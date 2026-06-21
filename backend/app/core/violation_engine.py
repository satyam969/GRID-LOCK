"""
Violation Engine — Core business logic for detecting each violation type.
Uses pre-trained model outputs + geometric/rule-based reasoning.
Models are run in parallel using ThreadPoolExecutor for maximum CPU throughput.
"""
import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import settings
from app.core.detector import Detector, detector
from app.models.violation import ViolationType, VehicleType
from app.core.red_light_detection import check_red_light_violation, detect_traffic_light_color
from app.core.illegal_parking_detection import check_illegal_parking
from app.core.wrong_side_detection import check_wrong_side_driving

logger = logging.getLogger(__name__)

# Shared thread pool — reused across requests (avoids thread creation overhead)
_THREAD_POOL = ThreadPoolExecutor(max_workers=4)




VIOLATION_DESCRIPTIONS = {
    ViolationType.HELMET_NON_COMPLIANCE: "Rider detected without wearing a helmet",
    ViolationType.SEATBELT_NON_COMPLIANCE: "Driver detected without wearing a seatbelt",
    ViolationType.TRIPLE_RIDING: "Three or more persons detected on a motorcycle",
    ViolationType.RED_LIGHT_VIOLATION: "Vehicle crossing intersection during red light",
    ViolationType.WRONG_SIDE_DRIVING: "Vehicle detected driving against traffic flow",
    ViolationType.STOP_LINE_VIOLATION: "Vehicle crossed the stop line at intersection",
    ViolationType.ILLEGAL_PARKING: "Vehicle parked in a no-parking zone",
}


class ViolationEngine:
    """
    Orchestrates all violation detection logic.
    Call `analyze(img)` to get all detected violations for one image.
    """

    def __init__(self):
        self.det = detector

    def compute_severity(self, violation_type: ViolationType, confidence: float) -> str:
        weights = {
            ViolationType.HELMET_NON_COMPLIANCE: 0.8,
            ViolationType.SEATBELT_NON_COMPLIANCE: 0.7,
            ViolationType.TRIPLE_RIDING: 0.9,
            ViolationType.RED_LIGHT_VIOLATION: 0.9,
            ViolationType.WRONG_SIDE_DRIVING: 0.9,
            ViolationType.STOP_LINE_VIOLATION: 0.4,
            ViolationType.ILLEGAL_PARKING: 0.3,
        }
        score = weights.get(violation_type, 0.5) * confidence
        if score >= 0.70: return "high"
        if score >= 0.45: return "medium"
        return "low"

    def analyze(
        self,
        img: np.ndarray,
        check_stop_line: bool = False,
        stop_line_y_ratio: Optional[float] = None,
        lane_direction: str = "right",
        detect_parking: bool = True,
        flow_direction: str = "none",
    ) -> Dict:
        """
        Full analysis pipeline for one image.

        Phase 1 (sequential): General detection — must run first to know
                              which vehicle types are present.
        Phase 2 (parallel):   Pose, helmet, seatbelt run concurrently
                              using a shared ThreadPoolExecutor.
        Phase 3 (sequential): Rule-based violation checks (no model needed).
        """
        h, w = img.shape[:2]
        violations = []
        metadata = {}

        # ── Phase 1: General detection (vehicles + persons) ───────────────
        general_detections = self.det.detect_vehicles_and_persons(img)
        vehicles = [d for d in general_detections if d["class_id"] in self.det.VEHICLE_CLASS_IDS]
        persons  = [d for d in general_detections if d["class_id"] == self.det.PERSON_CLASS_ID]
        motorcycles = [d for d in vehicles if d["class_id"] == self.det.MOTORCYCLE_CLASS_ID]
        cars        = [d for d in vehicles if d["class_id"] == self.det.CAR_CLASS_ID]

        metadata["general_detections"] = general_detections
        metadata["vehicle_count"] = len(vehicles)
        metadata["person_count"]  = len(persons)

        vehicle_type = self._get_dominant_vehicle_type(vehicles)

        # ── Phase 2: Parallel model inference ─────────────────────────────
        # Submit all relevant model calls concurrently; collect results
        futures = {}

        if motorcycles:
            futures["pose"]   = _THREAD_POOL.submit(self.det.detect_poses, img)
            futures["helmet"] = _THREAD_POOL.submit(self._check_helmet, img, motorcycles, persons)

        if cars or any(d["class_id"] in {5, 7} for d in vehicles) or persons:
            futures["seatbelt"] = _THREAD_POOL.submit(self._check_seatbelt, img, vehicles, persons)

        # Collect parallel results (block until all done)
        pose_detections = []
        if "pose" in futures:
            try:
                pose_detections = futures["pose"].result()
            except Exception as e:
                logger.warning(f"Pose model failed: {e}")

        if "helmet" in futures:
            try:
                violations.extend(futures["helmet"].result())
            except Exception as e:
                logger.warning(f"Helmet check failed: {e}")

        if "seatbelt" in futures:
            try:
                violations.extend(futures["seatbelt"].result())
            except Exception as e:
                logger.warning(f"Seatbelt check failed: {e}")

        # ── Triple riding (uses pose results already fetched) ──────────────
        if motorcycles:
            pose_persons = [p for p in pose_detections if p.get("class_id") == 0]
            person_source = pose_persons if len(pose_persons) > len(persons) else persons
            triple_violations = self._check_triple_riding_from_persons(motorcycles, person_source)
            violations.extend(triple_violations)

        # ── Phase 3: Rule-based checks (no model inference) ───────────────
        traffic_lights = [d for d in general_detections if d["class_id"] == 9]

        # Filter vehicles by flow direction
        target_vehicles = []
        for v in vehicles:
            v_cx = (v['bbox']['x1'] + v['bbox']['x2']) / 2
            if flow_direction == 'left'  and v_cx > w / 2: continue
            if flow_direction == 'right' and v_cx < w / 2: continue
            target_vehicles.append(v)

        # Stop-line violation
        if check_stop_line and target_vehicles:
            stop_y = (stop_line_y_ratio or settings.STOP_LINE_Y_RATIO) * h
            light_color = None
            if traffic_lights:
                for tl in traffic_lights:
                    light_color = detect_traffic_light_color(img, tl['bbox'])
                    if light_color:
                        break
            if light_color != 'GREEN':
                stop_violations = self._check_stop_line(target_vehicles, stop_y, h, w, flow_direction)
                violations.extend(stop_violations)

        # Red-light violation
        if traffic_lights and target_vehicles:
            stop_y = (stop_line_y_ratio or settings.STOP_LINE_Y_RATIO) * h
            red_violations = check_red_light_violation(
                vehicle_detections=target_vehicles,
                traffic_light_detections=traffic_lights,
                stop_line_y=stop_y,
                img=img,
                flow_direction=flow_direction,
            )
            violations.extend(red_violations)

        # Illegal parking
        if detect_parking:
            parking_violations = check_illegal_parking(
                all_detections=general_detections,
                img_shape=img.shape,
                no_parking_zones=[],
            )
            violations.extend(parking_violations)

        # Wrong-side driving
        if flow_direction != 'none':
            ws_violations = check_wrong_side_driving(
                general_detections, img.shape,
                {'flow_direction': flow_direction, 'lane_boundary_x': 0.5}
            )
            violations.extend(ws_violations)

        return {
            "violations":    violations,
            "vehicle_type":  vehicle_type,
            "person_count":  len(persons),
            "all_detections": general_detections,
            "metadata":      metadata,
        }

    # ── Helmet Detection ────────────────────────────────────────────────────

    def _is_rider(self, person: Dict, moto_box: Dict) -> bool:
        """Determines if a detected person is RIDING (not near) a motorcycle."""
        person_box = person["bbox"]

        # Simple bounding box intersection (Intersection over Person Area)
        # If the person bounding box overlaps with the motorcycle bounding box by > 20%
        # they are highly likely a rider.
        
        px1, py1, px2, py2 = person_box["x1"], person_box["y1"], person_box["x2"], person_box["y2"]
        mx1, my1, mx2, my2 = moto_box["x1"], moto_box["y1"], moto_box["x2"], moto_box["y2"]
        
        ix1 = max(px1, mx1)
        iy1 = max(py1, my1)
        ix2 = min(px2, mx2)
        iy2 = min(py2, my2)
        
        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        
        # Extremely forgiving heuristic for demo: Any overlap means they are a rider
        if inter_area > 0:
            return True
        return False

    def _check_helmet(
        self, img: np.ndarray, motorcycles: List[Dict], persons: List[Dict]
    ) -> List[Dict]:
        """
        Check if any rider's head is missing a helmet.
        Since community models were removed, uses an HSV skin-tone heuristic
        on the top 30% of the rider's bounding box.
        """
        violations = []

        for moto in motorcycles:
            # 1. Find the rider(s) on this motorcycle
            riders = [p for p in persons if self._is_rider(p, moto["bbox"])]
            
            for rider in riders:
                rb = rider["bbox"]
                rh = rb["y2"] - rb["y1"]
                
                # Head region = top 30% of person bbox
                y1 = int(max(0, rb["y1"]))
                y2 = int(min(img.shape[0], rb["y1"] + rh * 0.3))
                x1 = int(max(0, rb["x1"]))
                x2 = int(min(img.shape[1], rb["x2"]))
                
                head_crop = img[y1:y2, x1:x2]
                if head_crop.size == 0:
                    continue

                # Convert to HSV and check for skin tones (exposed face/head)
                hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([0, 20, 50], dtype=np.uint8)
                upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                
                skin_pixels = cv2.countNonZero(skin_mask)
                total_pixels = head_crop.shape[0] * head_crop.shape[1]
                
                # If > 10% of the head region is exposed skin, flag as No Helmet
                if total_pixels > 0 and (skin_pixels / total_pixels) > 0.10:
                    violations.append({
                        "violation_type": ViolationType.HELMET_NON_COMPLIANCE,
                        "confidence": 0.85, # Heuristic confidence
                        "severity": self.compute_severity(ViolationType.HELMET_NON_COMPLIANCE, 0.85),
                        "description": "Exposed head detected (No Helmet)",
                        "bbox": rider["bbox"],
                    })
                    # Flagging the first rider without a helmet is enough for the moto
                    break

        return violations

    # ── Triple Riding ───────────────────────────────────────────────────────

    def _check_triple_riding_from_persons(
        self, motorcycles: List[Dict], persons: List[Dict]
    ) -> List[Dict]:
        """Count persons per motorcycle using already-fetched person detections."""
        violations = []
        for moto in motorcycles:
            riders = [p for p in persons if self._is_rider(p, moto["bbox"])]
            count = len(riders)
            if count >= settings.TRIPLE_RIDING_PERSON_COUNT:
                violations.append({
                    "violation_type": ViolationType.TRIPLE_RIDING,
                    "confidence": min(0.95, 0.70 + count * 0.05),
                    "severity": self.compute_severity(ViolationType.TRIPLE_RIDING, min(0.95, 0.70 + count * 0.05)),
                    "description": f"{count} persons detected on motorcycle",
                    "bbox": moto["bbox"],
                    "person_count": count,
                })
        return violations


    # ── Seatbelt Detection ──────────────────────────────────────────────────

    def _check_seatbelt(
        self, img: np.ndarray, vehicles: List[Dict], persons: List[Dict]
    ) -> List[Dict]:
        """Run seatbelt model; flag '0' or 'no-seatbelt' detections in car driver region."""
        violations = []
        
        cars = [v for v in vehicles if v["class_id"] in {2, 5, 7}] # Car, Bus, Truck
        
        if cars:
            for car in cars:
                # Windshield crop for exterior cameras
                cb = car["bbox"]
                car_h = cb["y2"] - cb["y1"]
                car_w = cb["x2"] - cb["x1"]
                ws_x1 = int(max(0, cb["x1"] + car_w * 0.15))
                ws_x2 = int(min(img.shape[1], cb["x2"] - car_w * 0.15))
                ws_y1 = int(max(0, cb["y1"]))
                ws_y2 = int(min(img.shape[0], cb["y1"] + car_h * 0.50))
                
                if ws_x2 <= ws_x1 or ws_y2 <= ws_y1:
                    continue
                
                crop = img[ws_y1:ws_y2, ws_x1:ws_x2]
                if crop.size == 0:
                    continue
                    
                sb_dets = self.det.detect_seatbelts(crop)
                no_seatbelt = [d for d in sb_dets if d["class_name"] == "0" or "no" in d["class_name"].lower()]
                for det in no_seatbelt:
                    # Adjust crop coordinates back to original image
                    det["bbox"]["x1"] += ws_x1
                    det["bbox"]["x2"] += ws_x1
                    det["bbox"]["y1"] += ws_y1
                    det["bbox"]["y2"] += ws_y1
                    
                    violations.append({
                        "violation_type": ViolationType.SEATBELT_NON_COMPLIANCE,
                        "confidence": det["confidence"],
                        "severity": self.compute_severity(ViolationType.SEATBELT_NON_COMPLIANCE, det["confidence"]),
                        "description": VIOLATION_DESCRIPTIONS[ViolationType.SEATBELT_NON_COMPLIANCE],
                        "bbox": det["bbox"],
                    })
        else:
            # Interior dashcam shot where 'cars' is empty
            seatbelt_dets = self.det.detect_seatbelts(img)
            no_seatbelt = [
                d for d in seatbelt_dets
                if d["class_name"] == "0" or "no" in d["class_name"].lower() or "without" in d["class_name"].lower()
            ]
            for det in no_seatbelt:
                violations.append({
                    "violation_type": ViolationType.SEATBELT_NON_COMPLIANCE,
                    "confidence": det["confidence"],
                    "severity": self.compute_severity(ViolationType.SEATBELT_NON_COMPLIANCE, det["confidence"]),
                    "description": VIOLATION_DESCRIPTIONS[ViolationType.SEATBELT_NON_COMPLIANCE],
                    "bbox": det["bbox"],
                })

        return violations

    # ── Stop Line Violation ─────────────────────────────────────────────────

    def _check_stop_line(
        self, vehicles: List[Dict], stop_y: float, img_height: int, img_width: int, flow_direction: str
    ) -> List[Dict]:
        """
        If vehicle bottom edge is below (greater Y) than stop_y line
        AND vehicle is in lower half of frame → potential stop-line violation.
        Filters out oncoming traffic using flow_direction.
        """
        violations = []
        for v in vehicles:
            bottom = v["bbox"]["y2"]
            center_x = (v["bbox"]["x1"] + v["bbox"]["x2"]) / 2
            center_y = (v["bbox"]["y1"] + v["bbox"]["y2"]) / 2

            # Filter out oncoming traffic based on expected flow direction
            if flow_direction == 'left' and center_x > img_width / 2:
                continue
            if flow_direction == 'right' and center_x < img_width / 2:
                continue

            if bottom > stop_y and center_y > img_height * 0.4:
                violations.append({
                    "violation_type": ViolationType.STOP_LINE_VIOLATION,
                    "confidence": 0.70,
                    "severity": self.compute_severity(ViolationType.STOP_LINE_VIOLATION, 0.70),
                    "description": VIOLATION_DESCRIPTIONS[ViolationType.STOP_LINE_VIOLATION],
                    "bbox": v["bbox"],
                })
                break  # Typically one vehicle triggers this per frame

        return violations

    # ── Red Light Violation ─────────────────────────────────────────────────

    def _check_red_light(
        self,
        img: np.ndarray,
        traffic_lights: List[Dict],
        vehicles: List[Dict],
        h: int, w: int,
    ) -> List[Dict]:
        """
        Check if traffic light in frame is red AND vehicles are in intersection zone.
        Uses HSV color analysis on the traffic light bounding box.
        """
        violations = []

        for tl in traffic_lights:
            is_red = self._is_traffic_light_red(img, tl["bbox"])
            if not is_red:
                continue

            # Intersection zone = bottom 40% of frame
            intersection_y = h * 0.6
            moving_vehicles = [
                v for v in vehicles
                if v["bbox"]["y1"] > intersection_y
            ]

            if moving_vehicles:
                violations.append({
                    "violation_type": ViolationType.RED_LIGHT_VIOLATION,
                    "confidence": 0.75,
                    "severity": VIOLATION_SEVERITY[ViolationType.RED_LIGHT_VIOLATION],
                    "description": VIOLATION_DESCRIPTIONS[ViolationType.RED_LIGHT_VIOLATION],
                    "bbox": moving_vehicles[0]["bbox"],
                })
                break

        return violations

    def _is_traffic_light_red(self, img: np.ndarray, bbox: Dict) -> bool:
        """Crop traffic light region and check for red color dominance via HSV."""
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        h_img, w_img = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        # Top third = red light position
        top_third = crop[:crop.shape[0] // 3, :]
        hsv = cv2.cvtColor(top_third, cv2.COLOR_BGR2HSV)

        # Red color ranges in HSV
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_pixels = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)
        total_pixels = top_third.shape[0] * top_third.shape[1]

        return red_pixels > total_pixels * 0.15  # 15% red = light is red

    # ── Wrong Side Driving ──────────────────────────────────────────────────

    def _check_wrong_side(
        self, img: np.ndarray, vehicles: List[Dict], img_width: int, lane_direction: str
    ) -> List[Dict]:
        """
        Heuristic: if lane_direction='right', vehicles should be in right half of frame.
        Vehicles predominantly in left half may indicate wrong-side driving.
        This is a simplified heuristic — proper implementation uses optical flow.
        """
        violations = []
        mid_x = img_width / 2

        for v in vehicles:
            center_x = (v["bbox"]["x1"] + v["bbox"]["x2"]) / 2
            if lane_direction == "right" and center_x < mid_x * 0.3:
                violations.append({
                    "violation_type": ViolationType.WRONG_SIDE_DRIVING,
                    "confidence": 0.55,
                    "severity": VIOLATION_SEVERITY[ViolationType.WRONG_SIDE_DRIVING],
                    "description": VIOLATION_DESCRIPTIONS[ViolationType.WRONG_SIDE_DRIVING],
                    "bbox": v["bbox"],
                })

        return violations

    # ── Utilities ───────────────────────────────────────────────────────────

    @staticmethod
    def _expand_bbox(bbox: Dict, factor: float, img_shape: Tuple) -> Dict:
        """Expand a bounding box by a factor (for proximity checks)."""
        h, w = img_shape[:2]
        dx = (bbox["x2"] - bbox["x1"]) * factor
        dy = (bbox["y2"] - bbox["y1"]) * factor
        return {
            "x1": max(0, bbox["x1"] - dx),
            "y1": max(0, bbox["y1"] - dy),
            "x2": min(w, bbox["x2"] + dx),
            "y2": min(h, bbox["y2"] + dy),
        }

    @staticmethod
    def _get_dominant_vehicle_type(vehicles: List[Dict]) -> VehicleType:
        if not vehicles:
            return VehicleType.UNKNOWN
        class_map = {
            1: VehicleType.BICYCLE,
            2: VehicleType.CAR,
            3: VehicleType.MOTORCYCLE,
            5: VehicleType.BUS,
            7: VehicleType.TRUCK,
        }
        # Pick largest vehicle
        largest = max(
            vehicles,
            key=lambda v: (v["bbox"]["x2"] - v["bbox"]["x1"]) * (v["bbox"]["y2"] - v["bbox"]["y1"])
        )
        return class_map.get(largest["class_id"], VehicleType.UNKNOWN)


# Module-level singleton
violation_engine = ViolationEngine()
