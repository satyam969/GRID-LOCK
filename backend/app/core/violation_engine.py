"""
Violation Engine — Core business logic for detecting each violation type.
Uses pre-trained model outputs + geometric/rule-based reasoning.
"""
import logging
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple

from app.config import settings
from app.core.detector import Detector, detector
from app.models.violation import ViolationType, VehicleType

logger = logging.getLogger(__name__)




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
        check_stop_line: bool = True,
        stop_line_y_ratio: float = None,
        check_parking: bool = False,
        lane_direction: str = "right",  # "right" or "left"
    ) -> Dict:
        """
        Full analysis pipeline for one image.
        Returns structured result with all detected violations.
        """
        h, w = img.shape[:2]
        violations = []
        metadata = {}

        # ── Step 1: General detection (vehicles + persons) ─────────────────
        general_detections = self.det.detect_vehicles_and_persons(img)
        vehicles = [d for d in general_detections if d["class_id"] in self.det.VEHICLE_CLASS_IDS]
        persons = [d for d in general_detections if d["class_id"] == self.det.PERSON_CLASS_ID]
        motorcycles = [d for d in vehicles if d["class_id"] == self.det.MOTORCYCLE_CLASS_ID]
        cars = [d for d in vehicles if d["class_id"] == self.det.CAR_CLASS_ID]

        metadata["general_detections"] = general_detections
        metadata["vehicle_count"] = len(vehicles)
        metadata["person_count"] = len(persons)

        # Determine dominant vehicle type
        vehicle_type = self._get_dominant_vehicle_type(vehicles)

        # ── Step 2: Helmet detection (motorcycles only) ────────────────────
        if motorcycles:
            helmet_violations = self._check_helmet(img, motorcycles, persons)
            violations.extend(helmet_violations)
            metadata["helmet_checks"] = len(helmet_violations)

        # ── Step 3: Triple riding (motorcycles only) ───────────────────────
        if motorcycles:
            triple_violations = self._check_triple_riding(img, motorcycles, persons)
            violations.extend(triple_violations)

        # ── Step 4: Seatbelt (cars/trucks/buses/persons) ───────────────────
        if cars or any(d["class_id"] in {5, 7} for d in vehicles) or persons:
            seatbelt_violations = self._check_seatbelt(img, vehicles, persons)
            violations.extend(seatbelt_violations)

        # ── Step 5: Stop-line violation ────────────────────────────────────
        if check_stop_line and vehicles:
            stop_y = (stop_line_y_ratio or settings.STOP_LINE_Y_RATIO) * h
            stop_violations = self._check_stop_line(vehicles, stop_y, h)
            violations.extend(stop_violations)

        # ── Step 6: Red-light violation ────────────────────────────────────
        traffic_lights = [d for d in general_detections if d["class_id"] == 9]
        if traffic_lights and vehicles:
            red_violations = self._check_red_light(img, traffic_lights, vehicles, h, w)
            violations.extend(red_violations)

        # ── Step 7: Wrong-side driving ─────────────────────────────────────
        # Disabled per playbook recommendation (impossible to reliably detect without optical flow)
        # if vehicles:
        #     wrong_side = self._check_wrong_side(img, vehicles, w, lane_direction)
        #     violations.extend(wrong_side)

        return {
            "violations": violations,
            "vehicle_type": vehicle_type,
            "person_count": len(persons),
            "all_detections": general_detections,
            "metadata": metadata,
        }

    # ── Helmet Detection ────────────────────────────────────────────────────

    def _is_rider(self, person: Dict, moto_box: Dict) -> bool:
        """Determines if a detected person is RIDING (not near) a motorcycle."""
        person_box = person["bbox"]

        # 0. High-Accuracy Pose Keypoint Check
        if "keypoints" in person and person["keypoints"]:
            kpts = person["keypoints"]
            # Check if hip keypoints (11, 12) are valid and inside motorcycle
            if len(kpts) > 12:
                left_hip, right_hip = kpts[11], kpts[12]
                lx, ly = left_hip[0], left_hip[1]
                rx, ry = right_hip[0], right_hip[1]
                
                # If both hips are valid, check their location
                if lx > 0 and rx > 0 and ly > 0 and ry > 0:
                    hip_x = (lx + rx) / 2
                    hip_y = (ly + ry) / 2
                    
                    if (moto_box["x1"] <= hip_x <= moto_box["x2"] and 
                        moto_box["y1"] - 50 <= hip_y <= moto_box["y2"]):
                        return True
                    else:
                        return False # Hips are definitely NOT on the bike

        # 1. Person center-x must be within motorcycle horizontal span
        pcx = (person_box["x1"] + person_box["x2"]) / 2
        if not (moto_box["x1"] <= pcx <= moto_box["x2"]):
            return False

        # 2. Person bottom edge must not extend far below motorcycle
        if person_box["y2"] > moto_box["y2"] + 20:
            return False

        # 3. Person center-y must be in the riding zone
        pcy = (person_box["y1"] + person_box["y2"]) / 2
        moto_h = moto_box["y2"] - moto_box["y1"]
        zone_top = moto_box["y1"] - moto_h * 0.8
        zone_bot = moto_box["y2"]
        if not (zone_top <= pcy <= zone_bot):
            return False

        # 4. Minimum IoU to confirm physical overlap
        if self.det.compute_iou({"bbox": person_box}, {"bbox": moto_box}) < 0.05:
            return False

        return True

    def _check_helmet(
        self, img: np.ndarray, motorcycles: List[Dict], persons: List[Dict]
    ) -> List[Dict]:
        """
        Run helmet model on full image.
        Checks if any rider's head (top 40% of their bbox) is missing a helmet.
        """
        violations = []
        helmet_detections = self.det.detect_helmets(img)

        # The downloaded HF model uses 'Without Helmet'
        no_helmet_dets = [d for d in helmet_detections if "no" in d["class_name"].lower()
                          or "without" in d["class_name"].lower()]

        for moto in motorcycles:
            # 1. Find the rider(s) on this motorcycle
            riders = [p for p in persons if self._is_rider(p, moto["bbox"])]
            
            for rider in riders:
                rb = rider["bbox"]
                rh = rb["y2"] - rb["y1"]
                # Head region = top 40% of person bbox
                head_zone = {
                    "bbox": {
                        "x1": rb["x1"],
                        "y1": rb["y1"],
                        "x2": rb["x2"],
                        "y2": rb["y1"] + rh * 0.4,
                    }
                }

                for nh in no_helmet_dets:
                    iou = self.det.compute_iou({"bbox": head_zone["bbox"]}, nh)
                    if iou > 0.15:
                        violations.append({
                            "violation_type": ViolationType.HELMET_NON_COMPLIANCE,
                            "confidence": nh["confidence"],
                            "severity": self.compute_severity(ViolationType.HELMET_NON_COMPLIANCE, nh["confidence"]),
                            "description": VIOLATION_DESCRIPTIONS[ViolationType.HELMET_NON_COMPLIANCE],
                            "bbox": moto["bbox"],
                        })
                        break # One violation per rider is enough

        return violations

    # ── Triple Riding ───────────────────────────────────────────────────────

    def _check_triple_riding(
        self, img: np.ndarray, motorcycles: List[Dict], persons: List[Dict]
    ) -> List[Dict]:
        """Count persons associated with each motorcycle."""
        violations = []

        # Try pose model first for more accurate person count
        pose_dets = self.det.detect_poses(img)
        person_source = pose_dets if pose_dets else persons

        for moto in motorcycles:
            riders = [p for p in person_source if self._is_rider(p, moto["bbox"])]
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
        self, vehicles: List[Dict], stop_y: float, img_height: int
    ) -> List[Dict]:
        """
        If vehicle bottom edge is below (greater Y) than stop_y line
        AND vehicle is in lower half of frame → potential stop-line violation.
        """
        violations = []
        for v in vehicles:
            bottom = v["bbox"]["y2"]
            center_y = (v["bbox"]["y1"] + v["bbox"]["y2"]) / 2

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
