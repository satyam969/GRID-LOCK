"""
ModelRegistry — loads and caches all YOLO models at startup.
Handles missing community weights gracefully by falling back to COCO.
"""
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton that holds all loaded YOLO models."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._models: Dict[str, Any] = {}
        self._status: Dict[str, bool] = {}
        ModelRegistry._initialized = True

    def load_all(self):
        """Load all models at application startup.

        Priority for each model:
          1. INT8 ONNX file  (fastest, quantized)
          2. .pt file in models_weights/  (standard PyTorch weights)
          3. Auto-download from Ultralytics (official general/pose only)
          4. Fall back to general COCO model (community models only)
        """
        from ultralytics import YOLO

        # (onnx_filename, pt_filename)
        model_configs = {
            "general": ("yolov8n_int8.onnx",           "yolov8n.pt"),
            "pose":    ("yolov8n-pose_int8.onnx",       "yolov8n-pose.pt"),
            "helmet":  (settings.MODEL_HELMET.replace(".pt", "_int8.onnx"),   settings.MODEL_HELMET),
            "seatbelt":(settings.MODEL_SEATBELT.replace(".pt", "_int8.onnx"), settings.MODEL_SEATBELT),
            "plate":   (settings.MODEL_PLATE.replace(".pt", "_int8.onnx"),    settings.MODEL_PLATE),
        }

        for name, (onnx_file, pt_file) in model_configs.items():
            task_type = "pose" if name == "pose" else "detect"
            loaded = False

            # 1. Try INT8 ONNX
            onnx_path = settings.MODELS_DIR / onnx_file
            if onnx_path.exists():
                try:
                    logger.info(f"Loading {name} model (INT8 ONNX) from {onnx_path}")
                    self._models[name] = YOLO(str(onnx_path), task=task_type)
                    loaded = True
                except Exception as e:
                    logger.warning(f"Failed to load ONNX for '{name}': {e}. Trying .pt ...")

            # 2. Try .pt in models_weights/
            if not loaded:
                pt_path = settings.MODELS_DIR / pt_file
                if pt_path.exists():
                    try:
                        logger.info(f"Loading {name} model (.pt) from {pt_path}")
                        self._models[name] = YOLO(str(pt_path), task=task_type)
                        loaded = True
                    except Exception as e:
                        logger.warning(f"Failed to load .pt for '{name}': {e}")

            # 3. For official models, auto-download from Ultralytics Hub
            if not loaded and name in ("general", "pose"):
                try:
                    logger.info(f"Auto-downloading official model: {pt_file}")
                    self._models[name] = YOLO(pt_file, task=task_type)
                    loaded = True
                except Exception as e:
                    logger.error(f"❌ Auto-download failed for '{pt_file}': {e}")

            # 4. For community models, fall back to the general COCO model
            if not loaded and name not in ("general", "pose"):
                logger.warning(
                    f"'{onnx_file}' and '{pt_file}' not found in models_weights/. "
                    f"Run scripts/download_models.py to fetch them. "
                    f"Using COCO general as fallback for '{name}'."
                )
                self._models[name] = self._models.get("general")
                loaded = self._models.get("general") is not None

            self._status[name] = loaded and self._models.get(name) is not None
            if self._status[name]:
                logger.info(f"✅ Model '{name}' ready")
            else:
                logger.error(f"❌ Failed to load model '{name}'")

    def get(self, name: str):
        return self._models.get(name)

    def status(self) -> Dict[str, bool]:
        return self._status.copy()


# Global singleton
model_registry = ModelRegistry()


class Detector:
    """
    High-level detection interface.
    Wraps ModelRegistry and exposes task-specific methods.
    """

    # COCO class IDs relevant to traffic
    COCO_CLASSES = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck", 9: "traffic light", 10: "fire hydrant",
        11: "stop sign"
    }

    VEHICLE_CLASS_IDS = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
    PERSON_CLASS_ID = 0
    MOTORCYCLE_CLASS_ID = 3
    CAR_CLASS_ID = 2
    TRAFFIC_LIGHT_CLASS_ID = 9

    def __init__(self):
        self.registry = model_registry

    def detect_vehicles_and_persons(
        self, img: np.ndarray, conf: Optional[float] = None
    ) -> List[Dict]:
        """
        Run COCO general model. Returns vehicles and persons, and traffic lights.
        Uses lower threshold for traffic lights to improve recall.
        """
        model = self.registry.get("general")
        if model is None:
            return []

        # Run with lowest threshold to capture traffic lights
        base_threshold = conf or settings.CONF_GENERAL
        results = model(img, conf=0.12, iou=settings.NMS_IOU, verbose=False)
        return self._parse_results(
            results[0], 
            filter_classes=list(self.COCO_CLASSES.keys()),
            base_conf=base_threshold,
            traffic_light_conf=0.12
        )

    def detect_helmets(self, img: np.ndarray) -> List[Dict]:
        """Detect helmet / no-helmet on the image."""
        model = self.registry.get("helmet")
        if model is None:
            return []
        results = model(img, conf=settings.CONF_HELMET, iou=settings.NMS_IOU, imgsz=416, half=True, verbose=False)
        return self._parse_results(results[0])

    def detect_seatbelts(self, img: np.ndarray) -> List[Dict]:
        """Detect seatbelt / no-seatbelt."""
        model = self.registry.get("seatbelt")
        if model is None:
            return []
        results = model(img, conf=settings.CONF_SEATBELT, iou=settings.NMS_IOU, imgsz=416, half=True, verbose=False)
        return self._parse_results(results[0])

    def detect_license_plate(self, img: np.ndarray) -> List[Dict]:
        """Detect license plate bounding boxes."""
        model = self.registry.get("plate")
        if model is None:
            return []
        results = model(img, conf=settings.CONF_PLATE, iou=settings.NMS_IOU, verbose=False)
        return self._parse_results(results[0])

    def detect_poses(self, img: np.ndarray) -> List[Dict]:
        """Run pose estimation — returns keypoints for occupant counting."""
        model = self.registry.get("pose")
        if model is None:
            return []
        results = model(img, conf=settings.CONF_GENERAL, verbose=False)
        detections = self._parse_results(results[0])
        # Add keypoints if available
        if results[0].keypoints is not None:
            kpts = results[0].keypoints.xy.cpu().numpy()
            for i, det in enumerate(detections):
                if i < len(kpts):
                    det["keypoints"] = kpts[i].tolist()
        return detections

    @staticmethod
    def _parse_results(
        result, 
        filter_classes: Optional[List[int]] = None,
        base_conf: float = 0.0,
        traffic_light_conf: float = 0.0
    ) -> List[Dict]:
        """Parse ultralytics Results object into list of dicts."""
        detections = []
        if result.boxes is None:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if filter_classes and cls_id not in filter_classes:
                continue
            conf = float(boxes.conf[i].item())
            
            # IMPROVEMENT: Custom thresholding per class
            if cls_id == 9:
                if conf < traffic_light_conf:
                    continue
            else:
                if conf < base_conf:
                    continue

            xyxy = boxes.xyxy[i].cpu().numpy()
            cls_name = result.names.get(cls_id, str(cls_id))

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                },
                "keypoints": None,
            })

        return detections

    @staticmethod
    def compute_iou(box1: Dict, box2: Dict) -> float:
        """Compute Intersection over Union between two bbox dicts."""
        b1 = box1["bbox"]
        b2 = box2["bbox"]

        xi1 = max(b1["x1"], b2["x1"])
        yi1 = max(b1["y1"], b2["y1"])
        xi2 = min(b1["x2"], b2["x2"])
        yi2 = min(b1["y2"], b2["y2"])

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        b1_area = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
        b2_area = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
        union_area = b1_area + b2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def get_dominant_vehicle(vehicles: List[Dict]) -> Optional[Dict]:
        """Return the largest vehicle (most likely the subject of interest)."""
        if not vehicles:
            return None
        return max(vehicles, key=lambda v: (
            (v["bbox"]["x2"] - v["bbox"]["x1"]) * (v["bbox"]["y2"] - v["bbox"]["y1"])
        ))


# Module-level singleton
detector = Detector()
