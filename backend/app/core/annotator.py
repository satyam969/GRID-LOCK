"""
Evidence Annotator — Draws violation bboxes, labels, and metadata on images.
Produces annotated evidence images saved to disk.
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
import uuid

from app.config import settings


# Color palette per violation type (BGR)
VIOLATION_COLORS = {
    "helmet_non_compliance":   (0,   50,  255),  # Vivid Red
    "seatbelt_non_compliance": (0,   128, 255),  # Orange
    "triple_riding":           (0,   0,   200),  # Dark Red
    "wrong_side_driving":      (180, 0,   180),  # Purple
    "stop_line_violation":     (0,   200, 255),  # Yellow
    "red_light_violation":     (0,   0,   255),  # Pure Red
    "illegal_parking":         (128, 128, 0),    # Olive
    "vehicle":                 (0,   200, 0),    # Green (normal detection)
    "person":                  (255, 200, 0),    # Cyan-ish
    "default":                 (200, 200, 200),  # Gray
}

VIOLATION_SHORT_LABELS = {
    "helmet_non_compliance":   "NO HELMET",
    "seatbelt_non_compliance": "NO SEATBELT",
    "triple_riding":           "TRIPLE RIDING",
    "wrong_side_driving":      "WRONG SIDE",
    "stop_line_violation":     "STOP LINE",
    "red_light_violation":     "RED LIGHT",
    "illegal_parking":         "ILLEGAL PARK",
}


class Annotator:

    FONT = cv2.FONT_HERSHEY_DUPLEX
    FONT_SCALE_BASE = 0.6
    THICKNESS = 2

    @classmethod
    def annotate(
        cls,
        img: np.ndarray,
        violations: List[Dict],
        all_detections: List[Dict] = None,
        license_plate: Optional[str] = None,
        person_count: Optional[int] = None,
        show_all_detections: bool = True,
    ) -> np.ndarray:
        """
        Draw all detections and violations on the image.
        Returns annotated copy (original untouched).
        """
        canvas = img.copy()
        h, w = canvas.shape[:2]

        # Draw background detections first (green, lighter)
        if show_all_detections and all_detections:
            for det in all_detections:
                if det["class_name"] in ("person", "motorcycle", "car", "truck", "bus", "bicycle"):
                    color = VIOLATION_COLORS.get(det["class_name"], VIOLATION_COLORS["vehicle"])
                    cls._draw_box(canvas, det["bbox"], color, det["class_name"], det["confidence"], thin=True)

        # Draw violations (bold, colored)
        for violation in violations:
            v_type = violation["violation_type"]
            if hasattr(v_type, "value"):
                v_type = v_type.value
            color = VIOLATION_COLORS.get(v_type, VIOLATION_COLORS["default"])
            label = VIOLATION_SHORT_LABELS.get(v_type, v_type.replace("_", " ").upper())
            cls._draw_box(canvas, violation["bbox"], color, label, violation["confidence"], thin=False)

        # Info panel (top-left overlay)
        cls._draw_info_panel(canvas, violations, license_plate, person_count, w, h)

        # Timestamp watermark
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        cv2.putText(canvas, f"TrafficGuard AI  |  {ts}", (10, h - 10),
                    cls.FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        return canvas

    @classmethod
    def _draw_box(
        cls, canvas: np.ndarray, bbox: Dict, color: tuple, label: str,
        confidence: float, thin: bool = False
    ):
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        thickness = 1 if thin else cls.THICKNESS

        # Main rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        if thin:
            return  # Skip label for background detections

        # Label background
        label_text = f"{label} {confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cls.FONT, cls.FONT_SCALE_BASE, 1
        )
        label_y = max(y1 - 5, text_h + 5)
        cv2.rectangle(
            canvas,
            (x1, label_y - text_h - baseline),
            (x1 + text_w + 4, label_y + baseline),
            color, -1
        )
        # Label text
        cv2.putText(
            canvas, label_text,
            (x1 + 2, label_y),
            cls.FONT, cls.FONT_SCALE_BASE,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    @classmethod
    def _draw_info_panel(
        cls, canvas: np.ndarray, violations: List[Dict],
        license_plate: Optional[str], person_count: Optional[int],
        img_w: int, img_h: int
    ):
        """Semi-transparent info panel in top-right corner."""
        panel_w, panel_h = 280, max(120, 40 + len(violations) * 28)
        panel_x = img_w - panel_w - 10
        panel_y = 10

        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        # Header
        cv2.putText(canvas, "VIOLATION SUMMARY", (panel_x + 8, panel_y + 22),
                    cls.FONT, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        y_offset = panel_y + 50
        if not violations:
            cv2.putText(canvas, "No violations detected", (panel_x + 8, y_offset),
                        cls.FONT, 0.4, (100, 255, 100), 1, cv2.LINE_AA)
        else:
            for v in violations:
                v_type = v["violation_type"]
                if hasattr(v_type, "value"):
                    v_type = v_type.value
                label = VIOLATION_SHORT_LABELS.get(v_type, v_type[:20])
                color = VIOLATION_COLORS.get(v_type, VIOLATION_COLORS["default"])
                cv2.putText(canvas, f"• {label} ({v['confidence']:.0%})",
                            (panel_x + 8, y_offset), cls.FONT, 0.42, color, 1, cv2.LINE_AA)
                y_offset += 26

        # License plate
        if license_plate:
            cv2.putText(canvas, f"Plate: {license_plate}", (panel_x + 8, y_offset + 10),
                        cls.FONT, 0.45, (255, 230, 100), 1, cv2.LINE_AA)

    @classmethod
    def save_annotated(cls, canvas: np.ndarray, original_filename: str) -> str:
        """Save annotated image and return relative URL path."""
        settings.ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(original_filename).stem
        out_name = f"{stem}_annotated_{uuid.uuid4().hex[:8]}.jpg"
        out_path = settings.ANNOTATED_DIR / out_name
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return f"/annotated/{out_name}"


# Module-level singleton
annotator = Annotator()
