"""
OCR Engine — License plate text extraction using PaddleOCR (PP-OCRv4).
Replaces EasyOCR to fix duplicate text detection issues.
"""
import re
import logging
import cv2
from typing import Optional, Tuple, List, Dict
import numpy as np

logger = logging.getLogger(__name__)


class OCREngine:
    """
    High-performance license plate OCR using PaddleOCR PP-OCRv4.
    Lazy-loaded to avoid slow startup; initialized on first use.
    """

    _ocr = None
    confidence_threshold = 0.30

    @classmethod
    def _get_ocr(cls):
        if cls._ocr is None:
            try:
                # pyrefly: ignore [missing-import]
                from paddleocr import PaddleOCR
                logger.info("Loading PaddleOCR model (first use)...")
                cls._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    det_db_thresh=0.3,
                    rec_batch_num=6,
                )
                logger.info("✅ PaddleOCR ready")
            except ImportError:
                logger.error("❌ PaddleOCR not installed. Run: pip install paddlepaddle paddleocr")
                cls._ocr = None
            except Exception as e:
                logger.error(f"❌ PaddleOCR failed to load: {e}")
                cls._ocr = None
        return cls._ocr

    @staticmethod
    def preprocess_plate(plate_img: np.ndarray) -> np.ndarray:
        """Apply CLAHE + Bilateral Filtering."""
        if plate_img is None or plate_img.size == 0:
            return plate_img
            
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
        return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    @classmethod
    def extract_plate_text(
        cls,
        img: np.ndarray,
        plate_bbox: Optional[Dict] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Extract text from the license plate region.
        Returns (plate_text, confidence).
        """
        ocr = cls._get_ocr()
        if ocr is None:
            return None, 0.0

        # Crop plate region if bbox is provided
        if plate_bbox:
            # Add small padding
            h, w = img.shape[:2]
            padding = 5
            x1 = max(0, int(plate_bbox["x1"]) - padding)
            y1 = max(0, int(plate_bbox["y1"]) - padding)
            x2 = min(w, int(plate_bbox["x2"]) + padding)
            y2 = min(h, int(plate_bbox["y2"]) + padding)
            plate_img = img[y1:y2, x1:x2]
        else:
            plate_img = img
            
        if plate_img.size == 0:
            return None, 0.0

        processed = cls.preprocess_plate(plate_img)
        try:
            results = ocr.ocr(processed)
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return None, 0.0

        if not results or not results[0]:
            return None, 0.0

        # Pick SINGLE highest-confidence line to fix duplicate bug
        best_text, best_conf = '', 0.0
        for line in results[0]:
            text, conf = line[1][0], line[1][1]
            if conf > best_conf:
                best_text, best_conf = text, conf

        if best_conf < cls.confidence_threshold:
            return None, 0.0

        cleaned = cls._validate_indian_plate(best_text)
        return cleaned or best_text, round(best_conf, 4)

    @staticmethod
    def _validate_indian_plate(raw_text: str) -> Optional[str]:
        """Post-process OCR output to match Indian plate format."""
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        match = re.search(r'([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})', cleaned)
        if match:
            return f'{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}'
        
        # BH series match
        bh_match = re.search(r'(\d{2})(BH)(\d{4})([A-Z]{1,2})', cleaned)
        if bh_match:
            return f'{bh_match.group(1)} {bh_match.group(2)} {bh_match.group(3)} {bh_match.group(4)}'
            
        return cleaned

    @classmethod
    def extract_from_detections(
        cls, img: np.ndarray, plate_detections: List[Dict]
    ) -> Tuple[Optional[str], float]:
        """
        Given plate detection bboxes, pick the largest plate and run OCR.
        """
        if not plate_detections:
            return None, 0.0

        # Pick largest detected plate
        largest_plate = max(
            plate_detections,
            key=lambda d: (
                (d["bbox"]["x2"] - d["bbox"]["x1"]) *
                (d["bbox"]["y2"] - d["bbox"]["y1"])
            ),
        )
        return cls.extract_plate_text(img, largest_plate["bbox"])


# Module-level singleton
ocr_engine = OCREngine()
