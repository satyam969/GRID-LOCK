"""
OCR Engine — License plate text extraction using EasyOCR.
Uses EasyOCR for reliable CPU-based OCR without PaddlePaddle oneDNN issues.
"""
import re
import logging
import cv2
from typing import Optional, Tuple, List, Dict
import numpy as np

logger = logging.getLogger(__name__)


class OCREngine:
    """
    License plate OCR using EasyOCR.
    Lazy-loaded to avoid slow startup; initialized on first use.
    """

    _reader = None
    confidence_threshold = 0.30

    @classmethod
    def _get_reader(cls):
        if cls._reader is None:
            try:
                import easyocr
                logger.info("Loading EasyOCR model (first use)...")
                cls._reader = easyocr.Reader(['en'], gpu=False)
                logger.info("✅ EasyOCR ready")
            except ImportError:
                logger.error("❌ EasyOCR not installed. Run: pip install easyocr")
                cls._reader = None
            except Exception as e:
                logger.error(f"❌ EasyOCR failed to load: {e}")
                cls._reader = None
        return cls._reader

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
        reader = cls._get_reader()
        if reader is None:
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
            results = reader.readtext(processed)
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return None, 0.0

        if not results:
            return None, 0.0

        # Concatenate all detected text fragments and pick the best approach
        # Strategy: combine all texts, then validate as Indian plate
        all_texts = []
        total_conf = 0.0
        count = 0
        for detection in results:
            text = detection[1]
            conf = detection[2]
            if conf >= cls.confidence_threshold:
                all_texts.append(text)
                total_conf += conf
                count += 1

        if not all_texts:
            return None, 0.0

        # Try combined text first (handles split detections like "WB" + "06 J 2431")
        combined = ' '.join(all_texts)
        avg_conf = total_conf / count

        # Also check individual high-confidence results
        best_single = max(results, key=lambda r: r[2])
        best_text = best_single[1]
        best_conf = best_single[2]

        # Validate combined text against Indian plate format
        combined_plate = cls._validate_indian_plate(combined)
        single_plate = cls._validate_indian_plate(best_text)

        # Prefer the validated result that looks like a proper plate
        if combined_plate and re.search(r'[A-Z]{2}\s?\d{2}', combined_plate):
            return combined_plate, round(avg_conf, 4)
        elif single_plate and re.search(r'[A-Z]{2}\s?\d{2}', single_plate):
            return single_plate, round(best_conf, 4)
        elif combined_plate:
            return combined_plate, round(avg_conf, 4)
        elif single_plate:
            return single_plate, round(best_conf, 4)
        else:
            # Return best raw text if no validation matched
            cleaned = re.sub(r'[^A-Z0-9]', '', combined.upper())
            if len(cleaned) >= 4:
                return cleaned, round(avg_conf, 4)
            return None, 0.0

    @staticmethod
    def _validate_indian_plate(raw_text: str) -> Optional[str]:
        """Post-process OCR output to match Indian plate format."""
        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        # Fix common OCR misreads
        cleaned = cleaned.replace('O', '0').replace('I', '1').replace('S', '5').replace('B', '8')
        # But restore letters where they should be letters (first 2 chars are state code)
        # Try with original cleaned text first
        original_cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
        
        match = re.search(r'([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})', original_cleaned)
        if match:
            return f'{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}'
        
        # BH series match
        bh_match = re.search(r'(\d{2})(BH)(\d{4})([A-Z]{1,2})', original_cleaned)
        if bh_match:
            return f'{bh_match.group(1)} {bh_match.group(2)} {bh_match.group(3)} {bh_match.group(4)}'
            
        return original_cleaned if len(original_cleaned) >= 4 else None

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
