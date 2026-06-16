"""
OCR Engine — License plate text extraction using EasyOCR.
Includes Indian number plate format validation and confidence scoring.
"""
import re
import logging
from typing import Optional, Tuple, List, Dict
import numpy as np

from app.config import settings
from app.core.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class OCREngine:
    """
    Wraps EasyOCR for license plate text extraction.
    Lazy-loaded to avoid slow startup; initialized on first use.
    """

    _reader = None

    @classmethod
    def _get_reader(cls):
        if cls._reader is None:
            try:
                import easyocr
                logger.info("Loading EasyOCR model (first use)...")
                cls._reader = easyocr.Reader(settings.OCR_LANGUAGES, gpu=False, verbose=False)
                logger.info("✅ EasyOCR ready")
            except Exception as e:
                logger.error(f"❌ EasyOCR failed to load: {e}")
                cls._reader = None
        return cls._reader

    @staticmethod
    def preprocess_plate(crop):
        import cv2
        # Resize to standard height
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop
        target_h = 100
        scale = target_h / h
        resized = cv2.resize(crop, (int(w * scale), target_h))

        # Grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # CLAHE for shadow handling
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter (edge-preserving noise reduction)
        filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)

        # Otsu binarization
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological close (fill gaps in characters)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return closed

    @classmethod
    def extract_plate_text(
        cls,
        img: np.ndarray,
        plate_bbox: Optional[Dict] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Extract text from the license plate region.

        Args:
            img: Full image (BGR numpy array)
            plate_bbox: Optional bbox dict {x1,y1,x2,y2} of detected plate region

        Returns:
            (plate_text, confidence) — text is None if nothing detected
        """
        reader = cls._get_reader()
        if reader is None:
            return None, 0.0

        # Crop and preprocess plate region
        if plate_bbox:
            plate_img = ImagePreprocessor.preprocess_plate_crop(
                img,
                (plate_bbox["x1"], plate_bbox["y1"], plate_bbox["x2"], plate_bbox["y2"])
            )
        else:
            plate_img = img

        try:
            processed_plate = cls.preprocess_plate(plate_img)
            results = reader.readtext(processed_plate, detail=1, paragraph=False)
            results += reader.readtext(plate_img, detail=1, paragraph=False) # Fallback to raw
        except Exception as e:
            logger.error(f"EasyOCR readtext failed: {e}")
            return None, 0.0

        if not results:
            return None, 0.0

        # Combine all text pieces, pick highest confidence
        texts = []
        confidences = []
        for (_, text, conf) in results:
            clean = re.sub(r"[^A-Z0-9]", "", text.upper())
            if clean and conf > 0.30:
                texts.append(clean)
                confidences.append(conf)

        if not texts:
            return None, 0.0

        combined_text = "".join(texts)
        avg_confidence = sum(confidences) / len(confidences)

        # Validate and format Indian plate pattern
        validated = cls._validate_indian_plate(combined_text)
        return validated or combined_text, round(avg_confidence, 4)

    @staticmethod
    def _validate_indian_plate(text: str) -> Optional[str]:
        """
        Match Indian number plate formats:
        - Standard: MH12AB1234
        - New BH series: 22BH1234AA
        """
        patterns = [
            r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$",   # Standard state plates
            r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$",             # BH series
            r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$",       # Common format
        ]
        text = text.upper().replace(" ", "").replace("-", "")

        for pattern in patterns:
            if re.match(pattern, text):
                return text

        # Return None if no pattern matches (still return raw text from caller)
        return None

    @classmethod
    def extract_from_detections(
        cls, img: np.ndarray, plate_detections: List[Dict]
    ) -> Tuple[Optional[str], float]:
        """
        Given plate detection bboxes, pick the largest plate and run OCR.
        Returns (plate_text, confidence).
        """
        if not plate_detections:
            return None, 0.0  # Do not run OCR on the entire image

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
