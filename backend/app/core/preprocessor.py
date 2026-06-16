"""
Image Preprocessor
Handles: CLAHE enhancement, denoising, shadow removal, normalization.
Runs before YOLO inference to improve detection in poor conditions.
"""
import cv2
import numpy as np
from PIL import Image
import io
from typing import Union
from pathlib import Path


class ImagePreprocessor:

    @staticmethod
    def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
        """Convert uploaded bytes to OpenCV BGR array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes")
        return img

    @staticmethod
    def cv2_to_bytes(img: np.ndarray, format: str = ".jpg") -> bytes:
        """Convert OpenCV image to bytes."""
        success, encoded = cv2.imencode(format, img)
        if not success:
            raise ValueError("Could not encode image")
        return encoded.tobytes()

    @staticmethod
    def apply_clahe(img: np.ndarray) -> np.ndarray:
        """
        Contrast Limited Adaptive Histogram Equalization.
        Dramatically improves low-light and shadow visibility.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(img: np.ndarray, strength: int = 7) -> np.ndarray:
        """
        Fast Non-Local Means Denoising — handles rain/noise artifacts.
        strength=7 is a balanced setting for traffic images.
        """
        return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)

    @staticmethod
    def remove_shadows(img: np.ndarray) -> np.ndarray:
        """
        HSV-based shadow normalization.
        Brightens underexposed regions without blowing out highlights.
        """
        rgb_planes = cv2.split(img)
        result_planes = []
        for plane in rgb_planes:
            dilated = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(plane, bg_img)
            result_planes.append(diff)
        return cv2.merge(result_planes)

    @staticmethod
    def sharpen(img: np.ndarray) -> np.ndarray:
        """Unsharp masking — useful for motion-blurred images."""
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    @staticmethod
    def normalize_size(img: np.ndarray, target_size: int = 1280) -> np.ndarray:
        """
        Resize while maintaining aspect ratio.
        Upsamples small images, downsamples huge ones.
        YOLO works best with 640 or 1280px images.
        """
        h, w = img.shape[:2]
        if max(h, w) <= target_size and min(h, w) >= 320:
            return img  # Already good size
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @classmethod
    def preprocess(
        cls,
        image_bytes: bytes,
        enhance_contrast: bool = True,
        remove_shadow: bool = False,
        sharpen_blur: bool = False,
    ) -> np.ndarray:
        """
        Full preprocessing pipeline.
        Returns OpenCV BGR array ready for YOLO inference.
        """
        img = cls.bytes_to_cv2(image_bytes)
        img = cls.normalize_size(img)

        if enhance_contrast:
            img = cls.apply_clahe(img)

        if remove_shadow:
            img = cls.remove_shadows(img)

        if sharpen_blur:
            img = cls.sharpen(img)

        return img

    @classmethod
    def preprocess_plate_crop(cls, img: np.ndarray, bbox: tuple) -> np.ndarray:
        """
        Special preprocessing for license plate crops before OCR.
        Aggressive enhancement for character readability.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Add padding
        pad = 10
        h, w = img.shape[:2]
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        plate_crop = img[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return img

        # Upscale for better OCR
        plate_crop = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # CLAHE
        plate_crop = cls.apply_clahe(plate_crop)

        # Bilateral filter — preserves edges (character edges) while denoising
        plate_crop = cv2.bilateralFilter(plate_crop, 11, 75, 75)

        return plate_crop
