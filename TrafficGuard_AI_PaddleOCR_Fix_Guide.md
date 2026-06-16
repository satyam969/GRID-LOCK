# TrafficGuard AI — PaddleOCR Fix Guide

**Issue:** PaddleOCR fails to load or returns no plate text  
**Root Cause:** 3 bugs identified in the current `OCREngine` class  
**Scope:** Complete diagnosis, fixed code, standalone test, fallback plan

---

## Table of Contents

1. [Problem Diagnosis — 3 Bugs Found](#1-problem-diagnosis)
2. [Quick Fix Steps (Run These First)](#2-quick-fix-steps)
3. [Complete Fixed OCREngine Class](#3-complete-fixed-ocrengine-class)
4. [Standalone Test Script](#4-standalone-test-script)
5. [Integration Check — Detector ↔ OCR Bbox Format](#5-integration-check)
6. [Fallback — Fixed EasyOCR (If PaddleOCR Won't Install)](#6-fallback-fixed-easyocr)
7. [Troubleshooting FAQ](#7-troubleshooting-faq)

---

## 1. Problem Diagnosis

Three bugs were found in your current `OCREngine` class that cause PaddleOCR to silently fail:

### Bug 1: Missing `use_gpu=False` (Most Likely Cause)

Your `PaddleOCR()` initialization is **missing `use_gpu=False`**. Without it, PaddleOCR defaults to trying GPU. On a CPU-only machine, it either:
- Fails silently and returns `None` for every OCR call
- Throws an error that your `except Exception` block swallows without you seeing it

```python
# YOUR CODE (broken — no use_gpu flag):
cls._ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    det_db_thresh=0.3,
    rec_batch_num=6,
)

# FIX — add use_gpu=False:
cls._ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=False,       # ← CRITICAL FIX
    show_log=True,       # ← Shows download/load errors
    det_db_thresh=0.3,
    rec_batch_num=6,
)
```

### Bug 2: Corrupted / Incomplete Model Cache

PaddleOCR auto-downloads ~100MB of models on first use to `~/.paddleocr/`. If your network blocked it, timed out, or you killed the process mid-download, the models are **corrupted**. Every subsequent call silently returns `None` because the models can't load.

**Evidence:** You see `✅ PaddleOCR ready` in logs (because the `except` block didn't fire), but OCR always returns `None`.

### Bug 3: PaddlePaddle Version Mismatch

PaddleOCR 2.7+ requires `paddlepaddle >= 2.5`. PaddleOCR 3.x requires `paddlepaddle >= 3.0`. If you installed the wrong version, import succeeds but inference silently fails.

**How to check:**
```bash
python -c "import paddle; print(paddle.__version__)"
python -c "import paddleocr; print(paddleocr.__version__)"
```

**Compatible combos:**

| paddleocr | paddlepaddle | Status |
|-----------|-------------|--------|
| 2.7.x | 2.5.x - 2.6.x | ✅ Works on CPU |
| 2.8.x | 2.5.x - 2.6.x | ✅ Works on CPU |
| 3.0.x - 3.2.x | 3.0.x - 3.2.x | ✅ Works (latest) |
| 3.x | 2.x | ❌ Silent failure |
| 2.x | 3.x | ❌ API mismatch |

---

## 2. Quick Fix Steps

Run these commands **in order** before changing any code:

```bash
# Step 1: Check what you have
python -c "import paddle; print('Paddle:', paddle.__version__)"
python -c "import paddleocr; print('PaddleOCR:', paddleocr.__version__)"

# Step 2: Clean uninstall everything
pip uninstall paddlepaddle paddlepaddle-gpu paddleocr -y

# Step 3: Delete corrupted model cache
# Linux / Mac:
rm -rf ~/.paddleocr/
# Windows (PowerShell):
# Remove-Item -Recurse -Force $env:USERPROFILE\.paddleocr
# Windows (CMD):
# rmdir /s /q %USERPROFILE%\.paddleocr

# Step 4: Fresh install (OPTION A — Stable combo, recommended)
pip install paddlepaddle==2.6.2
pip install paddleocr==2.8.1

# Step 4: Fresh install (OPTION B — Latest 3.x)
# pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# pip install paddleocr

# Step 5: Verify installation
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(use_gpu=False, lang='en', show_log=True); print('SUCCESS')"
```

> **Note:** Step 5 will download models (~100MB) on first run. Make sure you have internet access and wait for it to complete fully. You'll see download progress bars in the terminal.

---

## 3. Complete Fixed OCREngine Class

Drop this into `backend/app/core/ocr_engine.py` — it replaces your current file entirely:

```python
"""
OCR Engine — License plate text extraction using PaddleOCR (PP-OCRv4).
Replaces EasyOCR to fix duplicate text detection issues.

FIXES APPLIED:
  1. Added use_gpu=False (prevents silent GPU failure on CPU machines)
  2. Added show_log=True during init (shows download/load errors)
  3. Added full traceback on exceptions (no more silent swallowing)
  4. Added self-test on first load (confirms OCR actually works)
  5. Added bbox format validation (catches detector↔OCR mismatches)
"""
import re
import logging
import traceback
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
    _initialized = False
    confidence_threshold = 0.30

    @classmethod
    def _get_ocr(cls):
        if cls._ocr is None and not cls._initialized:
            cls._initialized = True  # Prevent retry loops on failure
            try:
                from paddleocr import PaddleOCR

                logger.info("Loading PaddleOCR model (first use)...")
                logger.info("This will download ~100MB of models on first run.")

                cls._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=False,          # ← FIX 1: Force CPU mode
                    show_log=True,           # ← FIX 2: Show errors
                    det_db_thresh=0.3,
                    rec_batch_num=6,
                )

                # FIX 4: Self-test — confirm OCR actually works
                test_img = np.ones((60, 250, 3), dtype=np.uint8) * 255
                cv2.putText(
                    test_img, 'TEST123', (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2
                )
                test_result = cls._ocr.ocr(test_img, cls=True)

                if test_result and test_result[0]:
                    logger.info("PaddleOCR self-test PASSED")
                else:
                    logger.warning("PaddleOCR self-test returned empty — models may be corrupted")
                    logger.warning("Try: rm -rf ~/.paddleocr/ and restart")

                logger.info("PaddleOCR ready")

            except ImportError:
                logger.error("PaddleOCR not installed. Run:")
                logger.error("  pip install paddlepaddle==2.6.2")
                logger.error("  pip install paddleocr==2.8.1")
                cls._ocr = None
            except Exception as e:
                # FIX 3: Print FULL traceback instead of swallowing
                logger.error(f"PaddleOCR failed to load: {e}")
                logger.error(traceback.format_exc())
                cls._ocr = None

        return cls._ocr

    @staticmethod
    def preprocess_plate(plate_img: np.ndarray) -> np.ndarray:
        """Apply CLAHE + Bilateral Filtering for better OCR accuracy."""
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
            logger.warning("OCR engine not available — skipping plate text extraction")
            return None, 0.0

        # Crop plate region if bbox is provided
        if plate_bbox:
            # FIX 5: Validate bbox format before using it
            required_keys = {'x1', 'y1', 'x2', 'y2'}
            if not required_keys.issubset(plate_bbox.keys()):
                logger.error(f"Invalid plate_bbox format. Expected keys: {required_keys}, Got: {plate_bbox.keys()}")
                logger.error(f"Full bbox value: {plate_bbox}")
                return None, 0.0

            h, w = img.shape[:2]
            padding = 5
            x1 = max(0, int(plate_bbox['x1']) - padding)
            y1 = max(0, int(plate_bbox['y1']) - padding)
            x2 = min(w, int(plate_bbox['x2']) + padding)
            y2 = min(h, int(plate_bbox['y2']) + padding)

            # Validate crop dimensions
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid crop dimensions: ({x1},{y1})-({x2},{y2})")
                return None, 0.0

            plate_img = img[y1:y2, x1:x2]
            logger.debug(f"Cropped plate region: ({x1},{y1})-({x2},{y2}), shape={plate_img.shape}")
        else:
            plate_img = img

        if plate_img is None or plate_img.size == 0:
            logger.warning("Empty plate image — skipping OCR")
            return None, 0.0

        # Ensure minimum size for OCR (too small = no detection)
        min_h, min_w = 20, 60
        if plate_img.shape[0] < min_h or plate_img.shape[1] < min_w:
            logger.warning(f"Plate image too small: {plate_img.shape}. Resizing...")
            scale = max(min_h / plate_img.shape[0], min_w / plate_img.shape[1], 2.0)
            plate_img = cv2.resize(
                plate_img, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

        # Apply CLAHE + Bilateral preprocessing
        processed = cls.preprocess_plate(plate_img)

        try:
            results = ocr.ocr(processed, cls=True)
        except Exception as e:
            logger.error(f"PaddleOCR inference failed: {e}")
            logger.error(traceback.format_exc())
            return None, 0.0

        # Check for empty results
        if not results or not results[0]:
            logger.debug("PaddleOCR returned no text for this plate crop")
            return None, 0.0

        # Pick SINGLE highest-confidence line (fixes EasyOCR duplicate bug)
        best_text, best_conf = '', 0.0
        for line in results[0]:
            try:
                text, conf = line[1][0], line[1][1]
                if conf > best_conf:
                    best_text, best_conf = text, conf
            except (IndexError, TypeError) as e:
                logger.debug(f"Skipping malformed OCR line: {line}, error: {e}")
                continue

        logger.info(f"Raw OCR result: '{best_text}' (conf={best_conf:.2%})")

        if best_conf < cls.confidence_threshold:
            logger.debug(f"OCR confidence {best_conf:.2%} below threshold {cls.confidence_threshold:.0%}")
            return None, 0.0

        cleaned = cls._validate_indian_plate(best_text)
        logger.info(f"Cleaned plate text: '{cleaned}'")
        return cleaned or best_text, round(best_conf, 4)

    @staticmethod
    def _validate_indian_plate(raw_text: str) -> Optional[str]:
        """Post-process OCR output to match Indian plate format."""
        if not raw_text:
            return None

        cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

        # Standard Indian format: XX 00 XX 0000
        match = re.search(r'([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})', cleaned)
        if match:
            return f'{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}'

        # BH series (Bharat series): 00 BH 0000 XX
        bh_match = re.search(r'(\d{2})(BH)(\d{4})([A-Z]{1,2})', cleaned)
        if bh_match:
            return f'{bh_match.group(1)} {bh_match.group(2)} {bh_match.group(3)} {bh_match.group(4)}'

        # If no pattern matches, return cleaned text as-is
        return cleaned if len(cleaned) >= 4 else None

    @classmethod
    def extract_from_detections(
        cls, img: np.ndarray, plate_detections: List[Dict]
    ) -> Tuple[Optional[str], float]:
        """
        Given plate detection bboxes, pick the largest plate and run OCR.
        """
        if not plate_detections:
            logger.debug("No plate detections provided")
            return None, 0.0

        # Pick largest detected plate (most likely to be readable)
        largest_plate = max(
            plate_detections,
            key=lambda d: (
                (d['bbox']['x2'] - d['bbox']['x1']) *
                (d['bbox']['y2'] - d['bbox']['y1'])
            ),
        )
        logger.debug(f"Selected largest plate: {largest_plate['bbox']}")
        return cls.extract_plate_text(img, largest_plate['bbox'])


# Module-level singleton
ocr_engine = OCREngine()
```

---

## 4. Standalone Test Script

Save this as `test_paddle_ocr.py` and run it **outside your project** to confirm PaddleOCR works:

```python
#!/usr/bin/env python3
"""
test_paddle_ocr.py — Standalone PaddleOCR verification script.
Run this BEFORE integrating into TrafficGuard.

Usage: python test_paddle_ocr.py
"""
import sys
import cv2
import numpy as np

print('=' * 60)
print('PaddleOCR Standalone Verification Test')
print('=' * 60)

# Test 1: Check paddle version
print('\n[TEST 1] Checking PaddlePaddle installation...')
try:
    import paddle
    print(f'  PaddlePaddle version: {paddle.__version__}')
    print(f'  CUDA available: {paddle.device.is_compiled_with_cuda()}')
except ImportError:
    print('  ERROR: PaddlePaddle not installed!')
    print('  Run: pip install paddlepaddle==2.6.2')
    sys.exit(1)

# Test 2: Check paddleocr version
print('\n[TEST 2] Checking PaddleOCR installation...')
try:
    import paddleocr
    print(f'  PaddleOCR version: {paddleocr.__version__}')
except ImportError:
    print('  ERROR: PaddleOCR not installed!')
    print('  Run: pip install paddleocr==2.8.1')
    sys.exit(1)

# Test 3: Initialize OCR engine
print('\n[TEST 3] Initializing PaddleOCR (use_gpu=False)...')
print('  (This downloads ~100MB of models on first run — please wait)')
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=False,
        show_log=True,
    )
    print('  PaddleOCR initialized successfully!')
except Exception as e:
    print(f'  ERROR: Failed to initialize PaddleOCR: {e}')
    print('  Try: rm -rf ~/.paddleocr/ and reinstall')
    sys.exit(1)

# Test 4: OCR on synthetic plate image
print('\n[TEST 4] Running OCR on synthetic license plate...')
img = np.ones((80, 300, 3), dtype=np.uint8) * 255  # White background
cv2.putText(img, 'WB06J2431', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

result = ocr.ocr(img, cls=True)
if result and result[0]:
    for line in result[0]:
        text = line[1][0]
        conf = line[1][1]
        print(f'  Detected: "{text}" (confidence: {conf:.2%})')
    print('  PASSED — OCR is detecting text!')
else:
    print('  WARNING — OCR returned empty on synthetic image.')
    print('  This might be OK (synthetic text can be hard). Testing with CLAHE...')

# Test 5: OCR with CLAHE preprocessing
print('\n[TEST 5] Running OCR with CLAHE preprocessing...')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
filtered = cv2.bilateralFilter(enhanced, 11, 17, 17)
processed = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

result2 = ocr.ocr(processed, cls=True)
if result2 and result2[0]:
    for line in result2[0]:
        text = line[1][0]
        conf = line[1][1]
        print(f'  Detected: "{text}" (confidence: {conf:.2%})')
    print('  PASSED — OCR with CLAHE works!')
else:
    print('  WARNING — CLAHE test also returned empty.')

# Test 6: Save test image for manual verification
cv2.imwrite('test_plate_output.jpg', img)
print('\n[TEST 6] Saved test image to: test_plate_output.jpg')

print('\n' + '=' * 60)
print('TEST COMPLETE')
print('If Tests 4 or 5 passed, PaddleOCR is working.')
print('If both failed, try: rm -rf ~/.paddleocr/ and rerun.')
print('=' * 60)
```

**Run it:**
```bash
python test_paddle_ocr.py
```

---

## 5. Integration Check — Detector ↔ OCR Bbox Format

If PaddleOCR works in the standalone test but NOT in your project, the problem is likely the **bbox format** mismatch between your plate detector and the OCR engine.

### Required Bbox Format

Your `extract_plate_text()` expects this dictionary format:
```python
plate_bbox = {
    'x1': 150.0,   # Left edge
    'y1': 400.0,   # Top edge
    'x2': 350.0,   # Right edge
    'y2': 450.0,   # Bottom edge
}
```

### Common Bbox Format Mismatches

| Detector Output Format | Expected by OCR | Fix |
|---|---|---|
| `[x1, y1, x2, y2]` (list) | `{'x1':..., 'y1':..., 'x2':..., 'y2':...}` (dict) | `dict(zip(['x1','y1','x2','y2'], bbox))` |
| `[x, y, w, h]` (YOLO format) | `{'x1':..., 'y1':..., 'x2':..., 'y2':...}` | `{'x1': x, 'y1': y, 'x2': x+w, 'y2': y+h}` |
| `xyxy` tensor | `{'x1':..., 'y1':..., 'x2':..., 'y2':...}` | `{'x1': float(t[0]), ...}` |

### Debug: Add This to Your Violation Engine

```python
# Add this debug line RIGHT BEFORE calling ocr_engine
print(f'[DEBUG] Plate detections: {plate_detections}')
print(f'[DEBUG] Bbox format: {plate_detections[0]["bbox"] if plate_detections else "NONE"}')

plate_text, plate_conf = ocr_engine.extract_from_detections(img, plate_detections)
print(f'[DEBUG] OCR result: text={plate_text}, conf={plate_conf}')
```

---

## 6. Fallback — Fixed EasyOCR (If PaddleOCR Won't Install)

If PaddleOCR absolutely refuses to work on your machine, here's a **fixed version of EasyOCR** that solves the duplicate concatenation bug:

```python
"""
Fallback OCR Engine using EasyOCR.
KEY FIX: Picks single highest-confidence line instead of concatenating all.
"""
import re
import logging
import cv2
import numpy as np
import easyocr
from typing import Optional, Tuple, List, Dict

logger = logging.getLogger(__name__)


class OCREngineFallback:
    """Fixed EasyOCR — picks single best line instead of concatenating."""

    _reader = None
    confidence_threshold = 0.30

    @classmethod
    def _get_reader(cls):
        if cls._reader is None:
            cls._reader = easyocr.Reader(['en'], gpu=False)
            logger.info('EasyOCR (fallback) loaded')
        return cls._reader

    @staticmethod
    def preprocess_plate(plate_img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.bilateralFilter(enhanced, 11, 17, 17)

    @classmethod
    def extract_plate_text(
        cls, img: np.ndarray, plate_bbox: Optional[Dict] = None
    ) -> Tuple[Optional[str], float]:

        reader = cls._get_reader()
        if reader is None:
            return None, 0.0

        # Crop plate region
        if plate_bbox:
            h, w = img.shape[:2]
            x1 = max(0, int(plate_bbox['x1']) - 5)
            y1 = max(0, int(plate_bbox['y1']) - 5)
            x2 = min(w, int(plate_bbox['x2']) + 5)
            y2 = min(h, int(plate_bbox['y2']) + 5)
            plate_img = img[y1:y2, x1:x2]
        else:
            plate_img = img

        if plate_img.size == 0:
            return None, 0.0

        processed = cls.preprocess_plate(plate_img)

        # Run EasyOCR
        results = reader.readtext(processed)

        if not results:
            return None, 0.0

        # KEY FIX: Pick SINGLE highest-confidence result
        # (instead of concatenating all — which caused 'NB062431WBHB06J2431')
        best_text, best_conf = '', 0.0
        for (bbox, text, conf) in results:
            if conf > best_conf:
                best_text, best_conf = text, conf

        if best_conf < cls.confidence_threshold:
            return None, 0.0

        # Clean for Indian plate format
        cleaned = re.sub(r'[^A-Z0-9]', '', best_text.upper())
        match = re.search(r'([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})', cleaned)
        if match:
            cleaned = f'{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}'

        return cleaned, round(best_conf, 4)


# Use this if PaddleOCR fails:
# from app.core.ocr_engine_fallback import OCREngineFallback as OCREngine
```

---

## 7. Troubleshooting FAQ

| Error / Symptom | Cause | Fix |
|---|---|---|
| `PaddleOCR ready` in logs but OCR always returns `None` | Models corrupted or GPU mode failing silently | `rm -rf ~/.paddleocr/` + add `use_gpu=False` |
| `ModuleNotFoundError: No module named 'paddle'` | PaddlePaddle not installed | `pip install paddlepaddle==2.6.2` |
| `ModuleNotFoundError: No module named 'paddleocr'` | PaddleOCR not installed | `pip install paddleocr==2.8.1` |
| `AttributeError: 'AnalysisConfig' has no attribute 'set_optimization_level'` | paddleocr 3.x with paddlepaddle 2.x | Either both 2.x or both 3.x |
| `RuntimeError: No valid PaddlePaddle model found` | Corrupted model cache | `rm -rf ~/.paddleocr/` (or `~/.paddlex/`) |
| Models downloading every time server starts | Cache directory permissions issue | `chmod -R 755 ~/.paddleocr/` |
| OCR works standalone but returns `None` in project | Bbox format mismatch (see Section 5) | Add debug prints, check dict keys |
| `plate_img.size == 0` | Plate crop is outside image bounds | Check x1/y1/x2/y2 are within img shape |
| Very slow first inference (~30s) | Model compilation on first run | Normal — subsequent calls are fast (~2-3s) |
| OCR returns partial text like `9122` | Plate crop too small / low resolution | Auto-upscale (added in fixed class above) |
| Wrong characters (`W→N`, `B→H`) | EasyOCR CRAFT detector issue | Switch to PaddleOCR (DB detector is better) |
| Duplicate text `NB062431WBHB06J2431` | EasyOCR concatenating multiple text regions | Pick single highest-confidence line (fixed in both classes above) |

### Still Stuck? Debug Checklist

```bash
# 1. Check versions match
python -c "import paddle; print(paddle.__version__)"     # Should be 2.6.x or 3.x
python -c "import paddleocr; print(paddleocr.__version__)" # Match with paddle

# 2. Check model cache exists and is complete
ls -la ~/.paddleocr/whl/   # Should have det/, rec/, cls/ folders

# 3. Run standalone test
python test_paddle_ocr.py

# 4. Check server logs for errors
uvicorn main:app --reload --log-level debug 2>&1 | grep -i 'ocr\|paddle\|error'
```

---

*If PaddleOCR absolutely won't work on your machine, use the EasyOCR fallback in Section 6 — it fixes the duplicate concatenation bug that caused `NB062431WBHB06J2431` and will give you clean `WB 06 J 2431` output.*