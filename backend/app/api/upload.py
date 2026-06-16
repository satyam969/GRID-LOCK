"""
Upload & Analyze API — Core endpoint for traffic violation detection.
"""
import time
import uuid
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import aiofiles

from app.config import settings
from app.database import get_db
from app.models.violation import Violation, ViolationType, VehicleType
from app.schemas import AnalysisResult, DetectedViolation, BoundingBox, ViolationCreate
from app.core.preprocessor import ImagePreprocessor
from app.core.violation_engine import violation_engine
from app.core.ocr_engine import ocr_engine
from app.core.annotator import annotator
from app.core.detector import detector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analyze", tags=["Analysis"])


async def _save_upload(file: UploadFile) -> tuple[Path, str]:
    """Save uploaded file to uploads dir, return (path, filename)."""
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{suffix}"
    path = settings.UPLOADS_DIR / filename
    content = await file.read()
    async with aiofiles.open(path, "wb") as f:
        await f.write(content)
    return path, filename, content


async def _persist_violation(
    db: AsyncSession,
    violation: dict,
    vehicle_type: VehicleType,
    license_plate: Optional[str],
    plate_confidence: Optional[float],
    original_path: str,
    annotated_path: Optional[str],
    inference_ms: float,
    person_count: Optional[int],
    all_violations: List[dict],
) -> Violation:
    """Save a detected violation to database."""
    v_types = [v["violation_type"].value if hasattr(v["violation_type"], "value")
               else v["violation_type"] for v in all_violations]

    v_type = violation["violation_type"]
    if hasattr(v_type, "value"):
        v_type = v_type.value

    record = Violation(
        violation_type=v_type,
        violation_types=v_types,
        confidence=violation["confidence"],
        severity=violation.get("severity", "medium"),
        vehicle_type=vehicle_type.value if hasattr(vehicle_type, "value") else vehicle_type,
        license_plate=license_plate,
        plate_confidence=plate_confidence,
        original_image_path=original_path,
        annotated_image_path=annotated_path,
        inference_time_ms=inference_ms,
        person_count=person_count,
        detection_metadata=violation.get("bbox"),
    )
    db.add(record)
    await db.flush()
    return record


@router.post("/image", response_model=AnalysisResult)
async def analyze_image(
    file: UploadFile = File(..., description="Traffic image (JPG/PNG/WEBP)"),
    camera_id: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    enhance_contrast: bool = Form(True),
    check_stop_line: bool = Form(False),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze a single traffic image for violations.
    Returns detected violations, annotated image URL, license plate, and metadata.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (JPEG/PNG/WEBP)")

    start_time = time.time()

    # ── 1. Save and preprocess ─────────────────────────────────────────────
    upload_path, filename, content = await _save_upload(file)

    try:
        img = ImagePreprocessor.preprocess(
            content,
            enhance_contrast=enhance_contrast,
            remove_shadow=False,
        )
    except Exception as e:
        raise HTTPException(422, f"Image preprocessing failed: {e}")

    # ── 2. Run violation engine ────────────────────────────────────────────
    try:
        result = violation_engine.analyze(img, check_stop_line=check_stop_line)
    except Exception as e:
        logger.error(f"Violation engine error: {e}", exc_info=True)
        raise HTTPException(500, f"Detection failed: {e}")

    # ── 3. License plate OCR ───────────────────────────────────────────────
    plate_detections = detector.detect_license_plate(img)
    plate_text, plate_conf = ocr_engine.extract_from_detections(img, plate_detections)

    # ── 4. Annotate evidence image ─────────────────────────────────────────
    annotated_url = None
    try:
        canvas = annotator.annotate(
            img,
            violations=result["violations"],
            all_detections=result["all_detections"],
            license_plate=plate_text,
            person_count=result["person_count"],
        )
        annotated_url = annotator.save_annotated(canvas, filename)
    except Exception as e:
        logger.warning(f"Annotation failed (non-fatal): {e}")

    inference_ms = (time.time() - start_time) * 1000

    # ── 5. Persist to DB ───────────────────────────────────────────────────
    violations = result["violations"]
    if violations:
        try:
            await _persist_violation(
                db=db,
                violation=violations[0],  # Primary violation
                vehicle_type=result["vehicle_type"],
                license_plate=plate_text,
                plate_confidence=plate_conf if plate_text else None,
                original_path=str(upload_path),
                annotated_path=annotated_url,
                inference_ms=inference_ms,
                person_count=result["person_count"],
                all_violations=violations,
            )
        except Exception as e:
            logger.error(f"DB persist error: {e}")

    # ── 6. Build response ──────────────────────────────────────────────────
    response_violations = []
    for v in violations:
        bbox = v.get("bbox")
        v_type = v["violation_type"]
        if hasattr(v_type, "value"):
            v_type = v_type.value

        response_violations.append(DetectedViolation(
            violation_type=v_type,
            confidence=v["confidence"],
            severity=v.get("severity", "medium"),
            description=v.get("description", ""),
            bbox=BoundingBox(
                x1=bbox["x1"], y1=bbox["y1"],
                x2=bbox["x2"], y2=bbox["y2"],
                confidence=v["confidence"],
                class_name=v_type,
                class_id=0,
            ) if bbox else None,
        ))

    all_bbox = []
    for det in result.get("all_detections", []):
        b = det["bbox"]
        all_bbox.append(BoundingBox(
            x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
            confidence=det["confidence"],
            class_name=det["class_name"],
            class_id=det["class_id"],
        ))

    vehicle_type = result["vehicle_type"]
    if hasattr(vehicle_type, "value"):
        vehicle_type = vehicle_type.value

    return AnalysisResult(
        violations=response_violations,
        vehicle_type=vehicle_type,
        license_plate=plate_text,
        plate_confidence=plate_conf if plate_text else None,
        person_count=result["person_count"],
        all_detections=all_bbox,
        annotated_image_url=annotated_url,
        inference_time_ms=round(inference_ms, 2),
        violation_count=len(violations),
    )


@router.post("/batch")
async def analyze_batch(
    files: List[UploadFile] = File(..., description="Multiple traffic images"),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze multiple images. Returns list of AnalysisResult per image.
    """
    if len(files) > 50:
        raise HTTPException(400, "Maximum 50 images per batch")

    results = []
    for f in files:
        try:
            # Re-use single image endpoint logic
            content = await f.read()
            await f.seek(0)
            single_result = await analyze_image(
                file=f, camera_id=None, location=None,
                enhance_contrast=True, check_stop_line=False, db=db
            )
            results.append({"filename": f.filename, "result": single_result, "error": None})
        except Exception as e:
            results.append({"filename": f.filename, "result": None, "error": str(e)})

    total = len(results)
    violations_found = sum(1 for r in results if r["result"] and r["result"].violation_count > 0)

    return {
        "total_images": total,
        "violations_found": violations_found,
        "results": results,
    }
