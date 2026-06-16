"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Any, Dict
from datetime import datetime
from app.models.violation import ViolationType, ViolationStatus, VehicleType


# ─── Detection Result Schemas ────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    class_id: int


class DetectedViolation(BaseModel):
    violation_type: ViolationType
    confidence: float
    severity: str  # low / medium / high
    description: str
    bbox: Optional[BoundingBox] = None


class AnalysisResult(BaseModel):
    """Returned by the analyze endpoint"""
    violations: List[DetectedViolation]
    vehicle_type: VehicleType
    license_plate: Optional[str] = None
    plate_confidence: Optional[float] = None
    person_count: Optional[int] = None
    all_detections: List[BoundingBox] = []
    annotated_image_url: Optional[str] = None
    inference_time_ms: float
    violation_count: int


# ─── Violation Record Schemas ─────────────────────────────────────────────────

class ViolationCreate(BaseModel):
    violation_type: ViolationType
    violation_types: Optional[List[str]] = None
    confidence: float
    severity: str = "medium"
    vehicle_type: VehicleType = VehicleType.UNKNOWN
    license_plate: Optional[str] = None
    plate_confidence: Optional[float] = None
    original_image_path: Optional[str] = None
    annotated_image_path: Optional[str] = None
    camera_id: Optional[str] = None
    location: Optional[str] = None
    detection_metadata: Optional[Dict[str, Any]] = None
    inference_time_ms: Optional[float] = None
    person_count: Optional[int] = None


class ViolationResponse(ViolationCreate):
    id: str
    timestamp: datetime
    status: ViolationStatus

    model_config = ConfigDict(from_attributes=True)


class ViolationUpdate(BaseModel):
    status: Optional[ViolationStatus] = None
    notes: Optional[str] = None


class PaginatedViolations(BaseModel):
    items: List[ViolationResponse]
    total: int
    page: int
    page_size: int
    pages: int


# ─── Analytics Schemas ────────────────────────────────────────────────────────

class ViolationCount(BaseModel):
    violation_type: str
    count: int
    percentage: float


class DailyTrend(BaseModel):
    date: str
    count: int
    violation_types: Dict[str, int]


class AnalyticsSummary(BaseModel):
    total_violations: int
    today_violations: int
    pending_review: int
    top_violations: List[ViolationCount]
    vehicle_distribution: Dict[str, int]
    avg_confidence: float


# ─── Health Schema ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]
    database: str
