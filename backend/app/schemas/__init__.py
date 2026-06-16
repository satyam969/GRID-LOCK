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
    id: int
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


# ─── Vehicle Schemas ──────────────────────────────────────────────────────────

class VehicleResponse(BaseModel):
    id: int
    plate_number: str
    state_code: Optional[str] = None
    district_code: Optional[str] = None
    vehicle_type: Optional[str] = None
    owner_name: Optional[str] = None
    total_violations: int
    first_seen: datetime
    last_seen: datetime
    is_repeat_offender: bool
    flagged: bool

    model_config = ConfigDict(from_attributes=True)


# ─── Challan Schemas ──────────────────────────────────────────────────────────

class ChallanGenerateRequest(BaseModel):
    violation_id: int

class ChallanResponse(BaseModel):
    id: int
    violation_id: int
    vehicle_id: Optional[int] = None
    challan_number: str
    fine_amount: float
    payment_status: str
    payment_date: Optional[datetime] = None
    due_date: datetime
    issued_by: Optional[str] = None
    created_at: datetime
    
    # Extra relations to hydrate frontend
    plate_number: Optional[str] = None
    violation_type: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ─── Camera Schemas ───────────────────────────────────────────────────────────

class CameraCreate(BaseModel):
    name: str
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    zone: str
    direction: Optional[str] = None
    status: str = "active"

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    zone: Optional[str] = None
    direction: Optional[str] = None
    status: Optional[str] = None

class CameraResponse(CameraCreate):
    id: int
    installed_date: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ─── Settings Schemas ─────────────────────────────────────────────────────────

class AppSettings(BaseModel):
    # Detection
    conf_general: float
    conf_helmet: float
    conf_seatbelt: float
    conf_plate: float
    
    # Behavior
    enable_clahe: bool
    auto_generate_challans: bool
    
    # Notifications
    email_alerts: bool
    admin_email: str


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
    resolution_rate: float
    top_violations: List[ViolationCount]
    vehicle_distribution: Dict[str, int]
    avg_confidence: float


# ─── Health Schema ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, bool]
    database: str
