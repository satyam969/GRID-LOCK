"""
ORM Models — Violation and Camera entities
"""
import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum
from typing import Optional

from sqlalchemy import String, Float, DateTime, JSON, Enum, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class ViolationType(str, PyEnum):
    HELMET_NON_COMPLIANCE = "helmet_non_compliance"
    SEATBELT_NON_COMPLIANCE = "seatbelt_non_compliance"
    TRIPLE_RIDING = "triple_riding"
    WRONG_SIDE_DRIVING = "wrong_side_driving"
    STOP_LINE_VIOLATION = "stop_line_violation"
    RED_LIGHT_VIOLATION = "red_light_violation"
    ILLEGAL_PARKING = "illegal_parking"
    MULTIPLE = "multiple"


class ViolationStatus(str, PyEnum):
    PENDING = "pending"
    REVIEWED = "reviewed"
    DISMISSED = "dismissed"
    CHALLAN_ISSUED = "challan_issued"


class VehicleType(str, PyEnum):
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    BICYCLE = "bicycle"
    UNKNOWN = "unknown"


class Violation(Base):
    __tablename__ = "violations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Violation details
    violation_type: Mapped[str] = mapped_column(Enum(ViolationType), index=True)
    violation_types: Mapped[Optional[str]] = mapped_column(JSON, nullable=True)  # Multiple violations list
    confidence: Mapped[float] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(20), default="medium")  # low/medium/high

    # Vehicle info
    vehicle_type: Mapped[str] = mapped_column(Enum(VehicleType), default=VehicleType.UNKNOWN)
    license_plate: Mapped[Optional[str]] = mapped_column(String(20), nullable=True, index=True)
    plate_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Evidence
    original_image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    annotated_image_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Location / camera
    camera_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)

    # Status
    status: Mapped[str] = mapped_column(Enum(ViolationStatus), default=ViolationStatus.PENDING, index=True)

    # Raw detection metadata (bboxes, class scores, etc.)
    detection_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Processing info
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    person_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
