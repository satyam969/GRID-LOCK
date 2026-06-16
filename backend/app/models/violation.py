"""
ORM Models — Normalized database schema for TrafficGuard AI.
Includes Cameras, Vehicles, Officers, Violations, Challans, and Audit Logs.
"""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    ForeignKey, Text, Enum, Index, JSON
)
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import enum

from app.database import Base


class ViolationType(str, enum.Enum):
    HELMET_NON_COMPLIANCE = 'helmet_non_compliance'
    SEATBELT_NON_COMPLIANCE = 'seatbelt_non_compliance'
    TRIPLE_RIDING = 'triple_riding'
    WRONG_SIDE_DRIVING = 'wrong_side_driving'
    STOP_LINE_VIOLATION = 'stop_line_violation'
    RED_LIGHT_VIOLATION = 'red_light_violation'
    ILLEGAL_PARKING = 'illegal_parking'
    MULTIPLE = 'multiple'


class Severity(str, enum.Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'


class ViolationStatus(str, enum.Enum):
    PENDING = 'pending'
    UNDER_REVIEW = 'under_review'
    CONFIRMED = 'confirmed'
    FALSE_POSITIVE = 'false_positive'
    CHALLAN_ISSUED = 'challan_issued'
    RESOLVED = 'resolved'
    REVIEWED = 'reviewed'
    DISMISSED = 'dismissed'


class PaymentStatus(str, enum.Enum):
    UNPAID = 'UNPAID'
    PAID = 'PAID'
    OVERDUE = 'OVERDUE'
    WAIVED = 'WAIVED'


class VehicleType(str, enum.Enum):
    MOTORCYCLE = "motorcycle"
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    BICYCLE = "bicycle"
    UNKNOWN = "unknown"


class Camera(Base):
    __tablename__ = 'cameras'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    location = Column(String(255), nullable=False)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    direction = Column(String(50), nullable=True)
    zone = Column(String(100), nullable=True)
    status = Column(String(20), default='ACTIVE')
    installed_date = Column(DateTime, nullable=True)

    violations = relationship('Violation', back_populates='camera')


class Vehicle(Base):
    __tablename__ = 'vehicles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plate_number = Column(String(20), unique=True, nullable=False, index=True)
    state_code = Column(String(5), nullable=True)
    district_code = Column(String(5), nullable=True)
    vehicle_type = Column(String(30), nullable=True)
    owner_name = Column(String(100), nullable=True)
    total_violations = Column(Integer, default=0)
    first_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    is_repeat_offender = Column(Boolean, default=False)
    flagged = Column(Boolean, default=False)

    violations = relationship('Violation', back_populates='vehicle')
    challans = relationship('Challan', back_populates='vehicle')


class Officer(Base):
    __tablename__ = 'officers'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    badge_number = Column(String(30), unique=True, nullable=False)
    email = Column(String(100), nullable=True)
    phone = Column(String(15), nullable=True)
    role = Column(String(30), default='OFFICER')
    assigned_zone = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)


class Violation(Base):
    __tablename__ = 'violations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, ForeignKey('vehicles.id'), nullable=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=True)

    violation_type = Column(String(50), nullable=False)
    violation_types = Column(JSON, nullable=True) # Retaining for backward compat
    severity = Column(String(20), default='medium')
    confidence = Column(Float, nullable=False)

    # Legacy attributes mapped to Vehicle for backward compatibility in endpoints
    vehicle_type = Column(String(30), nullable=True)
    license_plate = Column(String(20), nullable=True, index=True)
    plate_confidence = Column(Float, nullable=True)

    # Image evidence
    original_image_path = Column(String(500), nullable=True)
    annotated_image_path = Column(String(500), nullable=True)

    # Context
    location = Column(String(200), nullable=True)
    rider_count = Column(Integer, nullable=True)
    person_count = Column(Integer, nullable=True)

    # AI performance
    inference_time_ms = Column(Float, nullable=True)
    models_used = Column(String(200), nullable=True)
    detection_metadata = Column(JSON, nullable=True)

    # Review workflow
    status = Column(String(20), default='pending')
    assigned_officer_id = Column(Integer, ForeignKey('officers.id'), nullable=True)
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    is_false_positive = Column(Boolean, default=False)

    # Timestamps (mapped 'timestamp' to 'created_at' for backward compat)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    vehicle = relationship('Vehicle', back_populates='violations')
    camera = relationship('Camera', back_populates='violations')
    audit_logs = relationship('AuditLog', back_populates='violation')
    challan = relationship('Challan', back_populates='violation', uselist=False)

    __table_args__ = (
        Index('idx_violation_type_date', 'violation_type', 'timestamp'),
        Index('idx_status_date', 'status', 'timestamp'),
    )


class Challan(Base):
    __tablename__ = 'challans'

    id = Column(Integer, primary_key=True, autoincrement=True)
    violation_id = Column(Integer, ForeignKey('violations.id'), unique=True)
    vehicle_id = Column(Integer, ForeignKey('vehicles.id'), nullable=True)

    challan_number = Column(String(30), unique=True, nullable=False)
    fine_amount = Column(Float, nullable=False)
    payment_status = Column(String(20), default='UNPAID')
    payment_date = Column(DateTime, nullable=True)
    due_date = Column(DateTime, nullable=False)
    issued_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    violation = relationship('Violation', back_populates='challan')
    vehicle = relationship('Vehicle', back_populates='challans')


class AuditLog(Base):
    __tablename__ = 'audit_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    violation_id = Column(Integer, ForeignKey('violations.id'), nullable=False)
    action = Column(String(50), nullable=False)
    performed_by = Column(String(100), nullable=True)
    old_status = Column(String(20), nullable=True)
    new_status = Column(String(20), nullable=True)
    notes = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    violation = relationship('Violation', back_populates='audit_logs')
