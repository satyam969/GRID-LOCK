"""
Violations CRUD API — list, filter, update, delete violation records.
"""
import math
from typing import Optional
from datetime import datetime, date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from app.database import get_db
from app.models.violation import Violation, ViolationType, ViolationStatus
from app.schemas import ViolationResponse, ViolationUpdate, PaginatedViolations

router = APIRouter(prefix="/violations", tags=["Violations"])


@router.get("", response_model=PaginatedViolations)
async def list_violations(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    violation_type: Optional[str] = None,
    status: Optional[str] = None,
    license_plate: Optional[str] = None,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    db: AsyncSession = Depends(get_db),
):
    """List violations with filtering and pagination."""
    stmt = select(Violation).order_by(Violation.timestamp.desc())

    if violation_type:
        stmt = stmt.where(Violation.violation_type == violation_type)
    if status:
        stmt = stmt.where(Violation.status == status)
    if license_plate:
        stmt = stmt.where(Violation.license_plate.ilike(f"%{license_plate}%"))
    if date_from:
        stmt = stmt.where(Violation.timestamp >= datetime.combine(date_from, datetime.min.time()))
    if date_to:
        stmt = stmt.where(Violation.timestamp <= datetime.combine(date_to, datetime.max.time()))

    # Count total
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    # Paginate
    offset = (page - 1) * page_size
    stmt = stmt.offset(offset).limit(page_size)
    rows = (await db.execute(stmt)).scalars().all()

    return PaginatedViolations(
        items=[ViolationResponse.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
        pages=math.ceil(total / page_size) if total else 0,
    )


@router.get("/{violation_id}", response_model=ViolationResponse)
async def get_violation(violation_id: str, db: AsyncSession = Depends(get_db)):
    """Get single violation by ID."""
    result = await db.get(Violation, violation_id)
    if not result:
        raise HTTPException(404, f"Violation {violation_id} not found")
    return ViolationResponse.model_validate(result)


@router.patch("/{violation_id}", response_model=ViolationResponse)
async def update_violation(
    violation_id: str,
    update: ViolationUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update violation status or notes (for officer review)."""
    record = await db.get(Violation, violation_id)
    if not record:
        raise HTTPException(404, f"Violation {violation_id} not found")

    if update.status is not None:
        record.status = update.status
    if update.notes is not None:
        record.notes = update.notes

    return ViolationResponse.model_validate(record)


@router.delete("/{violation_id}", status_code=204)
async def delete_violation(violation_id: str, db: AsyncSession = Depends(get_db)):
    """Soft delete — sets status to DISMISSED."""
    record = await db.get(Violation, violation_id)
    if not record:
        raise HTTPException(404, f"Violation {violation_id} not found")
    record.status = ViolationStatus.DISMISSED
