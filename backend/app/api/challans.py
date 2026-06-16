"""
Challans API — Generate, list, and pay traffic fines.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone, timedelta
import uuid

from app.database import get_db
from app.models.violation import Challan, Violation, ViolationStatus, Vehicle
from app.schemas import ChallanResponse, ChallanGenerateRequest

router = APIRouter(prefix="/challans", tags=["Challans"])

FINE_SCHEDULE = {
    'helmet_non_compliance': 1000,
    'seatbelt_non_compliance': 1000,
    'triple_riding': 2000,
    'wrong_side_driving': 5000,
    'stop_line_violation': 1000,
    'red_light_violation': 2000,
    'illegal_parking': 500,
    'multiple': 3000
}

@router.get("", response_model=List[ChallanResponse])
async def list_challans(
    status: str = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """List challans with optional status filter."""
    stmt = select(Challan, Vehicle.plate_number, Violation.violation_type).join(
        Violation, Challan.violation_id == Violation.id
    ).outerjoin(
        Vehicle, Challan.vehicle_id == Vehicle.id
    )

    if status:
        stmt = stmt.where(Challan.payment_status == status)

    result = await db.execute(stmt)
    rows = result.all()
    
    response_data = []
    for challan, plate, v_type in rows:
        c_dict = challan.__dict__.copy()
        c_dict['plate_number'] = plate
        c_dict['violation_type'] = v_type
        response_data.append(ChallanResponse.model_validate(c_dict))
        
    return response_data


@router.post("/generate", response_model=ChallanResponse)
async def generate_challan(
    req: ChallanGenerateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate a challan for a confirmed violation."""
    # Check if violation exists
    violation = await db.get(Violation, req.violation_id)
    if not violation:
        raise HTTPException(404, "Violation not found")
        
    # Check if challan already exists
    stmt = select(Challan).where(Challan.violation_id == violation.id)
    existing = (await db.execute(stmt)).scalars().first()
    if existing:
        raise HTTPException(400, "Challan already issued for this violation")
        
    fine = FINE_SCHEDULE.get(violation.violation_type, 1000)
    if violation.vehicle and violation.vehicle.is_repeat_offender:
        fine *= 2 # Repeat offenders pay double!
        
    challan = Challan(
        violation_id=violation.id,
        vehicle_id=violation.vehicle_id,
        challan_number=f"TG-{datetime.now().year}-{str(uuid.uuid4())[:8].upper()}",
        fine_amount=fine,
        due_date=datetime.now(timezone.utc) + timedelta(days=30),
        payment_status="UNPAID",
        issued_by="System"
    )
    
    violation.status = ViolationStatus.CHALLAN_ISSUED
    
    db.add(challan)
    await db.flush()
    
    # Refresh to load relationships
    plate = violation.vehicle.plate_number if violation.vehicle else None
    
    c_dict = challan.__dict__.copy()
    c_dict['plate_number'] = plate
    c_dict['violation_type'] = violation.violation_type
    
    return ChallanResponse.model_validate(c_dict)


@router.put("/{challan_id}/pay", response_model=ChallanResponse)
async def pay_challan(
    challan_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Mark a challan as PAID."""
    challan = await db.get(Challan, challan_id)
    if not challan:
        raise HTTPException(404, "Challan not found")
        
    challan.payment_status = "PAID"
    challan.payment_date = datetime.now(timezone.utc)
    
    # Also update the violation status to RESOLVED
    violation = await db.get(Violation, challan.violation_id)
    if violation:
        violation.status = ViolationStatus.RESOLVED
        
    await db.flush()
    
    # Just return without the joins for simplicity on update
    return ChallanResponse.model_validate(challan)
