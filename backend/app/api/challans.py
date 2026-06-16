"""
Challans API — Generate, list, and pay traffic fines.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_
from sqlalchemy.orm import joinedload
from datetime import datetime, timezone, timedelta
import uuid

from app.database import get_db
from app.models.violation import Challan, Violation, ViolationStatus, Vehicle
from app.schemas import ChallanResponse, ChallanGenerateRequest

router = APIRouter(prefix="/challans", tags=["Challans"])

# Indian Motor Vehicle Act 2019 fine schedule
FINE_SCHEDULE = {
    'helmet_non_compliance':   {'first': 1000, 'repeat': 2000, 'section': '194D'},
    'seatbelt_non_compliance': {'first': 1000, 'repeat': 1000, 'section': '194B(1)'},
    'triple_riding':           {'first': 2000, 'repeat': 5000, 'section': '194C'},
    'wrong_side_driving':      {'first': 5000, 'repeat': 10000, 'section': '184'},
    'stop_line_violation':     {'first':  500, 'repeat': 1500, 'section': '177'},
    'red_light_violation':     {'first': 2000, 'repeat': 5000, 'section': '177A'},
    'illegal_parking':         {'first':  500, 'repeat': 1500, 'section': '177'},
    'multiple':                {'first': 3000, 'repeat': 6000, 'section': 'Multiple'},
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
    stmt = select(Violation).options(joinedload(Violation.vehicle)).where(Violation.id == req.violation_id)
    violation = (await db.execute(stmt)).scalars().first()
    if not violation:
        raise HTTPException(404, "Violation not found")
        
    # Check if challan already exists
    stmt = select(Challan).where(Challan.violation_id == violation.id)
    existing = (await db.execute(stmt)).scalars().first()
    if existing:
        raise HTTPException(400, "Challan already issued for this violation")
        
    schedule = FINE_SCHEDULE.get(violation.violation_type, {'first': 1000, 'repeat': 2000})
    is_repeat = violation.vehicle and violation.vehicle.is_repeat_offender
    fine = schedule['repeat'] if is_repeat else schedule['first']
        
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


@router.get("/revenue-summary")
async def get_revenue_summary(db: AsyncSession = Depends(get_db)):
    """Revenue dashboard data — total fines, collected, pending, overdue."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    total_issued = (await db.execute(
        select(func.count(Challan.id))
    )).scalar() or 0

    total_amount = (await db.execute(
        select(func.coalesce(func.sum(Challan.fine_amount), 0))
    )).scalar() or 0

    collected = (await db.execute(
        select(func.coalesce(func.sum(Challan.fine_amount), 0)).where(
            Challan.payment_status == 'PAID'
        )
    )).scalar() or 0

    pending = (await db.execute(
        select(func.coalesce(func.sum(Challan.fine_amount), 0)).where(
            Challan.payment_status == 'UNPAID'
        )
    )).scalar() or 0

    overdue_count = (await db.execute(
        select(func.count(Challan.id)).where(
            and_(Challan.payment_status == 'UNPAID', Challan.due_date < now)
        )
    )).scalar() or 0

    return {
        'total_challans_issued': total_issued,
        'total_fine_amount': total_amount,
        'amount_collected': collected,
        'amount_pending': pending,
        'overdue_count': overdue_count,
        'collection_rate': round((collected / max(total_amount, 1)) * 100, 1),
    }
