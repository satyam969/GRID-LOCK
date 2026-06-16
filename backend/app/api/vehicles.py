"""
Vehicles API — Search vehicles and view their violation history.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.violation import Vehicle, Violation
from app.schemas import VehicleResponse, ViolationResponse

router = APIRouter(prefix="/vehicles", tags=["Vehicles"])


@router.get("/search", response_model=List[VehicleResponse])
async def search_vehicles(
    plate: str = Query(..., min_length=2),
    db: AsyncSession = Depends(get_db)
):
    """Search vehicles by license plate (partial match)."""
    stmt = select(Vehicle).where(Vehicle.plate_number.ilike(f"%{plate}%")).limit(10)
    result = await db.execute(stmt)
    return result.scalars().all()


@router.get("/{vehicle_id}", response_model=VehicleResponse)
async def get_vehicle(vehicle_id: int, db: AsyncSession = Depends(get_db)):
    """Get full vehicle profile."""
    record = await db.get(Vehicle, vehicle_id)
    if not record:
        raise HTTPException(404, f"Vehicle {vehicle_id} not found")
    return record


@router.get("/{vehicle_id}/violations", response_model=List[ViolationResponse])
async def get_vehicle_violations(
    vehicle_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get all violations for a specific vehicle."""
    stmt = select(Violation).where(Violation.vehicle_id == vehicle_id).order_by(Violation.timestamp.desc())
    result = await db.execute(stmt)
    violations = result.scalars().all()
    
    return [ViolationResponse.model_validate(v) for v in violations]


@router.post("/{vehicle_id}/flag")
async def flag_vehicle(
    vehicle_id: int,
    flagged: bool = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """Flag or unflag a vehicle for strict monitoring."""
    vehicle = await db.get(Vehicle, vehicle_id)
    if not vehicle:
        raise HTTPException(404, "Vehicle not found")
        
    vehicle.flagged = flagged
    await db.flush()
    return {"status": "success", "flagged": vehicle.flagged}
