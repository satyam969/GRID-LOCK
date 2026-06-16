"""
Cameras API — Register, update, and monitor traffic cameras.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.violation import Camera
from app.schemas import CameraCreate, CameraUpdate, CameraResponse

router = APIRouter(prefix="/cameras", tags=["Cameras"])


@router.get("", response_model=List[CameraResponse])
async def list_cameras(db: AsyncSession = Depends(get_db)):
    """List all registered cameras."""
    stmt = select(Camera).order_by(Camera.id.desc())
    result = await db.execute(stmt)
    return result.scalars().all()


@router.post("", response_model=CameraResponse)
async def create_camera(cam: CameraCreate, db: AsyncSession = Depends(get_db)):
    """Register a new traffic camera."""
    db_cam = Camera(**cam.model_dump())
    db.add(db_cam)
    await db.commit()
    await db.refresh(db_cam)
    return db_cam


@router.put("/{camera_id}", response_model=CameraResponse)
async def update_camera(
    camera_id: int, 
    cam_in: CameraUpdate, 
    db: AsyncSession = Depends(get_db)
):
    """Update camera details."""
    db_cam = await db.get(Camera, camera_id)
    if not db_cam:
        raise HTTPException(404, "Camera not found")
        
    update_data = cam_in.model_dump(exclude_unset=True)
    for k, v in update_data.items():
        setattr(db_cam, k, v)
        
    await db.commit()
    await db.refresh(db_cam)
    return db_cam


@router.delete("/{camera_id}")
async def delete_camera(camera_id: int, db: AsyncSession = Depends(get_db)):
    """Decommission a camera."""
    db_cam = await db.get(Camera, camera_id)
    if not db_cam:
        raise HTTPException(404, "Camera not found")
        
    await db.delete(db_cam)
    await db.commit()
    return {"status": "deleted"}
