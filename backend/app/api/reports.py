"""
Reports API — Generate aggregated traffic data exports.
"""
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
import io
import csv

from app.database import get_db
from app.models.violation import Violation, Challan

router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/generate")
async def generate_report(
    type: str = Query("daily_summary", description="Type of report: daily_summary, challan_revenue"),
    db: AsyncSession = Depends(get_db)
):
    """Generate a CSV report based on the requested type."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    if type == "daily_summary":
        stmt = select(Violation).order_by(Violation.timestamp.desc())
        violations = (await db.execute(stmt)).scalars().all()
        
        writer.writerow(["ID", "Timestamp", "Violation Type", "Confidence", "Status", "Camera ID"])
        for v in violations:
            writer.writerow([
                v.id, 
                v.timestamp.isoformat(), 
                v.violation_type, 
                f"{v.confidence:.2f}", 
                v.status, 
                v.camera_id
            ])
            
    elif type == "challan_revenue":
        stmt = select(Challan).order_by(Challan.created_at.desc())
        challans = (await db.execute(stmt)).scalars().all()
        
        writer.writerow(["Challan Number", "Date Issued", "Amount", "Status", "Payment Date"])
        for c in challans:
            writer.writerow([
                c.challan_number,
                c.created_at.isoformat(),
                c.fine_amount,
                c.payment_status,
                c.payment_date.isoformat() if c.payment_date else "N/A"
            ])
            
    output.seek(0)
    filename = f"report_{type}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
