"""
Analytics API — Statistics, trends, and report generation.
"""
import io
import csv
from datetime import datetime, timedelta, date, timezone
from typing import Optional
from collections import Counter, defaultdict

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database import get_db
from app.models.violation import Violation, ViolationType
from app.schemas import AnalyticsSummary, ViolationCount, DailyTrend

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
async def get_summary(db: AsyncSession = Depends(get_db)):
    today_start = datetime.now(timezone.utc).replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)

    all_violations = (await db.execute(select(Violation))).scalars().all()
    today_violations = [v for v in all_violations if v.timestamp >= today_start]
    pending = [v for v in all_violations if v.status == "pending"]

    # Violation type distribution
    type_counter = Counter(v.violation_type for v in all_violations)
    total = len(all_violations)
    top_violations = [
        ViolationCount(
            violation_type=vtype,
            count=count,
            percentage=round(count / total * 100, 1) if total else 0,
        )
        for vtype, count in type_counter.most_common(7)
    ]

    # Vehicle type distribution
    vehicle_counter = Counter(v.vehicle_type for v in all_violations)

    # Average confidence
    avg_conf = (
        sum(v.confidence for v in all_violations) / total if total else 0.0
    )

    # Resolution rate
    resolution_rate = ((total - len(pending)) / total * 100) if total > 0 else 0.0

    return AnalyticsSummary(
        total_violations=total,
        today_violations=len(today_violations),
        pending_review=len(pending),
        resolution_rate=round(resolution_rate, 1),
        top_violations=top_violations,
        vehicle_distribution=dict(vehicle_counter),
        avg_confidence=round(avg_conf, 3),
    )


@router.get("/trends")
async def get_trends(
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """Daily violation counts for trend charts."""
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    violations = (
        await db.execute(
            select(Violation).where(Violation.timestamp >= since)
        )
    ).scalars().all()

    # Group by date
    daily: dict = defaultdict(lambda: defaultdict(int))
    for v in violations:
        day_str = v.timestamp.strftime("%Y-%m-%d")
        daily[day_str]["total"] += 1
        daily[day_str][v.violation_type] += 1

    # Fill missing days with 0
    result = []
    for i in range(days):
        day = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        entry = daily.get(day, {})
        result.append({
            "date": day,
            "count": entry.get("total", 0),
            "violation_types": {
                k: v for k, v in entry.items() if k != "total"
            },
        })

    return result


@router.get("/by-type")
async def get_by_type(db: AsyncSession = Depends(get_db)):
    """Violation counts grouped by type — for bar/pie charts."""
    violations = (await db.execute(select(Violation))).scalars().all()
    counter = Counter(v.violation_type for v in violations)
    total = sum(counter.values())

    return {
        "labels": list(counter.keys()),
        "counts": list(counter.values()),
        "percentages": [
            round(c / total * 100, 1) if total else 0
            for c in counter.values()
        ],
    }


@router.get("/metrics")
async def get_performance_metrics(db: AsyncSession = Depends(get_db)):
    """
    System performance metrics — confidence scores, inference times.
    Used for the 'Performance Evaluation' section required by the hackathon.
    """
    violations = (await db.execute(select(Violation))).scalars().all()

    if not violations:
        return {"message": "No data available yet"}

    confidences = [v.confidence for v in violations]
    inf_times = [v.inference_time_ms for v in violations if v.inference_time_ms]

    # Per-type average confidence (proxy for precision)
    type_conf = defaultdict(list)
    for v in violations:
        type_conf[v.violation_type].append(v.confidence)

    per_type_metrics = {
        vtype: {
            "avg_confidence": round(sum(confs) / len(confs), 3),
            "count": len(confs),
            "min_conf": round(min(confs), 3),
            "max_conf": round(max(confs), 3),
        }
        for vtype, confs in type_conf.items()
    }

    return {
        "total_analyzed": len(violations),
        "overall_avg_confidence": round(sum(confidences) / len(confidences), 3),
        "avg_inference_ms": round(sum(inf_times) / len(inf_times), 1) if inf_times else None,
        "max_inference_ms": round(max(inf_times), 1) if inf_times else None,
        "per_violation_type": per_type_metrics,
        "note": "Confidence scores serve as proxy for precision. Ground-truth evaluation requires labeled test set.",
    }


@router.get("/export/csv")
async def export_csv(
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    db: AsyncSession = Depends(get_db),
):
    """Export violation records as CSV download."""
    stmt = select(Violation).order_by(Violation.timestamp.desc())
    if date_from:
        stmt = stmt.where(Violation.timestamp >= datetime.combine(date_from, datetime.min.time()))
    if date_to:
        stmt = stmt.where(Violation.timestamp <= datetime.combine(date_to, datetime.max.time()))

    violations = (await db.execute(stmt)).scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Timestamp", "Violation Type", "Confidence", "Severity",
        "Vehicle Type", "License Plate", "Status", "Camera ID", "Location",
    ])
    for v in violations:
        writer.writerow([
            v.id, v.timestamp.isoformat(), v.violation_type,
            f"{v.confidence:.2%}", v.severity, v.vehicle_type,
            v.license_plate or "", v.status,
            v.camera_id or "", v.location or "",
        ])

    output.seek(0)
    filename = f"violations_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
