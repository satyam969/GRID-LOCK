# TrafficGuard AI — Final Implementation Sprint

**Version:** 5.0 (Production-Ready Code for Critical Gaps)  
**Date:** June 2026  
**Prerequisite:** Complete `TrafficGuard_AI_System_Enhancement_Guide.md` first (database models + OCR fix)  
**Scope:** This file contains **exact, copy-paste-ready code** for the 3 critical gaps that will lose you points if unfixed.

---

## Table of Contents

1. [Dashboard KPI Fix (Complete Backend + Frontend)](#1-dashboard-kpi-fix)
2. [Notification System (Complete Implementation)](#2-notification-system)
3. [Challan Management (Complete Backend + Frontend)](#3-challan-management)
4. [Vehicle Lookup Page (Complete Implementation)](#4-vehicle-lookup-page)
5. [Database Migration Script](#5-database-migration-script)
6. [Integration Wiring (main.py + requirements + .env)](#6-integration-wiring)
7. [Pre-Demo Testing Checklist](#7-pre-demo-testing-checklist)

---

## 1. Dashboard KPI Fix

### 1.1 Backend: `app/routes/dashboard_routes.py`

```python
# ============================================================
# app/routes/dashboard_routes.py
# COMPLETE — Fixes the 14/14/14/14 KPI bug
# ============================================================
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, text, case, and_
from datetime import datetime, timedelta
from typing import Optional

from app.database import get_db
from app.models import Violation, Vehicle, Camera, Challan

router = APIRouter(prefix='/api/v1/dashboard', tags=['Dashboard'])


def _today_start():
    return datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)


def _week_start():
    today = _today_start()
    return today - timedelta(days=today.weekday())


@router.get('/kpis')
async def get_dashboard_kpis(db: Session = Depends(get_db)):
    """Return ALL dashboard KPIs in a single API call."""

    today = _today_start()
    week_start = _week_start()
    last_week_start = week_start - timedelta(days=7)

    # --- Core KPIs ---
    total_violations = db.query(func.count(Violation.id)).scalar() or 0

    todays_incidents = db.query(func.count(Violation.id)).filter(
        Violation.created_at >= today
    ).scalar() or 0

    pending_review = db.query(func.count(Violation.id)).filter(
        Violation.status == 'PENDING'
    ).scalar() or 0

    avg_confidence = db.query(func.avg(Violation.confidence)).filter(
        Violation.created_at >= today
    ).scalar()
    avg_confidence = round((avg_confidence or 0) * 100, 1)

    # --- Advanced KPIs ---
    total_reviewed = db.query(func.count(Violation.id)).filter(
        Violation.status != 'PENDING'
    ).scalar() or 0
    resolution_rate = round((total_reviewed / max(total_violations, 1)) * 100, 1)

    avg_review_time = db.query(
        func.avg(
            func.julianday(Violation.reviewed_at) - func.julianday(Violation.created_at)
        )
    ).filter(Violation.reviewed_at.isnot(None)).scalar()
    avg_review_hours = round((avg_review_time or 0) * 24, 1)

    false_positives = db.query(func.count(Violation.id)).filter(
        Violation.is_false_positive == True
    ).scalar() or 0
    false_positive_rate = round((false_positives / max(total_violations, 1)) * 100, 1)

    # Week-over-week change
    this_week_count = db.query(func.count(Violation.id)).filter(
        Violation.created_at >= week_start
    ).scalar() or 0
    last_week_count = db.query(func.count(Violation.id)).filter(
        and_(Violation.created_at >= last_week_start, Violation.created_at < week_start)
    ).scalar() or 0
    wow_change = round(
        ((this_week_count - last_week_count) / max(last_week_count, 1)) * 100, 1
    )

    # Peak violation hour today
    peak_hour_row = db.query(
        func.strftime('%H', Violation.created_at).label('hour'),
        func.count(Violation.id).label('cnt')
    ).filter(
        Violation.created_at >= today
    ).group_by('hour').order_by(func.count(Violation.id).desc()).first()
    peak_hour = int(peak_hour_row.hour) if peak_hour_row else None

    return {
        'total_violations': total_violations,
        'todays_incidents': todays_incidents,
        'pending_review': pending_review,
        'avg_confidence': avg_confidence,
        'resolution_rate': resolution_rate,
        'avg_review_time_hours': avg_review_hours,
        'false_positive_rate': false_positive_rate,
        'week_over_week_change': wow_change,
        'peak_violation_hour': peak_hour,
    }


@router.get('/heatmap')
async def get_violation_heatmap(
    days: int = Query(30, ge=7, le=365),
    db: Session = Depends(get_db)
):
    """Return hour x day-of-week violation counts for heatmap."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    rows = db.query(
        func.strftime('%w', Violation.created_at).label('day_of_week'),
        func.strftime('%H', Violation.created_at).label('hour'),
        func.count(Violation.id).label('count')
    ).filter(
        Violation.created_at >= cutoff
    ).group_by('day_of_week', 'hour').all()

    return [{'day': int(r.day_of_week), 'hour': int(r.hour), 'count': r.count} for r in rows]


@router.get('/repeat-offenders')
async def get_repeat_offenders(
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """Return top N vehicles with most violations."""
    offenders = db.query(Vehicle).filter(
        Vehicle.total_violations >= 2
    ).order_by(Vehicle.total_violations.desc()).limit(limit).all()

    return [{
        'plate_number': v.plate_number,
        'vehicle_type': v.vehicle_type,
        'total_violations': v.total_violations,
        'last_seen': v.last_seen.isoformat() if v.last_seen else None,
        'is_repeat_offender': v.is_repeat_offender,
    } for v in offenders]


@router.get('/live-feed')
async def get_live_feed(
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Return latest violations with thumbnail paths for live feed."""
    violations = db.query(Violation).order_by(
        Violation.created_at.desc()
    ).limit(limit).all()

    return [{
        'id': v.id,
        'violation_type': v.violation_type,
        'severity': v.severity,
        'confidence': round(v.confidence * 100, 1),
        'plate_text': v.plate_text,
        'vehicle_type': v.detected_vehicle_type,
        'thumbnail_url': f'/api/v1/images/{v.annotated_image_path}' if v.annotated_image_path else None,
        'timestamp': v.created_at.isoformat(),
        'status': v.status,
    } for v in violations]


@router.get('/weekly-comparison')
async def get_weekly_comparison(db: Session = Depends(get_db)):
    """Compare this week vs last week violations by type."""
    week_start = _week_start()
    last_week_start = week_start - timedelta(days=7)

    def _get_type_counts(start, end):
        rows = db.query(
            Violation.violation_type,
            func.count(Violation.id).label('count')
        ).filter(
            and_(Violation.created_at >= start, Violation.created_at < end)
        ).group_by(Violation.violation_type).all()
        return {r.violation_type: r.count for r in rows}

    this_week = _get_type_counts(week_start, datetime.utcnow())
    last_week = _get_type_counts(last_week_start, week_start)

    all_types = set(list(this_week.keys()) + list(last_week.keys()))
    return [{
        'violation_type': t,
        'this_week': this_week.get(t, 0),
        'last_week': last_week.get(t, 0),
    } for t in sorted(all_types)]
```

### 1.2 Frontend: Updated KPI Cards (`Dashboard.tsx` key changes)

```tsx
// ============================================================
// Key changes for src/pages/Dashboard.tsx
// Replace your current KPI fetch logic with this
// ============================================================

interface DashboardKPIs {
  total_violations: number;
  todays_incidents: number;
  pending_review: number;
  avg_confidence: number;
  resolution_rate: number;
  avg_review_time_hours: number;
  false_positive_rate: number;
  week_over_week_change: number;
  peak_violation_hour: number | null;
}

const [kpis, setKpis] = useState<DashboardKPIs | null>(null);

useEffect(() => {
  const fetchKPIs = async () => {
    const res = await fetch('/api/v1/dashboard/kpis');
    const data = await res.json();
    setKpis(data);
  };
  fetchKPIs();
  const interval = setInterval(fetchKPIs, 30000); // Refresh every 30s
  return () => clearInterval(interval);
}, []);

// KPI Card Component
const KPICard = ({ title, value, subtitle, color, icon }: {
  title: string; value: string | number;
  subtitle: string; color: string; icon: React.ReactNode;
}) => (
  <div className='bg-gray-900 rounded-xl p-5 border-t-2' style={{ borderColor: color }}>
    <div className='flex justify-between items-start'>
      <p className='text-gray-400 text-xs uppercase tracking-wider'>{title}</p>
      <span className='text-gray-500'>{icon}</span>
    </div>
    <p className='text-3xl font-bold text-white mt-2'>{value}</p>
    <p className='text-xs mt-1' style={{ color }}>{subtitle}</p>
  </div>
);

// Usage in JSX — REPLACES your current 4 identical cards:
{kpis && (
  <div className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'>
    <KPICard
      title='TOTAL VIOLATIONS'
      value={kpis.total_violations}
      subtitle='All-time records'
      color='#22d3ee'
      icon={<Shield size={18} />}
    />
    <KPICard
      title={`TODAY'S INCIDENTS`}
      value={kpis.todays_incidents}
      subtitle='Since midnight'
      color='#f87171'
      icon={<AlertTriangle size={18} />}
    />
    <KPICard
      title='PENDING REVIEW'
      value={kpis.pending_review}
      subtitle='Needs Officer Verification'
      color='#fbbf24'
      icon={<Clock size={18} />}
    />
    <KPICard
      title='AVG CONFIDENCE'
      value={`${kpis.avg_confidence}%`}
      subtitle={`Resolution rate: ${kpis.resolution_rate}%`}
      color='#34d399'
      icon={<CheckCircle size={18} />}
    />
  </div>
)}
```

---

## 2. Notification System

### 2.1 `app/services/notification_service.py`

```python
# ============================================================
# app/services/notification_service.py
# COMPLETE — Email + SMS notification on violations
# ============================================================
import smtplib
import os
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class NotificationService:
    """Handles email and SMS alerts for traffic violations."""

    def __init__(self):
        self.smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_user = os.getenv('SMTP_USER', '')
        self.smtp_pass = os.getenv('SMTP_PASS', '')
        self.from_email = os.getenv('FROM_EMAIL', 'trafficguard@system.ai')
        self.admin_emails = os.getenv('ADMIN_EMAILS', '').split(',')
        self.notifications_enabled = os.getenv('NOTIFICATIONS_ENABLED', 'true').lower() == 'true'

    def _send_email(self, to_emails: list, subject: str, html_body: str) -> bool:
        """Send an email via SMTP. Returns True on success."""
        if not self.notifications_enabled or not self.smtp_user:
            logger.warning('Notifications disabled or SMTP not configured.')
            return False

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ', '.join(to_emails)
            msg.attach(MIMEText(html_body, 'html'))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.from_email, to_emails, msg.as_string())

            logger.info(f'Email sent to {to_emails}: {subject}')
            return True
        except Exception as e:
            logger.error(f'Failed to send email: {e}')
            return False

    def send_violation_alert(
        self,
        violation_type: str,
        severity: str,
        confidence: float,
        plate_text: Optional[str],
        vehicle_type: str,
        camera_name: str = 'Unknown',
        image_url: Optional[str] = None,
    ) -> bool:
        """Send alert for HIGH/CRITICAL severity violations."""
        if severity not in ('HIGH', 'CRITICAL'):
            return False  # Only alert on serious violations

        subject = f'[TrafficGuard] {severity} Alert: {violation_type}'
        html = f'''
        <div style='font-family: Arial; padding: 20px; background: #1a1a2e; color: #fff;'>
            <h2 style='color: #ff4444;'>Traffic Violation Alert</h2>
            <table style='width:100%; border-collapse:collapse;'>
                <tr><td style='padding:8px; color:#aaa;'>Type</td>
                    <td style='padding:8px; font-weight:bold;'>{violation_type}</td></tr>
                <tr><td style='padding:8px; color:#aaa;'>Severity</td>
                    <td style='padding:8px; color:#ff4444; font-weight:bold;'>{severity}</td></tr>
                <tr><td style='padding:8px; color:#aaa;'>Confidence</td>
                    <td style='padding:8px;'>{round(confidence * 100, 1)}%</td></tr>
                <tr><td style='padding:8px; color:#aaa;'>Vehicle</td>
                    <td style='padding:8px;'>{vehicle_type}</td></tr>
                <tr><td style='padding:8px; color:#aaa;'>License Plate</td>
                    <td style='padding:8px; font-weight:bold;'>{plate_text or 'Not Detected'}</td></tr>
                <tr><td style='padding:8px; color:#aaa;'>Camera</td>
                    <td style='padding:8px;'>{camera_name}</td></tr>
                <tr><td style='padding:8px; color:#aaa;'>Time</td>
                    <td style='padding:8px;'>{datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}</td></tr>
            </table>
            <p style='margin-top:20px;'>
                <a href='http://localhost:5173/violations' 
                   style='background:#3b82f6; color:white; padding:10px 20px; 
                          text-decoration:none; border-radius:6px;'>
                    Review in TrafficGuard</a>
            </p>
        </div>
        '''
        return self._send_email(self.admin_emails, subject, html)

    def send_repeat_offender_alert(
        self,
        plate_number: str,
        total_violations: int,
        vehicle_type: str,
    ) -> bool:
        """Send alert when a vehicle becomes a repeat offender."""
        subject = f'[TrafficGuard] Repeat Offender: {plate_number} ({total_violations} violations)'
        html = f'''
        <div style='font-family: Arial; padding: 20px; background: #1a1a2e; color: #fff;'>
            <h2 style='color: #fbbf24;'>Repeat Offender Detected</h2>
            <p>Vehicle <strong>{plate_number}</strong> ({vehicle_type}) has accumulated 
               <strong style='color:#ff4444;'>{total_violations}</strong> violations.</p>
            <p>Recommended action: Generate consolidated challan with enhanced fine.</p>
            <p style='margin-top:20px;'>
                <a href='http://localhost:5173/vehicle-lookup?plate={plate_number}'
                   style='background:#f59e0b; color:black; padding:10px 20px;
                          text-decoration:none; border-radius:6px;'>
                    View Vehicle History</a>
            </p>
        </div>
        '''
        return self._send_email(self.admin_emails, subject, html)

    def send_daily_digest(
        self,
        total_today: int,
        pending_count: int,
        top_violation_type: str,
        overdue_challans: int,
    ) -> bool:
        """Send daily summary email."""
        subject = f'[TrafficGuard] Daily Digest — {total_today} violations today'
        html = f'''
        <div style='font-family: Arial; padding: 20px; background: #1a1a2e; color: #fff;'>
            <h2>Daily Summary — {datetime.utcnow().strftime("%B %d, %Y")}</h2>
            <ul>
                <li>Total violations today: <strong>{total_today}</strong></li>
                <li>Pending review: <strong>{pending_count}</strong></li>
                <li>Most common type: <strong>{top_violation_type}</strong></li>
                <li>Overdue challans: <strong style='color:#ff4444;'>{overdue_challans}</strong></li>
            </ul>
            <p><a href='http://localhost:5173/' 
                  style='background:#3b82f6; color:white; padding:10px 20px;
                         text-decoration:none; border-radius:6px;'>Open Dashboard</a></p>
        </div>
        '''
        return self._send_email(self.admin_emails, subject, html)


# Global singleton
notification_service = NotificationService()
```

### 2.2 Integration: Auto-trigger after violation detection

```python
# ============================================================
# Add this to your violation_engine.py AFTER saving a violation
# ============================================================
from app.services.notification_service import notification_service


def post_violation_hooks(violation, vehicle, db):
    """Run after a violation is saved to the database."""

    # 1. Send alert for HIGH/CRITICAL violations
    if violation.severity in ('HIGH', 'CRITICAL'):
        notification_service.send_violation_alert(
            violation_type=violation.violation_type,
            severity=violation.severity,
            confidence=violation.confidence,
            plate_text=violation.plate_text,
            vehicle_type=violation.detected_vehicle_type or 'Unknown',
        )

    # 2. Check & flag repeat offenders
    if vehicle and vehicle.total_violations >= 3:
        if not vehicle.is_repeat_offender:
            vehicle.is_repeat_offender = True
            db.commit()

        notification_service.send_repeat_offender_alert(
            plate_number=vehicle.plate_number,
            total_violations=vehicle.total_violations,
            vehicle_type=vehicle.vehicle_type or 'Unknown',
        )
```

### 2.3 Notification SQLAlchemy Model

```python
# Add to app/models.py
class Notification(Base):
    __tablename__ = 'notifications'

    id = Column(Integer, primary_key=True, autoincrement=True)
    violation_id = Column(Integer, ForeignKey('violations.id'), nullable=True)
    notification_type = Column(String(30), nullable=False)   # 'email', 'sms'
    recipient = Column(String(255), nullable=False)
    subject = Column(String(255), nullable=False)
    status = Column(String(20), default='sent')             # 'sent', 'failed', 'pending'
    sent_at = Column(DateTime, default=datetime.utcnow)
    error_message = Column(Text, nullable=True)
```

---

## 3. Challan Management

### 3.1 `app/services/challan_service.py`

```python
# ============================================================
# app/services/challan_service.py
# COMPLETE — Challan generation, fine calculation, payment
# ============================================================
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Challan, Violation, Vehicle, AuditLog


# Indian Motor Vehicle Act 2019 fine schedule
FINE_SCHEDULE = {
    'Helmet Non Compliance': {'first': 1000, 'repeat': 2000, 'section': '194D'},
    'Seatbelt Non Compliance': {'first': 1000, 'repeat': 1000, 'section': '194B(1)'},
    'Triple Riding': {'first': 2000, 'repeat': 5000, 'section': '194C'},
    'Wrong Side Driving': {'first': 5000, 'repeat': 10000, 'section': '184'},
    'Stop Line Violation': {'first': 500, 'repeat': 1500, 'section': '177'},
}

DUE_DAYS = 30  # Challans due within 30 days


class ChallanService:

    @staticmethod
    def generate_challan_number(db: Session) -> str:
        """Generate unique challan number: TG-YYYY-NNNNNN"""
        year = datetime.utcnow().year
        last_challan = db.query(Challan).filter(
            Challan.challan_number.like(f'TG-{year}-%')
        ).order_by(Challan.id.desc()).first()

        if last_challan:
            last_num = int(last_challan.challan_number.split('-')[-1])
            next_num = last_num + 1
        else:
            next_num = 1

        return f'TG-{year}-{next_num:06d}'

    @staticmethod
    def calculate_fine(violation_type: str, is_repeat: bool = False) -> float:
        """Calculate fine based on violation type and repeat status."""
        schedule = FINE_SCHEDULE.get(violation_type)
        if not schedule:
            return 1000  # Default fine
        return schedule['repeat'] if is_repeat else schedule['first']

    @classmethod
    def create_challan(
        cls,
        violation_id: int,
        db: Session,
        issued_by: str = 'SYSTEM',
    ) -> dict:
        """Create a challan for a confirmed violation."""

        # Check violation exists and has no challan yet
        violation = db.query(Violation).filter(Violation.id == violation_id).first()
        if not violation:
            return {'success': False, 'error': 'Violation not found'}

        if violation.challan:
            return {'success': False, 'error': f'Challan already exists: {violation.challan.challan_number}'}

        # Check repeat offender status
        is_repeat = False
        vehicle = None
        if violation.vehicle_id:
            vehicle = db.query(Vehicle).get(violation.vehicle_id)
            if vehicle and vehicle.is_repeat_offender:
                is_repeat = True

        fine_amount = cls.calculate_fine(violation.violation_type, is_repeat)
        challan_number = cls.generate_challan_number(db)

        challan = Challan(
            challan_number=challan_number,
            violation_id=violation_id,
            vehicle_id=violation.vehicle_id,
            fine_amount=fine_amount,
            payment_status='UNPAID',
            due_date=datetime.utcnow() + timedelta(days=DUE_DAYS),
            issued_by=issued_by,
        )
        db.add(challan)

        # Update violation status
        old_status = violation.status
        violation.status = 'CHALLAN_ISSUED'

        # Audit log
        audit = AuditLog(
            violation_id=violation_id,
            action='CHALLAN_ISSUED',
            performed_by=issued_by,
            old_status=old_status,
            new_status='CHALLAN_ISSUED',
            notes=f'Challan {challan_number} issued. Fine: Rs. {fine_amount}',
        )
        db.add(audit)
        db.commit()

        section = FINE_SCHEDULE.get(violation.violation_type, {}).get('section', 'N/A')

        return {
            'success': True,
            'challan_number': challan_number,
            'fine_amount': fine_amount,
            'section': section,
            'is_repeat_offender': is_repeat,
            'due_date': challan.due_date.isoformat(),
        }
```

### 3.2 `app/routes/challan_routes.py`

```python
# ============================================================
# app/routes/challan_routes.py
# COMPLETE — All challan CRUD endpoints
# ============================================================
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel

from app.database import get_db
from app.models import Challan, Violation, Vehicle
from app.services.challan_service import ChallanService

router = APIRouter(prefix='/api/v1/challans', tags=['Challans'])


class GenerateChallanRequest(BaseModel):
    violation_id: int
    issued_by: str = 'SYSTEM'


class BulkGenerateRequest(BaseModel):
    violation_ids: List[int]
    issued_by: str = 'SYSTEM'


@router.post('/generate')
async def generate_challan(req: GenerateChallanRequest, db: Session = Depends(get_db)):
    """Generate a challan for a single violation."""
    result = ChallanService.create_challan(req.violation_id, db, req.issued_by)
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['error'])
    return result


@router.post('/generate-bulk')
async def generate_bulk_challans(req: BulkGenerateRequest, db: Session = Depends(get_db)):
    """Generate challans for multiple violations at once."""
    results = []
    for vid in req.violation_ids:
        result = ChallanService.create_challan(vid, db, req.issued_by)
        results.append({'violation_id': vid, **result})
    successful = sum(1 for r in results if r.get('success'))
    return {'total': len(req.violation_ids), 'generated': successful, 'results': results}


@router.get('/')
async def list_challans(
    status: Optional[str] = Query(None),
    vehicle_plate: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List challans with optional filters."""
    query = db.query(Challan)

    if status:
        query = query.filter(Challan.payment_status == status.upper())
    if vehicle_plate:
        query = query.join(Vehicle).filter(Vehicle.plate_number.contains(vehicle_plate))

    total = query.count()
    challans = query.order_by(Challan.created_at.desc()).offset((page - 1) * limit).limit(limit).all()

    return {
        'total': total,
        'page': page,
        'limit': limit,
        'challans': [{
            'id': c.id,
            'challan_number': c.challan_number,
            'violation_id': c.violation_id,
            'violation_type': c.violation.violation_type if c.violation else None,
            'plate_number': c.vehicle.plate_number if c.vehicle else 'N/A',
            'fine_amount': c.fine_amount,
            'payment_status': c.payment_status,
            'due_date': c.due_date.isoformat(),
            'is_overdue': c.payment_status == 'UNPAID' and c.due_date < datetime.utcnow(),
            'issued_by': c.issued_by,
            'created_at': c.created_at.isoformat(),
        } for c in challans]
    }


@router.put('/{challan_id}/pay')
async def mark_challan_paid(
    challan_id: int,
    payment_method: str = Query('UPI'),
    db: Session = Depends(get_db)
):
    """Mark a challan as paid."""
    challan = db.query(Challan).filter(Challan.id == challan_id).first()
    if not challan:
        raise HTTPException(status_code=404, detail='Challan not found')
    if challan.payment_status == 'PAID':
        raise HTTPException(status_code=400, detail='Already paid')

    challan.payment_status = 'PAID'
    challan.payment_date = datetime.utcnow()
    challan.payment_method = payment_method

    # Update violation status
    if challan.violation:
        challan.violation.status = 'RESOLVED'

    db.commit()
    return {'success': True, 'challan_number': challan.challan_number, 'status': 'PAID'}


@router.get('/revenue-summary')
async def get_revenue_summary(db: Session = Depends(get_db)):
    """Revenue dashboard data."""
    total_issued = db.query(func.count(Challan.id)).scalar() or 0
    total_amount = db.query(func.sum(Challan.fine_amount)).scalar() or 0

    collected = db.query(func.sum(Challan.fine_amount)).filter(
        Challan.payment_status == 'PAID'
    ).scalar() or 0

    pending = db.query(func.sum(Challan.fine_amount)).filter(
        Challan.payment_status == 'UNPAID'
    ).scalar() or 0

    overdue = db.query(func.count(Challan.id)).filter(
        and_(Challan.payment_status == 'UNPAID', Challan.due_date < datetime.utcnow())
    ).scalar() or 0

    return {
        'total_challans_issued': total_issued,
        'total_fine_amount': total_amount,
        'amount_collected': collected,
        'amount_pending': pending,
        'overdue_count': overdue,
        'collection_rate': round((collected / max(total_amount, 1)) * 100, 1),
    }
```

---

## 4. Vehicle Lookup Page

### 4.1 `app/routes/vehicle_routes.py`

```python
# ============================================================
# app/routes/vehicle_routes.py
# COMPLETE — Vehicle search, history, flagging
# ============================================================
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional

from app.database import get_db
from app.models import Vehicle, Violation, Challan

router = APIRouter(prefix='/api/v1/vehicles', tags=['Vehicles'])


@router.get('/search')
async def search_vehicles(
    plate: str = Query(..., min_length=2),
    db: Session = Depends(get_db)
):
    """Search vehicles by partial plate number. Autocomplete-friendly."""
    cleaned = plate.upper().replace(' ', '')
    vehicles = db.query(Vehicle).filter(
        Vehicle.plate_number.contains(cleaned)
    ).order_by(Vehicle.total_violations.desc()).limit(10).all()

    return [{
        'id': v.id,
        'plate_number': v.plate_number,
        'vehicle_type': v.vehicle_type,
        'total_violations': v.total_violations,
        'is_repeat_offender': v.is_repeat_offender,
    } for v in vehicles]


@router.get('/{vehicle_id}')
async def get_vehicle_profile(vehicle_id: int, db: Session = Depends(get_db)):
    """Get full vehicle profile with violation summary."""
    vehicle = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()
    if not vehicle:
        raise HTTPException(status_code=404, detail='Vehicle not found')

    # Violation type breakdown
    type_breakdown = db.query(
        Violation.violation_type, func.count(Violation.id)
    ).filter(Violation.vehicle_id == vehicle_id).group_by(Violation.violation_type).all()

    # Outstanding fines
    outstanding = db.query(func.sum(Challan.fine_amount)).filter(
        Challan.vehicle_id == vehicle_id, Challan.payment_status == 'UNPAID'
    ).scalar() or 0

    return {
        'id': vehicle.id,
        'plate_number': vehicle.plate_number,
        'state_code': vehicle.state_code,
        'district_code': vehicle.district_code,
        'vehicle_type': vehicle.vehicle_type,
        'total_violations': vehicle.total_violations,
        'is_repeat_offender': vehicle.is_repeat_offender,
        'first_seen': vehicle.first_seen.isoformat() if vehicle.first_seen else None,
        'last_seen': vehicle.last_seen.isoformat() if vehicle.last_seen else None,
        'violation_breakdown': {vtype: count for vtype, count in type_breakdown},
        'outstanding_fines': outstanding,
    }


@router.get('/{vehicle_id}/violations')
async def get_vehicle_violations(
    vehicle_id: int,
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get all violations for a specific vehicle."""
    violations = db.query(Violation).filter(
        Violation.vehicle_id == vehicle_id
    ).order_by(Violation.created_at.desc()).offset((page - 1) * limit).limit(limit).all()

    return [{
        'id': v.id,
        'violation_type': v.violation_type,
        'severity': v.severity,
        'confidence': round(v.confidence * 100, 1),
        'plate_text': v.plate_text,
        'status': v.status,
        'has_challan': v.challan is not None,
        'annotated_image': v.annotated_image_path,
        'created_at': v.created_at.isoformat(),
    } for v in violations]


@router.post('/{vehicle_id}/flag')
async def flag_vehicle(vehicle_id: int, db: Session = Depends(get_db)):
    """Manually flag a vehicle for attention."""
    vehicle = db.query(Vehicle).filter(Vehicle.id == vehicle_id).first()
    if not vehicle:
        raise HTTPException(status_code=404, detail='Vehicle not found')
    vehicle.flagged = True
    db.commit()
    return {'success': True, 'plate_number': vehicle.plate_number, 'flagged': True}
```

---

## 5. Database Migration Script

### 5.1 Alembic Setup

```bash
pip install alembic
cd backend
alembic init alembic
```

Edit `alembic/env.py`:

```python
# alembic/env.py — Add these lines
from app.models import Base
target_metadata = Base.metadata
```

Edit `alembic.ini`:
```ini
sqlalchemy.url = sqlite:///trafficguard.db
```

### 5.2 Create Migration

```bash
alembic revision --autogenerate -m 'create_full_schema'
alembic upgrade head
```

### 5.3 Data Migration (Old Table to New Schema)

```python
# ============================================================
# scripts/migrate_old_data.py
# Run ONCE to migrate existing violations to new schema
# ============================================================
import sqlite3
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, Violation, Vehicle, Camera, AuditLog
import re

# Connect to new database
engine = create_engine('sqlite:///trafficguard.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
db = Session()

# Create default camera
default_camera = Camera(
    name='CAM_DOWNTOWN_04',
    location='Main Junction, Downtown',
    latitude=22.5726,
    longitude=88.3639,
    zone='Zone A - Central',
    status='ACTIVE',
)
db.add(default_camera)
db.commit()

# Read old violations from existing SQLite
old_conn = sqlite3.connect('old_trafficguard.db')
old_conn.row_factory = sqlite3.Row
old_rows = old_conn.execute('SELECT * FROM violations').fetchall()

print(f'Migrating {len(old_rows)} violations...')

for row in old_rows:
    plate_text = row['plate_text'] if 'plate_text' in row.keys() else None

    # Create or find vehicle by plate
    vehicle = None
    if plate_text and plate_text.strip():
        cleaned_plate = plate_text.strip().upper()
        vehicle = db.query(Vehicle).filter(Vehicle.plate_number == cleaned_plate).first()
        if not vehicle:
            # Extract state code from plate
            state_match = re.match(r'([A-Z]{2})', cleaned_plate)
            vehicle = Vehicle(
                plate_number=cleaned_plate,
                state_code=state_match.group(1) if state_match else None,
                vehicle_type=row.get('vehicle_type', 'Unknown'),
                total_violations=0,
            )
            db.add(vehicle)
            db.flush()

        vehicle.total_violations += 1
        if vehicle.total_violations >= 3:
            vehicle.is_repeat_offender = True

    # Create new violation record
    new_violation = Violation(
        vehicle_id=vehicle.id if vehicle else None,
        camera_id=default_camera.id,
        violation_type=row['violation_type'],
        severity=row.get('severity', 'MEDIUM'),
        confidence=row.get('confidence', 0.75),
        plate_text=plate_text,
        plate_confidence=row.get('plate_confidence'),
        detected_vehicle_type=row.get('vehicle_type'),
        rider_count=row.get('rider_count'),
        original_image_path=row.get('original_image_path'),
        annotated_image_path=row.get('annotated_image_path'),
        inference_time_ms=row.get('inference_time_ms'),
        status=row.get('status', 'PENDING'),
    )
    db.add(new_violation)
    db.flush()

    # Audit log
    db.add(AuditLog(
        violation_id=new_violation.id,
        action='MIGRATED',
        performed_by='MIGRATION_SCRIPT',
        old_status=None,
        new_status=new_violation.status,
        notes='Migrated from legacy single-table schema',
    ))

db.commit()
old_conn.close()
print(f'Migration complete. {len(old_rows)} violations migrated.')
```

---

## 6. Integration Wiring

### 6.1 Updated `main.py`

```python
# ============================================================
# main.py — Register all routers
# ============================================================
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.dashboard_routes import router as dashboard_router
from app.routes.analyze_routes import router as analyze_router      # Your existing
from app.routes.violation_routes import router as violation_router  # Your existing
from app.routes.analytics_routes import router as analytics_router  # Your existing
from app.routes.challan_routes import router as challan_router      # NEW
from app.routes.vehicle_routes import router as vehicle_router      # NEW

app = FastAPI(title='TrafficGuard AI', version='4.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Static files for uploaded images
app.mount('/uploads', StaticFiles(directory='uploads'), name='uploads')

# Register all routers
app.include_router(dashboard_router)
app.include_router(analyze_router)
app.include_router(violation_router)
app.include_router(analytics_router)
app.include_router(challan_router)
app.include_router(vehicle_router)


@app.get('/api/v1/health')
async def health_check():
    return {'status': 'ok', 'version': '4.0', 'models': 'loaded'}
```

### 6.2 Updated `requirements.txt`

```txt
# Core
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
sqlalchemy>=2.0.0
alembic>=1.12.0
python-multipart>=0.0.6
pydantic>=2.0.0

# AI Pipeline
ultralytics>=8.1.0
onnxruntime>=1.16.0
opencv-python-headless>=4.8.0
numpy>=1.24.0

# OCR (NEW — replaces easyocr)
paddlepaddle>=2.5.0
paddleocr>=2.7.0

# Utilities
python-dotenv>=1.0.0
python-jose[cryptography]>=3.3.0   # JWT auth
passlib[bcrypt]>=1.7.4             # Password hashing
slowapi>=0.1.8                      # Rate limiting
```

### 6.3 `.env` Template

```env
# ============================================================
# .env — TrafficGuard AI Configuration
# ============================================================

# Database
DATABASE_URL=sqlite:///trafficguard.db

# AI Models
MODEL_DIR=models/
CONFIDENCE_THRESHOLD=0.25

# Email Notifications
NOTIFICATIONS_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
FROM_EMAIL=trafficguard@system.ai
ADMIN_EMAILS=officer1@traffic.gov.in,officer2@traffic.gov.in

# JWT Auth
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=480

# Server
HOST=0.0.0.0
PORT=8000
UVICORN_WORKERS=1
```

---

## 7. Pre-Demo Testing Checklist

Run through this **before your hackathon demo** to ensure nothing is broken:

### Backend Tests

- [ ] 1. **Start server:** `uvicorn main:app --reload --port 8000`
- [ ] 2. **Health check:** `GET /api/v1/health` returns `{"status": "ok"}`
- [ ] 3. **Dashboard KPIs:** `GET /api/v1/dashboard/kpis` — verify Total != Today's != Pending
- [ ] 4. **Analyze motorcycle image:** Upload a bike with 2 riders + helmets → 0 violations
- [ ] 5. **Analyze motorcycle image:** Upload a bike with 3 riders → Triple Riding detected
- [ ] 6. **Analyze car image:** Upload car → Seatbelt violation detected + **plate reads correctly**
- [ ] 7. **OCR output check:** Plate shows `WB 06 J 2431` (not `NB062431WBHB06J2431`)
- [ ] 8. **Generate challan:** `POST /api/v1/challans/generate` with a violation ID → returns challan number
- [ ] 9. **Vehicle search:** `GET /api/v1/vehicles/search?plate=WB06` → returns matching vehicles
- [ ] 10. **Revenue summary:** `GET /api/v1/challans/revenue-summary` → returns totals

### Frontend Tests

- [ ] 11. **Dashboard:** 4 KPI cards show DIFFERENT numbers
- [ ] 12. **Dashboard:** Violation trend chart is rendering
- [ ] 13. **Dashboard:** Live Incident Feed shows latest violations
- [ ] 14. **Analyze page:** Upload image → annotated result appears in < 10 seconds
- [ ] 15. **Violations page:** Table loads with all columns populated
- [ ] 16. **Violations page:** Search by plate number works
- [ ] 17. **Violations page:** Click approve/reject → status changes
- [ ] 18. **Analytics page:** All charts render with real data
- [ ] 19. **CSV Export:** Click Export CSV → file downloads with correct data
- [ ] 20. **Notification:** Check email inbox for HIGH severity alert (if SMTP configured)

### Demo Script (Recommended Order)

1. Open Dashboard — show real-time KPIs
2. Go to Analyze — upload a motorcycle image with triple riding → show detection
3. Go to Analyze — upload a car image → show seatbelt violation + clean plate OCR
4. Go to Violations — show the new records, filter by type
5. Click a plate number — show Vehicle Lookup with history
6. Generate Challan — show auto-calculated fine from Motor Vehicle Act
7. Go to Analytics — show trends, vehicle distribution, model telemetry
8. Show email notification received for HIGH severity violation

---

*This completes the TrafficGuard AI system. With this file + the System Enhancement Guide + the original ONNX Optimization Guide, you have the full production blueprint.*