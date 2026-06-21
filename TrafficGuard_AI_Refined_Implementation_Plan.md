# TrafficGuard AI — Refined Implementation Plan (FINAL)

**Status:** Ready to execute  
**Estimated Time:** 6-7 hours  
**Deadline:** June 21, 2026  
**Supersedes:** All previous implementation plans

This is the definitive, final plan. Every code block is copy-paste ready.

---

## Table of Contents

0. [Pre-Implementation Checks](#0-pre-implementation-checks)
1. [EasyOCR Single-Line Fix](#1-easyocr-single-line-fix)
2. [Red-Light Violation (NEW + Improved)](#2-red-light-violation)
3. [Illegal Parking (NEW + Improved)](#3-illegal-parking)
4. [Stop-Line Violation (IMPROVED)](#4-stop-line-violation-improvement)
5. [Wrong-Side Driving (IMPROVED)](#5-wrong-side-driving-improvement)
6. [Seatbelt Non-Compliance (Production Note)](#6-seatbelt-production-note)
7. [Frontend Updates (All 7 Types)](#7-frontend-updates)
8. [Updated Fine Schedule](#8-updated-fine-schedule)
9. [File Changes Summary](#9-file-changes-summary)
10. [Execution Order](#10-execution-order)
11. [Post-Implementation Smoke Test](#11-smoke-test)
12. [Known Limitations + Demo Tips](#12-known-limitations)

---

## 0. Pre-Implementation Checks (15 min)

### Check 1: Does YOLOv8n detect traffic lights?

```python
from ultralytics import YOLO
model = YOLO('backend/models/yolov8n_int8.onnx')
results = model.predict('test_traffic_light_image.jpg', conf=0.12)
for det in results[0].boxes:
    if int(det.cls) == 9:
        print(f'Traffic light! conf={float(det.conf):.2f}')
# If nothing prints → lower your general detector conf for class 9 (see Section 2)
```

### Check 2: Stop-line Y passthrough
Open browser DevTools → Network → upload an image with stop-line toggle ON → check the request body includes `verify_stop_line: true`.

### Check 3: Detection output format
```python
# Add temporarily in violation_engine.py:
print(f'[DEBUG] Sample detection: {all_detections[0] if all_detections else "NONE"}')
# Verify it matches: {'class_id': int, 'bbox': {'x1':..,'y1':..,'x2':..,'y2':..}, 'confidence': float}
```

---

## 1. EasyOCR Single-Line Fix

**Time:** 30 min | **File:** `backend/app/core/ocr_engine.py` | **Action:** MODIFY

Find where EasyOCR results are processed and replace concatenation with single-best-line:

```python
# ============================================================
# REPLACE your result processing with this:
# ============================================================
results = reader.readtext(filtered_plate_img)

if not results:
    return None, 0.0

# KEY FIX: Pick SINGLE highest-confidence line
# BEFORE: concatenated all → 'NB062431WBHB06J2431'
# AFTER:  picks best     → 'WB06J2431'
best_text, best_conf = '', 0.0
for (bbox, text, conf) in results:
    if conf > best_conf:
        best_text, best_conf = text, conf

if best_conf < confidence_threshold:
    return None, 0.0

# Clean for Indian plate format
cleaned = clean_indian_plate(best_text)
return cleaned or best_text, round(best_conf, 4)


def clean_indian_plate(raw_text: str) -> str:
    """Post-process OCR output to match Indian plate format."""
    import re
    cleaned = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    # Standard: XX 00 XX 0000
    match = re.search(r'([A-Z]{2})(\d{2})([A-Z]{1,3})(\d{4})', cleaned)
    if match:
        return f'{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}'
    # BH series: 00 BH 0000 XX
    bh = re.search(r'(\d{2})(BH)(\d{4})([A-Z]{1,2})', cleaned)
    if bh:
        return f'{bh.group(1)} {bh.group(2)} {bh.group(3)} {bh.group(4)}'
    return cleaned
```

---

## 2. Red-Light Violation

**Time:** 2-3 hours | **New model:** NONE | **Action:** NEW FILE + MODIFY engine

### 2.1 Improvement: Lower Confidence for Traffic Lights

Traffic lights are small objects. YOLOv8n often detects them at 0.12-0.20 confidence, below the typical 0.25 threshold. Add this to your detector/engine where you filter detections:

```python
# In your detection filtering code (detector.py or violation_engine.py):
for det in raw_detections:
    cls_id = int(det.cls)
    conf = float(det.conf)
    
    # Standard threshold for vehicles/persons
    if conf >= 0.25:
        all_detections.append(format_detection(det))
    
    # IMPROVEMENT: Lower threshold specifically for traffic lights
    elif cls_id == 9 and conf >= 0.12:
        all_detections.append(format_detection(det))
```

### 2.2 Create `backend/app/core/red_light_detection.py`

```python
"""
Red-Light Violation Detection
Uses HSV color thresholding on traffic light crops (COCO class 9)
combined with stop-line position to detect violations.

IMPROVEMENTS:
  - Brightness check (V > 120) prevents false positives from reflections
  - 5% minimum pixel threshold prevents noise triggering
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def detect_traffic_light_color(img: np.ndarray, bbox: Dict) -> Optional[str]:
    """
    Classify a traffic light as RED, YELLOW, or GREEN.
    Divides the crop into 3 vertical zones and checks HSV ranges.
    Returns: 'RED', 'YELLOW', 'GREEN', or None.
    """
    x1 = max(0, int(bbox['x1']))
    y1 = max(0, int(bbox['y1']))
    x2 = min(img.shape[1], int(bbox['x2']))
    y2 = min(img.shape[0], int(bbox['y2']))

    crop = img[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 5:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = crop.shape[0]
    w = crop.shape[1]

    # Split into 3 vertical zones
    top_zone = hsv[0:h//3, :]          # RED position
    mid_zone = hsv[h//3:2*h//3, :]     # YELLOW position
    bot_zone = hsv[2*h//3:, :]          # GREEN position

    # --- RED detection (top zone) ---
    # Red wraps in HSV: H=0-10 OR H=160-180
    # IMPROVEMENT: Also require V > 120 (brightness) to prevent reflection false positives
    red_mask1 = cv2.inRange(top_zone, np.array([0, 100, 120]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(top_zone, np.array([160, 100, 120]), np.array([180, 255, 255]))
    red_pixels = cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2)

    # --- YELLOW detection (mid zone) ---
    yellow_mask = cv2.inRange(mid_zone, np.array([15, 100, 120]), np.array([35, 255, 255]))
    yellow_pixels = cv2.countNonZero(yellow_mask)

    # --- GREEN detection (bottom zone) ---
    green_mask = cv2.inRange(bot_zone, np.array([35, 100, 120]), np.array([85, 255, 255]))
    green_pixels = cv2.countNonZero(green_mask)

    # Minimum: at least 5% of zone area must match
    zone_area = (h // 3) * w
    min_threshold = zone_area * 0.05

    scores = {'RED': red_pixels, 'YELLOW': yellow_pixels, 'GREEN': green_pixels}
    best = max(scores, key=scores.get)

    if scores[best] < min_threshold:
        return None

    logger.info(f'Traffic light: {best} (R={red_pixels}, Y={yellow_pixels}, G={green_pixels})')
    return best


def check_red_light_violation(
    vehicle_detections: List[Dict],
    traffic_light_detections: List[Dict],
    stop_line_y: int,
    img: np.ndarray,
) -> List[Dict]:
    """Check if vehicles crossed stop line while light is RED."""
    violations = []

    if not traffic_light_detections:
        return violations

    # Check if ANY traffic light is RED
    is_red = False
    red_conf = 0.0
    for tl in traffic_light_detections:
        color = detect_traffic_light_color(img, tl['bbox'])
        if color == 'RED':
            is_red = True
            red_conf = tl.get('confidence', 0.8)
            break

    if not is_red:
        return violations

    # Check vehicles past stop line
    vehicle_classes = {2, 3, 5, 7}
    for det in vehicle_detections:
        if det.get('class_id') not in vehicle_classes:
            continue
        if det['bbox']['y2'] > stop_line_y:
            violations.append({
                'type': 'Red Light Violation',
                'severity': 'HIGH',
                'confidence': round(min(det.get('confidence', 0.8), red_conf), 4),
                'vehicle_bbox': det['bbox'],
                'vehicle_type': det.get('class_name', 'Vehicle'),
                'details': f"Vehicle past stop line (y={int(det['bbox']['y2'])}) while RED",
            })

    return violations
```

### 2.3 Integration + Annotation

```python
# In violation_engine.py — add import:
from app.core.red_light_detection import check_red_light_violation, detect_traffic_light_color

# Add this stage AFTER existing detection stages:
traffic_lights = [d for d in all_detections if d.get('class_id') == 9]
vehicle_dets = [d for d in all_detections if d.get('class_id') in {2, 3, 5, 7}]

if traffic_lights and pipeline_settings.get('verify_stop_line', False):
    stop_line_y = pipeline_settings.get('stop_line_y', int(img.shape[0] * 0.65))
    red_light_violations = check_red_light_violation(
        vehicle_dets, traffic_lights, stop_line_y, img
    )
    all_violations.extend(red_light_violations)

# Annotation:
for v in [v for v in all_violations if v['type'] == 'Red Light Violation']:
    b = v['vehicle_bbox']
    x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    lbl = 'RED LIGHT VIOLATION'
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated_img, (x1, y1-th-10), (x1+tw, y1), (0, 0, 255), -1)
    cv2.putText(annotated_img, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
```

---

## 3. Illegal Parking

**Time:** 2 hours | **New model:** NONE | **Action:** NEW FILE + MODIFY engine

### 3.1 Create `backend/app/core/illegal_parking_detection.py`

```python
"""
Illegal Parking Detection
Strategy 1: Vehicle center inside admin-configured no-parking zone
Strategy 2: Vehicle at road edge (15%) + no person within 120px

IMPROVEMENTS:
  - Minimum vehicle area threshold (1% of frame) prevents distant-car false positives
  - Motorcycles excluded (class_id 3) to prevent mass false positives
"""
import math
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def check_illegal_parking(
    all_detections: List[Dict],
    img_shape: Tuple[int, ...],
    no_parking_zones: Optional[List[Dict]] = None,
    person_proximity: float = 120.0,
    edge_pct: float = 0.15,
) -> List[Dict]:
    """Detect illegally parked vehicles."""
    violations = []
    zones = no_parking_zones or []
    img_h, img_w = img_shape[:2]

    # Only cars(2), buses(5), trucks(7). Motorcycles(3) excluded.
    vehicles = [d for d in all_detections if d.get('class_id') in {2, 5, 7}]
    persons = [d for d in all_detections if d.get('class_id') == 0]

    # IMPROVEMENT: Minimum area threshold
    min_area = img_h * img_w * 0.01  # Vehicle must be >=1% of frame

    for v in vehicles:
        vx1, vy1 = v['bbox']['x1'], v['bbox']['y1']
        vx2, vy2 = v['bbox']['x2'], v['bbox']['y2']
        v_cx = (vx1 + vx2) / 2
        v_cy = (vy1 + vy2) / 2
        v_area = (vx2 - vx1) * (vy2 - vy1)

        # Skip tiny distant vehicles
        if v_area < min_area:
            continue

        is_violation = False
        reason = ''

        # Strategy 1: No-Parking Zone
        for zone in zones:
            if (zone['x1'] <= v_cx <= zone['x2'] and zone['y1'] <= v_cy <= zone['y2']):
                is_violation = True
                reason = f"Vehicle in no-parking zone: {zone.get('label', 'Restricted')}"
                break

        # Strategy 2: Road Edge + No Person Nearby
        if not is_violation:
            edge_thresh = img_w * edge_pct
            at_left = v_cx < edge_thresh
            at_right = v_cx > (img_w - edge_thresh)

            if at_left or at_right:
                has_person = False
                for p in persons:
                    px = (p['bbox']['x1'] + p['bbox']['x2']) / 2
                    py = (p['bbox']['y1'] + p['bbox']['y2']) / 2
                    dist = math.sqrt((v_cx - px)**2 + (v_cy - py)**2)
                    if dist < person_proximity:
                        has_person = True
                        break

                if not has_person:
                    is_violation = True
                    side = 'left' if at_left else 'right'
                    reason = f'Unoccupied vehicle parked at {side} road edge'

        if is_violation:
            violations.append({
                'type': 'Illegal Parking',
                'severity': 'MEDIUM',
                'confidence': round(v.get('confidence', 0.7) * 0.85, 4),
                'vehicle_bbox': v['bbox'],
                'vehicle_type': v.get('class_name', 'Vehicle'),
                'details': reason,
            })

    logger.info(f'Parking: {len(vehicles)} vehicles checked, {len(violations)} violations')
    return violations
```

### 3.2 Integration + Annotation

```python
# In violation_engine.py:
from app.core.illegal_parking_detection import check_illegal_parking

# Stage:
if pipeline_settings.get('detect_parking', True):
    parking_violations = check_illegal_parking(
        all_detections=all_detections,
        img_shape=img.shape,
        no_parking_zones=pipeline_settings.get('no_parking_zones', []),
    )
    all_violations.extend(parking_violations)

# Annotation (blue box):
for v in [v for v in all_violations if v['type'] == 'Illegal Parking']:
    b = v['vehicle_bbox']
    x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 100, 0), 3)
    lbl = 'ILLEGAL PARKING'
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated_img, (x1, y1-th-10), (x1+tw, y1), (255, 100, 0), -1)
    cv2.putText(annotated_img, lbl, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
```

---

## 4. Stop-Line Violation Improvement

**Time:** 20 min | **Action:** MODIFY existing stop-line logic in `violation_engine.py`

**Problem:** Currently, ANY vehicle past the line is flagged. But when the light is GREEN, vehicles are SUPPOSED to be past the line.

**Fix:** Couple stop-line detection with traffic light state. Skip if GREEN.

```python
# Replace your existing stop-line violation logic with:

# ========== STAGE: Stop-Line Violation (IMPROVED) ==========
if pipeline_settings.get('verify_stop_line', False):
    stop_line_y = pipeline_settings.get('stop_line_y', int(img.shape[0] * 0.65))

    # Check traffic light state first
    light_color = None
    traffic_lights = [d for d in all_detections if d.get('class_id') == 9]
    if traffic_lights:
        from app.core.red_light_detection import detect_traffic_light_color
        for tl in traffic_lights:
            light_color = detect_traffic_light_color(img, tl['bbox'])
            if light_color:
                break

    # Only flag if light is RED, YELLOW, or not detected
    # If GREEN → vehicles SHOULD be past the line → no violation
    if light_color != 'GREEN':
        vehicle_classes = {2, 3, 5, 7}
        for det in all_detections:
            if det.get('class_id') not in vehicle_classes:
                continue
            if det['bbox']['y2'] > stop_line_y:
                all_violations.append({
                    'type': 'Stop Line Violation',
                    'severity': 'LOW',
                    'confidence': round(det.get('confidence', 0.7), 4),
                    'vehicle_bbox': det['bbox'],
                    'vehicle_type': det.get('class_name', 'Vehicle'),
                    'details': f"Light: {light_color or 'not detected'}",
                })

    # Draw stop line on annotated image
    h, w = annotated_img.shape[:2]
    cv2.line(annotated_img, (0, stop_line_y), (w, stop_line_y), (0, 255, 255), 2)
    cv2.putText(annotated_img, 'STOP LINE', (10, stop_line_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
```

---

## 5. Wrong-Side Driving Improvement

**Time:** 30-45 min | **Action:** CREATE new file + MODIFY engine

**Problem:** Current logic (vehicle in left half = wrong side) produces false positives on parked cars, two-way roads, and turning vehicles.

**Fix:** Per-camera configurable lane direction + distance filter + lower confidence.

### 5.1 Create `backend/app/core/wrong_side_detection.py`

```python
"""
Wrong-Side Driving Detection (Improved)

Uses per-camera configurable lane direction instead of naive bisect.
IMPROVEMENTS:
  - Configurable flow_direction per camera ('left' or 'right')
  - Adjustable lane boundary position (default 50%)
  - Distance filter: ignores vehicles in top 30% of frame (too far to judge)
  - Lower confidence multiplier (0.7x) to reflect heuristic uncertainty
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


def check_wrong_side_driving(
    vehicle_detections: List[Dict],
    img_shape: tuple,
    camera_config: Dict,
) -> List[Dict]:
    """
    Detect wrong-side driving using camera-specific lane config.

    camera_config:
        'flow_direction': 'right' | 'left' | 'none'
        'lane_boundary_x': 0.5  (fraction of image width)
    """
    violations = []

    flow = camera_config.get('flow_direction', 'none')
    if flow == 'none' or not flow:
        return violations  # Feature disabled for this camera

    boundary_pct = camera_config.get('lane_boundary_x', 0.5)
    img_h, img_w = img_shape[:2]
    boundary_x = img_w * boundary_pct

    vehicle_classes = {2, 3, 5, 7}
    for det in vehicle_detections:
        if det.get('class_id') not in vehicle_classes:
            continue

        v_cx = (det['bbox']['x1'] + det['bbox']['x2']) / 2
        v_cy = (det['bbox']['y1'] + det['bbox']['y2']) / 2

        # Distance filter: skip vehicles in top 30% (too far to judge)
        if v_cy < img_h * 0.3:
            continue

        is_wrong = False
        if flow == 'right' and v_cx < boundary_x:
            is_wrong = True  # Should be on right, but is on left
        elif flow == 'left' and v_cx > boundary_x:
            is_wrong = True  # Should be on left, but is on right

        if is_wrong:
            violations.append({
                'type': 'Wrong Side Driving',
                'severity': 'HIGH',
                'confidence': round(det.get('confidence', 0.7) * 0.7, 4),  # 0.7x heuristic penalty
                'vehicle_bbox': det['bbox'],
                'vehicle_type': det.get('class_name', 'Vehicle'),
                'details': f"Vehicle on wrong side (expected flow: {flow})",
            })

    logger.info(f'Wrong-side: flow={flow}, violations={len(violations)}')
    return violations
```

### 5.2 Integration

```python
# In violation_engine.py:
from app.core.wrong_side_detection import check_wrong_side_driving

# Stage:
wrong_side_config = {
    'flow_direction': pipeline_settings.get('flow_direction', 'none'),
    'lane_boundary_x': pipeline_settings.get('lane_boundary_x', 0.5),
}
if wrong_side_config['flow_direction'] != 'none':
    ws_violations = check_wrong_side_driving(all_detections, img.shape, wrong_side_config)
    all_violations.extend(ws_violations)
```

---

## 6. Seatbelt Production Note

**No code change needed.** Current 45% top crop works well for sedans.

**Demo tip:** If asked, say: *"In production, the crop percentage would be adaptive based on vehicle subclass — SUVs have higher windshields than sedans. For the prototype, 45% covers the vast majority of passenger vehicles."*

---

## 7. Frontend Updates

**Time:** 30-45 min

### 7.1 Violation Color Mapping
```tsx
const VIOLATION_COLORS: Record<string, string> = {
  'Helmet Non Compliance': '#f59e0b',     // Amber
  'Seatbelt Non Compliance': '#f97316',   // Orange
  'Triple Riding': '#ef4444',             // Red
  'Wrong Side Driving': '#dc2626',        // Dark Red
  'Stop Line Violation': '#eab308',       // Yellow
  'Red Light Violation': '#ff0000',       // Bright Red   ← NEW
  'Illegal Parking': '#3b82f6',           // Blue         ← NEW
};
```

### 7.2 Filter Dropdown Options
```tsx
const VIOLATION_TYPE_OPTIONS = [
  { value: 'all', label: 'All Violations' },
  { value: 'Helmet Non Compliance', label: 'Helmet Non Compliance' },
  { value: 'Seatbelt Non Compliance', label: 'Seatbelt Non Compliance' },
  { value: 'Triple Riding', label: 'Triple Riding' },
  { value: 'Wrong Side Driving', label: 'Wrong Side Driving' },
  { value: 'Stop Line Violation', label: 'Stop Line Violation' },
  { value: 'Red Light Violation', label: 'Red Light Violation' },      // NEW
  { value: 'Illegal Parking', label: 'Illegal Parking' },              // NEW
];
```

### 7.3 Analyze.tsx — New Config Controls
```tsx
// States
const [detectParking, setDetectParking] = useState(true);
const [flowDirection, setFlowDirection] = useState('none');

// JSX — add to Configuration panel:
<label className='flex items-center gap-2 text-sm text-gray-300'>
  <input type='checkbox' checked={detectParking}
    onChange={(e) => setDetectParking(e.target.checked)}
    className='rounded bg-gray-700' />
  Detect Illegal Parking
  <span className='text-gray-500 text-xs'>Flags vehicles at road edges</span>
</label>

<div className='flex flex-col gap-1'>
  <label className='text-sm text-gray-400'>Traffic Flow Direction</label>
  <select value={flowDirection} onChange={(e) => setFlowDirection(e.target.value)}
    className='bg-gray-700 text-gray-200 rounded px-2 py-1 text-sm'>
    <option value='none'>Wrong-Side: Off</option>
    <option value='right'>Expected: Keep Right</option>
    <option value='left'>Expected: Keep Left</option>
  </select>
</div>

// Include in FormData:
formData.append('detect_parking', String(detectParking));
formData.append('flow_direction', flowDirection);
```

---

## 8. Updated Fine Schedule

```python
FINE_SCHEDULE = {
    'Helmet Non Compliance':    {'first': 1000,  'repeat': 2000,  'section': '194D',    'severity': 'MEDIUM'},
    'Seatbelt Non Compliance':  {'first': 1000,  'repeat': 1000,  'section': '194B(1)', 'severity': 'MEDIUM'},
    'Triple Riding':            {'first': 2000,  'repeat': 5000,  'section': '194C',    'severity': 'HIGH'},
    'Wrong Side Driving':       {'first': 5000,  'repeat': 10000, 'section': '184',     'severity': 'HIGH'},
    'Stop Line Violation':      {'first': 500,   'repeat': 1500,  'section': '177',     'severity': 'LOW'},
    'Red Light Violation':      {'first': 5000,  'repeat': 10000, 'section': '184',     'severity': 'HIGH'},
    'Illegal Parking':          {'first': 500,   'repeat': 1500,  'section': '177',     'severity': 'MEDIUM'},
}

VIOLATION_TYPES = [
    'Helmet Non Compliance', 'Seatbelt Non Compliance', 'Triple Riding',
    'Wrong Side Driving', 'Stop Line Violation', 'Red Light Violation', 'Illegal Parking',
]
```

---

## 9. File Changes Summary

| Action | File | Change |
|--------|------|--------|
| MODIFY | `backend/app/core/ocr_engine.py` | Single highest-confidence EasyOCR line |
| **NEW** | `backend/app/core/red_light_detection.py` | HSV color + stop-line + brightness check |
| **NEW** | `backend/app/core/illegal_parking_detection.py` | Zone + edge proximity + min area |
| **NEW** | `backend/app/core/wrong_side_detection.py` | Per-camera lane config + distance filter |
| MODIFY | `backend/app/core/violation_engine.py` | Import + integrate all 3 new modules + improved stop-line |
| MODIFY | `backend/app/core/detector.py` (or equivalent) | Lower conf threshold for class_id 9 |
| MODIFY | `frontend/src/pages/Analyze.tsx` | Parking toggle + flow direction dropdown |
| MODIFY | `frontend/src/pages/Violations.tsx` | 2 new types in dropdown + colors |
| MODIFY | Fine schedule file | 2 new entries (Red Light, Illegal Parking) |

**Total: 3 new files + 5-6 modified files**

---

## 10. Execution Order

| Step | Task | Time | Depends On |
|------|------|------|-----------|
| 0 | Pre-checks (Section 0) | 15 min | — |
| 1 | EasyOCR single-line fix | 30 min | — |
| 2 | Lower conf for class 9 traffic lights | 10 min | Check 0 |
| 3 | Create `red_light_detection.py` | 45 min | — |
| 4 | Create `illegal_parking_detection.py` | 45 min | — |
| 5 | Create `wrong_side_detection.py` | 30 min | — |
| 6 | Integrate ALL into `violation_engine.py` | 45 min | Steps 3-5 |
| 7 | Improve stop-line (couple with light state) | 20 min | Step 3 |
| 8 | Frontend: toggle + dropdown + colors | 30 min | Step 6 |
| 9 | Update fine schedule | 5 min | — |
| 10 | Smoke test (Section 11) | 30 min | All above |
| | **Total** | **~5.5 hours** | |

---

## 11. Smoke Test

```bash
# 1. Start server
cd backend && python run.py

# 2. Health check
curl http://localhost:8000/api/v1/health

# 3. Test analyze with all features ON
curl -X POST http://localhost:8000/api/v1/analyze \
  -F 'image=@test_image.jpg' \
  -F 'verify_stop_line=true' \
  -F 'detect_parking=true' \
  -F 'flow_direction=right' | python -m json.tool
```

### Frontend Walkthrough
```
1. Dashboard → KPI cards show different numbers
2. Analyze → motorcycle → helmet/triple riding works
3. Analyze → car → seatbelt + clean OCR (no duplicates)
4. Analyze → red light image → red-light violation detected
5. Analyze → parked car at edge → illegal parking detected
6. Violations → filter dropdown shows all 7 types
7. Violations → new violations have correct colors/badges
8. Challans → generate challan for red-light → fine = Rs 5000
9. Challans → generate challan for parking → fine = Rs 500
```

---

## 12. Known Limitations + Demo Tips

| Feature | Limitation | What to Say in Demo |
|---------|-----------|-------------------|
| Red-Light | YOLOv8n may miss small/distant traffic lights | "Works best with clear, close signals. We lowered the detection threshold for traffic lights specifically." |
| Red-Light | HSV may fail in extreme lighting (night/harsh sun) | "We added a brightness floor (V>120) to prevent false positives from reflections." |
| Illegal Parking | Single-frame can't tell 'stopped at light' vs 'parked' | "We use a conservative heuristic: only flags edge vehicles with no nearby person." |
| Illegal Parking | No predefined zones unless configured | "Strategy 2 works without config. In production, admins would configure zones per camera." |
| Stop-Line | Needs traffic light coupling to avoid false positives | "We improved this to skip green-light scenarios — vehicles should be past the line when it's green." |
| Wrong-Side | Requires per-camera flow direction config | "This is intentional — wrong-side detection must be camera-aware. We provide a simple dropdown." |
| Wrong-Side | Can't handle intersections/turning vehicles | "In production, optical flow across video frames would replace this single-frame heuristic." |
| EasyOCR | Occasional misreads on low-quality plates | "We pick the single highest-confidence detection and apply Indian plate regex cleaning." |

> **Golden demo tip:** When presenting heuristic features, lead with *"For the prototype, we use an efficient HSV/geometric approach. In production, this would be enhanced with..."* — judges love hearing you understand the production path.

---

> **After this plan, move to README + Demo Video + Submission (Sections 6-10 of `TrafficGuard_AI_Final_Phase_Completion.md`). Ship it! 🚀🏆**