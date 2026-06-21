# Implementation Plan: Final Phase Completion (Updated)

**Status:** Ready to execute  
**Estimated Time:** 5-6 hours  
**Deadline:** June 21, 2026

---

## 0. Pre-Implementation Quick Checks (Do These FIRST — 15 min)

Before writing any code, verify these 3 things:

### Check 1: Does YOLOv8n detect traffic lights?

Your `yolov8n_int8.onnx` is a nano model. Traffic lights (COCO class 9) are small objects — nano models have lower recall on them. Run this quick test:

```python
# Run in Python shell or a test script
from ultralytics import YOLO

model = YOLO('backend/models/yolov8n_int8.onnx')
results = model.predict('path/to/image_with_traffic_light.jpg', conf=0.15)

for det in results[0].boxes:
    cls_id = int(det.cls)
    if cls_id == 9:
        print(f'Traffic light found! conf={float(det.conf):.2f}, bbox={det.xyxy[0].tolist()}')

# If nothing prints → YOLOv8n can't detect traffic lights in this image.
# Fallback: Lower conf threshold to 0.10 for class 9 specifically,
# OR note in demo that 'red-light detection works best with clear signals'.
```

### Check 2: Does stop_line_y reach the backend?

Check your Analyze page's API call. When the 'Verify Stop-line Crossing' toggle is ON, confirm the request body includes `stop_line_y`:

```
# Check browser DevTools → Network tab → look at the POST /api/v1/analyze request body:
# Should contain something like:
# { ..., "verify_stop_line": true, "stop_line_y": 450 }
#
# If stop_line_y is missing, the red-light detection will use the default:
# int(img.shape[0] * 0.65)  — which is 65% down the image
```

### Check 3: Confirm your detection output format

The new detection modules expect this bbox dict format from your general detector:
```python
# Expected format per detection:
{
    'class_id': 9,              # int — COCO class ID
    'class_name': 'traffic light',  # str
    'confidence': 0.72,          # float 0-1
    'bbox': {
        'x1': 150.0, 'y1': 50.0,
        'x2': 180.0, 'y2': 120.0
    },
}
```

Add a quick `print(all_detections[:3])` in your `violation_engine.py` to confirm this matches your actual output format. If your keys are different (e.g., `cls` instead of `class_id`), adapt the new modules accordingly.

---

## 1. EasyOCR Single-Line Fix

**Time:** 30 minutes  
**File:** `backend/app/core/ocr_engine.py`  
**Action:** MODIFY

### What to Change

Find the section in your `extract_plate_text` (or equivalent method) where EasyOCR results are processed. Replace the concatenation logic with single-best-line logic:

```python
# ==========================================================
# BEFORE (broken — concatenates all lines):
# ==========================================================
# plate_text = ''
# for (bbox, text, conf) in results:
#     plate_text += text  # ← THIS causes 'NB062431WBHB06J2431'

# ==========================================================
# AFTER (fixed — picks single highest-confidence line):
# ==========================================================
best_text, best_conf = '', 0.0
for (bbox, text, conf) in results:
    if conf > best_conf:
        best_text, best_conf = text, conf

if best_conf < self.confidence_threshold:  # or cls.confidence_threshold
    return None, 0.0

# Clean for Indian plate format
cleaned = self._validate_indian_plate(best_text)  # Your existing cleanup
return cleaned or best_text, round(best_conf, 4)
```

### Verification
After this change, upload a car image with a visible plate. The Violations page should show something like `WB 06 J 2431` instead of `NB062431WBHB06J2431`.

---

## 2. Red-Light Violation Detection

**Time:** 2-3 hours  
**New Model:** NONE — uses HSV color thresholding + existing YOLO detections

### 2.1 Create `backend/app/core/red_light_detection.py`

**Action:** NEW FILE

Copy the full `red_light_detection.py` from **Section 2.1** of `TrafficGuard_AI_Final_Phase_Completion.md`. It contains:
- `detect_traffic_light_color(img, bbox)` — HSV thresholding on traffic light crop
- `check_red_light_violation(vehicle_detections, traffic_light_detections, stop_line_y, img)` — combines color + stop line

### 2.2 Integrate into `violation_engine.py`

**Action:** MODIFY `backend/app/core/violation_engine.py`

```python
# Add import at top
from app.core.red_light_detection import check_red_light_violation

# Add this stage AFTER your existing detection stages in the analyze() function:

# ========== STAGE: Red-Light Violation ==========
traffic_lights = [d for d in all_detections if d.get('class_id') == 9]
vehicle_dets = [d for d in all_detections if d.get('class_id') in {2, 3, 5, 7}]

if traffic_lights and pipeline_settings.get('verify_stop_line', False):
    stop_line_y = pipeline_settings.get('stop_line_y', int(img.shape[0] * 0.65))
    red_light_violations = check_red_light_violation(
        vehicle_detections=vehicle_dets,
        traffic_light_detections=traffic_lights,
        stop_line_y=stop_line_y,
        img=img,
    )
    all_violations.extend(red_light_violations)
    logger.info(f'Red-light check: {len(red_light_violations)} violations')
```

### 2.3 Annotation Drawing

**Action:** MODIFY your annotation drawing code (wherever you draw bounding boxes on the output image)

```python
# Add to your annotation loop for red-light violations:
for v in red_light_violations:
    bbox = v['vehicle_bbox']
    x1, y1 = int(bbox['x1']), int(bbox['y1'])
    x2, y2 = int(bbox['x2']), int(bbox['y2'])
    # Red box with thick border
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    # Label above box
    label = 'RED LIGHT VIOLATION'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated_img, (x1, y1 - th - 10), (x1 + tw, y1), (0, 0, 255), -1)
    cv2.putText(annotated_img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
```

### 2.4 Important: Adapt to YOUR detection format

If your detections use different keys (e.g., `cls` instead of `class_id`, or `xyxy` instead of `bbox` dict), update the references in `red_light_detection.py` accordingly. The logic stays the same — only the key names change.

---

## 3. Illegal Parking Detection

**Time:** 2 hours  
**New Model:** NONE — position + proximity heuristic

### 3.1 Create `backend/app/core/illegal_parking_detection.py`

**Action:** NEW FILE

Copy the full `illegal_parking_detection.py` from **Section 3.1** of `TrafficGuard_AI_Final_Phase_Completion.md`. It contains:
- `check_illegal_parking(all_detections, img_shape, no_parking_zones, ...)` — two strategies
  - Strategy 1: Vehicle center inside a no-parking zone rect
  - Strategy 2: Vehicle at road edge (left/right 15%) + no person within 120px

> **Note:** Motorcycles (`class_id: 3`) are intentionally EXCLUDED. Parked motorcycles are normal and would cause mass false positives. Only cars (2), buses (5), trucks (7) trigger illegal parking.

### 3.2 Integrate into `violation_engine.py`

**Action:** MODIFY `backend/app/core/violation_engine.py`

```python
# Add import at top
from app.core.illegal_parking_detection import check_illegal_parking

# Add this stage AFTER your existing detection stages:

# ========== STAGE: Illegal Parking ==========
if pipeline_settings.get('detect_parking', True):  # Default ON
    parking_violations = check_illegal_parking(
        all_detections=all_detections,
        img_shape=img.shape,
        no_parking_zones=pipeline_settings.get('no_parking_zones', []),
    )
    all_violations.extend(parking_violations)
    logger.info(f'Parking check: {len(parking_violations)} violations')
```

### 3.3 Annotation Drawing

```python
# Add to your annotation loop for parking violations:
for v in parking_violations:
    bbox = v['vehicle_bbox']
    x1, y1 = int(bbox['x1']), int(bbox['y1'])
    x2, y2 = int(bbox['x2']), int(bbox['y2'])
    # Blue box (parking = blue zone)
    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 100, 0), 3)
    label = 'ILLEGAL PARKING'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated_img, (x1, y1 - th - 10), (x1 + tw, y1), (255, 100, 0), -1)
    cv2.putText(annotated_img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
```

### 3.4 Frontend: Add Toggle in Analyze.tsx

**Action:** MODIFY `frontend/src/pages/Analyze.tsx`

Add in the Configuration panel, next to your existing toggles:

```tsx
// State
const [detectParking, setDetectParking] = useState(true);

// JSX — add to Configuration panel
<label className='flex items-center gap-2 text-sm text-gray-300'>
  <input
    type='checkbox'
    checked={detectParking}
    onChange={(e) => setDetectParking(e.target.checked)}
    className='rounded bg-gray-700 border-gray-600'
  />
  Detect Illegal Parking
</label>

// Include in API request body:
const formData = new FormData();
// ... existing fields ...
formData.append('detect_parking', String(detectParking));
```

---

## 4. Frontend: Violation Type Color Mapping

**Action:** MODIFY wherever you map violation types to colors/badges in the frontend

```tsx
// Update your violation type → color mapping
// (This might be in Violations.tsx, Dashboard.tsx, or a shared constants file)

const VIOLATION_COLORS: Record<string, string> = {
  'Helmet Non Compliance': '#f59e0b',     // Amber
  'Seatbelt Non Compliance': '#f97316',   // Orange
  'Triple Riding': '#ef4444',             // Red
  'Wrong Side Driving': '#dc2626',        // Dark Red
  'Stop Line Violation': '#eab308',       // Yellow
  'Red Light Violation': '#ff0000',       // Bright Red  ← NEW
  'Illegal Parking': '#3b82f6',           // Blue        ← NEW
};

const SEVERITY_COLORS: Record<string, string> = {
  'LOW': '#22c55e',       // Green
  'MEDIUM': '#f59e0b',    // Amber
  'HIGH': '#ef4444',      // Red
  'CRITICAL': '#dc2626',  // Dark Red
};
```

Also update any dropdown filters that list violation types:
```tsx
const VIOLATION_TYPE_OPTIONS = [
  { value: 'all', label: 'All Types' },
  { value: 'Helmet Non Compliance', label: 'Helmet Non Compliance' },
  { value: 'Seatbelt Non Compliance', label: 'Seatbelt Non Compliance' },
  { value: 'Triple Riding', label: 'Triple Riding' },
  { value: 'Wrong Side Driving', label: 'Wrong Side Driving' },
  { value: 'Stop Line Violation', label: 'Stop Line Violation' },
  { value: 'Red Light Violation', label: 'Red Light Violation' },      // NEW
  { value: 'Illegal Parking', label: 'Illegal Parking' },              // NEW
];
```

---

## 5. Test Images — What to Use for Each Violation

| Violation | What Image to Test With | What to Look For |
|-----------|------------------------|-----------------|
| Helmet Non Compliance | Motorcycle rider without helmet | Yellow bbox on rider |
| Seatbelt Non Compliance | Car with visible driver through windshield | Orange bbox on car |
| Triple Riding | Motorcycle with 3+ riders | Red bbox with rider count label |
| Wrong Side Driving | Vehicle heading against lane flow | Dark red bbox |
| Stop Line Violation | Vehicle past the yellow stop line | Yellow bbox |
| **Red Light Violation** | **Image with visible red traffic light + vehicle past line** | **Red bbox + 'RED LIGHT VIOLATION' label** |
| **Illegal Parking** | **Car at extreme left/right edge, no people nearby** | **Blue bbox + 'ILLEGAL PARKING' label** |

### Red Light Test — Finding Good Images

The red-light detection depends on YOLO detecting the traffic light (COCO class 9). For best results:
- Use images where the traffic light is **clearly visible and not too small**
- The traffic light should be **vertically oriented** (standard 3-circle layout)
- The red light should be **lit** (bright red circle in top third)
- If YOLOv8n misses the traffic light entirely, the detection will gracefully return 0 violations (no crash)

### Illegal Parking Test

- Use a CCTV-style image with a car sitting at the very left or right edge
- Ensure no person is standing within ~120px of the car in the image
- Cars in the center of the image will NOT trigger (by design)

---

## 6. Post-Implementation Smoke Test

After implementing all 3 features, run these tests:

### 6.1 Backend Health
```bash
# Start server
cd backend && python run.py

# Check health
curl http://localhost:8000/api/v1/health
# Expected: {"status": "ok", ...}
```

### 6.2 Analyze Image (via curl)
```bash
# Test with an image file
curl -X POST http://localhost:8000/api/v1/analyze \
  -F 'image=@test_image.jpg' \
  -F 'verify_stop_line=true' \
  -F 'detect_parking=true' | python -m json.tool

# Check the response for:
# - 'violations' array contains new types ('Red Light Violation', 'Illegal Parking')
# - 'plate_text' is clean (not duplicated)
# - 'annotated_image_path' exists
```

### 6.3 Frontend Walkthrough
```
1. Open http://localhost:5173
2. Dashboard → verify KPIs load (not all same number)
3. Analyze → upload motorcycle image → check helmet/triple riding detection
4. Analyze → upload car image → check seatbelt + clean plate OCR
5. Analyze → upload red-light image → check red-light violation appears
6. Analyze → upload parked car image → check illegal parking appears
7. Violations → confirm new violation types appear in the table
8. Violations → filter dropdown shows 'Red Light Violation' and 'Illegal Parking'
9. Challans → generate a challan for a red-light violation → verify correct fine (₹5000)
```

---

## 7. Recommended Execution Order

| Step | Task | Time | Depends On |
|------|------|------|-----------|
| **0** | Pre-implementation checks (Section 0) | 15 min | Nothing |
| **1** | EasyOCR single-line fix | 30 min | Nothing |
| **2** | Create `red_light_detection.py` | 45 min | Check 0 done |
| **3** | Integrate red-light into `violation_engine.py` | 30 min | Step 2 |
| **4** | Create `illegal_parking_detection.py` | 45 min | Nothing |
| **5** | Integrate parking into `violation_engine.py` | 30 min | Step 4 |
| **6** | Add annotation drawing for both | 30 min | Steps 3, 5 |
| **7** | Frontend: parking toggle + color mapping + dropdown | 30 min | Steps 3, 5 |
| **8** | Smoke test (Section 6) | 30 min | All above |
| | **Total** | **~4.5 hours** | |

---

## 8. Complete File Changes Summary

| Action | File | What Changes |
|--------|------|-------------|
| MODIFY | `backend/app/core/ocr_engine.py` | Pick single highest-confidence EasyOCR line |
| **NEW** | `backend/app/core/red_light_detection.py` | HSV traffic light color + stop-line check |
| **NEW** | `backend/app/core/illegal_parking_detection.py` | Zone-based + edge proximity heuristic |
| MODIFY | `backend/app/core/violation_engine.py` | Import + call both new modules, add annotation |
| MODIFY | `frontend/src/pages/Analyze.tsx` | Add 'Detect Illegal Parking' toggle |
| MODIFY | `frontend/src/pages/Violations.tsx` (or constants) | Add 2 new types to filter dropdown + color map |
| MODIFY | `frontend/src/pages/Dashboard.tsx` (or constants) | Add 2 new types to color map (if separate) |

**Total: 2 new files + 4-5 modified files**

---

## 9. Known Limitations (Be Honest in Demo)

These are heuristic-based detections. Be transparent about their limitations — judges respect honesty:

| Feature | Limitation | Mitigation |
|---------|-----------|-----------|
| Red-Light | YOLOv8n may miss small/distant traffic lights | Works best with clear, close-up signals. Gracefully returns 0 violations if no light detected |
| Red-Light | HSV thresholds may fail in extreme lighting (night/harsh sun) | CLAHE preprocessing helps. Thresholds are tunable |
| Illegal Parking | Single-frame heuristic can't distinguish 'stopped at light' from 'parked' | Only flags edge vehicles with no nearby person — conservative approach |
| Illegal Parking | No predefined zones unless configured per camera | Strategy 2 (edge heuristic) works without zone config |
| EasyOCR | Still occasional misreads vs PaddleOCR | Indian plate regex catches most errors. Best on clear, front-facing plates |

> **Demo tip:** When presenting, say *"These use HSV thresholding and proximity heuristics for efficiency — no additional model overhead. In production, we'd add a dedicated traffic light classifier for higher accuracy."*

---

> **After completing this plan, move to README + Demo Video + Submission using Sections 6-10 of `TrafficGuard_AI_Final_Phase_Completion.md`. You're in the final stretch! 🚀**