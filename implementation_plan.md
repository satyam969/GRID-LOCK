# Implementation Plan: Final Phase Completion

## Goal
Implement the remaining features from the `TrafficGuard_AI_Final_Phase_Completion.md` document to finalize the prototype for the hackathon submission.

## Current Status of Final Phase Tasks
1. **EasyOCR Single-Line Fix:** Needs implementation. Our current code concatenates all strings (which the guide points out causes garbage output). We need to pick the single highest-confidence line instead.
2. **Red-Light Violation Detection:** Needs implementation.
3. **Illegal Parking Detection:** Needs implementation.
4. **Updated Fine Schedule:** ✅ Already completed in the previous sprint!
5. **GitHub Repository Preparation:** We can add `red_light_detection.py` and `illegal_parking_detection.py` to the core folder, fulfilling the architecture layout requested.

## Proposed Changes

### 1. EasyOCR Single-Line Fix
#### [MODIFY] [backend/app/core/ocr_engine.py](file:///d:/trafficguard-ai/trafficguard-ai/backend/app/core/ocr_engine.py)
- Change `extract_plate_text` to pick the single highest-confidence result from the EasyOCR output rather than concatenating all texts.

### 2. Red-Light Violation Detection
#### [NEW] [backend/app/core/red_light_detection.py](file:///d:/trafficguard-ai/trafficguard-ai/backend/app/core/red_light_detection.py)
- Create new file with `detect_traffic_light_color` (using HSV color thresholding) and `check_red_light_violation` functions.
#### [MODIFY] [backend/app/core/violation_engine.py](file:///d:/trafficguard-ai/trafficguard-ai/backend/app/core/violation_engine.py)
- Import `check_red_light_violation`.
- Integrate the red-light check stage into the `analyze` function (using `class_id == 9` for traffic lights).

### 3. Illegal Parking Detection
#### [NEW] [backend/app/core/illegal_parking_detection.py](file:///d:/trafficguard-ai/trafficguard-ai/backend/app/core/illegal_parking_detection.py)
- Create new file with `check_illegal_parking` function (heuristic based on road edge proximity and lack of nearby persons).
#### [MODIFY] [backend/app/core/violation_engine.py](file:///d:/trafficguard-ai/trafficguard-ai/backend/app/core/violation_engine.py)
- Import `check_illegal_parking`.
- Integrate the illegal parking check stage into the `analyze` function.
#### [MODIFY] [frontend/src/pages/Analyze.tsx](file:///d:/trafficguard-ai/trafficguard-ai/frontend/src/pages/Analyze.tsx)
- Add a UI toggle in the Configuration panel for "Detect Illegal Parking".

## User Review Required
> [!IMPORTANT]
> The Red-Light and Illegal Parking features use heuristics (color thresholding and position/proximity, respectively) rather than dedicated machine learning models. This is highly efficient and perfectly acceptable for the hackathon, but may require parameter tuning (like the red HSV bounds or edge percentages) depending on your test images. I will implement them exactly as suggested in the guide.

Please approve this plan so I can begin the implementation!
