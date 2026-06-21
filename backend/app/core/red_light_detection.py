"""
Red-Light Violation Detection
Uses HSV color thresholding on traffic light crops (COCO class 9)
combined with stop-line position to detect violations.
No new ML model required.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def detect_traffic_light_color(img: np.ndarray, bbox: Dict) -> Optional[str]:
    """
    Classify a traffic light as RED, YELLOW, or GREEN
    using HSV color thresholding on the cropped region.

    The traffic light is divided into 3 vertical zones:
    - Top third: RED light position
    - Middle third: YELLOW light position
    - Bottom third: GREEN light position

    Returns: 'RED', 'YELLOW', 'GREEN', or None if uncertain
    """
    x1, y1 = int(bbox['x1']), int(bbox['y1'])
    x2, y2 = int(bbox['x2']), int(bbox['y2'])

    # Validate bounds
    h_img, w_img = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]

    if h < 15 or w < 5:  # Too small to analyze
        return None

    # Split into 3 vertical zones
    top_zone = hsv[0:h//3, :]
    mid_zone = hsv[h//3:2*h//3, :]
    bot_zone = hsv[2*h//3:, :]

    # --- RED detection (top zone) ---
    # Red wraps around in HSV: H=0-10 OR H=160-180
    red_lower1 = np.array([0, 100, 120])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 120])
    red_upper2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(top_zone, red_lower1, red_upper1) | \
               cv2.inRange(top_zone, red_lower2, red_upper2)
    red_pixels = cv2.countNonZero(red_mask)

    # --- YELLOW detection (mid zone) ---
    yellow_lower = np.array([15, 100, 120])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(mid_zone, yellow_lower, yellow_upper)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    # --- GREEN detection (bottom zone) ---
    green_lower = np.array([35, 100, 120])
    green_upper = np.array([85, 255, 255])
    green_mask = cv2.inRange(bot_zone, green_lower, green_upper)
    green_pixels = cv2.countNonZero(green_mask)

    # Minimum pixel threshold: at least 5% of zone area must match
    zone_area = (h // 3) * w
    min_threshold = zone_area * 0.05

    scores = {'RED': red_pixels, 'YELLOW': yellow_pixels, 'GREEN': green_pixels}
    best = max(scores, key=scores.get)

    if scores[best] < min_threshold:
        logger.debug(f'Traffic light color uncertain: {scores}')
        return None

    logger.info(f'Traffic light detected: {best} (R={red_pixels}, Y={yellow_pixels}, G={green_pixels})')
    return best


def check_red_light_violation(
    vehicle_detections: List[Dict],
    traffic_light_detections: List[Dict],
    stop_line_y: int,
    img: np.ndarray,
    flow_direction: str = 'none',
) -> List[Dict]:
    """
    Check if any vehicle is past the stop line while the traffic light is RED.

    Args:
        vehicle_detections: vehicles from general detector (class 2,3,5,7)
        traffic_light_detections: traffic lights from general detector (class 9)
        stop_line_y: y-coordinate of the stop line on the image
        img: original BGR image

    Returns:
        List of violation dicts
    """
    violations = []

    if not traffic_light_detections:
        return violations  # No traffic lights in frame

    # Step 1: Check if ANY traffic light in the frame is RED
    from app.models.violation import ViolationType, Severity
    
    is_red = False
    red_confidence = 0.0
    for tl in traffic_light_detections:
        color = detect_traffic_light_color(img, tl['bbox'])
        if color == 'RED':
            is_red = True
            red_confidence = tl.get('confidence', 0.8)
            break

    if not is_red:
        return violations  # Light not red — no violation possible

    # Step 2: Check which vehicles have crossed the stop line
    vehicle_class_ids = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    img_h, img_w = img.shape[:2]
    
    for det in vehicle_detections:
        if det.get('class_id') not in vehicle_class_ids:
            continue

        vehicle_front_y = det['bbox']['y2']  # Bottom edge = front of vehicle
        v_cx = (det['bbox']['x1'] + det['bbox']['x2']) / 2

        # Filter out oncoming traffic based on flow direction
        if flow_direction == 'left' and v_cx > img_w / 2:
            continue
        if flow_direction == 'right' and v_cx < img_w / 2:
            continue

        # Vehicle's front is PAST (below in image coords) the stop line
        if vehicle_front_y > stop_line_y:
            violations.append({
                'violation_type': ViolationType.RED_LIGHT_VIOLATION,
                'severity': Severity.HIGH,
                'confidence': round(min(det.get('confidence', 0.8), red_confidence), 4),
                'bbox': det['bbox'],
                'vehicle_type': det.get('class_name', 'Vehicle'),
                'description': f'Vehicle crossed stop line at y={int(vehicle_front_y)} while signal is RED (stop_line_y={stop_line_y})',
            })

    logger.info(f'Red-light check: is_red={is_red}, vehicles_past_line={len(violations)}')
    return violations
