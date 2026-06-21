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
from app.models.violation import ViolationType, Severity

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
                'violation_type': ViolationType.WRONG_SIDE_DRIVING,
                'severity': Severity.HIGH,
                'confidence': round(det.get('confidence', 0.7) * 0.7, 4),  # 0.7x heuristic penalty
                'bbox': det['bbox'],
                'vehicle_type': det.get('class_name', 'Vehicle'),
                'description': f"Vehicle on wrong side (expected flow: {flow})",
            })

    logger.info(f'Wrong-side: flow={flow}, violations={len(violations)}')
    return violations
