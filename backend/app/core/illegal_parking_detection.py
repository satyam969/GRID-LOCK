"""
Illegal Parking Detection
Two strategies:
  1. Vehicle center inside a predefined no-parking polygon zone
  2. Vehicle at road edge with no person nearby (unoccupied = likely parked)
No new ML model required.
"""
import logging
import math
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Default no-parking zones (can be configured per camera)
# Format: list of rectangular zones with optional labels
DEFAULT_NO_PARKING_ZONES = [
    # Example: Bus stop area (left side of typical CCTV frame)
    # {'x1': 0, 'y1': 300, 'x2': 200, 'y2': 600, 'label': 'Bus Stop'},
    # Configure these per camera in Pipeline Settings
]


def point_in_rect(px: float, py: float, zone: Dict) -> bool:
    """Check if a point is inside a rectangular zone."""
    return (zone['x1'] <= px <= zone['x2'] and
            zone['y1'] <= py <= zone['y2'])


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def check_illegal_parking(
    all_detections: List[Dict],
    img_shape: Tuple[int, int],
    no_parking_zones: Optional[List[Dict]] = None,
    person_proximity_threshold: float = 120.0,
    edge_percentage: float = 0.15,
) -> List[Dict]:
    """
    Detect illegally parked vehicles.

    Strategy 1: Vehicle center is inside a predefined no-parking zone.
    Strategy 2: Vehicle is at the road edge AND no person is nearby
                (unoccupied vehicle at edge = likely parked).

    Args:
        all_detections: all detections from general detector
        img_shape: (height, width) of the image
        no_parking_zones: list of zone dicts [{'x1','y1','x2','y2','label'}]
        person_proximity_threshold: max pixel distance to consider a person 'near' a vehicle
        edge_percentage: fraction of image width considered 'road edge'

    Returns:
        List of violation dicts
    """
    violations = []
    zones = no_parking_zones or DEFAULT_NO_PARKING_ZONES

    # Separate vehicles and persons
    vehicles = [d for d in all_detections if d.get('class_id') in {2, 5, 7}]  # car, bus, truck
    persons = [d for d in all_detections if d.get('class_id') == 0]

    img_h, img_w = img_shape[:2]

    # IMPROVEMENT: Minimum area threshold
    min_area = img_h * img_w * 0.01  # Vehicle must be >=1% of frame

    for vehicle in vehicles:
        vx1 = vehicle['bbox']['x1']
        vy1 = vehicle['bbox']['y1']
        vx2 = vehicle['bbox']['x2']
        vy2 = vehicle['bbox']['y2']
        v_cx = (vx1 + vx2) / 2  # Vehicle center x
        v_cy = (vy1 + vy2) / 2  # Vehicle center y
        v_area = (vx2 - vx1) * (vy2 - vy1)

        # Skip tiny distant vehicles
        if v_area < min_area:
            continue

        is_violation = False
        reason = ''

        # --- Strategy 1: No-Parking Zone ---
        from app.models.violation import ViolationType, Severity
        
        if zones:
            for zone in zones:
                if point_in_rect(v_cx, v_cy, zone):
                    is_violation = True
                    reason = f"Vehicle in no-parking zone: {zone.get('label', 'Restricted Area')}"
                    break

        # --- Strategy 2: Road Edge + No Person Nearby ---
        if not is_violation:
            # Check if vehicle is at the extreme left or right edge of the frame
            edge_threshold = img_w * edge_percentage
            is_at_left_edge = v_cx < edge_threshold
            is_at_right_edge = v_cx > (img_w - edge_threshold)
            is_at_edge = is_at_left_edge or is_at_right_edge

            if is_at_edge:
                # Check if any person is near this vehicle
                has_nearby_person = False
                for person in persons:
                    px = (person['bbox']['x1'] + person['bbox']['x2']) / 2
                    py = (person['bbox']['y1'] + person['bbox']['y2']) / 2
                    dist = distance_2d(v_cx, v_cy, px, py)
                    if dist < person_proximity_threshold:
                        has_nearby_person = True
                        break

                if not has_nearby_person:
                    is_violation = True
                    side = 'left' if is_at_left_edge else 'right'
                    reason = f'Unoccupied vehicle parked at {side} road edge'

        if is_violation:
            violations.append({
                'violation_type': ViolationType.ILLEGAL_PARKING,
                'severity': Severity.MEDIUM,
                'confidence': round(vehicle.get('confidence', 0.7) * 0.85, 4),
                'bbox': vehicle['bbox'],
                'vehicle_type': vehicle.get('class_name', 'Vehicle'),
                'description': reason,
            })

    logger.info(f'Parking check: {len(vehicles)} vehicles, {len(violations)} violations')
    return violations
