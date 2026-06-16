import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import MagicMock

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.violation_engine import ViolationEngine
from app.models.violation import ViolationType

@pytest.fixture
def mock_engine():
    """Create a ViolationEngine with mocked detector methods."""
    engine = ViolationEngine()
    engine.det = MagicMock()
    
    # Standard class mapping
    engine.det.VEHICLE_CLASS_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck
    engine.det.PERSON_CLASS_ID = 0
    engine.det.MOTORCYCLE_CLASS_ID = 3
    engine.det.CAR_CLASS_ID = 2
    
    return engine

def test_helmet_violation_logic(mock_engine):
    """Test that helmet violation logic correctly flags no-helmet detections on a motorcycle."""
    # Mock helmet detector to return a "no-helmet" detection
    mock_engine.det.detect_helmets = MagicMock(return_value=[{
        "class_id": 1,
        "class_name": "no-helmet",
        "confidence": 0.85,
        "bbox": {"x1": 110, "y1": 100, "x2": 150, "y2": 150}
    }])
    
    # Motorcyclist at (100, 100, 200, 300)
    motorcycles = [{
        "class_id": 3,
        "class_name": "motorcycle",
        "confidence": 0.90,
        "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 300}
    }]
    persons = [{
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.88,
        "bbox": {"x1": 100, "y1": 100, "x2": 180, "y2": 250}
    }]
    
    # Mock IOU calculations
    mock_engine.det.compute_iou = MagicMock(side_effect=lambda box1, box2: 0.80)
    
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    violations = mock_engine._check_helmet(img, motorcycles, persons)
    
    assert len(violations) == 1
    assert violations[0]["violation_type"] == ViolationType.HELMET_NON_COMPLIANCE
    assert violations[0]["confidence"] == 0.85

def test_triple_riding_logic(mock_engine):
    """Test triple riding checks when 3 persons are detected on a motorcycle."""
    # Mock pose model to be empty, so it uses persons list
    mock_engine.det.detect_poses = MagicMock(return_value=[])
    
    motorcycles = [{
        "class_id": 3,
        "class_name": "motorcycle",
        "confidence": 0.90,
        "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 300}
    }]
    
    # 3 persons overlapping the motorcycle
    persons = [
        {"class_id": 0, "class_name": "person", "confidence": 0.80, "bbox": {"x1": 110, "y1": 110, "x2": 150, "y2": 200}},
        {"class_id": 0, "class_name": "person", "confidence": 0.81, "bbox": {"x1": 130, "y1": 110, "x2": 170, "y2": 200}},
        {"class_id": 0, "class_name": "person", "confidence": 0.82, "bbox": {"x1": 150, "y1": 110, "x2": 190, "y2": 200}},
    ]
    
    # Mock IOU compute so that all 3 persons overlap the motorcycle bbox
    mock_engine.det.compute_iou = MagicMock(side_effect=lambda box1, box2: 0.50)
    
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    violations = mock_engine._check_triple_riding(img, motorcycles, persons)
    
    assert len(violations) == 1
    assert violations[0]["violation_type"] == ViolationType.TRIPLE_RIDING
    assert violations[0]["person_count"] == 3

def test_seatbelt_violation_logic(mock_engine):
    """Test seatbelt logic when seatbelt model returns a no-seatbelt detection."""
    mock_engine.det.detect_seatbelts = MagicMock(return_value=[{
        "class_id": 1,
        "class_name": "no-seatbelt",
        "confidence": 0.92,
        "bbox": {"x1": 150, "y1": 120, "x2": 250, "y2": 220}
    }])
    
    vehicles = [{
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.95,
        "bbox": {"x1": 100, "y1": 100, "x2": 300, "y2": 300}
    }]
    persons = [{
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.90,
        "bbox": {"x1": 120, "y1": 110, "x2": 260, "y2": 250}
    }]
    
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    violations = mock_engine._check_seatbelt(img, vehicles, persons)
    
    assert len(violations) == 1
    assert violations[0]["violation_type"] == ViolationType.SEATBELT_NON_COMPLIANCE
    assert violations[0]["confidence"] == 0.92

def test_stop_line_violation_logic(mock_engine):
    """Test stop-line violation when vehicle center is past stop-line."""
    vehicles = [{
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.95,
        "bbox": {"x1": 100, "y1": 310, "x2": 300, "y2": 450} # bottom at 450
    }]
    
    violations = mock_engine._check_stop_line(vehicles, stop_y=400, img_height=480)
    
    assert len(violations) == 1
    assert violations[0]["violation_type"] == ViolationType.STOP_LINE_VIOLATION

def test_red_light_violation_logic(mock_engine):
    """Test red-light violation checks when light is analyzed as red and vehicle is moving."""
    traffic_lights = [{
        "class_id": 9,
        "class_name": "traffic light",
        "confidence": 0.90,
        "bbox": {"x1": 200, "y1": 20, "x2": 240, "y2": 100}
    }]
    vehicles = [{
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.95,
        "bbox": {"x1": 100, "y1": 300, "x2": 300, "y2": 450}
    }]
    
    # Mock red light detector check to return True
    mock_engine._is_traffic_light_red = MagicMock(return_value=True)
    
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    violations = mock_engine._check_red_light(img, traffic_lights, vehicles, h=480, w=640)
    
    assert len(violations) == 1
    assert violations[0]["violation_type"] == ViolationType.RED_LIGHT_VIOLATION

def test_wrong_side_driving_logic(mock_engine):
    """Test wrong side driving check heuristics."""
    vehicles = [{
        "class_id": 2,
        "class_name": "car",
        "confidence": 0.95,
        "bbox": {"x1": 20, "y1": 200, "x2": 80, "y2": 300} # center around 50 (far left)
    }]
    
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    violations = mock_engine._check_wrong_side(img, vehicles, img_width=640, lane_direction="right")
    
    assert len(violations) == 1
    assert violations[0]["violation_type"] == ViolationType.WRONG_SIDE_DRIVING
