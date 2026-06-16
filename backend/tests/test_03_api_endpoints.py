import io
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.config import settings

@pytest.fixture(scope="module")
def client():
    """Fixture that initializes the FastAPI TestClient within a lifespan context manager."""
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    """Verify that root endpoint returns project metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["version"] == settings.APP_VERSION

def test_health_endpoint(client):
    """Verify system health report."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert data["database"] == "connected"

def test_list_violations_empty_or_paginated(client):
    """Verify that violations records are returned in a paginated format."""
    response = client.get("/api/v1/violations")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "page_size" in data

def test_analytics_summary(client):
    """Verify analytics summary report."""
    response = client.get("/api/v1/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_violations" in data
    assert "pending_review" in data
    assert "top_violations" in data
    assert "vehicle_distribution" in data

def test_analytics_trends(client):
    """Verify trends endpoint with parameter handling."""
    response = client.get("/api/v1/analytics/trends?days=7")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_upload_image_endpoint(client):
    """Test image upload and processing pipeline on a small synthetic image."""
    # Create a small 100x100 white image in memory
    img_byte_arr = io.BytesIO()
    image = Image.new("RGB", (100, 100), color="white")
    image.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()
    
    files = {
        "file": ("test.jpg", img_bytes, "image/jpeg")
    }
    
    # We turn check_stop_line off to avoid index issues with small synthetic shape
    data = {
        "check_stop_line": "false",
        "enhance_contrast": "false",
        "camera_id": "TEST_CAM"
    }
    
    response = client.post("/api/v1/analyze/image", files=files, data=data)
    assert response.status_code == 200
    res_json = response.json()
    assert "violations" in res_json
    assert "vehicle_type" in res_json
    assert "inference_time_ms" in res_json
    assert "violation_count" in res_json
