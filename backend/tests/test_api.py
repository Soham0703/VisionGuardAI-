"""
VisionGuard AI - Backend API Integration Tests
"""

import io
import pytest
from pathlib import Path
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.database.database import init_db

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    init_db()


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "VisionGuard AI" in data["service"]
    assert len(data["classes"]) == 6


def test_model_metrics_endpoint():
    response = client.get("/api/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "yolo_detector" in data
    assert "anomaly_detector" in data
    assert "mAP50" in data["yolo_detector"]["overall_metrics"]
    assert "roc_auc" in data["anomaly_detector"]["metrics"]


def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0


def test_image_inspection_endpoint():
    # Create a synthetic 512x512 image with a scratch
    img = np.full((512, 512, 3), 160, dtype=np.uint8)
    cv2.line(img, (100, 100), (350, 350), (20, 20, 20), 3)
    _, buffer = cv2.imencode(".jpg", img)
    file_bytes = io.BytesIO(buffer.tobytes())

    response = client.post(
        "/api/inspect/image",
        files={"file": ("test_scratch.jpg", file_bytes, "image/jpeg")},
        params={"product_type": "metal_sheet", "conf_threshold": 0.20}
    )
    assert response.status_code == 200
    data = response.json()

    assert "inspection_id" in data
    assert "verdict" in data
    assert data["verdict"]["verdict"] in ["PASS", "REVIEW", "FAIL"]
    assert "anomaly" in data
    assert "latency" in data
    assert "images" in data
    assert "annotated" in data["images"]
    assert "heatmap" in data["images"]
    assert "side_by_side" in data["images"]


def test_analytics_and_history_endpoints():
    response = client.get("/api/analytics")
    assert response.status_code == 200
    analytics = response.json()
    assert "total_inspected" in analytics
    assert analytics["total_inspected"] >= 1
    assert "yield_rate_percentage" in analytics

    hist_response = client.get("/api/history?limit=10")
    assert hist_response.status_code == 200
    hist = hist_response.json()
    assert "items" in hist
    assert len(hist["items"]) >= 1


def test_csv_export_endpoint():
    response = client.get("/api/export/csv")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert "ID,Timestamp,Product,Verdict" in response.text
