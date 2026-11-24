"""
VisionGuard AI - Computer Vision Pipeline Unit Tests
"""

import pytest
import numpy as np
import cv2

from app.core.severity_engine import SeverityEngine
from app.core.heatmap_generator import HeatmapGenerator
from app.core.yolo_detector import YOLODefectDetector
from app.core.anomaly_detector import IndustrialAnomalyDetector


def test_severity_engine_calculation():
    img_shape = (512, 512)
    # Test 1: Critical crack with high area and anomaly score
    res_crit = SeverityEngine.calculate_defect_severity(
        class_name="crack",
        bbox=[100, 100, 250, 250],
        img_shape=img_shape,
        confidence=0.92,
        local_anomaly_score=0.85
    )
    assert res_crit["severity_level"] == "CRITICAL"
    assert res_crit["severity_score"] >= 0.68

    # Test 2: Low-risk minor scratch
    res_low = SeverityEngine.calculate_defect_severity(
        class_name="scratch",
        bbox=[50, 50, 60, 60],
        img_shape=img_shape,
        confidence=0.45,
        local_anomaly_score=0.10
    )
    assert res_low["severity_level"] == "LOW"
    assert res_low["severity_score"] < 0.40


def test_verdict_decision_logic():
    # Verdict with critical defect -> FAIL
    defects = [{"severity_level": "CRITICAL"}]
    v1 = SeverityEngine.evaluate_inspection_verdict(defects, global_anomaly_score=0.2)
    assert v1["verdict"] == "FAIL"

    # Verdict with 0 defects & low anomaly -> PASS
    v2 = SeverityEngine.evaluate_inspection_verdict([], global_anomaly_score=0.1)
    assert v2["verdict"] == "PASS"

    # Verdict with 1 medium defect -> REVIEW
    v3 = SeverityEngine.evaluate_inspection_verdict([{"severity_level": "MEDIUM"}], global_anomaly_score=0.3)
    assert v3["verdict"] == "REVIEW"


def test_heatmap_generator():
    img = np.full((128, 128, 3), 180, dtype=np.uint8)
    hmap = np.zeros((128, 128), dtype=np.float32)
    hmap[30:60, 30:60] = 0.9

    overlay = HeatmapGenerator.generate_heatmap_overlay(img, hmap, alpha=0.5)
    assert overlay.shape == img.shape
    assert overlay.dtype == np.uint8

    defects = [{
        "bbox": [20, 20, 80, 80],
        "class_name": "crack",
        "severity_level": "CRITICAL",
        "confidence": 0.95
    }]
    annotated = HeatmapGenerator.draw_annotations(img, defects, draw_contours=True, heatmap_norm=hmap)
    assert annotated.shape == img.shape

    side_by_side = HeatmapGenerator.create_side_by_side(img, annotated, overlay)
    assert side_by_side.shape[0] == 128
    assert side_by_side.shape[1] > 128 * 2


def test_anomaly_detector_inference():
    detector = IndustrialAnomalyDetector()
    dummy = np.full((256, 256, 3), 150, dtype=np.uint8)
    res = detector.inspect(dummy)
    assert "anomaly_score" in res
    assert 0.0 <= res["anomaly_score"] <= 1.0
    assert "heatmap" in res
    assert res["heatmap"].shape == (256, 256)
