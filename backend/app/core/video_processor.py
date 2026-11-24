"""
VisionGuard AI - Video Stream and Live Camera Inspection Processor
Handles simulated high-speed conveyor belt video feeds and live camera frame streaming.
"""

import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

from app.core.yolo_detector import YOLODefectDetector
from app.core.anomaly_detector import IndustrialAnomalyDetector
from app.core.severity_engine import SeverityEngine
from app.core.heatmap_generator import HeatmapGenerator


class InspectionStreamProcessor:
    """
    Processes video frames in real time, computing FPS, defect bounding boxes,
    heatmaps, and industrial pass/fail decisions.
    """
    def __init__(
        self,
        yolo_detector: YOLODefectDetector,
        anomaly_detector: IndustrialAnomalyDetector
    ):
        self.yolo = yolo_detector
        self.anomaly = anomaly_detector
        self.frame_count = 0
        self.fps_history = []
        self.last_timestamp = time.perf_counter()

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        enable_anomaly: bool = True,
        conf_threshold: float = 0.25,
        render_mode: str = "annotated"  # "annotated", "heatmap", "side_by_side"
    ) -> Dict:
        """
        Executes real-time inference on a single video/camera frame.
        """
        self.frame_count += 1
        now = time.perf_counter()
        delta = now - self.last_timestamp
        self.last_timestamp = now
        instant_fps = 1.0 / max(delta, 1e-4)
        self.fps_history.append(instant_fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = round(float(np.mean(self.fps_history)), 1)

        t_start = time.perf_counter()

        # Anomaly detection (optional per frame to optimize video FPS)
        anomaly_score = 0.0
        heatmap_norm = None
        if enable_anomaly:
            anomaly_res = self.anomaly.inspect(frame_bgr)
            anomaly_score = anomaly_res["anomaly_score"]
            heatmap_norm = anomaly_res["heatmap"]

        # YOLO object detection
        yolo_res = self.yolo.detect(
            image_bgr=frame_bgr,
            conf_threshold=conf_threshold,
            local_anomaly_map=heatmap_norm
        )
        defects = yolo_res["defects"]

        # Quality verdict
        verdict_res = SeverityEngine.evaluate_inspection_verdict(
            defects=defects,
            global_anomaly_score=anomaly_score
        )

        # Rendering
        if render_mode == "heatmap" and heatmap_norm is not None:
            rendered = HeatmapGenerator.generate_heatmap_overlay(frame_bgr, heatmap_norm, alpha=0.5)
            rendered = HeatmapGenerator.draw_annotations(rendered, defects, draw_contours=False)
        elif render_mode == "side_by_side":
            annotated = HeatmapGenerator.draw_annotations(frame_bgr, defects, draw_contours=True, heatmap_norm=heatmap_norm)
            heatmap_img = HeatmapGenerator.generate_heatmap_overlay(frame_bgr, heatmap_norm) if heatmap_norm is not None else frame_bgr
            rendered = HeatmapGenerator.create_side_by_side(frame_bgr, annotated, heatmap_img)
        else:
            # Default annotated
            rendered = HeatmapGenerator.draw_annotations(
                frame_bgr, defects, draw_contours=True, heatmap_norm=heatmap_norm
            )

        # Draw HUD overlay (FPS, Defect count, Verdict badge)
        cv2.putText(
            rendered,
            f"VisionGuard AI | FPS: {avg_fps} | Frame #{self.frame_count}",
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 180),
            1,
            cv2.LINE_AA
        )
        verdict_color = (40, 240, 40) if verdict_res["verdict"] == "PASS" else ((40, 40, 240) if verdict_res["verdict"] == "FAIL" else (15, 180, 245))
        cv2.putText(
            rendered,
            f"STATUS: {verdict_res['verdict']}",
            (16, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            verdict_color,
            2,
            cv2.LINE_AA
        )

        t_end = time.perf_counter()
        total_latency_ms = round((t_end - t_start) * 1000.0, 2)
        frame_b64 = HeatmapGenerator.encode_image_to_base64(rendered, quality=80)

        return {
            "frame_id": self.frame_count,
            "fps": avg_fps,
            "latency_ms": total_latency_ms,
            "defects": defects,
            "verdict": verdict_res,
            "image_base64": f"data:image/jpeg;base64,{frame_b64}"
        }
