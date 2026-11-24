"""
VisionGuard AI - YOLO Defect Localization Inference Engine
Performs real-time defect localization, bounding box extraction, and multi-class detection.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from app.core.severity_engine import SeverityEngine

DEFECT_CLASSES = [
    "scratch",
    "crack",
    "hole",
    "surface_abnormality",
    "contamination",
    "deformation"
]


class YOLODefectDetector:
    """
    Production inference wrapper for fine-tuned YOLOv8 industrial defect model.
    """
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25):
        self.conf_threshold = conf_threshold
        if model_path is None:
            # Default model storage location
            default_p = Path(__file__).parent.parent / "models_storage" / "yolov8_defect.pt"
            if default_p.exists():
                model_path = str(default_p)
            else:
                model_path = "yolov8n.pt"  # Pretrained fallback

        print(f"Loading YOLO defect model from {model_path}...")
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.classes = DEFECT_CLASSES

    def detect(
        self,
        image_bgr: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: float = 0.45,
        local_anomaly_map: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Runs object detection on input image and computes defect severity.
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        h, w = image_bgr.shape[:2]

        t0 = time.perf_counter()
        results = self.model.predict(
            source=image_bgr,
            conf=conf,
            iou=iou_threshold,
            verbose=False,
            imgsz=512
        )
        t1 = time.perf_counter()
        inference_time_ms = round((t1 - t0) * 1000.0, 2)

        defects = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                box_conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                class_name = (
                    self.classes[cls_id] if cls_id < len(self.classes)
                    else results[0].names.get(cls_id, f"defect_{cls_id}")
                )

                x1, y1, x2, y2 = xyxy
                # Clamp coordinates
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(x1 + 1, min(w, x2))
                y2 = max(y1 + 1, min(h, y2))
                bbox = [int(x1), int(y1), int(x2), int(y2)]

                # Extract local anomaly score within bbox region if map provided
                local_anomaly = 0.0
                if local_anomaly_map is not None:
                    patch = local_anomaly_map[y1:y2, x1:x2]
                    if patch.size > 0:
                        local_anomaly = float(np.mean(patch))

                # Calculate severity using multi-factor risk engine
                severity_meta = SeverityEngine.calculate_defect_severity(
                    class_name=class_name,
                    bbox=bbox,
                    img_shape=(h, w),
                    confidence=box_conf,
                    local_anomaly_score=local_anomaly
                )

                defects.append({
                    "id": f"defect_{i + 1}",
                    "class_name": class_name,
                    "confidence": round(box_conf, 4),
                    "bbox": bbox,
                    "normalized_bbox": [
                        round((x1 + x2) / (2.0 * w), 4),
                        round((y1 + y2) / (2.0 * h), 4),
                        round((x2 - x1) / float(w), 4),
                        round((y2 - y1) / float(h), 4)
                    ],
                    **severity_meta
                })

        return {
            "defects": defects,
            "total_defects": len(defects),
            "inference_time_ms": inference_time_ms,
            "image_dimensions": {"width": w, "height": h}
        }
