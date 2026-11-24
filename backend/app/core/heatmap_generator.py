"""
VisionGuard AI - Computer Vision Visual Heatmaps & Annotation Overlays
Generates visual heatmaps highlighting suspicious defect regions,
draws bounding boxes with severity classification, and prepares visual comparisons.
"""

import base64
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np


class HeatmapGenerator:
    """
    Renders visual thermal heatmaps and bounding box annotations on industrial images.
    """

    @staticmethod
    def generate_heatmap_overlay(
        image_bgr: np.ndarray,
        heatmap_norm: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlays a normalized 2D anomaly heatmap (0.0 to 1.0) onto an image.
        """
        h, w = image_bgr.shape[:2]
        if heatmap_norm.shape[:2] != (h, w):
            heatmap_norm = cv2.resize(heatmap_norm, (w, h), interpolation=cv2.INTER_CUBIC)

        # Convert to 8-bit [0, 255]
        heatmap_uint8 = np.clip(heatmap_norm * 255.0, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(heatmap_uint8, colormap)

        # Blend with original image
        blended = cv2.addWeighted(color_map, alpha, image_bgr, 1.0 - alpha, 0)
        return blended

    @staticmethod
    def draw_annotations(
        image_bgr: np.ndarray,
        defects: List[Dict],
        draw_contours: bool = True,
        heatmap_norm: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Draws defect bounding boxes, corner accents, severity-colored tags,
        and optional anomaly contours.
        """
        annotated = image_bgr.copy()
        h, w = annotated.shape[:2]

        # Draw contour highlights around high anomaly zones
        if draw_contours and heatmap_norm is not None:
            if heatmap_norm.shape[:2] != (h, w):
                heatmap_norm = cv2.resize(heatmap_norm, (w, h), interpolation=cv2.INTER_CUBIC)
            thresh = (heatmap_norm > 0.45).astype(np.uint8) * 255
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (0, 140, 255), 1)  # Amber contour

        for d in defects:
            x1, y1, x2, y2 = d["bbox"]
            severity = d.get("severity_level", "MEDIUM")
            class_name = d.get("class_name", "defect")
            conf = d.get("confidence", 0.0)

            # Severity-coded BGR colors
            if severity == "CRITICAL":
                color = (40, 40, 240)    # Red (BGR)
            elif severity == "MEDIUM":
                color = (15, 175, 245)   # Amber (BGR)
            else:
                color = (245, 160, 40)   # Blue (BGR)

            # Primary bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # High-tech corner bracket accents
            line_len = min(12, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + line_len, y1), color, 3)
            cv2.line(annotated, (x1, y1), (x1, y1 + line_len), color, 3)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - line_len, y1), color, 3)
            cv2.line(annotated, (x2, y1), (x2, y1 + line_len), color, 3)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + line_len, y2), color, 3)
            cv2.line(annotated, (x1, y2), (x1, y2 - line_len), color, 3)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - line_len, y2), color, 3)
            cv2.line(annotated, (x2, y2), (x2, y2 - line_len), color, 3)

            # Label text & background chip
            label = f"{class_name.upper()} {conf * 100:.1f}% [{severity}]"
            font_scale = 0.45
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            chip_y1 = max(0, y1 - th - 8)
            chip_y2 = y1
            chip_x2 = min(w, x1 + tw + 10)

            # Semi-transparent dark background for chip
            sub_rect = annotated[chip_y1:chip_y2, x1:chip_x2]
            if sub_rect.shape[0] > 0 and sub_rect.shape[1] > 0:
                dark_chip = np.zeros_like(sub_rect)
                cv2.rectangle(dark_chip, (0, 0), (sub_rect.shape[1], sub_rect.shape[0]), (20, 20, 25), -1)
                annotated[chip_y1:chip_y2, x1:chip_x2] = cv2.addWeighted(sub_rect, 0.25, dark_chip, 0.75, 0)

            cv2.rectangle(annotated, (x1, chip_y1), (chip_x2, chip_y2), color, 1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, chip_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        return annotated

    @staticmethod
    def create_side_by_side(
        original_bgr: np.ndarray,
        annotated_bgr: np.ndarray,
        heatmap_bgr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Creates a side-by-side composite image with inspection banners.
        """
        images = [original_bgr, annotated_bgr]
        if heatmap_bgr is not None:
            images.append(heatmap_bgr)

        # Ensure all same height
        h = max(img.shape[0] for img in images)
        resized_imgs = []
        for img in images:
            if img.shape[0] != h:
                scaled_w = int(img.shape[1] * (h / img.shape[0]))
                resized_imgs.append(cv2.resize(img, (scaled_w, h)))
            else:
                resized_imgs.append(img)

        # Add vertical dividing borders
        divider = np.full((h, 4, 3), 45, dtype=np.uint8)
        composite = resized_imgs[0]
        for nxt in resized_imgs[1:]:
            composite = np.hstack([composite, divider, nxt])

        return composite

    @staticmethod
    def encode_image_to_base64(image_bgr: np.ndarray, quality: int = 88) -> str:
        """Encodes an OpenCV BGR image to a base64 JPEG string."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", image_bgr, encode_params)
        if not success:
            return ""
        return base64.b64encode(buffer).decode("utf-8")
