"""
VisionGuard AI - MVTec AD Handler & Converter
Handles downloading, extracting, and standardizing real MVTec AD datasets,
extracting ground-truth bounding boxes from masks, and formatting for YOLO/Anomaly detection.
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class MVTecADHandler:
    """
    Parser for MVTec Anomaly Detection dataset directory structure:
    <category>/
        train/
            good/
        test/
            good/
            crack/
            scratch/
            ...
        ground_truth/
            crack/
            scratch/
            ...
    """
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def scan_categories(self) -> List[str]:
        """Returns detected MVTec categories."""
        if not self.root_dir.exists():
            return []
        categories = [
            p.name for p in self.root_dir.iterdir()
            if p.is_dir() and (p / "train" / "good").exists()
        ]
        return categories

    def mask_to_bounding_boxes(
        self,
        mask: np.ndarray,
        min_area: int = 16
    ) -> List[Tuple[int, int, int, int]]:
        """
        Converts a binary defect mask into bounding box coordinates (x1, y1, x2, y2).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((x, y, x + w, y + h))
        return boxes

    def export_category_to_yolo(
        self,
        category: str,
        output_dir: str,
        class_mapping: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """
        Extracts test defect images and ground_truth masks, generates YOLO annotations.
        """
        cat_path = self.root_dir / category
        out_path = Path(output_dir)
        out_images = out_path / "images"
        out_labels = out_path / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        if class_mapping is None:
            # Dynamically infer defect types in test folder (excluding 'good')
            defect_types = [
                p.name for p in (cat_path / "test").iterdir()
                if p.is_dir() and p.name != "good"
            ]
            class_mapping = {d: i for i, d in enumerate(defect_types)}

        stats = {c: 0 for c in class_mapping}
        stats["good"] = 0

        # Process defect test samples
        for defect_type, class_id in class_mapping.items():
            test_defect_dir = cat_path / "test" / defect_type
            gt_defect_dir = cat_path / "ground_truth" / defect_type
            if not test_defect_dir.exists():
                continue

            for img_file in test_defect_dir.glob("*.png"):
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                h, w = img.shape[:2]
                mask_file = gt_defect_dir / f"{img_file.stem}_mask.png"

                bboxes = []
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        boxes = self.mask_to_bounding_boxes(mask)
                        for x1, y1, x2, y2 in boxes:
                            xc = (x1 + x2) / (2.0 * w)
                            yc = (y1 + y2) / (2.0 * h)
                            bw = (x2 - x1) / float(w)
                            bh = (y2 - y1) / float(h)
                            bboxes.append((class_id, xc, yc, bw, bh))

                target_name = f"{category}_{defect_type}_{img_file.stem}"
                cv2.imwrite(str(out_images / f"{target_name}.jpg"), img)
                with open(out_labels / f"{target_name}.txt", "w", encoding="utf-8") as lf:
                    for b in bboxes:
                        lf.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                stats[defect_type] += 1

        # Process good samples (background/negative examples)
        test_good_dir = cat_path / "test" / "good"
        if test_good_dir.exists():
            for img_file in list(test_good_dir.glob("*.png"))[:len(class_mapping) * 5]:
                img = cv2.imread(str(img_file))
                if img is not None:
                    target_name = f"{category}_good_{img_file.stem}"
                    cv2.imwrite(str(out_images / f"{target_name}.jpg"), img)
                    # Create empty label file for negative sample
                    open(out_labels / f"{target_name}.txt", "w", encoding="utf-8").close()
                    stats["good"] += 1

        return stats
