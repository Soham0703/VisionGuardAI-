"""
VisionGuard AI - Industrial Defect Dataset Handler & Preprocessor
Supports MVTec AD formats, industrial defect categories, data augmentation,
and dataset generation with exact pixel masks and YOLO bounding boxes.
"""

import os
import json
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

# Industrial Defect Taxonomy
DEFECT_CLASSES = [
    "scratch",
    "crack",
    "hole",
    "surface_abnormality",
    "contamination",
    "deformation"
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(DEFECT_CLASSES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(DEFECT_CLASSES)}

PRODUCT_CATEGORIES = [
    "metal_sheet",
    "bottle",
    "capsule",
    "pcb",
    "ceramic_tile"
]


class IndustrialDatasetGenerator:
    """
    Generates realistic industrial product substrates with precise defects,
    bounding box labels (YOLO format), and segmentation masks.
    """
    def __init__(self, img_size: int = 512, seed: int = 42):
        self.img_size = img_size
        random.seed(seed)
        np.random.seed(seed)

    def _generate_substrate(self, product_type: str) -> np.ndarray:
        """Generates realistic industrial substrate textures."""
        size = self.img_size
        img = np.zeros((size, size, 3), dtype=np.uint8)

        if product_type == "metal_sheet":
            # Machined/brushed steel with directional grain
            base_val = np.random.randint(140, 180)
            noise = np.random.normal(0, 12, (size, size)).astype(np.float32)
            grain = np.tile(np.random.normal(0, 8, (1, size)), (size, 1))
            channel = np.clip(base_val + noise * 0.4 + grain * 0.6, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
            # Add subtle horizontal machining lines
            for _ in range(size // 8):
                y = np.random.randint(0, size)
                val_shift = np.random.randint(-15, 15)
                cv2.line(img, (0, y), (size, y), (int(base_val + val_shift),) * 3, 1)

        elif product_type == "ceramic_tile":
            # Matte ceramic with fine stippling and gradient
            base_color = np.array([210, 205, 195], dtype=np.float32)
            noise = np.random.normal(0, 10, (size, size, 3)).astype(np.float32)
            img = np.clip(base_color + noise, 0, 255).astype(np.uint8)
            # Edge border bevel
            cv2.rectangle(img, (10, 10), (size - 10, size - 10), (180, 175, 165), 2)

        elif product_type == "pcb":
            # Circuit board: green or dark blue substrate with copper/gold traces
            bg_green = np.array([30, 90, 45], dtype=np.uint8)
            img[:] = bg_green
            # Add copper traces
            trace_color = (60, 180, 140)
            pad_color = (80, 210, 240) # Golden solder pads
            for _ in range(15):
                pt1 = (np.random.randint(30, size - 30), np.random.randint(30, size - 30))
                length = np.random.randint(50, 200)
                horizontal = np.random.rand() > 0.5
                pt2 = (pt1[0] + length, pt1[1]) if horizontal else (pt1[0], pt1[1] + length)
                pt2 = (min(pt2[0], size - 30), min(pt2[1], size - 30))
                cv2.line(img, pt1, pt2, trace_color, 2)
                cv2.circle(img, pt1, 4, pad_color, -1)
                cv2.circle(img, pt2, 4, pad_color, -1)

        elif product_type == "bottle":
            # Glass container surface with specular gradient & reflections
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            xx, yy = np.meshgrid(x, y)
            cylindrical = 1.0 - 0.4 * (xx ** 2)
            base_glass = (cylindrical * 190).astype(np.uint8)
            img = cv2.cvtColor(base_glass, cv2.COLOR_GRAY2BGR)
            # Specular highlight streak
            spec_x = int(size * 0.35)
            cv2.line(img, (spec_x, 0), (spec_x, size), (240, 240, 240), 4)

        elif product_type == "capsule":
            # Dual-color gelatin capsule
            img[:] = (230, 230, 230)
            cap_w, cap_h = int(size * 0.45), int(size * 0.75)
            cx, cy = size // 2, size // 2
            x1, y1 = cx - cap_w // 2, cy - cap_h // 2
            x2, y2 = cx + cap_w // 2, cy + cap_h // 2
            # Half red, half white
            cv2.rectangle(img, (x1, y1), (x2, cy), (50, 50, 220), -1) # Red top
            cv2.rectangle(img, (x1, cy), (x2, y2), (240, 240, 240), -1) # White bottom
            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 2)

        return img

    def _inject_scratch(self, img: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Injects a linear or curvilinear scratch."""
        size = self.img_size
        x1 = np.random.randint(int(size * 0.15), int(size * 0.75))
        y1 = np.random.randint(int(size * 0.15), int(size * 0.75))
        length = np.random.randint(40, 140)
        angle = np.random.uniform(0, 2 * math.pi)
        x2 = int(np.clip(x1 + length * math.cos(angle), 10, size - 10))
        y2 = int(np.clip(y1 + length * math.sin(angle), 10, size - 10))

        # Control point for bezier curve
        cx = int((x1 + x2) / 2 + np.random.randint(-20, 20))
        cy = int((y1 + y2) / 2 + np.random.randint(-20, 20))

        pts = []
        for t in np.linspace(0, 1, 30):
            bx = int((1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2)
            by = int((1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2)
            pts.append((bx, by))

        # Draw on image and mask
        scratch_color = (255, 255, 255) if np.random.rand() > 0.4 else (20, 20, 20)
        thickness = np.random.randint(1, 3)
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], scratch_color, thickness)
            cv2.line(mask, pts[i], pts[i + 1], 255, thickness + 2)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs) - 5, min(ys) - 5, max(xs) + 5, max(ys) + 5

    def _inject_crack(self, img: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Injects a fractured, branching jagged crack."""
        size = self.img_size
        start_x = np.random.randint(int(size * 0.2), int(size * 0.7))
        start_y = np.random.randint(int(size * 0.2), int(size * 0.7))
        segments = np.random.randint(5, 12)
        curr = (start_x, start_y)
        all_pts = [curr]

        for _ in range(segments):
            dx = np.random.randint(-25, 25)
            dy = np.random.randint(-25, 25)
            nxt = (int(np.clip(curr[0] + dx, 5, size - 5)), int(np.clip(curr[1] + dy, 5, size - 5)))
            cv2.line(img, curr, nxt, (15, 15, 15), 2)
            cv2.line(mask, curr, nxt, 255, 3)
            all_pts.append(nxt)
            # 30% chance of small branch
            if np.random.rand() < 0.35:
                branch = (int(np.clip(nxt[0] + np.random.randint(-15, 15), 5, size - 5)),
                          int(np.clip(nxt[1] + np.random.randint(-15, 15), 5, size - 5)))
                cv2.line(img, nxt, branch, (25, 25, 25), 1)
                cv2.line(mask, nxt, branch, 255, 2)
                all_pts.append(branch)
            curr = nxt

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        return min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4

    def _inject_hole(self, img: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Injects a puncture hole/void with rim edge shadow."""
        size = self.img_size
        cx = np.random.randint(int(size * 0.25), int(size * 0.75))
        cy = np.random.randint(int(size * 0.25), int(size * 0.75))
        rx = np.random.randint(12, 35)
        ry = int(rx * np.random.uniform(0.7, 1.3))
        angle = np.random.randint(0, 180)

        # Draw void (dark center, highlighted edge)
        cv2.ellipse(img, (cx, cy), (rx + 2, ry + 2), angle, 0, 360, (230, 230, 230), 2)
        cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, (10, 10, 10), -1)
        cv2.ellipse(mask, (cx, cy), (rx + 2, ry + 2), angle, 0, 360, 255, -1)

        max_r = max(rx, ry) + 4
        return cx - max_r, cy - max_r, cx + max_r, cy + max_r

    def _inject_surface_abnormality(self, img: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Injects chemical discoloration, oxidation or texture roughness."""
        size = self.img_size
        cx = np.random.randint(int(size * 0.2), int(size * 0.8))
        cy = np.random.randint(int(size * 0.2), int(size * 0.8))
        radius = np.random.randint(25, 60)

        # Blotchy irregular mask
        sub_mask = np.zeros((size, size), dtype=np.uint8)
        num_blobs = np.random.randint(3, 7)
        for _ in range(num_blobs):
            bx = cx + np.random.randint(-radius // 2, radius // 2)
            by = cy + np.random.randint(-radius // 2, radius // 2)
            br = np.random.randint(radius // 3, radius)
            cv2.circle(sub_mask, (bx, by), br, 255, -1)

        sub_mask = cv2.GaussianBlur(sub_mask, (21, 21), 0)
        norm_sub = sub_mask.astype(np.float32) / 255.0

        # Discoloration: oxidize towards brownish-orange or dark blotch
        discolor_color = np.array([40, 80, 160], dtype=np.float32) # rust/oxidation
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * (1.0 - norm_sub * 0.7) + discolor_color[c] * norm_sub * 0.7).astype(np.uint8)

        mask[sub_mask > 40] = 255
        return cx - radius - 10, cy - radius - 10, cx + radius + 10, cy + radius + 10

    def _inject_contamination(self, img: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Injects oil spatter or foreign particle cluster."""
        size = self.img_size
        cx = np.random.randint(int(size * 0.2), int(size * 0.8))
        cy = np.random.randint(int(size * 0.2), int(size * 0.8))
        num_droplets = np.random.randint(6, 18)
        radius_spread = np.random.randint(20, 50)
        pts = []

        oil_color = (20, 45, 60) # dark oil
        for _ in range(num_droplets):
            ox = int(np.clip(cx + np.random.normal(0, radius_spread // 2), 5, size - 5))
            oy = int(np.clip(cy + np.random.normal(0, radius_spread // 2), 5, size - 5))
            orad = np.random.randint(2, 8)
            cv2.circle(img, (ox, oy), orad, oil_color, -1)
            cv2.circle(mask, (ox, oy), orad, 255, -1)
            pts.append((ox - orad, oy - orad))
            pts.append((ox + orad, oy + orad))

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4

    def _inject_deformation(self, img: np.ndarray, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Injects physical dent / warp / boundary deformation."""
        size = self.img_size
        cx = np.random.randint(int(size * 0.3), int(size * 0.7))
        cy = np.random.randint(int(size * 0.3), int(size * 0.7))
        radius = np.random.randint(30, 65)

        # Create optical warp using mesh displacement
        y_indices, x_indices = np.indices((size, size))
        dist = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
        dent_mask = dist < radius

        # Shadow and highlight ring simulating depth
        dent_highlight = np.clip(img.astype(np.float32) + 40, 0, 255).astype(np.uint8)
        dent_shadow = np.clip(img.astype(np.float32) - 50, 0, 255).astype(np.uint8)

        # Upper crescent gets highlight, lower crescent gets shadow
        crescent_hi = dent_mask & (y_indices < cy)
        crescent_lo = dent_mask & (y_indices >= cy)

        img[crescent_hi] = dent_highlight[crescent_hi]
        img[crescent_lo] = dent_shadow[crescent_lo]
        mask[dent_mask] = 255

        return cx - radius - 5, cy - radius - 5, cx + radius + 5, cy + radius + 5

    def generate_sample(
        self,
        product_type: Optional[str] = None,
        defect_class: Optional[str] = None
    ) -> Dict:
        """
        Generates a single industrial sample. If defect_class is None, produces
        a nominal (defect-free) sample.
        """
        if product_type is None:
            product_type = random.choice(PRODUCT_CATEGORIES)

        img = self._generate_substrate(product_type)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        bboxes = []

        is_defective = defect_class is not None
        if is_defective:
            if defect_class == "scratch":
                bbox = self._inject_scratch(img, mask)
            elif defect_class == "crack":
                bbox = self._inject_crack(img, mask)
            elif defect_class == "hole":
                bbox = self._inject_hole(img, mask)
            elif defect_class == "surface_abnormality":
                bbox = self._inject_surface_abnormality(img, mask)
            elif defect_class == "contamination":
                bbox = self._inject_contamination(img, mask)
            elif defect_class == "deformation":
                bbox = self._inject_deformation(img, mask)
            else:
                bbox = self._inject_scratch(img, mask)

            # Clamp bounding box coordinates
            x1 = max(0, min(self.img_size - 1, bbox[0]))
            y1 = max(0, min(self.img_size - 1, bbox[1]))
            x2 = max(x1 + 4, min(self.img_size, bbox[2]))
            y2 = max(y1 + 4, min(self.img_size, bbox[3]))

            # Normalize to YOLO format (class, x_center, y_center, width, height)
            w_norm = (x2 - x1) / self.img_size
            h_norm = (y2 - y1) / self.img_size
            xc_norm = (x1 + x2) / (2.0 * self.img_size)
            yc_norm = (y1 + y2) / (2.0 * self.img_size)
            class_id = CLASS_TO_ID[defect_class]

            bboxes.append({
                "class_id": class_id,
                "class_name": defect_class,
                "yolo_bbox": [xc_norm, yc_norm, w_norm, h_norm],
                "pixel_bbox": [x1, y1, x2, y2]
            })

        # Apply realistic sensor noise & lighting variance
        img = self.apply_photometric_augmentation(img)

        return {
            "image": img,
            "mask": mask,
            "product_type": product_type,
            "is_defective": is_defective,
            "defect_class": defect_class if is_defective else "good",
            "bboxes": bboxes
        }

    def apply_photometric_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Applies subtle factory lighting and optical variance."""
        # Brightness adjustment
        factor = np.random.uniform(0.92, 1.08)
        img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Subtle Gaussian noise (sensor grain)
        noise = np.random.normal(0, 2.5, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return img


def build_full_dataset(
    output_root: str,
    train_count: int = 120,
    val_count: int = 40,
    test_count: int = 40,
    img_size: int = 512
) -> Dict[str, any]:
    """
    Builds and exports a complete industrial defect dataset with train/val/test splits,
    YOLO annotations, MVTec AD anomaly structure, and dataset metadata.
    """
    generator = IndustrialDatasetGenerator(img_size=img_size)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_count,
        "val": val_count,
        "test": test_count
    }

    stats = {
        "total_samples": train_count + val_count + test_count,
        "classes": DEFECT_CLASSES,
        "splits": {},
        "category_counts": {}
    }

    # Setup directories
    for split in splits:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        mask_dir = root / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

    # MVTec structure for Anomaly Detection (train/good, test/good, test/<defect>)
    mvtec_dir = root / "mvtec_structure"
    (mvtec_dir / "train" / "good").mkdir(parents=True, exist_ok=True)
    (mvtec_dir / "test" / "good").mkdir(parents=True, exist_ok=True)
    for c in DEFECT_CLASSES:
        (mvtec_dir / "test" / c).mkdir(parents=True, exist_ok=True)

    for split, count in splits.items():
        stats["splits"][split] = {"good": 0, "defective": 0, "by_class": {c: 0 for c in DEFECT_CLASSES}}

        for i in range(count):
            # In training, 40% good samples (critical for anomaly detection & false positive suppression)
            # In test, 25% good, 75% defects
            good_prob = 0.40 if split == "train" else 0.25
            is_good = random.random() < good_prob

            product = random.choice(PRODUCT_CATEGORIES)
            defect_type = None if is_good else random.choice(DEFECT_CLASSES)

            sample = generator.generate_sample(product_type=product, defect_class=defect_type)
            img = sample["image"]
            mask = sample["mask"]
            bboxes = sample["bboxes"]

            sample_name = f"{split}_{i:04d}_{product}_{sample['defect_class']}"
            img_path = root / split / "images" / f"{sample_name}.jpg"
            lbl_path = root / split / "labels" / f"{sample_name}.txt"
            mask_path = root / split / "masks" / f"{sample_name}.png"

            # Save image and mask
            cv2.imwrite(str(img_path), img)
            cv2.imwrite(str(mask_path), mask)

            # Write YOLO label format: class xc yc w h
            with open(lbl_path, "w", encoding="utf-8") as f:
                for b in bboxes:
                    xc, yc, w, h = b["yolo_bbox"]
                    f.write(f"{b['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            # Also populate MVTec structure
            if is_good:
                stats["splits"][split]["good"] += 1
                if split == "train":
                    cv2.imwrite(str(mvtec_dir / "train" / "good" / f"{sample_name}.jpg"), img)
                else:
                    cv2.imwrite(str(mvtec_dir / "test" / "good" / f"{sample_name}.jpg"), img)
            else:
                stats["splits"][split]["defective"] += 1
                stats["splits"][split]["by_class"][defect_type] += 1
                stats["category_counts"][defect_type] = stats["category_counts"].get(defect_type, 0) + 1
                cv2.imwrite(str(mvtec_dir / "test" / defect_type / f"{sample_name}.jpg"), img)

    # Write YOLO data.yaml
    yolo_yaml_content = f"""# VisionGuard AI Industrial Defect Dataset
path: {root.resolve().as_posix()}
train: train/images
val: val/images
test: test/images

nc: {len(DEFECT_CLASSES)}
names: {DEFECT_CLASSES}
"""
    with open(root / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yolo_yaml_content)

    # Write metadata manifest
    with open(root / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


if __name__ == "__main__":
    out_dir = "D:/Python Projects/2.VisonGuardAI/backend/app/dataset/data_root"
    print("Generating industrial defect dataset...")
    s = build_full_dataset(out_dir, train_count=100, val_count=30, test_count=30, img_size=512)
    print("Dataset generation complete. Stats:", json.dumps(s, indent=2))
