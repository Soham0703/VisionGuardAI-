"""
VisionGuard AI - Core Anomaly Inference Module
Loads fitted PatchCore feature memory bank and generates calibrated anomaly scores & heatmaps.
"""

import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from app.training.train_anomaly import PatchFeatureExtractor


class IndustrialAnomalyDetector:
    """
    Production anomaly detector for detecting zero-shot / unseen defects.
    """
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model_path is None:
            default_p = Path(__file__).parent.parent / "models_storage" / "anomaly_model.pt"
            model_path = str(default_p)

        self.model_path = model_path
        self.img_size = 256
        self.threshold = 0.45
        self.memory_bank: Optional[torch.Tensor] = None

        print(f"Initializing feature extractor on {self.device}...")
        self.extractor = PatchFeatureExtractor(device=self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if Path(model_path).exists():
            self._load(model_path)
        else:
            print(f"Warning: anomaly model not found at {model_path}. Anomaly inference will use fallback.")

    def _load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.memory_bank = state["memory_bank"].to(self.device)
        self.threshold = state.get("threshold", 0.45)
        self.img_size = state.get("img_size", 256)
        print(f"Loaded Anomaly Feature Memory Bank: {self.memory_bank.shape[0]} embeddings.")

    def inspect(self, image_bgr: np.ndarray) -> Dict:
        """
        Runs anomaly detection on image, returning anomaly score, binary decision,
        normalized heatmap (0 to 1), and execution latency.
        """
        h_orig, w_orig = image_bgr.shape[:2]
        t0 = time.perf_counter()

        if self.memory_bank is None:
            # Fallback if model weights not yet loaded
            heatmap = np.zeros((h_orig, w_orig), dtype=np.float32)
            return {
                "anomaly_score": 0.05,
                "is_anomalous": False,
                "threshold": self.threshold,
                "heatmap": heatmap,
                "latency_ms": 1.0
            }

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        t_img = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feats = self.extractor(t_img)
            B, C, H, W = feats.shape
            patches = feats.permute(0, 2, 3, 1).reshape(-1, C)
            patches = F.normalize(patches, p=2, dim=-1)

            # Cosine similarity to nominal feature memory bank
            similarity = torch.matmul(patches, self.memory_bank.t())
            max_sim, _ = similarity.max(dim=1)
            distances = 1.0 - max_sim  # (N_patches,)

            # Spatial anomaly map
            dist_map = distances.reshape(H, W).cpu().numpy()
            dist_map = cv2.GaussianBlur(dist_map, (3, 3), 0)

            # Top 5% anomaly patches contribute to overall image score
            k = max(1, int(len(distances) * 0.05))
            top_k_dists, _ = torch.topk(distances, k)
            raw_score = float(top_k_dists.mean().cpu().numpy())

            # Calibrated probability
            calibrated_score = float(np.clip((raw_score - 0.12) / 0.25, 0.0, 1.0))

            # Resize to original resolution
            heatmap = cv2.resize(dist_map, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
            h_min, h_max = heatmap.min(), heatmap.max()
            if h_max > h_min:
                heatmap = (heatmap - h_min) / (h_max - h_min)
            else:
                heatmap = np.zeros_like(heatmap)

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000.0, 2)

        return {
            "anomaly_score": round(calibrated_score, 4),
            "is_anomalous": bool(calibrated_score >= self.threshold),
            "threshold": round(self.threshold, 4),
            "heatmap": heatmap,
            "latency_ms": latency_ms
        }
