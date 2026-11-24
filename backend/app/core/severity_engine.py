"""
VisionGuard AI - Defect Severity Classification Engine
Implements a multi-factor risk assessment model to classify industrial defects
into LOW, MEDIUM, or CRITICAL severity with pass/review/fail decisions.
"""

from typing import Dict, List, Tuple
import numpy as np

# Defect Class Inherent Criticality Weights (W_c)
# Safety-critical structural voids/fractures are rated highest.
CLASS_CRITICALITY_WEIGHTS = {
    "crack": 1.00,                 # High risk of catastrophic structural failure
    "hole": 0.95,                  # High risk of fluid leakage, pressure drop, puncture
    "deformation": 0.85,           # Mechanical misalignment, dimensional out-of-spec
    "surface_abnormality": 0.65,   # Corrosion, chemical oxidation, degradation
    "contamination": 0.50,         # Foreign matter, particulate / cleaning defect
    "scratch": 0.40                # Surface cosmetic flaw / minor abrasion
}

# Decision Thresholds
CRITICAL_SEVERITY_THRESHOLD = 0.68
MEDIUM_SEVERITY_THRESHOLD = 0.40


class SeverityEngine:
    """
    Computes defect severity and overall quality inspection verdict:
    PASS (Yield approved), REVIEW (Manual verification required), or FAIL (Reject/Scrap).
    """

    @classmethod
    def calculate_defect_severity(
        cls,
        class_name: str,
        bbox: List[int],          # [x1, y1, x2, y2]
        img_shape: Tuple[int, int],  # (H, W)
        confidence: float,        # 0.0 - 1.0
        local_anomaly_score: float = 0.0  # 0.0 - 1.0
    ) -> Dict:
        """
        Calculates individual defect severity:
        Formula:
          Severity = 0.35 * W_c + 0.30 * min(1.0, Area_ratio * 20.0) + 0.25 * S_anomaly + 0.10 * Confidence
        """
        w_c = CLASS_CRITICALITY_WEIGHTS.get(class_name.lower(), 0.50)
        h_img, w_img = img_shape[:2]
        total_pixels = float(h_img * w_img)

        x1, y1, x2, y2 = bbox
        defect_w = max(1, x2 - x1)
        defect_h = max(1, y2 - y1)
        defect_area = float(defect_w * defect_h)
        area_ratio = min(1.0, defect_area / total_pixels)

        # Normalized area factor (scaled so 5% of product area gives 1.0)
        area_factor = min(1.0, area_ratio * 20.0)

        # Composite severity score
        score = (
            0.35 * w_c +
            0.30 * area_factor +
            0.25 * float(np.clip(local_anomaly_score, 0.0, 1.0)) +
            0.10 * float(np.clip(confidence, 0.0, 1.0))
        )
        score = round(float(np.clip(score, 0.0, 1.0)), 4)

        if score >= CRITICAL_SEVERITY_THRESHOLD:
            level = "CRITICAL"
            color_hex = "#EF4444"  # Red
            badge_bg = "bg-red-500/10 text-red-400 border-red-500/30"
        elif score >= MEDIUM_SEVERITY_THRESHOLD:
            level = "MEDIUM"
            color_hex = "#F59E0B"  # Amber
            badge_bg = "bg-amber-500/10 text-amber-400 border-amber-500/30"
        else:
            level = "LOW"
            color_hex = "#3B82F6"  # Blue/Green
            badge_bg = "bg-blue-500/10 text-blue-400 border-blue-500/30"

        return {
            "class_name": class_name,
            "severity_level": level,
            "severity_score": score,
            "area_pixels": int(defect_area),
            "area_percentage": round(area_ratio * 100.0, 2),
            "confidence": round(confidence, 4),
            "color_hex": color_hex,
            "badge_class": badge_bg
        }

    @classmethod
    def evaluate_inspection_verdict(
        cls,
        defects: List[Dict],
        global_anomaly_score: float,
        anomaly_threshold: float = 0.45
    ) -> Dict:
        """
        Evaluates overall manufacturing quality decision: PASS, REVIEW, or FAIL.
        """
        has_critical = any(d["severity_level"] == "CRITICAL" for d in defects)
        medium_count = sum(1 for d in defects if d["severity_level"] == "MEDIUM")
        high_anomaly = global_anomaly_score >= (anomaly_threshold * 1.35)

        if has_critical or medium_count >= 2 or high_anomaly:
            decision = "FAIL"
            decision_color = "#EF4444"
            status_text = "Rejected – Defect Criticality Exceeds Quality Tolerance"
        elif medium_count == 1 or global_anomaly_score >= anomaly_threshold:
            decision = "REVIEW"
            decision_color = "#F59E0B"
            status_text = "Quarantined – Secondary Operator Review Required"
        else:
            decision = "PASS"
            decision_color = "#10B981"
            status_text = "Approved – Nominal Surface Quality Standards Met"

        return {
            "verdict": decision,
            "verdict_color": decision_color,
            "status_text": status_text,
            "total_defects": len(defects),
            "critical_defects": sum(1 for d in defects if d["severity_level"] == "CRITICAL"),
            "medium_defects": medium_count,
            "low_defects": sum(1 for d in defects if d["severity_level"] == "LOW"),
            "global_anomaly_score": round(global_anomaly_score, 4)
        }
