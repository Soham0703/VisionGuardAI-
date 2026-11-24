"""
VisionGuard AI - Database Operations (CRUD) & Analytics Aggregation
"""

import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.database.models import InspectionRecord, DefectItem


def create_inspection(
    db: Session,
    product_type: str,
    verdict_meta: Dict,
    defects: List[Dict],
    latency_ms: float,
    image_name: str = "capture.jpg",
    thumbnail_base64: Optional[str] = None
) -> InspectionRecord:
    """Creates and persists an inspection record and its defects."""
    rec = InspectionRecord(
        timestamp=datetime.utcnow(),
        product_type=product_type,
        verdict=verdict_meta["verdict"],
        verdict_reason=verdict_meta.get("status_text", ""),
        total_defects=verdict_meta["total_defects"],
        critical_defects=verdict_meta["critical_defects"],
        medium_defects=verdict_meta["medium_defects"],
        low_defects=verdict_meta["low_defects"],
        global_anomaly_score=verdict_meta.get("global_anomaly_score", 0.0),
        latency_ms=latency_ms,
        image_name=image_name,
        thumbnail_base64=thumbnail_base64
    )
    db.add(rec)
    db.flush()  # Assigns rec.id

    for d in defects:
        defect_item = DefectItem(
            inspection_id=rec.id,
            class_name=d["class_name"],
            confidence=d.get("confidence", 0.0),
            severity_level=d.get("severity_level", "MEDIUM"),
            severity_score=d.get("severity_score", 0.0),
            area_pixels=d.get("area_pixels", 0),
            area_percentage=d.get("area_percentage", 0.0),
            bbox_json=json.dumps(d.get("bbox", []))
        )
        db.add(defect_item)

    db.commit()
    db.refresh(rec)
    return rec


def get_inspections(
    db: Session,
    limit: int = 50,
    offset: int = 0,
    verdict: Optional[str] = None
) -> List[InspectionRecord]:
    """Retrieves paginated inspection records."""
    query = db.query(InspectionRecord)
    if verdict and verdict.upper() != "ALL":
        query = query.filter(InspectionRecord.verdict == verdict.upper())
    return query.order_by(desc(InspectionRecord.timestamp)).offset(offset).limit(limit).all()


def get_analytics_metrics(db: Session) -> Dict:
    """
    Computes real-time executive dashboard KPIs and defect analytics.
    """
    total = db.query(InspectionRecord).count()
    if total == 0:
        return {
            "total_inspected": 0,
            "pass_count": 0,
            "review_count": 0,
            "fail_count": 0,
            "defective_count": 0,
            "defect_rate_percentage": 0.0,
            "yield_rate_percentage": 100.0,
            "avg_latency_ms": 0.0,
            "category_distribution": {},
            "severity_distribution": {"CRITICAL": 0, "MEDIUM": 0, "LOW": 0},
            "recent_verdict_trend": []
        }

    pass_cnt = db.query(InspectionRecord).filter(InspectionRecord.verdict == "PASS").count()
    review_cnt = db.query(InspectionRecord).filter(InspectionRecord.verdict == "REVIEW").count()
    fail_cnt = db.query(InspectionRecord).filter(InspectionRecord.verdict == "FAIL").count()

    defective_cnt = review_cnt + fail_cnt
    defect_rate = round((defective_cnt / total) * 100.0, 2)
    yield_rate = round((pass_cnt / total) * 100.0, 2)

    avg_lat = db.query(func.avg(InspectionRecord.latency_ms)).scalar() or 0.0

    # Defect category distribution
    cat_counts = (
        db.query(DefectItem.class_name, func.count(DefectItem.id))
        .group_by(DefectItem.class_name)
        .all()
    )
    category_dist = {cat: count for cat, count in cat_counts}

    # Defect severity distribution
    sev_counts = (
        db.query(DefectItem.severity_level, func.count(DefectItem.id))
        .group_by(DefectItem.severity_level)
        .all()
    )
    severity_dist = {"CRITICAL": 0, "MEDIUM": 0, "LOW": 0}
    for sev, count in sev_counts:
        if sev in severity_dist:
            severity_dist[sev] = count

    # Recent 10 inspection trend
    recent_trend = []
    recent_records = (
        db.query(InspectionRecord)
        .order_by(desc(InspectionRecord.timestamp))
        .limit(15)
        .all()
    )
    for r in reversed(recent_records):
        recent_trend.append({
            "id": r.id,
            "time": r.timestamp.strftime("%H:%M:%S") if r.timestamp else "",
            "verdict": r.verdict,
            "defects": r.total_defects,
            "anomaly_score": r.global_anomaly_score,
            "latency": r.latency_ms
        })

    return {
        "total_inspected": total,
        "pass_count": pass_cnt,
        "review_count": review_cnt,
        "fail_count": fail_cnt,
        "defective_count": defective_cnt,
        "defect_rate_percentage": defect_rate,
        "yield_rate_percentage": yield_rate,
        "avg_latency_ms": round(float(avg_lat), 1),
        "category_distribution": category_dist,
        "severity_distribution": severity_dist,
        "recent_verdict_trend": recent_trend
    }
