"""
VisionGuard AI - FastAPI Inference and Management API Endpoints
"""

import io
import time
import json
import base64
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, Query, HTTPException, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.database import crud
from app.core.yolo_detector import YOLODefectDetector
from app.core.anomaly_detector import IndustrialAnomalyDetector
from app.core.severity_engine import SeverityEngine
from app.core.heatmap_generator import HeatmapGenerator
from app.core.video_processor import InspectionStreamProcessor

router = APIRouter()

# Global Model Singletons
yolo_engine: Optional[YOLODefectDetector] = None
anomaly_engine: Optional[IndustrialAnomalyDetector] = None
stream_processor: Optional[InspectionStreamProcessor] = None


def get_cv_engines():
    global yolo_engine, anomaly_engine, stream_processor
    if yolo_engine is None:
        yolo_engine = YOLODefectDetector()
    if anomaly_engine is None:
        anomaly_engine = IndustrialAnomalyDetector()
    if stream_processor is None:
        stream_processor = InspectionStreamProcessor(yolo_engine, anomaly_engine)
    return yolo_engine, anomaly_engine, stream_processor


# Pydantic Request Models
class StreamFramePayload(BaseModel):
    image_base64: str
    conf_threshold: float = 0.25
    enable_anomaly: bool = True
    render_mode: str = "annotated"


@router.get("/health")
def health_check():
    """Health check endpoint showing active engines and hardware info."""
    yolo, anomaly, _ = get_cv_engines()
    return {
        "status": "online",
        "service": "VisionGuard AI Industrial Quality Platform",
        "device": anomaly.device,
        "yolo_model": yolo.model_path,
        "anomaly_model": anomaly.model_path,
        "classes": yolo.classes
    }


@router.post("/inspect/image")
async def inspect_single_image(
    file: UploadFile = File(...),
    product_type: str = Query("industrial_substrate"),
    conf_threshold: float = Query(0.25),
    alpha: float = Query(0.45),
    db: Session = Depends(get_db)
):
    """
    Main image inspection endpoint.
    Performs simultaneous YOLO defect localization & PatchCore anomaly detection.
    """
    yolo, anomaly, _ = get_cv_engines()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image payload.")

    h, w = image_bgr.shape[:2]
    t0 = time.perf_counter()

    # 1. Anomaly Detection (Zero-shot / Unseen defect detection)
    anomaly_result = anomaly.inspect(image_bgr)
    anomaly_score = anomaly_result["anomaly_score"]
    anomaly_map = anomaly_result["heatmap"]
    anomaly_latency = anomaly_result["latency_ms"]

    # 2. YOLO Defect Localization
    yolo_result = yolo.detect(
        image_bgr=image_bgr,
        conf_threshold=conf_threshold,
        local_anomaly_map=anomaly_map
    )
    defects = yolo_result["defects"]
    yolo_latency = yolo_result["inference_time_ms"]

    # 3. Defect Severity & Overall Quality Verdict
    verdict = SeverityEngine.evaluate_inspection_verdict(
        defects=defects,
        global_anomaly_score=anomaly_score
    )

    # 4. Generate Visual Heatmaps and Annotations
    annotated_bgr = HeatmapGenerator.draw_annotations(
        image_bgr=image_bgr,
        defects=defects,
        draw_contours=True,
        heatmap_norm=anomaly_map
    )

    heatmap_overlay_bgr = HeatmapGenerator.generate_heatmap_overlay(
        image_bgr=image_bgr,
        heatmap_norm=anomaly_map,
        alpha=alpha,
        colormap=cv2.COLORMAP_JET
    )

    side_by_side_bgr = HeatmapGenerator.create_side_by_side(
        original_bgr=image_bgr,
        annotated_bgr=annotated_bgr,
        heatmap_bgr=heatmap_overlay_bgr
    )

    t1 = time.perf_counter()
    total_latency_ms = round((t1 - t0) * 1000.0, 2)

    # Encode images to Base64 for instant UI display
    orig_b64 = HeatmapGenerator.encode_image_to_base64(image_bgr, quality=85)
    annotated_b64 = HeatmapGenerator.encode_image_to_base64(annotated_bgr, quality=88)
    heatmap_b64 = HeatmapGenerator.encode_image_to_base64(heatmap_overlay_bgr, quality=85)
    side_by_side_b64 = HeatmapGenerator.encode_image_to_base64(side_by_side_bgr, quality=85)

    # Persist record in database
    rec = crud.create_inspection(
        db=db,
        product_type=product_type,
        verdict_meta=verdict,
        defects=defects,
        latency_ms=total_latency_ms,
        image_name=file.filename or "uploaded.jpg",
        thumbnail_base64=f"data:image/jpeg;base64,{annotated_b64}"
    )

    return {
        "inspection_id": rec.id,
        "timestamp": rec.timestamp.isoformat(),
        "product_type": product_type,
        "verdict": verdict,
        "defects": defects,
        "anomaly": {
            "score": anomaly_score,
            "is_anomalous": anomaly_result["is_anomalous"],
            "threshold": anomaly_result["threshold"]
        },
        "latency": {
            "total_ms": total_latency_ms,
            "yolo_ms": yolo_latency,
            "anomaly_ms": anomaly_latency
        },
        "images": {
            "original": f"data:image/jpeg;base64,{orig_b64}",
            "annotated": f"data:image/jpeg;base64,{annotated_b64}",
            "heatmap": f"data:image/jpeg;base64,{heatmap_b64}",
            "side_by_side": f"data:image/jpeg;base64,{side_by_side_b64}"
        }
    }


@router.post("/inspect/stream-frame")
def inspect_stream_frame(payload: StreamFramePayload):
    """
    High-throughput endpoint for live camera frame stream or video feed.
    """
    _, _, stream = get_cv_engines()

    raw_data = payload.image_base64
    if "," in raw_data:
        raw_data = raw_data.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise ValueError("Corrupt frame decode.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame decode error: {e}")

    result = stream.process_frame(
        frame_bgr=frame_bgr,
        enable_anomaly=payload.enable_anomaly,
        conf_threshold=payload.conf_threshold,
        render_mode=payload.render_mode
    )
    return result


@router.get("/samples")
def get_sample_test_images():
    """
    Returns pre-loaded sample images across multiple product substrates & defects
    for immediate 1-click evaluation in the UI.
    """
    samples_dir = Path(__file__).parent.parent / "dataset" / "data_root" / "test" / "images"
    if not samples_dir.exists():
        return {"samples": []}

    samples = []
    for p in list(samples_dir.glob("*.jpg"))[:16]:
        # Filename pattern: test_0001_metal_sheet_scratch.jpg
        parts = p.stem.split("_")
        defect_class = parts[-1] if len(parts) >= 4 else "unknown"
        product = parts[-2] if len(parts) >= 4 else "substrate"

        # Read small thumbnail
        img = cv2.imread(str(p))
        if img is not None:
            thumb = cv2.resize(img, (120, 120))
            thumb_b64 = HeatmapGenerator.encode_image_to_base64(thumb, quality=70)
            full_b64 = HeatmapGenerator.encode_image_to_base64(img, quality=85)
            samples.append({
                "filename": p.name,
                "product_type": product,
                "expected_class": defect_class,
                "is_good": defect_class == "good",
                "thumbnail_url": f"data:image/jpeg;base64,{thumb_b64}",
                "full_image_url": f"data:image/jpeg;base64,{full_b64}"
            })

    return {"samples": samples}


@router.get("/analytics")
def get_analytics(db: Session = Depends(get_db)):
    """Executive summary metrics and interactive charts data."""
    return crud.get_analytics_metrics(db)


@router.get("/history")
def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    verdict: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Retrieves paginated inspection log history."""
    records = crud.get_inspections(db, limit=limit, offset=offset, verdict=verdict)
    total_count = db.query(crud.InspectionRecord).count()
    return {
        "total": total_count,
        "items": [r.to_dict() for r in records]
    }


@router.delete("/history/{record_id}")
def delete_history_record(record_id: int, db: Session = Depends(get_db)):
    """Deletes a single inspection record."""
    rec = db.query(crud.InspectionRecord).filter(crud.InspectionRecord.id == record_id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Record not found.")
    db.delete(rec)
    db.commit()
    return {"message": f"Record {record_id} deleted."}


@router.post("/history/clear")
def clear_all_history(db: Session = Depends(get_db)):
    """Clears all historical inspection records."""
    db.query(crud.DefectItem).delete()
    db.query(crud.InspectionRecord).delete()
    db.commit()
    return {"message": "All inspection records cleared."}


@router.get("/export/csv")
def export_history_csv(db: Session = Depends(get_db)):
    """Exports inspection history as CSV."""
    records = db.query(crud.InspectionRecord).order_by(crud.desc(crud.InspectionRecord.timestamp)).all()
    output = io.StringIO()
    output.write("ID,Timestamp,Product,Verdict,Defects,Critical,Medium,Low,AnomalyScore,Latency_ms\n")
    for r in records:
        ts = r.timestamp.strftime("%Y-%m-%d %H:%M:%S") if r.timestamp else ""
        output.write(f"{r.id},{ts},{r.product_type},{r.verdict},{r.total_defects},{r.critical_defects},{r.medium_defects},{r.low_defects},{r.global_anomaly_score:.4f},{r.latency_ms:.2f}\n")

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=visionguard_inspection_report.csv"}
    )


@router.get("/model/metrics")
def get_model_evaluation_metrics():
    """
    Returns actual documented model evaluation metrics from real training runs.
    """
    storage = Path(__file__).parent.parent / "models_storage"
    yolo_file = storage / "yolo_metrics.json"
    anomaly_file = storage / "anomaly_metrics.json"

    yolo_data = {}
    if yolo_file.exists():
        with open(yolo_file, "r", encoding="utf-8") as f:
            yolo_data = json.load(f)

    anomaly_data = {}
    if anomaly_file.exists():
        with open(anomaly_file, "r", encoding="utf-8") as f:
            anomaly_data = json.load(f)

    return {
        "yolo_detector": yolo_data,
        "anomaly_detector": anomaly_data
    }
