"""
VisionGuard AI - Database Models (SQLAlchemy ORM)
Persists inspection history, defect records, and operational telemetry.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class InspectionRecord(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    product_type = Column(String(64), default="industrial_substrate")
    verdict = Column(String(16), index=True)  # PASS, REVIEW, FAIL
    verdict_reason = Column(String(256))
    total_defects = Column(Integer, default=0)
    critical_defects = Column(Integer, default=0)
    medium_defects = Column(Integer, default=0)
    low_defects = Column(Integer, default=0)
    global_anomaly_score = Column(Float, default=0.0)
    latency_ms = Column(Float, default=0.0)
    image_name = Column(String(128))
    thumbnail_base64 = Column(Text, nullable=True)

    # Relationships
    defects = relationship("DefectItem", back_populates="inspection", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "product_type": self.product_type,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "total_defects": self.total_defects,
            "critical_defects": self.critical_defects,
            "medium_defects": self.medium_defects,
            "low_defects": self.low_defects,
            "global_anomaly_score": self.global_anomaly_score,
            "latency_ms": self.latency_ms,
            "image_name": self.image_name,
            "thumbnail_base64": self.thumbnail_base64,
            "defects": [d.to_dict() for d in self.defects]
        }


class DefectItem(Base):
    __tablename__ = "defect_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id", ondelete="CASCADE"), index=True)
    class_name = Column(String(64), index=True)
    confidence = Column(Float, default=0.0)
    severity_level = Column(String(16), index=True)  # CRITICAL, MEDIUM, LOW
    severity_score = Column(Float, default=0.0)
    area_pixels = Column(Integer, default=0)
    area_percentage = Column(Float, default=0.0)
    bbox_json = Column(String(128))  # "[x1, y1, x2, y2]"

    inspection = relationship("InspectionRecord", back_populates="defects")

    def to_dict(self):
        return {
            "id": self.id,
            "inspection_id": self.inspection_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "severity_level": self.severity_level,
            "severity_score": self.severity_score,
            "area_pixels": self.area_pixels,
            "area_percentage": self.area_percentage,
            "bbox_json": self.bbox_json
        }
