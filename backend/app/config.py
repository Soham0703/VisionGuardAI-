"""
VisionGuard AI - System Configuration & Environment Loader
Loads configuration from .env and environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Search for .env in current directory, backend directory, or project root
CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CURRENT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

for env_path in [BACKEND_DIR / ".env", PROJECT_ROOT / ".env", CURRENT_DIR / ".env"]:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break

BASE_DIR = BACKEND_DIR
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "app" / "models_storage")))
DATASET_DIR = Path(os.getenv("DATASET_DIR", str(BASE_DIR / "app" / "dataset" / "data_root")))

APP_NAME = os.getenv("APP_NAME", "VisionGuard AI")
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 2))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Database URL
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{(BASE_DIR / 'visionguard.db').resolve().as_posix()}")

# Hardware Accelerator
DEVICE = os.getenv("DEVICE", "cpu")

# Model Paths
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(MODELS_DIR / "yolov8_defect.pt"))
ANOMALY_MODEL_PATH = os.getenv("ANOMALY_MODEL_PATH", str(MODELS_DIR / "anomaly_model.pt"))

# Quality Inspection Defaults
DEFAULT_CONF_THRESHOLD = float(os.getenv("DEFAULT_CONF_THRESHOLD", 0.25))
DEFAULT_IOU_THRESHOLD = float(os.getenv("DEFAULT_IOU_THRESHOLD", 0.45))
DEFAULT_HEATMAP_ALPHA = float(os.getenv("DEFAULT_HEATMAP_ALPHA", 0.45))
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", 0.2532))

# CORS Origins
cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000,*")
CORS_ORIGINS = [orig.strip() for orig in cors_raw.split(",") if orig.strip()]
