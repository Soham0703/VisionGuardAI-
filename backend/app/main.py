"""
VisionGuard AI - Real-Time Industrial Defect Detection & Quality Inspection Platform
FastAPI Application Entry Point
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.database.database import init_db
from app.config import CORS_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database on startup
    print("Initializing VisionGuard AI Database...")
    init_db()
    yield
    print("Shutting down VisionGuard AI Services...")


app = FastAPI(
    title="VisionGuard AI",
    description="Real-Time Industrial Defect Detection & Quality Inspection Platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router, prefix="/api")

# Serve compiled frontend static files if available
FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/")
    def root():
        return {
            "platform": "VisionGuard AI",
            "tagline": "Real-Time Industrial Defect Detection & Quality Inspection",
            "version": "1.0.0",
            "docs": "/docs",
            "api_status": "/api/health"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
