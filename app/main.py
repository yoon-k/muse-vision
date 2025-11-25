from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.core.config import settings
from app.api.router import api_router

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("Starting MUSE Vision", version=settings.APP_VERSION)

    # Load models on startup
    from app.services.detection.yolo import YOLODetector
    from app.services.recognition.face import FaceRecognizer
    from app.services.search.clip import CLIPSearcher

    app.state.detector = YOLODetector()
    app.state.face_recognizer = FaceRecognizer()
    app.state.searcher = CLIPSearcher()

    yield

    logger.info("Shutting down MUSE Vision")


app = FastAPI(
    title="MUSE Vision",
    description="""
# MUSE Vision - AI-Powered Image Recognition Platform

엔터프라이즈급 컴퓨터 비전 API

## Features
- Object Detection (YOLOv8)
- Face Detection & Recognition (ArcFace)
- Image Similarity Search (CLIP)
- CCTV Stream Analysis
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "MUSE Vision"}


@app.get("/")
async def root():
    return {
        "name": "MUSE Vision",
        "version": settings.APP_VERSION,
        "description": "AI-Powered Image Recognition Platform",
        "docs": "/docs"
    }
