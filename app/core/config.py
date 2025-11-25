from functools import lru_cache
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # App
    APP_NAME: str = "MUSE-Vision"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://localhost:5432/muse_vision"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Vector DB (Milvus)
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION: str = "muse_images"

    # Models
    YOLO_MODEL: str = "yolov8x.pt"
    FACE_DETECTION_MODEL: str = "retinaface_resnet50"
    FACE_RECOGNITION_MODEL: str = "arcface_r100"
    CLIP_MODEL: str = "ViT-L/14"

    # Detection Settings
    DETECTION_CONFIDENCE: float = 0.5
    FACE_SIMILARITY_THRESHOLD: float = 0.6
    MAX_FACES_PER_IMAGE: int = 100

    # Storage
    UPLOAD_DIR: str = "/tmp/muse_vision/uploads"
    MODEL_DIR: str = "models"

    # GPU
    DEVICE: str = "cuda"  # cuda or cpu
    CUDA_VISIBLE_DEVICES: str = "0"

    # Limits
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    MAX_BATCH_SIZE: int = 32


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
