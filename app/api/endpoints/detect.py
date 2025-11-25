from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import time

router = APIRouter()


class DetectionResult(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]
    attributes: Optional[dict] = None


class DetectionResponse(BaseModel):
    detections: List[DetectionResult]
    image_size: List[int]
    processing_time_ms: float


@router.post("", response_model=DetectionResponse)
async def detect_objects(
    request: Request,
    image: UploadFile = File(...),
    confidence: float = Form(0.5),
    classes: Optional[str] = Form(None)
):
    """Detect objects in an image.

    - **image**: Image file (JPEG, PNG)
    - **confidence**: Minimum confidence threshold (0-1)
    - **classes**: Comma-separated class names to filter (e.g., "person,dog,car")
    """
    start_time = time.time()

    # Read and validate image
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
        img_array = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Get detector from app state
    detector = request.app.state.detector

    # Parse class filter
    class_filter = None
    if classes:
        class_names = [c.strip().lower() for c in classes.split(",")]
        name_to_id = {v.lower(): k for k, v in detector.get_class_names().items()}
        class_filter = [name_to_id[name] for name in class_names if name in name_to_id]

    # Run detection
    detections = detector.detect(
        img_array,
        confidence=confidence,
        classes=class_filter
    )

    processing_time = (time.time() - start_time) * 1000

    return DetectionResponse(
        detections=[
            DetectionResult(
                class_id=d["class_id"],
                class_name=d["class"],
                confidence=d["confidence"],
                bbox=d["bbox"]
            )
            for d in detections
        ],
        image_size=[pil_image.width, pil_image.height],
        processing_time_ms=round(processing_time, 2)
    )


@router.post("/persons")
async def detect_persons(
    request: Request,
    image: UploadFile = File(...),
    confidence: float = Form(0.5)
):
    """Detect only persons in an image."""
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    img_array = np.array(pil_image)

    detector = request.app.state.detector
    detections = detector.detect_persons(img_array, confidence=confidence)

    return {
        "person_count": len(detections),
        "detections": detections
    }


@router.post("/animals")
async def detect_animals(
    request: Request,
    image: UploadFile = File(...),
    confidence: float = Form(0.5)
):
    """Detect animals in an image (dogs, cats, birds, etc.)."""
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    img_array = np.array(pil_image)

    detector = request.app.state.detector
    detections = detector.detect_animals(img_array, confidence=confidence)

    return {
        "animal_count": len(detections),
        "detections": detections
    }


@router.get("/classes")
async def get_classes(request: Request):
    """Get list of detectable object classes."""
    detector = request.app.state.detector
    return {"classes": detector.get_class_names()}
