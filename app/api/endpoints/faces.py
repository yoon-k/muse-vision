from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import uuid
import time

router = APIRouter()


class FaceDetection(BaseModel):
    bbox: List[int]
    confidence: float
    age: Optional[int] = None
    gender: Optional[str] = None
    match: Optional[dict] = None


class RegisterRequest(BaseModel):
    name: str
    metadata: Optional[dict] = None


class SearchResult(BaseModel):
    person_id: str
    name: str
    similarity: float
    metadata: Optional[dict] = None


@router.post("/detect")
async def detect_faces(
    request: Request,
    image: UploadFile = File(...),
    max_faces: int = Form(100)
):
    """Detect faces in an image.

    Returns bounding boxes, landmarks, and attributes (age, gender) for each face.
    """
    start_time = time.time()

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
        img_array = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    recognizer = request.app.state.face_recognizer
    faces = recognizer.detect_faces(img_array, max_faces=max_faces)

    # Remove embeddings from response
    for face in faces:
        if "embedding" in face:
            del face["embedding"]

    processing_time = (time.time() - start_time) * 1000

    return {
        "face_count": len(faces),
        "faces": faces,
        "processing_time_ms": round(processing_time, 2)
    }


@router.post("/register")
async def register_face(
    request: Request,
    image: UploadFile = File(...),
    name: str = Form(...),
    metadata: Optional[str] = Form(None)
):
    """Register a face for recognition.

    - **image**: Image containing the face to register
    - **name**: Person's name
    - **metadata**: Optional JSON metadata
    """
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
        img_array = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Parse metadata
    import json
    meta_dict = {}
    if metadata:
        try:
            meta_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON")

    recognizer = request.app.state.face_recognizer
    person_id = str(uuid.uuid4())

    result = recognizer.register_face(
        person_id=person_id,
        name=name,
        image=img_array,
        metadata=meta_dict
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Registration failed"))

    return result


@router.post("/search", response_model=dict)
async def search_face(
    request: Request,
    image: UploadFile = File(...),
    top_k: int = Form(5)
):
    """Search for a face in the registered database.

    Returns matching persons with similarity scores.
    """
    start_time = time.time()

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
        img_array = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    recognizer = request.app.state.face_recognizer
    matches = recognizer.search_face(img_array, top_k=top_k)

    processing_time = (time.time() - start_time) * 1000

    return {
        "matches": matches,
        "match_count": len(matches),
        "processing_time_ms": round(processing_time, 2)
    }


@router.post("/identify")
async def identify_faces(
    request: Request,
    image: UploadFile = File(...)
):
    """Detect and identify all faces in an image.

    Combines face detection with recognition to identify registered persons.
    """
    start_time = time.time()

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
        img_array = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    recognizer = request.app.state.face_recognizer
    faces = recognizer.identify_faces(img_array)

    processing_time = (time.time() - start_time) * 1000

    identified_count = sum(1 for f in faces if f.get("match"))

    return {
        "face_count": len(faces),
        "identified_count": identified_count,
        "faces": faces,
        "processing_time_ms": round(processing_time, 2)
    }


@router.get("/registered")
async def list_registered_faces(request: Request):
    """List all registered faces."""
    recognizer = request.app.state.face_recognizer
    faces = recognizer.list_registered_faces()
    return {
        "total": len(faces),
        "faces": faces
    }


@router.delete("/{person_id}")
async def delete_face(request: Request, person_id: str):
    """Delete a registered face."""
    recognizer = request.app.state.face_recognizer
    deleted = recognizer.delete_face(person_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Person not found")

    return {"message": "Face deleted", "person_id": person_id}


@router.post("/liveness")
async def check_liveness(
    request: Request,
    image: UploadFile = File(...)
):
    """Check if the face is real (anti-spoofing).

    Detects printed photos, screens, and masks.
    """
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
        img_array = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    from app.services.recognition.face import LivenessDetector
    liveness = LivenessDetector()
    result = liveness.check_liveness(img_array)

    return result
