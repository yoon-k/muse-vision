from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import uuid
import time

router = APIRouter()


class SearchResult(BaseModel):
    image_id: str
    similarity: float
    metadata: Optional[dict] = None


class IndexRequest(BaseModel):
    image_id: Optional[str] = None
    metadata: Optional[dict] = None


@router.post("/similar")
async def search_similar_images(
    request: Request,
    image: UploadFile = File(...),
    top_k: int = Form(10)
):
    """Find visually similar images.

    Uses CLIP embeddings for semantic similarity search.
    """
    start_time = time.time()

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    searcher = request.app.state.searcher
    results = searcher.search_by_image(pil_image, top_k=top_k)

    processing_time = (time.time() - start_time) * 1000

    return {
        "results": results,
        "result_count": len(results),
        "processing_time_ms": round(processing_time, 2)
    }


@router.post("/text")
async def search_by_text(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(10)
):
    """Search images using text description.

    Examples:
    - "a dog playing in the park"
    - "sunset over the ocean"
    - "people walking on a busy street"
    """
    start_time = time.time()

    if len(query) < 3:
        raise HTTPException(status_code=400, detail="Query too short")

    searcher = request.app.state.searcher
    results = searcher.search_by_text(query, top_k=top_k)

    processing_time = (time.time() - start_time) * 1000

    return {
        "query": query,
        "results": results,
        "result_count": len(results),
        "processing_time_ms": round(processing_time, 2)
    }


@router.post("/index")
async def index_image(
    request: Request,
    image: UploadFile = File(...),
    image_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """Add an image to the search index.

    - **image**: Image to index
    - **image_id**: Optional custom ID (auto-generated if not provided)
    - **metadata**: Optional JSON metadata
    """
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
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

    # Generate ID if not provided
    img_id = image_id or str(uuid.uuid4())

    searcher = request.app.state.searcher
    success = searcher.index_image(img_id, pil_image, meta_dict)

    if not success:
        raise HTTPException(status_code=500, detail="Failed to index image")

    return {
        "success": True,
        "image_id": img_id,
        "index_size": searcher.get_index_size()
    }


@router.delete("/index/{image_id}")
async def delete_from_index(request: Request, image_id: str):
    """Remove an image from the search index."""
    searcher = request.app.state.searcher
    deleted = searcher.delete_image(image_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Image not found in index")

    return {"message": "Image removed from index", "image_id": image_id}


@router.get("/index/stats")
async def get_index_stats(request: Request):
    """Get search index statistics."""
    searcher = request.app.state.searcher

    return {
        "total_images": searcher.get_index_size(),
        "model": "CLIP ViT-L/14"
    }


@router.post("/duplicates")
async def find_duplicates(
    request: Request,
    image: UploadFile = File(...),
    threshold: float = Form(0.95)
):
    """Find duplicate or near-duplicate images.

    - **threshold**: Similarity threshold (0.9-1.0), higher = more strict
    """
    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    from app.services.search.clip import SemanticImageSearch
    searcher = SemanticImageSearch()
    duplicates = searcher.find_duplicates(pil_image, threshold=threshold)

    return {
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "threshold": threshold
    }
