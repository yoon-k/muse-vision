from fastapi import APIRouter

from app.api.endpoints import detect, faces, search, streams

api_router = APIRouter()

api_router.include_router(detect.router, prefix="/detect", tags=["Detection"])
api_router.include_router(faces.router, prefix="/faces", tags=["Face Recognition"])
api_router.include_router(search.router, prefix="/search", tags=["Image Search"])
api_router.include_router(streams.router, prefix="/streams", tags=["CCTV Streams"])
