from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import cv2
import numpy as np
from datetime import datetime
import uuid
import structlog

logger = structlog.get_logger()
router = APIRouter()


class StreamConfig(BaseModel):
    name: str
    url: str  # RTSP, HTTP, or file path
    detection_enabled: bool = True
    face_recognition_enabled: bool = False
    tracking_enabled: bool = False
    alert_on_person: bool = False
    target_person_ids: Optional[List[str]] = None  # Face IDs to watch for


class StreamInfo(BaseModel):
    stream_id: str
    name: str
    url: str
    status: str  # running, stopped, error
    fps: float
    resolution: Optional[List[int]] = None
    detection_enabled: bool
    face_recognition_enabled: bool
    event_count: int = 0


class DetectionEvent(BaseModel):
    event_id: str
    stream_id: str
    timestamp: datetime
    event_type: str  # person_detected, face_recognized, motion_detected
    data: Dict[str, Any]
    thumbnail_base64: Optional[str] = None


# In-memory storage for streams and events
active_streams: Dict[str, Dict[str, Any]] = {}
stream_events: Dict[str, List[DetectionEvent]] = {}


@router.post("", response_model=StreamInfo)
async def create_stream(
    request: Request,
    config: StreamConfig,
    background_tasks: BackgroundTasks
):
    """Register and start a new CCTV stream.

    Supports:
    - RTSP: rtsp://username:password@ip:port/path
    - HTTP: http://ip:port/stream
    - Local file: /path/to/video.mp4
    """
    stream_id = str(uuid.uuid4())

    # Validate stream URL
    cap = cv2.VideoCapture(config.url)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Cannot connect to stream")

    # Get stream info
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    stream_info = {
        "stream_id": stream_id,
        "name": config.name,
        "url": config.url,
        "status": "running",
        "fps": fps,
        "resolution": [width, height],
        "detection_enabled": config.detection_enabled,
        "face_recognition_enabled": config.face_recognition_enabled,
        "tracking_enabled": config.tracking_enabled,
        "alert_on_person": config.alert_on_person,
        "target_person_ids": config.target_person_ids or [],
        "event_count": 0,
        "stop_flag": False
    }

    active_streams[stream_id] = stream_info
    stream_events[stream_id] = []

    # Start background processing
    background_tasks.add_task(
        process_stream,
        stream_id,
        config.url,
        request.app.state
    )

    return StreamInfo(**{k: v for k, v in stream_info.items() if k != "stop_flag"})


async def process_stream(stream_id: str, url: str, app_state):
    """Background task to process stream frames."""
    logger.info("Starting stream processing", stream_id=stream_id)

    cap = cv2.VideoCapture(url)
    frame_count = 0
    skip_frames = 5  # Process every 5th frame for efficiency

    detector = app_state.detector
    face_recognizer = app_state.face_recognizer

    try:
        while stream_id in active_streams and not active_streams[stream_id].get("stop_flag"):
            ret, frame = cap.read()
            if not ret:
                # Try to reconnect
                cap.release()
                await asyncio.sleep(5)
                cap = cv2.VideoCapture(url)
                continue

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            stream_config = active_streams[stream_id]

            # Object detection
            if stream_config["detection_enabled"]:
                detections = detector.detect_persons(frame, confidence=0.5)

                if detections and stream_config["alert_on_person"]:
                    event = DetectionEvent(
                        event_id=str(uuid.uuid4()),
                        stream_id=stream_id,
                        timestamp=datetime.utcnow(),
                        event_type="person_detected",
                        data={
                            "person_count": len(detections),
                            "detections": detections
                        }
                    )
                    stream_events[stream_id].append(event)
                    active_streams[stream_id]["event_count"] += 1

            # Face recognition
            if stream_config["face_recognition_enabled"]:
                faces = face_recognizer.identify_faces(frame)

                for face in faces:
                    if face.get("match"):
                        person_id = face["match"]["person_id"]

                        # Check if this is a target person
                        if (not stream_config["target_person_ids"] or
                                person_id in stream_config["target_person_ids"]):

                            event = DetectionEvent(
                                event_id=str(uuid.uuid4()),
                                stream_id=stream_id,
                                timestamp=datetime.utcnow(),
                                event_type="face_recognized",
                                data={
                                    "person_id": person_id,
                                    "name": face["match"]["name"],
                                    "similarity": face["match"]["similarity"],
                                    "bbox": face["bbox"]
                                }
                            )
                            stream_events[stream_id].append(event)
                            active_streams[stream_id]["event_count"] += 1
                            logger.info(
                                "Face recognized",
                                stream_id=stream_id,
                                person=face["match"]["name"]
                            )

            await asyncio.sleep(0.01)  # Yield to event loop

    except Exception as e:
        logger.error("Stream processing error", stream_id=stream_id, error=str(e))
        if stream_id in active_streams:
            active_streams[stream_id]["status"] = "error"
    finally:
        cap.release()
        if stream_id in active_streams:
            active_streams[stream_id]["status"] = "stopped"

    logger.info("Stream processing stopped", stream_id=stream_id)


@router.get("", response_model=List[StreamInfo])
async def list_streams():
    """List all registered streams."""
    return [
        StreamInfo(**{k: v for k, v in stream.items() if k != "stop_flag"})
        for stream in active_streams.values()
    ]


@router.get("/{stream_id}", response_model=StreamInfo)
async def get_stream(stream_id: str):
    """Get stream details."""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    stream = active_streams[stream_id]
    return StreamInfo(**{k: v for k, v in stream.items() if k != "stop_flag"})


@router.delete("/{stream_id}")
async def stop_stream(stream_id: str):
    """Stop and remove a stream."""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    active_streams[stream_id]["stop_flag"] = True
    await asyncio.sleep(1)  # Wait for processing to stop

    del active_streams[stream_id]
    if stream_id in stream_events:
        del stream_events[stream_id]

    return {"message": "Stream stopped", "stream_id": stream_id}


@router.get("/{stream_id}/events")
async def get_stream_events(
    stream_id: str,
    event_type: Optional[str] = None,
    limit: int = 100
):
    """Get detection events from a stream.

    - **event_type**: Filter by type (person_detected, face_recognized)
    - **limit**: Maximum number of events to return
    """
    if stream_id not in stream_events:
        raise HTTPException(status_code=404, detail="Stream not found")

    events = stream_events[stream_id]

    if event_type:
        events = [e for e in events if e.event_type == event_type]

    # Return most recent events
    events = events[-limit:]

    return {
        "stream_id": stream_id,
        "event_count": len(events),
        "events": [e.dict() for e in events]
    }


@router.post("/{stream_id}/watch")
async def add_watch_target(
    stream_id: str,
    person_id: str
):
    """Add a person to watch for in the stream."""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    if person_id not in active_streams[stream_id]["target_person_ids"]:
        active_streams[stream_id]["target_person_ids"].append(person_id)

    return {
        "message": "Watch target added",
        "person_id": person_id,
        "targets": active_streams[stream_id]["target_person_ids"]
    }


@router.delete("/{stream_id}/watch/{person_id}")
async def remove_watch_target(stream_id: str, person_id: str):
    """Remove a person from watch list."""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    targets = active_streams[stream_id]["target_person_ids"]
    if person_id in targets:
        targets.remove(person_id)

    return {
        "message": "Watch target removed",
        "person_id": person_id,
        "targets": targets
    }


@router.get("/{stream_id}/snapshot")
async def get_snapshot(request: Request, stream_id: str):
    """Get current frame from stream."""
    if stream_id not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    url = active_streams[stream_id]["url"]
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot read from stream")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Failed to capture frame")

    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame)

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg"
    )


# Need to import io at the top
import io
