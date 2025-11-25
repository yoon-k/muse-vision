# MUSE Vision - AI-Powered Image Recognition & Search Platform

<p align="center">
  <img src="assets/muse-vision-logo.png" alt="MUSE Vision Logo" width="200"/>
</p>

MUSE Vision은 딥러닝 기반 이미지 인식, 얼굴 검출/인식, 객체 탐지, 유사 이미지 검색을 제공하는 엔터프라이즈 비전 AI 플랫폼입니다.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Features

### Object Detection
- **Multi-Object Detection** - YOLO v8 기반 실시간 객체 탐지
- **100+ Categories** - 사람, 동물, 차량, 가구 등 100개 이상 카테고리
- **Custom Training** - 커스텀 객체 학습 지원
- **Video Analysis** - 영상 스트림 실시간 분석

### Face Recognition
- **Face Detection** - RetinaFace 기반 정밀 얼굴 검출
- **Face Recognition** - ArcFace 기반 얼굴 인식 (99.8% 정확도)
- **Face Clustering** - 자동 얼굴 클러스터링/그룹화
- **Liveness Detection** - 스푸핑 방지 생체 인식

### Image Search
- **Visual Similarity** - 임베딩 기반 유사 이미지 검색
- **Reverse Image Search** - 이미지로 이미지 검색
- **Semantic Search** - 텍스트로 이미지 검색 (CLIP)
- **Real-time Indexing** - 수백만 이미지 실시간 인덱싱

### CCTV Integration
- **Live Streaming** - RTSP/HLS 스트림 지원
- **Motion Detection** - 움직임 감지 및 알림
- **Person Tracking** - 다중 카메라 인물 추적
- **Alert System** - 실시간 이벤트 알림

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Sources                           │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │  Image  │ │  Video  │ │  CCTV   │ │ Webcam  │            │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘            │
└───────┼───────────┼───────────┼───────────┼─────────────────┘
        │           │           │           │
        └───────────┴─────┬─────┴───────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    MUSE Vision API                           │
│                      (FastAPI)                               │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│   Detection   │ │  Recognition  │ │    Search     │
│   Service     │ │   Service     │ │   Service     │
│   (YOLO v8)   │ │  (ArcFace)    │ │   (CLIP)      │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│  PostgreSQL   │ │    Milvus     │ │    Redis      │
│   (Metadata)  │ │  (Vectors)    │ │   (Cache)     │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Quick Start

### Docker

```bash
cp .env.example .env
docker-compose up -d

# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 모델 다운로드
python -m app.models.download

# 서버 실행
uvicorn app.main:app --reload --port 8000
```

## API Examples

### Object Detection

```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "image=@photo.jpg" \
  -F "confidence=0.5"
```

Response:
```json
{
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "bbox": [100, 150, 300, 500],
      "attributes": {"gender": "male", "age_range": "25-35"}
    },
    {
      "class": "dog",
      "confidence": 0.89,
      "bbox": [400, 300, 550, 450],
      "attributes": {"breed": "golden_retriever"}
    }
  ],
  "processing_time_ms": 45
}
```

### Face Recognition

```bash
# 얼굴 등록
curl -X POST http://localhost:8000/api/v1/faces/register \
  -F "image=@face.jpg" \
  -F "name=John Doe" \
  -F "metadata={\"employee_id\": \"E001\"}"

# 얼굴 검색
curl -X POST http://localhost:8000/api/v1/faces/search \
  -F "image=@unknown.jpg"
```

Response:
```json
{
  "matches": [
    {
      "person_id": "person_abc123",
      "name": "John Doe",
      "similarity": 0.94,
      "metadata": {"employee_id": "E001"}
    }
  ],
  "face_count": 1,
  "processing_time_ms": 120
}
```

### Image Search

```bash
# 유사 이미지 검색
curl -X POST http://localhost:8000/api/v1/search/similar \
  -F "image=@query.jpg" \
  -F "top_k=10"

# 텍스트로 이미지 검색
curl -X POST http://localhost:8000/api/v1/search/text \
  -H "Content-Type: application/json" \
  -d '{"query": "a dog playing in the park", "top_k": 10}'
```

### CCTV Streaming

```bash
# 스트림 등록
curl -X POST http://localhost:8000/api/v1/streams \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Entrance Camera",
    "url": "rtsp://192.168.1.100/stream1",
    "detection_enabled": true,
    "face_recognition_enabled": true
  }'

# 이벤트 조회
curl http://localhost:8000/api/v1/streams/{stream_id}/events
```

## Model Performance

| Task | Model | Accuracy | Speed (GPU) |
|------|-------|----------|-------------|
| Object Detection | YOLOv8x | mAP 53.9 | 12ms |
| Face Detection | RetinaFace | 99.5% | 8ms |
| Face Recognition | ArcFace | 99.8% | 5ms |
| Image Embedding | CLIP ViT-L | - | 15ms |

## Configuration

```env
# Models
YOLO_MODEL=yolov8x.pt
FACE_DETECTION_MODEL=retinaface_resnet50
FACE_RECOGNITION_MODEL=arcface_r100

# Vector DB
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Database
DATABASE_URL=postgresql://localhost:5432/muse_vision

# GPU
CUDA_VISIBLE_DEVICES=0
```

## License

MIT License
