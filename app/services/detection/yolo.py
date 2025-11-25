from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
import structlog

from app.core.config import settings

logger = structlog.get_logger()


class YOLODetector:
    """YOLOv8 based object detection service."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or settings.YOLO_MODEL
        self.device = settings.DEVICE
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load YOLO model."""
        try:
            self.model = YOLO(self.model_path)
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda")
            logger.info("YOLO model loaded", model=self.model_path, device=self.device)
        except Exception as e:
            logger.error("Failed to load YOLO model", error=str(e))
            raise

    def detect(
        self,
        image: np.ndarray,
        confidence: float = None,
        classes: List[int] = None,
        max_detections: int = 100
    ) -> List[Dict[str, Any]]:
        """Detect objects in an image.

        Args:
            image: Input image as numpy array (BGR or RGB)
            confidence: Minimum confidence threshold
            classes: Filter by specific class IDs
            max_detections: Maximum number of detections to return

        Returns:
            List of detection results
        """
        confidence = confidence or settings.DETECTION_CONFIDENCE

        results = self.model(
            image,
            conf=confidence,
            classes=classes,
            max_det=max_detections,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            detection = {
                "class_id": int(box.cls[0]),
                "class": results.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": [int(x) for x in box.xyxy[0].tolist()],  # [x1, y1, x2, y2]
            }

            # Add segmentation mask if available
            if results.masks is not None:
                mask_idx = list(results.boxes).index(box)
                if mask_idx < len(results.masks):
                    detection["mask"] = results.masks[mask_idx].data.cpu().numpy()

            detections.append(detection)

        return detections

    def detect_batch(
        self,
        images: List[np.ndarray],
        confidence: float = None
    ) -> List[List[Dict[str, Any]]]:
        """Detect objects in multiple images.

        Args:
            images: List of input images
            confidence: Minimum confidence threshold

        Returns:
            List of detection results for each image
        """
        confidence = confidence or settings.DETECTION_CONFIDENCE

        results = self.model(
            images,
            conf=confidence,
            verbose=False
        )

        all_detections = []
        for result in results:
            detections = []
            for box in result.boxes:
                detection = {
                    "class_id": int(box.cls[0]),
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [int(x) for x in box.xyxy[0].tolist()],
                }
                detections.append(detection)
            all_detections.append(detections)

        return all_detections

    def detect_persons(
        self,
        image: np.ndarray,
        confidence: float = None
    ) -> List[Dict[str, Any]]:
        """Detect only persons in an image.

        Args:
            image: Input image
            confidence: Minimum confidence threshold

        Returns:
            List of person detections
        """
        # Class 0 is 'person' in COCO dataset
        return self.detect(image, confidence=confidence, classes=[0])

    def detect_animals(
        self,
        image: np.ndarray,
        confidence: float = None
    ) -> List[Dict[str, Any]]:
        """Detect animals in an image.

        Args:
            image: Input image
            confidence: Minimum confidence threshold

        Returns:
            List of animal detections
        """
        # Animal classes in COCO: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
        animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        return self.detect(image, confidence=confidence, classes=animal_classes)

    def get_class_names(self) -> Dict[int, str]:
        """Get mapping of class IDs to names."""
        return self.model.names

    def count_objects(
        self,
        image: np.ndarray,
        target_class: str = None
    ) -> Dict[str, int]:
        """Count objects in an image.

        Args:
            image: Input image
            target_class: Optional specific class to count

        Returns:
            Dictionary of class names to counts
        """
        detections = self.detect(image)

        counts = {}
        for det in detections:
            class_name = det["class"]
            if target_class is None or class_name == target_class:
                counts[class_name] = counts.get(class_name, 0) + 1

        return counts


class ObjectTracker:
    """Multi-object tracking using YOLO + ByteTrack."""

    def __init__(self):
        self.detector = YOLODetector()
        self.tracks = {}
        self.next_id = 0

    def update(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Update tracker with new frame.

        Args:
            image: Current frame

        Returns:
            List of tracked objects with IDs
        """
        # Use YOLO's built-in tracking
        results = self.detector.model.track(
            image,
            persist=True,
            verbose=False
        )[0]

        tracked_objects = []
        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id):
                tracked_objects.append({
                    "track_id": int(track_id),
                    "class": results.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": [int(x) for x in box.xyxy[0].tolist()]
                })

        return tracked_objects

    def reset(self):
        """Reset tracker state."""
        self.tracks = {}
        self.detector.model.predictor = None
