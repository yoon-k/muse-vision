from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image
import torch
import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import structlog

from app.core.config import settings

logger = structlog.get_logger()


class FaceRecognizer:
    """Face detection and recognition service using InsightFace."""

    def __init__(self):
        self.device = settings.DEVICE
        self.similarity_threshold = settings.FACE_SIMILARITY_THRESHOLD
        self.app = None
        self.face_database = {}  # In-memory face database
        self._load_models()

    def _load_models(self):
        """Load face detection and recognition models."""
        try:
            # Initialize InsightFace
            providers = ['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=providers
            )
            self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1)
            logger.info("Face models loaded", device=self.device)
        except Exception as e:
            logger.error("Failed to load face models", error=str(e))
            raise

    def detect_faces(
        self,
        image: np.ndarray,
        max_faces: int = None
    ) -> List[Dict[str, Any]]:
        """Detect faces in an image.

        Args:
            image: Input image (BGR format)
            max_faces: Maximum number of faces to detect

        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        max_faces = max_faces or settings.MAX_FACES_PER_IMAGE

        faces = self.app.get(image, max_num=max_faces)

        results = []
        for face in faces:
            result = {
                "bbox": [int(x) for x in face.bbox],  # [x1, y1, x2, y2]
                "confidence": float(face.det_score),
                "landmarks": face.kps.tolist() if face.kps is not None else None,
                "embedding": face.embedding.tolist() if face.embedding is not None else None,
            }

            # Add face attributes if available
            if hasattr(face, 'age'):
                result["age"] = int(face.age)
            if hasattr(face, 'gender'):
                result["gender"] = "male" if face.gender == 1 else "female"

            results.append(result)

        return results

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding from image.

        Args:
            image: Input image with a face

        Returns:
            512-dimensional face embedding or None if no face found
        """
        faces = self.app.get(image, max_num=1)
        if faces and faces[0].embedding is not None:
            return faces[0].embedding
        return None

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two face embeddings.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Similarity score (0-1)
        """
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        return float(np.dot(embedding1, embedding2))

    def register_face(
        self,
        person_id: str,
        name: str,
        image: np.ndarray,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Register a face in the database.

        Args:
            person_id: Unique person identifier
            name: Person's name
            image: Image containing the face
            metadata: Additional metadata

        Returns:
            Registration result
        """
        embedding = self.get_embedding(image)
        if embedding is None:
            return {"success": False, "error": "No face detected"}

        self.face_database[person_id] = {
            "name": name,
            "embedding": embedding,
            "metadata": metadata or {}
        }

        logger.info("Face registered", person_id=person_id, name=name)
        return {
            "success": True,
            "person_id": person_id,
            "name": name
        }

    def search_face(
        self,
        image: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for matching faces in the database.

        Args:
            image: Image containing the face to search
            top_k: Number of top matches to return

        Returns:
            List of matching persons with similarity scores
        """
        query_embedding = self.get_embedding(image)
        if query_embedding is None:
            return []

        # Calculate similarities with all registered faces
        matches = []
        for person_id, data in self.face_database.items():
            similarity = self.compute_similarity(query_embedding, data["embedding"])
            if similarity >= self.similarity_threshold:
                matches.append({
                    "person_id": person_id,
                    "name": data["name"],
                    "similarity": similarity,
                    "metadata": data["metadata"]
                })

        # Sort by similarity and return top_k
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches[:top_k]

    def identify_faces(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Detect and identify all faces in an image.

        Args:
            image: Input image

        Returns:
            List of detected faces with identification results
        """
        faces = self.detect_faces(image)

        for face in faces:
            if face.get("embedding"):
                embedding = np.array(face["embedding"])

                # Find best match
                best_match = None
                best_similarity = 0

                for person_id, data in self.face_database.items():
                    similarity = self.compute_similarity(embedding, data["embedding"])
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_match = {
                            "person_id": person_id,
                            "name": data["name"],
                            "similarity": similarity
                        }

                face["match"] = best_match
                # Remove raw embedding from response
                del face["embedding"]

        return faces

    def delete_face(self, person_id: str) -> bool:
        """Delete a face from the database.

        Args:
            person_id: Person ID to delete

        Returns:
            True if deleted, False if not found
        """
        if person_id in self.face_database:
            del self.face_database[person_id]
            logger.info("Face deleted", person_id=person_id)
            return True
        return False

    def list_registered_faces(self) -> List[Dict[str, Any]]:
        """List all registered faces.

        Returns:
            List of registered persons (without embeddings)
        """
        return [
            {
                "person_id": pid,
                "name": data["name"],
                "metadata": data["metadata"]
            }
            for pid, data in self.face_database.items()
        ]


class LivenessDetector:
    """Anti-spoofing liveness detection."""

    def __init__(self):
        self.model = None
        # In production, load a trained anti-spoofing model

    def check_liveness(self, image: np.ndarray) -> Dict[str, Any]:
        """Check if the face is real or spoofed.

        Args:
            image: Input image

        Returns:
            Liveness check result
        """
        # Simple heuristic-based checks (production would use ML model)
        # 1. Check image quality
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 2. Check color distribution (printed photos have different distribution)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_mean = hsv[:, :, 1].mean()

        is_live = laplacian_var > 100 and saturation_mean > 20

        return {
            "is_live": is_live,
            "confidence": min(laplacian_var / 200, 1.0),
            "checks": {
                "blur_score": laplacian_var,
                "saturation_score": saturation_mean
            }
        }
