from typing import List, Dict, Any, Optional, Union
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
import structlog

from app.core.config import settings

logger = structlog.get_logger()


class CLIPSearcher:
    """CLIP-based image and text search service."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or "openai/clip-vit-large-patch14"
        self.device = "cuda" if settings.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.image_index = {}  # In-memory image index
        self._load_model()

    def _load_model(self):
        """Load CLIP model and processor."""
        try:
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("CLIP model loaded", model=self.model_name, device=self.device)
        except Exception as e:
            logger.error("Failed to load CLIP model", error=str(e))
            raise

    @torch.no_grad()
    def get_image_embedding(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Get CLIP embedding for an image.

        Args:
            image: Input image (numpy array or PIL Image)

        Returns:
            512-dimensional image embedding
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        features = self.model.get_image_features(**inputs)
        embedding = features.cpu().numpy()[0]

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    @torch.no_grad()
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Get CLIP embedding for text.

        Args:
            text: Input text

        Returns:
            512-dimensional text embedding
        """
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        features = self.model.get_text_features(**inputs)
        embedding = features.cpu().numpy()[0]

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    @torch.no_grad()
    def get_batch_embeddings(
        self,
        images: List[Union[np.ndarray, Image.Image]]
    ) -> np.ndarray:
        """Get embeddings for multiple images.

        Args:
            images: List of input images

        Returns:
            Array of embeddings (N x 512)
        """
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)

        inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        features = self.model.get_image_features(**inputs)
        embeddings = features.cpu().numpy()

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        return embeddings

    def index_image(
        self,
        image_id: str,
        image: Union[np.ndarray, Image.Image],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add an image to the search index.

        Args:
            image_id: Unique image identifier
            image: Image to index
            metadata: Optional metadata

        Returns:
            True if indexed successfully
        """
        try:
            embedding = self.get_image_embedding(image)
            self.image_index[image_id] = {
                "embedding": embedding,
                "metadata": metadata or {}
            }
            logger.info("Image indexed", image_id=image_id)
            return True
        except Exception as e:
            logger.error("Failed to index image", image_id=image_id, error=str(e))
            return False

    def search_by_image(
        self,
        query_image: Union[np.ndarray, Image.Image],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for similar images.

        Args:
            query_image: Query image
            top_k: Number of results to return

        Returns:
            List of similar images with scores
        """
        query_embedding = self.get_image_embedding(query_image)
        return self._search(query_embedding, top_k)

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search images by text description.

        Args:
            query_text: Text query
            top_k: Number of results to return

        Returns:
            List of matching images with scores
        """
        query_embedding = self.get_text_embedding(query_text)
        return self._search(query_embedding, top_k)

    def _search(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Internal search method.

        Args:
            query_embedding: Query embedding
            top_k: Number of results

        Returns:
            Sorted search results
        """
        if not self.image_index:
            return []

        results = []
        for image_id, data in self.image_index.items():
            similarity = float(np.dot(query_embedding, data["embedding"]))
            results.append({
                "image_id": image_id,
                "similarity": similarity,
                "metadata": data["metadata"]
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def delete_image(self, image_id: str) -> bool:
        """Remove an image from the index.

        Args:
            image_id: Image ID to delete

        Returns:
            True if deleted, False if not found
        """
        if image_id in self.image_index:
            del self.image_index[image_id]
            logger.info("Image deleted from index", image_id=image_id)
            return True
        return False

    def get_index_size(self) -> int:
        """Get the number of indexed images."""
        return len(self.image_index)


class SemanticImageSearch:
    """Advanced semantic image search with multiple models."""

    def __init__(self):
        self.clip_searcher = CLIPSearcher()
        # Additional models for enhanced search
        self.sentence_model = SentenceTransformer('clip-ViT-B-32')

    def hybrid_search(
        self,
        query: Union[str, np.ndarray, Image.Image],
        top_k: int = 10,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining multiple methods.

        Args:
            query: Text query or image
            top_k: Number of results
            rerank: Whether to rerank results

        Returns:
            Search results
        """
        if isinstance(query, str):
            results = self.clip_searcher.search_by_text(query, top_k * 2)
        else:
            results = self.clip_searcher.search_by_image(query, top_k * 2)

        if rerank and results:
            # Simple reranking based on metadata (can be enhanced)
            # In production, use a cross-encoder or other reranking model
            pass

        return results[:top_k]

    def find_duplicates(
        self,
        image: Union[np.ndarray, Image.Image],
        threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """Find duplicate or near-duplicate images.

        Args:
            image: Query image
            threshold: Similarity threshold for duplicates

        Returns:
            List of potential duplicates
        """
        results = self.clip_searcher.search_by_image(image, top_k=100)
        return [r for r in results if r["similarity"] >= threshold]
