"""Text encoders for VIF Critic training.

This module defines the TextEncoder protocol and implementations for encoding
journal entry text into dense vector representations. The protocol pattern
enables easy swapping of encoders for ablation studies.

Supported encoders:
- SBERTEncoder: Sentence-BERT models via sentence-transformers library
  - "all-MiniLM-L6-v2" (384 dim, fast, good quality) - default
  - "all-mpnet-base-v2" (768 dim, higher quality, slower)
  - "paraphrase-MiniLM-L3-v2" (384 dim, fastest, lower quality)

Usage:
    from src.vif.encoders import SBERTEncoder, create_encoder

    # Direct instantiation
    encoder = SBERTEncoder("all-MiniLM-L6-v2")
    embeddings = encoder.encode(["Hello world", "Another text"])

    # From config
    encoder = create_encoder({"type": "sbert", "model_name": "all-mpnet-base-v2"})
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TextEncoder(Protocol):
    """Protocol for text encoders used in the VIF pipeline.

    This protocol defines the interface that any text encoder must implement
    to be used with the StateEncoder and VIF training pipeline.

    The protocol pattern allows easy swapping of encoder implementations
    for ablation studies without changing downstream code.
    """

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        ...

    @property
    def model_name(self) -> str:
        """Human-readable name of the encoder model."""
        ...

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into dense vectors.

        Args:
            texts: List of text strings to encode

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        ...

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches for memory efficiency.

        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to encode per batch

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        ...


class SBERTEncoder:
    """Sentence-BERT encoder using the sentence-transformers library.

    This encoder wraps sentence-transformers models for encoding journal
    entries into dense semantic embeddings.

    Common models and their properties:
    - all-MiniLM-L6-v2: 384 dim, fast, good quality (recommended default)
    - all-mpnet-base-v2: 768 dim, higher quality, slower
    - paraphrase-MiniLM-L3-v2: 384 dim, fastest, lower quality

    Example:
        encoder = SBERTEncoder("all-MiniLM-L6-v2")
        embeddings = encoder.encode(["Journal entry text here"])
        print(embeddings.shape)  # (1, 384)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the SBERT encoder.

        Args:
            model_name: Name of the sentence-transformers model to load
        """
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._embedding_dim = self._model.get_sentence_embedding_dimension()

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings."""
        return self._embedding_dim

    @property
    def model_name(self) -> str:
        """Name of the underlying model."""
        return self._model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into dense vectors.

        Args:
            texts: List of text strings to encode

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        # sentence-transformers returns numpy array directly
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches for memory efficiency.

        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to encode per batch

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.astype(np.float32)


def create_encoder(config: dict) -> TextEncoder:
    """Factory function to create an encoder from configuration.

    This enables configuration-driven encoder selection for ablation studies.

    Args:
        config: Dict with keys:
            - type: Encoder type ("sbert")
            - model_name: Model identifier (e.g., "all-MiniLM-L6-v2")

    Returns:
        TextEncoder instance

    Raises:
        ValueError: If encoder type is unknown

    Example:
        # From YAML config
        config = {"type": "sbert", "model_name": "all-mpnet-base-v2"}
        encoder = create_encoder(config)
    """
    encoder_type = config.get("type", "sbert")

    if encoder_type == "sbert":
        model_name = config.get("model_name", "all-MiniLM-L6-v2")
        return SBERTEncoder(model_name)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
