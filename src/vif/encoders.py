"""Text encoders for VIF Critic training.

This module defines the TextEncoder protocol and implementations for encoding
journal entry text into dense vector representations. The protocol pattern
enables easy swapping of encoders for ablation studies.

Supported encoders:
- SBERTEncoder: Sentence-BERT models via sentence-transformers library
  - "nomic-ai/nomic-embed-text-v1.5" (768 native, Matryoshka-truncatable,
    8192 token context, MTEB 61.04 @ 256d) - recommended
  - "all-MiniLM-L6-v2" (384 dim, fast, good quality)
  - "all-mpnet-base-v2" (768 dim, higher quality, slower)

Usage:
    from src.vif.encoders import SBERTEncoder, create_encoder

    # Direct instantiation with Matryoshka truncation
    encoder = SBERTEncoder(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        truncate_dim=256,
        text_prefix="classification: ",
    )
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
    entries into dense semantic embeddings. Supports Matryoshka truncation
    for models trained with Matryoshka Representation Learning (e.g.
    nomic-embed-text-v1.5).

    Common models and their properties:
    - nomic-ai/nomic-embed-text-v1.5: 768 dim native, Matryoshka to 256/128/64,
      8192 token context (recommended)
    - all-MiniLM-L6-v2: 384 dim, fast, good quality
    - all-mpnet-base-v2: 768 dim, higher quality, slower

    Example:
        encoder = SBERTEncoder(
            "nomic-ai/nomic-embed-text-v1.5",
            trust_remote_code=True,
            truncate_dim=256,
            text_prefix="classification: ",
        )
        embeddings = encoder.encode(["Journal entry text here"])
        print(embeddings.shape)  # (1, 256)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        trust_remote_code: bool = False,
        truncate_dim: int | None = None,
        text_prefix: str = "",
    ):
        """Initialize the SBERT encoder.

        Args:
            model_name: Name of the sentence-transformers model to load
            trust_remote_code: Allow loading custom model code from HuggingFace
                (required for nomic-embed-text-v1.5)
            truncate_dim: If set, apply Matryoshka truncation to this
                dimensionality. The model must have been trained with
                Matryoshka Representation Learning for this to be valid.
            text_prefix: Prefix prepended to all input texts before encoding.
                Nomic models require task-specific prefixes (e.g.
                "classification: " for downstream classification tasks).
        """
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(
            model_name, trust_remote_code=trust_remote_code
        )
        self._native_dim = self._model.get_sentence_embedding_dimension()
        self._truncate_dim = truncate_dim
        self._text_prefix = text_prefix

        if truncate_dim is not None and truncate_dim > self._native_dim:
            raise ValueError(
                f"truncate_dim ({truncate_dim}) exceeds native embedding "
                f"dimension ({self._native_dim})"
            )

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embeddings (after truncation if set)."""
        if self._truncate_dim is not None:
            return self._truncate_dim
        return self._native_dim

    @property
    def model_name(self) -> str:
        """Name of the underlying model."""
        return self._model_name

    def _apply_prefix(self, texts: list[str]) -> list[str]:
        """Prepend task prefix to each text if configured."""
        if self._text_prefix:
            return [self._text_prefix + t for t in texts]
        return texts

    def _matryoshka_truncate(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply Matryoshka truncation: LayerNorm → slice → L2 normalize."""
        if self._truncate_dim is None:
            return embeddings
        # LayerNorm across embedding dimension (per-vector)
        mean = embeddings.mean(axis=-1, keepdims=True)
        var = embeddings.var(axis=-1, keepdims=True)
        embeddings = (embeddings - mean) / np.sqrt(var + 1e-5)
        # Truncate to target dimension
        embeddings = embeddings[:, : self._truncate_dim]
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms
        return embeddings

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into dense vectors.

        Args:
            texts: List of text strings to encode

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        prefixed = self._apply_prefix(texts)
        embeddings = self._model.encode(prefixed, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        return self._matryoshka_truncate(embeddings)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches for memory efficiency.

        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to encode per batch

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        prefixed = self._apply_prefix(texts)
        embeddings = self._model.encode(
            prefixed,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )
        embeddings = embeddings.astype(np.float32)
        return self._matryoshka_truncate(embeddings)


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
        return SBERTEncoder(
            model_name,
            trust_remote_code=config.get("trust_remote_code", False),
            truncate_dim=config.get("truncate_dim"),
            text_prefix=config.get("text_prefix", ""),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
