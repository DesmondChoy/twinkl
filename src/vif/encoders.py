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

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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

    def render_inputs(self, texts: list[str]) -> list[str]:
        """Render the exact text strings that will be tokenized by the model."""
        ...

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Return token counts for the rendered model inputs."""
        ...


@dataclass(frozen=True)
class _EncodeRequest:
    """Prepared inputs for both model encoding and token-count analysis."""

    model_inputs: list[str]
    rendered_inputs: list[str]
    encode_kwargs: dict[str, Any]


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
        text_prefix: str | None = "",
        prompt_name: str | None = None,
        prompt: str | None = None,
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
            prompt_name: Native sentence-transformers prompt name to use
                during encoding.
            prompt: Native sentence-transformers prompt string to use
                during encoding.
        """
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(
            model_name, trust_remote_code=trust_remote_code
        )
        self._native_dim = self._model.get_sentence_embedding_dimension()
        self._truncate_dim = truncate_dim
        self._text_prefix = text_prefix or ""
        self._prompt_name = prompt_name or None
        self._prompt = prompt or None
        self._use_native_truncate = self._should_use_native_truncate()

        active_input_modes = [
            mode
            for mode, value in (
                ("text_prefix", self._text_prefix),
                ("prompt_name", self._prompt_name),
                ("prompt", self._prompt),
            )
            if value
        ]
        if len(active_input_modes) > 1:
            joined = ", ".join(active_input_modes)
            raise ValueError(
                "Configure at most one of text_prefix, prompt_name, or prompt; "
                f"got {joined}."
            )

        if truncate_dim is not None and truncate_dim > self._native_dim:
            raise ValueError(
                f"truncate_dim ({truncate_dim}) exceeds native embedding "
                f"dimension ({self._native_dim})"
            )

        if self._prompt_name is not None:
            prompts = getattr(self._model, "prompts", {}) or {}
            if self._prompt_name not in prompts:
                available = ", ".join(sorted(prompts)) or "none"
                raise ValueError(
                    f"Unknown prompt_name '{self._prompt_name}' for "
                    f"{self._model_name}. Available prompts: {available}"
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

    @property
    def max_seq_length(self) -> int:
        """Maximum context window exposed by the underlying model."""
        return self._model.max_seq_length

    def _should_use_native_truncate(self) -> bool:
        """Use model-native truncation only for models known to support it."""
        lower = self._model_name.lower()
        return "qwen" in lower and "embedding" in lower

    def _resolve_prompt_text(self) -> str:
        """Resolve the literal prompt string used for tokenization analysis."""
        if self._prompt is not None:
            return self._prompt
        if self._prompt_name is None:
            return ""
        prompts = getattr(self._model, "prompts", {}) or {}
        return prompts[self._prompt_name]

    def _build_encode_request(self, texts: list[str]) -> _EncodeRequest:
        """Prepare model inputs and rendered inputs from one config source."""
        raw_inputs = list(texts)
        prompt_text = self._resolve_prompt_text()

        if self._text_prefix:
            rendered_inputs = [self._text_prefix + text for text in raw_inputs]
            return _EncodeRequest(
                model_inputs=rendered_inputs,
                rendered_inputs=rendered_inputs,
                encode_kwargs={},
            )

        if prompt_text:
            rendered_inputs = [prompt_text + text for text in raw_inputs]
        else:
            rendered_inputs = raw_inputs

        encode_kwargs: dict[str, Any] = {}
        if self._prompt_name is not None:
            encode_kwargs["prompt_name"] = self._prompt_name
        elif self._prompt is not None:
            encode_kwargs["prompt"] = self._prompt

        return _EncodeRequest(
            model_inputs=raw_inputs,
            rendered_inputs=rendered_inputs,
            encode_kwargs=encode_kwargs,
        )

    def render_inputs(self, texts: list[str]) -> list[str]:
        """Render the exact text strings that will be tokenized by the model."""
        return self._build_encode_request(texts).rendered_inputs

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Count tokens for the rendered model inputs."""
        rendered_inputs = self.render_inputs(texts)
        tokenizer = self._model.tokenizer
        return [
            len(tokenizer.encode(rendered_text, add_special_tokens=True))
            for rendered_text in rendered_inputs
        ]

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

    def _encode_internal(
        self,
        texts: list[str],
        *,
        batch_size: int | None,
        show_progress_bar: bool | None,
    ) -> np.ndarray:
        """Encode texts through one shared preparation path."""
        request = self._build_encode_request(texts)
        encode_kwargs = dict(request.encode_kwargs)
        encode_kwargs["convert_to_numpy"] = True

        if batch_size is not None:
            encode_kwargs["batch_size"] = batch_size
        if show_progress_bar is not None:
            encode_kwargs["show_progress_bar"] = show_progress_bar
        if self._use_native_truncate and self._truncate_dim is not None:
            encode_kwargs["truncate_dim"] = self._truncate_dim

        embeddings = self._model.encode(request.model_inputs, **encode_kwargs)
        embeddings = embeddings.astype(np.float32)
        if self._use_native_truncate:
            return embeddings
        return self._matryoshka_truncate(embeddings)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into dense vectors.

        Args:
            texts: List of text strings to encode

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        return self._encode_internal(
            texts,
            batch_size=None,
            show_progress_bar=None,
        )

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches for memory efficiency.

        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to encode per batch

        Returns:
            np.ndarray of shape (len(texts), embedding_dim)
        """
        return self._encode_internal(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
        )


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
            prompt_name=config.get("prompt_name"),
            prompt=config.get("prompt"),
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
