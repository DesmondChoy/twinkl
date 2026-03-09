"""Shared fixtures for VIF test suite."""

import hashlib

import numpy as np
import pytest


class MockTextEncoder:
    """Deterministic text encoder satisfying the TextEncoder protocol.

    Returns all-ones embeddings of configurable dimension so tests can
    verify state assembly logic without real model weights.
    """

    embedding_dim = 8
    model_name = "mock"

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), self.embedding_dim), dtype=np.float32)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(texts)


class ContentAwareMockTextEncoder:
    """Deterministic text encoder that preserves content differences in tests."""

    embedding_dim = 8
    model_name = "content-aware-mock"

    def _encode_text(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()[: self.embedding_dim]
        return np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.stack([self._encode_text(text) for text in texts]).astype(np.float32)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(texts)


@pytest.fixture
def mock_text_encoder() -> MockTextEncoder:
    """Provide a MockTextEncoder instance for state encoder tests."""
    return MockTextEncoder()


@pytest.fixture
def content_aware_text_encoder() -> ContentAwareMockTextEncoder:
    """Provide a deterministic encoder whose embeddings depend on input text."""
    return ContentAwareMockTextEncoder()
