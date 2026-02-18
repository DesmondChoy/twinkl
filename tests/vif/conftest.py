"""Shared fixtures for VIF test suite."""

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


@pytest.fixture
def mock_text_encoder() -> MockTextEncoder:
    """Provide a MockTextEncoder instance for state encoder tests."""
    return MockTextEncoder()
