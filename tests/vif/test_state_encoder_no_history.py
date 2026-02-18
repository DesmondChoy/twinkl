"""Regression tests: EMA alignment history removed from state vector.

These tests verify that the train/serve skew fix (removing EMA alignment
history) is correctly implemented and that backward compatibility is
maintained for v1 notebooks via **kwargs absorption.
"""

import inspect

import numpy as np
import pytest

from src.vif.state_encoder import StateEncoder

from .conftest import MockTextEncoder


class TestNoAlignmentHistory:
    """Verify EMA alignment history is fully removed from the state vector."""

    def test_state_dim_excludes_ema(self, mock_text_encoder):
        """state_dim equals window_size * emb_dim + (window_size-1) + 10 (no EMA)."""
        for ws in [1, 3, 5]:
            enc = StateEncoder(mock_text_encoder, window_size=ws)
            d_e = mock_text_encoder.embedding_dim  # 8
            expected = ws * d_e + (ws - 1) + 10
            assert enc.state_dim == expected, (
                f"window_size={ws}: expected {expected}, got {enc.state_dim}"
            )

    def test_build_state_vector_no_alignment_history_param(self, mock_text_encoder):
        """build_state_vector() signature has no alignment_history parameter."""
        enc = StateEncoder(mock_text_encoder)
        sig = inspect.signature(enc.build_state_vector)
        param_names = list(sig.parameters.keys())
        assert "alignment_history" not in param_names, (
            f"alignment_history should be removed, but found in: {param_names}"
        )

    def test_state_vector_shape(self, mock_text_encoder):
        """Output shape matches state_dim for various window sizes."""
        for ws in [1, 3, 5]:
            enc = StateEncoder(mock_text_encoder, window_size=ws)
            texts = [f"Entry {i}" for i in range(ws)]
            dates = [f"2025-01-{i+1:02d}" for i in range(ws)]
            state = enc.build_state_vector(
                texts=texts,
                dates=dates,
                core_values=["Security"],
            )
            assert state.shape == (enc.state_dim,), (
                f"window_size={ws}: expected ({enc.state_dim},), got {state.shape}"
            )

    def test_unknown_kwargs_absorbed(self, mock_text_encoder):
        """StateEncoder(encoder, ema_alpha=0.3) does not raise (backward compat)."""
        # This simulates v1 notebooks passing ema_alpha
        enc = StateEncoder(mock_text_encoder, ema_alpha=0.3)
        assert enc.window_size == 3  # default
        assert not hasattr(enc, "ema_alpha")

    def test_unknown_kwargs_raise(self, mock_text_encoder):
        """Unknown kwargs still fail fast to avoid silent config mistakes."""
        with pytest.raises(TypeError, match="Unexpected StateEncoder kwargs"):
            StateEncoder(mock_text_encoder, window_sze=3)
