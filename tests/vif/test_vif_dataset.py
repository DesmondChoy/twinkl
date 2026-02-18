"""Tests for VIFDataset: persona indexing, getitem, sliding window, caching, and parity."""

import numpy as np
import polars as pl
import pytest
import torch

from src.vif.dataset import VIFDataset
from src.vif.state_encoder import StateEncoder

from .conftest import MockTextEncoder


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_merged_df(
    n_personas: int = 2,
    entries_per_persona: int = 4,
) -> pl.DataFrame:
    """Create a minimal merged DataFrame mimicking merge_labels_and_entries output.

    Each persona gets sequential dates and non-zero alignment vectors so
    history calculations are testable.
    """
    rows = []
    for p in range(n_personas):
        pid = f"persona_{p:03d}"
        for t in range(entries_per_persona):
            # Alignment vector: use t_index-dependent values for history testing
            alignment = [float(t) * 0.1] * 10
            rows.append({
                "persona_id": pid,
                "t_index": t,
                "date": f"2025-01-{t + 1:02d}",
                "initial_entry": f"Entry {t} for {pid}",
                "nudge_text": f"Nudge {t}" if t % 2 == 0 else None,
                "response_text": f"Response {t}" if t % 2 == 0 else None,
                "core_values": "achievement,benevolence",
                "alignment_vector": alignment,
            })
    return pl.DataFrame(rows)


def _make_encoder(window_size: int = 3) -> StateEncoder:
    """Create a StateEncoder with MockTextEncoder."""
    return StateEncoder(MockTextEncoder(), window_size=window_size)


# ── TestPersonaIndex ─────────────────────────────────────────────────────────


class TestPersonaIndex:
    """_build_persona_index() correctness."""

    def test_persona_entries_keys(self):
        """One key per persona in persona_entries."""
        df = _make_merged_df(n_personas=3, entries_per_persona=4)
        ds = VIFDataset(df, _make_encoder(), cache_embeddings=False)
        assert set(ds.persona_entries.keys()) == {
            "persona_000", "persona_001", "persona_002",
        }

    def test_entry_lookup_returns_correct_row(self):
        """entry_lookup[(pid, t)] returns the matching row dict."""
        df = _make_merged_df(n_personas=2, entries_per_persona=3)
        ds = VIFDataset(df, _make_encoder(), cache_embeddings=False)
        row = ds.entry_lookup[("persona_001", 2)]
        assert row["persona_id"] == "persona_001"
        assert row["t_index"] == 2
        assert row["initial_entry"] == "Entry 2 for persona_001"

    def test_index_map_length(self):
        """index_map length matches total entry count."""
        df = _make_merged_df(n_personas=2, entries_per_persona=5)
        ds = VIFDataset(df, _make_encoder(), cache_embeddings=False)
        assert len(ds.index_map) == 10

    def test_len_matches_index_map(self):
        """__len__() equals len(index_map)."""
        df = _make_merged_df(n_personas=3, entries_per_persona=4)
        ds = VIFDataset(df, _make_encoder(), cache_embeddings=False)
        assert len(ds) == len(ds.index_map) == 12


# ── TestGetItem ──────────────────────────────────────────────────────────────


class TestGetItem:
    """__getitem__() output shape, dtype, and content."""

    def test_returns_tuple(self):
        df = _make_merged_df()
        ds = VIFDataset(df, _make_encoder(), cache_embeddings=False)
        result = ds[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_state_tensor_shape(self):
        enc = _make_encoder(window_size=3)
        df = _make_merged_df()
        ds = VIFDataset(df, enc, cache_embeddings=False)
        state, _ = ds[0]
        assert state.shape == (enc.state_dim,)

    def test_target_tensor_shape(self):
        df = _make_merged_df()
        ds = VIFDataset(df, _make_encoder(), cache_embeddings=False)
        _, target = ds[0]
        assert target.shape == (10,)

    def test_dtype_float32(self):
        df = _make_merged_df()
        ds = VIFDataset(df, _make_encoder(), cache_embeddings=False)
        state, target = ds[0]
        assert state.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_first_entry_profile_weights(self):
        """t_index=0 → profile weights are present in state vector."""
        enc = _make_encoder(window_size=1)
        df = _make_merged_df(n_personas=1, entries_per_persona=3)
        ds = VIFDataset(df, enc, cache_embeddings=False)

        # First entry: t_index=0
        state, _ = ds[0]
        d_e = 8  # MockTextEncoder.embedding_dim
        # With window_size=1: state = [text(8)] + [gaps(0)] + [profile(10)]
        profile_start = d_e  # no gaps for window_size=1
        profile_part = state[profile_start : profile_start + 10].numpy()
        # core_values="achievement,benevolence" → indices 3 and 8 each get 0.5
        assert profile_part[3] == pytest.approx(0.5, abs=1e-6)
        assert profile_part[8] == pytest.approx(0.5, abs=1e-6)

    def test_target_matches_alignment_vector(self):
        """Target tensor matches the entry's alignment_vector."""
        df = _make_merged_df(n_personas=1, entries_per_persona=3)
        ds = VIFDataset(df, _make_encoder(), cache_embeddings=False)
        _, target = ds[2]  # t_index=2, alignment = [0.2]*10
        np.testing.assert_allclose(target.numpy(), np.full(10, 0.2), atol=1e-6)


# ── TestSlidingWindow ────────────────────────────────────────────────────────


class TestSlidingWindow:
    """Window assembly edge cases for different window sizes and positions."""

    def test_window_size_1_no_gaps(self):
        """window_size=1: only current entry, no time gap features."""
        enc = _make_encoder(window_size=1)
        df = _make_merged_df(n_personas=1, entries_per_persona=3)
        ds = VIFDataset(df, enc, cache_embeddings=False)
        state, _ = ds[1]
        # state_dim for W=1: 1*8 + 0 + 10 = 18
        assert state.shape == (18,)

    def test_window_size_3_at_t0_padded(self):
        """window_size=3 at t_index=0: 2 entries zero-padded."""
        enc = _make_encoder(window_size=3)
        df = _make_merged_df(n_personas=1, entries_per_persona=4)
        ds = VIFDataset(df, enc, cache_embeddings=False)
        state, _ = ds[0]  # t_index=0

        d_e = 8
        # First embedding (t=0): ones from MockTextEncoder
        first_emb = state[:d_e].numpy()
        np.testing.assert_allclose(first_emb, np.ones(d_e), atol=1e-6)

        # Second and third embeddings (t=-1, t=-2): zero-padded
        second_emb = state[d_e : 2 * d_e].numpy()
        third_emb = state[2 * d_e : 3 * d_e].numpy()
        np.testing.assert_allclose(second_emb, np.zeros(d_e), atol=1e-6)
        np.testing.assert_allclose(third_emb, np.zeros(d_e), atol=1e-6)

    def test_window_size_3_at_t2_full_window(self):
        """window_size=3 at t_index=2: full window, no padding."""
        enc = _make_encoder(window_size=3)
        df = _make_merged_df(n_personas=1, entries_per_persona=4)
        ds = VIFDataset(df, enc, cache_embeddings=False)
        state, _ = ds[2]  # t_index=2

        d_e = 8
        # All three embeddings should be ones (MockTextEncoder)
        for i in range(3):
            emb = state[i * d_e : (i + 1) * d_e].numpy()
            np.testing.assert_allclose(emb, np.ones(d_e), atol=1e-6)

    def test_state_dim_consistent(self):
        """State tensor dimension matches encoder.state_dim regardless of t_index."""
        enc = _make_encoder(window_size=3)
        df = _make_merged_df(n_personas=1, entries_per_persona=5)
        ds = VIFDataset(df, enc, cache_embeddings=False)
        for i in range(len(ds)):
            state, _ = ds[i]
            assert state.shape == (enc.state_dim,), f"Shape mismatch at index {i}"


# ── TestEmbeddingCaching ─────────────────────────────────────────────────────


class TestEmbeddingCaching:
    """Embedding cache behavior."""

    def test_cache_populated_on_init(self):
        """cache_embeddings=True populates embedding_cache."""
        df = _make_merged_df(n_personas=1, entries_per_persona=3)
        ds = VIFDataset(df, _make_encoder(), cache_embeddings=True)
        assert hasattr(ds, "embedding_cache")
        assert len(ds.embedding_cache) == 3

    def test_no_cache_when_disabled(self):
        """cache_embeddings=False does not create embedding_cache."""
        df = _make_merged_df(n_personas=1, entries_per_persona=3)
        ds = VIFDataset(df, _make_encoder(), cache_embeddings=False)
        assert not hasattr(ds, "embedding_cache")

    def test_cached_and_uncached_produce_same_output(self):
        """Both modes produce identical __getitem__ output."""
        df = _make_merged_df(n_personas=2, entries_per_persona=4)
        enc_kwargs = dict(window_size=3)

        ds_cached = VIFDataset(df, _make_encoder(**enc_kwargs), cache_embeddings=True)
        ds_uncached = VIFDataset(df, _make_encoder(**enc_kwargs), cache_embeddings=False)

        for i in range(len(ds_cached)):
            state_c, target_c = ds_cached[i]
            state_u, target_u = ds_uncached[i]
            torch.testing.assert_close(state_c, state_u)
            torch.testing.assert_close(target_c, target_u)


# ── TestTrainInferenceParity ─────────────────────────────────────────────────


class TestTrainInferenceParity:
    """Feature-availability contract: no train-only or inference-only features."""

    def test_state_vectors_identical_across_cache_modes(self):
        """State vectors from the same data are identical regardless of cache flag.

        This documents that the current architecture has no train-only or
        inference-only features — the same state construction path is used
        in both modes, differing only in whether embeddings are cached.
        """
        df = _make_merged_df(n_personas=2, entries_per_persona=5)

        ds_train = VIFDataset(df, _make_encoder(window_size=3), cache_embeddings=True)
        ds_infer = VIFDataset(df, _make_encoder(window_size=3), cache_embeddings=False)

        assert len(ds_train) == len(ds_infer)

        for i in range(len(ds_train)):
            state_t, target_t = ds_train[i]
            state_i, target_i = ds_infer[i]
            torch.testing.assert_close(
                state_t, state_i,
                msg=f"State mismatch at index {i}",
            )
            torch.testing.assert_close(
                target_t, target_i,
                msg=f"Target mismatch at index {i}",
            )
