"""Tests for StateEncoder: state dimension, value parsing, time gaps, and assembly."""

import numpy as np
import pytest

from src.vif.state_encoder import StateEncoder

from .conftest import MockTextEncoder


# ── TestStateDim ─────────────────────────────────────────────────────────────


class TestStateDim:
    """state_dim property follows the formula: W*d_e + (W-1) + 10."""

    def test_window_size_1(self, mock_text_encoder):
        enc = StateEncoder(mock_text_encoder, window_size=1)
        # 1*8 + 0 + 10 = 18
        assert enc.state_dim == 18

    def test_window_size_3(self, mock_text_encoder):
        enc = StateEncoder(mock_text_encoder, window_size=3)
        # 3*8 + 2 + 10 = 36
        assert enc.state_dim == 36

    def test_window_size_5(self, mock_text_encoder):
        enc = StateEncoder(mock_text_encoder, window_size=5)
        # 5*8 + 4 + 10 = 54
        assert enc.state_dim == 54


# ── TestParseCoreValuesToWeights ─────────────────────────────────────────────


class TestParseCoreValuesToWeights:
    """Value-name normalization and weight assignment."""

    def test_title_case(self, mock_text_encoder):
        """Title-case names map via VALUE_NAME_TO_KEY."""
        enc = StateEncoder(mock_text_encoder)
        weights = enc.parse_core_values_to_weights(["Security", "Benevolence"])
        # security → index 5, benevolence → index 8
        assert weights[5] == pytest.approx(0.5)
        assert weights[8] == pytest.approx(0.5)
        # All others zero
        for i in range(10):
            if i not in (5, 8):
                assert weights[i] == 0.0

    def test_hyphenated(self, mock_text_encoder):
        """Hyphenated 'Self-Direction' maps to index 0."""
        enc = StateEncoder(mock_text_encoder)
        weights = enc.parse_core_values_to_weights(["Self-Direction"])
        assert weights[0] == pytest.approx(1.0)

    def test_case_variant(self, mock_text_encoder):
        """Lowercase-d 'Self-direction' also in VALUE_NAME_TO_KEY."""
        enc = StateEncoder(mock_text_encoder)
        weights = enc.parse_core_values_to_weights(["Self-direction"])
        assert weights[0] == pytest.approx(1.0)

    def test_snake_case_fallback(self, mock_text_encoder):
        """Snake-case 'self_direction' matched via SCHWARTZ_VALUE_ORDER fallback."""
        enc = StateEncoder(mock_text_encoder)
        weights = enc.parse_core_values_to_weights(["self_direction"])
        assert weights[0] == pytest.approx(1.0)

    def test_sum_to_one(self, mock_text_encoder):
        """Weights always sum to 1.0 for any valid input."""
        enc = StateEncoder(mock_text_encoder)
        for values in [
            ["Security"],
            ["Achievement", "Power", "Universalism"],
            ["Self-Direction", "Stimulation", "Hedonism", "Achievement", "Power"],
        ]:
            weights = enc.parse_core_values_to_weights(values)
            assert weights.sum() == pytest.approx(1.0)

    def test_empty_list_uniform(self, mock_text_encoder):
        """Empty list → uniform weights (0.1 each)."""
        enc = StateEncoder(mock_text_encoder)
        weights = enc.parse_core_values_to_weights([])
        np.testing.assert_allclose(weights, np.full(10, 0.1), atol=1e-6)

    def test_all_unrecognized_uniform(self, mock_text_encoder):
        """All-unrecognized values → uniform fallback."""
        enc = StateEncoder(mock_text_encoder)
        weights = enc.parse_core_values_to_weights(["Nonsense", "FakeValue"])
        np.testing.assert_allclose(weights, np.full(10, 0.1), atol=1e-6)

    def test_unrecognized_values_skipped(self, mock_text_encoder):
        """Unrecognized values are skipped; recognized ones still weighted."""
        enc = StateEncoder(mock_text_encoder)
        weights = enc.parse_core_values_to_weights(["Security", "FakeValue"])
        assert weights[5] == pytest.approx(1.0)
        assert weights.sum() == pytest.approx(1.0)

    def test_single_value(self, mock_text_encoder):
        """Single recognized value gets 1.0."""
        enc = StateEncoder(mock_text_encoder)
        weights = enc.parse_core_values_to_weights(["Tradition"])
        assert weights[7] == pytest.approx(1.0)


# ── TestComputeTimeGaps ──────────────────────────────────────────────────────


class TestComputeTimeGaps:
    """Temporal feature computation."""

    def test_consecutive_days(self, mock_text_encoder):
        """1-day gap between consecutive dates → 1/30."""
        enc = StateEncoder(mock_text_encoder, window_size=2)
        # Dates ordered: current first, then older
        gaps = enc.compute_time_gaps(["2025-01-02", "2025-01-01"])
        assert gaps.shape == (1,)
        assert gaps[0] == pytest.approx(1.0 / 30.0)

    def test_large_gap_clamped(self, mock_text_encoder):
        """Gap > 30 days → clamped to 30/30 = 1.0."""
        enc = StateEncoder(mock_text_encoder, window_size=2)
        gaps = enc.compute_time_gaps(["2025-03-01", "2025-01-01"])
        assert gaps[0] == pytest.approx(1.0)

    def test_none_dates_default(self, mock_text_encoder):
        """None dates → default gap of 7/30."""
        enc = StateEncoder(mock_text_encoder, window_size=2)
        gaps = enc.compute_time_gaps([None, None])
        assert gaps[0] == pytest.approx(7.0 / 30.0)

    def test_invalid_date_string(self, mock_text_encoder):
        """Invalid date string → default gap of 7/30."""
        enc = StateEncoder(mock_text_encoder, window_size=2)
        gaps = enc.compute_time_gaps(["not-a-date", "also-bad"])
        assert gaps[0] == pytest.approx(7.0 / 30.0)

    def test_window_size_1_empty(self, mock_text_encoder):
        """window_size=1 → 0 gaps (empty array)."""
        enc = StateEncoder(mock_text_encoder, window_size=1)
        gaps = enc.compute_time_gaps(["2025-01-01"])
        assert gaps.shape == (0,)

    def test_padding_fewer_dates(self, mock_text_encoder):
        """Fewer dates than window_size → padded with default 7/30."""
        enc = StateEncoder(mock_text_encoder, window_size=3)
        # Only 1 date provided, but window_size needs 2 gaps
        gaps = enc.compute_time_gaps(["2025-01-01"])
        assert gaps.shape == (2,)
        np.testing.assert_allclose(gaps, [7.0 / 30.0, 7.0 / 30.0], atol=1e-6)

    def test_multi_gap_values(self, mock_text_encoder):
        """Multiple gaps with varying intervals."""
        enc = StateEncoder(mock_text_encoder, window_size=3)
        # Current: Jan 10, prev: Jan 7 (3 days), oldest: Jan 1 (6 days)
        gaps = enc.compute_time_gaps(["2025-01-10", "2025-01-07", "2025-01-01"])
        assert gaps.shape == (2,)
        assert gaps[0] == pytest.approx(3.0 / 30.0)
        assert gaps[1] == pytest.approx(6.0 / 30.0)


# ── TestBuildStateVector ─────────────────────────────────────────────────────


class TestBuildStateVector:
    """Full state vector assembly."""

    def test_output_shape_window_1(self, mock_text_encoder):
        enc = StateEncoder(mock_text_encoder, window_size=1)
        state = enc.build_state_vector(
            texts=["Hello"],
            dates=["2025-01-01"],
            core_values=["Security"],
        )
        assert state.shape == (enc.state_dim,)
        assert state.dtype == np.float32

    def test_output_shape_window_3(self, mock_text_encoder):
        enc = StateEncoder(mock_text_encoder, window_size=3)
        state = enc.build_state_vector(
            texts=["Entry 3", "Entry 2", "Entry 1"],
            dates=["2025-01-03", "2025-01-02", "2025-01-01"],
            core_values=["Achievement", "Benevolence"],
        )
        assert state.shape == (enc.state_dim,)

    def test_concatenation_order(self, mock_text_encoder):
        """Verify: text_window | time_gaps | profile_weights."""
        enc = StateEncoder(mock_text_encoder, window_size=2)
        state = enc.build_state_vector(
            texts=["Current", "Previous"],
            dates=["2025-01-02", "2025-01-01"],
            core_values=["Security"],
        )
        d_e = mock_text_encoder.embedding_dim  # 8

        # text_window: 2 * 8 = 16 (all ones from MockTextEncoder)
        text_part = state[:2 * d_e]
        np.testing.assert_allclose(text_part, np.ones(2 * d_e), atol=1e-6)

        # time_gaps: 1 gap (1 day → 1/30)
        gap_part = state[2 * d_e : 2 * d_e + 1]
        assert gap_part[0] == pytest.approx(1.0 / 30.0)

        # profile_weights: Security at index 5 → 1.0
        profile_part = state[2 * d_e + 1:]
        assert len(profile_part) == 10
        assert profile_part[5] == pytest.approx(1.0)

    def test_build_state_vector_matches_from_embeddings(self, mock_text_encoder):
        """build_state_vector and build_state_vector_from_embeddings produce identical output."""
        enc = StateEncoder(mock_text_encoder, window_size=2)
        texts = ["Current entry", "Previous entry"]
        dates = ["2025-01-02", "2025-01-01"]
        core_values = ["Achievement"]

        state_from_text = enc.build_state_vector(
            texts=texts, dates=dates,
            core_values=core_values,
        )

        # Pre-compute embeddings the same way build_state_vector does
        embeddings = [enc.text_encoder.encode_batch([t])[0] for t in texts]
        state_from_emb = enc.build_state_vector_from_embeddings(
            embeddings=embeddings, dates=dates,
            core_values=core_values,
        )

        np.testing.assert_array_equal(state_from_text, state_from_emb)


# ── TestConcatenateEntryText ─────────────────────────────────────────────────


class TestConcatenateEntryText:
    """Text concatenation for embedding input."""

    def test_all_three_parts(self, mock_text_encoder):
        enc = StateEncoder(mock_text_encoder)
        result = enc.concatenate_entry_text(
            initial_entry="I went for a walk.",
            nudge_text="What did you notice?",
            response_text="The trees were beautiful.",
        )
        assert "I went for a walk." in result
        assert 'Nudge: "What did you notice?"' in result
        assert "Response: The trees were beautiful." in result
        # Parts joined by double newline
        parts = result.split("\n\n")
        assert len(parts) == 3

    def test_only_initial_entry(self, mock_text_encoder):
        enc = StateEncoder(mock_text_encoder)
        result = enc.concatenate_entry_text(
            initial_entry="Just a journal entry.",
            nudge_text=None,
            response_text=None,
        )
        assert result == "Just a journal entry."

    def test_all_none(self, mock_text_encoder):
        enc = StateEncoder(mock_text_encoder)
        result = enc.concatenate_entry_text(None, None, None)
        assert result == ""

    def test_nudge_format(self, mock_text_encoder):
        """Nudge is quoted with 'Nudge: "..."' format."""
        enc = StateEncoder(mock_text_encoder)
        result = enc.concatenate_entry_text(None, "How are you?", None)
        assert result == 'Nudge: "How are you?"'

    def test_response_format(self, mock_text_encoder):
        """Response is prefixed with 'Response: ...' (no quotes)."""
        enc = StateEncoder(mock_text_encoder)
        result = enc.concatenate_entry_text(None, None, "I feel great.")
        assert result == "Response: I feel great."
