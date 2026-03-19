"""Tests for ordinal and two-stage class-balance helpers."""

import numpy as np
import polars as pl
import pytest

from src.vif.class_balance import (
    class_counts_to_priors,
    compute_activation_class_counts,
    compute_long_tail_statistics_from_dataframe,
    compute_ordinal_class_counts,
    compute_polarity_class_counts,
)


def _sample_targets() -> np.ndarray:
    return np.array(
        [
            [-1, 0, 1, 0, 1, -1, 0, 1, 0, -1],
            [0, 0, 1, 0, -1, 0, 0, 1, 0, 1],
            [1, -1, 0, 0, 0, 1, 0, 0, -1, 0],
        ],
        dtype=np.int64,
    )


def test_class_counts_to_priors_supports_binary_counts():
    counts = np.array([[8.0, 2.0]] * 10, dtype=np.float64)

    priors = class_counts_to_priors(counts)

    assert priors.shape == (10, 2)
    assert np.allclose(priors.sum(axis=1), 1.0)
    assert np.allclose(priors[:, 0], 0.8)
    assert np.allclose(priors[:, 1], 0.2)


def test_compute_activation_class_counts():
    targets = _sample_targets()

    counts = compute_activation_class_counts(targets)

    assert counts.shape == (10, 2)
    np.testing.assert_array_equal(counts[0], [1.0, 2.0])
    np.testing.assert_array_equal(counts[1], [2.0, 1.0])
    np.testing.assert_array_equal(counts[3], [3.0, 0.0])


def test_compute_polarity_class_counts():
    targets = _sample_targets()

    counts = compute_polarity_class_counts(targets)

    assert counts.shape == (10, 2)
    np.testing.assert_array_equal(counts[0], [1.0, 1.0])
    np.testing.assert_array_equal(counts[1], [1.0, 0.0])
    np.testing.assert_array_equal(counts[2], [0.0, 2.0])


def test_compute_long_tail_statistics_from_dataframe_includes_two_stage_stats():
    targets = _sample_targets()
    train_df = pl.DataFrame({"alignment_vector": targets.tolist()})

    stats = compute_long_tail_statistics_from_dataframe(train_df)

    assert set(stats) == {
        "class_counts",
        "class_priors",
        "activation_counts",
        "activation_priors",
        "polarity_counts",
        "polarity_priors",
    }
    assert stats["class_counts"].shape == (10, 3)
    assert stats["class_priors"].shape == (10, 3)
    assert stats["activation_counts"].shape == (10, 2)
    assert stats["activation_priors"].shape == (10, 2)
    assert stats["polarity_counts"].shape == (10, 2)
    assert stats["polarity_priors"].shape == (10, 2)
    assert np.allclose(stats["activation_priors"].sum(axis=1), 1.0)
    assert np.allclose(stats["polarity_priors"].sum(axis=1), 1.0)


def test_compute_long_tail_statistics_requires_alignment_vector_column():
    with pytest.raises(ValueError, match="alignment_vector"):
        compute_long_tail_statistics_from_dataframe(pl.DataFrame({"x": [1, 2, 3]}))


def test_activation_and_polarity_counts_match_ordinal_projection():
    targets = _sample_targets()

    ordinal_counts = compute_ordinal_class_counts(targets)
    activation_counts = compute_activation_class_counts(targets)
    polarity_counts = compute_polarity_class_counts(targets)

    np.testing.assert_allclose(activation_counts[:, 0], ordinal_counts[:, 1])
    np.testing.assert_allclose(
        activation_counts[:, 1],
        ordinal_counts[:, 0] + ordinal_counts[:, 2],
    )
    np.testing.assert_allclose(polarity_counts[:, 0], ordinal_counts[:, 0])
    np.testing.assert_allclose(polarity_counts[:, 1], ordinal_counts[:, 2])
