"""Tests for nudge ranking metrics."""

from __future__ import annotations

import pytest

from src.evals.nudge_ranking_eval import hit_at_k, ndcg_at_k


def test_hit_at_k():
    assert hit_at_k([0, 0, 1], 1) == 0.0
    assert hit_at_k([0, 1, 0], 2) == 1.0


def test_ndcg_at_k_basic():
    score = ndcg_at_k([3, 2, 1], 3)
    assert score == 1.0

    lower_score = ndcg_at_k([1, 3, 2], 3)
    assert 0.0 < lower_score < 1.0


def test_metric_k_validation():
    with pytest.raises(ValueError):
        hit_at_k([1], 0)
    with pytest.raises(ValueError):
        ndcg_at_k([1.0], 0)
