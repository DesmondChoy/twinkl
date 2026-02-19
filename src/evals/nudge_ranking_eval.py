"""Ranking metrics for nudge-relevance evaluation.

This script replaces the old notebook placeholder for ranking metrics with
importable, testable Python functions.
"""

from __future__ import annotations

import math


def hit_at_k(relevances: list[int], k: int) -> float:
    """Return 1.0 if any relevant item appears in the top-k list, else 0.0."""
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = relevances[:k]
    return 1.0 if any(rel > 0 for rel in top_k) else 0.0


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Compute normalized discounted cumulative gain at rank k."""
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = relevances[:k]
    if not top_k:
        return 0.0

    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(top_k))

    ideal = sorted(relevances, reverse=True)[:k]
    if not ideal:
        return 0.0
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal))

    if idcg == 0:
        return 0.0
    return dcg / idcg
