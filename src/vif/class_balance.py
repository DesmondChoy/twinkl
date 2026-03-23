"""Helpers for per-dimension ordinal class balance statistics.

These utilities centralize train-split class counts/priors for the
3-class ordinal VIF heads so training-time long-tail losses and post-hoc
logit adjustment use the same corrected-split statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

NUM_DIMS = 10
NUM_CLASSES = 3
NUM_BINARY_CLASSES = 2


def _validate_targets(targets: np.ndarray) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    if targets.ndim != 2:
        raise ValueError(f"targets must have shape (n_samples, {NUM_DIMS}), got {targets.shape}")
    if targets.shape[1] != NUM_DIMS:
        raise ValueError(f"targets second dimension must be {NUM_DIMS}, got {targets.shape[1]}")
    if np.any((targets < -1) | (targets > 1)):
        raise ValueError("targets must contain only ordinal labels in {-1, 0, 1}")
    return targets


def compute_ordinal_class_counts(targets: np.ndarray) -> np.ndarray:
    """Count ordinal labels per dimension in [-1, 0, +1] class order."""
    targets = _validate_targets(targets)
    class_indices = targets + 1

    counts = np.zeros((NUM_DIMS, NUM_CLASSES), dtype=np.float64)
    for dim_idx in range(NUM_DIMS):
        counts[dim_idx] = np.bincount(class_indices[:, dim_idx], minlength=NUM_CLASSES)
    return counts


def class_counts_to_priors(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert per-dimension class counts into strictly positive priors."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] != NUM_DIMS or counts.shape[1] <= 0:
        raise ValueError(
            "counts must have shape "
            f"({NUM_DIMS}, n_classes>0), got {counts.shape}"
        )

    stabilized = np.clip(counts, a_min=0.0, a_max=None) + float(eps)
    normalizer = stabilized.sum(axis=1, keepdims=True)
    return stabilized / normalizer


def compute_ordinal_class_priors(counts: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Backward-compatible alias for count -> prior conversion."""
    return class_counts_to_priors(counts, eps=eps)


def compute_activation_class_counts(targets: np.ndarray) -> np.ndarray:
    """Count binary activation labels per dimension in [inactive, active] order."""
    targets = _validate_targets(targets)
    inactive = (targets == 0).sum(axis=0, dtype=np.int64)
    active = (targets != 0).sum(axis=0, dtype=np.int64)
    return np.stack([inactive, active], axis=1).astype(np.float64)


def compute_polarity_class_counts(targets: np.ndarray) -> np.ndarray:
    """Count active-only polarity labels per dimension in [misaligned, aligned] order."""
    targets = _validate_targets(targets)
    misaligned = (targets == -1).sum(axis=0, dtype=np.int64)
    aligned = (targets == 1).sum(axis=0, dtype=np.int64)
    return np.stack([misaligned, aligned], axis=1).astype(np.float64)


def compute_ldam_margins(counts: np.ndarray, max_m: float = 0.5, eps: float = 1e-12) -> np.ndarray:
    """Compute LDAM margins per dimension/class.

    Uses the standard inverse fourth-root schedule and rescales each
    dimension so its largest class margin equals ``max_m``.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (NUM_DIMS, NUM_CLASSES):
        raise ValueError(f"counts must have shape ({NUM_DIMS}, {NUM_CLASSES}), got {counts.shape}")

    safe_counts = np.clip(counts, a_min=1.0, a_max=None)
    margins = safe_counts ** (-0.25)
    per_dim_max = np.maximum(margins.max(axis=1, keepdims=True), float(eps))
    return (float(max_m) * margins / per_dim_max).astype(np.float32)


def compute_effective_number_weights(
    counts: np.ndarray,
    beta: float = 0.9999,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute effective-number class weights with per-dimension mean 1."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (NUM_DIMS, NUM_CLASSES):
        raise ValueError(f"counts must have shape ({NUM_DIMS}, {NUM_CLASSES}), got {counts.shape}")

    beta = float(beta)
    safe_counts = np.clip(counts, a_min=1.0, a_max=None)
    effective_num = 1.0 - np.power(beta, safe_counts)
    weights = (1.0 - beta) / np.maximum(effective_num, float(eps))
    weights = weights / np.maximum(weights.mean(axis=1, keepdims=True), float(eps))
    return weights.astype(np.float32)


def compute_long_tail_statistics_from_dataframe(train_df: pl.DataFrame) -> dict[str, Any]:
    """Extract shared ordinal class statistics from the corrected train split."""
    if "alignment_vector" not in train_df.columns:
        raise ValueError("train_df must contain an 'alignment_vector' column")

    targets = np.asarray(train_df.get_column("alignment_vector").to_list(), dtype=np.int64)
    counts = compute_ordinal_class_counts(targets)
    priors = class_counts_to_priors(counts, eps=1e-12)
    activation_counts = compute_activation_class_counts(targets)
    activation_priors = class_counts_to_priors(activation_counts, eps=1e-12)
    polarity_counts = compute_polarity_class_counts(targets)
    polarity_priors = class_counts_to_priors(polarity_counts, eps=1e-12)
    return {
        "class_counts": counts.astype(np.float32),
        "class_priors": priors.astype(np.float32),
        "activation_counts": activation_counts.astype(np.float32),
        "activation_priors": activation_priors.astype(np.float32),
        "polarity_counts": polarity_counts.astype(np.float32),
        "polarity_priors": polarity_priors.astype(np.float32),
    }
