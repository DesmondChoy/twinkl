"""Deterministic value-evolution classification on top of weekly VIF signals."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, Field

from src.models.judge import SCHWARTZ_VALUE_ORDER

EvolutionClassification = Literal["stable", "evolution", "drift"]
EvolutionDirection = Literal["up", "down", "flat"]


class DimensionEvolutionSignal(BaseModel):
    """Per-dimension evolution summary for one analysis window."""

    dimension: str
    weight: float = Field(ge=0.0, le=1.0)
    expected_alignment: float
    mean_alignment: float
    volatility: float = Field(ge=0.0)
    residual: float
    classification: EvolutionClassification
    direction: EvolutionDirection


class ProfileUpdateSuggestion(BaseModel):
    """Suggested user-profile update when evolution is detected."""

    blend_rate: float = Field(ge=0.0, le=1.0)
    original_profile: dict[str, float]
    suggested_profile: dict[str, float]
    evolved_dimensions: list[str] = Field(default_factory=list)


class EvolutionDetectionResult(BaseModel):
    """Evolution-filter result for one persona/week window."""

    persona_id: str
    week_start: str
    week_end: str
    n_weeks_observed: int = Field(ge=1)
    min_weeks_required: int = Field(ge=1)
    dimensions: list[DimensionEvolutionSignal]
    suggested_profile: ProfileUpdateSuggestion | None = None


def _expected_alignment_from_weights(
    weights: np.ndarray,
    *,
    importance_floor: float,
) -> np.ndarray:
    """Map declared profile weights to expected alignment in [0, 1]."""
    expected = np.maximum(weights - importance_floor, 0.0)
    denom = max(1e-6, 1.0 - importance_floor)
    expected = expected / denom
    return np.clip(expected, 0.0, 1.0)


def _normalize_profile(weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0:
        return np.full_like(weights, 1.0 / len(weights), dtype=np.float64)
    return weights / total


def _direction_from_mean(mean_alignment: float, *, eps: float = 1e-6) -> EvolutionDirection:
    if mean_alignment > eps:
        return "up"
    if mean_alignment < -eps:
        return "down"
    return "flat"


def _build_profile_suggestion(
    dimension_rows: list[DimensionEvolutionSignal],
    *,
    blend_rate: float,
) -> ProfileUpdateSuggestion | None:
    evolved = [row for row in dimension_rows if row.classification == "evolution"]
    if not evolved:
        return None

    original = np.asarray([row.weight for row in dimension_rows], dtype=np.float64)
    behavioral = original.copy()
    for row in evolved:
        idx = SCHWARTZ_VALUE_ORDER.index(row.dimension)
        behavioral[idx] = (row.mean_alignment + 1.0) / 2.0

    normalized_behavioral = _normalize_profile(behavioral)
    blended = ((1.0 - blend_rate) * original) + (blend_rate * normalized_behavioral)
    suggested = _normalize_profile(blended)

    return ProfileUpdateSuggestion(
        blend_rate=blend_rate,
        original_profile={
            dim: float(value) for dim, value in zip(SCHWARTZ_VALUE_ORDER, original.tolist())
        },
        suggested_profile={
            dim: float(value) for dim, value in zip(SCHWARTZ_VALUE_ORDER, suggested.tolist())
        },
        evolved_dimensions=[row.dimension for row in evolved],
    )


def classify_weekly_evolution(
    weekly_df: pl.DataFrame,
    *,
    target_week_end: str | None = None,
    lookback_weeks: int = 6,
    min_weeks: int = 3,
    residual_threshold: float = 0.4,
    volatility_threshold: float = 0.5,
    blend_rate: float = 0.3,
    importance_floor: float = 0.15,
) -> EvolutionDetectionResult:
    """Classify each value dimension as stable, evolution, or drift."""
    if weekly_df.is_empty():
        raise ValueError("weekly_df must contain at least one weekly signal row")

    persona_ids = weekly_df["persona_id"].unique().to_list()
    if len(persona_ids) != 1:
        raise ValueError(
            f"classify_weekly_evolution expects one persona, found {len(persona_ids)}"
        )

    ordered = weekly_df.sort("week_end")
    if target_week_end is not None:
        ordered = ordered.filter(pl.col("week_end") <= target_week_end)
    if ordered.is_empty():
        raise ValueError("No weekly rows remain after applying target_week_end filter")

    window = ordered.tail(lookback_weeks)
    first_row = window.row(0, named=True)
    last_row = window.row(-1, named=True)
    n_weeks_observed = window.height

    weights = np.asarray(
        [float(first_row[f"profile_weight_{dim}"]) for dim in SCHWARTZ_VALUE_ORDER],
        dtype=np.float64,
    )
    expected = _expected_alignment_from_weights(
        weights,
        importance_floor=importance_floor,
    )

    dimension_rows: list[DimensionEvolutionSignal] = []
    enough_history = n_weeks_observed >= min_weeks
    for idx, dim in enumerate(SCHWARTZ_VALUE_ORDER):
        series = np.asarray(window[f"alignment_{dim}"].to_list(), dtype=np.float64)
        mean_alignment = float(series.mean())
        volatility = float(series.std(ddof=0))
        residual = mean_alignment - float(expected[idx])

        if not enough_history or abs(residual) < residual_threshold:
            classification: EvolutionClassification = "stable"
        elif volatility < volatility_threshold:
            classification = "evolution"
        else:
            classification = "drift"

        dimension_rows.append(
            DimensionEvolutionSignal(
                dimension=dim,
                weight=float(weights[idx]),
                expected_alignment=float(expected[idx]),
                mean_alignment=mean_alignment,
                volatility=volatility,
                residual=float(residual),
                classification=classification,
                direction=_direction_from_mean(mean_alignment),
            )
        )

    suggestion = None
    if enough_history:
        suggestion = _build_profile_suggestion(
            dimension_rows,
            blend_rate=blend_rate,
        )

    return EvolutionDetectionResult(
        persona_id=str(first_row["persona_id"]),
        week_start=str(first_row["week_start"]),
        week_end=str(last_row["week_end"]),
        n_weeks_observed=n_weeks_observed,
        min_weeks_required=min_weeks,
        dimensions=dimension_rows,
        suggested_profile=suggestion,
    )
