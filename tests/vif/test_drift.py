"""Tests for uncertainty-gated weekly drift detection."""

import polars as pl
import pytest

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.drift import detect_weekly_drift
from src.vif.weekly_schema import (
    ALIGNMENT_VECTOR,
    CORE_VALUES,
    N_ENTRIES,
    OVERALL_MEAN,
    OVERALL_UNCERTAINTY,
    PERSONA_ID,
    PERSONA_NAME,
    UNCERTAINTY_VECTOR,
    WEEK_END,
    WEEK_START,
    alignment_col,
    profile_weight_col,
    uncertainty_col,
)


def _weekly_row(
    *,
    week_start: str,
    week_end: str,
    overall_mean: float,
    overall_uncertainty: float = 0.1,
    achievement: float = 0.0,
    self_direction: float = 0.0,
    benevolence: float = 0.0,
    achievement_weight: float = 0.4,
    self_direction_weight: float = 0.3,
    benevolence_weight: float = 0.1,
) -> dict:
    row = {
        PERSONA_ID: "deadbeef",
        PERSONA_NAME: "Casey",
        WEEK_START: week_start,
        WEEK_END: week_end,
        N_ENTRIES: 2,
        CORE_VALUES: ["Achievement", "Self-Direction"],
        OVERALL_MEAN: overall_mean,
        OVERALL_UNCERTAINTY: overall_uncertainty,
    }
    for dim in SCHWARTZ_VALUE_ORDER:
        row[alignment_col(dim)] = 0.0
        row[uncertainty_col(dim)] = 0.1
        row[profile_weight_col(dim)] = 0.0
    row[alignment_col("achievement")] = achievement
    row[alignment_col("self_direction")] = self_direction
    row[alignment_col("benevolence")] = benevolence
    row[profile_weight_col("achievement")] = achievement_weight
    row[profile_weight_col("self_direction")] = self_direction_weight
    row[profile_weight_col("benevolence")] = benevolence_weight
    row[ALIGNMENT_VECTOR] = [row[alignment_col(d)] for d in SCHWARTZ_VALUE_ORDER]
    row[UNCERTAINTY_VECTOR] = [row[uncertainty_col(d)] for d in SCHWARTZ_VALUE_ORDER]
    return row


def test_detect_weekly_drift_returns_high_uncertainty_before_other_modes():
    weekly_df = pl.DataFrame(
        [
            _weekly_row(
                week_start="2025-01-06",
                week_end="2025-01-12",
                overall_mean=-0.2,
                overall_uncertainty=0.45,
                achievement=-0.9,
            )
        ]
    )

    result = detect_weekly_drift(
        weekly_df,
        uncertainty_threshold=0.3,
    )

    assert result.response_mode == "high_uncertainty"
    assert result.trigger_type == "high_uncertainty"
    assert "overall_uncertainty_above_threshold" in result.reasons


def test_detect_weekly_drift_returns_crash_on_large_week_over_week_drop():
    weekly_df = pl.DataFrame(
        [
            _weekly_row(
                week_start="2025-01-06",
                week_end="2025-01-12",
                overall_mean=0.45,
                achievement=0.6,
                self_direction=0.4,
            ),
            _weekly_row(
                week_start="2025-01-13",
                week_end="2025-01-19",
                overall_mean=-0.35,
                achievement=-0.6,
                self_direction=-0.3,
            ),
        ]
    )

    result = detect_weekly_drift(
        weekly_df,
        crash_delta=0.5,
        min_rut_weeks=3,
    )

    assert result.response_mode == "crash"
    assert result.trigger_type == "crash"
    assert "achievement" in result.triggered_dimensions


def test_detect_weekly_drift_returns_rut_for_consecutive_low_core_value_weeks():
    weekly_df = pl.DataFrame(
        [
            _weekly_row(
                week_start="2025-01-06",
                week_end="2025-01-12",
                overall_mean=-0.4,
                self_direction=-0.8,
            ),
            _weekly_row(
                week_start="2025-01-13",
                week_end="2025-01-19",
                overall_mean=-0.35,
                self_direction=-0.6,
            ),
            _weekly_row(
                week_start="2025-01-20",
                week_end="2025-01-26",
                overall_mean=-0.45,
                self_direction=-0.9,
            ),
        ]
    )

    result = detect_weekly_drift(
        weekly_df,
        min_rut_weeks=3,
        rut_threshold=-0.4,
        volatility_threshold=0.05,
    )

    assert result.response_mode == "rut"
    assert result.trigger_type == "rut"
    assert result.triggered_dimensions == ["self_direction"]


def test_detect_weekly_drift_routes_low_volatility_shift_to_evolution():
    weekly_df = pl.DataFrame(
        [
            _weekly_row(
                week_start="2025-01-06",
                week_end="2025-01-12",
                overall_mean=-0.2,
                achievement=-0.9,
                benevolence=0.8,
            ),
            _weekly_row(
                week_start="2025-01-13",
                week_end="2025-01-19",
                overall_mean=-0.15,
                achievement=-0.85,
                benevolence=0.85,
            ),
            _weekly_row(
                week_start="2025-01-20",
                week_end="2025-01-26",
                overall_mean=-0.1,
                achievement=-0.8,
                benevolence=0.9,
            ),
        ]
    )

    result = detect_weekly_drift(
        weekly_df,
        min_rut_weeks=3,
        residual_threshold=0.3,
        volatility_threshold=0.45,
    )

    assert result.response_mode == "evolution"
    assert result.trigger_type == "evolution"
    assert "achievement" in result.triggered_dimensions
    assert result.profile_update is not None


def test_detect_weekly_drift_raises_on_missing_required_column():
    row = _weekly_row(week_start="2025-01-06", week_end="2025-01-12", overall_mean=0.1)
    weekly_df = pl.DataFrame([row]).drop(alignment_col("achievement"))

    with pytest.raises(ValueError, match="alignment_achievement"):
        detect_weekly_drift(weekly_df)
