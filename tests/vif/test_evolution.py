"""Tests for deterministic evolution detection on weekly VIF signals."""

import polars as pl

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.evolution import classify_weekly_evolution


def _weekly_row(
    *,
    week_start: str,
    week_end: str,
    achievement: float = 0.0,
    benevolence: float = 0.0,
    self_direction: float = 0.0,
    achievement_weight: float = 0.4,
    benevolence_weight: float = 0.1,
    self_direction_weight: float = 0.3,
) -> dict:
    row = {
        "persona_id": "deadbeef",
        "persona_name": "Casey",
        "week_start": week_start,
        "week_end": week_end,
        "n_entries": 2,
        "core_values": ["Achievement", "Self-Direction"],
        "overall_mean": 0.0,
        "overall_uncertainty": 0.1,
    }
    for dim in SCHWARTZ_VALUE_ORDER:
        row[f"alignment_{dim}"] = 0.0
        row[f"uncertainty_{dim}"] = 0.1
        row[f"profile_weight_{dim}"] = 0.0
    row["alignment_achievement"] = achievement
    row["alignment_benevolence"] = benevolence
    row["alignment_self_direction"] = self_direction
    row["profile_weight_achievement"] = achievement_weight
    row["profile_weight_benevolence"] = benevolence_weight
    row["profile_weight_self_direction"] = self_direction_weight
    row["alignment_vector"] = [row[f"alignment_{dim}"] for dim in SCHWARTZ_VALUE_ORDER]
    row["uncertainty_vector"] = [row[f"uncertainty_{dim}"] for dim in SCHWARTZ_VALUE_ORDER]
    return row


def test_classify_weekly_evolution_distinguishes_stable_evolution_and_drift():
    weekly_df = pl.DataFrame(
        [
            _weekly_row(
                week_start="2025-01-06",
                week_end="2025-01-12",
                achievement=-0.9,
                benevolence=0.8,
                self_direction=-1.0,
            ),
            _weekly_row(
                week_start="2025-01-13",
                week_end="2025-01-19",
                achievement=-0.8,
                benevolence=0.9,
                self_direction=1.0,
            ),
            _weekly_row(
                week_start="2025-01-20",
                week_end="2025-01-26",
                achievement=-0.85,
                benevolence=0.85,
                self_direction=-1.0,
            ),
        ]
    )

    result = classify_weekly_evolution(
        weekly_df,
        min_weeks=3,
        residual_threshold=0.3,
        volatility_threshold=0.45,
        blend_rate=0.4,
    )

    by_dim = {row.dimension: row for row in result.dimensions}

    assert by_dim["achievement"].classification == "evolution"
    assert by_dim["benevolence"].classification == "evolution"
    assert by_dim["self_direction"].classification == "drift"
    assert by_dim["achievement"].direction == "down"
    assert by_dim["benevolence"].direction == "up"
    assert by_dim["self_direction"].volatility > 0.9

    assert result.suggested_profile is not None
    assert set(result.suggested_profile.evolved_dimensions) == {
        "achievement",
        "benevolence",
    }
    assert (
        result.suggested_profile.suggested_profile["achievement"]
        < result.suggested_profile.original_profile["achievement"]
    )
    assert (
        result.suggested_profile.suggested_profile["benevolence"]
        > result.suggested_profile.original_profile["benevolence"]
    )


def test_classify_weekly_evolution_requires_enough_history_before_relabeling():
    weekly_df = pl.DataFrame(
        [
            _weekly_row(
                week_start="2025-01-06",
                week_end="2025-01-12",
                achievement=-1.0,
                self_direction=-1.0,
            ),
            _weekly_row(
                week_start="2025-01-13",
                week_end="2025-01-19",
                achievement=-1.0,
                self_direction=1.0,
            ),
        ]
    )

    result = classify_weekly_evolution(
        weekly_df,
        min_weeks=3,
        residual_threshold=0.3,
        volatility_threshold=0.45,
    )

    assert result.n_weeks_observed == 2
    assert all(row.classification == "stable" for row in result.dimensions)
    assert result.suggested_profile is None
