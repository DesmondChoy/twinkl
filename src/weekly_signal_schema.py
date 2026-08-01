"""Shared schema for the weekly VIF-signal frame."""

from __future__ import annotations

import polars as pl

from src.models.judge import SCHWARTZ_VALUE_ORDER


def alignment_col(dim: str) -> str:
    return f"alignment_{dim}"


def uncertainty_col(dim: str) -> str:
    return f"uncertainty_{dim}"


def profile_weight_col(dim: str) -> str:
    return f"profile_weight_{dim}"


ALIGNMENT_COLUMNS = [alignment_col(dim) for dim in SCHWARTZ_VALUE_ORDER]
UNCERTAINTY_COLUMNS = [uncertainty_col(dim) for dim in SCHWARTZ_VALUE_ORDER]
PROFILE_WEIGHT_COLUMNS = [profile_weight_col(dim) for dim in SCHWARTZ_VALUE_ORDER]

PERSONA_ID = "persona_id"
PERSONA_NAME = "persona_name"
WEEK_START = "week_start"
WEEK_END = "week_end"
N_ENTRIES = "n_entries"
CORE_VALUES = "core_values"
ALIGNMENT_VECTOR = "alignment_vector"
UNCERTAINTY_VECTOR = "uncertainty_vector"
OVERALL_MEAN = "overall_mean"
OVERALL_UNCERTAINTY = "overall_uncertainty"

WEEKLY_SIGNAL_COLUMNS = [
    PERSONA_ID,
    PERSONA_NAME,
    WEEK_START,
    WEEK_END,
    N_ENTRIES,
    CORE_VALUES,
    ALIGNMENT_VECTOR,
    UNCERTAINTY_VECTOR,
    *ALIGNMENT_COLUMNS,
    *UNCERTAINTY_COLUMNS,
    *PROFILE_WEIGHT_COLUMNS,
    OVERALL_MEAN,
    OVERALL_UNCERTAINTY,
]

REQUIRED_WEEKLY_COLUMNS = [
    PERSONA_ID,
    WEEK_START,
    WEEK_END,
    OVERALL_MEAN,
    OVERALL_UNCERTAINTY,
    *ALIGNMENT_COLUMNS,
    *UNCERTAINTY_COLUMNS,
    *PROFILE_WEIGHT_COLUMNS,
]


def validate_weekly_frame(df: pl.DataFrame) -> None:
    """Reject a weekly frame that omits a required consumer column."""
    missing = [col for col in REQUIRED_WEEKLY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("weekly_df is missing required columns: " + ", ".join(missing))
