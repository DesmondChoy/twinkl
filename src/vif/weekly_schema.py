"""Shared schema for the weekly VIF-signal frame.

This module is the single source of truth for the column contract between
``aggregate_timeline_by_week`` (producer, ``src/vif/runtime.py``) and
``detect_weekly_drift`` (consumer, ``src/vif/drift.py``). Centralizing the
column vocabulary, dimension accessors, and a presence guard here keeps the
two sides from independently re-deriving the same string keys.

The wire format stays a Polars DataFrame; this schema governs column names
and presence, which is where the contract is actually fragile.
"""

from __future__ import annotations

import polars as pl

from src.models.judge import SCHWARTZ_VALUE_ORDER

# ── Per-dimension column accessors ───────────────────────────────────────────

def alignment_col(dim: str) -> str:
    return f"alignment_{dim}"


def uncertainty_col(dim: str) -> str:
    return f"uncertainty_{dim}"


def profile_weight_col(dim: str) -> str:
    return f"profile_weight_{dim}"


ALIGNMENT_COLUMNS = [alignment_col(dim) for dim in SCHWARTZ_VALUE_ORDER]
UNCERTAINTY_COLUMNS = [uncertainty_col(dim) for dim in SCHWARTZ_VALUE_ORDER]
PROFILE_WEIGHT_COLUMNS = [profile_weight_col(dim) for dim in SCHWARTZ_VALUE_ORDER]


# ── Scalar / list column names ───────────────────────────────────────────────

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


# Ordered full column list for the weekly frame. The producer selects in this
# order so the layout is defined exactly once.
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


# Columns required for drift/coach consumers to operate. The vector columns
# are convenience denormalizations, so they are not part of the hard contract.
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
    """Assert the weekly frame carries every column drift/coach consumers need.

    Raises ``ValueError`` naming the missing columns so a schema mismatch fails
    at the boundary rather than as a late ``KeyError`` deep inside drift logic.
    """
    missing = [col for col in REQUIRED_WEEKLY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "weekly_df is missing required columns: " + ", ".join(missing)
        )

