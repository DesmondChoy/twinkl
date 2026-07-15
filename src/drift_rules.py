"""Lightweight shared rules for Drift construction, matching, and coverage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

MATCH_SCHEMA = {
    "reference_episode_id": pl.String,
    "predicted_episode_id": pl.String,
    "persona_id": pl.String,
    "dimension": pl.String,
    "reference_confirmation_t_index": pl.Int64,
    "predicted_confirmation_t_index": pl.Int64,
    "latency_entries": pl.Int64,
    "reference_delivery_state": pl.String,
    "predicted_delivery_state": pl.String,
}


def drift_spans(
    labels: Sequence[bool | None], t_indices: Sequence[int]
) -> list[tuple[int, int, int]]:
    """Collapse consecutive Conflict labels into Drift spans.

    Each tuple contains onset, confirmation, and end ``t_index`` values.
    ``False``, ``None``, and gaps all break the current Conflict run.
    """
    if len(labels) != len(t_indices):
        raise ValueError("labels and t_indices must have the same length")

    spans: list[tuple[int, int, int]] = []
    run: list[int] = []

    def finish() -> None:
        if len(run) >= 2:
            spans.append((run[0], run[1], run[-1]))

    for label, t_index in zip(labels, t_indices, strict=True):
        adjacent = not run or int(t_index) == run[-1] + 1
        if label is True and adjacent:
            run.append(int(t_index))
            continue
        finish()
        run = [int(t_index)] if label is True else []
    finish()
    return spans


def trajectory_covered(labels: Sequence[bool | None], t_indices: Sequence[int]) -> bool:
    """Apply the Weekly Drift Reviewer trajectory coverage definition."""
    if len(labels) != len(t_indices):
        raise ValueError("labels and t_indices must have the same length")
    adjacent_pairs = [
        (first, second)
        for index, (first, second) in enumerate(zip(labels, labels[1:], strict=False))
        if int(t_indices[index + 1]) == int(t_indices[index]) + 1
    ]
    if any(first is True and second is True for first, second in adjacent_pairs):
        return True
    return bool(adjacent_pairs) and all(
        first is False or second is False for first, second in adjacent_pairs
    )


def _match_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=MATCH_SCHEMA)
    return pl.DataFrame(rows, schema=MATCH_SCHEMA, strict=False)


def match_episodes(
    reference_df: pl.DataFrame,
    predicted_df: pl.DataFrame,
    *,
    max_confirmation_lag: int = 2,
) -> pl.DataFrame:
    """Greedily match predictions from reference onset through the lag window."""
    if max_confirmation_lag < 0:
        raise ValueError("max_confirmation_lag must be non-negative")
    matches = []
    predicted_rows = predicted_df.sort(
        "persona_id", "dimension", "confirmation_t_index"
    ).to_dicts()
    used_prediction_ids: set[str] = set()
    for reference in reference_df.sort(
        "persona_id", "dimension", "confirmation_t_index"
    ).to_dicts():
        candidates = []
        for predicted in predicted_rows:
            if predicted["episode_id"] in used_prediction_ids:
                continue
            if (
                predicted["persona_id"] != reference["persona_id"]
                or predicted["dimension"] != reference["dimension"]
            ):
                continue
            predicted_confirmation = int(predicted["confirmation_t_index"])
            latency = predicted_confirmation - int(reference["confirmation_t_index"])
            latest_match = int(reference["end_t_index"]) + max_confirmation_lag
            if (
                int(reference["onset_t_index"])
                <= predicted_confirmation
                <= latest_match
            ):
                candidates.append(
                    (latency, int(predicted["confirmation_t_index"]), predicted)
                )
        if not candidates:
            continue
        latency, _confirmation, predicted = min(candidates, key=lambda item: item[:2])
        used_prediction_ids.add(str(predicted["episode_id"]))
        matches.append(
            {
                "reference_episode_id": str(reference["episode_id"]),
                "predicted_episode_id": str(predicted["episode_id"]),
                "persona_id": str(reference["persona_id"]),
                "dimension": str(reference["dimension"]),
                "reference_confirmation_t_index": int(
                    reference["confirmation_t_index"]
                ),
                "predicted_confirmation_t_index": int(
                    predicted["confirmation_t_index"]
                ),
                "latency_entries": int(latency),
                "reference_delivery_state": str(reference["delivery_state"]),
                "predicted_delivery_state": str(predicted["delivery_state"]),
            }
        )
    return _match_frame(matches)
