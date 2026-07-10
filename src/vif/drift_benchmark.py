"""Decision-level benchmark helpers for sustained-conflict drift v1.

This module keeps the strict label-side reference separate from model evidence.
Reference episodes come from stored consensus labels. Predicted episodes come
from per-entry ``P(-1)`` plus uncertainty, or from hard LLM class outputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean, median
from typing import Any

import polars as pl

from src.models.judge import SCHWARTZ_VALUE_ORDER

EPISODE_SCHEMA = {
    "episode_id": pl.String,
    "source": pl.String,
    "persona_id": pl.String,
    "dimension": pl.String,
    "onset_t_index": pl.Int64,
    "confirmation_t_index": pl.Int64,
    "end_t_index": pl.Int64,
    "onset_date": pl.String,
    "confirmation_date": pl.String,
    "end_date": pl.String,
    "supporting_t_indices": pl.List(pl.Int64),
    "supporting_dates": pl.List(pl.String),
    "length": pl.Int64,
    "open_at_cutoff": pl.Boolean,
    "delivery_state": pl.String,
    "termination_t_index": pl.Int64,
    "termination_date": pl.String,
    "termination_label": pl.Int64,
    "mean_conflict_evidence": pl.Float64,
    "mean_uncertainty": pl.Float64,
    "confidence_metadata": pl.List(pl.String),
    "agreement_metadata": pl.List(pl.Float64),
}

DECISION_SCHEMA = {
    "source": pl.String,
    "persona_id": pl.String,
    "dimension": pl.String,
    "window_start_t_index": pl.Int64,
    "window_end_t_index": pl.Int64,
    "window_start_date": pl.String,
    "window_end_date": pl.String,
    "pair_probability": pl.Float64,
    "pair_uncertainty": pl.Float64,
    "probability_passed": pl.Boolean,
    "uncertainty_passed": pl.Boolean,
    "alert": pl.Boolean,
    "suppression_reason": pl.String,
}

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


def normalize_value_name(value: str) -> str:
    """Normalize display names to the canonical Schwartz dimension key."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _core_value_map(profiles_df: pl.DataFrame) -> dict[str, tuple[str, ...]]:
    required = {"persona_id", "core_values"}
    missing = required - set(profiles_df.columns)
    if missing:
        raise ValueError(
            "profiles_df is missing required columns: " + ", ".join(sorted(missing))
        )

    result: dict[str, tuple[str, ...]] = {}
    for row in profiles_df.select("persona_id", "core_values").to_dicts():
        raw_values = row["core_values"] or []
        if isinstance(raw_values, str):
            raw_values = [part.strip() for part in raw_values.split(",")]
        normalized_values = [
            normalize_value_name(str(raw)) for raw in raw_values if str(raw).strip()
        ]
        invalid = [
            value for value in normalized_values if value not in SCHWARTZ_VALUE_ORDER
        ]
        if invalid:
            raise ValueError(
                f"persona {row['persona_id']} has invalid core value(s): "
                + ", ".join(invalid)
            )
        values = tuple(dict.fromkeys(normalized_values))
        result[str(row["persona_id"])] = values
    return result


def _frame(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, strict=False)


def _validate_timeline_keys(df: pl.DataFrame) -> None:
    required = {"persona_id", "t_index", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "timeline is missing required columns: " + ", ".join(sorted(missing))
        )
    duplicate_count = (
        df.group_by("persona_id", "t_index").len().filter(pl.col("len") > 1).height
    )
    if duplicate_count:
        raise ValueError(
            "timeline contains duplicate (persona_id, t_index) rows: "
            f"{duplicate_count} duplicate key(s)"
        )


def _validate_evidence_keys(df: pl.DataFrame) -> None:
    required = {
        "source",
        "persona_id",
        "dimension",
        "t_index",
        "date",
        "p_minus1",
        "uncertainty",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "evidence is missing required columns: " + ", ".join(sorted(missing))
        )
    duplicate_count = (
        df.group_by("source", "persona_id", "dimension", "t_index")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError(
            "evidence contains duplicate (source, persona_id, dimension, t_index) "
            f"rows: {duplicate_count} duplicate key(s)"
        )


def build_eligible_trajectories(
    timeline_df: pl.DataFrame,
    profiles_df: pl.DataFrame,
) -> pl.DataFrame:
    """Return one eligibility row per observed persona and declared core value."""
    _validate_timeline_keys(timeline_df)
    core_map = _core_value_map(profiles_df)
    rows = []
    for persona_id, count in (
        timeline_df.group_by("persona_id").len().select("persona_id", "len").iter_rows()
    ):
        for dimension in core_map.get(str(persona_id), ()):
            rows.append(
                {
                    "persona_id": str(persona_id),
                    "dimension": dimension,
                    "n_entries": int(count),
                }
            )
    return _frame(
        rows,
        {"persona_id": pl.String, "dimension": pl.String, "n_entries": pl.Int64},
    ).sort("persona_id", "dimension")


def build_reference_episodes(
    labels_df: pl.DataFrame,
    profiles_df: pl.DataFrame,
    *,
    source: str = "five_pass_consensus",
) -> pl.DataFrame:
    """Materialize strict sustained-conflict episodes from stored labels.

    Each declared core value is scanned independently. Two adjacent observed
    ``-1`` labels confirm an episode. Further adjacent ``-1`` labels extend the
    same episode. Any other or missing label breaks the run.
    """
    _validate_timeline_keys(labels_df)
    core_map = _core_value_map(profiles_df)
    rows: list[dict[str, Any]] = []

    for persona_key, timeline in (
        labels_df.sort("persona_id", "t_index", "date")
        .partition_by("persona_id", as_dict=True)
        .items()
    ):
        persona_id = str(
            persona_key[0] if isinstance(persona_key, tuple) else persona_key
        )
        entry_rows = timeline.to_dicts()
        for dimension in core_map.get(persona_id, ()):
            label_col = f"alignment_{dimension}"
            if label_col not in labels_df.columns:
                raise ValueError(f"labels_df is missing required column: {label_col}")
            confidence_col = f"confidence_{dimension}"
            agreement_col = f"consensus_agreement_{dimension}"
            index = 0
            while index < len(entry_rows):
                if entry_rows[index].get(label_col) != -1:
                    index += 1
                    continue
                run_start = index
                while (
                    index < len(entry_rows) and entry_rows[index].get(label_col) == -1
                ):
                    if (
                        index > run_start
                        and int(entry_rows[index]["t_index"])
                        != int(entry_rows[index - 1]["t_index"]) + 1
                    ):
                        break
                    index += 1
                run = entry_rows[run_start:index]
                if len(run) < 2:
                    continue

                next_row = entry_rows[index] if index < len(entry_rows) else None
                gap_after_run = (
                    next_row is not None
                    and int(next_row["t_index"]) != int(run[-1]["t_index"]) + 1
                )
                if next_row is None:
                    delivery_state = "active"
                elif (
                    gap_after_run
                    or next_row.get(label_col) is None
                    or str(next_row.get(confidence_col) or "").lower() == "no_majority"
                ):
                    delivery_state = "uncertain"
                else:
                    delivery_state = "recovered"

                confidence_values = [
                    str(row.get(confidence_col))
                    for row in run
                    if row.get(confidence_col) is not None
                ]
                agreement_values = [
                    float(row[agreement_col])
                    for row in run
                    if row.get(agreement_col) is not None
                ]
                onset = run[0]
                confirmation = run[1]
                end = run[-1]
                rows.append(
                    {
                        "episode_id": (
                            f"{source}::{persona_id}::{dimension}::"
                            f"{int(onset['t_index'])}"
                        ),
                        "source": source,
                        "persona_id": persona_id,
                        "dimension": dimension,
                        "onset_t_index": int(onset["t_index"]),
                        "confirmation_t_index": int(confirmation["t_index"]),
                        "end_t_index": int(end["t_index"]),
                        "onset_date": str(onset["date"]),
                        "confirmation_date": str(confirmation["date"]),
                        "end_date": str(end["date"]),
                        "supporting_t_indices": [int(row["t_index"]) for row in run],
                        "supporting_dates": [str(row["date"]) for row in run],
                        "length": len(run),
                        "open_at_cutoff": next_row is None,
                        "delivery_state": delivery_state,
                        "termination_t_index": (
                            None if next_row is None else int(next_row["t_index"])
                        ),
                        "termination_date": (
                            None if next_row is None else str(next_row["date"])
                        ),
                        "termination_label": (
                            None
                            if next_row is None or next_row.get(label_col) is None
                            else int(next_row[label_col])
                        ),
                        "mean_conflict_evidence": 1.0,
                        "mean_uncertainty": None,
                        "confidence_metadata": confidence_values,
                        "agreement_metadata": agreement_values,
                    }
                )

    return _frame(rows, EPISODE_SCHEMA).sort("persona_id", "dimension", "onset_t_index")


def evidence_from_ordinal_artifact(
    outputs_df: pl.DataFrame,
    profiles_df: pl.DataFrame,
    *,
    source: str,
) -> pl.DataFrame:
    """Convert exported ordinal outputs into core-gated benchmark evidence."""
    required = {
        "persona_id",
        "t_index",
        "date",
        "dimension",
        "predicted_class",
        "uncertainty",
        "class_probabilities",
    }
    missing = required - set(outputs_df.columns)
    if missing:
        raise ValueError(
            "ordinal outputs are missing required columns: "
            + ", ".join(sorted(missing))
        )
    core_map = _core_value_map(profiles_df)
    rows = []
    for row in outputs_df.sort("persona_id", "t_index", "dimension").to_dicts():
        persona_id = str(row["persona_id"])
        dimension = normalize_value_name(str(row["dimension"]))
        if dimension not in core_map.get(persona_id, ()):
            continue
        probabilities = row["class_probabilities"]
        if probabilities is None or len(probabilities) != 3:
            raise ValueError(
                "class_probabilities must contain [-1, 0, +1] probabilities"
            )
        rows.append(
            {
                "source": source,
                "persona_id": persona_id,
                "dimension": dimension,
                "t_index": int(row["t_index"]),
                "date": str(row["date"]),
                "p_minus1": float(probabilities[0]),
                "uncertainty": float(row["uncertainty"]),
                "predicted_class": int(row["predicted_class"]),
                "evidence_kind": "soft_probability",
            }
        )
    return _frame(
        rows,
        {
            "source": pl.String,
            "persona_id": pl.String,
            "dimension": pl.String,
            "t_index": pl.Int64,
            "date": pl.String,
            "p_minus1": pl.Float64,
            "uncertainty": pl.Float64,
            "predicted_class": pl.Int64,
            "evidence_kind": pl.String,
        },
    )


def evidence_from_llm_records(
    records: Iterable[dict[str, Any]],
    *,
    source: str,
) -> pl.DataFrame:
    """Convert hard LLM scores to the common core-gated evidence schema."""
    rows = []
    for record_index, record in enumerate(records):
        if record.get("status") != "ok" or not isinstance(record.get("scores"), dict):
            raise ValueError(
                "LLM evidence record is not scoreable: "
                f"index={record_index}, persona_id={record.get('persona_id')}, "
                f"t_index={record.get('t_index')}, status={record.get('status')}"
            )
        normalized_core_values = {
            normalize_value_name(str(value))
            for value in record.get("core_values") or []
        }
        invalid_core_values = normalized_core_values - set(SCHWARTZ_VALUE_ORDER)
        if invalid_core_values:
            raise ValueError(
                "LLM evidence record has invalid core values: "
                + ", ".join(sorted(invalid_core_values))
            )
        if not normalized_core_values:
            raise ValueError("LLM evidence record has no declared core values")
        core_values = [
            dimension
            for dimension in SCHWARTZ_VALUE_ORDER
            if dimension in normalized_core_values
        ]
        for dimension in core_values:
            score = record["scores"].get(dimension)
            rows.append(
                {
                    "source": source,
                    "persona_id": str(record["persona_id"]),
                    "dimension": dimension,
                    "t_index": int(record["t_index"]),
                    "date": str(record["date"]),
                    "p_minus1": None if score is None else float(int(score) == -1),
                    "uncertainty": None,
                    "predicted_class": None if score is None else int(score),
                    "evidence_kind": "hard_class",
                }
            )
    return _frame(
        rows,
        {
            "source": pl.String,
            "persona_id": pl.String,
            "dimension": pl.String,
            "t_index": pl.Int64,
            "date": pl.String,
            "p_minus1": pl.Float64,
            "uncertainty": pl.Float64,
            "predicted_class": pl.Int64,
            "evidence_kind": pl.String,
        },
    )


def detect_sustained_conflict_episodes(
    evidence_df: pl.DataFrame,
    *,
    probability_threshold: float,
    uncertainty_threshold: float | None,
) -> pl.DataFrame:
    """Detect two-entry soft-evidence episodes on declared-core evidence rows."""
    required = {
        "source",
        "persona_id",
        "dimension",
        "t_index",
        "date",
        "p_minus1",
        "uncertainty",
    }
    missing = required - set(evidence_df.columns)
    if missing:
        raise ValueError(
            "evidence_df is missing required columns: " + ", ".join(sorted(missing))
        )
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    if uncertainty_threshold is not None and uncertainty_threshold < 0.0:
        raise ValueError("uncertainty_threshold must be non-negative")

    decisions = build_detection_decisions(
        evidence_df,
        probability_threshold=probability_threshold,
        uncertainty_threshold=uncertainty_threshold,
    )

    rows: list[dict[str, Any]] = []
    for group_key, timeline in (
        evidence_df.sort("source", "persona_id", "dimension", "t_index", "date")
        .partition_by("source", "persona_id", "dimension", as_dict=True)
        .items()
    ):
        source, persona_id, dimension = group_key
        entry_rows = timeline.to_dicts()
        group_decisions = decisions.filter(
            (pl.col("source") == str(source))
            & (pl.col("persona_id") == str(persona_id))
            & (pl.col("dimension") == str(dimension))
            & pl.col("alert")
        ).sort("window_start_t_index")
        alert_rows = group_decisions.to_dicts()
        decision_index = 0
        while decision_index < len(alert_rows):
            first_alert = alert_rows[decision_index]
            merged_alerts = [first_alert]
            decision_index += 1
            while decision_index < len(alert_rows) and int(
                alert_rows[decision_index]["window_start_t_index"]
            ) == int(merged_alerts[-1]["window_end_t_index"]):
                merged_alerts.append(alert_rows[decision_index])
                decision_index += 1

            onset_t_index = int(first_alert["window_start_t_index"])
            end_t_index = int(merged_alerts[-1]["window_end_t_index"])
            entry_by_t_index = {int(row["t_index"]): row for row in entry_rows}
            run = [
                entry_by_t_index[t_index]
                for t_index in range(onset_t_index, end_t_index + 1)
                if t_index in entry_by_t_index
            ]
            next_row = entry_by_t_index.get(end_t_index + 1)
            if next_row is None:
                delivery_state = "active"
            elif next_row.get("p_minus1") is None or (
                uncertainty_threshold is not None
                and (
                    next_row.get("uncertainty") is None
                    or float(next_row["uncertainty"]) > uncertainty_threshold
                )
            ):
                delivery_state = "uncertain"
            elif next_row.get("predicted_class") is None:
                delivery_state = "uncertain"
            elif int(next_row["predicted_class"]) == -1:
                delivery_state = "active"
            else:
                delivery_state = "recovered"

            onset, confirmation, end = run[0], run[1], run[-1]
            source = str(source)
            rows.append(
                {
                    "episode_id": (
                        f"{source}::{persona_id}::{dimension}::{int(onset['t_index'])}"
                    ),
                    "source": source,
                    "persona_id": str(persona_id),
                    "dimension": str(dimension),
                    "onset_t_index": int(onset["t_index"]),
                    "confirmation_t_index": int(confirmation["t_index"]),
                    "end_t_index": int(end["t_index"]),
                    "onset_date": str(onset["date"]),
                    "confirmation_date": str(confirmation["date"]),
                    "end_date": str(end["date"]),
                    "supporting_t_indices": [int(row["t_index"]) for row in run],
                    "supporting_dates": [str(row["date"]) for row in run],
                    "length": len(run),
                    "open_at_cutoff": next_row is None,
                    "delivery_state": delivery_state,
                    "termination_t_index": (
                        None if next_row is None else int(next_row["t_index"])
                    ),
                    "termination_date": (
                        None if next_row is None else str(next_row["date"])
                    ),
                    "termination_label": (
                        None
                        if next_row is None or next_row.get("predicted_class") is None
                        else int(next_row["predicted_class"])
                    ),
                    "mean_conflict_evidence": mean(
                        float(row["p_minus1"]) for row in run
                    ),
                    "mean_uncertainty": (
                        mean(
                            float(row["uncertainty"])
                            for row in run
                            if row.get("uncertainty") is not None
                        )
                        if any(row.get("uncertainty") is not None for row in run)
                        else None
                    ),
                    "confidence_metadata": [],
                    "agreement_metadata": [],
                }
            )

    return _frame(rows, EPISODE_SCHEMA).sort("persona_id", "dimension", "onset_t_index")


def build_detection_decisions(
    evidence_df: pl.DataFrame,
    *,
    probability_threshold: float,
    uncertainty_threshold: float | None,
) -> pl.DataFrame:
    """Evaluate every adjacent evidence pair and retain alert denominators."""
    _validate_evidence_keys(evidence_df)
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    if uncertainty_threshold is not None and uncertainty_threshold < 0.0:
        raise ValueError("uncertainty_threshold must be non-negative")

    rows = []
    for group_key, timeline in (
        evidence_df.sort("source", "persona_id", "dimension", "t_index", "date")
        .partition_by("source", "persona_id", "dimension", as_dict=True)
        .items()
    ):
        source, persona_id, dimension = group_key
        entries = timeline.to_dicts()
        for previous, current in zip(entries, entries[1:], strict=False):
            if int(current["t_index"]) != int(previous["t_index"]) + 1:
                continue
            probabilities = [previous.get("p_minus1"), current.get("p_minus1")]
            uncertainties = [previous.get("uncertainty"), current.get("uncertainty")]
            pair_probability = (
                None
                if any(value is None for value in probabilities)
                else mean(float(value) for value in probabilities)
            )
            probability_passed = (
                pair_probability is not None
                and pair_probability >= probability_threshold
            )
            if uncertainty_threshold is None:
                pair_uncertainty = (
                    None
                    if any(value is None for value in uncertainties)
                    else max(float(value) for value in uncertainties)
                )
                uncertainty_passed = True
            elif any(value is None for value in uncertainties):
                pair_uncertainty = None
                uncertainty_passed = False
            else:
                pair_uncertainty = max(float(value) for value in uncertainties)
                uncertainty_passed = pair_uncertainty <= uncertainty_threshold
            alert = probability_passed and uncertainty_passed
            if alert:
                suppression_reason = None
            elif pair_probability is None:
                suppression_reason = "missing_probability"
            elif not probability_passed:
                suppression_reason = "low_probability"
            elif pair_uncertainty is None:
                suppression_reason = "missing_uncertainty"
            else:
                suppression_reason = "high_uncertainty"
            rows.append(
                {
                    "source": str(source),
                    "persona_id": str(persona_id),
                    "dimension": str(dimension),
                    "window_start_t_index": int(previous["t_index"]),
                    "window_end_t_index": int(current["t_index"]),
                    "window_start_date": str(previous["date"]),
                    "window_end_date": str(current["date"]),
                    "pair_probability": pair_probability,
                    "pair_uncertainty": pair_uncertainty,
                    "probability_passed": probability_passed,
                    "uncertainty_passed": uncertainty_passed,
                    "alert": alert,
                    "suppression_reason": suppression_reason,
                }
            )
    return _frame(rows, DECISION_SCHEMA)


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
    return _frame(matches, MATCH_SCHEMA)


def episode_metrics(
    reference_df: pl.DataFrame,
    predicted_df: pl.DataFrame,
    eligible_df: pl.DataFrame,
    *,
    max_confirmation_lag: int = 2,
    decisions_df: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute episode hits, false alarms, latency, and recovery agreement."""
    matches = match_episodes(
        reference_df,
        predicted_df,
        max_confirmation_lag=max_confirmation_lag,
    )
    true_positive = matches.height
    false_positive = predicted_df.height - true_positive
    false_negative = reference_df.height - true_positive
    precision = (
        true_positive / (true_positive + false_positive) if predicted_df.height else 0.0
    )
    recall = true_positive / reference_df.height if reference_df.height else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    if "n_entries" not in eligible_df.columns:
        raise ValueError("eligible_df is missing required column: n_entries")
    decision_eligible = eligible_df.filter(pl.col("n_entries") >= 2)
    eligible_keys = set(decision_eligible.select("persona_id", "dimension").iter_rows())
    reference_keys = set(reference_df.select("persona_id", "dimension").iter_rows())
    predicted_keys = set(predicted_df.select("persona_id", "dimension").iter_rows())
    negative_keys = eligible_keys - reference_keys
    false_alarm_keys = negative_keys & predicted_keys
    trajectory_false_alarm_rate = (
        len(false_alarm_keys) / len(negative_keys) if negative_keys else 0.0
    )

    negative_windows = 0
    alerted_negative_windows = 0
    if decisions_df is not None:
        reference_by_key: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for row in reference_df.to_dicts():
            reference_by_key.setdefault(
                (str(row["persona_id"]), str(row["dimension"])), []
            ).append((int(row["onset_t_index"]), int(row["end_t_index"])))
        for decision in decisions_df.to_dicts():
            key = (str(decision["persona_id"]), str(decision["dimension"]))
            end_index = int(decision["window_end_t_index"])
            inside_reference = any(
                onset <= end_index <= end
                for onset, end in reference_by_key.get(key, [])
            )
            if inside_reference:
                continue
            negative_windows += 1
            alerted_negative_windows += bool(decision["alert"])
    false_positive_rate = (
        alerted_negative_windows / negative_windows if negative_windows else None
    )

    latencies = matches["latency_entries"].to_list() if matches.height else []
    delivery_rows = matches
    delivery_correct = (
        delivery_rows.filter(
            pl.col("reference_delivery_state") == pl.col("predicted_delivery_state")
        ).height
        if delivery_rows.height
        else 0
    )
    recovery_rows = matches.filter(pl.col("reference_delivery_state") == "recovered")
    recovery_correct = (
        recovery_rows.filter(
            pl.col("reference_delivery_state") == pl.col("predicted_delivery_state")
        ).height
        if recovery_rows.height
        else 0
    )
    return {
        "reference_episodes": reference_df.height,
        "predicted_episodes": predicted_df.height,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "hit_rate": recall,
        "f1": f1,
        "negative_trajectories": len(negative_keys),
        "false_alarm_trajectories": len(false_alarm_keys),
        "trajectory_false_alarm_rate": trajectory_false_alarm_rate,
        "negative_decision_windows": negative_windows,
        "alerted_negative_windows": alerted_negative_windows,
        "false_positive_rate": false_positive_rate,
        "mean_latency_entries": mean(latencies) if latencies else None,
        "median_latency_entries": median(latencies) if latencies else None,
        "max_latency_entries": max(latencies) if latencies else None,
        "delivery_state_cases": delivery_rows.height,
        "delivery_state_correct": delivery_correct,
        "delivery_state_accuracy": (
            delivery_correct / delivery_rows.height if delivery_rows.height else None
        ),
        "recovery_cases": recovery_rows.height,
        "recovery_correct": recovery_correct,
        "recovery_accuracy": (
            recovery_correct / recovery_rows.height if recovery_rows.height else None
        ),
    }


def tune_detector_thresholds(
    evidence_df: pl.DataFrame,
    reference_df: pl.DataFrame,
    eligible_df: pl.DataFrame,
    *,
    probability_thresholds: Iterable[float],
    uncertainty_thresholds: Iterable[float],
    minimum_precision: float = 0.6,
    maximum_false_alarm_rate: float = 0.2,
    max_confirmation_lag: int = 2,
) -> tuple[dict[str, float], pl.DataFrame]:
    """Choose thresholds on development data using event F1 and guardrails."""
    candidates = []
    for probability_threshold in probability_thresholds:
        for uncertainty_threshold in uncertainty_thresholds:
            predicted = detect_sustained_conflict_episodes(
                evidence_df,
                probability_threshold=float(probability_threshold),
                uncertainty_threshold=float(uncertainty_threshold),
            )
            metrics = episode_metrics(
                reference_df,
                predicted,
                eligible_df,
                max_confirmation_lag=max_confirmation_lag,
                decisions_df=build_detection_decisions(
                    evidence_df,
                    probability_threshold=float(probability_threshold),
                    uncertainty_threshold=float(uncertainty_threshold),
                ),
            )
            candidates.append(
                {
                    "probability_threshold": float(probability_threshold),
                    "uncertainty_threshold": float(uncertainty_threshold),
                    "meets_guardrails": (
                        metrics["precision"] >= minimum_precision
                        and (
                            metrics["false_positive_rate"] is None
                            or metrics["false_positive_rate"]
                            <= maximum_false_alarm_rate
                        )
                    ),
                    **metrics,
                }
            )
    if not candidates:
        raise ValueError("threshold grids must contain at least one candidate")

    guarded = [row for row in candidates if row["meets_guardrails"]]
    pool = guarded or candidates

    def rank(row: dict[str, Any]) -> tuple[float, ...]:
        latency = row["mean_latency_entries"]
        # When observed metrics tie exactly, prefer the more recall-sensitive
        # operating point: lower conflict threshold and wider uncertainty gate.
        return (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
            -float(
                row["false_positive_rate"]
                if row["false_positive_rate"] is not None
                else 0.0
            ),
            -float(latency if latency is not None else 999.0),
            -float(row["probability_threshold"]),
            float(row["uncertainty_threshold"]),
        )

    selected = max(pool, key=rank)
    grid = pl.DataFrame(candidates).sort(
        "meets_guardrails",
        "f1",
        "recall",
        "precision",
        descending=[True, True, True, True],
    )
    return {
        "probability_threshold": float(selected["probability_threshold"]),
        "uncertainty_threshold": float(selected["uncertainty_threshold"]),
    }, grid
