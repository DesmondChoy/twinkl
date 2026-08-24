"""Weekly Drift Detection output helpers for the Coach Digest.

This module builds structured Weekly Drift Detection output from offline files.
It renders the Coach Digest prompt, validates generated responses, and persists
the output for later analysis.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
from collections.abc import Awaitable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol, cast

import polars as pl
import yaml

from prompts import load_prompt
from src.coach.mode_logic import (
    WeeklyModeSignals,
    has_acute_distress_context,
    has_background_strain_context,
    infer_response_mode,
)
from src.coach.schemas import (
    WEEKLY_DIGEST_COACH_RESPONSE_FORMAT,
    CoachDigestDiagnostic,
    CoachDigestPolicy,
    CoachNarrative,
    CoachResponseMode,
    CoreValueDigestDetail,
    CoreValueDriftState,
    CoreValueStateChange,
    CoreValueWeekComparison,
    DigestValidation,
    DimensionDigest,
    DriftDetectionResult,
    EvidenceSnippet,
    JournalHistoryEntry,
    ValidationCheck,
    WeeklyDigest,
)
from src.drift_detector import DriftDetectorResult, detect_drift
from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.prompt_boundary import (
    protect_instructions,
    render_live_prompt_receipt,
    serialize_untrusted_data,
)
from src.weekly_drift_reviewer import (
    WeeklyDriftReviewerDecision,
    WeeklyDriftReviewerEntry,
)
from src.weekly_signal_schema import (
    ALIGNMENT_COLUMNS,
    UNCERTAINTY_COLUMNS,
    alignment_col,
)
from src.wrangling.parse_wrangled_data import parse_wrangled_file

SCHWARTZ_CONFIG_PATH = Path("config/schwartz_values.yaml")


class LLMCompleteFn(Protocol):
    """Provider call with an optional higher-priority instruction message."""

    def __call__(
        self,
        prompt: str,
        response_format: dict | None,
        instructions: str | None = None,
    ) -> Awaitable[str | None]: ...


def _parse_iso_date(raw: str) -> date:
    """Parse YYYY-MM-DD date strings."""
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _format_dim_name(dim: str) -> str:
    """Format snake_case value names for human-readable output."""
    return dim.replace("_", " ").title()


def _truncate_excerpt(text: str, max_words: int = 40) -> str:
    """Keep excerpts compact for digest readability."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "..."


def _format_optional_score(score: float | None) -> str:
    """Render a score only when the compatibility path supplied one."""
    return f"{score:.3f}" if score is not None else "N/A"


def _to_snake_case(value_name: str) -> str:
    """Convert value labels like 'Self Direction' to 'self_direction'."""
    return value_name.strip().lower().replace("-", " ").replace(" ", "_")


def _load_schwartz_value_map(
    config_path: Path = SCHWARTZ_CONFIG_PATH,
) -> dict[str, dict]:
    """Load Schwartz value elaborations keyed by snake_case dimension name."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    values = raw.get("values", {})
    return {
        _to_snake_case(display_name): details
        for display_name, details in values.items()
    }


def _summarize_compass_context(
    value_map: dict[str, dict],
    dimensions: list[str],
) -> list[str]:
    """Pair internal Schwartz labels with the phrases confirmed in onboarding."""
    lines: list[str] = []
    for dim in dimensions:
        details = value_map.get(dim, {})
        user_phrase = str(details.get("user_phrase", "")).strip()
        line = f"- Internal Schwartz label: {_format_dim_name(dim)}"
        if user_phrase:
            line += f' | user-facing compass phrase: "{user_phrase}"'
        lines.append(line)

    return lines


def _coach_digest_policy(digest: WeeklyDigest) -> CoachDigestPolicy:
    """Reduce auditable Drift states to three Coach Digest delivery policies."""
    states = set(digest.drift_states.values())
    if "active_drift" in states:
        return "drift_detected"
    if "insufficient_evidence" in states:
        return "more_reflection_needed"
    if digest.response_mode in {
        "uncertain",
        "mixed",
        "rut",
        "crash",
        "evolution",
        "high_uncertainty",
        "mixed_state",
        "background_strain",
    }:
        return "more_reflection_needed"
    return "no_current_drift"


def _drift_summary_lines(digest: WeeklyDigest) -> list[str]:
    """Explain upstream Drift states without asking the Coach Digest to infer them."""
    if not digest.drift_states:
        return [
            "- Weekly Drift Detection did not find two consecutive Journal Entries "
            "showing clear behavior or choices against the same Core Value. This "
            "does not by itself prove positive behavior."
        ]

    meanings = {
        "active_drift": (
            "Drift is active: two consecutive Journal Entries each showed a clear "
            "behavior or choice against this Core Value, with no later Journal "
            "Entry ending the pattern."
        ),
        "no_active_drift": (
            "No active Drift is confirmed for this Core Value at the cutoff. This "
            "does not prove supportive behavior, improvement, or success."
        ),
        "insufficient_evidence": (
            "Current Drift status has insufficient evidence because a review "
            "failure prevents a current claim, or an Abstain or Journal Entry gap "
            "blocked recent Conflict evidence."
        ),
    }
    lines: list[str] = []
    for core_value, state in digest.drift_states.items():
        line = (
            f"- Internal Schwartz label: {_format_dim_name(core_value)} | "
            f"{meanings[state]}"
        )
        detail = digest.drift_details.get(core_value)
        if detail is not None:
            line += (
                f" | current run length: {detail.current_run_length}"
                f" | last decision: {detail.last_decision}"
                f" | last review status: {detail.last_review_status}"
            )
        lines.append(line)
    return lines


def _state_comparison_lines(digest: WeeklyDigest) -> list[str]:
    """Render deterministic prior-week facts without asking the Coach to infer them."""
    if not digest.state_comparisons:
        return ["- No prior closed-week comparison is available."]
    lines: list[str] = []
    for comparison in digest.state_comparisons:
        line = (
            f"- Internal Schwartz label: {_format_dim_name(comparison.core_value)} | "
            f"previous state: {comparison.previous_state} | "
            f"current state: {comparison.current_state} | "
            f"deterministic change: {comparison.change}"
        )
        if comparison.end_reason is not None:
            line += f" | end reason: {comparison.end_reason}"
        lines.append(line)
        lines.extend(
            f"  prior evidence {item.date}: "
            f"{json.dumps(item.excerpt, ensure_ascii=False)}"
            for item in comparison.previous_evidence
        )
        lines.extend(
            f"  current evidence {item.date}: "
            f"{json.dumps(item.excerpt, ensure_ascii=False)}"
            for item in comparison.current_evidence
        )
    return lines


def _evidence_role(direction: str) -> str:
    """Translate stored evidence directions into prompt-facing roles."""
    return {
        "misaligned": "Drift evidence",
        "aligned": "supportive evidence",
        "strain": "strain context",
        "recovery": "recovery context",
        "context": "weekly context",
    }[direction]


def _load_persona_context(
    persona_id: str,
    wrangled_dir: Path,
    history_end: date | None = None,
) -> tuple[dict, list[JournalHistoryEntry], dict[tuple[str, int], str]]:
    """Load profile, full journal history, and entry texts for one persona."""
    wrangled_file = wrangled_dir / f"persona_{persona_id}.md"
    if not wrangled_file.exists():
        raise FileNotFoundError(f"Wrangled file not found: {wrangled_file}")

    profile, entries, _warnings = parse_wrangled_file(wrangled_file)

    text_by_key: dict[tuple[str, int], str] = {}
    journal_history: list[JournalHistoryEntry] = []
    for entry in entries:
        entry_date = _parse_iso_date(entry["date"])
        if history_end is not None and entry_date > history_end:
            continue

        combined = entry["initial_entry"] or ""
        if entry.get("has_response") and entry.get("response_text"):
            combined = f"{combined}\n\nResponse: {entry['response_text']}"
        combined = combined.strip()

        key = (entry["date"], entry["t_index"])
        text_by_key[key] = combined
        journal_history.append(
            JournalHistoryEntry(
                date=entry["date"],
                t_index=int(entry["t_index"]),
                content=combined,
                has_response=bool(entry.get("has_response")),
            )
        )

    profile["core_values"] = [
        _to_snake_case(value_name)
        for value_name in (profile.get("core_values") or [])
        if isinstance(value_name, str)
    ]
    return profile, journal_history, text_by_key


def _resolve_week_window(
    labels: pl.DataFrame, start_date: str | None, end_date: str | None
) -> tuple[date, date]:
    """Resolve digest window, defaulting to the latest 7-day range."""
    min_date = _parse_iso_date(labels["date"].min())
    max_date = _parse_iso_date(labels["date"].max())

    resolved_end = _parse_iso_date(end_date) if end_date else max_date
    resolved_start = (
        _parse_iso_date(start_date) if start_date else resolved_end - timedelta(days=6)
    )

    if resolved_start > resolved_end:
        raise ValueError("start_date must be on or before end_date")
    if resolved_end < min_date or resolved_start > max_date:
        raise ValueError(
            f"Requested window [{resolved_start}, {resolved_end}] is outside available "
            f"label range [{min_date}, {max_date}] for this persona."
        )

    return resolved_start, resolved_end


def _load_signal_frame(
    *,
    persona_id: str,
    labels_path: Path | None,
    signals_path: Path | None,
    signals_df: pl.DataFrame | None,
) -> tuple[pl.DataFrame, str]:
    """Load either live VIF signals or judge labels for one persona."""
    if signals_df is not None:
        frame = signals_df.filter(pl.col("persona_id") == persona_id)
        source = "vif_runtime"
    elif signals_path is not None:
        frame = pl.read_parquet(signals_path).filter(pl.col("persona_id") == persona_id)
        source = "vif_runtime"
    else:
        if labels_path is None:
            raise ValueError(
                "Either signals_path/signals_df or labels_path must be provided."
            )
        frame = pl.read_parquet(labels_path).filter(pl.col("persona_id") == persona_id)
        source = "judge_labels"

    if frame.is_empty():
        raise ValueError(f"No numeric signals found for persona_id={persona_id}")
    return frame, source


def _prioritize_dimensions_from_drift(
    ranked_dims: list[str],
    drift_result: DriftDetectionResult | None,
    *,
    direction: str,
) -> list[str]:
    """Prioritize dimensions highlighted by upstream drift/evolution output."""
    if drift_result is None or not drift_result.dimension_signals:
        return ranked_dims

    prioritized: list[str] = []
    for signal in drift_result.dimension_signals:
        if (
            direction == "tension"
            and signal.mean_alignment < 0
            and signal.trigger
            in {
                "crash",
                "rut",
                "evolution",
            }
        ):
            prioritized.append(signal.dimension)
        if (
            direction == "strength"
            and signal.mean_alignment > 0
            and signal.trigger
            in {
                "evolution",
                "acknowledgement",
            }
        ):
            prioritized.append(signal.dimension)

    for dim in ranked_dims:
        if dim not in prioritized:
            prioritized.append(dim)
    return prioritized


def _rank_dimensions(
    dim_means: list[tuple[str, float]],
    core_values: list[str],
    direction: str,
    limit: int,
) -> list[str]:
    """Rank digest focus dimensions without forcing contradictory fallback buckets."""
    if direction not in {"tension", "strength"}:
        raise ValueError("direction must be 'tension' or 'strength'")

    if direction == "tension":
        primary_core = [
            item for item in dim_means if item[0] in core_values and item[1] < 0
        ]
        primary_other = [
            item for item in dim_means if item[0] not in core_values and item[1] < 0
        ]
        primary_core.sort(key=lambda item: item[1])
        primary_other.sort(key=lambda item: item[1])
    else:
        primary_core = [
            item for item in dim_means if item[0] in core_values and item[1] > 0
        ]
        primary_other = [
            item for item in dim_means if item[0] not in core_values and item[1] > 0
        ]
        primary_core.sort(key=lambda item: item[1], reverse=True)
        primary_other.sort(key=lambda item: item[1], reverse=True)

    ranked: list[str] = []
    for bucket in (primary_core, primary_other):
        for dim, _score in bucket:
            if dim not in ranked:
                ranked.append(dim)
            if len(ranked) >= limit:
                return ranked

    return ranked[:limit]


def _select_row_dimensions(
    row: dict, candidate_dims: list[str], direction: str
) -> list[str]:
    """Select the most representative focus dimensions for one evidence row."""
    scored_dims = [(dim, float(row[alignment_col(dim)])) for dim in candidate_dims]
    reverse = direction == "aligned"
    scored_dims.sort(key=lambda item: item[1], reverse=reverse)

    if direction == "misaligned":
        filtered = [dim for dim, score in scored_dims if score < 0]
    else:
        filtered = [dim for dim, score in scored_dims if score > 0]

    if filtered:
        return filtered[:2]
    return [scored_dims[0][0]] if scored_dims else []


def _build_mean_expr(columns: list[str], alias: str) -> pl.Expr:
    """Build a horizontal-mean expression with a safe empty fallback."""
    if not columns:
        return pl.lit(0.0).alias(alias)
    return pl.mean_horizontal(columns).alias(alias)


def _find_dimension_polarity(
    labels: pl.DataFrame,
    dimensions: list[str],
) -> bool:
    """Detect within-week polarity flips on the declared core dimensions."""
    for dim in dimensions:
        col = alignment_col(dim)
        if col not in labels.columns:
            continue
        series = labels[col]
        if bool((series == -1).any()) and bool((series == 1).any()):
            return True
    return False


def _select_strain_dimensions(
    core_values: list[str],
    top_tensions: list[str],
    top_strengths: list[str],
) -> list[str]:
    """Choose soft focus dimensions for strain evidence when no clean tension exists."""
    candidates = list(dict.fromkeys(core_values + top_tensions + top_strengths))
    return candidates[:2]


def _append_mode_specific_strain_evidence(
    evidence_rows: list[EvidenceSnippet],
    selected_keys: set[tuple[str, int]],
    response_mode: CoachResponseMode,
    journal_history: list[JournalHistoryEntry],
    entry_texts: dict[tuple[str, int], str],
    core_values: list[str],
    top_tensions: list[str],
    top_strengths: list[str],
    with_entry_scores: pl.DataFrame,
) -> None:
    """Add strain evidence for uncertainty or background-strain fallback modes."""
    if response_mode not in {"high_uncertainty", "background_strain"}:
        return

    if response_mode == "high_uncertainty":
        matched_entries = [
            entry for entry in journal_history if has_acute_distress_context([entry])
        ]
    else:
        matched_entries = [
            entry for entry in journal_history if has_background_strain_context([entry])
        ]

    row_by_key = {
        (row["date"], int(row["t_index"])): row for row in with_entry_scores.to_dicts()
    }
    strain_dimensions = _select_strain_dimensions(
        core_values, top_tensions, top_strengths
    )

    for entry in matched_entries:
        key = (entry.date, entry.t_index)
        row = row_by_key.get(key)
        score_mean = float(row["entry_mean"]) if row is not None else 0.0
        snippet = EvidenceSnippet(
            date=entry.date,
            t_index=entry.t_index,
            direction="strain",
            dimensions=strain_dimensions,
            score_mean=score_mean,
            excerpt=_truncate_excerpt(entry_texts.get(key, entry.content)),
        )

        if key in selected_keys:
            for index, existing in enumerate(evidence_rows):
                if (existing.date, existing.t_index) == key:
                    evidence_rows[index] = snippet
                    return
            continue

        evidence_rows.insert(0, snippet)
        selected_keys.add(key)
        return


def build_weekly_digest(
    persona_id: str,
    labels_path: Path | None,
    wrangled_dir: Path,
    signals_path: Path | None = None,
    signals_df: pl.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    drift_result: DriftDetectionResult | None = None,
    response_mode: CoachResponseMode | None = None,
) -> WeeklyDigest:
    """Build a structured Weekly Drift Detection output for one persona."""
    labels, signal_source = _load_signal_frame(
        persona_id=persona_id,
        labels_path=labels_path,
        signals_path=signals_path,
        signals_df=signals_df,
    )

    resolved_start, resolved_end = _resolve_week_window(labels, start_date, end_date)

    labels = labels.filter(
        (pl.col("date") >= resolved_start.isoformat())
        & (pl.col("date") <= resolved_end.isoformat())
    ).sort(["date", "t_index"])
    if labels.is_empty():
        raise ValueError(
            f"No labels in window [{resolved_start}, {resolved_end}] "
            f"for persona_id={persona_id}"
        )

    profile, window_entries, entry_texts = _load_persona_context(
        persona_id,
        wrangled_dir,
        history_end=resolved_end,
    )
    core_values = profile.get("core_values") or []

    dim_rows: list[DimensionDigest] = []
    dim_means: list[tuple[str, float]] = []
    n_rows = labels.height

    for dim, col in zip(SCHWARTZ_VALUE_ORDER, ALIGNMENT_COLUMNS, strict=True):
        mean_score = float(labels[col].mean())
        neg_count = int((labels[col] == -1).sum())
        neutral_count = int((labels[col] == 0).sum())
        pos_count = int((labels[col] == 1).sum())

        dim_rows.append(
            DimensionDigest(
                dimension=dim,
                mean_score=mean_score,
                pct_neg=neg_count / n_rows,
                pct_neutral=neutral_count / n_rows,
                pct_pos=pos_count / n_rows,
            )
        )
        dim_means.append((dim, mean_score))

    top_tensions = _rank_dimensions(
        dim_means, core_values, direction="tension", limit=3
    )
    top_strengths = _rank_dimensions(
        dim_means, core_values, direction="strength", limit=2
    )
    top_tensions = _prioritize_dimensions_from_drift(
        top_tensions,
        drift_result,
        direction="tension",
    )
    top_strengths = _prioritize_dimensions_from_drift(
        top_strengths,
        drift_result,
        direction="strength",
    )
    top_strengths = [dim for dim in top_strengths if dim not in top_tensions]
    top_tensions = top_tensions[:3]
    top_strengths = top_strengths[:2]

    tension_cols = [alignment_col(dim) for dim in top_tensions]
    strength_cols = [alignment_col(dim) for dim in top_strengths]
    with_entry_scores = labels.with_columns(
        [
            pl.mean_horizontal(ALIGNMENT_COLUMNS).alias("entry_mean"),
            _build_mean_expr(tension_cols, "tension_score"),
            _build_mean_expr(strength_cols, "strength_score"),
        ]
    )

    n_entries = with_entry_scores.height
    mis_target = 1 if n_entries <= 2 else 2
    aligned_target = 1

    mis_primary = with_entry_scores.filter(pl.col("tension_score") < 0).sort(
        ["tension_score", "entry_mean"]
    )
    mis_fallback = with_entry_scores.sort(["tension_score", "entry_mean"])
    mis_candidates = mis_primary.to_dicts() + mis_fallback.to_dicts()

    aligned_primary = with_entry_scores.filter(pl.col("strength_score") > 0).sort(
        ["strength_score", "entry_mean"], descending=[True, True]
    )
    aligned_fallback = with_entry_scores.sort(
        ["strength_score", "entry_mean"], descending=[True, True]
    )
    aligned_candidates = aligned_primary.to_dicts() + aligned_fallback.to_dicts()

    evidence_rows: list[EvidenceSnippet] = []
    selected_keys: set[tuple[str, int]] = set()

    for row in mis_candidates:
        if len(evidence_rows) >= mis_target:
            break
        key = (row["date"], int(row["t_index"]))
        if key in selected_keys:
            continue
        selected_keys.add(key)
        raw_text = entry_texts.get(key, "")
        evidence_rows.append(
            EvidenceSnippet(
                date=row["date"],
                t_index=int(row["t_index"]),
                direction="misaligned",
                dimensions=_select_row_dimensions(row, top_tensions, "misaligned"),
                score_mean=float(row["entry_mean"]),
                excerpt=_truncate_excerpt(raw_text),
            )
        )

    aligned_added = 0
    for row in aligned_candidates:
        if aligned_added >= aligned_target:
            break
        key = (row["date"], int(row["t_index"]))
        if key in selected_keys:
            continue
        selected_keys.add(key)
        raw_text = entry_texts.get(key, "")
        evidence_rows.append(
            EvidenceSnippet(
                date=row["date"],
                t_index=int(row["t_index"]),
                direction="aligned",
                dimensions=_select_row_dimensions(row, top_strengths, "aligned"),
                score_mean=float(row["entry_mean"]),
                excerpt=_truncate_excerpt(raw_text),
            )
        )
        aligned_added += 1

    overall_mean = float(with_entry_scores["entry_mean"].mean())
    overall_uncertainty = None
    if "overall_uncertainty" in labels.columns:
        overall_uncertainty = float(labels["overall_uncertainty"].mean())
    elif all(column in labels.columns for column in UNCERTAINTY_COLUMNS):
        overall_uncertainty = float(
            labels.select(pl.mean_horizontal(UNCERTAINTY_COLUMNS).alias("u"))[
                "u"
            ].mean()
        )
    has_mixed_core_polarity = _find_dimension_polarity(labels, core_values)

    if response_mode is not None:
        resolved_mode = response_mode
        mode_source = "manual_override"
        mode_rationale = (
            "Response mode supplied explicitly for testing or manual review."
        )
        drift_reasons: list[str] = []
    elif drift_result is not None:
        resolved_mode = drift_result.response_mode
        mode_source = drift_result.source
        mode_rationale = drift_result.rationale
        drift_reasons = list(drift_result.reasons)
    else:
        mode_decision = infer_response_mode(
            WeeklyModeSignals(
                overall_mean=overall_mean,
                overall_uncertainty=overall_uncertainty,
                top_tensions=top_tensions,
                top_strengths=top_strengths,
                core_values=core_values,
                window_entries=window_entries,
                has_mixed_core_polarity=has_mixed_core_polarity,
            )
        )
        resolved_mode = mode_decision.response_mode
        mode_source = mode_decision.mode_source
        mode_rationale = mode_decision.mode_rationale
        drift_reasons = []

    if resolved_mode in {"high_uncertainty", "background_strain"}:
        top_tensions = []

    _append_mode_specific_strain_evidence(
        evidence_rows=evidence_rows,
        selected_keys=selected_keys,
        response_mode=resolved_mode,
        journal_history=window_entries,
        entry_texts=entry_texts,
        core_values=core_values,
        top_tensions=top_tensions,
        top_strengths=top_strengths,
        with_entry_scores=with_entry_scores,
    )

    return WeeklyDigest(
        persona_id=persona_id,
        persona_name=profile.get("name"),
        week_start=resolved_start.isoformat(),
        week_end=resolved_end.isoformat(),
        response_mode=resolved_mode,
        mode_source=mode_source,
        mode_rationale=mode_rationale,
        signal_source=signal_source,
        n_entries=n_rows,
        overall_mean=overall_mean,
        overall_uncertainty=overall_uncertainty,
        core_values=core_values,
        drift_reasons=drift_reasons,
        top_tensions=top_tensions,
        top_strengths=top_strengths,
        dimensions=dim_rows,
        evidence=evidence_rows,
    )


def _evidence_from_decisions(
    decisions: Sequence[WeeklyDriftReviewerDecision],
    *,
    entry_texts: dict[tuple[str, int], str],
    limit: int = 3,
) -> list[EvidenceSnippet]:
    """Build capped evidence with exact Conflict quotes and displayed text."""
    snippets: list[EvidenceSnippet] = []
    for decision in sorted(decisions, key=lambda row: row.t_index)[-limit:]:
        key = (decision.date, decision.t_index)
        excerpt = (
            decision.evidence_quote.strip()
            if decision.verdict == "conflict"
            else entry_texts.get(key, "")
        ) or entry_texts.get(key, "")
        if not excerpt:
            continue
        snippets.append(
            EvidenceSnippet(
                date=decision.date,
                t_index=decision.t_index,
                direction=(
                    "misaligned" if decision.verdict == "conflict" else "context"
                ),
                dimensions=[decision.core_value],
                excerpt=_truncate_excerpt(excerpt),
            )
        )
    return snippets


def _relevant_state_decisions(
    *,
    core_value: str,
    state: CoreValueDriftState,
    decisions: Sequence[WeeklyDriftReviewerDecision],
    drift_result: DriftDetectorResult,
    window_start: date | None = None,
    window_end: date | None = None,
) -> list[WeeklyDriftReviewerDecision]:
    """Select the smallest decision set that explains one current state."""
    value_decisions = [
        decision for decision in decisions if decision.core_value == core_value
    ]
    if window_start is not None and window_end is not None:
        window_decisions = [
            decision
            for decision in value_decisions
            if window_start <= _parse_iso_date(decision.date) <= window_end
        ]
        if window_decisions:
            return window_decisions[-3:]
    if state == "active_drift":
        active = next(
            (
                drift
                for drift in reversed(drift_result.drifts)
                if drift.core_value == core_value and drift.termination_reason is None
            ),
            None,
        )
        if active is not None:
            indices = set(active.supporting_t_indices[-3:])
            return [
                decision
                for decision in value_decisions
                if decision.t_index in indices
            ]
    return value_decisions[-3:]


def _comparison_change(
    previous_state: CoreValueDriftState,
    current_state: CoreValueDriftState,
    *,
    restarted: bool,
) -> CoreValueStateChange:
    if restarted:
        return "active_drift_restarted"
    if current_state == "active_drift" and previous_state != "active_drift":
        return "active_drift_started"
    if previous_state == "active_drift" and current_state == "no_active_drift":
        return "active_drift_ended"
    if (
        current_state == "insufficient_evidence"
        and previous_state != "insufficient_evidence"
    ):
        return "evidence_became_insufficient"
    if (
        previous_state == "insufficient_evidence"
        and current_state != "insufficient_evidence"
    ):
        return "evidence_resolved"
    return "unchanged"


def _comparison_end_reason(
    *,
    previous_state: CoreValueDriftState,
    current_decisions: Sequence[WeeklyDriftReviewerDecision],
    previous_last_t_index: int,
) -> Literal["not_conflict", "abstain", "gap"] | None:
    if previous_state != "active_drift":
        return None
    ordered = sorted(current_decisions, key=lambda row: row.t_index)
    expected_t_index = previous_last_t_index + 1
    for decision in ordered:
        if decision.t_index != expected_t_index:
            return "gap"
        if decision.review_status != "ok" or decision.verdict == "abstain":
            return "abstain"
        if decision.verdict == "not_conflict":
            return "not_conflict"
        expected_t_index = decision.t_index + 1
    return None


def build_weekly_drift_reviewer_digest_from_entries(
    *,
    persona_id: str,
    persona_name: str | None,
    week_start: str,
    week_end: str,
    core_values: list[str],
    entries: Sequence[WeeklyDriftReviewerEntry],
    decisions: Sequence[WeeklyDriftReviewerDecision],
    drift_result: DriftDetectorResult,
) -> WeeklyDigest:
    """Build Weekly Drift Detection output from displayed Journal Entries."""
    resolved_start = _parse_iso_date(week_start)
    resolved_end = _parse_iso_date(week_end)
    if resolved_start > resolved_end:
        raise ValueError("week_start must be on or before week_end")
    if drift_result.persona_id != persona_id:
        raise ValueError("Drift Detector result does not match persona_id")

    history = sorted(entries, key=lambda entry: entry.t_index)
    if any(_parse_iso_date(entry.date) > resolved_end for entry in history):
        raise ValueError(
            "Weekly Drift Detection entries cannot extend beyond week_end"
        )
    window_entries = [
        entry
        for entry in history
        if resolved_start <= _parse_iso_date(entry.date) <= resolved_end
    ]
    if not window_entries:
        raise ValueError(
            f"No Journal Entries in window [{week_start}, {week_end}] "
            f"for persona_id={persona_id}"
        )

    entry_texts = {(entry.date, entry.t_index): entry.text for entry in history}
    current_decisions = [
        decision
        for decision in decisions
        if resolved_start <= _parse_iso_date(decision.date) <= resolved_end
    ]
    previous_end = resolved_start - timedelta(days=1)
    previous_start = resolved_start - timedelta(days=7)
    previous_cutoff_decisions = [
        decision
        for decision in decisions
        if _parse_iso_date(decision.date) <= previous_end
    ]
    previous_result = (
        detect_drift(previous_cutoff_decisions, persona_id=persona_id)
        if previous_cutoff_decisions
        else None
    )

    state_comparisons: list[CoreValueWeekComparison] = []
    evidence_by_coordinate: dict[tuple[int, str], EvidenceSnippet] = {}
    for core_value in core_values:
        current_state = drift_result.core_value_states[core_value]
        current_relevant = _relevant_state_decisions(
            core_value=core_value,
            state=current_state,
            decisions=decisions,
            drift_result=drift_result,
            window_start=resolved_start,
            window_end=resolved_end,
        )
        current_evidence = _evidence_from_decisions(
            current_relevant,
            entry_texts=entry_texts,
        )
        for item in current_evidence:
            evidence_by_coordinate[(item.t_index, core_value)] = item

        if (
            previous_result is None
            or core_value not in previous_result.core_value_states
        ):
            continue
        previous_state = previous_result.core_value_states[core_value]
        previous_relevant = _relevant_state_decisions(
            core_value=core_value,
            state=previous_state,
            decisions=previous_cutoff_decisions,
            drift_result=previous_result,
            window_start=previous_start,
            window_end=previous_end,
        )
        previous_evidence = _evidence_from_decisions(
            previous_relevant,
            entry_texts=entry_texts,
        )
        value_current_decisions = [
            decision
            for decision in current_decisions
            if decision.core_value == core_value
        ]
        previous_open_ids = {
            drift.drift_id
            for drift in previous_result.drifts
            if drift.core_value == core_value and drift.termination_reason is None
        }
        current_drifts = [
            drift
            for drift in drift_result.drifts
            if drift.core_value == core_value
        ]
        restarted = (
            previous_state == "active_drift"
            and current_state == "active_drift"
            and any(
                drift.drift_id not in previous_open_ids
                and resolved_start
                <= _parse_iso_date(drift.confirmation_date)
                <= resolved_end
                for drift in current_drifts
            )
            and any(
                drift.drift_id in previous_open_ids
                and drift.termination_reason is not None
                for drift in current_drifts
            )
        )
        end_reason = _comparison_end_reason(
            previous_state=previous_state,
            current_decisions=value_current_decisions,
            previous_last_t_index=(
                previous_result.core_value_details[core_value].last_t_index
            ),
        )
        state_comparisons.append(
            CoreValueWeekComparison(
                core_value=core_value,
                previous_week_start=previous_start.isoformat(),
                previous_week_end=previous_end.isoformat(),
                current_week_start=resolved_start.isoformat(),
                current_week_end=resolved_end.isoformat(),
                previous_state=previous_state,
                current_state=current_state,
                change=_comparison_change(
                    previous_state,
                    current_state,
                    restarted=restarted,
                ),
                end_reason=end_reason,
                previous_evidence=previous_evidence,
                current_evidence=current_evidence,
            )
        )
        for item in previous_evidence:
            evidence_by_coordinate.setdefault((item.t_index, core_value), item)

    evidence = sorted(
        evidence_by_coordinate.values(),
        key=lambda item: (item.t_index, item.dimensions),
    )
    if not evidence:
        evidence = [
            EvidenceSnippet(
                date=entry.date,
                t_index=entry.t_index,
                direction="context",
                dimensions=[],
                excerpt=_truncate_excerpt(entry.text),
            )
            for entry in window_entries[-2:]
        ]

    state = drift_result.delivery_state
    review_unavailable = not any(
        decision.review_status == "ok"
        and resolved_start <= _parse_iso_date(decision.date) <= resolved_end
        for decision in decisions
    )
    if review_unavailable:
        rationale = (
            "The Weekly Drift Reviewer could not return usable evidence for this week."
        )
    elif state == "no_active_drift":
        rationale = (
            "No Core Value has active Drift at this cutoff. This does not prove "
            "supportive behavior or improvement."
        )
    elif state == "insufficient_evidence":
        rationale = "A recent Conflict run lacks enough evidence for a current claim."
    else:
        rationale = "At least one Core Value has active Drift."

    drift_reasons = [
        f"{drift.drift_id}:{drift.termination_reason or 'open'}"
        for drift in drift_result.drifts
    ]
    return WeeklyDigest(
        persona_id=persona_id,
        persona_name=persona_name,
        week_start=week_start,
        week_end=week_end,
        response_mode=state,
        mode_source="drift_detector",
        mode_rationale=rationale,
        signal_source="weekly_drift_reviewer",
        n_entries=len(window_entries),
        overall_mean=None,
        overall_uncertainty=None,
        core_values=core_values,
        drift_states=drift_result.core_value_states,
        drift_details={
            core_value: CoreValueDigestDetail.model_validate(detail.model_dump())
            for core_value, detail in drift_result.core_value_details.items()
        },
        state_comparisons=state_comparisons,
        drift_reasons=drift_reasons,
        top_tensions=[
            core_value
            for core_value, core_state in drift_result.core_value_states.items()
            if core_state != "no_active_drift"
        ],
        top_strengths=[],
        dimensions=[],
        evidence=evidence,
    )


def build_weekly_drift_reviewer_digest(
    *,
    persona_id: str,
    wrangled_dir: Path,
    week_start: str,
    week_end: str,
    core_values: list[str],
    decisions: list[WeeklyDriftReviewerDecision],
    drift_result: DriftDetectorResult,
) -> WeeklyDigest:
    """Build approved Weekly Drift Detection output from Reviewer decisions."""
    resolved_end = _parse_iso_date(week_end)
    profile, history, entry_texts = _load_persona_context(
        persona_id,
        wrangled_dir,
        history_end=resolved_end,
    )
    return build_weekly_drift_reviewer_digest_from_entries(
        persona_id=persona_id,
        persona_name=profile.get("name"),
        week_start=week_start,
        week_end=week_end,
        core_values=core_values,
        entries=[
            WeeklyDriftReviewerEntry(
                date=entry.date,
                t_index=entry.t_index,
                text=entry_texts[(entry.date, entry.t_index)],
            )
            for entry in history
        ],
        decisions=decisions,
        drift_result=drift_result,
    )


def build_coach_digest_prompt_inputs(
    digest: WeeklyDigest,
    config_path: Path = SCHWARTZ_CONFIG_PATH,
) -> dict[str, object]:
    """Build the shared factual inputs for Coach Digest generation and review."""
    value_map = _load_schwartz_value_map(config_path)
    compass_context_lines = _summarize_compass_context(
        value_map,
        digest.core_values,
    )
    if digest.goal_context:
        compass_context_lines.append(
            f'- User-confirmed current focus: "{digest.goal_context}"'
        )
    evidence_lines = [
        (
            f"- {snippet.date} | {_evidence_role(snippet.direction)} | "
            f"internal Schwartz label(s): "
            f"{', '.join(_format_dim_name(d) for d in snippet.dimensions) or 'none'} "
            f"| excerpt: {json.dumps(snippet.excerpt, ensure_ascii=False)}"
        )
        for snippet in digest.evidence
    ]
    return {
        "persona_name": digest.persona_name or "Not supplied; address the user as you",
        "week_window": f"{digest.week_start} to {digest.week_end}",
        "response_policy": _coach_digest_policy(digest),
        "compass_context_lines": compass_context_lines,
        "drift_summary_lines": _drift_summary_lines(digest),
        "state_comparison_lines": _state_comparison_lines(digest),
        "evidence_lines": evidence_lines,
    }


def render_digest_messages(digest: WeeklyDigest) -> tuple[str, str]:
    """Render trusted instructions and serialized Coach Digest input data."""
    prompt = load_prompt("weekly_digest_coach")
    prompt_inputs = build_coach_digest_prompt_inputs(digest)
    instruction_placeholders = {
        "persona_name": "UNTRUSTED_INPUT_SUPPLIED_SEPARATELY",
        "week_window": "UNTRUSTED_INPUT_SUPPLIED_SEPARATELY",
        "response_policy": "UNTRUSTED_INPUT_SUPPLIED_SEPARATELY",
        "compass_context_lines": [],
        "drift_summary_lines": [],
        "state_comparison_lines": [],
        "evidence_lines": [],
    }
    rendered = cast(str, prompt.render(**instruction_placeholders)).strip()
    lead, data_marker, remainder = rendered.partition("\nPreferred name:")
    _data_section, output_marker, output_rules = remainder.partition(
        "\n\nReturn JSON with exactly these keys:"
    )
    if not data_marker or not output_marker:
        raise ValueError(
            "Coach Digest prompt does not contain the expected data boundary"
        )
    instructions = f"{lead}\n\nReturn JSON with exactly these keys:{output_rules}"
    return protect_instructions(instructions), serialize_untrusted_data(prompt_inputs)


def render_digest_prompt(digest: WeeklyDigest) -> str:
    """Render the exact two-message Coach Digest request for Inspect."""
    instructions, input_data = render_digest_messages(digest)
    return render_live_prompt_receipt(
        instructions=instructions,
        input_data=input_data,
    )


def render_digest_markdown(digest: WeeklyDigest) -> str:
    """Render a deterministic markdown digest artifact."""
    persona_label = digest.persona_name or digest.persona_id
    core_value_text = (
        ", ".join(_format_dim_name(dim) for dim in digest.core_values)
        or "None captured"
    )
    drift_state_text = (
        ", ".join(
            f"{_format_dim_name(core_value)}={state}"
            for core_value, state in digest.drift_states.items()
        )
        or "No confirmed Drift"
    )
    lines = [
        f"# Weekly Drift Detection: {persona_label}",
        "",
        f"- Persona ID: `{digest.persona_id}`",
        f"- Window: `{digest.week_start}` to `{digest.week_end}`",
        f"- Response mode: `{digest.response_mode}` (`{digest.mode_source}`)",
        f"- Mode rationale: {digest.mode_rationale}",
        f"- Signal source: `{digest.signal_source}`",
        f"- Journal Entries reviewed: `{digest.n_entries}`",
        f"- Overall mean alignment: `{_format_optional_score(digest.overall_mean)}`",
        f"- Declared Core Values: {core_value_text}",
        f"- Drift states: {drift_state_text}",
        "",
        "## Tensions",
        ", ".join(_format_dim_name(d) for d in digest.top_tensions)
        or "None clear this week",
        "",
        "## Strengths",
        ", ".join(_format_dim_name(d) for d in digest.top_strengths)
        or "None clear this week",
        "",
        "## Dimension Summary",
        "",
        "| Dimension | Mean | -1 | 0 | +1 |",
        "|---|---:|---:|---:|---:|",
    ]
    if digest.overall_uncertainty is not None:
        lines.insert(8, f"- Overall uncertainty: `{digest.overall_uncertainty:.3f}`")

    if digest.dimensions:
        for dim in digest.dimensions:
            lines.append(
                "| "
                f"{_format_dim_name(dim.dimension)} | {dim.mean_score:.3f} | "
                f"{dim.pct_neg:.0%} | {dim.pct_neutral:.0%} | {dim.pct_pos:.0%} |"
            )
    else:
        lines.append(
            "| No numeric summary in the approved path | N/A | N/A | N/A | N/A |"
        )

    lines.extend(["", "## Evidence Snippets", ""])
    for snippet in digest.evidence:
        dimension_text = ", ".join(
            _format_dim_name(dimension) for dimension in snippet.dimensions
        )
        lines.append(
            f"- `{snippet.date}` entry `{snippet.t_index}` "
            f"({snippet.direction}, dims={dimension_text}, "
            f"mean={_format_optional_score(snippet.score_mean)}): {snippet.excerpt}"
        )

    if digest.coach_narrative is not None:
        lines.extend(
            [
                "",
                "## Coach Digest Response",
                "",
                f"### Weekly Mirror\n{digest.coach_narrative.weekly_mirror}",
                "",
                "### Tension Explanation\n"
                f"{digest.coach_narrative.tension_explanation}",
                "",
                "### Reflective Question\n"
                f"{digest.coach_narrative.reflective_question}",
            ]
        )

    if digest.validation is not None:
        lines.extend(["", "## Validation", ""])
        for check in digest.validation.checks:
            status = "pass" if check.passed else "fail"
            lines.append(f"- `{check.name}`: {status} - {check.details}")

    return "\n".join(lines).strip() + "\n"


async def generate_weekly_digest_coach(
    digest: WeeklyDigest,
    llm_complete: LLMCompleteFn,
) -> tuple[CoachNarrative | None, str]:
    """Generate a Coach Digest response from Weekly Drift Detection output."""
    diagnostic, prompt = await generate_weekly_digest_coach_diagnostic(
        digest,
        llm_complete,
    )
    return diagnostic.narrative, prompt


async def generate_weekly_digest_coach_diagnostic(
    digest: WeeklyDigest,
    llm_complete: LLMCompleteFn,
    *,
    repair_requirements: Sequence[str] | None = None,
) -> tuple[CoachDigestDiagnostic, str]:
    """Generate and diagnose one Coach Digest response."""
    instructions, input_data = render_digest_messages(digest)
    if repair_requirements:
        requirements = "\n".join(
            f"- {requirement}" for requirement in repair_requirements
        )
        instructions += (
            "\n\nA prior response failed Coach Digest Validations. Generate a "
            "new response. Correct each failure below. Do not discuss the "
            f"failures in the response.\n{requirements}"
        )
    prompt = render_live_prompt_receipt(
        instructions=instructions,
        input_data=input_data,
    )
    raw_json = await llm_complete(
        input_data,
        WEEKLY_DIGEST_COACH_RESPONSE_FORMAT,
        instructions,
    )
    if not raw_json:
        return (
            CoachDigestDiagnostic(
                persona_id=digest.persona_id,
                week_start=digest.week_start,
                week_end=digest.week_end,
                accepted=False,
                failure_stage="no_response",
                failure_details=[
                    "The provider adapter returned no response text."
                ],
            ),
            prompt,
        )

    try:
        payload = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError) as exc:
        return (
            CoachDigestDiagnostic(
                persona_id=digest.persona_id,
                week_start=digest.week_start,
                week_end=digest.week_end,
                accepted=False,
                failure_stage="json_parse",
                failure_details=[str(exc)],
                raw_output=raw_json,
            ),
            prompt,
        )
    if not isinstance(payload, dict):
        return (
            CoachDigestDiagnostic(
                persona_id=digest.persona_id,
                week_start=digest.week_start,
                week_end=digest.week_end,
                accepted=False,
                failure_stage="json_parse",
                failure_details=["The JSON response was not an object."],
                raw_output=raw_json,
            ),
            prompt,
        )

    try:
        narrative = CoachNarrative.model_validate(payload)
    except Exception as exc:
        return (
            CoachDigestDiagnostic(
                persona_id=digest.persona_id,
                week_start=digest.week_start,
                week_end=digest.week_end,
                accepted=False,
                failure_stage="schema_validation",
                failure_details=[str(exc)],
                raw_output=raw_json,
            ),
            prompt,
        )

    validation = validate_weekly_digest_narrative(digest, narrative)
    failed_checks = [
        f"{check.name}: {check.details}"
        for check in validation.checks
        if not check.passed
    ]
    accepted = validation.all_passed
    return (
        CoachDigestDiagnostic(
            persona_id=digest.persona_id,
            week_start=digest.week_start,
            week_end=digest.week_end,
            accepted=accepted,
            failure_stage=None if accepted else "coach_validation",
            failure_details=failed_checks,
            raw_output=raw_json,
            narrative=narrative,
            validation=validation,
        ),
        prompt,
    )


def _extract_quoted_phrases(text: str) -> list[str]:
    """Extract straight or typographic quotes for groundedness checks."""
    return [
        match.strip()
        for match in re.findall(r'["“]([^"”]+)["”]', text)
        if match.strip()
    ]


def _detect_value_label_leakage(
    text: str,
    config_path: Path = SCHWARTZ_CONFIG_PATH,
) -> list[str]:
    """Return Schwartz value labels surfaced verbatim in a Coach Digest response."""
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    labels = list((raw.get("values") or {}).keys())

    lowered = text.lower()
    leaked: list[str] = []
    for label in labels:
        pattern = rf"\b{re.escape(label.lower())}\b"
        if re.search(pattern, lowered):
            leaked.append(label)
    return leaked


def validate_weekly_digest_narrative(
    digest: WeeklyDigest,
    narrative: CoachNarrative,
    min_words: int = 25,
    max_words: int = 180,
    config_path: Path = SCHWARTZ_CONFIG_PATH,
) -> DigestValidation:
    """Run Coach Digest Validations on one response."""
    combined_text = " ".join(
        [
            narrative.weekly_mirror.strip(),
            narrative.tension_explanation.strip(),
            narrative.reflective_question.strip(),
        ]
    ).strip()
    word_count = len(combined_text.split())

    source_texts = [snippet.excerpt for snippet in digest.evidence]
    grounded_quotes = [
        quote
        for quote in _extract_quoted_phrases(combined_text)
        if any(quote.lower() in source.lower() for source in source_texts)
    ]

    score_language = [
        "score",
        "scores",
        "scored",
        "alignment",
        "aligned",
        "misaligned",
        "mean=",
    ]
    non_circularity_passed = not any(
        term in combined_text.lower() for term in score_language
    )

    leaked_values = _detect_value_label_leakage(combined_text, config_path)
    value_leakage_passed = not leaked_values

    positive_transition_patterns = {
        "better": (
            r"\b(?:are|became|become|becoming|feel|feels|felt|get|gets|getting|"
            r"got|is|look|looks|seem|seems|was|were)\s+(?:\w+\s+){0,2}better\b|"
            r"\bbetter\s+(?:now|than before|this week)\b"
        ),
        "improve": r"\bimprov\w*\b",
        "progress": r"\bprogress\w*\b",
        "recover": r"\brecover\w*\b",
        "success": r"\bsuccess\w*\b",
    }
    lowered_text = combined_text.lower()
    found_transition_terms = sorted(
        label
        for label, pattern in positive_transition_patterns.items()
        if re.search(pattern, lowered_text)
    )
    end_claim_patterns = [
        r"\bdid not continue\b",
        r"\bdidn't continue\b",
        r"\bpattern (?:has )?ended\b",
        r"\bpattern (?:has )?stopped\b",
    ]
    has_end_claim = any(
        re.search(pattern, lowered_text) for pattern in end_claim_patterns
    )
    supports_end_claim = any(
        comparison.change == "active_drift_ended"
        and comparison.end_reason == "not_conflict"
        for comparison in digest.state_comparisons
    )
    state_claims_passed = not found_transition_terms and (
        not has_end_claim or supports_end_claim
    )
    state_claim_details = "Response makes only supported current-state claims."
    if found_transition_terms:
        state_claim_details = (
            "Response makes an unsupported positive transition claim with: "
            f"{', '.join(found_transition_terms)}."
        )
    elif has_end_claim and not supports_end_claim:
        state_claim_details = (
            "Response says a pattern ended without an active-to-no-active "
            "comparison ended by a Not Conflict decision."
        )

    checks = [
        ValidationCheck(
            name="groundedness",
            passed=bool(grounded_quotes),
            details=(
                f"Found {len(grounded_quotes)} quoted phrase(s) that match "
                "selected evidence."
                if grounded_quotes
                else "No quoted phrase from selected evidence was detected."
            ),
        ),
        ValidationCheck(
            name="non_circularity",
            passed=non_circularity_passed,
            details=(
                "Response avoids raw scoring and alignment terminology."
                if non_circularity_passed
                else (
                    "Response uses raw scoring or alignment terminology instead of "
                    "reflective language."
                )
            ),
        ),
        ValidationCheck(
            name="value_leakage",
            passed=value_leakage_passed,
            details=(
                "Response avoids naming raw Schwartz value labels."
                if value_leakage_passed
                else (
                    "Response names raw Schwartz value labels: "
                    f"{', '.join(leaked_values)}."
                )
            ),
        ),
        ValidationCheck(
            name="state_claims",
            passed=state_claims_passed,
            details=state_claim_details,
        ),
        ValidationCheck(
            name="length",
            passed=min_words <= word_count <= max_words,
            details=(
                f"Combined response length is {word_count} words "
                f"(target {min_words}-{max_words})."
            ),
        ),
    ]

    return DigestValidation(
        grounded_quotes=grounded_quotes,
        word_count=word_count,
        checks=checks,
    )


def attach_coach_artifacts(
    digest: WeeklyDigest,
    coach_narrative: CoachNarrative | None,
    validation: DigestValidation | None = None,
) -> WeeklyDigest:
    """Attach a Coach Digest response to Weekly Drift Detection output."""
    return digest.model_copy(
        update={
            "coach_narrative": coach_narrative,
            "validation": validation,
        }
    )


def persist_weekly_digest_record(
    digest: WeeklyDigest, parquet_path: Path
) -> pl.DataFrame:
    """Upsert one digest row into a consolidated parquet artifact."""
    record = {
        "persona_id": digest.persona_id,
        "week_start": digest.week_start,
        "week_end": digest.week_end,
        "persona_name": digest.persona_name,
        "response_mode": digest.response_mode,
        "mode_source": digest.mode_source,
        "mode_rationale": digest.mode_rationale,
        "signal_source": digest.signal_source,
        "n_entries": digest.n_entries,
        "overall_mean": digest.overall_mean,
        "overall_uncertainty": digest.overall_uncertainty,
        "core_values_json": json.dumps(digest.core_values),
        "goal_context": digest.goal_context,
        "drift_states_json": json.dumps(digest.drift_states),
        "drift_details_json": json.dumps(
            {
                core_value: detail.model_dump()
                for core_value, detail in digest.drift_details.items()
            }
        ),
        "state_comparisons_json": json.dumps(
            [comparison.model_dump() for comparison in digest.state_comparisons]
        ),
        "drift_reasons_json": json.dumps(digest.drift_reasons),
        "top_tensions_json": json.dumps(digest.top_tensions),
        "top_strengths_json": json.dumps(digest.top_strengths),
        "dimensions_json": json.dumps([row.model_dump() for row in digest.dimensions]),
        "evidence_json": json.dumps([row.model_dump() for row in digest.evidence]),
        "coach_narrative_json": (
            json.dumps(digest.coach_narrative.model_dump())
            if digest.coach_narrative is not None
            else None
        ),
        "validation_json": (
            json.dumps(digest.validation.model_dump())
            if digest.validation is not None
            else None
        ),
    }

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = parquet_path.with_suffix(f"{parquet_path.suffix}.lock")

    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        new_df = pl.DataFrame([record])
        if parquet_path.exists():
            existing = pl.read_parquet(parquet_path)
            for name, dtype in new_df.schema.items():
                if name not in existing.columns:
                    existing = existing.with_columns(
                        pl.lit(None).cast(dtype).alias(name)
                    )
            for name, dtype in existing.schema.items():
                if name not in new_df.columns:
                    new_df = new_df.with_columns(pl.lit(None).cast(dtype).alias(name))
            # Reconcile dtype drift: older rows may carry all-null (Null dtype)
            # columns (e.g. coach_narrative_json before narratives existed) that
            # clash with a now-populated dtype. Cast the all-null side to match.
            for name, new_dtype in new_df.schema.items():
                existing_dtype = existing.schema.get(name)
                if existing_dtype is not None and existing_dtype != new_dtype:
                    if existing_dtype == pl.Null:
                        existing = existing.with_columns(pl.col(name).cast(new_dtype))
                    elif new_dtype == pl.Null:
                        new_df = new_df.with_columns(pl.col(name).cast(existing_dtype))
            existing = existing.filter(
                ~(
                    (pl.col("persona_id") == digest.persona_id)
                    & (pl.col("week_start") == digest.week_start)
                    & (pl.col("week_end") == digest.week_end)
                )
            )
            existing = existing.select(new_df.columns)
            new_df = pl.concat([existing, new_df], how="vertical")

        new_df.write_parquet(parquet_path)

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return new_df


def _default_output_stem(digest: WeeklyDigest) -> str:
    return f"{digest.persona_id}_{digest.week_end}"


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Weekly Drift Detection output."
    )
    parser.add_argument(
        "--persona-id", required=True, help="Persona ID (without prefix)."
    )
    parser.add_argument("--start-date", default=None, help="Window start YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Window end YYYY-MM-DD.")
    parser.add_argument(
        "--response-mode",
        default=None,
        choices=[
            "stable",
            "rut",
            "crash",
            "evolution",
            "high_uncertainty",
            "mixed_state",
            "background_strain",
        ],
        help="Optional manual response mode override.",
    )
    parser.add_argument(
        "--drift-result-json",
        default=None,
        help="Path to an upstream drift-detection JSON payload.",
    )
    parser.add_argument(
        "--labels-path",
        default="logs/judge_labels/judge_labels.parquet",
        help="Path to consolidated judge labels parquet.",
    )
    parser.add_argument(
        "--signals-path",
        default=None,
        help="Optional path to live VIF timeline signals parquet.",
    )
    parser.add_argument(
        "--wrangled-dir",
        default="logs/wrangled",
        help="Directory containing wrangled persona markdown files.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/exports/weekly_digests",
        help="Directory for output artifacts.",
    )
    parser.add_argument(
        "--parquet-path",
        default="logs/exports/weekly_digests/weekly_digests.parquet",
        help="Path to consolidated Weekly Drift Detection output parquet.",
    )
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()
    drift_result = None
    if args.drift_result_json:
        drift_result = DriftDetectionResult.model_validate_json(
            Path(args.drift_result_json).read_text()
        )

    digest = build_weekly_digest(
        persona_id=args.persona_id,
        labels_path=Path(args.labels_path),
        wrangled_dir=Path(args.wrangled_dir),
        signals_path=Path(args.signals_path) if args.signals_path else None,
        start_date=args.start_date,
        end_date=args.end_date,
        drift_result=drift_result,
        response_mode=args.response_mode,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _default_output_stem(digest)

    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    prompt_path = output_dir / f"{stem}.prompt.txt"

    json_path.write_text(json.dumps(digest.model_dump(), indent=2) + "\n")
    md_path.write_text(render_digest_markdown(digest))
    prompt_path.write_text(render_digest_prompt(digest))
    persist_weekly_digest_record(digest, Path(args.parquet_path))

    print(f"Wrote digest JSON: {json_path}")
    print(f"Wrote digest markdown: {md_path}")
    print(f"Wrote coach prompt: {prompt_path}")
    print(f"Updated digest parquet: {args.parquet_path}")


if __name__ == "__main__":
    main()
