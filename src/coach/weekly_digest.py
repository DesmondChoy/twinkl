"""Weekly digest helpers for the Coach layer.

This module builds a structured weekly digest from offline artifacts,
renders a full-context Coach prompt, validates generated narratives,
and persists digest records for later analysis.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable

import polars as pl
import yaml

from prompts import load_prompt
from src.coach.mode_logic import (
    WeeklyModeSignals,
    has_acute_distress_context,
    infer_response_mode,
)
from src.coach.schemas import (
    CoachNarrative,
    CoachResponseMode,
    DriftDetectionResult,
    DigestValidation,
    DimensionDigest,
    EvidenceSnippet,
    JournalHistoryEntry,
    ValidationCheck,
    WEEKLY_DIGEST_COACH_RESPONSE_FORMAT,
    WeeklyDigest,
)
from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.wrangling.parse_wrangled_data import parse_wrangled_file

ALIGNMENT_COLUMNS = [f"alignment_{value}" for value in SCHWARTZ_VALUE_ORDER]
SCHWARTZ_CONFIG_PATH = Path("config/schwartz_values.yaml")
LLMCompleteFn = Callable[[str, dict | None], Awaitable[str | None]]


def _parse_iso_date(raw: str) -> date:
    """Parse YYYY-MM-DD date strings."""
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _format_dim_name(dim: str) -> str:
    """Format snake_case value names for human-readable output."""
    return dim.replace("_", " ").title()


def _format_dimension_list(dimensions: list[str], empty_label: str) -> str:
    """Format a dimension list for human-readable output."""
    if not dimensions:
        return empty_label
    return ", ".join(_format_dim_name(dim) for dim in dimensions)


def _truncate_excerpt(text: str, max_words: int = 40) -> str:
    """Keep excerpts compact for digest readability."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip() + "..."


def _to_snake_case(value_name: str) -> str:
    """Convert value labels like 'Self Direction' to 'self_direction'."""
    return value_name.strip().lower().replace("-", " ").replace(" ", "_")


def _load_schwartz_value_map(config_path: Path = SCHWARTZ_CONFIG_PATH) -> dict[str, dict]:
    """Load Schwartz value elaborations keyed by snake_case dimension name."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    values = raw.get("values", {})
    return {_to_snake_case(display_name): details for display_name, details in values.items()}


def _summarize_value_context(
    value_map: dict[str, dict],
    dimensions: list[str],
) -> list[str]:
    """Build compact value elaborations for prompt injection."""
    lines: list[str] = []
    for dim in dimensions:
        details = value_map.get(dim, {})
        core_motivation = str(details.get("core_motivation", "")).strip()
        tension_values = details.get("opposing_tension_values") or []

        context_bits: list[str] = [f"- {_format_dim_name(dim)}"]
        if core_motivation:
            context_bits.append(f"motivation: {core_motivation}")
        if tension_values:
            tensions = ", ".join(str(value) for value in tension_values[:3])
            context_bits.append(f"common tensions: {tensions}")
        lines.append(" | ".join(context_bits))

    return lines


def _load_persona_context(
    persona_id: str,
    wrangled_dir: Path,
    history_end_date: date | None = None,
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
        if history_end_date is not None and entry_date > history_end_date:
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


def _window_history(
    journal_history: list[JournalHistoryEntry],
    start_date: date,
    end_date: date,
) -> list[JournalHistoryEntry]:
    """Return journal entries that fall within the requested digest window."""
    return [
        entry
        for entry in journal_history
        if start_date <= _parse_iso_date(entry.date) <= end_date
    ]


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


def _rank_dimensions(
    dim_means: list[tuple[str, float]],
    core_values: list[str],
    direction: str,
    limit: int,
    exclude: set[str] | None = None,
) -> list[str]:
    """Rank digest focus dimensions, preferring declared core values when useful."""
    if direction not in {"tension", "strength"}:
        raise ValueError("direction must be 'tension' or 'strength'")
    excluded = exclude or set()

    if direction == "tension":
        primary_core = [item for item in dim_means if item[0] in core_values and item[1] < 0]
        primary_other = [item for item in dim_means if item[0] not in core_values and item[1] < 0]
        primary_core.sort(key=lambda item: item[1])
        primary_other.sort(key=lambda item: item[1])
        buckets = (primary_core, primary_other)
    else:
        primary_core = [item for item in dim_means if item[0] in core_values and item[1] > 0]
        primary_other = [item for item in dim_means if item[0] not in core_values and item[1] > 0]
        primary_core.sort(key=lambda item: item[1], reverse=True)
        primary_other.sort(key=lambda item: item[1], reverse=True)
        buckets = (primary_core, primary_other)

    ranked: list[str] = []
    for bucket in buckets:
        for dim, _score in bucket:
            if dim not in ranked and dim not in excluded:
                ranked.append(dim)
            if len(ranked) >= limit:
                return ranked

    return ranked[:limit]


def _select_row_dimensions(row: dict, candidate_dims: list[str], direction: str) -> list[str]:
    """Select the most representative focus dimensions for one evidence row."""
    scored_dims = [(dim, float(row[f"alignment_{dim}"])) for dim in candidate_dims]
    reverse = direction == "aligned"
    scored_dims.sort(key=lambda item: item[1], reverse=reverse)

    if direction == "misaligned":
        filtered = [dim for dim, score in scored_dims if score < 0]
    else:
        filtered = [dim for dim, score in scored_dims if score > 0]

    if filtered:
        return filtered[:2]
    return [scored_dims[0][0]] if scored_dims else []


def _select_soft_strain_entry(
    rows: list[dict],
    selected_keys: set[tuple[str, int]],
) -> dict | None:
    """Select a softer strain example when the week is not cleanly misaligned."""
    for row in rows:
        key = (row["date"], int(row["t_index"]))
        if key not in selected_keys:
            return row
    return None


def _has_mixed_core_polarity(labels: pl.DataFrame, core_values: list[str]) -> bool:
    """Detect whether any declared core value swings both negative and positive in-window."""
    for dim in core_values:
        column = f"alignment_{dim}"
        if column not in labels.columns:
            continue
        if int(labels[column].min()) < 0 and int(labels[column].max()) > 0:
            return True
    return False


def _format_evidence_dimensions(dimensions: list[str]) -> str:
    """Format evidence dimensions with an explicit empty-state label."""
    if not dimensions:
        return "none"
    return ", ".join(_format_dim_name(dim) for dim in dimensions)


def build_weekly_digest(
    persona_id: str,
    labels_path: Path,
    wrangled_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    drift_result: DriftDetectionResult | None = None,
    response_mode: CoachResponseMode | None = None,
) -> WeeklyDigest:
    """Build a structured weekly digest payload for one persona."""
    labels = pl.read_parquet(labels_path).filter(pl.col("persona_id") == persona_id)
    if labels.is_empty():
        raise ValueError(f"No labels found for persona_id={persona_id}")

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

    profile, journal_history, entry_texts = _load_persona_context(
        persona_id,
        wrangled_dir,
        history_end_date=resolved_end,
    )
    core_values = profile.get("core_values") or []
    window_history = _window_history(journal_history, resolved_start, resolved_end)
    acute_distress = has_acute_distress_context(window_history)
    dim_rows: list[DimensionDigest] = []
    dim_means: list[tuple[str, float]] = []
    n_rows = labels.height

    for dim, col in zip(SCHWARTZ_VALUE_ORDER, ALIGNMENT_COLUMNS):
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

    top_strengths = _rank_dimensions(dim_means, core_values, direction="strength", limit=2)
    if not top_strengths:
        positive_dims = [dim for dim, score in dim_means if score > 0]
        top_strengths = positive_dims[:2]

    suppressed_tensions: set[str] = set()
    if acute_distress:
        negative_core_tensions = [
            dim for dim, score in dim_means if dim in core_values and score < 0
        ]
        if not negative_core_tensions:
            suppressed_tensions.update({"hedonism", "stimulation"})

    top_tensions = _rank_dimensions(
        dim_means,
        core_values,
        direction="tension",
        limit=3,
        exclude=set(top_strengths) | suppressed_tensions,
    )

    tension_cols = [f"alignment_{dim}" for dim in top_tensions]
    strength_cols = [f"alignment_{dim}" for dim in top_strengths]
    score_columns = [pl.mean_horizontal(ALIGNMENT_COLUMNS).alias("entry_mean")]
    if tension_cols:
        score_columns.append(pl.mean_horizontal(tension_cols).alias("tension_score"))
    else:
        score_columns.append(pl.lit(0.0).alias("tension_score"))
    if strength_cols:
        score_columns.append(pl.mean_horizontal(strength_cols).alias("strength_score"))
    else:
        score_columns.append(pl.lit(0.0).alias("strength_score"))

    with_entry_scores = labels.with_columns(score_columns)

    n_entries = with_entry_scores.height
    mis_target = 0 if not top_tensions else (1 if n_entries <= 2 else 2)
    aligned_target = 1 if top_strengths else 0

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

    drift_reasons: list[str] = []
    if drift_result is not None:
        resolved_mode = drift_result.response_mode
        mode_source = drift_result.source
        mode_rationale = drift_result.rationale
        drift_reasons = list(drift_result.reasons)
    elif response_mode is not None:
        resolved_mode = response_mode
        mode_source = "manual_override"
        mode_rationale = "Response mode supplied explicitly for testing or manual review."
    else:
        mode_decision = infer_response_mode(
            WeeklyModeSignals(
                overall_mean=overall_mean,
                top_tensions=top_tensions,
                top_strengths=top_strengths,
                core_values=core_values,
                window_entries=window_history,
                has_mixed_core_polarity=_has_mixed_core_polarity(labels, core_values),
            )
        )
        resolved_mode = mode_decision.response_mode
        mode_source = mode_decision.mode_source
        mode_rationale = mode_decision.mode_rationale

    if resolved_mode == "high_uncertainty":
        normalized_evidence: list[EvidenceSnippet] = []
        for snippet in evidence_rows:
            if snippet.direction == "misaligned":
                normalized_evidence.append(
                    snippet.model_copy(update={"direction": "strain"})
                )
            else:
                normalized_evidence.append(snippet)
        evidence_rows = normalized_evidence

    if resolved_mode in {"high_uncertainty", "background_strain"}:
        soft_row = _select_soft_strain_entry(mis_candidates + aligned_candidates, selected_keys)
        if soft_row is not None:
            key = (soft_row["date"], int(soft_row["t_index"]))
            selected_keys.add(key)
            raw_text = entry_texts.get(key, "")
            soft_dims = top_tensions[:2]
            strain_snippet = EvidenceSnippet(
                date=soft_row["date"],
                t_index=int(soft_row["t_index"]),
                direction="strain",
                dimensions=soft_dims,
                score_mean=float(soft_row["entry_mean"]),
                excerpt=_truncate_excerpt(raw_text),
            )
            evidence_rows.insert(0, strain_snippet)

    return WeeklyDigest(
        persona_id=persona_id,
        persona_name=profile.get("name"),
        week_start=resolved_start.isoformat(),
        week_end=resolved_end.isoformat(),
        response_mode=resolved_mode,
        mode_source=mode_source,
        mode_rationale=mode_rationale,
        n_entries=n_rows,
        overall_mean=overall_mean,
        core_values=core_values,
        drift_reasons=drift_reasons,
        top_tensions=top_tensions,
        top_strengths=top_strengths,
        dimensions=dim_rows,
        evidence=evidence_rows,
        journal_history=journal_history,
    )


def _build_prompt_inputs(
    digest: WeeklyDigest,
    config_path: Path = SCHWARTZ_CONFIG_PATH,
) -> dict[str, object]:
    """Build template inputs for the weekly Coach prompt."""
    value_map = _load_schwartz_value_map(config_path)
    focus_dimensions = list(dict.fromkeys(digest.core_values + digest.top_tensions + digest.top_strengths))

    dimension_lines = [
        (
            f"- {_format_dim_name(dim.dimension)}: mean={dim.mean_score:.3f}, "
            f"neg={dim.pct_neg:.0%}, neutral={dim.pct_neutral:.0%}, pos={dim.pct_pos:.0%}"
        )
        for dim in digest.dimensions
    ]
    evidence_lines = [
        (
            f"- {snippet.date} (entry {snippet.t_index}, {snippet.direction}, "
            f"dims={_format_evidence_dimensions(snippet.dimensions)}, "
            f"mean={snippet.score_mean:.3f}): {snippet.excerpt}"
        )
        for snippet in digest.evidence
    ]
    history_lines = [
        (
            f"- {entry.date} (entry {entry.t_index}"
            f"{', has_response' if entry.has_response else ''}): {entry.content}"
        )
        for entry in digest.journal_history
    ]

    return {
        "persona_id": digest.persona_id,
        "persona_name": digest.persona_name or "Unknown Persona",
        "week_start": digest.week_start,
        "week_end": digest.week_end,
        "response_mode": digest.response_mode,
        "mode_source": digest.mode_source,
        "mode_rationale": digest.mode_rationale,
        "n_entries": digest.n_entries,
        "overall_mean": f"{digest.overall_mean:.3f}",
        "core_values": ", ".join(_format_dim_name(dim) for dim in digest.core_values) or "None captured",
        "drift_reasons": digest.drift_reasons,
        "top_tensions": _format_dimension_list(
            digest.top_tensions, "None clear this week"
        ),
        "top_strengths": _format_dimension_list(
            digest.top_strengths, "None clear this week"
        ),
        "dimension_lines": dimension_lines,
        "evidence_lines": evidence_lines,
        "history_lines": history_lines,
        "value_context_lines": _summarize_value_context(value_map, focus_dimensions),
    }


def render_digest_prompt(digest: WeeklyDigest) -> str:
    """Render Coach-generation prompt using full digest context."""
    prompt = load_prompt("weekly_digest_coach")
    return prompt.render(**_build_prompt_inputs(digest))


def render_digest_markdown(digest: WeeklyDigest) -> str:
    """Render a deterministic markdown digest artifact."""
    persona_label = digest.persona_name or digest.persona_id
    lines = [
        f"# Weekly Alignment Digest: {persona_label}",
        "",
        f"- Persona ID: `{digest.persona_id}`",
        f"- Window: `{digest.week_start}` to `{digest.week_end}`",
        f"- Response mode: `{digest.response_mode}` (`{digest.mode_source}`)",
        f"- Mode rationale: {digest.mode_rationale}",
        (
            f"- Drift reasons: {', '.join(digest.drift_reasons)}"
            if digest.drift_reasons
            else "- Drift reasons: none recorded"
        ),
        f"- Entries scored: `{digest.n_entries}`",
        f"- Overall mean alignment: `{digest.overall_mean:.3f}`",
        f"- Declared core values: {', '.join(_format_dim_name(dim) for dim in digest.core_values) or 'None captured'}",
        "",
        "## Tensions",
        _format_dimension_list(digest.top_tensions, "None clear this week"),
        "",
        "## Strengths",
        _format_dimension_list(digest.top_strengths, "None clear this week"),
        "",
        "## Dimension Summary",
        "",
        "| Dimension | Mean | -1 | 0 | +1 |",
        "|---|---:|---:|---:|---:|",
    ]

    for dim in digest.dimensions:
        lines.append(
            "| "
            f"{_format_dim_name(dim.dimension)} | {dim.mean_score:.3f} | "
            f"{dim.pct_neg:.0%} | {dim.pct_neutral:.0%} | {dim.pct_pos:.0%} |"
        )

    lines.extend(["", "## Evidence Snippets", ""])
    for snippet in digest.evidence:
        lines.append(
            f"- `{snippet.date}` entry `{snippet.t_index}` "
            f"({snippet.direction}, dims={_format_evidence_dimensions(snippet.dimensions)}, "
            f"mean={snippet.score_mean:.3f}): {snippet.excerpt}"
        )

    lines.extend(["", "## Full Journal History", ""])
    for entry in digest.journal_history:
        suffix = ", has_response" if entry.has_response else ""
        lines.append(f"- `{entry.date}` entry `{entry.t_index}`{suffix}: {entry.content}")

    if digest.coach_narrative is not None:
        lines.extend(
            [
                "",
                "## Coach Narrative",
                "",
                f"### Weekly Mirror\n{digest.coach_narrative.weekly_mirror}",
                "",
                f"### Tension Explanation\n{digest.coach_narrative.tension_explanation}",
                "",
                f"### Reflective Question\n{digest.coach_narrative.reflective_question}",
            ]
        )

    if digest.validation is not None:
        lines.extend(["", "## Validation", ""])
        for check in digest.validation.checks:
            status = "pass" if check.passed else "fail"
            lines.append(f"- `{check.name}`: {status} - {check.details}")

    return "\n".join(lines).strip() + "\n"


def _safe_load_json_object(raw_json: str) -> dict | None:
    """Parse a JSON object safely, returning None on malformed payloads."""
    try:
        data = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


async def generate_weekly_digest_coach(
    digest: WeeklyDigest,
    llm_complete: LLMCompleteFn,
) -> tuple[CoachNarrative | None, str]:
    """Generate a structured Coach narrative from a weekly digest."""
    prompt = render_digest_prompt(digest)
    raw_json = await llm_complete(prompt, WEEKLY_DIGEST_COACH_RESPONSE_FORMAT)
    if not raw_json:
        return None, prompt

    payload = _safe_load_json_object(raw_json)
    if payload is None:
        return None, prompt

    try:
        narrative = CoachNarrative.model_validate(payload)
    except Exception:
        return None, prompt

    return narrative, prompt


def _extract_quoted_phrases(text: str) -> list[str]:
    """Extract straight-quoted snippets for groundedness checks."""
    return [match.strip() for match in re.findall(r'"([^"]+)"', text) if match.strip()]


def validate_weekly_digest_narrative(
    digest: WeeklyDigest,
    narrative: CoachNarrative,
    min_words: int = 25,
    max_words: int = 180,
) -> DigestValidation:
    """Run Tier 1 automated validation checks on Coach narrative output."""
    combined_text = " ".join(
        [
            narrative.weekly_mirror.strip(),
            narrative.tension_explanation.strip(),
            narrative.reflective_question.strip(),
        ]
    ).strip()
    word_count = len(combined_text.split())

    source_texts = [entry.content for entry in digest.journal_history] + [
        snippet.excerpt for snippet in digest.evidence
    ]
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
    non_circularity_passed = not any(term in combined_text.lower() for term in score_language)

    checks = [
        ValidationCheck(
            name="groundedness",
            passed=bool(grounded_quotes),
            details=(
                f"Found {len(grounded_quotes)} quoted phrase(s) that match journal history."
                if grounded_quotes
                else "No quoted evidence from journal history was detected."
            ),
        ),
        ValidationCheck(
            name="non_circularity",
            passed=non_circularity_passed,
            details=(
                "Narrative avoids raw scoring and alignment terminology."
                if non_circularity_passed
                else "Narrative uses raw scoring or alignment terminology instead of reflective language."
            ),
        ),
        ValidationCheck(
            name="length",
            passed=min_words <= word_count <= max_words,
            details=f"Combined narrative length is {word_count} words (target {min_words}-{max_words}).",
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
    """Return a new digest with Coach narrative artifacts attached."""
    return digest.model_copy(
        update={
            "coach_narrative": coach_narrative,
            "validation": validation,
        }
    )


def persist_weekly_digest_record(digest: WeeklyDigest, parquet_path: Path) -> pl.DataFrame:
    """Upsert one digest row into a consolidated parquet artifact."""
    record = {
        "persona_id": digest.persona_id,
        "week_start": digest.week_start,
        "week_end": digest.week_end,
        "persona_name": digest.persona_name,
        "response_mode": digest.response_mode,
        "mode_source": digest.mode_source,
        "mode_rationale": digest.mode_rationale,
        "drift_reasons_json": json.dumps(digest.drift_reasons),
        "n_entries": digest.n_entries,
        "overall_mean": digest.overall_mean,
        "core_values_json": json.dumps(digest.core_values),
        "top_tensions_json": json.dumps(digest.top_tensions),
        "top_strengths_json": json.dumps(digest.top_strengths),
        "dimensions_json": json.dumps([row.model_dump() for row in digest.dimensions]),
        "evidence_json": json.dumps([row.model_dump() for row in digest.evidence]),
        "journal_history_json": json.dumps(
            [row.model_dump() for row in digest.journal_history]
        ),
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
            existing = existing.filter(
                ~(
                    (pl.col("persona_id") == digest.persona_id)
                    & (pl.col("week_start") == digest.week_start)
                    & (pl.col("week_end") == digest.week_end)
                )
            )
            new_df = pl.concat([existing, new_df], how="vertical")

        new_df.write_parquet(parquet_path)

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    return new_df


def _default_output_stem(digest: WeeklyDigest) -> str:
    return f"{digest.persona_id}_{digest.week_end}"


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate weekly coach digest artifact.")
    parser.add_argument("--persona-id", required=True, help="Persona ID (without prefix).")
    parser.add_argument("--start-date", default=None, help="Window start YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Window end YYYY-MM-DD.")
    parser.add_argument(
        "--response-mode",
        default=None,
        choices=[
            "stable",
            "rut",
            "crash",
            "high_uncertainty",
            "mixed_state",
            "background_strain",
        ],
        help="Optional manual response mode override. Prefer upstream drift detection in normal use.",
    )
    parser.add_argument(
        "--drift-result-json",
        default=None,
        help="Optional JSON file containing upstream drift detection output for this week.",
    )
    parser.add_argument(
        "--labels-path",
        default="logs/judge_labels/judge_labels.parquet",
        help="Path to consolidated judge labels parquet.",
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
        help="Path to consolidated weekly digest parquet.",
    )
    return parser


def main() -> None:
    args = _build_cli_parser().parse_args()

    drift_result = None
    if args.drift_result_json:
        drift_payload = json.loads(Path(args.drift_result_json).read_text())
        drift_result = DriftDetectionResult.model_validate(drift_payload)

    digest = build_weekly_digest(
        persona_id=args.persona_id,
        labels_path=Path(args.labels_path),
        wrangled_dir=Path(args.wrangled_dir),
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
