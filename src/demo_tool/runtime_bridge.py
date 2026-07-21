"""Bridge between the demo app and the approved Weekly Drift Reviewer path.

The demo runs the approved user-facing architecture: Weekly Drift Reviewer
Decisions per Journal Entry and Core Value, then the deterministic Drift
Detector (two consecutive Conflicts for the same Core Value). There is no VIF
Critic input on this path.

Input comes from the app session (onboarding Core Values plus typed Journal
Entries, or a preloaded demo persona), so nothing here reads persona files.
Weekly receipts are cached in the session keyed by prompt hash, so re-running
after new entries only pays for new or changed weeks.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.drift_detector import DriftDetectorResult, detect_drift
from src.vif.state_encoder import concatenate_entry_text
from src.weekly_drift_reviewer import (
    OpenAIWeeklyDriftReviewer,
    VerifierAssessment,
    WeeklyDriftReviewerDecision,
    WeeklyDriftReviewerEntry,
    WeeklyDriftReviewerFn,
    WeeklyDriftReviewerReceipt,
    WeeklyDriftReviewerRequest,
    WeeklyVerifierResponse,
    _receipt,  # reused so replayed receipts share the exact live-call schema
    build_weekly_drift_reviewer_request,
    validate_weekly_drift_reviewer_response,
)

MAX_CORE_VALUES = 2

# Frozen research run under the exact approved contract (gpt-5.6-luna,
# reasoning effort low, arm weekly_without_critic) covering all synthetic demo
# personas' full journals. Reused so loading a demo persona doesn't repeat
# paid calls for weeks that were already reviewed under this contract.
FROZEN_LUNA_LOW_PATH = Path(
    "logs/experiments/artifacts/twinkl_52zz_luna_low_20260714/"
    "responses_gpt_5_6_luna_low.jsonl"
)
_FROZEN_ARM = "weekly_without_critic"


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _week_bounds(raw: str) -> tuple[date, date]:
    entry_date = _parse_date(raw)
    start = entry_date - timedelta(days=entry_date.weekday())
    return start, start + timedelta(days=6)


def normalize_core_value(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def demo_journal_from_persona(
    persona: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Convert one catalog persona into (core_values, entries) session input.

    Core Values are normalized and capped at MAX_CORE_VALUES to match the
    onboarding contract. Entries keep initial_entry/nudge_text/response_text
    separate (the same shape a live composer entry uses) rather than
    pre-joining them, so the journal UI can render the nudge thread and
    entry_display_text() can build the reviewer/digest text on demand.
    """
    core_values = list(
        dict.fromkeys(
            normalize_core_value(value)
            for value in persona.get("persona_core_values") or []
            if isinstance(value, str) and value.strip()
        )
    )[:MAX_CORE_VALUES]

    entries = [
        {
            "t_index": index,
            "date": str(entry["date"]),
            "initial_entry": entry.get("initial_entry") or "",
            "nudge_text": entry.get("nudge_text"),
            "response_text": entry.get("response_text"),
        }
        for index, entry in enumerate(persona["entries"])
    ]
    return core_values, entries


def entry_display_text(entry: dict[str, Any]) -> str:
    """The full displayed text for one Journal Entry: entry, nudge, response."""
    return concatenate_entry_text(
        entry.get("initial_entry"),
        entry.get("nudge_text"),
        entry.get("response_text"),
    )


@dataclass(frozen=True)
class _FrozenWeeklyRecord:
    """One frozen response row, reduced to what a replayed receipt needs."""

    status: str
    assessments: list[VerifierAssessment]
    resolved_model: str | None
    response_id: str | None
    usage: dict[str, int]


_FrozenIndex = dict[tuple[str, str, str], list[_FrozenWeeklyRecord]]


def _load_frozen_luna_low(path: Path = FROZEN_LUNA_LOW_PATH) -> _FrozenIndex:
    """Index frozen weekly_without_critic responses by (persona_id, week_start,
    week_end), records ordered by repeat. Callers use the first "ok" record."""
    index: _FrozenIndex = defaultdict(list)
    if not path.exists():
        return index

    rows = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    rows.sort(key=lambda row: row.get("repeat", 0))

    for row in rows:
        if row.get("arm") != _FROZEN_ARM:
            continue
        parsed = row.get("parsed") or {}
        assessments = [
            VerifierAssessment.model_validate(item)
            for item in parsed.get("assessments", [])
        ]
        key = (row["persona_id"], row["week_start"], row["week_end"])
        usage = {
            field: value
            for field, value in (row.get("usage") or {}).items()
            if isinstance(value, int)
        }
        index[key].append(
            _FrozenWeeklyRecord(
                status=row["status"],
                assessments=assessments,
                resolved_model=row.get("resolved_model"),
                response_id=row.get("response_id"),
                usage=usage,
            )
        )
    return index


_frozen_luna_low_index: _FrozenIndex | None = None


def _get_frozen_luna_low_index() -> _FrozenIndex:
    global _frozen_luna_low_index
    if _frozen_luna_low_index is None:
        _frozen_luna_low_index = _load_frozen_luna_low()
    return _frozen_luna_low_index


def build_demo_reviewer(
    persona_id: str,
    live_reviewer: WeeklyDriftReviewerFn | None = None,
) -> WeeklyDriftReviewerFn:
    """Reviewer for a demo persona: replays the frozen Luna-low run for any
    persona-week it covers, and falls back to a live call otherwise (e.g. a
    week where the user added entries beyond the persona's original journal,
    or the rare frozen record marked invalid).
    """
    index = _get_frozen_luna_low_index()
    live_reviewer = live_reviewer or OpenAIWeeklyDriftReviewer()

    async def _reviewer(
        request: WeeklyDriftReviewerRequest,
    ) -> WeeklyDriftReviewerReceipt:
        records = index.get((persona_id, request.week_start, request.week_end), [])
        record = next((r for r in records if r.status == "ok"), None)
        if record is None:
            return await live_reviewer(request)

        response = WeeklyVerifierResponse(assessments=record.assessments)
        try:
            validate_weekly_drift_reviewer_response(response, request)
        except ValueError:
            # Frozen coordinates or evidence text no longer match this
            # request (e.g. wrangled data changed since the frozen run) —
            # fall back rather than serve a stale, mismatched receipt.
            return await live_reviewer(request)

        return _receipt(
            request,
            status="ok",
            attempts=1,
            latency_seconds=0.0,
            response=response,
            resolved_model=f"{record.resolved_model or 'gpt-5.6-luna'} (frozen run)",
            response_id=record.response_id,
            usage=record.usage,
        )

    return _reviewer


async def _review_journal_async(
    *,
    user_id: str,
    core_values: list[str],
    entries: list[dict[str, Any]],
    reviewer: WeeklyDriftReviewerFn,
    receipt_cache: dict[str, WeeklyDriftReviewerReceipt],
) -> list[WeeklyDriftReviewerReceipt]:
    entries_by_week: dict[tuple[date, date], list[dict[str, Any]]] = {}
    for entry in sorted(entries, key=lambda row: int(row["t_index"])):
        entries_by_week.setdefault(_week_bounds(str(entry["date"])), []).append(entry)

    receipts: list[WeeklyDriftReviewerReceipt] = []
    for week_start, week_end in sorted(entries_by_week):
        history = [
            WeeklyDriftReviewerEntry(
                t_index=int(entry["t_index"]),
                date=str(entry["date"]),
                text=entry_display_text(entry),
            )
            for entry in entries
            if _parse_date(str(entry["date"])) <= week_end
        ]
        request = build_weekly_drift_reviewer_request(
            persona_id=user_id,
            week_start=week_start.isoformat(),
            week_end=week_end.isoformat(),
            core_values=core_values,
            history=history,
            current_t_indices=[
                int(entry["t_index"])
                for entry in entries_by_week[(week_start, week_end)]
            ],
        )
        cache_key = f"{request.prompt_sha256}:{request.week_end}"
        receipt = receipt_cache.get(cache_key)
        if receipt is None:
            receipt = await reviewer(request)
            if receipt.status == "ok":
                receipt_cache[cache_key] = receipt
        receipts.append(receipt)
    return receipts


def review_journal(
    *,
    user_id: str,
    core_values: list[str],
    entries: list[dict[str, Any]],
    reviewer: WeeklyDriftReviewerFn | None = None,
    receipt_cache: dict[str, WeeklyDriftReviewerReceipt] | None = None,
) -> dict[str, Any]:
    """Review the journal week by week, then apply the Drift Detector.

    Makes one paid Weekly Drift Reviewer call per uncached persona-week.
    Returns receipts, the flat Weekly Drift Reviewer Decisions, and the
    Drift Detector result.
    """
    if not core_values:
        raise ValueError("At least one Core Value is required")
    if len(core_values) > MAX_CORE_VALUES:
        raise ValueError(f"At most {MAX_CORE_VALUES} Core Values are supported")
    if not entries:
        raise ValueError("At least one Journal Entry is required")

    receipts = asyncio.run(
        _review_journal_async(
            user_id=user_id,
            core_values=core_values,
            entries=entries,
            reviewer=reviewer or OpenAIWeeklyDriftReviewer(),
            receipt_cache=receipt_cache if receipt_cache is not None else {},
        )
    )
    decisions: list[WeeklyDriftReviewerDecision] = [
        decision for receipt in receipts for decision in receipt.decisions
    ]
    drift: DriftDetectorResult = detect_drift(decisions, persona_id=user_id)
    return {
        "receipts": receipts,
        "decisions": decisions,
        "drift": drift,
    }
