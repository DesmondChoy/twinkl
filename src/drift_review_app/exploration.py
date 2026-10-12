"""Count verified contributors and route each one to its Journal Entries."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, get_args

from src.drift_review_app.data import (
    CURRENT_SETUP_KEY,
    CaseRecord,
    DriftSpan,
    EntryRecord,
    ReviewData,
    value_label,
)

QueryKind = Literal[
    "corpus_personas",
    "corpus_entries",
    "cases",
    "core_values",
    "entry_value_cells",
    "reference_conflict",
    "reference_not_conflict",
    "uncertain",
    "reference_drifts",
    "known_cases",
    "no_known_cases",
    "reference_same_week",
    "reference_cross_week",
    "detected",
    "found",
    "missed",
    "false",
    "covered",
    "unresolved",
    "invalid",
    "abstain",
    "conflict",
    "not_conflict",
]
ReferenceStatus = Literal["all", "has", "none"]

_ACCURACY_QUERIES = {"found", "missed", "false"}
_DECISION_QUERIES = {"invalid", "abstain", "conflict", "not_conflict"}
_REFERENCE_CELL_QUERIES = {
    "entry_value_cells",
    "reference_conflict",
    "reference_not_conflict",
    "uncertain",
}
_TITLES: dict[str, str] = {
    "corpus_personas": "Personas represented in these histories",
    "corpus_entries": "Journal Entries in these histories",
    "cases": "Persona/Core Value histories",
    "core_values": "Core Values represented in these histories",
    "entry_value_cells": "Journal Entry/Core Value combinations",
    "reference_conflict": "Reference Conflict labels",
    "reference_not_conflict": "Reference Not Conflict labels",
    "uncertain": "Uncertain reference labels",
    "reference_drifts": "Known reference Drifts",
    "known_cases": "Histories with a known reference Drift",
    "no_known_cases": "Histories with no known reference Drift",
    "reference_same_week": "Reference Drifts within one week",
    "reference_cross_week": "Reference Drifts across weeks",
    "detected": "Detected Drifts",
    "found": "Known reference Drifts found",
    "missed": "Known reference Drifts missed",
    "false": "False Drift alerts",
    "covered": "Covered histories",
    "unresolved": "Unresolved histories",
    "invalid": "Unavailable reviewer decisions",
    "abstain": "Reviewer Abstain decisions",
    "conflict": "Reviewer Conflict decisions",
    "not_conflict": "Reviewer Not Conflict decisions",
}


@dataclass(frozen=True)
class ExplorationQuery:
    """One count with explicit history filters and one preserved Run."""

    kind: QueryKind
    setup_key: str = CURRENT_SETUP_KEY
    run: int = 1
    reference_status: ReferenceStatus = "all"
    dimension: str = ""
    persona_id: str = ""


@dataclass(frozen=True)
class ExplorationRow:
    """An exact contributor with coordinates accepted by the Journal inspector."""

    case_id: str
    persona_id: str
    dimension: str
    label: str
    t_index: int | None = None
    period: str | None = None
    run: int | None = None
    reference_id: str | None = None
    drift_id: str | None = None
    span: tuple[int, int, int] | None = None
    start_date: str | None = None
    end_date: str | None = None
    status: str | None = None
    related_case_ids: tuple[str, ...] = ()
    dimensions: tuple[str, ...] = ()

    @property
    def case_ids(self) -> tuple[str, ...]:
        return self.related_case_ids


@dataclass(frozen=True)
class ExplorationResult:
    """The displayed count and the complete contributor population agree."""

    query: ExplorationQuery
    title: str
    unit: str
    description: str
    rows: tuple[ExplorationRow, ...]

    @property
    def count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class CoreValueCount:
    """One chart bar and the query that retrieves its exact contributors."""

    dimension: str
    label: str
    count: int
    query: ExplorationQuery


def _validate(data: ReviewData, query: ExplorationQuery) -> None:
    if query.kind not in get_args(QueryKind):
        raise ValueError(f"Unknown exploration query: {query.kind}")
    if query.setup_key not in data.setup_specs:
        raise ValueError(f"Unknown reviewer setup: {query.setup_key}")
    if isinstance(query.run, bool) or query.run not in (1, 2, 3):
        raise ValueError("Exploration requires one preserved Run: 1, 2, or 3")
    if query.reference_status not in get_args(ReferenceStatus):
        raise ValueError(f"Unknown reference Drift filter: {query.reference_status}")
    dimensions = {case.dimension for case in data.cases.values()}
    if query.dimension not in {"", "all", *dimensions}:
        raise ValueError(f"Unknown Core Value: {query.dimension}")
    if query.persona_id and query.persona_id not in data.profiles:
        raise ValueError(f"Unknown Persona: {query.persona_id}")
    if (
        query.kind in _ACCURACY_QUERIES
        and not data.setup_specs[query.setup_key].accuracy_evaluated
    ):
        raise ValueError(
            "The definitions study reports detected Drift identities; "
            "found, missed, and false-alert accuracy queries are unavailable"
        )


def _cases(data: ReviewData, query: ExplorationQuery) -> tuple[CaseRecord, ...]:
    cases = []
    for case in sorted(data.cases.values(), key=lambda item: item.case_id):
        if query.persona_id and case.persona_id != query.persona_id:
            continue
        if query.dimension not in ("", "all", case.dimension):
            continue
        has_drift = bool(data.reference_drifts[case.case_id])
        if query.reference_status == "has" and not has_drift:
            continue
        if query.reference_status == "none" and has_drift:
            continue
        if query.kind == "known_cases" and not has_drift:
            continue
        if query.kind == "no_known_cases" and has_drift:
            continue
        cases.append(case)
    return tuple(cases)


def _entry_row(
    data: ReviewData,
    query: ExplorationQuery,
    case: CaseRecord,
    entry: EntryRecord,
    *,
    status: str | None = None,
) -> ExplorationRow:
    decision = data.decision(query.setup_key, query.run, case.case_id, entry.t_index)
    return ExplorationRow(
        case_id=case.case_id,
        persona_id=case.persona_id,
        dimension=case.dimension,
        label=(
            f"{case.persona_id} · {value_label(case.dimension)} · Entry {entry.t_index}"
        ),
        t_index=entry.t_index,
        period=f"week|{decision.week_start}",
        run=query.run,
        start_date=entry.date,
        end_date=entry.date,
        status=status,
        related_case_ids=(case.case_id,),
        dimensions=(case.dimension,),
    )


def _case_row(
    data: ReviewData, query: ExplorationQuery, case: CaseRecord
) -> ExplorationRow:
    row = _entry_row(data, query, case, case.entries[0])
    return replace(
        row,
        label=f"{case.persona_id} · {value_label(case.dimension)}",
        period="full",
        t_index=None,
    )


def _span_row(
    data: ReviewData,
    query: ExplorationQuery,
    case: CaseRecord,
    span: DriftSpan,
    *,
    predicted: bool = False,
    prediction: DriftSpan | None = None,
) -> ExplorationRow:
    entry = next(
        entry for entry in case.entries if entry.t_index == span.confirmation_t_index
    )
    row = _entry_row(data, query, case, entry)
    return replace(
        row,
        label=(
            f"{case.persona_id} · {value_label(case.dimension)} · "
            f"Entries {span.onset_t_index}–{span.end_t_index}"
        )
        + (
            f" · Matched alert confirmed at Entry {prediction.confirmation_t_index}"
            if prediction
            else ""
        ),
        t_index=prediction.confirmation_t_index if prediction else row.t_index,
        reference_id=(
            span.matched_reference_id
            if predicted and data.setup_specs[query.setup_key].accuracy_evaluated
            else None
            if predicted
            else span.drift_id
        ),
        drift_id=(
            span.drift_id if predicted else prediction.drift_id if prediction else None
        ),
        span=(span.onset_t_index, span.confirmation_t_index, span.end_t_index),
        period=(
            "full"
            if prediction
            else (
                f"pred|{query.run}|{span.onset_t_index}|{span.end_t_index}|"
                f"{span.detection_date}"
            )
            if predicted
            else f"ref|{span.drift_id}|{span.onset_t_index}|{span.end_t_index}"
        ),
        start_date=span.onset_date,
        end_date=span.end_date,
        status=(
            "found"
            if prediction
            else span.result
            if predicted and data.setup_specs[query.setup_key].accuracy_evaluated
            else "detected"
            if predicted
            else "reference"
        ),
    )


def _corpus_rows(
    data: ReviewData,
    query: ExplorationQuery,
    cases: tuple[CaseRecord, ...],
) -> tuple[tuple[ExplorationRow, ...], str, str]:
    if query.kind in {"cases", "known_cases", "no_known_cases"}:
        return (
            tuple(_case_row(data, query, case) for case in cases),
            "Persona/Core Value histories",
            "Each row is one Persona's Journal Entries for one declared Core Value.",
        )
    grouped: dict[str, list[CaseRecord]] = {}
    for case in cases:
        grouped.setdefault(case.persona_id, []).append(case)
    rows = []
    for persona_id, persona_cases in grouped.items():
        first = persona_cases[0]
        related_case_ids = tuple(case.case_id for case in persona_cases)
        dimensions = tuple(case.dimension for case in persona_cases)
        if query.kind == "corpus_personas":
            rows.append(
                replace(
                    _case_row(data, query, first),
                    label=persona_id,
                    related_case_ids=related_case_ids,
                    dimensions=dimensions,
                )
            )
        elif query.kind == "corpus_entries":
            for entry in first.entries:
                rows.append(
                    replace(
                        _entry_row(data, query, first, entry),
                        label=f"{persona_id} · Entry {entry.t_index}",
                        related_case_ids=related_case_ids,
                        dimensions=dimensions,
                    )
                )
    if query.kind == "core_values":
        by_value: dict[str, CaseRecord] = {}
        for case in cases:
            by_value.setdefault(case.dimension, case)
        rows = [
            replace(_case_row(data, query, case), label=value_label(dimension))
            for dimension, case in sorted(by_value.items())
        ]
        return (
            tuple(rows),
            "Core Values",
            "Each Core Value represented is counted once.",
        )
    if query.kind == "corpus_personas":
        return (
            tuple(rows),
            "Personas",
            (
                "Each Persona is counted once across matching Core Value histories. "
                "A row lists every matching history for that Persona."
            ),
        )
    return (
        tuple(rows),
        "Journal Entries",
        (
            "Each Persona/t_index Journal Entry is counted once across matching "
            "Core Values. "
            "A row lists the matching histories that share this Journal Entry."
        ),
    )


def _cell_rows(
    data: ReviewData,
    query: ExplorationQuery,
    cases: tuple[CaseRecord, ...],
) -> tuple[ExplorationRow, ...]:
    rows = []
    for case in cases:
        for entry in case.entries:
            if query.kind in _REFERENCE_CELL_QUERIES:
                if (
                    query.kind == "reference_conflict"
                    and entry.final_conflict is not True
                ):
                    continue
                if (
                    query.kind == "reference_not_conflict"
                    and entry.final_conflict is not False
                ):
                    continue
                if query.kind == "uncertain" and entry.final_conflict is not None:
                    continue
                status = (
                    "Uncertain"
                    if entry.final_conflict is None
                    else "Conflict"
                    if entry.final_conflict
                    else "Not Conflict"
                )
            else:
                decision = data.decision(
                    query.setup_key, query.run, case.case_id, entry.t_index
                )
                if query.kind == "invalid":
                    if decision.response_status == "ok":
                        continue
                    status = decision.response_status
                else:
                    if (
                        decision.response_status != "ok"
                        or decision.verdict != query.kind
                    ):
                        continue
                    status = decision.verdict
            rows.append(_entry_row(data, query, case, entry, status=status))
    return tuple(rows)


def explore(data: ReviewData, query: ExplorationQuery) -> ExplorationResult:
    """Return the precise contributors to one filtered count, without Run voting."""
    _validate(data, query)
    cases = _cases(data, query)
    title = _TITLES[query.kind]
    if query.kind in {
        "corpus_personas",
        "corpus_entries",
        "cases",
        "core_values",
        "known_cases",
        "no_known_cases",
    }:
        rows, unit, description = _corpus_rows(data, query, cases)
    elif query.kind in _REFERENCE_CELL_QUERIES | _DECISION_QUERIES:
        rows = _cell_rows(data, query, cases)
        unit = "entry/Core Value combinations"
        description = (
            "Each row is one Journal Entry paired with one declared Core Value. "
            "Reference labels come from the AI-reviewed development data."
            if query.kind in _REFERENCE_CELL_QUERIES
            else "Each row is one reviewer decision in the selected preserved Run. "
            "Valid Abstain decisions and unavailable reviews are counted separately."
        )
    elif query.kind in {"covered", "unresolved"}:
        rows = tuple(
            _case_row(data, query, case)
            for case in cases
            if bool(
                data.case_metrics[(query.setup_key, query.run, case.case_id)]["covered"]
            )
            == (query.kind == "covered")
        )
        unit = "Persona/Core Value histories"
        description = (
            "Covered histories contain a Conflict pair or rule out Drift in every "
            "adjacent pair. Others remain unresolved, including histories with no "
            "adjacent pair. This follows the selected preserved Run."
        )
    else:
        event_rows = []
        for case in cases:
            predictions = data.predicted_drifts[
                (query.setup_key, query.run, case.case_id)
            ]
            if query.kind in {"detected", "false"}:
                for prediction in predictions:
                    if (
                        query.kind == "false"
                        and prediction.result != "false Drift alert"
                    ):
                        continue
                    event_rows.append(
                        _span_row(data, query, case, prediction, predicted=True)
                    )
                continue
            matched = {
                span.matched_reference_id: span
                for span in predictions
                if span.matched_reference_id is not None
            }
            for reference in data.reference_drifts[case.case_id]:
                if query.kind == "reference_same_week" and reference.crosses_week:
                    continue
                if query.kind == "reference_cross_week" and not reference.crosses_week:
                    continue
                if query.kind == "found" and reference.drift_id not in matched:
                    continue
                if query.kind == "missed" and reference.drift_id in matched:
                    continue
                event_rows.append(
                    _span_row(
                        data,
                        query,
                        case,
                        reference,
                        prediction=(
                            matched.get(reference.drift_id)
                            if query.kind == "found"
                            else None
                        ),
                    )
                )
        rows = tuple(event_rows)
        unit = (
            "detected Drifts"
            if query.kind in {"detected", "false"}
            else "reference Drifts"
        )
        description = (
            "Each row is one predicted Drift span in the selected preserved Run."
            if query.kind in {"detected", "false"}
            else "Each row is one AI-reviewed reference Drift span. Found and missed "
            "rows use the selected preserved Run's original one-to-one matching."
        )
    if query.kind in _DECISION_QUERIES | _ACCURACY_QUERIES | {
        "detected",
        "covered",
        "unresolved",
    }:
        title = f"{title} · {data.setup_specs[query.setup_key].label} · Run {query.run}"
    return ExplorationResult(query, title, unit, description, rows)


def core_value_breakdown(
    data: ReviewData, query: ExplorationQuery
) -> tuple[CoreValueCount, ...]:
    """Return counts per Core Value with identical history, setup, and Run filters."""
    _validate(data, query)
    dimensions = sorted({case.dimension for case in data.cases.values()})
    if query.dimension not in ("", "all"):
        dimensions = [query.dimension]
    bars = []
    for dimension in dimensions:
        scoped = replace(query, dimension=dimension)
        bars.append(
            CoreValueCount(
                dimension, value_label(dimension), explore(data, scoped).count, scoped
            )
        )
    return tuple(bars)
