"""Deterministic Drift Detector for Weekly Drift Reviewer Decisions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from src.drift_rules import drift_spans
from src.weekly_drift_reviewer import (
    ReviewStatus,
    Verdict,
    WeeklyDriftReviewerDecision,
)

CoreValueDriftState = Literal[
    "active_drift",
    "no_active_drift",
    "insufficient_evidence",
]
DriftTerminationReason = Literal["not_conflict", "abstain", "gap"]
DriftStepEffect = Literal["start", "confirm", "extend", "end", "insufficient", "reset"]


class DriftRecord(BaseModel):
    """One confirmed historical Drift for one Core Value."""

    drift_id: str
    persona_id: str
    core_value: str
    onset_t_index: int = Field(ge=0)
    confirmation_t_index: int = Field(ge=0)
    end_t_index: int = Field(ge=0)
    onset_date: str
    confirmation_date: str
    end_date: str
    supporting_t_indices: list[int]
    evidence_quotes: list[str]
    termination_reason: DriftTerminationReason | None = None
    termination_t_index: int | None = Field(default=None, ge=0)
    termination_date: str | None = None
    termination_verdict: Literal["not_conflict", "abstain"] | None = None


class CoreValueDriftDetail(BaseModel):
    """Current derived state and compact decision metadata for one Core Value."""

    state: CoreValueDriftState
    current_run_length: int = Field(ge=0)
    last_decision: Verdict
    last_review_status: ReviewStatus
    last_t_index: int = Field(ge=0)
    last_date: str


class DriftDetectorResult(BaseModel):
    """Auditable current state and historical Drifts at one delivery cutoff."""

    schema_version: Literal["drift-detector-result-v2"] = "drift-detector-result-v2"
    persona_id: str
    cutoff_t_index: int = Field(ge=0)
    cutoff_date: str
    delivery_state: CoreValueDriftState
    core_value_states: dict[str, CoreValueDriftState]
    core_value_details: dict[str, CoreValueDriftDetail]
    drifts: list[DriftRecord] = Field(default_factory=list)


@dataclass(frozen=True)
class DriftDecisionTransition:
    """One decision transition shared by the Drift Detector and Inspect."""

    decision: WeeklyDriftReviewerDecision
    current_state: CoreValueDriftState
    current_run_length: int
    gap_before: bool
    effect: DriftStepEffect


def _aggregate_delivery_state(
    core_value_states: dict[str, CoreValueDriftState],
) -> CoreValueDriftState:
    states = set(core_value_states.values())
    if "active_drift" in states:
        return "active_drift"
    if "insufficient_evidence" in states:
        return "insufficient_evidence"
    return "no_active_drift"


def drift_decision_transitions(
    decisions: Sequence[WeeklyDriftReviewerDecision],
) -> list[DriftDecisionTransition]:
    """Derive current state after each decision for one Core Value."""
    if not decisions:
        return []
    core_values = {decision.core_value for decision in decisions}
    if len(core_values) != 1:
        raise ValueError("Drift transitions require one Core Value")

    ordered = sorted(decisions, key=lambda item: item.t_index)
    transitions: list[DriftDecisionTransition] = []
    state: CoreValueDriftState = "no_active_drift"
    run_length = 0
    unresolved_conflict = False
    previous_index: int | None = None
    previous_verdict: Verdict | None = None

    for decision in ordered:
        gap_before = (
            previous_index is not None and decision.t_index != previous_index + 1
        )
        if gap_before:
            if run_length > 0 or state != "no_active_drift":
                unresolved_conflict = True
                state = "insufficient_evidence"
            run_length = 0
            previous_verdict = None

        prior_state = state
        prior_run_length = run_length
        if decision.review_status != "ok":
            state = "insufficient_evidence"
            unresolved_conflict = True
            run_length = 0
            effect: DriftStepEffect = "insufficient"
        elif decision.verdict == "not_conflict":
            state = "no_active_drift"
            unresolved_conflict = False
            run_length = 0
            effect = (
                "end"
                if prior_run_length > 0 or prior_state != "no_active_drift"
                else "reset"
            )
        elif decision.verdict == "abstain":
            if prior_run_length > 0 or prior_state != "no_active_drift":
                state = "insufficient_evidence"
                unresolved_conflict = True
                effect = "insufficient"
            else:
                state = "no_active_drift"
                unresolved_conflict = False
                effect = "reset"
            run_length = 0
        else:
            run_length = (
                prior_run_length + 1
                if previous_verdict == "conflict" and not gap_before
                else 1
            )
            if run_length >= 2:
                state = "active_drift"
                unresolved_conflict = False
                effect = "confirm" if run_length == 2 else "extend"
            elif unresolved_conflict:
                state = "insufficient_evidence"
                effect = "start"
            else:
                state = "no_active_drift"
                effect = "start"

        transitions.append(
            DriftDecisionTransition(
                decision=decision,
                current_state=state,
                current_run_length=run_length,
                gap_before=gap_before,
                effect=effect,
            )
        )
        previous_index = decision.t_index
        previous_verdict = decision.verdict

    return transitions


def _historical_drifts(
    value_decisions: Sequence[WeeklyDriftReviewerDecision],
    *,
    persona_id: str,
    core_value: str,
) -> list[DriftRecord]:
    labels = [
        True
        if row.verdict == "conflict" and row.review_status == "ok"
        else False
        if row.verdict == "not_conflict" and row.review_status == "ok"
        else None
        for row in value_decisions
    ]
    t_indices = [row.t_index for row in value_decisions]
    decision_by_index = {row.t_index: row for row in value_decisions}
    drifts: list[DriftRecord] = []

    for onset, confirmation, end in drift_spans(labels, t_indices):
        run = [
            decision_by_index[t_index]
            for t_index in range(onset, end + 1)
            if t_index in decision_by_index
        ]
        next_decision = next(
            (row for row in value_decisions if row.t_index > end),
            None,
        )
        termination_reason: DriftTerminationReason | None = None
        termination_t_index: int | None = None
        termination_date: str | None = None
        termination_verdict: Literal["not_conflict", "abstain"] | None = None
        if next_decision is not None:
            if next_decision.t_index != end + 1:
                termination_reason = "gap"
            elif next_decision.verdict == "not_conflict":
                termination_reason = "not_conflict"
                termination_t_index = next_decision.t_index
                termination_date = next_decision.date
                termination_verdict = "not_conflict"
            else:
                termination_reason = "abstain"
                termination_t_index = next_decision.t_index
                termination_date = next_decision.date
                termination_verdict = "abstain"

        drifts.append(
            DriftRecord(
                drift_id=f"{persona_id}:{core_value}:{onset}",
                persona_id=persona_id,
                core_value=core_value,
                onset_t_index=onset,
                confirmation_t_index=confirmation,
                end_t_index=end,
                onset_date=decision_by_index[onset].date,
                confirmation_date=decision_by_index[confirmation].date,
                end_date=decision_by_index[end].date,
                supporting_t_indices=[row.t_index for row in run],
                evidence_quotes=[
                    row.evidence_quote for row in run if row.evidence_quote.strip()
                ],
                termination_reason=termination_reason,
                termination_t_index=termination_t_index,
                termination_date=termination_date,
                termination_verdict=termination_verdict,
            )
        )
    return drifts


def detect_drift(
    decisions: Sequence[WeeklyDriftReviewerDecision],
    *,
    persona_id: str,
) -> DriftDetectorResult:
    """Derive current state and historical Drifts independently per Core Value."""
    if not decisions:
        raise ValueError("At least one Weekly Drift Reviewer Decision is required")
    if any(decision.persona_id != persona_id for decision in decisions):
        raise ValueError(
            "All Weekly Drift Reviewer Decisions must belong to persona_id"
        )

    ordered = sorted(decisions, key=lambda item: (item.t_index, item.core_value))
    duplicate_coordinates = len({(row.t_index, row.core_value) for row in ordered})
    if duplicate_coordinates != len(ordered):
        raise ValueError(
            "Weekly Drift Reviewer Decisions contain duplicate coordinates"
        )

    by_core_value: dict[str, list[WeeklyDriftReviewerDecision]] = defaultdict(list)
    for decision in ordered:
        by_core_value[decision.core_value].append(decision)

    drifts: list[DriftRecord] = []
    core_value_states: dict[str, CoreValueDriftState] = {}
    core_value_details: dict[str, CoreValueDriftDetail] = {}
    for core_value, value_decisions in sorted(by_core_value.items()):
        value_decisions.sort(key=lambda item: item.t_index)
        transitions = drift_decision_transitions(value_decisions)
        latest = transitions[-1]
        core_value_states[core_value] = latest.current_state
        core_value_details[core_value] = CoreValueDriftDetail(
            state=latest.current_state,
            current_run_length=latest.current_run_length,
            last_decision=latest.decision.verdict,
            last_review_status=latest.decision.review_status,
            last_t_index=latest.decision.t_index,
            last_date=latest.decision.date,
        )
        drifts.extend(
            _historical_drifts(
                value_decisions,
                persona_id=persona_id,
                core_value=core_value,
            )
        )

    cutoff = max(ordered, key=lambda item: item.t_index)
    return DriftDetectorResult(
        persona_id=persona_id,
        cutoff_t_index=cutoff.t_index,
        cutoff_date=cutoff.date,
        delivery_state=_aggregate_delivery_state(core_value_states),
        core_value_states=core_value_states,
        core_value_details=core_value_details,
        drifts=sorted(
            drifts,
            key=lambda row: (row.onset_t_index, row.core_value),
        ),
    )
