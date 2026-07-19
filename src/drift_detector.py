"""Deterministic Drift Detector for Weekly Drift Reviewer Decisions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from src.drift_rules import drift_spans
from src.weekly_drift_reviewer import WeeklyDriftReviewerDecision

DriftDeliveryState = Literal["active", "recovered", "uncertain"]
WeeklyDeliveryState = Literal["stable", "active", "recovered", "uncertain", "mixed"]


class DriftRecord(BaseModel):
    """One confirmed Drift for one Core Value."""

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
    delivery_state: DriftDeliveryState
    termination_t_index: int | None = Field(default=None, ge=0)
    termination_date: str | None = None
    termination_verdict: Literal["not_conflict", "abstain"] | None = None


class DriftDetectorResult(BaseModel):
    """Auditable Drift Detector result at one delivery cutoff."""

    schema_version: Literal["drift-detector-result-v1"] = "drift-detector-result-v1"
    persona_id: str
    cutoff_t_index: int = Field(ge=0)
    cutoff_date: str
    delivery_state: WeeklyDeliveryState
    core_value_states: dict[str, DriftDeliveryState] = Field(default_factory=dict)
    drifts: list[DriftRecord] = Field(default_factory=list)


def _aggregate_delivery_state(
    core_value_states: dict[str, DriftDeliveryState],
) -> WeeklyDeliveryState:
    states = set(core_value_states.values())
    if not states:
        return "stable"
    if len(states) == 1:
        return states.pop()
    return "mixed"


def detect_drift(
    decisions: Sequence[WeeklyDriftReviewerDecision],
    *,
    persona_id: str,
) -> DriftDetectorResult:
    """Apply the two-consecutive-Conflict rule independently per Core Value."""
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
    core_value_states: dict[str, DriftDeliveryState] = {}
    for core_value, value_decisions in sorted(by_core_value.items()):
        value_decisions.sort(key=lambda item: item.t_index)
        labels = [
            True
            if row.verdict == "conflict"
            else False
            if row.verdict == "not_conflict"
            else None
            for row in value_decisions
        ]
        t_indices = [row.t_index for row in value_decisions]
        decision_by_index = {row.t_index: row for row in value_decisions}

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
            if next_decision is None:
                delivery_state: DriftDeliveryState = "active"
                termination_verdict: Literal["not_conflict", "abstain"] | None = None
            elif (
                next_decision.t_index == end + 1
                and next_decision.verdict == "not_conflict"
            ):
                delivery_state = "recovered"
                termination_verdict = "not_conflict"
            else:
                delivery_state = "uncertain"
                termination_verdict = "abstain"

            drift = DriftRecord(
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
                delivery_state=delivery_state,
                termination_t_index=(
                    next_decision.t_index if next_decision is not None else None
                ),
                termination_date=(
                    next_decision.date if next_decision is not None else None
                ),
                termination_verdict=termination_verdict,
            )
            drifts.append(drift)
            core_value_states[core_value] = delivery_state

    cutoff = max(ordered, key=lambda item: item.t_index)
    return DriftDetectorResult(
        persona_id=persona_id,
        cutoff_t_index=cutoff.t_index,
        cutoff_date=cutoff.date,
        delivery_state=_aggregate_delivery_state(core_value_states),
        core_value_states=core_value_states,
        drifts=sorted(
            drifts,
            key=lambda row: (row.onset_t_index, row.core_value),
        ),
    )
