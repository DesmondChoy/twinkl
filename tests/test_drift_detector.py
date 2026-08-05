"""Tests for the deterministic three-state Drift Detector."""

import pytest

from src.demo.contracts import build_drift_rule_steps
from src.drift_detector import detect_drift
from src.weekly_drift_reviewer import WeeklyDriftReviewerDecision


def _decision(
    t_index: int,
    core_value: str,
    verdict: str,
    *,
    review_status: str = "ok",
) -> WeeklyDriftReviewerDecision:
    return WeeklyDriftReviewerDecision(
        persona_id="deadbeef",
        week_start="2025-01-06" if t_index == 0 else "2025-01-13",
        week_end="2025-01-12" if t_index == 0 else "2025-01-19",
        t_index=t_index,
        date=f"2025-01-{6 + t_index:02d}",
        core_value=core_value,
        verdict=verdict,
        confidence="high" if review_status == "ok" else None,
        reason_code=(
            "direct_behavior_or_choice"
            if verdict == "conflict" and review_status == "ok"
            else "direct_aligned_or_neutral_behavior"
            if verdict == "not_conflict" and review_status == "ok"
            else "ambiguous"
            if review_status == "ok"
            else None
        ),
        evidence_quote=f"evidence-{t_index}" if verdict == "conflict" else "",
        review_status=review_status,
    )


def test_cross_week_conflicts_form_active_drift_then_not_conflict_ends_it():
    active = detect_drift(
        [
            _decision(0, "benevolence", "conflict"),
            _decision(1, "benevolence", "conflict"),
            _decision(2, "benevolence", "conflict"),
        ],
        persona_id="deadbeef",
    )
    ended = detect_drift(
        [
            _decision(0, "benevolence", "conflict"),
            _decision(1, "benevolence", "conflict"),
            _decision(2, "benevolence", "conflict"),
            _decision(3, "benevolence", "not_conflict"),
        ],
        persona_id="deadbeef",
    )

    assert active.delivery_state == "active_drift"
    assert active.core_value_details["benevolence"].current_run_length == 3
    assert ended.delivery_state == "no_active_drift"
    assert len(ended.drifts) == 1
    assert ended.drifts[0].termination_reason == "not_conflict"
    assert ended.drifts[0].termination_t_index == 3


def test_conflict_conflict_abstain_becomes_insufficient_evidence():
    result = detect_drift(
        [
            _decision(0, "benevolence", "conflict"),
            _decision(1, "benevolence", "conflict"),
            _decision(2, "benevolence", "abstain"),
        ],
        persona_id="deadbeef",
    )

    assert result.delivery_state == "insufficient_evidence"
    assert result.core_value_states == {"benevolence": "insufficient_evidence"}
    assert result.drifts[0].termination_reason == "abstain"


def test_unrelated_valid_abstain_does_not_create_insufficient_evidence():
    result = detect_drift(
        [_decision(0, "benevolence", "abstain")],
        persona_id="deadbeef",
    )

    assert result.delivery_state == "no_active_drift"
    assert result.core_value_details["benevolence"].last_decision == "abstain"


@pytest.mark.parametrize("review_status", ["refusal", "invalid", "error"])
def test_unavailable_review_is_insufficient_evidence(review_status: str):
    result = detect_drift(
        [
            _decision(
                0,
                "benevolence",
                "abstain",
                review_status=review_status,
            )
        ],
        persona_id="deadbeef",
    )

    assert result.delivery_state == "insufficient_evidence"


def test_multi_value_precedence_does_not_create_mixed_state():
    result = detect_drift(
        [
            _decision(0, "benevolence", "conflict"),
            _decision(0, "self_direction", "conflict"),
            _decision(1, "benevolence", "conflict"),
            _decision(1, "self_direction", "abstain"),
        ],
        persona_id="deadbeef",
    )

    assert result.delivery_state == "active_drift"
    assert result.core_value_states == {
        "benevolence": "active_drift",
        "self_direction": "insufficient_evidence",
    }


def test_gap_breaks_conflict_run_and_inspect_uses_the_same_transition():
    decisions = [
        _decision(0, "benevolence", "conflict"),
        _decision(2, "benevolence", "conflict"),
    ]

    result = detect_drift(decisions, persona_id="deadbeef")
    steps = build_drift_rule_steps(decisions)

    assert result.delivery_state == "insufficient_evidence"
    assert result.drifts == []
    assert [step.effect for step in steps] == ["start", "start"]
    assert [step.gap_before for step in steps] == [False, True]
    assert steps[-1].current_state == "insufficient_evidence"


def test_later_valid_decision_resolves_insufficient_evidence():
    decisions = [
        _decision(0, "benevolence", "conflict"),
        _decision(1, "benevolence", "conflict"),
        _decision(2, "benevolence", "abstain"),
        _decision(3, "benevolence", "not_conflict"),
    ]

    result = detect_drift(decisions, persona_id="deadbeef")

    assert result.delivery_state == "no_active_drift"
    assert result.core_value_details["benevolence"].current_run_length == 0
