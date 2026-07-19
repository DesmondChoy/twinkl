"""Tests for the approved deterministic Drift Detector."""

from src.drift_detector import detect_drift
from src.weekly_drift_reviewer import WeeklyDriftReviewerDecision


def _decision(
    t_index: int,
    core_value: str,
    verdict: str,
) -> WeeklyDriftReviewerDecision:
    week_start = "2025-01-06" if t_index == 0 else "2025-01-13"
    week_end = "2025-01-12" if t_index == 0 else "2025-01-19"
    return WeeklyDriftReviewerDecision(
        persona_id="deadbeef",
        week_start=week_start,
        week_end=week_end,
        t_index=t_index,
        date=f"2025-01-{6 + t_index * 7:02d}",
        core_value=core_value,
        verdict=verdict,
        confidence="high" if verdict != "abstain" else None,
        reason_code=("direct_behavior_or_choice" if verdict == "conflict" else None),
        evidence_quote=f"evidence-{t_index}" if verdict == "conflict" else "",
        review_status="ok",
    )


def test_cross_week_conflicts_form_one_extended_drift_then_recover():
    result = detect_drift(
        [
            _decision(0, "benevolence", "conflict"),
            _decision(1, "benevolence", "conflict"),
            _decision(2, "benevolence", "conflict"),
            _decision(3, "benevolence", "not_conflict"),
        ],
        persona_id="deadbeef",
    )

    assert result.delivery_state == "recovered"
    assert len(result.drifts) == 1
    assert result.drifts[0].confirmation_t_index == 1
    assert result.drifts[0].end_t_index == 2
    assert result.drifts[0].termination_t_index == 3


def test_abstain_after_confirmed_drift_makes_delivery_uncertain():
    result = detect_drift(
        [
            _decision(0, "benevolence", "conflict"),
            _decision(1, "benevolence", "conflict"),
            _decision(2, "benevolence", "abstain"),
        ],
        persona_id="deadbeef",
    )

    assert result.delivery_state == "uncertain"
    assert result.drifts[0].delivery_state == "uncertain"


def test_different_core_value_states_produce_mixed_weekly_delivery():
    result = detect_drift(
        [
            _decision(0, "benevolence", "conflict"),
            _decision(0, "self_direction", "conflict"),
            _decision(1, "benevolence", "conflict"),
            _decision(1, "self_direction", "conflict"),
            _decision(2, "benevolence", "not_conflict"),
            _decision(2, "self_direction", "conflict"),
        ],
        persona_id="deadbeef",
    )

    assert result.delivery_state == "mixed"
    assert result.core_value_states == {
        "benevolence": "recovered",
        "self_direction": "active",
    }
