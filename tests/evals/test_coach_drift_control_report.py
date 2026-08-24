"""Tests for the Drift against control Coach Digest comparison."""

from __future__ import annotations

import json

from src.evals.coach_drift_control_report import (
    build_report,
    render_markdown,
    wilson_interval,
)


def _entry(
    persona_id: str,
    arm: str,
    *,
    grounded: bool,
    delivery_state: str | None = None,
    n_entries: int = 8,
) -> dict:
    return {
        "digest": {
            "persona_id": persona_id,
            "week_end": "2025-06-08",
            "response_mode": "active_drift" if arm == "drift" else "stable",
            "n_entries": n_entries,
            "evidence": [{"excerpt": "x"}, {"excerpt": "y"}],
        },
        "narrative": {"weekly_mirror": "m"},
        "validation": {
            "all_passed": grounded,
            "checks": [
                {"name": "groundedness", "passed": grounded, "details": ""},
                {"name": "non_circularity", "passed": True, "details": ""},
            ],
        },
        "generator_model": "openai:m1",
        "target": {
            "arm": arm,
            "delivery_state": delivery_state,
            "match_quality": "exact",
            "n_truncated_weeks": 4,
        },
    }


def _verdict_record(persona_id: str, specificity: int) -> dict:
    return {
        "key": f"{persona_id}:2025-06-08",
        "verdict": {
            "correctness": 4,
            "specificity": specificity,
            "non_prescriptive_tone": 5,
            "tension_honesty": 4,
            "needs_review": False,
            "justification": "ok",
        },
    }


def test_wilson_interval_brackets_the_rate_and_stays_in_range():
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    assert 0.0 <= low and high <= 1.0

    # A perfect rate must not produce an upper bound above one.
    low, high = wilson_interval(4, 4)
    assert high == 1.0
    assert low < 1.0

    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_build_report_splits_validations_and_scores_by_arm():
    manifest = [
        _entry("a1", "drift", grounded=True, delivery_state="active"),
        _entry("a2", "drift", grounded=True, delivery_state="recovered"),
        _entry("b1", "control", grounded=False),
        _entry("b2", "control", grounded=False),
    ]
    verdicts = [
        _verdict_record("a1", 4),
        _verdict_record("a2", 4),
        _verdict_record("b1", 2),
        _verdict_record("b2", 2),
    ]

    report = build_report(manifest, verdicts)

    assert report["n_by_arm"] == {"drift": 2, "control": 2}
    drift = report["validations_by_arm"]["drift"]["groundedness"]
    control = report["validations_by_arm"]["control"]["groundedness"]
    assert drift["passed"] == 2 and drift["rate"] == 1.0
    assert control["passed"] == 0 and control["rate"] == 0.0
    # Every rate carries an interval, so a small sample cannot read as certain.
    assert len(drift["ci95"]) == 2

    assert report["scores_by_arm"]["drift"]["specificity"]["mean"] == 4.0
    assert report["scores_by_arm"]["control"]["specificity"]["mean"] == 2.0
    assert report["generator_model"] == "openai:m1"
    assert report["n_without_verdict"] == 0


def test_build_report_splits_by_delivery_state():
    manifest = [
        _entry("a1", "drift", grounded=True, delivery_state="active"),
        _entry("a2", "drift", grounded=False, delivery_state="recovered"),
    ]
    report = build_report(manifest, [])

    by_state = report["validations_by_delivery_state"]
    assert by_state["active"]["groundedness"]["rate"] == 1.0
    assert by_state["recovered"]["groundedness"]["rate"] == 0.0
    # Without verdicts the report still holds the validation results.
    assert report["n_without_verdict"] == 2
    assert report["scores_by_arm"] == {}


def test_history_length_check_reports_confound_inputs():
    manifest = [
        _entry("a1", "drift", grounded=True, n_entries=12),
        _entry("b1", "control", grounded=False, n_entries=4),
    ]
    check = build_report(manifest, [])["history_length_check"]

    assert check["drift"]["mean_entries"] == 12
    assert check["control"]["mean_entries"] == 4
    assert check["drift"]["mean_entries_grounded"] == 12
    assert check["control"]["mean_entries_ungrounded"] == 4


def test_render_markdown_states_the_limits():
    manifest = [_entry("a1", "drift", grounded=True, delivery_state="active")]
    markdown = render_markdown(build_report(manifest, []))

    assert "not human validation" in markdown.lower()
    assert "no detected Drift" in markdown
    assert "tension_honesty and specificity are expected to differ" in markdown
    assert "95% CI" in markdown


def test_report_is_json_serializable():
    manifest = [_entry("a1", "drift", grounded=True, delivery_state="active")]
    report = build_report(manifest, [_verdict_record("a1", 3)])

    assert json.loads(json.dumps(report))["n_rows"] == 1
