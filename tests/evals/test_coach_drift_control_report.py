"""Tests for the Coach Digest Drift/control comparison report."""

from __future__ import annotations

import json

import pytest

from src.evals.coach_drift_control_report import (
    build_report,
    render_markdown,
    wilson_interval,
)

CHECK_NAMES = (
    "groundedness",
    "non_circularity",
    "value_leakage",
    "state_claims",
    "length",
)


def _entry(
    target_id: str,
    group: str,
    *,
    grounded: bool,
    delivery_state: str | None = None,
    n_entries: int = 8,
) -> dict:
    checks = [
        {
            "name": name,
            "passed": grounded if name == "groundedness" else True,
            "details": "",
        }
        for name in CHECK_NAMES
    ]
    return {
        "digest": {
            "persona_id": target_id,
            "week_end": "2025-06-08",
            "response_mode": "active_drift" if group == "drift" else "stable",
            "n_entries": n_entries,
            "evidence": [{"excerpt": "x"}, {"excerpt": "y"}],
        },
        "narrative": {"weekly_mirror": "m"},
        "validation": {"all_passed": grounded, "checks": checks},
        "generator_model": "openai:model-one",
        "target": {
            "target_id": target_id,
            "group": group,
            "delivery_state": delivery_state,
            "match_quality": "exact",
            "reviewed_week_count": 4,
        },
    }


def _eval_result(target_id: str, specificity: int) -> dict:
    return {
        "sample_id": target_id,
        "status": "scored",
        "correctness": 4,
        "specificity": specificity,
        "non_prescriptive_tone": 5,
        "tension_honesty": 4,
        "needs_review": False,
        "justification": "ok",
    }


def test_wilson_interval_brackets_rate_and_stays_in_range():
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    assert 0.0 <= low and high <= 1.0

    low, high = wilson_interval(4, 4)
    assert high == 1.0
    assert low < 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_report_compares_validations_scores_and_known_drift_state():
    manifest = [
        _entry("d1", "drift", grounded=True, delivery_state="active"),
        _entry("d2", "drift", grounded=True, delivery_state="ended"),
        _entry("c1", "control", grounded=False),
        _entry("c2", "control", grounded=False),
    ]
    eval_metrics = {
        "judge_model": "gemini:model-two",
        "sample_results": [
            _eval_result("d1", 4),
            _eval_result("d2", 4),
            _eval_result("c1", 2),
            _eval_result("c2", 2),
        ]
    }

    report = build_report(manifest, eval_metrics)

    assert report["n_by_group"] == {"drift": 2, "control": 2}
    assert report["generator_model"] == "openai:model-one"
    assert report["evaluator_model"] == "gemini:model-two"
    assert report["cross_provider"] is True
    assert report["self_evaluation"] is False
    drift = report["validations_by_group"]["drift"]["groundedness"]
    control = report["validations_by_group"]["control"]["groundedness"]
    assert drift["passed"] == 2 and drift["rate"] == 1.0
    assert control["passed"] == 0 and control["rate"] == 0.0
    assert len(drift["ci95"]) == 2
    assert report["scores_by_group"]["drift"]["specificity"]["mean"] == 4.0
    assert report["scores_by_group"]["control"]["specificity"]["mean"] == 2.0
    assert report["scores_by_known_delivery_state"]["active"]["correctness"][
        "mean"
    ] == 4.0
    assert report["n_without_eval_result"] == 0


def test_report_recomputes_all_passed_and_exposes_history_inputs():
    manifest = [
        _entry("d1", "drift", grounded=True, n_entries=12),
        _entry("c1", "control", grounded=False, n_entries=4),
    ]
    manifest[1]["validation"]["all_passed"] = True

    report = build_report(manifest, {"sample_results": []})

    rows = {row["sample_id"]: row for row in report["rows"]}
    assert rows["d1"]["all_passed"] is True
    assert rows["c1"]["all_passed"] is False
    assert report["history_summary"]["drift"]["mean_n_entries"] == 12
    assert report["history_summary"]["control"]["mean_n_entries"] == 4
    assert report["n_without_eval_result"] == 2


def test_report_rejects_duplicate_target_ids():
    manifest = [
        _entry("d1", "drift", grounded=True),
        _entry("d1", "drift", grounded=True),
    ]

    with pytest.raises(ValueError, match="Duplicate manifest target"):
        build_report(manifest, {"sample_results": []})


def test_report_rejects_eval_results_from_another_manifest():
    manifest = [_entry("d1", "drift", grounded=True)]
    eval_metrics = {"sample_results": [_eval_result("another-target", 4)]}

    with pytest.raises(ValueError, match="do not match the manifest"):
        build_report(manifest, eval_metrics)


def test_report_rejects_a_different_generator_model():
    manifest = [_entry("d1", "drift", grounded=True)]
    eval_metrics = {
        "generator_model": "openai:different-model",
        "sample_results": [_eval_result("d1", 4)],
    }

    with pytest.raises(ValueError, match="different generator model"):
        build_report(manifest, eval_metrics)


def test_report_rejects_multiple_manifest_generator_models():
    manifest = [
        _entry("d1", "drift", grounded=True),
        _entry("c1", "control", grounded=True),
    ]
    manifest[1]["generator_model"] = "openai:model-two"

    with pytest.raises(ValueError, match="multiple Coach Digest generator"):
        build_report(manifest, {"sample_results": []})


def test_markdown_states_sources_and_limits():
    report = build_report(
        [_entry("d1", "drift", grounded=True, delivery_state="active")],
        {"sample_results": []},
    )
    markdown = render_markdown(report)

    assert "not human validation" in markdown.lower()
    assert "AI-reviewed synthetic development data" in markdown
    assert "Input history by group" in markdown
    assert "Cross-provider AI review: unrecorded" in markdown
    assert "95% interval" in markdown
    assert json.loads(json.dumps(report))["n_rows"] == 1
