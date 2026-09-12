"""Tests for the Coach Digest Validations batch report."""

from __future__ import annotations

import json

import pytest

from src.coach.schemas import CoachNarrative, CoreValueWeekComparison, WeeklyDigest
from src.coach.weekly_digest import (
    persist_weekly_digest_record,
    validate_weekly_digest_narrative,
)
from src.evals.coach_digest_validations import (
    evaluate_manifest,
    evaluate_parquet,
    evaluate_rows,
    main,
    render_markdown,
)


def _row(
    *,
    persona_id: str,
    week_end: str,
    narrative: dict[str, str] | None,
    evidence_excerpt: str = "called my mom and helped a colleague debug",
    signal_source: str = "weekly_drift_reviewer",
) -> dict[str, object]:
    """Build a parquet-shaped digest row for the evaluator."""
    return {
        "persona_id": persona_id,
        "persona_name": "Casey",
        "week_start": "2025-01-01",
        "week_end": week_end,
        "response_mode": "stable",
        "mode_source": "fallback_heuristic",
        "mode_rationale": "No confirmed Drift this week.",
        "signal_source": signal_source,
        "n_entries": 3,
        "overall_mean": 0.4,
        "overall_uncertainty": None,
        "core_values_json": json.dumps(["benevolence"]),
        "drift_states_json": json.dumps({}),
        "drift_reasons_json": json.dumps([]),
        "top_tensions_json": json.dumps([]),
        "top_strengths_json": json.dumps(["benevolence"]),
        "dimensions_json": json.dumps([]),
        "evidence_json": json.dumps(
            [
                {
                    "date": "2025-01-03",
                    "t_index": 1,
                    "direction": "aligned",
                    "dimensions": ["benevolence"],
                    "excerpt": evidence_excerpt,
                }
            ]
        ),
        "coach_narrative_json": json.dumps(narrative) if narrative else None,
    }


_CLEAN = {
    "weekly_mirror": (
        'A steady week of showing up for people, like when you "helped a '
        'colleague debug" without being asked, holding a calm rhythm across '
        "the days and into the weekend."
    ),
    "tension_explanation": (
        "Nothing pulled against what matters to you this week; the pattern was "
        "showing up for the people around you and it felt unforced."
    ),
    "reflective_question": "What let you keep showing up with intention this week?",
}

_JARGON = {
    "weekly_mirror": (
        "Your alignment score dipped midweek before recovering, and the mean= "
        "reading stayed low for several days in a row before it steadied."
    ),
    "tension_explanation": (
        "The misaligned days pulled your weekly scores down before things "
        "steadied again toward the end of the week and settled."
    ),
    "reflective_question": "What shifted between the low and the recovery here?",
}


def test_evaluate_rows_aggregates_pass_rates():
    rows = [
        _row(persona_id="aaaa", week_end="2025-01-07", narrative=_CLEAN),
        _row(persona_id="bbbb", week_end="2025-01-07", narrative=_JARGON),
        _row(persona_id="cccc", week_end="2025-01-07", narrative=None),
    ]

    report = evaluate_rows(rows, input_source="in-memory")

    assert report.n_rows == 3
    assert report.n_with_narrative == 2
    assert report.n_evaluated == 2
    # Clean passes non_circularity, jargon fails -> 1/2.
    assert report.checks["non_circularity"].passed == 1
    assert report.checks["non_circularity"].total == 2
    assert report.checks["non_circularity"].pass_rate == 0.5
    # non_circularity target is 0.95, so 0.5 does not meet it.
    assert report.checks["non_circularity"].meets_target is False
    # value_leakage has no target -> meets_target is None (informational).
    assert report.checks["value_leakage"].meets_target is None
    assert report.checks["state_claims"].meets_target is None


def test_evaluate_rows_skips_unparseable_narrative():
    bad = _row(persona_id="dddd", week_end="2025-01-07", narrative=None)
    bad["coach_narrative_json"] = "{not valid json"
    rows = [
        _row(persona_id="aaaa", week_end="2025-01-07", narrative=_CLEAN),
        bad,
    ]

    report = evaluate_rows(rows, input_source="in-memory")

    assert report.n_with_narrative == 2
    assert report.n_evaluated == 1
    assert "dddd:2025-01-07" in report.skipped


def test_render_markdown_contains_source_disclaimer():
    rows = [_row(persona_id="aaaa", week_end="2025-01-07", narrative=_CLEAN)]
    report = evaluate_rows(rows, input_source="in-memory")

    markdown = render_markdown(report)

    assert markdown.startswith("# Coach Digest Validations — Batch Report")
    assert "not human validation" in markdown
    assert "groundedness" in markdown
    assert report.to_dict()["eval"] == "coach_digest_validations"


def test_evaluate_rows_defaults_to_approved_path_only():
    rows = [
        _row(persona_id="aaaa", week_end="2025-01-07", narrative=_CLEAN),
        _row(
            persona_id="bbbb",
            week_end="2025-01-07",
            narrative=_JARGON,
            signal_source="vif_runtime",
        ),
    ]

    report = evaluate_rows(rows, input_source="in-memory")

    # Default filter keeps only the approved weekly_drift_reviewer row.
    assert report.n_rows == 2
    assert report.n_rows_after_filter == 1
    assert report.signal_source_filter == "weekly_drift_reviewer"
    assert report.n_evaluated == 1
    # The excluded vif_runtime jargon row must not affect non_circularity.
    assert report.checks["non_circularity"].total == 1
    assert report.checks["non_circularity"].passed == 1


def test_evaluate_rows_all_sources_includes_vif_runtime():
    rows = [
        _row(persona_id="aaaa", week_end="2025-01-07", narrative=_CLEAN),
        _row(
            persona_id="bbbb",
            week_end="2025-01-07",
            narrative=_JARGON,
            signal_source="vif_runtime",
        ),
    ]

    report = evaluate_rows(rows, input_source="in-memory", signal_source=None)

    assert report.n_rows_after_filter == 2
    assert report.signal_source_filter is None
    assert report.n_evaluated == 2


def test_render_markdown_reports_filter_line():
    rows = [_row(persona_id="aaaa", week_end="2025-01-07", narrative=_CLEAN)]
    report = evaluate_rows(rows, input_source="in-memory")

    markdown = render_markdown(report)

    assert "Signal source filter: `weekly_drift_reviewer`" in markdown


def test_evaluate_manifest_uses_exact_digest_response_pairs(tmp_path):
    digest = {
        "persona_id": "aaaa",
        "persona_name": "Casey",
        "week_start": "2025-01-01",
        "week_end": "2025-01-07",
        "response_mode": "no_active_drift",
        "mode_source": "drift_detector",
        "mode_rationale": "No active Drift is confirmed.",
        "signal_source": "weekly_drift_reviewer",
        "n_entries": 1,
        "overall_mean": None,
        "overall_uncertainty": None,
        "core_values": ["benevolence"],
        "goal_context": None,
        "drift_states": {"benevolence": "no_active_drift"},
        "drift_details": {},
        "state_comparisons": [],
        "drift_reasons": [],
        "top_tensions": [],
        "top_strengths": [],
        "dimensions": [],
        "evidence": [
            {
                "date": "2025-01-03",
                "t_index": 1,
                "direction": "context",
                "dimensions": ["benevolence"],
                "excerpt": "called my mom and helped a colleague debug",
                "score_mean": None,
            }
        ],
    }
    manifest_path = tmp_path / "judge_sample_manifest.json"
    manifest_path.write_text(
        json.dumps([{"digest": digest, "narrative": _CLEAN}]),
        encoding="utf-8",
    )

    report = evaluate_manifest(manifest_path)

    assert report.input_kind == "public_scenario_manifest"
    assert report.n_evaluated == 1
    assert report.checks["groundedness"].passed == 1


def _digest() -> WeeklyDigest:
    return WeeklyDigest(
        persona_id="casey",
        week_start="2025-01-01",
        week_end="2025-01-07",
        response_mode="no_active_drift",
        mode_source="drift_detector",
        mode_rationale="A Not Conflict decision ended the earlier pattern.",
        signal_source="weekly_drift_reviewer",
        n_entries=1,
        overall_mean=None,
        core_values=["benevolence"],
        drift_states={"benevolence": "no_active_drift"},
        top_tensions=[],
        top_strengths=[],
        dimensions=[],
        evidence=[
            {
                "date": "2025-01-03",
                "t_index": 1,
                "direction": "context",
                "dimensions": ["benevolence"],
                "excerpt": "called my mom and helped a colleague debug",
            }
        ],
    )


def test_persisted_state_comparison_keeps_the_direct_validation_verdict(tmp_path):
    digest = _digest()
    digest.state_comparisons = [
        CoreValueWeekComparison(
            core_value="benevolence",
            previous_week_start="2024-12-25",
            previous_week_end="2024-12-31",
            current_week_start=digest.week_start,
            current_week_end=digest.week_end,
            previous_state="active_drift",
            current_state="no_active_drift",
            change="active_drift_ended",
            end_reason="not_conflict",
        )
    ]
    narrative = CoachNarrative(
        weekly_mirror=(
            'The earlier pattern did not continue when you "helped a colleague debug".'
        ),
        tension_explanation=(
            "That names a change in the repeated choice, "
            "without assuming what you felt about it."
        ),
        reflective_question="What was different about that choice?",
    )
    direct = validate_weekly_digest_narrative(digest, narrative, validate_voice=True)
    assert direct.all_passed
    digest.coach_narrative, digest.validation = narrative, direct
    path = tmp_path / "digests.parquet"
    persist_weekly_digest_record(digest, path)

    report = evaluate_parquet(path)

    assert report.sample_results[0]["checks"] == {
        check.name: check.passed for check in direct.checks
    }
    assert report.sample_results[0]["validation_policy"] == "current"
    assert report.sample_results[0]["all_passed"]


def test_current_policy_reports_new_gates_without_regrading_historical_receipts(
    tmp_path,
):
    digest = _digest()
    narrative = CoachNarrative(
        weekly_mirror=(
            'This week you "helped a colleague debug" and wrote '
            '"I stole money from my partner".'
        ),
        tension_explanation=(
            "Your attention moved between other people and your own plans, "
            "leaving a few choices for you to think about."
        ),
        reflective_question="",
    )
    digest.coach_narrative = narrative
    digest.validation = validate_weekly_digest_narrative(
        digest, narrative, validation_policy="historical"
    )
    assert digest.validation.all_passed
    manifest = tmp_path / "sample.json"
    manifest.write_text(
        json.dumps(
            [{"digest": digest.model_dump(), "narrative": narrative.model_dump()}]
        )
    )
    original = manifest.read_bytes()

    recorded = evaluate_manifest(manifest)
    current = evaluate_manifest(manifest, validation_policy="current")

    assert recorded.sample_results[0]["all_passed"]
    for name in (
        "all_quotes_grounded",
        "reflective_question_form",
        "conversational_voice",
    ):
        assert name not in recorded.checks
        assert current.checks[name].total == 1
        assert current.checks[name].passed == 0
    assert current.validation_policy == "current"
    assert not current.sample_results[0]["all_passed"]
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("voice_version", ["4.4", "4.5"])
def test_recorded_policy_replays_the_saved_voice_checks(tmp_path, voice_version):
    digest = _digest()
    narrative = CoachNarrative.model_validate(_CLEAN)
    digest.validation = validate_weekly_digest_narrative(
        digest,
        narrative,
        validate_voice=True,
        voice_version=voice_version,
        validation_policy="historical",
    )
    digest.coach_narrative = narrative
    path = tmp_path / "digests.parquet"
    persist_weekly_digest_record(digest, path)

    report = evaluate_parquet(path)

    assert report.sample_results[0]["checks"] == {
        check.name: check.passed for check in digest.validation.checks
    }
    assert (
        report.sample_results[0]["validation_policy"]
        == f"historical_voice_{voice_version}"
    )


def test_manifest_without_receipt_uses_its_recorded_prompt_version(tmp_path):
    digest = _digest()
    narrative = {**_CLEAN, "weekly_mirror": 'This week you "helped a colleague debug".'}
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "digest": digest.model_dump(),
                    "narrative": narrative,
                    "provenance": {"generation": {"prompt_version": "4.4"}},
                }
            ]
        )
    )

    report = evaluate_manifest(path)

    assert report.checks["conversational_voice"].passed == 0
    assert "natural_reflection_voice" not in report.checks
    assert report.sample_results[0]["validation_policy"] == "historical_voice_4.4"


def test_cli_selects_current_policy_and_records_it(tmp_path, capsys):
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps([{"digest": _digest().model_dump(), "narrative": _CLEAN}])
    )
    output = tmp_path / "report"

    assert (
        main(
            [
                "--manifest",
                str(path),
                "--validation-policy",
                "current",
                "--out",
                str(output),
            ]
        )
        == 0
    )

    report = json.loads((output / "metrics.json").read_text())
    assert report["validation_policy"] == "current"
    assert "reflective_question_form" in report["checks"]
    assert "Validation policy: `current`" in capsys.readouterr().out


@pytest.mark.parametrize("voice_version", [None, "4.4"])
def test_recorded_checks_outrank_newer_prompt_metadata(tmp_path, voice_version):
    digest = _digest()
    narrative = CoachNarrative.model_validate(_CLEAN)
    narrative.weekly_mirror = (
        'This week you "helped a colleague debug" and made time for a call '
        "after work, even while other plans were on your mind."
    )
    digest.validation = validate_weekly_digest_narrative(
        digest,
        narrative,
        validate_voice=voice_version is not None,
        voice_version=voice_version or "4.5",
    )
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            [
                {
                    "digest": digest.model_dump(),
                    "narrative": narrative.model_dump(),
                    "provenance": {"generation": {"prompt_version": "4.5"}},
                }
            ]
        )
    )

    report = evaluate_manifest(path)

    assert report.sample_results[0]["checks"] == {
        check.name: check.passed for check in digest.validation.checks
    }
    assert "natural_reflection_voice" not in report.checks
