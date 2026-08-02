"""Tests for the Coach Digest Validations batch report."""

from __future__ import annotations

import json

from src.evals.coach_digest_validations import (
    evaluate_rows,
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

    report = evaluate_rows(rows, parquet_source="in-memory")

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


def test_evaluate_rows_skips_unparseable_narrative():
    bad = _row(persona_id="dddd", week_end="2025-01-07", narrative=None)
    bad["coach_narrative_json"] = "{not valid json"
    rows = [
        _row(persona_id="aaaa", week_end="2025-01-07", narrative=_CLEAN),
        bad,
    ]

    report = evaluate_rows(rows, parquet_source="in-memory")

    assert report.n_with_narrative == 2
    assert report.n_evaluated == 1
    assert "dddd:2025-01-07" in report.skipped


def test_render_markdown_contains_source_disclaimer():
    rows = [_row(persona_id="aaaa", week_end="2025-01-07", narrative=_CLEAN)]
    report = evaluate_rows(rows, parquet_source="in-memory")

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

    report = evaluate_rows(rows, parquet_source="in-memory")

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

    report = evaluate_rows(rows, parquet_source="in-memory", signal_source=None)

    assert report.n_rows_after_filter == 2
    assert report.signal_source_filter is None
    assert report.n_evaluated == 2


def test_render_markdown_reports_filter_line():
    rows = [_row(persona_id="aaaa", week_end="2025-01-07", narrative=_CLEAN)]
    report = evaluate_rows(rows, parquet_source="in-memory")

    markdown = render_markdown(report)

    assert "Signal source filter: `weekly_drift_reviewer`" in markdown
