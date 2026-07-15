from __future__ import annotations

from pathlib import Path

import pytest

from src.drift_review_app.data import (
    EXPECTED_COUNTS,
    QUEUE_LABELS,
    ReviewData,
    load_review_data,
    preserved_group_signature,
)
from src.drift_rules import drift_spans, trajectory_covered

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def review_data() -> ReviewData:
    return load_review_data(ROOT)


def test_source_counts_and_join_completeness(review_data: ReviewData) -> None:
    for key, expected in EXPECTED_COUNTS.items():
        assert review_data.integrity[key] == expected
    assert len(review_data.cases) == 292
    assert len(review_data.decisions) == 2377 * 3 * 3
    assert {
        (entry.persona_id, entry.t_index)
        for case in review_data.cases.values()
        for entry in case.entries
    } == {
        (persona_id, t_index)
        for persona_id in review_data.profiles
        for t_index in {
            entry.t_index
            for case in review_data.cases_for_persona(persona_id)
            for entry in case.entries
        }
    }


def test_three_manifest_backed_setup_mappings(review_data: ReviewData) -> None:
    assert {
        key: (spec.model, spec.reasoning_effort)
        for key, spec in review_data.setup_specs.items()
    } == {
        "mini_none": ("gpt-5.4-mini-2026-03-17", "none"),
        "luna_none": ("gpt-5.6-luna", "none"),
        "luna_low": ("gpt-5.6-luna", "low"),
    }
    provenance = review_data.integrity["reasoning_effort_provenance"]
    assert "Frozen manifest and registered configuration" in provenance
    assert "individual receipts do not record reasoning effort" in provenance


def test_each_run_has_exactly_951_receipts(review_data: ReviewData) -> None:
    for setup_key in review_data.setup_specs:
        for run in (1, 2, 3):
            rows = [
                key
                for key in review_data.receipt_statuses
                if key[0] == setup_key and key[1] == run
            ]
            assert len(rows) == 951
    assert review_data.integrity["response_summaries"] == {
        "mini_none": {"ok": 2815, "invalid": 38},
        "luna_none": {"ok": 2837, "invalid": 16},
        "luna_low": {"ok": 2845, "invalid": 8},
    }


def test_frozen_prompt_boundaries_exclude_later_journal_entries(
    review_data: ReviewData,
) -> None:
    assert len(review_data.prompt_boundaries) == 951
    boundaries = review_data.boundaries_for_persona("02fb94f3")
    first, second = boundaries[:2]
    assert first.week_start == "2025-04-14"
    assert first.visible_t_indices == (0, 1)
    assert first.current_t_indices == (0, 1)
    assert first.cutoff_t_index == 1
    assert first.declared_values == ("self_direction", "tradition")
    assert second.visible_t_indices == (0, 1, 2, 3)
    assert second.current_t_indices == (2, 3)
    assert second.cutoff_t_index == 3
    later_t_indices = {
        entry.t_index
        for entry in review_data.cases["02fb94f3:tradition"].entries
        if entry.t_index > first.cutoff_t_index
    }
    assert later_t_indices
    assert later_t_indices.isdisjoint(first.visible_t_indices)
    assert (
        "empty VIF Critic input block"
        in review_data.integrity["reviewer_input_provenance"]
    )


def test_runs_are_isolated_and_align_to_shared_entries(review_data: ReviewData) -> None:
    for case_id, case in review_data.cases.items():
        expected = {entry.t_index for entry in case.entries}
        for setup_key in review_data.setup_specs:
            for run in (1, 2, 3):
                observed = {
                    key[3]
                    for key in review_data.decisions
                    if key[:3] == (setup_key, run, case_id)
                }
                assert observed == expected
    example = review_data.integrity["examples"]["run_disagreement"]
    assert review_data.run_disagreement("luna_low", example)


def test_reference_and_predicted_drift_use_consecutive_t_indices(
    review_data: ReviewData,
) -> None:
    case_id = "02fb94f3:tradition"
    assert [
        (row.onset_t_index, row.confirmation_t_index, row.end_t_index)
        for row in review_data.reference_drifts[case_id]
    ] == [(4, 5, 6), (9, 10, 10)]
    assert [
        (row.onset_t_index, row.confirmation_t_index, row.end_t_index)
        for row in review_data.predicted_drifts[("luna_low", 1, case_id)]
    ] == [(4, 5, 5)]
    assert drift_spans([True, True, True, False], [4, 5, 6, 7]) == [(4, 5, 6)]
    assert drift_spans([True, True], [4, 6]) == []


def test_uncertain_reference_labels_stay_distinct(
    review_data: ReviewData,
) -> None:
    case = review_data.cases["3a3b15e4:tradition"]
    uncertain = [entry for entry in case.entries if entry.final_conflict is None]
    assert len(uncertain) == 2
    assert all(entry.resolution_status != "resolved" for entry in uncertain)
    for entry in uncertain:
        decisions = [
            review_data.decision("luna_low", run, case.case_id, entry.t_index)
            for run in (1, 2, 3)
        ]
        assert all(
            decision.response_status in {"ok", "invalid"} for decision in decisions
        )
    assert len(review_data.reference_drifts[case.case_id]) == 1


def test_cross_week_reference_and_prediction_flags(review_data: ReviewData) -> None:
    references = review_data.reference_drifts["02fb94f3:tradition"]
    assert references and all(row.crosses_week for row in references)
    predicted = [
        row
        for rows in review_data.predicted_drifts.values()
        for row in rows
        if row.crosses_week
    ]
    assert predicted


def test_missed_and_false_drift_alert_classification(review_data: ReviewData) -> None:
    known = review_data.case_metrics[("luna_low", 1, "02fb94f3:tradition")]
    assert known["drift_hits"] == 1
    assert known["missed_drifts"] == 1
    false_case = review_data.integrity["examples"]["false_drift_alert"]
    assert any(
        review_data.case_metrics[("luna_low", run, false_case)]["false_drift_alerts"]
        for run in (1, 2, 3)
    )
    assert false_case in review_data.queue_case_ids("False Drift alert", "luna_low")


def test_detection_dates_and_signed_delays(review_data: ReviewData) -> None:
    rows = review_data.predicted_drifts[("mini_none", 1, "abf1ce49:security")]
    early = next(row for row in rows if row.delay_entries == -1)
    assert early.detection_date == "2025-10-12"
    assert early.delay_days == -2
    assert early.delay_entries == -1
    false_rows = [
        row
        for rows in review_data.predicted_drifts.values()
        for row in rows
        if row.result == "false Drift alert"
    ]
    assert false_rows
    assert all(row.detection_date and row.delay_days is None for row in false_rows)


def test_exact_coverage_and_abstain_behavior(review_data: ReviewData) -> None:
    assert trajectory_covered([True, True], [0, 1])
    assert trajectory_covered([False, None, False], [0, 1, 2])
    assert not trajectory_covered([True, None], [0, 1])
    case_id = review_data.integrity["examples"]["unresolved_abstain"]
    assert case_id in review_data.queue_case_ids(
        "Unresolved because of Abstain", "luna_low"
    )
    unresolved = [
        (run, pair)
        for run in (1, 2, 3)
        for pair in review_data.unresolved_pairs("luna_low", run, case_id)
    ]
    assert unresolved
    for run, pair in unresolved:
        verdicts = {
            review_data.decision("luna_low", run, case_id, t_index).verdict
            for t_index in pair
        }
        assert "abstain" in verdicts
        assert "not_conflict" not in verdicts


def test_invalid_responses_remain_fail_closed(review_data: ReviewData) -> None:
    case_id = review_data.integrity["examples"]["invalid_response"]
    invalid = [
        review_data.decision("luna_low", run, case_id, entry.t_index)
        for run in (1, 2, 3)
        for entry in review_data.cases[case_id].entries
        if review_data.decision("luna_low", run, case_id, entry.t_index).response_status
        != "ok"
    ]
    assert invalid
    assert all(row.verdict is None and row.confidence is None for row in invalid)
    assert case_id in review_data.queue_case_ids("Invalid response", "luna_low")


def test_model_disagreement_compares_groups_without_voting(
    review_data: ReviewData,
) -> None:
    assert preserved_group_signature(
        ["conflict", "not_conflict", "conflict"]
    ) == preserved_group_signature(["conflict", "conflict", "not_conflict"])
    assert preserved_group_signature(
        ["conflict", "conflict", "not_conflict"]
    ) != preserved_group_signature(["conflict", "conflict", "conflict"])
    example = review_data.integrity["examples"]["model_disagreement"]
    assert review_data.model_disagreement(example)
    assert example in review_data.queue_case_ids("Model disagreement", "mini_none")


def test_all_review_queues_have_live_examples(review_data: ReviewData) -> None:
    for queue in QUEUE_LABELS:
        assert review_data.queue_case_ids(queue, "luna_low"), queue


def test_known_02fb94f3_tradition_cross_setup_difference(
    review_data: ReviewData,
) -> None:
    case_id = "02fb94f3:tradition"
    mini_hits = sum(
        review_data.case_metrics[("mini_none", run, case_id)]["drift_hits"]
        for run in (1, 2, 3)
    )
    low_hits = sum(
        review_data.case_metrics[("luna_low", run, case_id)]["drift_hits"]
        for run in (1, 2, 3)
    )
    assert mini_hits == 0
    assert low_hits == 1
