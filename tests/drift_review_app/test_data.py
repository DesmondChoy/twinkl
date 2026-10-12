from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import median

import pytest

from src.drift_review_app.data import (
    CURRENT_SETUP_KEY,
    EXPECTED_COUNTS,
    HISTORICAL_SETUP_KEYS,
    QUEUE_LABELS,
    ReviewData,
    _read_jsonl,
    _sha256_file,
    load_review_data,
    preserved_group_signature,
)
from src.drift_review_app.experiments import (
    _normalize_prompts,
    _normalize_responses,
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
    assert len(review_data.decisions) == 2377 * 3 * len(review_data.setup_specs)
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
        if key in HISTORICAL_SETUP_KEYS
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
    assert {
        key: review_data.integrity["response_summaries"][key]
        for key in HISTORICAL_SETUP_KEYS
    } == {
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
    observed_by_run = defaultdict(set)
    for setup_key, run, case_id, t_index in review_data.decisions:
        observed_by_run[(setup_key, run, case_id)].add(t_index)
    for case_id, case in review_data.cases.items():
        expected = {entry.t_index for entry in case.entries}
        for setup_key in review_data.setup_specs:
            for run in (1, 2, 3):
                observed = observed_by_run[(setup_key, run, case_id)]
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


def test_report_prompt_groups_preserve_all_studies(review_data: ReviewData) -> None:
    assert len(review_data.setup_specs) == 18
    assert set(review_data.setups_for_prompt("v0")) == {
        "mini_none",
        "luna_none",
        "luna_low",
        "luna_medium",
        "luna_high",
        "luna_xhigh",
    }
    assert set(review_data.setups_for_prompt("v1")) == {"definitions_off_low"}
    assert len(review_data.setups_for_prompt("v2")) == 11
    for version, archive in (("v0", "2.0"), ("v1", "3.0"), ("v2", "4.0")):
        assert all(
            spec.archive_prompt_version == archive
            for spec in review_data.setups_for_prompt(version).values()
        )
    assert review_data.setup_specs[CURRENT_SETUP_KEY].selected
    assert not review_data.setup_specs["definitions_on_low"].accuracy_evaluated
    assert review_data.setup_specs["definitions_on_low"].study_id != (
        review_data.setup_specs[CURRENT_SETUP_KEY].study_id
    )


def test_submitted_report_current_and_historical_medians(
    review_data: ReviewData,
) -> None:
    for key, hits, false_alerts, coverage in (
        ("mini_none", 7, 5, 0.740),
        ("luna_none", 20, 13, 0.777),
        ("luna_low", 23, 4, 0.637),
        (CURRENT_SETUP_KEY, 17, 2, 0.990),
    ):
        rows = review_data.aggregate_results[key]
        assert median(row["drift_hits"] for row in rows) == hits
        assert median(row["false_drift_alerts"] for row in rows) == false_alerts
        assert round(median(row["coverage"] for row in rows), 3) == coverage
        assert median(row["drift_recall"] for row in rows) == hits / 42
    assert [
        row["drift_hits"] for row in review_data.aggregate_results[CURRENT_SETUP_KEY]
    ] == [17, 15, 17]
    costs = review_data.integrity["calculated_cost_usd"]
    assert round(costs["gpt56_low"], 4) == 1.2730
    assert round(costs[CURRENT_SETUP_KEY], 4) == 0.5083


def test_definition_counts_have_original_scope(review_data: ReviewData) -> None:
    summary = review_data.integrity["definitions_study"]
    assert summary["baseline_counts"] == [22, 22, 21]
    assert summary["definitions_counts"] == [19, 21, 18]
    assert summary["changed_entry_decisions"] == [305, 308, 319]
    assert summary["response_statuses"] == {"ok": 5684, "invalid": 22}
    for key in ("definitions_off_low", "definitions_on_low"):
        assert all(
            "drift_recall" not in row and "false_drift_alerts" not in row
            for row in review_data.aggregate_results[key]
        )


def test_inspector_boundaries_belong_to_selected_prompt(
    review_data: ReviewData,
) -> None:
    before = review_data.boundaries_for_persona("02fb94f3", "luna_low")
    split = review_data.boundaries_for_persona("02fb94f3", "definitions_off_low")
    after = review_data.boundaries_for_persona("02fb94f3", CURRENT_SETUP_KEY)
    assert len(before) == len(split) == len(after)
    assert before[0].prompt_sha256 != split[0].prompt_sha256 != after[0].prompt_sha256
    assert (
        before[0].visible_t_indices
        == split[0].visible_t_indices
        == after[0].visible_t_indices
    )
    assert after == review_data.boundaries_for_persona("02fb94f3", "gpt56_none")
    assert before == review_data.boundaries_for_persona("02fb94f3")


def test_current_error_and_invalid_receipts_stay_fail_closed(
    review_data: ReviewData,
) -> None:
    failures = [
        decision
        for key, decision in review_data.decisions.items()
        if key[0] == "gpt6_none" and decision.response_status != "ok"
    ]
    assert {decision.response_status for decision in failures} == {"error", "invalid"}
    assert all(
        decision.verdict is None
        and decision.confidence is None
        and not decision.evidence_quote
        for decision in failures
    )
    assert review_data.integrity["response_summaries"]["gpt6_none"] == {
        "ok": 2841,
        "invalid": 7,
        "error": 5,
    }


def test_model_disagreement_does_not_cross_prompt_studies(
    review_data: ReviewData,
) -> None:
    assert not any(
        review_data.model_disagreement(case_id, "definitions_off_low")
        for case_id in review_data.cases
    )


@pytest.fixture(scope="module")
def frozen_current_example() -> tuple[dict, list[dict], list[dict]]:
    directory = ROOT / "logs/experiments/artifacts/twinkl_q6pt_luna_model_20260923"
    with (directory / "responses.jsonl").open() as stream:
        receipt = json.loads(stream.readline())
    source = (
        ROOT
        / "logs/experiments/artifacts"
        / "twinkl_j3k7_core_value_definitions_20260907/requests.jsonl"
    )
    row = next(
        row for row in _read_jsonl(source) if row["case_id"] == receipt["case_id"]
    )
    row["variants"] = {receipt["variant"]: row["variants"]["definitions"]}
    receipts = [
        item
        for item in _read_jsonl(directory / "responses.jsonl")
        if item["variant"] == receipt["variant"]
        and item["case_id"] == receipt["case_id"]
    ]
    events = [
        event
        for event in _read_jsonl(directory / "attempts.jsonl")
        if event["request_key"] in {item["request_key"] for item in receipts}
    ]
    return row, receipts, events


def test_frozen_prompt_payload_hash_rejects_edits(
    frozen_current_example: tuple,
) -> None:
    row = deepcopy(frozen_current_example[0])
    row["variants"]["gpt_6_none"]["instructions"] += " altered"
    with pytest.raises(ValueError, match="request hash mismatch"):
        _normalize_prompts([row], "gpt_6_none")


def test_frozen_history_hash_rejects_edits(frozen_current_example: tuple) -> None:
    row = deepcopy(frozen_current_example[0])
    row["request"]["history"][0]["text"] += " altered"
    with pytest.raises(ValueError, match="history hash mismatch"):
        _normalize_prompts([row], "gpt_6_none")


@pytest.mark.parametrize("tamper", ["model", "request", "coordinate", "failed_output"])
def test_current_receipt_provenance_and_fail_closed_validation(
    frozen_current_example: tuple,
    tmp_path: Path,
    tamper: str,
) -> None:
    row, receipts, events = deepcopy(frozen_current_example)
    receipt = receipts[0]
    final = next(
        event
        for event in reversed(events)
        if event["request_key"] == receipt["request_key"]
    )
    if tamper == "model":
        final["resolved_model"] = "gpt-5.6-luna"
        expected_error = "Unexpected provider model"
    elif tamper == "request":
        receipt["request_sha256"] = "incorrect"
        expected_error = "receipt request hash differs"
    elif tamper == "coordinate":
        receipt["decisions"][0]["week_end"] = "2025-03-03"
        expected_error = "decision boundary differs"
    else:
        final["status"] = "error"
        receipt["status"] = "error"
        for decision in receipt["decisions"]:
            decision["review_status"] = "error"
        expected_error = "decision differs from validated provider output"
    (tmp_path / "responses.jsonl").write_text(
        "\n".join(json.dumps(item) for item in receipts) + "\n"
    )
    (tmp_path / "attempts.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    metrics = {
        "responses_sha256": _sha256_file(tmp_path / "responses.jsonl"),
        "attempts_sha256": _sha256_file(tmp_path / "attempts.jsonl"),
    }
    with pytest.raises(ValueError, match=expected_error):
        _normalize_responses(
            directory=tmp_path,
            rows=[row],
            prompts={"gpt_6_none": _normalize_prompts([row], "gpt_6_none")},
            arm_keys={"gpt_6_none": "gpt6_none"},
            metrics=metrics,
        )
