"""Tests for the bounded twinkl-752.3 prompt-alignment study."""

from pathlib import Path

import pytest

from scripts.experiments import weekly_drift_reviewer_prompt_alignment as alignment
from scripts.experiments import weekly_verifier_ablation as baseline

ROOT = Path(__file__).resolve().parents[2]
CONFIG = baseline._read_yaml(ROOT / alignment.DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="module")
def prepared() -> tuple[list[dict], dict]:
    return alignment.build_prompt_records(CONFIG, ROOT)


def _not_conflict_response(record: dict) -> baseline.AlignedWeeklyVerifierResponse:
    return baseline.AlignedWeeklyVerifierResponse(
        rubric_version="drift_v1_conflict_rubric_v1",
        assessments=[
            baseline.VerifierAssessment(
                t_index=item["t_index"],
                dimension=item["dimension"],
                verdict="not_conflict",
                confidence="high",
                reason_code="direct_aligned_or_neutral_behavior",
                evidence_quote="",
            )
            for item in record["expected_coordinates"]
        ],
        pair_assessments=[
            baseline.DriftPairAssessment(
                first_t_index=item["first_t_index"],
                second_t_index=item["second_t_index"],
                dimension=item["dimension"],
                sustained_conflict="no",
            )
            for item in record["expected_pairs"]
        ],
    )


def test_aligned_surface_is_frozen(prepared) -> None:
    records, manifest = prepared

    assert len(records) == 126
    assert manifest["planned_successful_calls"] == 378
    assert sum(len(row["current_expected_coordinates"]) for row in records) == 335
    assert sum(len(row["expected_coordinates"]) for row in records) == 487
    assert sum(len(row["expected_pairs"]) for row in records) == 293
    assert (
        sum(
            item["crosses_week_boundary"]
            for row in records
            for item in row["expected_pairs"]
        )
        == 152
    )
    assert manifest["locked_final_test_used"] is False
    assert manifest["retired_benchmark_used"] is False
    for reviewer in ("reviewer_a", "reviewer_b"):
        path = baseline._rooted(CONFIG["population"][f"{reviewer}_path"], ROOT)
        assert manifest[f"{reviewer}_sha256"] == baseline._sha256_file(path)


def test_pair_blocks_repeat_both_complete_journal_entries(prepared) -> None:
    records, _manifest = prepared
    record = next(row for row in records if row["expected_pairs"])
    pair = record["expected_pairs"][0]
    first = record["entry_text_by_t_index"][str(pair["first_t_index"])]
    second = record["entry_text_by_t_index"][str(pair["second_t_index"])]

    assert first in record["prompt"]
    assert second in record["prompt"]
    assert (
        f"[PAIR t_index={pair['first_t_index']} -> t_index={pair['second_t_index']}]"
    ) in record["prompt"]


def test_aligned_validation_enforces_quotes_and_pair_truth_table(prepared) -> None:
    records, _manifest = prepared
    record = next(row for row in records if row["expected_pairs"])
    parsed = _not_conflict_response(record)
    baseline.validate_parsed_response(parsed=parsed, record=record)

    bad_pair = parsed.model_copy(deep=True)
    bad_pair.pair_assessments[0] = bad_pair.pair_assessments[0].model_copy(
        update={"sustained_conflict": "yes"}
    )
    with pytest.raises(ValueError, match="inconsistent"):
        baseline.validate_parsed_response(parsed=bad_pair, record=record)

    bad_quote = parsed.model_copy(deep=True)
    bad_quote.assessments[0] = bad_quote.assessments[0].model_copy(
        update={
            "verdict": "conflict",
            "reason_code": "direct_behavior_or_choice",
            "evidence_quote": "fabricated evidence",
        }
    )
    with pytest.raises(ValueError, match="not present"):
        baseline.validate_parsed_response(parsed=bad_quote, record=record)


def test_boundary_reassessment_is_excluded_from_entry_metrics(prepared) -> None:
    records, _manifest = prepared
    boundary = next(
        row
        for row in records
        if any(item["crosses_week_boundary"] for item in row["expected_pairs"])
    )
    parsed = _not_conflict_response(boundary)
    response = {
        "status": "ok",
        "persona_id": boundary["persona_id"],
        "week_start": boundary["week_start"],
        "arm": boundary["arm"],
        "repeat": 1,
        "parsed": parsed.model_dump(mode="json"),
    }

    entry_predictions, _pairs = alignment._aligned_predictions([response], [boundary])
    assert len(entry_predictions) == len(boundary["current_expected_coordinates"])
    assert len(boundary["expected_coordinates"]) > len(entry_predictions)


def test_overlapping_pair_claims_collapse_to_one_drift() -> None:
    case = {
        "persona_id": "p",
        "dimension": "security",
        "entries": [
            {"t_index": index, "date": f"2026-01-0{index + 1}"} for index in range(3)
        ],
    }
    pair_predictions = {
        (1, "p", first, second, "security"): baseline.DriftPairAssessment(
            first_t_index=first,
            second_t_index=second,
            dimension="security",
            sustained_conflict="yes",
        )
        for first, second in ((0, 1), (1, 2))
    }

    metrics, case_predictions, _timing, _slices = alignment._aligned_drift_metrics(
        cases={"case_001": case},
        episode_targets={"case_001": False},
        conflict_by_coordinate={("p", "security", index): False for index in range(3)},
        pair_predictions=pair_predictions,
        repeat=1,
    )

    assert metrics["fp"] == 1
    assert case_predictions == {"case_001": True}


@pytest.mark.parametrize(
    ("aligned_recall", "cross_week_repeats", "expected"),
    [(0.6, 2, "prompt_limited"), (0.4, 0, "unchanged"), (0.6, 1, "inconclusive")],
)
def test_preregistered_decision_rule(
    aligned_recall: float, cross_week_repeats: int, expected: str
) -> None:
    current = [
        {"episode": {"recall": 0.4, "fp": 1.0, "coverage": 0.75}} for _ in range(3)
    ]
    aligned = [
        {
            "episode": {"recall": aligned_recall, "fp": 1.0, "coverage": 0.75},
            "drift_slices": {
                "cross_week": {"detected": int(index < cross_week_repeats)}
            },
        }
        for index in range(3)
    ]

    assert alignment._decision(current, aligned)["verdict"] == expected


def test_paid_run_guard(prepared) -> None:
    records, _manifest = prepared
    estimate = baseline.estimate_plan(records, CONFIG)

    assert estimate["successful_calls"] == 378
    assert estimate["estimated_cost_usd"] < estimate["max_budget_usd"]
    with pytest.raises(SystemExit, match="Refusing paid calls"):
        alignment.command_run(
            baseline.argparse.Namespace(
                execute=False,
                root=ROOT,
                config=alignment.DEFAULT_CONFIG_PATH,
            )
        )


def test_saved_result_is_complete_and_reproducible() -> None:
    records, paths = alignment._load_prepared(CONFIG, ROOT)
    responses = baseline._load_jsonl(paths["responses"])
    saved = baseline._read_json(paths["metrics"])
    rescored = alignment.score_responses(
        config=CONFIG,
        root=ROOT,
        records=records,
        responses=responses,
    )

    assert len(responses) == 378
    assert {
        repeat: sum(row["repeat"] == repeat for row in responses)
        for repeat in range(1, 4)
    } == {1: 126, 2: 126, 3: 126}
    assert all(row["requested_model"] == CONFIG["api"]["model"] for row in responses)
    assert all(row["resolved_model"] == CONFIG["api"]["model"] for row in responses)
    assert all(row["response_id"] for row in responses)
    assert all((row.get("usage") or {}).get("total_tokens", 0) > 0 for row in responses)
    assert saved["response_summary"]["statuses"] == {"invalid": 27, "ok": 351}
    assert saved["decision"]["verdict"] == "unchanged"
    assert saved["decision"]["aligned_prompt"] == pytest.approx(
        {
            "median_drift_recall": 0.2,
            "median_false_drift_alerts": 5.0,
            "median_coverage": 0.8292682926829268,
            "cross_week_reference_drift_recovered_repeats": 0,
        }
    )
    saved.pop("scored_at")
    saved.pop("artifact_provenance")
    rescored.pop("scored_at")
    assert rescored == saved
