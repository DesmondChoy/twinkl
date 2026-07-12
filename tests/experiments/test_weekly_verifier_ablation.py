"""Tests for the bounded twinkl-752.1 weekly verifier ablation."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "scripts/experiments/weekly_verifier_ablation.py"
SPEC = importlib.util.spec_from_file_location("weekly_verifier_ablation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ablation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ablation
SPEC.loader.exec_module(ablation)
CONFIG = ablation._read_yaml(ROOT / ablation.DEFAULT_CONFIG_PATH)


@pytest.fixture(scope="module")
def prepared() -> tuple[list[dict], dict]:
    return ablation.build_prompt_records(CONFIG, ROOT)


def test_live_population_and_call_surface_are_frozen(prepared) -> None:
    records, manifest = prepared

    assert manifest["persona_count"] == 28
    assert manifest["case_count"] == 42
    assert manifest["entry_count"] == 217
    assert manifest["persona_week_count"] == 126
    assert manifest["prompt_count"] == 252
    assert manifest["planned_successful_calls"] == 756
    assert manifest["target_split"] == "development"
    assert manifest["locked_promotion_used"] is False
    assert manifest["retired_benchmark_used"] is False
    assert manifest["mlp_seed_input_divergence"]["run_ids"] == [
        "run_019",
        "run_020",
        "run_021",
    ]
    assert manifest["mlp_seed_input_divergence"]["cell_count"] == 335
    assert len({(row["persona_id"], row["week_start"]) for row in records}) == 126


def test_iso_weeks_use_cumulative_history_and_current_week_outputs(prepared) -> None:
    records, _manifest = prepared
    without = [row for row in records if row["arm"] == "without_critic"]

    assert all(
        ablation.date.fromisoformat(row["week_start"]).weekday() == 0 for row in without
    )
    assert all(
        ablation.date.fromisoformat(row["week_end"])
        - ablation.date.fromisoformat(row["week_start"])
        == ablation.timedelta(days=6)
        for row in without
    )
    by_persona: dict[str, list[dict]] = {}
    for row in without:
        by_persona.setdefault(row["persona_id"], []).append(row)
    multiweek = next(rows for rows in by_persona.values() if len(rows) > 1)
    multiweek.sort(key=lambda row: row["week_start"])
    first_sections = ablation._entry_sections(multiweek[0]["prompt"])
    second_sections = ablation._entry_sections(multiweek[1]["prompt"])

    assert len(second_sections) > len(first_sections)
    assert set(multiweek[0]["current_t_indices"]).isdisjoint(
        set(multiweek[1]["current_t_indices"])
    )


def test_arms_are_identical_except_for_the_critic_block(prepared) -> None:
    records, _manifest = prepared
    paired: dict[tuple[str, str], dict[str, dict]] = {}
    for row in records:
        paired.setdefault((row["persona_id"], row["week_start"]), {})[row["arm"]] = row
    pair = next(iter(paired.values()))
    without = pair["without_critic"]
    with_critic = pair["with_critic"]
    prompt = with_critic["prompt"]
    start = prompt.index("\nCRITIC SIGNALS FOR CURRENT-WEEK ENTRIES")
    end = prompt.index("\n\nThe Critic, when present", start)
    stripped = prompt[:start] + prompt[end:]

    assert stripped == without["prompt"]
    assert "P(-1)" not in without["prompt"]
    assert "MC uncertainty" not in without["prompt"]
    assert "P(-1)" in with_critic["prompt"]
    assert without["expected_coordinates"] == with_critic["expected_coordinates"]
    assert without["runtime_text_sha256"] == with_critic["runtime_text_sha256"]


def test_strict_response_validation_fails_closed(prepared) -> None:
    records, _manifest = prepared
    record = next(row for row in records if row["arm"] == "without_critic")
    assessments = [
        ablation.VerifierAssessment(
            t_index=item["t_index"],
            dimension=item["dimension"],
            verdict="not_conflict",
            confidence="high",
            reason_code="direct_aligned_or_neutral_behavior",
            evidence_quote="",
        )
        for item in record["expected_coordinates"]
    ]
    parsed = ablation.WeeklyVerifierResponse(assessments=assessments)
    ablation.validate_parsed_response(parsed=parsed, record=record)

    with pytest.raises(ValueError, match="coordinate mismatch"):
        ablation.validate_parsed_response(
            parsed=ablation.WeeklyVerifierResponse(assessments=assessments[:-1]),
            record=record,
        )
    bad_quote = assessments.copy()
    bad_quote[0] = bad_quote[0].model_copy(
        update={"verdict": "conflict", "evidence_quote": "not in the entry"}
    )
    with pytest.raises(ValueError, match="not present"):
        ablation.validate_parsed_response(
            parsed=ablation.WeeklyVerifierResponse(assessments=bad_quote),
            record=record,
        )

    optional_quote = assessments.copy()
    optional_quote[0] = optional_quote[0].model_copy(
        update={"evidence_quote": "A concise model explanation, not a quote."}
    )
    ablation.validate_parsed_response(
        parsed=ablation.WeeklyVerifierResponse(assessments=optional_quote),
        record=record,
    )


def _all_not_conflict_responses(records: list[dict]) -> list[dict]:
    responses = []
    for repeat in range(1, CONFIG["study"]["repeats"] + 1):
        for record in records:
            responses.append(
                {
                    "status": "ok",
                    "persona_id": record["persona_id"],
                    "week_start": record["week_start"],
                    "week_end": record["week_end"],
                    "arm": record["arm"],
                    "repeat": repeat,
                    "prompt_sha256": record["prompt_sha256"],
                    "runtime_text_sha256": record["runtime_text_sha256"],
                    "requested_model": CONFIG["api"]["model"],
                    "parsed": {
                        "assessments": [
                            {
                                "t_index": item["t_index"],
                                "dimension": item["dimension"],
                                "verdict": "not_conflict",
                                "confidence": "high",
                                "reason_code": "direct_aligned_or_neutral_behavior",
                                "evidence_quote": "",
                            }
                            for item in record["expected_coordinates"]
                        ]
                    },
                }
            )
    return responses


def test_scoring_masks_unresolved_targets_and_case_037(prepared) -> None:
    records, _manifest = prepared
    metrics = ablation.score_responses(
        config=CONFIG,
        root=ROOT,
        records=records,
        responses=_all_not_conflict_responses(records),
    )

    assert metrics["resolved_entry_cells"] == 316
    assert metrics["resolved_trajectories"] == 41
    assert metrics["positive_episodes"] == 5
    assert metrics["negative_trajectories"] == 36
    assert metrics["decision"]["verdict"] == "negative"
    assert all(row["entry"]["coverage"] == 1.0 for row in metrics["results"])
    assert all(row["episode"]["recall"] == 0.0 for row in metrics["results"])


def test_cross_week_adjacent_conflicts_form_an_episode() -> None:
    case = {
        "persona_id": "persona",
        "dimension": "security",
        "entries": [
            {"t_index": 4, "date": "2026-01-11"},
            {"t_index": 5, "date": "2026-01-12"},
        ],
    }
    predictions = {
        ("with_critic", 1, "persona", t_index, "security"): ablation.VerifierAssessment(
            t_index=t_index,
            dimension="security",
            verdict="conflict",
            confidence="high",
            reason_code="direct_behavior_or_choice",
            evidence_quote="evidence",
        )
        for t_index in (4, 5)
    }

    episode, covered, confirmation = ablation._predicted_episode(
        case=case,
        arm="with_critic",
        repeat=1,
        predictions=predictions,
    )

    assert ablation._week_start("2026-01-11") != ablation._week_start("2026-01-12")
    assert episode is True
    assert covered is True
    assert confirmation == 5


def test_abstention_suppresses_recall_and_reduces_coverage() -> None:
    metrics = ablation._binary_metrics(
        [True, False, True],
        [None, False, True],
    )

    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["coverage"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("with_recall", "with_fp", "with_coverage", "expected"),
    [
        (0.4, 0.0, 1.0, "positive"),
        (0.2, 0.0, 1.0, "negative"),
        (0.4, 1.0, 1.0, "inconclusive"),
    ],
)
def test_paired_decision_rule(
    with_recall: float, with_fp: float, with_coverage: float, expected: str
) -> None:
    rows = []
    for arm, recall, fp, coverage in (
        ("without_critic", 0.2, 0.0, 1.0),
        ("with_critic", with_recall, with_fp, with_coverage),
    ):
        for repeat in (1, 2, 3):
            rows.append(
                {
                    "arm": arm,
                    "repeat": repeat,
                    "episode": {"recall": recall, "fp": fp, "coverage": coverage},
                }
            )

    assert ablation._paired_decision(rows)["verdict"] == expected


def test_paired_decision_requires_all_registered_repeats() -> None:
    rows = [
        {
            "arm": arm,
            "repeat": repeat,
            "episode": {"recall": 0.2, "fp": 0.0, "coverage": 1.0},
        }
        for arm in ablation.ARMS
        for repeat in (1, 2)
    ]

    assert ablation._paired_decision(rows)["verdict"] == "incomplete"


def test_cost_estimate_and_paid_run_guard(prepared) -> None:
    records, _manifest = prepared
    estimate = ablation.estimate_plan(records, CONFIG)

    assert estimate["successful_calls"] == 756
    assert estimate["estimated_cost_usd"] < estimate["max_budget_usd"]
    with pytest.raises(SystemExit, match="Refusing paid calls"):
        ablation.command_run(
            ablation.argparse.Namespace(
                execute=False,
                root=ROOT,
                config=ablation.DEFAULT_CONFIG_PATH,
            )
        )


def test_refusal_and_invalid_receipts_are_terminal_for_resume(prepared) -> None:
    records, _manifest = prepared
    record_map = {
        (row["persona_id"], row["week_start"], row["arm"]): row for row in records
    }
    selected = records[:2]
    receipts = [
        {
            "status": status,
            "persona_id": row["persona_id"],
            "week_start": row["week_start"],
            "arm": row["arm"],
            "repeat": 1,
            "prompt_sha256": row["prompt_sha256"],
            "runtime_text_sha256": row["runtime_text_sha256"],
            "week_end": row["week_end"],
            "requested_model": CONFIG["api"]["model"],
        }
        for status, row in zip(("refusal", "invalid"), selected, strict=True)
    ]

    completed = ablation._completed_keys(
        receipts,
        record_map,
        repeats=CONFIG["study"]["repeats"],
        requested_model=CONFIG["api"]["model"],
    )

    assert len(completed) == 2


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("runtime_text_sha256", "stale", "Stale or unknown"),
        ("week_end", "2099-01-01", "Stale or unknown"),
        ("repeat", 4, "Out-of-range"),
        ("requested_model", "wrong-model", "Unexpected requested model"),
    ],
)
def test_terminal_receipt_provenance_is_validated(
    prepared, field: str, bad_value, message: str
) -> None:
    records, _manifest = prepared
    record = records[0]
    record_map = {
        (row["persona_id"], row["week_start"], row["arm"]): row for row in records
    }
    receipt = {
        "status": "refusal",
        "persona_id": record["persona_id"],
        "week_start": record["week_start"],
        "week_end": record["week_end"],
        "arm": record["arm"],
        "repeat": 1,
        "prompt_sha256": record["prompt_sha256"],
        "runtime_text_sha256": record["runtime_text_sha256"],
        "requested_model": CONFIG["api"]["model"],
    }
    receipt[field] = bad_value

    with pytest.raises(ValueError, match=message):
        ablation._completed_keys(
            [receipt],
            record_map,
            repeats=CONFIG["study"]["repeats"],
            requested_model=CONFIG["api"]["model"],
        )


def test_saved_result_is_complete_and_reproducible() -> None:
    records, paths = ablation._load_prepared(CONFIG, ROOT)
    responses = ablation._load_jsonl(paths["responses"])
    saved = ablation._read_json(paths["metrics"])
    rescored = ablation.score_responses(
        config=CONFIG,
        root=ROOT,
        records=records,
        responses=responses,
    )

    assert len(responses) == 756
    assert all(row["requested_model"] == CONFIG["api"]["model"] for row in responses)
    assert all(row["resolved_model"] == CONFIG["api"]["model"] for row in responses)
    assert all(row["response_id"] for row in responses)
    assert all((row.get("usage") or {}).get("total_tokens", 0) > 0 for row in responses)
    assert saved["response_summary"]["statuses"] == {"invalid": 79, "ok": 677}
    assert saved["response_summary"]["recovered_prompt_contract_valid"] == 69
    assert saved["response_summary"]["effective_invalid"] == 10
    assert saved["decision"]["verdict"] == "negative"
    assert saved["decision"]["summary"]["without_critic"] == pytest.approx(
        {
            "median_episode_recall": 0.4,
            "median_false_alerts": 1.0,
            "median_coverage": 0.7560975609756098,
        }
    )
    assert saved["decision"]["summary"]["with_critic"] == pytest.approx(
        {
            "median_episode_recall": 0.2,
            "median_false_alerts": 0.0,
            "median_coverage": 0.7317073170731707,
        }
    )
    saved.pop("scored_at")
    saved.pop("artifact_provenance")
    rescored.pop("scored_at")
    assert rescored == saved
