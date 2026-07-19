"""Protocol checks for the ``twinkl-52zz`` Luna reasoning comparison."""

from __future__ import annotations

from pathlib import Path

from scripts.experiments import compare_twinkl_52zz_luna_reasoning as study
from scripts.experiments import weekly_verifier_ablation as baseline

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/evals/twinkl_52zz_luna_low_v1.yaml"


def _config() -> dict:
    return baseline._read_yaml(CONFIG_PATH)


def _comparison(
    *,
    recall: float,
    recall_interval: list[float],
    false_alerts: float,
    false_alerts_interval: list[float],
    coverage: float,
) -> dict:
    return {
        "deltas": {
            "recall": {
                "observed_median": recall,
                "interval": recall_interval,
            },
            "false_alerts": {
                "observed_median": false_alerts,
                "interval": false_alerts_interval,
            },
            "coverage": {"observed_median": coverage, "interval": [-0.1, 0.1]},
        }
    }


def test_frozen_prompts_cover_all_personas_and_repeats() -> None:
    config = _config()
    paths = study._baseline_paths(config, ROOT)
    study._validate_hashes(config, paths)
    base_config = baseline._read_yaml(paths["config"])
    records = baseline._load_jsonl(paths["prompts"])
    cases = study.model_study._load_complete_development(base_config, ROOT)[0]
    none_responses = baseline._load_jsonl(paths["luna_none_responses"])

    assert len({case["persona_id"] for case in cases}) == 204
    assert len(records) == 951
    assert len(records) * config["study"]["repeats"] == 2853
    assert len(none_responses) == 2853
    assert baseline._sha256_file(paths["prompts"]) == (
        "f0c7e68b5906c3ceeaf27dfc5d5b305252ee2298d688193363d79f6ac370c539"
    )


def test_smoke_selection_is_deterministic_and_spans_prompt_lengths() -> None:
    config = _config()
    paths = study._baseline_paths(config, ROOT)
    study._validate_hashes(config, paths)
    records = baseline._load_jsonl(paths["prompts"])
    first = study._smoke_records(records, 24)
    second = study._smoke_records(records, 24)

    assert [row["prompt_sha256"] for row in first] == [
        row["prompt_sha256"] for row in second
    ]
    assert len({row["prompt_sha256"] for row in first}) == 24
    assert len(first[0]["prompt"]) == min(len(row["prompt"]) for row in records)
    assert len(first[-1]["prompt"]) == max(len(row["prompt"]) for row in records)


def test_usage_payload_preserves_reasoning_and_cache_details() -> None:
    class Usage:
        def model_dump(self, *, mode: str) -> dict:
            assert mode == "json"
            return {
                "input_tokens": 100,
                "input_tokens_details": {
                    "cached_tokens": 80,
                    "cache_write_tokens": 10,
                },
                "output_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 20},
                "total_tokens": 130,
            }

    class Response:
        usage = Usage()

    assert study._usage_payload(Response()) == {
        "input_tokens": 100,
        "input_tokens_details": {
            "cached_tokens": 80,
            "cache_write_tokens": 10,
        },
        "output_tokens": 30,
        "output_tokens_details": {"reasoning_tokens": 20},
        "total_tokens": 130,
    }


def test_cache_aware_cost_partitions_input_tokens() -> None:
    assert (
        study._cache_aware_cost_usd(
            {
                "input_tokens": 100,
                "cached_input_tokens": 80,
                "cache_write_tokens": 10,
                "output_tokens": 30,
            },
            {"input": 1.0, "cached_input": 0.1, "cache_write": 1.25, "output": 6.0},
        )
        == 0.0002105
    )


def test_decision_selects_only_a_paired_pareto_improvement() -> None:
    assert (
        study._decision(
            _comparison(
                recall=0.1,
                recall_interval=[0.01, 0.2],
                false_alerts=0,
                false_alerts_interval=[-2, 2],
                coverage=-0.01,
            )
        )
        == "select_luna_low"
    )
    assert (
        study._decision(
            _comparison(
                recall=0,
                recall_interval=[-0.1, 0.1],
                false_alerts=-3,
                false_alerts_interval=[-5, -1],
                coverage=0,
            )
        )
        == "select_luna_low"
    )
    assert (
        study._decision(
            _comparison(
                recall=0.1,
                recall_interval=[-0.01, 0.2],
                false_alerts=1,
                false_alerts_interval=[-2, 3],
                coverage=-0.06,
            )
        )
        == "keep_luna_none"
    )
