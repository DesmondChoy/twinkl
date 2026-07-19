"""Protocol checks for the ``twinkl-52zz`` model comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from prompts import get_prompt_metadata
from scripts.experiments import compare_twinkl_52zz_models as study
from scripts.experiments import weekly_verifier_ablation as baseline

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config/evals/twinkl_52zz_model_comparison_v1.yaml"


def _config() -> dict:
    return baseline._read_yaml(CONFIG_PATH)


def test_complete_development_population_and_prompt_counts() -> None:
    config = _config()
    records, cases, outcomes, targets, episodes = study._build_prompt_records(
        config, ROOT
    )
    counts = study._observed_counts(
        records=records,
        cases=cases,
        outcomes=outcomes,
        targets=targets,
        episodes=episodes,
    )

    assert counts == {
        "personas": 204,
        "cases": 292,
        "entries": 1651,
        "entry_value_cells": 2377,
        "resolved_entry_value_cells": 2375,
        "drifts": 42,
        "drift_trajectories": 36,
        "persona_weeks": 951,
    }
    assert all(record["arm"] == study.WEEKLY_WITHOUT for record in records)
    assert all("CRITIC SIGNALS" not in record["prompt"] for record in records)
    assert all("critic_block_sha256" not in record for record in records)


def test_canonical_prompt_metadata_rejects_vif_critic_input() -> None:
    metadata = get_prompt_metadata("weekly_vif_verifier")

    assert metadata["version"] == "2.0"
    assert metadata["input_variables"] == [
        "declared_values",
        "cumulative_history",
        "current_week_entries",
    ]


def test_models_change_only_api_identity_and_share_one_prompt_file() -> None:
    config = _config()
    specs = study._model_specs(config)
    assert set(specs) == {"gpt_5_4_mini", "gpt_5_6_luna"}
    assert {spec["reasoning_effort"] for spec in specs.values()} == {"none"}
    assert specs["gpt_5_4_mini"]["model"] == "gpt-5.4-mini-2026-03-17"
    assert specs["gpt_5_6_luna"]["model"] == "gpt-5.6-luna"
    assert config["api"]["store"] is False
    assert sum(float(spec["max_budget_usd"]) for spec in specs.values()) == 15.0


def test_incomplete_model_responses_fail_closed() -> None:
    config = _config()
    records, cases, outcomes, targets, episodes = study._build_prompt_records(
        config, ROOT
    )
    with pytest.raises(ValueError, match="Response set is incomplete"):
        study._score_model(
            config=config,
            model_spec=study._model_specs(config)["gpt_5_6_luna"],
            records=records,
            cases=cases,
            outcomes=outcomes,
            targets=targets,
            episodes=episodes,
            responses=[],
        )
