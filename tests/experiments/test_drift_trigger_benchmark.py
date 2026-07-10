"""Integration-light tests for the twinkl-wq9p benchmark script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl

from scripts.experiments.llm_critic_baseline import render_prompt
from src.vif.drift_benchmark import build_reference_episodes
from src.vif.holdout import load_fixed_holdout_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiments" / "drift_trigger_benchmark.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "drift_trigger_benchmark_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()


def test_designed_holdout_is_locked_balanced_and_isolated():
    fixture, cases, profiles, labels = mod.load_designed_holdout(mod.DEFAULT_HOLDOUT)
    reference = build_reference_episodes(labels, profiles, source="designed_holdout")
    registry = pl.read_parquet(REPO_ROOT / mod.DEFAULT_REGISTRY)

    assert fixture["locked_before_scoring"] is True
    assert fixture["review_status"] == "designed_not_human_reviewed"
    assert len(cases) == 20
    assert reference.height == 10
    assert reference["dimension"].n_unique() == 10
    assert set(profiles["persona_id"].to_list()).isdisjoint(
        registry["persona_id"].to_list()
    )


def test_live_consensus_reference_counts_match_benchmark_manifest_contract():
    consensus = pl.read_parquet(REPO_ROOT / mod.DEFAULT_CONSENSUS)
    registry = pl.read_parquet(REPO_ROOT / mod.DEFAULT_REGISTRY)
    validation_ids, test_ids = load_fixed_holdout_ids(
        REPO_ROOT / mod.DEFAULT_HOLDOUT_MANIFEST
    )

    all_reference = build_reference_episodes(consensus, registry)
    validation_reference = build_reference_episodes(
        consensus.filter(pl.col("persona_id").is_in(validation_ids)), registry
    )
    test_reference = build_reference_episodes(
        consensus.filter(pl.col("persona_id").is_in(test_ids)), registry
    )

    assert all_reference.height == 52
    assert validation_reference.height == 6
    assert test_reference.height == 5


def test_holdout_prompt_does_not_include_reference_labels():
    _fixture, cases, _profiles, _labels = mod.load_designed_holdout(mod.DEFAULT_HOLDOUT)
    row = mod._holdout_experiment_rows(cases)[0]

    prompt = render_prompt(row, context_arm="student_visible")

    assert "core_labels" not in prompt
    assert "target_vector" not in prompt
    assert '"security": -1' not in prompt
    assert row.session_content in prompt
