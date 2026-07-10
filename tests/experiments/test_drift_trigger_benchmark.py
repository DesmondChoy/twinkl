"""Integration-light tests for the twinkl-wq9p benchmark script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

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

    actual_counts = (
        all_reference.height,
        validation_reference.height,
        test_reference.height,
    )
    assert actual_counts == (52, 6, 5), (
        "Consensus benchmark input changed; rebuild and review the drift benchmark "
        "artifacts before accepting new counts. Actual all/validation/test: "
        f"{actual_counts}"
    )


def test_holdout_prompt_does_not_include_reference_labels():
    _fixture, cases, _profiles, _labels = mod.load_designed_holdout(mod.DEFAULT_HOLDOUT)
    row = mod._holdout_experiment_rows(cases)[0]

    prompt = render_prompt(row, context_arm="student_visible")

    assert "core_labels" not in prompt
    assert "target_vector" not in prompt
    assert '"security": -1' not in prompt
    assert row.session_content in prompt


def test_llm_record_validation_rejects_failed_duplicate_and_missing_rows():
    scores = {dimension: 0 for dimension in mod.SCHWARTZ_VALUE_ORDER}
    valid = {
        "status": "ok",
        "persona_id": "p1",
        "t_index": 0,
        "date": "2026-01-01",
        "core_values": ["Security"],
        "scores": scores,
    }

    with pytest.raises(ValueError, match="failed LLM row"):
        mod._validate_llm_records(
            [{**valid, "status": "error"}],
            arm_id="arm",
            expected_keys={("p1", 0)},
        )
    with pytest.raises(ValueError, match="duplicate LLM row keys"):
        mod._validate_llm_records(
            [valid, valid],
            arm_id="arm",
            expected_keys={("p1", 0)},
        )
    with pytest.raises(ValueError, match="row coverage mismatch"):
        mod._validate_llm_records(
            [valid],
            arm_id="arm",
            expected_keys={("p1", 0), ("p1", 1)},
        )
    with pytest.raises(ValueError, match="invalid core values"):
        mod._validate_llm_records(
            [{**valid, "core_values": ["Not A Schwartz Value"]}],
            arm_id="arm",
            expected_keys={("p1", 0)},
        )


def _result(arm_id: str, *, hits: int, reference: int) -> dict:
    recall = hits / reference
    precision = 1.0 if hits else 0.0
    return {
        "arm_id": arm_id,
        "evidence_kind": "hard_class"
        if arm_id.startswith("llm_")
        else "soft_probability",
        "overall": {
            "reference_episodes": reference,
            "predicted_episodes": hits,
            "true_positive": hits,
            "precision": precision,
            "recall": recall,
            "f1": recall,
            "false_positive_rate": 0.0,
            "max_latency_entries": 0 if hits else None,
            "recovery_accuracy": None,
        },
    }


def test_report_derives_cross_set_counts_and_requires_both_surfaces_to_pass():
    report = mod.render_report(
        fixture={
            "holdout_id": "fixture",
            "sha256": "abc123",
            "review_status": "designed_not_human_reviewed",
        },
        frozen_results=[
            _result("run_020_selected", hits=1, reference=7),
            _result("run_052_consensus", hits=2, reference=7),
            _result("llm_custom", hits=2, reference=7),
        ],
        holdout_results=[
            _result("run_020_selected", hits=3, reference=12),
            _result("run_052_consensus", hits=4, reference=12),
            _result("llm_custom", hits=10, reference=12),
        ],
    )

    assert "Designed LLM episode hits: `llm_custom` 10/12" in report
    assert "Frozen LLM episode hits: `llm_custom` 2/7" in report
    assert "that split has only 7 strict episodes" in report
    assert "Designed incumbent MLP episode hits: `run_020_selected` 3/12" in report
    assert (
        "Designed consensus-trained MLP episode hits: `run_052_consensus` 4/12"
        in report
    )
    assert "No scorer is promotion-ready" in report
