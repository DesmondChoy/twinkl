"""Integration-light checks for the active v8pb evaluation entrypoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest

from src.vif.dataset import load_entries
from src.vif.drift_target import validate_target_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/experiments/evaluate_v8pb_student_visible_target.py"
MANIFEST_PATH = REPO_ROOT / "config/evals/drift_v1_student_visible_v1.yaml"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("v8pb_target_evaluation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()


def test_run_020_is_the_only_allowed_fresh_population_checkpoint():
    manifest = mod._read_yaml(MANIFEST_PATH)
    run, run_path, checkpoint, validation_outputs = mod._load_run_020(manifest)

    assert run["metadata"]["run_id"] == "run_020"
    assert sum(run["data"].values()) == 1460
    assert run_path.is_file()
    assert checkpoint.is_file()
    assert validation_outputs.is_file()


def test_fresh_promotion_cases_are_derived_from_the_original_manifest_gap():
    registry = pl.read_parquet(REPO_ROOT / "logs/registry/personas.parquet")
    entries = load_entries(REPO_ROOT / "logs/wrangled")
    _manifest, split = validate_target_manifest(MANIFEST_PATH, registry, entries)
    cases = mod._model_cases(entries, registry, split.promotion_persona_ids)

    assert len(cases) == 24
    assert sum(len(case["entries"]) for case in cases) == 191
    assert {"initial_entry", "nudge_text", "response_text"} <= set(
        cases[0]["entries"][0]
    )
    assert any(
        entry["nudge_text"] or entry["response_text"]
        for case in cases
        for entry in case["entries"]
    )
    assert {case["persona_id"] for case in cases} == set(split.promotion_persona_ids)
    assert set(split.promotion_persona_ids).isdisjoint(split.training_persona_ids)
    assert set(split.promotion_persona_ids).isdisjoint(split.development_persona_ids)
    assert set(split.promotion_persona_ids).isdisjoint(split.retired_persona_ids)


def test_threshold_must_precede_the_first_promotion_submission():
    promotion_summary = {
        "reviewer_submission_timestamps": [
            "2026-07-11T09:00:00+08:00",
            "2026-07-11T10:00:00+08:00",
        ]
    }

    with pytest.raises(ValueError, match="first promotion-review"):
        mod._validate_threshold_timing(
            {"created_at": "2026-07-11T01:30:00+00:00"},
            promotion_summary,
        )

    mod._validate_threshold_timing(
        {"created_at": "2026-07-11T00:30:00+00:00"},
        promotion_summary,
    )


def test_threshold_timing_rejects_naive_timestamps():
    with pytest.raises(ValueError, match="timezone"):
        mod._validate_threshold_timing(
            {"created_at": "2026-07-11T00:30:00"},
            {
                "reviewer_submission_timestamps": [
                    "2026-07-11T09:00:00+08:00",
                    "2026-07-11T10:00:00+08:00",
                ]
            },
        )
