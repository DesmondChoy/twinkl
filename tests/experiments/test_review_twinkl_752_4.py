"""Live-contract checks for the ``twinkl-752.4`` review entrypoint."""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts/experiments/review_twinkl_752_4_legacy_drift_candidates.py"
)
CONFIG_PATH = REPO_ROOT / "config/evals/twinkl_752_4_legacy_drift_review_v1.yaml"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("twinkl_752_4_review", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()


def test_live_cohort_matches_frozen_pre_review_contract():
    config, _registry, inventory, pairs, _cases, _paths = mod._load_live(CONFIG_PATH)
    selected = mod.selected_case_metadata(inventory, pairs)

    assert config["expected_counts"]["candidates"] == 52
    assert pairs.height == 52
    assert selected.height == 104
    assert selected["persona_id"].n_unique() == 103
    assert selected.filter(pl.col("analysis_role") == "retired_audit_only").height == 16


def test_prepare_writes_four_complete_blind_shards(tmp_path: Path):
    output = tmp_path / "review"
    mod.prepare(Namespace(config=CONFIG_PATH, output=output))

    manifest = mod._read_json(output / "parent_control/cohort_manifest.json")
    selected = pl.read_parquet(output / "parent_control/selected_cases.parquet")
    assert manifest["case_count"] == 104
    assert len(manifest["shards"]) == 4
    assert sum(shard["case_count"] for shard in manifest["shards"]) == 104
    assert sum(shard["entry_count"] for shard in manifest["shards"]) == int(
        selected["trajectory_length"].sum()
    )
    for shard in manifest["shards"]:
        packet_text = (REPO_ROOT / shard["packet_path"]).read_text(encoding="utf-8")
        assert "persona_id" not in packet_text
        assert "historical_split" not in packet_text
        assert "legacy_candidate" not in packet_text
