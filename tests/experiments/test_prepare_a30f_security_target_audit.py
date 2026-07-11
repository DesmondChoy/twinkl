"""Tests for the receipt-bound twinkl-a30f active-state prompt bundle."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

from src.judge.labeling import load_schwartz_values
from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.security_target import (
    ACTIVE_CRITIC_STATE_CONTRACT_VERSION,
    EXPECTED_CONTEXT_FLAGS,
)
from src.vif.state_encoder import concatenate_entry_text

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "experiments" / "prepare_a30f_security_target_audit.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_a30f_security_target_audit_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()


def test_active_state_manifest_matches_runtime_state_and_excludes_hidden_context():
    initial_entry = "I chose the predictable option after checking the budget."
    nudge = "What made that feel right?"
    response = "I need the next few months to be stable."
    manifest = mod.build_active_state_manifest(
        joined_results=_joined_results(),
        entries=_entries(initial_entry, nudge, response),
        schwartz_config=load_schwartz_values(
            REPO_ROOT / "config" / "schwartz_values.yaml"
        ),
    )

    assert len(manifest) == 1
    record = manifest[0]
    expected_session = concatenate_entry_text(initial_entry, nudge, response)
    assert record["state_contract_version"] == ACTIVE_CRITIC_STATE_CONTRACT_VERSION
    assert record["context_flags"] == EXPECTED_CONTEXT_FLAGS
    assert record["state_input"]["window_size"] == 1
    assert record["state_input"]["session_content"] == expected_session
    assert record["state_input"]["profile_weights"] == {
        dimension: 0.5 if dimension in {"security", "benevolence"} else 0.0
        for dimension in SCHWARTZ_VALUE_ORDER
    }
    assert expected_session in record["prompt"]
    assert "- security: 0.500000" in record["prompt"]
    for forbidden in (
        "## Persona Context",
        "- **Name:**",
        "- **Age:**",
        "- **Profession:**",
        "- **Culture:**",
        "- **Bio:**",
        "## Recent Entries",
        "**Date:**",
    ):
        assert forbidden not in record["prompt"]


def test_bundle_writer_creates_empty_result_template_without_overwriting(tmp_path):
    output_dir = tmp_path / "active-state"
    manifest = mod.build_active_state_manifest(
        joined_results=_joined_results(),
        entries=_entries("Entry.", None, None),
        schwartz_config=load_schwartz_values(
            REPO_ROOT / "config" / "schwartz_values.yaml"
        ),
    )

    manifest_path, results_path = mod.write_active_state_bundle(
        output_dir=output_dir,
        manifest=manifest,
    )

    written = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    assert written == manifest
    assert results_path.read_text() == ""
    readme = (output_dir / "README.md").read_text()
    assert "not a training or evaluation target" in readme


def test_manifest_rejects_duplicate_entry_coordinates_even_if_counts_cancel():
    entries = pl.concat(
        [
            _entries("First entry.", None, None),
            _entries("Duplicate entry.", None, None),
        ]
    )
    joined_results = pl.concat(
        [
            _joined_results(),
            _joined_results().with_columns(
                pl.lit("security__missing__2").alias("case_id"),
                pl.lit("missing").alias("persona_id"),
                pl.lit(2).alias("t_index"),
            ),
        ],
        how="vertical_relaxed",
    )

    with pytest.raises(ValueError, match="duplicate entry coordinates"):
        mod.build_active_state_manifest(
            joined_results=joined_results,
            entries=entries,
            schwartz_config=load_schwartz_values(
                REPO_ROOT / "config" / "schwartz_values.yaml"
            ),
        )


def _joined_results() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": ["security__example__1"],
            "dimension": ["security"],
            "persona_id": ["example"],
            "t_index": [1],
            "date": ["2026-01-01"],
            "persisted_label": [1],
            "student_visible_label": [0],
            "profile_only_label": [0],
            "full_context_label": [0],
        }
    )


def _entries(initial_entry: str, nudge_text: str | None, response_text: str | None):
    return pl.DataFrame(
        {
            "persona_id": ["example"],
            "t_index": [1],
            "date": ["2026-01-01"],
            "initial_entry": [initial_entry],
            "nudge_text": [nudge_text],
            "response_text": [response_text],
            "core_values": [["Security", "Benevolence"]],
        }
    )
