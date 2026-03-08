"""Tests for targeted synthetic-batch verification helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.synthetic.batch_verification import (
    load_baseline_persona_ids,
    render_spot_check_report,
    summarize_raw_persona_file,
    verify_targeted_batch,
)


RAW_PERSONA_TEMPLATE = """\
# Persona {persona_id}: Test Persona

## Profile
- **Persona ID:** {persona_id}
- **Generated:** 2026-03-08_10-00-00
- Age: 25-34
- Profession: Teacher
- Culture: Western European
- Core Values: {core_values}
- Bio: A short persona bio.

## Entry 1 - 2025-03-01

### Initial Entry
**Tone**: Self-reflective | **Verbosity**: Medium (1-2 paragraphs) | **Reflection Mode**: Unsettled

Something small but annoying happened today and I let it slide even though it made me uneasy.

### Nudge (tension_surfacing)
**Trigger**: Mild unresolved tension.
"What sat wrong with you?"

### Response
**Mode**: Answering directly

It was not catastrophic, just the kind of thing that chips away at my sense that things are under control.

---

## Entry 2 - 2025-03-04

### Initial Entry
**Tone**: Brief and factual | **Verbosity**: Short (1-3 sentences) | **Reflection Mode**: Neutral

Routine day. Did laundry and answered email.

---
"""

PLAIN_METADATA_PERSONA_TEMPLATE = """\
# Persona {persona_id}: Test Persona

## Profile

Persona ID: **{persona_id}**
Generated: **2026-03-08_10-00-00**
Age: 25-34
Profession: Teacher
Culture: Western European
Core Values: {core_values}

A short persona bio.

## Entry 1 - 2025-03-01

### Initial Entry
Tone: Self-reflective
Verbosity: Medium (1-2 paragraphs)
Mode: Unsettled

Something small but annoying happened today and I let it slide even though it made me uneasy.

### Nudge (tension_surfacing)
**Trigger**: Mild unresolved tension.
"What sat wrong with you?"

### Response

It was not catastrophic, just the kind of thing that chips away at my sense that things are under control.

---
"""


def test_load_baseline_persona_ids(tmp_path: Path):
    snapshot_path = tmp_path / "snapshot.yaml"
    snapshot_path.write_text(
        "registry_persona_ids:\n- aaa11111\n- bbb22222\n",
        encoding="utf-8",
    )

    assert load_baseline_persona_ids(snapshot_path) == {"aaa11111", "bbb22222"}


def test_summarize_raw_persona_file_extracts_unsettled_entry(tmp_path: Path):
    raw_path = tmp_path / "persona_abc12345.md"
    raw_path.write_text(
        RAW_PERSONA_TEMPLATE.format(
            persona_id="abc12345",
            core_values="Power, Security",
        ),
        encoding="utf-8",
    )

    summary = summarize_raw_persona_file(raw_path)

    assert summary["persona_id"] == "abc12345"
    assert summary["entry_count"] == 2
    assert summary["unsettled_entry_count"] == 1
    assert summary["first_unsettled_entry"]["t_index"] == 0
    assert "Something small but annoying happened today" in summary["first_unsettled_entry"][
        "initial_entry_excerpt"
    ]


def test_summarize_raw_persona_file_supports_plain_metadata_variant(tmp_path: Path):
    raw_path = tmp_path / "persona_abc12345.md"
    raw_path.write_text(
        PLAIN_METADATA_PERSONA_TEMPLATE.format(
            persona_id="abc12345",
            core_values="Power",
        ),
        encoding="utf-8",
    )

    summary = summarize_raw_persona_file(raw_path)

    assert summary["core_values"] == ["Power"]
    assert summary["unsettled_entry_count"] == 1
    assert summary["first_unsettled_entry"]["t_index"] == 0


def test_verify_targeted_batch_accepts_expected_batch(tmp_path: Path):
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()

    baseline_ids = {"0aa00001", "0bb00002"}
    new_personas = [
        ("0cc00001", "Power"),
        ("0dd00002", "Power, Achievement"),
        ("0ee00003", "Security"),
        ("0ff00004", "Security, Tradition"),
    ]

    for persona_id, core_values in new_personas:
        (synthetic_dir / f"persona_{persona_id}.md").write_text(
            RAW_PERSONA_TEMPLATE.format(
                persona_id=persona_id,
                core_values=core_values,
            ),
            encoding="utf-8",
        )

    registry = pl.DataFrame(
        {
            "persona_id": ["0aa00001", "0bb00002", "0cc00001", "0dd00002", "0ee00003", "0ff00004"],
            "name": ["Old 1", "Old 2", "New 1", "New 2", "New 3", "New 4"],
            "age": ["25-34"] * 6,
            "profession": ["Teacher"] * 6,
            "culture": ["Western European"] * 6,
            "core_values": [["Power"]] * 6,
            "entry_count": [2] * 6,
            "created_at": [None] * 6,
            "stage_synthetic": [True] * 6,
            "stage_wrangled": [False] * 6,
            "stage_labeled": [False] * 6,
            "nudge_enabled": [True] * 6,
            "annotation_order": [None] * 6,
        }
    )
    registry_path = tmp_path / "personas.parquet"
    registry.write_parquet(registry_path)

    summary = verify_targeted_batch(
        baseline_persona_ids=baseline_ids,
        registry_path=registry_path,
        synthetic_dir=synthetic_dir,
        required_targets=["Power", "Security"],
        expected_new_persona_count=4,
        expected_min_personas_per_target=2,
        min_entries=2,
        max_entries=4,
    )

    assert summary["accepted"] is True
    assert summary["target_counts"] == {"Power": 2, "Security": 2}
    assert summary["new_persona_ids"] == ["0cc00001", "0dd00002", "0ee00003", "0ff00004"]
    assert "Accepted: yes" in render_spot_check_report(summary)


def test_verify_targeted_batch_reports_missing_unsettled_entries(tmp_path: Path):
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    raw_path = synthetic_dir / "persona_0cc00001.md"
    raw_path.write_text(
        RAW_PERSONA_TEMPLATE.replace("Reflection Mode**: Unsettled", "Reflection Mode**: Neutral").format(
            persona_id="0cc00001",
            core_values="Power",
        ),
        encoding="utf-8",
    )

    registry = pl.DataFrame(
        {
            "persona_id": ["0aa00001", "0cc00001"],
            "name": ["Base", "New"],
            "age": ["25-34", "25-34"],
            "profession": ["Teacher", "Teacher"],
            "culture": ["Western European", "Western European"],
            "core_values": [["Power"], ["Power"]],
            "entry_count": [2, 2],
            "created_at": [None, None],
            "stage_synthetic": [True, True],
            "stage_wrangled": [False, False],
            "stage_labeled": [False, False],
            "nudge_enabled": [True, True],
            "annotation_order": [None, None],
        }
    )
    registry_path = tmp_path / "personas.parquet"
    registry.write_parquet(registry_path)

    summary = verify_targeted_batch(
        baseline_persona_ids={"0aa00001"},
        registry_path=registry_path,
        synthetic_dir=synthetic_dir,
        required_targets=["Power", "Security"],
        expected_new_persona_count=1,
        expected_min_personas_per_target=1,
        min_entries=2,
        max_entries=4,
    )

    assert summary["accepted"] is False
    assert any("no Unsettled entries" in failure for failure in summary["failures"])


def test_verify_targeted_batch_requires_stage_synthetic_flag(tmp_path: Path):
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    (synthetic_dir / "persona_0cc00001.md").write_text(
        RAW_PERSONA_TEMPLATE.format(
            persona_id="0cc00001",
            core_values="Power",
        ),
        encoding="utf-8",
    )

    registry = pl.DataFrame(
        {
            "persona_id": ["0aa00001", "0cc00001"],
            "name": ["Base", "New"],
            "age": ["25-34", "25-34"],
            "profession": ["Teacher", "Teacher"],
            "culture": ["Western European", "Western European"],
            "core_values": [["Power"], ["Power"]],
            "entry_count": [2, 2],
            "created_at": [None, None],
            "stage_synthetic": [True, False],
            "stage_wrangled": [False, False],
            "stage_labeled": [False, False],
            "nudge_enabled": [True, True],
            "annotation_order": [None, None],
        }
    )
    registry_path = tmp_path / "personas.parquet"
    registry.write_parquet(registry_path)

    summary = verify_targeted_batch(
        baseline_persona_ids={"0aa00001"},
        registry_path=registry_path,
        synthetic_dir=synthetic_dir,
        required_targets=["Power"],
        expected_new_persona_count=1,
        expected_min_personas_per_target=1,
        min_entries=2,
        max_entries=4,
    )

    assert summary["accepted"] is False
    assert any("stage_synthetic=true" in failure for failure in summary["failures"])
