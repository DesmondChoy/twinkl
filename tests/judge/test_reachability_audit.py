"""Tests for the twinkl-747 judge reachability audit scripts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl

from src.judge.labeling import load_schwartz_values
from src.models.judge import SCHWARTZ_VALUE_ORDER

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script_module(filename: str, module_name: str):
    script_path = REPO_ROOT / "scripts" / "journalling" / filename
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_mod = _load_script_module(
    "twinkl_747_prepare_audit.py", "twinkl_747_prepare_audit_test"
)
summarize_mod = _load_script_module(
    "twinkl_747_summarize_audit.py", "twinkl_747_summarize_audit_test"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _score_payload(dimension: str, value: int) -> dict[str, int]:
    payload = {name: 0 for name in SCHWARTZ_VALUE_ORDER}
    payload[dimension] = value
    return payload


def test_prepare_bundle_writes_expected_counts_and_is_deterministic(tmp_path):
    cases, _entries = prepare_mod._build_case_universe(
        labels_path=REPO_ROOT / "logs" / "judge_labels" / "judge_labels.parquet",
        wrangled_dir=REPO_ROOT / "logs" / "wrangled",
        annotations_dir=REPO_ROOT / "logs" / "annotations",
        runs_dir=REPO_ROOT / "logs" / "experiments" / "runs",
    )

    manifest_a = prepare_mod.select_audit_manifest(cases)
    manifest_b = prepare_mod.select_audit_manifest(cases)

    assert manifest_a["case_id"].to_list() == manifest_b["case_id"].to_list()
    assert manifest_a.height == 50
    assert (
        manifest_a.group_by("dimension").len().sort("dimension").to_dicts()
        == [
            {"dimension": "hedonism", "len": 14},
            {"dimension": "security", "len": 14},
            {"dimension": "self_direction", "len": 4},
            {"dimension": "stimulation", "len": 14},
            {"dimension": "universalism", "len": 4},
        ]
    )

    bundle_dir = tmp_path / "bundle"
    written_manifest = prepare_mod.prepare_audit_bundle(
        output_dir=bundle_dir,
        labels_path=REPO_ROOT / "logs" / "judge_labels" / "judge_labels.parquet",
        wrangled_dir=REPO_ROOT / "logs" / "wrangled",
        annotations_dir=REPO_ROOT / "logs" / "annotations",
        runs_dir=REPO_ROOT / "logs" / "experiments" / "runs",
        schwartz_path=REPO_ROOT / "config" / "schwartz_values.yaml",
    )

    assert written_manifest.height == 50
    assert (bundle_dir / "sample_manifest.csv").exists()
    assert (bundle_dir / "prompts" / "full_context.jsonl").exists()
    assert (bundle_dir / "prompts" / "profile_only.jsonl").exists()
    assert (bundle_dir / "prompts" / "student_visible.jsonl").exists()
    assert (bundle_dir / "manual_review_workbook.csv").exists()
    assert (bundle_dir / "manual_review_blind_packet.md").exists()
    assert (bundle_dir / "manual_review_reference.md").exists()


def test_prompt_rendering_strips_bio_core_values_and_history_as_expected():
    cases, entries = prepare_mod._build_case_universe(
        labels_path=REPO_ROOT / "logs" / "judge_labels" / "judge_labels.parquet",
        wrangled_dir=REPO_ROOT / "logs" / "wrangled",
        annotations_dir=REPO_ROOT / "logs" / "annotations",
        runs_dir=REPO_ROOT / "logs" / "experiments" / "runs",
    )
    manifest = prepare_mod.select_audit_manifest(cases)
    candidate = (
        manifest.filter(pl.col("t_index") > 0)
        .sort(["dimension", "selection_rank_within_dimension", "case_id"])
        .to_dicts()[0]
    )

    entry_lookup, entries_by_persona = prepare_mod._build_entry_maps(entries)
    entry = entry_lookup[(candidate["persona_id"], int(candidate["t_index"]))]
    schwartz_config = load_schwartz_values(REPO_ROOT / "config" / "schwartz_values.yaml")

    full = prepare_mod._render_prompt_record(
        candidate,
        condition="full_context",
        entry_lookup=entry_lookup,
        entries_by_persona=entries_by_persona,
        schwartz_config=schwartz_config,
    )
    profile_only = prepare_mod._render_prompt_record(
        candidate,
        condition="profile_only",
        entry_lookup=entry_lookup,
        entries_by_persona=entries_by_persona,
        schwartz_config=schwartz_config,
    )
    student_visible = prepare_mod._render_prompt_record(
        candidate,
        condition="student_visible",
        entry_lookup=entry_lookup,
        entries_by_persona=entries_by_persona,
        schwartz_config=schwartz_config,
    )

    core_values_line = (
        "- **Core Values (from profile):** "
        + ", ".join(entry["persona_core_values"] or [])
    )
    bio_line = f"- **Bio:** {entry['persona_bio'] or ''}"

    assert "## Recent Entries (for context)" in full["prompt"]
    assert "## Recent Entries (for context)" not in profile_only["prompt"]
    assert "## Recent Entries (for context)" not in student_visible["prompt"]

    assert core_values_line in full["prompt"]
    assert core_values_line in profile_only["prompt"]
    assert core_values_line not in student_visible["prompt"]

    assert bio_line in full["prompt"]
    assert bio_line not in profile_only["prompt"]
    assert bio_line not in student_visible["prompt"]


def test_summarizer_computes_flip_counts_and_recommendation_bands(tmp_path):
    bundle_dir = tmp_path / "audit_bundle"
    (bundle_dir / "results").mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for dimension in ("security", "hedonism", "stimulation"):
        for idx in range(14):
            manifest_rows.append(
                {
                    "case_id": f"{dimension}__persona_{idx}__0",
                    "dimension": dimension,
                    "persisted_label": 1,
                }
            )
    manifest = pl.DataFrame(manifest_rows)
    manifest.write_csv(bundle_dir / "sample_manifest.csv")

    full_rows: list[dict] = []
    profile_rows: list[dict] = []
    student_rows: list[dict] = []
    workbook_rows: list[dict] = []

    for row in manifest.to_dicts():
        dimension = row["dimension"]
        case_id = row["case_id"]
        index = int(case_id.split("__")[1].split("_")[-1])

        full_value = 1
        if dimension == "security":
            profile_value = 0 if index < 4 else 1
        elif dimension == "hedonism":
            profile_value = 0 if index < 2 else 1
        else:
            profile_value = 1

        student_value = profile_value

        full_rows.append({"case_id": case_id, "scores": _score_payload(dimension, full_value)})
        profile_rows.append(
            {"case_id": case_id, "scores": _score_payload(dimension, profile_value)}
        )
        student_rows.append(
            {"case_id": case_id, "scores": _score_payload(dimension, student_value)}
        )

        workbook_rows.append(
            {
                "case_id": case_id,
                "dimension": dimension,
                "persona_id": "persona",
                "t_index": 0,
                "date": "2025-01-01",
                "selection_reason": "fixture",
                "persisted_label": 1,
                "human_majority_label": "",
                "blind_review_priority": "true" if case_id == "hedonism__persona_0__0" else "false",
                "blind_review_rank": "1" if case_id == "hedonism__persona_0__0" else "",
                "reviewer_id": "tester" if case_id == "hedonism__persona_0__0" else "",
                "blind_text_label": "0" if case_id == "hedonism__persona_0__0" else "",
                "blind_text_notes": "",
                "rich_context_label": "1" if case_id == "hedonism__persona_0__0" else "",
                "rich_context_notes": "",
                "label_changed_after_context": "true" if case_id == "hedonism__persona_0__0" else "",
                "preferred_target": "reduced_context" if case_id == "hedonism__persona_0__0" else "",
                "recommendation_notes": "",
            }
        )

    _write_jsonl(bundle_dir / "results" / "full_context_results.jsonl", full_rows)
    _write_jsonl(bundle_dir / "results" / "profile_only_results.jsonl", profile_rows)
    _write_jsonl(bundle_dir / "results" / "student_visible_results.jsonl", student_rows)
    pl.DataFrame(workbook_rows).write_csv(bundle_dir / "manual_review_workbook.csv")

    report, results, comparisons, flip_summary = summarize_mod.summarize_bundle(bundle_dir)

    security_row = flip_summary.filter(
        (pl.col("comparison") == "full_context_vs_profile_only")
        & (pl.col("dimension") == "security")
        & (pl.col("source_label") == 1)
    ).to_dicts()[0]
    hedonism_row = flip_summary.filter(
        (pl.col("comparison") == "full_context_vs_profile_only")
        & (pl.col("dimension") == "hedonism")
        & (pl.col("source_label") == 1)
    ).to_dicts()[0]
    stimulation_row = flip_summary.filter(
        (pl.col("comparison") == "full_context_vs_profile_only")
        & (pl.col("dimension") == "stimulation")
        & (pl.col("source_label") == 1)
    ).to_dicts()[0]

    assert results.height == 42
    assert comparisons.height == 126
    assert security_row["flip_count"] == 4
    assert security_row["band"] == "substantive"
    assert hedonism_row["flip_count"] == 2
    assert hedonism_row["band"] == "ambiguous"
    assert stimulation_row["flip_count"] == 0
    assert stimulation_row["band"] == "low"
    assert "Overall recommendation: `change_distillation_target`" in report
    assert "Reviewed manual rows: `1`" in report
