#!/usr/bin/env python3
"""Prepare the one-off judge reachability audit bundle for twinkl-747.

This script builds a deterministic 50-case sample from the active
`run_019`-`run_021` BalancedSoftmax frontier, renders judge prompt bundles for
the three rerun conditions, and writes manual-review materials for the same
cases.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation_tool.data_loader import load_entries
from src.judge.labeling import (
    build_session_content,
    load_schwartz_values,
    render_judge_prompt,
)

FRONTIER_RUN_IDS = ("run_019", "run_020", "run_021")
HARD_DIMENSIONS = ("security", "hedonism", "stimulation")
CONTROL_DIMENSIONS = ("self_direction", "universalism")
SAMPLE_DIMENSIONS = HARD_DIMENSIONS + CONTROL_DIMENSIONS
BUCKET_SIZES = {
    "security": 14,
    "hedonism": 14,
    "stimulation": 14,
    "self_direction": 4,
    "universalism": 4,
}
PROMPT_CONDITIONS = ("full_context", "profile_only", "student_visible")


def _case_id(dimension: str, persona_id: str, t_index: int) -> str:
    return f"{dimension}__{persona_id}__{t_index}"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_frontier_outputs(runs_dir: Path) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []

    for run_id in FRONTIER_RUN_IDS:
        run_path = runs_dir / f"{run_id}_BalancedSoftmax.yaml"
        payload = yaml.safe_load(run_path.read_text(encoding="utf-8"))
        artifact_path = Path(payload["artifacts"]["test_outputs"])
        frame = pl.read_parquet(artifact_path).with_columns(
            pl.lit(run_id).alias("run_id")
        )
        frames.append(frame)

    return pl.concat(frames, how="vertical")


def _load_human_majorities(annotations_dir: Path) -> pl.DataFrame:
    parquet_files = sorted(annotations_dir.glob("*.parquet"))
    if not parquet_files:
        return pl.DataFrame(
            {
                "persona_id": [],
                "t_index": [],
                "dimension": [],
                "human_majority_label": [],
                "annotator_count": [],
            },
            schema={
                "persona_id": pl.Utf8,
                "t_index": pl.Int64,
                "dimension": pl.Utf8,
                "human_majority_label": pl.Int64,
                "annotator_count": pl.Int64,
            },
        )

    annotations = pl.concat([pl.read_parquet(path) for path in parquet_files], how="vertical")
    expected_annotators = len(parquet_files)

    grouped = annotations.group_by(["persona_id", "t_index"]).agg(
        [pl.len().alias("annotator_count")]
        + [
            pl.col(f"alignment_{dimension}")
            .drop_nulls()
            .alias(f"votes_{dimension}")
            for dimension in SAMPLE_DIMENSIONS
        ]
    )
    shared = grouped.filter(pl.col("annotator_count") == expected_annotators)

    majority_columns = shared.select(
        ["persona_id", "t_index", "annotator_count"]
        + [
            pl.col(f"votes_{dimension}")
            .list.sort()
            .list.get(expected_annotators // 2)
            .alias(f"human_majority_{dimension}")
            for dimension in SAMPLE_DIMENSIONS
        ]
    )

    return (
        majority_columns.unpivot(
            index=["persona_id", "t_index", "annotator_count"],
            on=[f"human_majority_{dimension}" for dimension in SAMPLE_DIMENSIONS],
            variable_name="human_majority_column",
            value_name="human_majority_label",
        )
        .with_columns(
            pl.col("human_majority_column")
            .str.strip_prefix("human_majority_")
            .alias("dimension")
        )
        .drop("human_majority_column")
    )


def _build_case_universe(
    *,
    labels_path: Path,
    wrangled_dir: Path,
    annotations_dir: Path,
    runs_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    labels = pl.read_parquet(labels_path)
    entries = load_entries(wrangled_dir)

    base = labels.join(entries, on=["persona_id", "t_index", "date"], how="inner")
    base_long = (
        base.unpivot(
            index=[
                "persona_id",
                "t_index",
                "date",
                "rationales_json",
                "persona_name",
                "persona_age",
                "persona_profession",
                "persona_culture",
                "persona_core_values",
                "persona_bio",
                "initial_entry",
                "nudge_text",
                "response_text",
                "has_nudge",
                "has_response",
                "annotation_order",
            ]
            if "annotation_order" in base.columns
            else [
                "persona_id",
                "t_index",
                "date",
                "rationales_json",
                "persona_name",
                "persona_age",
                "persona_profession",
                "persona_culture",
                "persona_core_values",
                "persona_bio",
                "initial_entry",
                "nudge_text",
                "response_text",
                "has_nudge",
                "has_response",
            ],
            on=[f"alignment_{dimension}" for dimension in SAMPLE_DIMENSIONS],
            variable_name="alignment_column",
            value_name="persisted_label",
        )
        .with_columns(
            pl.col("alignment_column").str.strip_prefix("alignment_").alias("dimension")
        )
        .drop("alignment_column")
        .filter(pl.col("dimension").is_in(SAMPLE_DIMENSIONS))
    )

    frontier = _load_frontier_outputs(runs_dir).filter(
        pl.col("dimension").is_in(SAMPLE_DIMENSIONS)
    )
    frontier_long = frontier.group_by(["persona_id", "t_index", "date", "dimension"]).agg(
        [
            pl.first("target").alias("frontier_target"),
            pl.first("split").alias("frontier_split"),
            pl.mean("uncertainty").alias("mean_uncertainty"),
            (pl.col("predicted_class") != pl.col("target")).sum().alias("n_errors"),
            (
                (pl.col("target") != 0) & (pl.col("predicted_class") != pl.col("target"))
            )
            .sum()
            .alias("n_non_neutral_errors"),
            (
                (pl.col("target") != 0) & (pl.col("predicted_class") == 0)
            )
            .sum()
            .alias("n_missed_active"),
            (
                ((pl.col("target") == -1) & (pl.col("predicted_class") == 1))
                | ((pl.col("target") == 1) & (pl.col("predicted_class") == -1))
            )
            .sum()
            .alias("n_severe_flip"),
            (pl.col("predicted_class") == pl.col("target")).sum().alias("n_correct"),
            *[
                pl.col("predicted_class")
                .filter(pl.col("run_id") == run_id)
                .first()
                .alias(f"{run_id}_pred")
                for run_id in FRONTIER_RUN_IDS
            ],
            *[
                pl.col("uncertainty")
                .filter(pl.col("run_id") == run_id)
                .first()
                .alias(f"{run_id}_uncertainty")
                for run_id in FRONTIER_RUN_IDS
            ],
        ]
    )

    human_long = _load_human_majorities(annotations_dir)

    cases = (
        base_long.join(
            frontier_long, on=["persona_id", "t_index", "date", "dimension"], how="inner"
        )
        .join(
            human_long, on=["persona_id", "t_index", "dimension"], how="left"
        )
        .with_columns(
            [
                pl.lit(True).alias("has_frontier_artifact"),
                pl.col("human_majority_label").is_not_null().alias("has_human_overlap"),
                (
                    pl.col("human_majority_label").is_not_null()
                    & (pl.col("human_majority_label") != pl.col("persisted_label"))
                ).alias("judge_human_majority_disagree"),
                ((pl.col("persisted_label") != 0) & (pl.col("n_errors") == 3)).alias(
                    "stable_non_neutral_miss"
                ),
                (pl.col("n_severe_flip") == 3).alias("stable_severe_polarity_flip"),
                (pl.col("n_correct") == 3).alias("stable_correct_control"),
                pl.struct(["initial_entry", "nudge_text", "response_text"])
                .map_elements(
                    lambda row: build_session_content(
                        row["initial_entry"],
                        row["nudge_text"],
                        row["response_text"],
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias("session_content"),
            ]
        )
        .with_columns(
            pl.struct(["dimension", "persona_id", "t_index"]).map_elements(
                lambda row: _case_id(
                    row["dimension"], row["persona_id"], int(row["t_index"])
                ),
                return_dtype=pl.Utf8,
            )
            .alias("case_id")
        )
    )

    mismatches = cases.filter(pl.col("frontier_target") != pl.col("persisted_label"))
    if mismatches.height > 0:
        raise ValueError(
            "Frontier outputs no longer match persisted labels for some sampled "
            "dimensions. This audit expects the frontier target to equal the "
            "stored parquet label."
        )

    return cases, entries


def _iter_ranked_rows(frame: pl.DataFrame) -> list[dict]:
    return frame.to_dicts()


def _select_dimension_bucket(cases: pl.DataFrame, dimension: str, target_count: int) -> list[dict]:
    frame = cases.filter(pl.col("dimension") == dimension)
    selected: list[dict] = []
    selected_ids: set[str] = set()

    tiers = [
        (
            "stable_non_neutral_miss",
            frame.filter(pl.col("stable_non_neutral_miss")).sort(
                ["mean_uncertainty", "persona_id", "t_index", "case_id"]
            ),
        ),
        (
            "stable_severe_polarity_flip",
            frame.filter(pl.col("stable_severe_polarity_flip")).sort(
                ["mean_uncertainty", "persona_id", "t_index", "case_id"]
            ),
        ),
        (
            "judge_human_majority_disagree",
            frame.filter(pl.col("judge_human_majority_disagree")).sort(
                ["n_errors", "mean_uncertainty", "persona_id", "t_index", "case_id"],
                descending=[True, False, False, False, False],
            ),
        ),
        (
            "low_uncertainty_correct_control",
            frame.filter(pl.col("stable_correct_control")).sort(
                ["mean_uncertainty", "persona_id", "t_index", "case_id"]
            ),
        ),
    ]

    for reason, tier in tiers:
        for row in _iter_ranked_rows(tier):
            if row["case_id"] in selected_ids:
                continue
            row["selection_reason"] = reason
            selected.append(row)
            selected_ids.add(row["case_id"])
            if len(selected) == target_count:
                return selected

    fallback = frame.filter(~pl.col("case_id").is_in(list(selected_ids))).sort(
        ["n_errors", "mean_uncertainty", "persona_id", "t_index", "case_id"],
        descending=[True, False, False, False, False],
    )
    for row in _iter_ranked_rows(fallback):
        row["selection_reason"] = "fallback_remaining_case"
        selected.append(row)
        selected_ids.add(row["case_id"])
        if len(selected) == target_count:
            return selected

    raise ValueError(
        f"Could not fill the {dimension} bucket to {target_count} cases. "
        f"Only found {len(selected)} candidates."
    )


def select_audit_manifest(cases: pl.DataFrame) -> pl.DataFrame:
    selected_rows: list[dict] = []

    for dimension in SAMPLE_DIMENSIONS:
        bucket_rows = _select_dimension_bucket(cases, dimension, BUCKET_SIZES[dimension])
        for rank, row in enumerate(bucket_rows, start=1):
            row["selection_rank_within_dimension"] = rank
            row["blind_review_priority"] = dimension in HARD_DIMENSIONS and rank <= 3
            row["blind_review_rank"] = rank if dimension in HARD_DIMENSIONS and rank <= 3 else None
            selected_rows.append(row)

    manifest = pl.DataFrame(selected_rows).select(
        [
            "case_id",
            "dimension",
            "persona_id",
            "t_index",
            "date",
            "selection_reason",
            "selection_rank_within_dimension",
            "persisted_label",
            "run_019_pred",
            "run_020_pred",
            "run_021_pred",
            "run_019_uncertainty",
            "run_020_uncertainty",
            "run_021_uncertainty",
            "mean_uncertainty",
            "n_errors",
            "n_non_neutral_errors",
            "n_missed_active",
            "n_severe_flip",
            "n_correct",
            "has_human_overlap",
            "human_majority_label",
            "judge_human_majority_disagree",
            "blind_review_priority",
            "blind_review_rank",
        ]
    )

    counts = (
        manifest.group_by("dimension")
        .len()
        .sort("dimension")
        .to_dicts()
    )
    expected = {dimension: BUCKET_SIZES[dimension] for dimension in SAMPLE_DIMENSIONS}
    observed = {row["dimension"]: row["len"] for row in counts}
    if observed != expected:
        raise ValueError(f"Unexpected sample counts: expected {expected}, observed {observed}")

    return manifest.sort(["dimension", "selection_rank_within_dimension", "case_id"])


def _build_entry_maps(entries: pl.DataFrame) -> tuple[dict[tuple[str, int], dict], dict[str, list[dict]]]:
    entry_lookup: dict[tuple[str, int], dict] = {}
    entries_by_persona: dict[str, list[dict]] = {}

    sorted_entries = entries.sort(["persona_id", "t_index"]).to_dicts()
    for row in sorted_entries:
        key = (row["persona_id"], int(row["t_index"]))
        entry_lookup[key] = row
        entries_by_persona.setdefault(row["persona_id"], []).append(row)

    return entry_lookup, entries_by_persona


def _render_prompt_record(
    row: dict,
    *,
    condition: str,
    entry_lookup: dict[tuple[str, int], dict],
    entries_by_persona: dict[str, list[dict]],
    schwartz_config: dict,
) -> dict:
    entry = entry_lookup[(row["persona_id"], int(row["t_index"]))]
    session_content = build_session_content(
        entry["initial_entry"], entry["nudge_text"], entry["response_text"]
    )

    previous_entries = [
        {
            "date": previous["date"],
            "content": build_session_content(
                previous["initial_entry"],
                previous["nudge_text"],
                previous["response_text"],
            ),
        }
        for previous in entries_by_persona[entry["persona_id"]]
        if int(previous["t_index"]) < int(entry["t_index"])
    ]

    persona_core_values = list(entry["persona_core_values"] or [])
    persona_bio = entry.get("persona_bio") or ""
    previous_context = previous_entries or None

    if condition == "profile_only":
        persona_bio = ""
        previous_context = None
    elif condition == "student_visible":
        persona_core_values = []
        persona_bio = ""
        previous_context = None

    prompt = render_judge_prompt(
        session_content=session_content,
        entry_date=entry["date"],
        persona_name=entry.get("persona_name") or "",
        persona_age=entry.get("persona_age") or "",
        persona_profession=entry.get("persona_profession") or "",
        persona_culture=entry.get("persona_culture") or "",
        persona_core_values=persona_core_values,
        persona_bio=persona_bio,
        schwartz_config=schwartz_config,
        previous_entries=previous_context,
    )

    return {
        "case_id": row["case_id"],
        "condition": condition,
        "dimension": row["dimension"],
        "persona_id": row["persona_id"],
        "t_index": int(row["t_index"]),
        "date": row["date"],
        "persisted_label": int(row["persisted_label"]),
        "selection_reason": row["selection_reason"],
        "prompt": prompt,
        "context_flags": {
            "bio_included": condition == "full_context",
            "previous_entries_included": condition == "full_context",
            "core_values_included": condition != "student_visible",
        },
        "context_stats": {
            "previous_entries_count": len(previous_entries),
            "current_session_chars": len(session_content),
        },
    }


def _build_reference_section(
    row: dict,
    *,
    entry_lookup: dict[tuple[str, int], dict],
    entries_by_persona: dict[str, list[dict]],
) -> list[str]:
    entry = entry_lookup[(row["persona_id"], int(row["t_index"]))]
    previous_entries = [
        previous
        for previous in entries_by_persona[entry["persona_id"]]
        if int(previous["t_index"]) < int(entry["t_index"])
    ]

    lines = [
        f"## {row['case_id']}",
        f"- Dimension: `{row['dimension']}`",
        f"- Selection reason: `{row['selection_reason']}`",
        f"- Persisted label: `{int(row['persisted_label'])}`",
        f"- Frontier predictions: `run_019={int(row['run_019_pred'])}`, `run_020={int(row['run_020_pred'])}`, `run_021={int(row['run_021_pred'])}`",
        f"- Mean uncertainty: `{float(row['mean_uncertainty']):.6f}`",
        f"- Human majority label: `{'' if row['human_majority_label'] is None else int(row['human_majority_label'])}`",
        "",
        "**Profile**",
        f"- Name: {entry.get('persona_name') or ''}",
        f"- Age: {entry.get('persona_age') or ''}",
        f"- Profession: {entry.get('persona_profession') or ''}",
        f"- Culture: {entry.get('persona_culture') or ''}",
        f"- Core Values: {', '.join(entry.get('persona_core_values') or [])}",
        f"- Bio: {entry.get('persona_bio') or ''}",
        "",
        "**Previous Entries**",
    ]

    if previous_entries:
        for previous in previous_entries:
            lines.append(
                f"- {previous['date']}: {build_session_content(previous['initial_entry'], previous['nudge_text'], previous['response_text'])}"
            )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "**Current Session**",
            "",
            build_session_content(
                entry["initial_entry"], entry["nudge_text"], entry["response_text"]
            ),
            "",
        ]
    )
    return lines


def _write_manual_review_artifacts(
    output_dir: Path,
    manifest: pl.DataFrame,
    *,
    entry_lookup: dict[tuple[str, int], dict],
    entries_by_persona: dict[str, list[dict]],
) -> None:
    workbook = manifest.select(
        [
            "case_id",
            "dimension",
            "persona_id",
            "t_index",
            "date",
            "selection_reason",
            "persisted_label",
            "human_majority_label",
            "blind_review_priority",
            "blind_review_rank",
        ]
    ).with_columns(
        [
            pl.lit("").alias("reviewer_id"),
            pl.lit("").alias("blind_text_label"),
            pl.lit("").alias("blind_text_notes"),
            pl.lit("").alias("rich_context_label"),
            pl.lit("").alias("rich_context_notes"),
            pl.lit("").alias("label_changed_after_context"),
            pl.lit("").alias("preferred_target"),
            pl.lit("").alias("recommendation_notes"),
        ]
    )
    workbook.write_csv(output_dir / "manual_review_workbook.csv")

    blind_lines = [
        "# twinkl-747 Blind Review Packet",
        "",
        "Use `manual_review_workbook.csv` to record answers.",
        "For each case below, label only the focal dimension from session text alone before opening the richer context packet.",
        "",
    ]
    for row in manifest.filter(pl.col("blind_review_priority")).sort(
        ["dimension", "blind_review_rank", "case_id"]
    ).to_dicts():
        entry = entry_lookup[(row["persona_id"], int(row["t_index"]))]
        blind_lines.extend(
            [
                f"## {row['case_id']}",
                f"- Dimension under review: `{row['dimension']}`",
                f"- Workbook row: `{row['case_id']}`",
                "",
                build_session_content(
                    entry["initial_entry"], entry["nudge_text"], entry["response_text"]
                ),
                "",
            ]
        )
    (output_dir / "manual_review_blind_packet.md").write_text(
        "\n".join(blind_lines).rstrip() + "\n",
        encoding="utf-8",
    )

    reference_lines = [
        "# twinkl-747 Rich Context Reference Packet",
        "",
        "Use this after completing any blind text-only judgments for the priority cases.",
        "",
    ]
    for row in manifest.to_dicts():
        reference_lines.extend(
            _build_reference_section(
                row,
                entry_lookup=entry_lookup,
                entries_by_persona=entries_by_persona,
            )
        )
    (output_dir / "manual_review_reference.md").write_text(
        "\n".join(reference_lines).rstrip() + "\n",
        encoding="utf-8",
    )


def _write_readme(output_dir: Path) -> None:
    lines = [
        "# twinkl-747 Audit Bundle",
        "",
        "This bundle was prepared by `scripts/journalling/twinkl_747_prepare_audit.py`.",
        "",
        "## Files",
        "",
        "- `sample_manifest.csv`: the deterministic 50-case audit sample.",
        "- `prompts/<condition>.jsonl`: rendered judge prompts for `full_context`, `profile_only`, and `student_visible`.",
        "- `results/<condition>_results.jsonl`: empty templates to fill with externally run judge outputs.",
        "- `manual_review_workbook.csv`: answer sheet for human review.",
        "- `manual_review_blind_packet.md`: text-only packet for the top 3 hard cases per hard dimension.",
        "- `manual_review_reference.md`: rich-context reference for all sampled cases.",
        "",
        "## Result File Format",
        "",
        "Write one JSON object per line to each results file:",
        "",
        "```json",
        '{"case_id":"security__013d8101__1","scores":{"self_direction":0,"stimulation":0,"hedonism":0,"achievement":0,"power":0,"security":1,"conformity":0,"tradition":0,"benevolence":1,"universalism":0},"rationales":{"security":"...","benevolence":"..."}}',
        "```",
        "",
        "Only `case_id` and `scores` are required by the summarizer. The `scores` object must contain all 10 Schwartz dimensions.",
        "",
        "## Next Step",
        "",
        "After filling the three results files, run `scripts/journalling/twinkl_747_summarize_audit.py` against this bundle directory.",
        "",
    ]
    (output_dir / "README.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def prepare_audit_bundle(
    *,
    output_dir: Path,
    labels_path: Path,
    wrangled_dir: Path,
    annotations_dir: Path,
    runs_dir: Path,
    schwartz_path: Path,
) -> pl.DataFrame:
    cases, entries = _build_case_universe(
        labels_path=labels_path,
        wrangled_dir=wrangled_dir,
        annotations_dir=annotations_dir,
        runs_dir=runs_dir,
    )
    manifest = select_audit_manifest(cases)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_csv(output_dir / "sample_manifest.csv")

    schwartz_config = load_schwartz_values(schwartz_path)
    entry_lookup, entries_by_persona = _build_entry_maps(entries)

    prompt_dir = output_dir / "prompts"
    for condition in PROMPT_CONDITIONS:
        prompt_rows = [
            _render_prompt_record(
                row,
                condition=condition,
                entry_lookup=entry_lookup,
                entries_by_persona=entries_by_persona,
                schwartz_config=schwartz_config,
            )
            for row in manifest.to_dicts()
        ]
        _write_jsonl(prompt_dir / f"{condition}.jsonl", prompt_rows)

    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for condition in PROMPT_CONDITIONS:
        (results_dir / f"{condition}_results.jsonl").write_text("", encoding="utf-8")

    _write_manual_review_artifacts(
        output_dir,
        manifest,
        entry_lookup=entry_lookup,
        entries_by_persona=entries_by_persona,
    )
    _write_readme(output_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the twinkl-747 judge reachability audit bundle."
    )
    parser.add_argument(
        "--output-dir",
        default="logs/exports/twinkl_747",
        help="Directory to write the audit bundle.",
    )
    parser.add_argument(
        "--labels-path",
        default="logs/judge_labels/judge_labels.parquet",
        help="Path to the persisted judge labels parquet.",
    )
    parser.add_argument(
        "--wrangled-dir",
        default="logs/wrangled",
        help="Directory containing wrangled persona markdown files.",
    )
    parser.add_argument(
        "--annotations-dir",
        default="logs/annotations",
        help="Directory containing human annotation parquet files.",
    )
    parser.add_argument(
        "--runs-dir",
        default="logs/experiments/runs",
        help="Directory containing run metadata YAML files.",
    )
    parser.add_argument(
        "--schwartz-path",
        default="config/schwartz_values.yaml",
        help="Path to the Schwartz value config used to render prompts.",
    )
    args = parser.parse_args()

    manifest = prepare_audit_bundle(
        output_dir=Path(args.output_dir),
        labels_path=Path(args.labels_path),
        wrangled_dir=Path(args.wrangled_dir),
        annotations_dir=Path(args.annotations_dir),
        runs_dir=Path(args.runs_dir),
        schwartz_path=Path(args.schwartz_path),
    )

    counts = (
        manifest.group_by("dimension").len().sort("dimension").to_dicts()
    )
    print(f"Wrote audit bundle: {Path(args.output_dir)}")
    print(f"Sample size: {manifest.height}")
    for row in counts:
        print(f"  {row['dimension']}: {row['len']}")


if __name__ == "__main__":
    main()
