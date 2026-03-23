#!/usr/bin/env python3
"""Summarize externally run twinkl-747 judge reachability audit results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.judge import AlignmentScores

HARD_DIMENSIONS = ("security", "hedonism", "stimulation")
PROMPT_CONDITIONS = ("full_context", "profile_only", "student_visible")
COMPARISON_SPECS = (
    ("persisted_label", "full_context_label", "persisted_vs_full_context"),
    ("full_context_label", "profile_only_label", "full_context_vs_profile_only"),
    ("profile_only_label", "student_visible_label", "profile_only_vs_student_visible"),
)
REDUCED_CONTEXT_PREFERENCES = {"reduced_context", "text_only_ambiguous"}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_condition_results(
    *,
    bundle_dir: Path,
    manifest: pl.DataFrame,
    condition: str,
) -> pl.DataFrame:
    result_path = bundle_dir / "results" / f"{condition}_results.jsonl"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result file: {result_path}")

    manifest_lookup = {
        row["case_id"]: row["dimension"] for row in manifest.select(["case_id", "dimension"]).to_dicts()
    }
    normalized_rows: list[dict] = []

    for payload in _read_jsonl(result_path):
        case_id = payload.get("case_id")
        if not isinstance(case_id, str) or case_id not in manifest_lookup:
            raise ValueError(
                f"{result_path} contains an unknown or missing case_id: {case_id!r}"
            )
        scores_payload = payload.get("scores")
        if not isinstance(scores_payload, dict):
            raise ValueError(
                f"{result_path} case {case_id} is missing the `scores` object."
            )
        scores = AlignmentScores.model_validate(scores_payload)
        dimension = manifest_lookup[case_id]
        normalized_rows.append(
            {
                "case_id": case_id,
                f"{condition}_label": getattr(scores, dimension),
                f"{condition}_rationales_json": json.dumps(
                    payload.get("rationales") or {}, ensure_ascii=True
                ),
            }
        )

    frame = pl.DataFrame(normalized_rows)
    duplicates = frame.group_by("case_id").len().filter(pl.col("len") > 1)
    if duplicates.height > 0:
        raise ValueError(f"{result_path} contains duplicate case_ids.")

    expected_case_ids = set(manifest["case_id"].to_list())
    observed_case_ids = set(frame["case_id"].to_list())
    missing = sorted(expected_case_ids - observed_case_ids)
    extra = sorted(observed_case_ids - expected_case_ids)
    if missing or extra:
        raise ValueError(
            f"{result_path} does not match the manifest. Missing={missing[:5]}, extra={extra[:5]}"
        )

    return frame


def _band_label(flip_count: int) -> str:
    if flip_count <= 1:
        return "low"
    if flip_count <= 3:
        return "ambiguous"
    return "substantive"


def _load_manual_review(workbook_path: Path) -> list[dict]:
    if not workbook_path.exists():
        return []

    workbook = pl.read_csv(workbook_path)
    rows = workbook.to_dicts()
    reviewed_rows: list[dict] = []
    for row in rows:
        if not any(
            str(row.get(column) or "").strip()
            for column in (
                "blind_text_label",
                "rich_context_label",
                "preferred_target",
                "recommendation_notes",
            )
        ):
            continue

        normalized = dict(row)
        for column in ("blind_text_label", "rich_context_label", "blind_review_rank"):
            value = normalized.get(column)
            if isinstance(value, (int, float)):
                normalized[column] = int(value)
            elif value not in (None, ""):
                normalized[column] = int(value)
            else:
                normalized[column] = None
        normalized["blind_review_priority"] = str(
            normalized.get("blind_review_priority") or ""
        ).lower() == "true"
        normalized["preferred_target"] = (normalized.get("preferred_target") or "").strip()
        reviewed_rows.append(normalized)

    return reviewed_rows


def _build_comparison_rows(results: pl.DataFrame) -> pl.DataFrame:
    comparison_frames: list[pl.DataFrame] = []

    for source_col, target_col, name in COMPARISON_SPECS:
        comparison_frames.append(
            results.select(
                [
                    "case_id",
                    "dimension",
                    pl.lit(name).alias("comparison"),
                    pl.col(source_col).alias("source_label"),
                    pl.col(target_col).alias("target_label"),
                ]
            ).with_columns((pl.col("source_label") != pl.col("target_label")).alias("is_flip"))
        )

    return pl.concat(comparison_frames, how="vertical").sort(
        ["comparison", "dimension", "case_id"]
    )


def _build_flip_summary(comparisons: pl.DataFrame) -> pl.DataFrame:
    summary = (
        comparisons.group_by(["comparison", "dimension", "source_label"])
        .agg(
            [
                pl.len().alias("n_cases"),
                pl.col("is_flip").sum().alias("flip_count"),
            ]
        )
        .with_columns(
            pl.col("flip_count")
            .map_elements(_band_label, return_dtype=pl.Utf8)
            .alias("band")
        )
        .sort(["comparison", "dimension", "source_label"])
    )
    return summary


def _manual_dimension_summary(review_rows: list[dict]) -> dict[str, dict[str, int]]:
    summary = {
        dimension: {
            "reviewed_rows": 0,
            "blind_reviewed_rows": 0,
            "blind_unrecoverable_count": 0,
            "preferred_reduced_or_text_count": 0,
        }
        for dimension in HARD_DIMENSIONS
    }

    for row in review_rows:
        dimension = row["dimension"]
        if dimension not in summary:
            continue
        summary[dimension]["reviewed_rows"] += 1
        if row.get("preferred_target") in REDUCED_CONTEXT_PREFERENCES:
            summary[dimension]["preferred_reduced_or_text_count"] += 1
        if row.get("blind_review_priority"):
            summary[dimension]["blind_reviewed_rows"] += 1
            if (
                row.get("blind_text_label") is not None
                and row.get("rich_context_label") is not None
                and row["blind_text_label"] != row["rich_context_label"]
            ):
                summary[dimension]["blind_unrecoverable_count"] += 1

    return summary


def _format_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        rows = [["-"] * len(headers)]
    separator = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _comparison_table(summary: pl.DataFrame, comparison: str) -> str:
    rows = []
    for row in summary.filter(pl.col("comparison") == comparison).to_dicts():
        rows.append(
            [
                row["dimension"],
                str(int(row["source_label"])),
                str(int(row["flip_count"])),
                str(int(row["n_cases"])),
                row["band"],
            ]
        )
    return _format_markdown_table(
        ["Dimension", "Source Sign", "Flip Count", "Cases", "Band"],
        rows,
    )


def _recommendation_summary(
    *,
    flip_summary: pl.DataFrame,
    review_rows: list[dict],
) -> tuple[str, list[dict]]:
    review_summary = _manual_dimension_summary(review_rows)
    dimension_rows: list[dict] = []

    for dimension in HARD_DIMENSIONS:
        path_rows = flip_summary.filter(
            (pl.col("dimension") == dimension)
            & pl.col("comparison").is_in(
                ["full_context_vs_profile_only", "profile_only_vs_student_visible"]
            )
        )
        max_path_flip = (
            int(path_rows["flip_count"].max()) if path_rows.height > 0 else 0
        )
        max_path_band = _band_label(max_path_flip)
        reproducibility_rows = flip_summary.filter(
            (pl.col("dimension") == dimension)
            & (pl.col("comparison") == "persisted_vs_full_context")
        )
        reproducibility_flip = (
            int(reproducibility_rows["flip_count"].max())
            if reproducibility_rows.height > 0
            else 0
        )

        manual = review_summary[dimension]

        if max_path_flip >= 4 or manual["blind_unrecoverable_count"] >= 2:
            recommendation = "change_distillation_target"
        elif max_path_band == "ambiguous" or manual["preferred_reduced_or_text_count"] >= 2:
            recommendation = "targeted_relabeling"
        else:
            recommendation = "keep_current_labels"

        dimension_rows.append(
            {
                "dimension": dimension,
                "max_path_flip": max_path_flip,
                "max_path_band": max_path_band,
                "reproducibility_flip": reproducibility_flip,
                "reviewed_rows": manual["reviewed_rows"],
                "blind_unrecoverable_count": manual["blind_unrecoverable_count"],
                "preferred_reduced_or_text_count": manual["preferred_reduced_or_text_count"],
                "recommendation": recommendation,
            }
        )

    if any(row["recommendation"] == "change_distillation_target" for row in dimension_rows):
        overall = "change_distillation_target"
    elif any(row["recommendation"] == "targeted_relabeling" for row in dimension_rows):
        overall = "targeted_relabeling"
    else:
        overall = "keep_current_labels"

    return overall, dimension_rows


def build_report(
    *,
    manifest: pl.DataFrame,
    results: pl.DataFrame,
    comparisons: pl.DataFrame,
    flip_summary: pl.DataFrame,
    review_rows: list[dict],
    overall_recommendation: str,
    dimension_recommendations: list[dict],
) -> str:
    lines = [
        "# twinkl-747 Judge Reachability Audit",
        "",
        "## Audit Scope",
        "",
        f"- Sample size: `{manifest.height}` cases",
        f"- Hard dimensions: `{', '.join(HARD_DIMENSIONS)}`",
        f"- Control dimensions: `self_direction`, `universalism`",
        f"- Reviewed manual rows: `{len(review_rows)}`",
        f"- Overall recommendation: `{overall_recommendation}`",
        "",
        "## 1. Persisted Label vs Full Context",
        "",
        _comparison_table(flip_summary, "persisted_vs_full_context"),
        "",
        "This table tests whether a fresh rich-context rerun reproduces the persisted training labels or surfaces pre-existing label drift.",
        "",
        "## 2. Full Context vs Profile Only",
        "",
        _comparison_table(flip_summary, "full_context_vs_profile_only"),
        "",
        "These flips isolate how much the richer teacher signal depends on biography and prior-entry trajectory.",
        "",
        "## 3. Profile Only vs Student Visible",
        "",
        _comparison_table(flip_summary, "profile_only_vs_student_visible"),
        "",
        "These flips estimate how much of the remaining signal depends on declared profile hints rather than the current session text alone.",
        "",
        "## Hard-Dimension Recommendation Grid",
        "",
    ]

    recommendation_rows = [
        [
            row["dimension"],
            str(row["max_path_flip"]),
            row["max_path_band"],
            str(row["reproducibility_flip"]),
            str(row["blind_unrecoverable_count"]),
            str(row["preferred_reduced_or_text_count"]),
            row["recommendation"],
        ]
        for row in dimension_recommendations
    ]
    lines.append(
        _format_markdown_table(
            [
                "Dimension",
                "Max Path Flip",
                "Path Band",
                "Persisted↔Full Flip",
                "Blind Unrecoverable",
                "Manual Reduced/Text Pref",
                "Recommendation",
            ],
            recommendation_rows,
        )
    )
    lines.extend(
        [
            "",
            "Band interpretation for hard dimensions:",
            "- `0-1` flips: low concern",
            "- `2-3` flips: ambiguous, use manual review to break the tie",
            "- `4+` flips: substantive mismatch",
            "",
        ]
    )

    if review_rows:
        lines.extend(
            [
                "## Manual Review Notes",
                "",
                "Manual review rows were detected in `manual_review_workbook.csv` and were incorporated into the recommendation grid above.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Manual Review Notes",
                "",
                "No completed manual review rows were detected. The recommendation therefore rests only on flip counts and should be revisited after reviewers fill the workbook.",
                "",
            ]
        )

    lines.extend(
        [
            "## Output Files",
            "",
            "- `joined_results.csv`: manifest plus focal labels for each rerun condition",
            "- `comparison_rows.csv`: case-level flip rows for each comparison pair",
            "- `flip_summary.csv`: aggregated flip counts by comparison, dimension, and sign bucket",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def summarize_bundle(bundle_dir: Path) -> tuple[str, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    manifest = pl.read_csv(bundle_dir / "sample_manifest.csv")
    results = manifest
    for condition in PROMPT_CONDITIONS:
        results = results.join(
            _load_condition_results(
                bundle_dir=bundle_dir,
                manifest=manifest,
                condition=condition,
            ),
            on="case_id",
            how="inner",
        )

    comparisons = _build_comparison_rows(results)
    flip_summary = _build_flip_summary(comparisons)
    review_rows = _load_manual_review(bundle_dir / "manual_review_workbook.csv")
    overall_recommendation, dimension_recommendations = _recommendation_summary(
        flip_summary=flip_summary,
        review_rows=review_rows,
    )
    report = build_report(
        manifest=manifest,
        results=results,
        comparisons=comparisons,
        flip_summary=flip_summary,
        review_rows=review_rows,
        overall_recommendation=overall_recommendation,
        dimension_recommendations=dimension_recommendations,
    )
    return report, results, comparisons, flip_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize externally run twinkl-747 reachability audit results."
    )
    parser.add_argument(
        "--bundle-dir",
        default="logs/exports/twinkl_747",
        help="Directory produced by twinkl_747_prepare_audit.py.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional report output path. Defaults to <bundle-dir>/reachability_audit_report.md",
    )
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    output_path = (
        Path(args.output)
        if args.output
        else bundle_dir / "reachability_audit_report.md"
    )
    report, results, comparisons, flip_summary = summarize_bundle(bundle_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    results.write_csv(bundle_dir / "joined_results.csv")
    comparisons.write_csv(bundle_dir / "comparison_rows.csv")
    flip_summary.write_csv(bundle_dir / "flip_summary.csv")

    print(f"Wrote report: {output_path}")
    print(f"Joined rows: {results.height}")
    print(f"Comparison rows: {comparisons.height}")


if __name__ == "__main__":
    main()
