#!/usr/bin/env python3
"""Summarize twinkl-754 consensus re-judging results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.annotation_tool.agreement_metrics import (
    calculate_cohen_kappa,
    calculate_fleiss_kappa,
)
from src.judge.consensus_utils import (
    aggregate_rationale_coverage,
    compute_score_only_hash,
    count_entry_vector_differences,
    hash_file,
    load_bundle_status,
    load_expected_ids,
    read_jsonl,
    validate_result_rows_against_ids,
    write_bundle_status,
)
from src.judge.consolidate import consolidate_consensus_labels
from src.models.judge import SCHWARTZ_VALUE_ORDER

HARD_DIMENSIONS = ("security", "hedonism", "stimulation")
HUMAN_FLEISS_BASELINES = {
    "security": 0.48,
    "hedonism": 0.64,
    "stimulation": 0.58,
}
N_PASSES = 5


def _load_manifest(bundle_dir: Path) -> pl.DataFrame:
    manifest_path = bundle_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return pl.read_csv(manifest_path).sort(["persona_id", "t_index"])


def _manifest_entry_ids(manifest: pl.DataFrame) -> list[str]:
    return manifest["entry_id"].to_list()


def _is_valid_kappa(value: object) -> bool:
    return isinstance(value, int | float) and not math.isnan(float(value))


def _validate_pass_results(
    *,
    bundle_dir: Path,
    manifest: pl.DataFrame,
    pass_index: int,
) -> tuple[pl.DataFrame, dict[str, dict]]:
    pass_name = f"pass_{pass_index}"
    prompt_path = bundle_dir / "prompts" / f"{pass_name}.jsonl"
    result_path = bundle_dir / "results" / f"{pass_name}_results.jsonl"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing prompt file: {prompt_path}")
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result file: {result_path}")

    expected_entry_ids = load_expected_ids(prompt_path)
    if expected_entry_ids != _manifest_entry_ids(manifest):
        raise ValueError(
            f"{prompt_path} no longer matches manifest ordering or membership."
        )

    valid_manifest_ids = set(expected_entry_ids)
    normalized_rows, _coverage = validate_result_rows_against_ids(
        read_jsonl(result_path),
        valid_manifest_ids=valid_manifest_ids,
        expected_entry_ids=expected_entry_ids,
    )

    payloads_by_entry: dict[str, dict] = {}
    frame_rows: list[dict] = []

    for payload in normalized_rows:
        entry_id = payload["entry_id"]
        rationales = payload["rationales"]
        payloads_by_entry[entry_id] = {
            "scores": payload["scores"],
            "rationales": rationales,
        }
        frame_row = {
            "entry_id": entry_id,
            f"pass_{pass_index}_rationales_json": (
                json.dumps(rationales, ensure_ascii=True) if rationales else None
            ),
        }
        frame_rows.append(frame_row)

    frame = pl.DataFrame(frame_rows).sort("entry_id")
    return frame, payloads_by_entry


def _load_provenance_frame(bundle_dir: Path, filename: str) -> pl.DataFrame:
    provenance_path = bundle_dir / "provenance" / filename
    if not provenance_path.exists():
        raise FileNotFoundError(
            f"Missing provenance file: {provenance_path}. "
            "Run twinkl_754_merge_pass_results.py before summarizing."
        )
    return pl.read_csv(provenance_path)


def _payload_rows(payloads_by_entry: dict[str, dict]) -> list[dict]:
    return [
        {
            "entry_id": entry_id,
            "scores": payload["scores"],
            "rationales": payload["rationales"],
        }
        for entry_id, payload in sorted(payloads_by_entry.items())
    ]


def _verify_pass_provenance(
    *,
    bundle_dir: Path,
    pass_payloads: dict[int, dict[str, dict]],
    pass_provenance: pl.DataFrame,
    pass_similarity: pl.DataFrame,
) -> None:
    for pass_index in range(1, N_PASSES + 1):
        provenance_rows = pass_provenance.filter(pl.col("pass_index") == pass_index)
        if provenance_rows.height != 1:
            raise ValueError(f"Expected exactly one pass provenance row for pass_{pass_index}.")
        provenance = provenance_rows.to_dicts()[0]
        pass_name = f"pass_{pass_index}"
        prompt_path = bundle_dir / "prompts" / f"{pass_name}.jsonl"
        result_path = bundle_dir / "results" / f"{pass_name}_results.jsonl"
        payload_rows = _payload_rows(pass_payloads[pass_index])
        coverage = aggregate_rationale_coverage(payload_rows)

        if provenance["prompt_sha256"] != hash_file(prompt_path):
            raise ValueError(f"{pass_name} prompt hash does not match recorded provenance.")
        if provenance["raw_result_sha256"] != hash_file(result_path):
            raise ValueError(f"{pass_name} raw result hash does not match recorded provenance.")
        if provenance["score_only_sha256"] != compute_score_only_hash(payload_rows):
            raise ValueError(f"{pass_name} score-only hash does not match recorded provenance.")
        if int(provenance["non_zero_score_count"]) != coverage.non_zero_score_count:
            raise ValueError(
                f"{pass_name} non-zero score count does not match recorded provenance."
            )
        if int(provenance["non_zero_rationale_count"]) != coverage.non_zero_rationale_count:
            raise ValueError(
                f"{pass_name} rationale coverage count does not match recorded provenance."
            )
        if float(provenance["non_zero_rationale_coverage"]) < 1.0:
            raise ValueError(
                f"{pass_name} provenance indicates incomplete rationale coverage."
            )

    for row in pass_similarity.to_dicts():
        left_pass = int(row["left_pass_index"])
        right_pass = int(row["right_pass_index"])
        left_payloads = _payload_rows(pass_payloads[left_pass])
        right_payloads = _payload_rows(pass_payloads[right_pass])
        left_provenance = pass_provenance.filter(pl.col("pass_index") == left_pass).to_dicts()[0]
        right_provenance = pass_provenance.filter(pl.col("pass_index") == right_pass).to_dicts()[0]
        differing_entry_vectors = count_entry_vector_differences(
            left_payloads,
            right_payloads,
        )
        if bool(row["raw_hash_match"]) != (
            left_provenance["raw_result_sha256"] == right_provenance["raw_result_sha256"]
        ):
            raise ValueError(
                f"Stored raw-hash similarity does not match current pass files for "
                f"pass_{left_pass} vs pass_{right_pass}."
            )
        if bool(row["score_hash_match"]) != (
            left_provenance["score_only_sha256"] == right_provenance["score_only_sha256"]
        ):
            raise ValueError(
                f"Stored score-hash similarity does not match current pass files for "
                f"pass_{left_pass} vs pass_{right_pass}."
            )
        if int(row["differing_entry_vectors"]) != differing_entry_vectors:
            raise ValueError(
                f"Stored entry-vector diff count does not match current pass files for "
                f"pass_{left_pass} vs pass_{right_pass}."
            )

    duplicate_pairs = pass_similarity.filter(
        pl.col("raw_hash_match") | pl.col("score_hash_match")
    )
    if duplicate_pairs.height > 0:
        formatted_pairs = ", ".join(
            f"{row['left_pass_name']} vs {row['right_pass_name']}"
            for row in duplicate_pairs.to_dicts()
        )
        raise ValueError(
            "Duplicate pass outputs detected before consensus aggregation: "
            + formatted_pairs
        )


def _resolve_votes(votes: list[int]) -> tuple[int, str, int]:
    active_count = sum(value != 0 for value in votes)
    inactive_count = len(votes) - active_count
    if inactive_count >= 3:
        if inactive_count == 5:
            tier = "unanimous"
        elif inactive_count >= 4:
            tier = "strong"
        else:
            tier = "bare_majority"
        return 0, tier, inactive_count

    positive_count = sum(value == 1 for value in votes)
    negative_count = sum(value == -1 for value in votes)
    if positive_count == negative_count:
        return 0, "no_majority", active_count

    consensus_label = 1 if positive_count > negative_count else -1
    winning_count = max(positive_count, negative_count)
    if winning_count == len(votes):
        tier = "unanimous"
    elif winning_count >= 4:
        tier = "strong"
    else:
        tier = "bare_majority"
    return consensus_label, tier, winning_count


def _load_human_majority_wide(annotations_dir: Path) -> pl.DataFrame:
    parquet_files = sorted(annotations_dir.glob("*.parquet"))
    if not parquet_files:
        return pl.DataFrame(
            schema={
                "persona_id": pl.Utf8,
                "t_index": pl.Int64,
                **{f"alignment_{value_name}": pl.Int64 for value_name in SCHWARTZ_VALUE_ORDER},
            }
        )

    annotations = pl.concat([pl.read_parquet(path) for path in parquet_files], how="vertical")
    expected_annotators = len(parquet_files)
    grouped = annotations.group_by(["persona_id", "t_index"]).agg(
        [pl.len().alias("annotator_count")]
        + [
            pl.col(f"alignment_{value_name}")
            .drop_nulls()
            .alias(f"votes_{value_name}")
            for value_name in SCHWARTZ_VALUE_ORDER
        ]
    )
    shared = grouped.filter(pl.col("annotator_count") == expected_annotators)
    if shared.height == 0:
        return pl.DataFrame(
            schema={
                "persona_id": pl.Utf8,
                "t_index": pl.Int64,
                **{f"alignment_{value_name}": pl.Int64 for value_name in SCHWARTZ_VALUE_ORDER},
            }
        )

    return shared.select(
        ["persona_id", "t_index"]
        + [
            pl.col(f"votes_{value_name}")
            .list.sort()
            .list.get(expected_annotators // 2)
            .alias(f"alignment_{value_name}")
            for value_name in SCHWARTZ_VALUE_ORDER
        ]
    )


def _build_wide_label_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).sort(["persona_id", "t_index"])


def _build_consensus_outputs(
    *,
    manifest: pl.DataFrame,
    pass_payloads: dict[int, dict[str, dict]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    consensus_rows: list[dict] = []
    joined_rows: list[dict] = []

    for row in manifest.sort(["persona_id", "t_index"]).to_dicts():
        entry_id = row["entry_id"]
        consensus_scores: dict[str, int] = {}
        confidence_tiers: dict[str, str] = {}
        agreement_counts: dict[str, int] = {}
        changed_flags: dict[str, bool] = {}

        per_pass_scores = {
            pass_index: pass_payloads[pass_index][entry_id]["scores"]
            for pass_index in range(1, N_PASSES + 1)
        }
        pass_match_counts: dict[int, int] = {}

        for value_name in SCHWARTZ_VALUE_ORDER:
            votes = [per_pass_scores[pass_index][value_name] for pass_index in range(1, N_PASSES + 1)]
            consensus_label, tier, decisive_count = _resolve_votes(votes)
            persisted_label = int(row[f"alignment_{value_name}"])
            consensus_scores[value_name] = consensus_label
            confidence_tiers[value_name] = tier
            agreement_counts[value_name] = decisive_count
            changed_flags[value_name] = consensus_label != persisted_label

        for pass_index in range(1, N_PASSES + 1):
            pass_match_counts[pass_index] = sum(
                per_pass_scores[pass_index][value_name] == consensus_scores[value_name]
                for value_name in SCHWARTZ_VALUE_ORDER
            )

        perfect_passes = [
            pass_index for pass_index, count in pass_match_counts.items()
            if count == len(SCHWARTZ_VALUE_ORDER)
        ]
        rationale_pass = perfect_passes[0] if perfect_passes else max(
            pass_match_counts,
            key=lambda pass_index: (pass_match_counts[pass_index], -pass_index),
        )
        rationale_payload = pass_payloads[rationale_pass][entry_id]["rationales"]
        rationales_json = json.dumps(rationale_payload, ensure_ascii=True) if rationale_payload else None
        mismatch_count = len(SCHWARTZ_VALUE_ORDER) - pass_match_counts[rationale_pass]

        consensus_row = {
            "entry_id": entry_id,
            "persona_id": row["persona_id"],
            "t_index": int(row["t_index"]),
            "date": row["date"],
            "alignment_vector": [consensus_scores[value_name] for value_name in SCHWARTZ_VALUE_ORDER],
            "rationales_json": rationales_json,
            "rationale_source_pass": rationale_pass,
            "rationale_match_count": pass_match_counts[rationale_pass],
            "rationale_mismatch_count": mismatch_count,
        }
        joined_row = {
            "entry_id": entry_id,
            "persona_id": row["persona_id"],
            "t_index": int(row["t_index"]),
            "date": row["date"],
            "rationale_source_pass": rationale_pass,
            "rationale_match_count": pass_match_counts[rationale_pass],
            "rationale_mismatch_count": mismatch_count,
            "selected_rationales_json": rationales_json,
        }
        for value_name in SCHWARTZ_VALUE_ORDER:
            persisted_label = int(row[f"alignment_{value_name}"])
            consensus_label = consensus_scores[value_name]
            consensus_row[f"alignment_{value_name}"] = consensus_label
            consensus_row[f"confidence_{value_name}"] = confidence_tiers[value_name]
            consensus_row[f"consensus_agreement_{value_name}"] = agreement_counts[value_name]
            consensus_row[f"label_changed_{value_name}"] = changed_flags[value_name]

            joined_row[f"persisted_alignment_{value_name}"] = persisted_label
            joined_row[f"consensus_alignment_{value_name}"] = consensus_label
            joined_row[f"confidence_{value_name}"] = confidence_tiers[value_name]
            joined_row[f"consensus_agreement_{value_name}"] = agreement_counts[value_name]
            joined_row[f"label_changed_{value_name}"] = changed_flags[value_name]
            for pass_index in range(1, N_PASSES + 1):
                joined_row[f"pass_{pass_index}_alignment_{value_name}"] = per_pass_scores[pass_index][value_name]

        consensus_rows.append(consensus_row)
        joined_rows.append(joined_row)

    return _build_wide_label_frame(consensus_rows), pl.DataFrame(joined_rows).sort(
        ["persona_id", "t_index"]
    )


def _build_comparison_rows(joined: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict] = []
    for row in joined.to_dicts():
        for value_name in SCHWARTZ_VALUE_ORDER:
            long_row = {
                "entry_id": row["entry_id"],
                "persona_id": row["persona_id"],
                "t_index": int(row["t_index"]),
                "date": row["date"],
                "dimension": value_name,
                "persisted_label": int(row[f"persisted_alignment_{value_name}"]),
                "consensus_label": int(row[f"consensus_alignment_{value_name}"]),
                "confidence_tier": row[f"confidence_{value_name}"],
                "consensus_agreement": int(row[f"consensus_agreement_{value_name}"]),
                "label_changed": bool(row[f"label_changed_{value_name}"]),
            }
            for pass_index in range(1, N_PASSES + 1):
                long_row[f"pass_{pass_index}_label"] = int(
                    row[f"pass_{pass_index}_alignment_{value_name}"]
                )
            rows.append(long_row)
    return pl.DataFrame(rows).sort(["dimension", "persona_id", "t_index"])


def _build_flip_summary(comparison_rows: pl.DataFrame) -> pl.DataFrame:
    return comparison_rows.group_by(
        ["dimension", "persisted_label", "consensus_label"]
    ).agg(pl.len().alias("n_entries")).sort(
        ["dimension", "persisted_label", "consensus_label"]
    )


def _build_confidence_summary(comparison_rows: pl.DataFrame) -> pl.DataFrame:
    return comparison_rows.group_by(["dimension", "confidence_tier"]).agg(
        [
            pl.len().alias("n_entries"),
            (pl.col("consensus_label") != 0).sum().alias("n_non_neutral"),
        ]
    ).sort(["dimension", "confidence_tier"])


def _build_persisted_wide(manifest: pl.DataFrame) -> pl.DataFrame:
    return manifest.select(
        ["persona_id", "t_index"]
        + [f"alignment_{value_name}" for value_name in SCHWARTZ_VALUE_ORDER]
    ).sort(["persona_id", "t_index"])


def _build_consensus_wide(consensus: pl.DataFrame) -> pl.DataFrame:
    return consensus.select(
        ["persona_id", "t_index"]
        + [f"alignment_{value_name}" for value_name in SCHWARTZ_VALUE_ORDER]
    ).sort(["persona_id", "t_index"])


def _build_pass_wide(pass_payloads: dict[str, dict]) -> pl.DataFrame:
    rows: list[dict] = []
    for entry_id, payload in sorted(pass_payloads.items()):
        persona_id, t_index_str = entry_id.split("__")
        row = {
            "persona_id": persona_id,
            "t_index": int(t_index_str),
        }
        for value_name in SCHWARTZ_VALUE_ORDER:
            row[f"alignment_{value_name}"] = payload["scores"][value_name]
        rows.append(row)
    return _build_wide_label_frame(rows)


def _build_irr_summary(
    *,
    persisted_wide: pl.DataFrame,
    consensus_wide: pl.DataFrame,
    human_majority_wide: pl.DataFrame,
    pass_payloads: dict[int, dict[str, dict]],
) -> tuple[pl.DataFrame, dict[str, float], dict[str, float], dict[str, float]]:
    fleiss = calculate_fleiss_kappa(
        [_build_pass_wide(pass_payloads[pass_index]) for pass_index in range(1, N_PASSES + 1)]
    )
    consensus_vs_persisted = calculate_cohen_kappa(consensus_wide, persisted_wide)
    consensus_vs_human = calculate_cohen_kappa(human_majority_wide, consensus_wide)
    persisted_vs_human = calculate_cohen_kappa(human_majority_wide, persisted_wide)

    rows: list[dict] = []
    for value_name in [*SCHWARTZ_VALUE_ORDER, "aggregate"]:
        rows.append(
            {
                "metric": "fleiss_kappa",
                "dimension": value_name,
                "value": fleiss.get(value_name),
                "reference": (
                    HUMAN_FLEISS_BASELINES.get(value_name)
                    if value_name in HUMAN_FLEISS_BASELINES
                    else None
                ),
            }
        )
        rows.append(
            {
                "metric": "consensus_vs_persisted_cohen_kappa",
                "dimension": value_name,
                "value": consensus_vs_persisted.get(value_name),
                "reference": None,
            }
        )
        rows.append(
            {
                "metric": "consensus_vs_human_cohen_kappa",
                "dimension": value_name,
                "value": consensus_vs_human.get(value_name),
                "reference": persisted_vs_human.get(value_name),
            }
        )
        rows.append(
            {
                "metric": "persisted_vs_human_cohen_kappa",
                "dimension": value_name,
                "value": persisted_vs_human.get(value_name),
                "reference": None,
            }
        )

    rows.append(
        {
            "metric": "fleiss_n_shared",
            "dimension": "aggregate",
            "value": fleiss.get("n_shared"),
            "reference": None,
        }
    )
    return (
        pl.DataFrame(rows),
        fleiss,
        consensus_vs_human,
        persisted_vs_human,
    )


def _evaluate_gate(comparison_rows: pl.DataFrame, *, consensus_vs_human: dict[str, float], persisted_vs_human: dict[str, float]) -> dict:
    aggregate_consensus = consensus_vs_human.get("aggregate")
    aggregate_persisted = persisted_vs_human.get("aggregate")
    agreement_passed = (
        _is_valid_kappa(aggregate_consensus)
        and _is_valid_kappa(aggregate_persisted)
        and aggregate_consensus >= aggregate_persisted
    )

    hard_rows: list[dict] = []
    confidence_passed = True
    for value_name in HARD_DIMENSIONS:
        subset = comparison_rows.filter(pl.col("dimension") == value_name)
        non_neutral = subset.filter(pl.col("consensus_label") != 0)
        low_confidence = non_neutral.filter(
            pl.col("confidence_tier").is_in(["bare_majority", "no_majority"])
        )
        ratio = (
            float(low_confidence.height) / float(non_neutral.height)
            if non_neutral.height > 0
            else 0.0
        )
        passes = ratio < 0.5
        confidence_passed = confidence_passed and passes
        hard_rows.append(
            {
                "dimension": value_name,
                "non_neutral_count": non_neutral.height,
                "low_confidence_count": low_confidence.height,
                "low_confidence_ratio": ratio,
                "passes": passes,
            }
        )

    return {
        "agreement_passed": agreement_passed,
        "confidence_passed": confidence_passed,
        "overall_passed": agreement_passed and confidence_passed,
        "aggregate_consensus_vs_human": aggregate_consensus,
        "aggregate_persisted_vs_human": aggregate_persisted,
        "hard_dimension_rows": hard_rows,
    }


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        rows = [["-"] * len(headers)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt_float(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int | float):
        if math.isnan(float(value)):
            return "N/A"
        return f"{value:.3f}"
    return str(value)


def build_report(
    *,
    manifest: pl.DataFrame,
    flip_summary: pl.DataFrame,
    confidence_summary: pl.DataFrame,
    irr_summary: pl.DataFrame,
    joined_results: pl.DataFrame,
    comparison_rows: pl.DataFrame,
    gate_summary: dict,
    pass_provenance: pl.DataFrame,
    pass_similarity: pl.DataFrame,
    bundle_status: dict | None = None,
) -> str:
    bundle_status = bundle_status or {}
    worker_models = sorted(
        {
            model
            for model_list in pass_provenance["worker_model"].to_list()
            for model in str(model_list).split(",")
            if model
        }
    )
    bundle_mode = str(bundle_status.get("bundle_mode", "full"))
    selected_entry_count = int(bundle_status.get("selected_entry_count", manifest.height))
    lines = [
        "# twinkl-754 Consensus Re-judging Report",
        "",
    ]
    warning_text = str(bundle_status.get("warning") or "").strip()
    if bundle_status.get("invalidated") and warning_text:
        lines.extend(
            [
                "> Warning: this bundle is marked invalidated.",
                f"> {warning_text}",
                "",
            ]
        )
    lines.extend(
        [
            "## Scope Summary",
            "",
            f"- Prompt condition: `profile_only`",
            f"- Bundle mode: `{bundle_mode}`",
            f"- Entries: `{selected_entry_count}`",
            f"- Passes: `{N_PASSES}`",
            f"- Personas: `{manifest.select('persona_id').n_unique()}`",
            f"- Worker model: `{', '.join(worker_models)}`",
            "",
            "## 1. Judge Repeated-Call Self-Consistency",
            "",
            "These kappas measure repeated-call consistency of the same judge workflow, not agreement among independent raters.",
            "",
        ]
    )
    if bundle_mode == "pilot":
        selection_summary = bundle_status.get("selection_summary") or {}
        lines.extend(
            [
                "Pilot selection:",
                (
                    f"- Requested entries: `{selection_summary.get('requested_size', selected_entry_count)}`"
                ),
                (
                    f"- Selected entries: `{selection_summary.get('selected_size', selected_entry_count)}`"
                ),
            ]
        )
        for value_name, count in (
            selection_summary.get("selected_non_zero_counts") or {}
        ).items():
            lines.append(f"- Selected non-zero `{value_name}` entries: `{int(count)}`")
        lines.append("")
    fleiss_rows = []
    for value_name in SCHWARTZ_VALUE_ORDER:
        row = irr_summary.filter(
            (pl.col("metric") == "fleiss_kappa")
            & (pl.col("dimension") == value_name)
        ).to_dicts()[0]
        fleiss_rows.append(
            [
                value_name,
                _fmt_float(row["value"]),
                _fmt_float(row["reference"]),
            ]
        )
    lines.append(
        _format_table(
            ["Dimension", "Fleiss kappa", "Human baseline"],
            fleiss_rows,
        )
    )
    lines.extend(
        [
            "",
            "## 2. Consensus vs Persisted",
            "",
        ]
    )
    persisted_kappa_rows = []
    for value_name in [*SCHWARTZ_VALUE_ORDER, "aggregate"]:
        kappa_row = irr_summary.filter(
            (pl.col("metric") == "consensus_vs_persisted_cohen_kappa")
            & (pl.col("dimension") == value_name)
        ).to_dicts()[0]
        persisted_kappa_rows.append(
            [
                value_name,
                _fmt_float(kappa_row["value"]),
            ]
        )
    lines.append(
        _format_table(
            ["Dimension", "Consensus vs persisted Cohen kappa"],
            persisted_kappa_rows,
        )
    )
    lines.extend(["", "Confusion counts:", ""])
    flip_rows = []
    for row in flip_summary.to_dicts():
        flip_rows.append(
            [
                row["dimension"],
                str(int(row["persisted_label"])),
                str(int(row["consensus_label"])),
                str(int(row["n_entries"])),
            ]
        )
    lines.append(
        _format_table(
            ["Dimension", "Persisted", "Consensus", "Count"],
            flip_rows,
        )
    )
    lines.extend(
        [
            "",
            "## 3. Consensus vs Human",
            "",
        ]
    )
    human_rows = []
    for value_name in [*SCHWARTZ_VALUE_ORDER, "aggregate"]:
        consensus_row = irr_summary.filter(
            (pl.col("metric") == "consensus_vs_human_cohen_kappa")
            & (pl.col("dimension") == value_name)
        ).to_dicts()[0]
        human_rows.append(
            [
                value_name,
                _fmt_float(consensus_row["value"]),
                _fmt_float(consensus_row["reference"]),
            ]
        )
    lines.append(
        _format_table(
            ["Dimension", "Consensus vs human", "Persisted vs human"],
            human_rows,
        )
    )
    lines.extend(
        [
            "",
            "## 4. Confidence Tier Distribution",
            "",
        ]
    )
    confidence_rows = []
    for row in confidence_summary.to_dicts():
        confidence_rows.append(
            [
                row["dimension"],
                row["confidence_tier"],
                str(int(row["n_entries"])),
                str(int(row["n_non_neutral"])),
            ]
        )
    lines.append(
        _format_table(
            ["Dimension", "Tier", "Entries", "Non-neutral entries"],
            confidence_rows,
        )
    )
    lines.extend(
        [
            "",
            "## 5. Pass Diagnostics",
            "",
        ]
    )
    pass_rows = []
    for row in pass_provenance.to_dicts():
        pass_rows.append(
            [
                row["pass_name"],
                str(int(row["attempt"])),
                row["worker_model"],
                row["raw_result_sha256"][:12],
                row["score_only_sha256"][:12],
                f"{float(row['non_zero_rationale_coverage']):.3f}",
                row["completion_timestamp"],
            ]
        )
    lines.append(
        _format_table(
            [
                "Pass",
                "Attempt",
                "Worker model",
                "Raw hash",
                "Score hash",
                "Rationale coverage",
                "Completed",
            ],
            pass_rows,
        )
    )
    lines.extend(["", "Pairwise similarity:", ""])
    similarity_rows = []
    for row in pass_similarity.to_dicts():
        similarity_rows.append(
            [
                f"{row['left_pass_name']} vs {row['right_pass_name']}",
                str(bool(row["raw_hash_match"])),
                str(bool(row["score_hash_match"])),
                str(int(row["identical_entry_vectors"])),
                str(int(row["differing_entry_vectors"])),
            ]
        )
    lines.append(
        _format_table(
            [
                "Pass pair",
                "Raw hash match",
                "Score hash match",
                "Identical entry vectors",
                "Differing entry vectors",
            ],
            similarity_rows,
        )
    )
    lines.extend(
        [
            "",
            "## 6. Hard-Dimension Gate",
            "",
            f"- Aggregate consensus-vs-human kappa: `{_fmt_float(gate_summary['aggregate_consensus_vs_human'])}`",
            f"- Aggregate persisted-vs-human kappa: `{_fmt_float(gate_summary['aggregate_persisted_vs_human'])}`",
            f"- Agreement gate passed: `{gate_summary['agreement_passed']}`",
            f"- Confidence gate passed: `{gate_summary['confidence_passed']}`",
            f"- Overall retrain gate passed: `{gate_summary['overall_passed']}`",
            "",
        ]
    )
    hard_rows = [
        [
            row["dimension"],
            str(int(row["non_neutral_count"])),
            str(int(row["low_confidence_count"])),
            f"{row['low_confidence_ratio']:.3f}",
            str(row["passes"]),
        ]
        for row in gate_summary["hard_dimension_rows"]
    ]
    lines.append(
        _format_table(
            [
                "Dimension",
                "Non-neutral labels",
                "Bare/no-majority labels",
                "Low-confidence ratio",
                "Passes",
            ],
            hard_rows,
        )
    )
    lines.extend(
        [
            "",
            "## 7. Hard-Dimension Deep Dive",
            "",
        ]
    )
    for value_name in HARD_DIMENSIONS:
        persisted_kappa = irr_summary.filter(
            (pl.col("metric") == "consensus_vs_persisted_cohen_kappa")
            & (pl.col("dimension") == value_name)
        ).to_dicts()[0]
        human_kappa = irr_summary.filter(
            (pl.col("metric") == "consensus_vs_human_cohen_kappa")
            & (pl.col("dimension") == value_name)
        ).to_dicts()[0]
        hard_gate_row = next(
            row for row in gate_summary["hard_dimension_rows"] if row["dimension"] == value_name
        )
        lines.extend(
            [
                f"### {value_name}",
                "",
                (
                    "Persisted-vs-consensus kappa: "
                    f"`{_fmt_float(persisted_kappa['value'])}`; "
                    "consensus-vs-human kappa: "
                    f"`{_fmt_float(human_kappa['value'])}`; "
                    "low-confidence non-neutral labels: "
                    f"`{hard_gate_row['low_confidence_count']}/{hard_gate_row['non_neutral_count']}`."
                ),
                "",
            ]
        )
        hard_confusion_rows = []
        for row in flip_summary.filter(pl.col("dimension") == value_name).to_dicts():
            hard_confusion_rows.append(
                [
                    str(int(row["persisted_label"])),
                    str(int(row["consensus_label"])),
                    str(int(row["n_entries"])),
                ]
            )
        lines.append(
            _format_table(
                ["Persisted", "Consensus", "Count"],
                hard_confusion_rows,
            )
        )
        confidence_subset_rows = []
        for row in confidence_summary.filter(pl.col("dimension") == value_name).to_dicts():
            confidence_subset_rows.append(
                [
                    row["confidence_tier"],
                    str(int(row["n_entries"])),
                    str(int(row["n_non_neutral"])),
                ]
            )
        lines.extend(["", "Confidence breakdown:", ""])
        lines.append(
            _format_table(
                ["Tier", "Entries", "Non-neutral entries"],
                confidence_subset_rows,
            )
        )
        lines.append("")

    fallback_rationale_count = joined_results.filter(
        pl.col("rationale_mismatch_count") > 0
    ).height
    max_rationale_mismatch = (
        int(joined_results["rationale_mismatch_count"].max())
        if joined_results.height > 0
        else 0
    )
    rationale_rows = []
    for row in (
        joined_results.group_by(["rationale_source_pass", "rationale_mismatch_count"])
        .agg(pl.len().alias("n_entries"))
        .sort(["rationale_source_pass", "rationale_mismatch_count"])
        .to_dicts()
    ):
        rationale_rows.append(
            [
                str(int(row["rationale_source_pass"])),
                str(int(row["rationale_mismatch_count"])),
                str(int(row["n_entries"])),
            ]
        )
    lines.extend(
        [
            "",
            "## 8. Rationale Source Summary",
            "",
            f"- Entries with a perfect 10/10 rationale-source match: `{manifest.height - fallback_rationale_count}`",
            f"- Entries using fallback rationale selection: `{fallback_rationale_count}`",
            f"- Maximum label mismatches on a chosen rationale source: `{max_rationale_mismatch}`",
            "",
        ]
    )
    lines.append(
        _format_table(
            ["Source pass", "Mismatch count", "Entries"],
            rationale_rows,
        )
    )

    migration_rows = []
    migration_summary = comparison_rows.filter(pl.col("label_changed")).group_by(
        ["dimension", "persisted_label", "consensus_label"]
    ).agg(pl.len().alias("n_entries")).sort(
        ["dimension", "persisted_label", "consensus_label"]
    )
    lines.extend(
        [
            "",
            "## 9. Label Migration Summary",
            "",
        ]
    )
    for row in migration_summary.to_dicts():
        migration_rows.append(
            [
                row["dimension"],
                str(int(row["persisted_label"])),
                str(int(row["consensus_label"])),
                str(int(row["n_entries"])),
            ]
        )
    lines.append(
        _format_table(
            ["Dimension", "Persisted", "Consensus", "Changed entries"],
            migration_rows,
        )
    )

    recommendation = (
        "Proceed to `twinkl-754.6` retraining."
        if gate_summary["overall_passed"]
        else "Stop after repeated-call diagnostics review; do not retrain until the gate is addressed."
    )
    lines.extend(
        [
            "",
            "## 10. Recommendation",
            "",
            recommendation,
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def summarize_bundle(
    bundle_dir: Path,
    *,
    annotations_dir: Path | None = None,
) -> tuple[
    str,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict,
]:
    manifest = _load_manifest(bundle_dir)
    bundle_status = load_bundle_status(bundle_dir)
    pass_provenance = _load_provenance_frame(bundle_dir, "pass_provenance.csv").sort(
        "pass_index"
    )
    pass_similarity = _load_provenance_frame(bundle_dir, "pass_similarity.csv").sort(
        ["left_pass_index", "right_pass_index"]
    )
    pass_frames: list[pl.DataFrame] = []
    pass_payloads: dict[int, dict[str, dict]] = {}

    for pass_index in range(1, N_PASSES + 1):
        pass_frame, payloads = _validate_pass_results(
            bundle_dir=bundle_dir,
            manifest=manifest,
            pass_index=pass_index,
        )
        pass_frames.append(pass_frame)
        pass_payloads[pass_index] = payloads
    _verify_pass_provenance(
        bundle_dir=bundle_dir,
        pass_payloads=pass_payloads,
        pass_provenance=pass_provenance,
        pass_similarity=pass_similarity,
    )

    consensus_frame, joined_results = _build_consensus_outputs(
        manifest=manifest,
        pass_payloads=pass_payloads,
    )
    for pass_frame in pass_frames:
        joined_results = joined_results.join(pass_frame, on="entry_id", how="left")
    comparison_rows = _build_comparison_rows(joined_results)
    flip_summary = _build_flip_summary(comparison_rows)
    confidence_summary = _build_confidence_summary(comparison_rows)

    persisted_wide = _build_persisted_wide(manifest)
    consensus_wide = _build_consensus_wide(consensus_frame)
    human_majority_wide = _load_human_majority_wide(
        annotations_dir or (ROOT / "logs" / "annotations")
    )
    irr_summary, _fleiss, consensus_vs_human, persisted_vs_human = _build_irr_summary(
        persisted_wide=persisted_wide,
        consensus_wide=consensus_wide,
        human_majority_wide=human_majority_wide,
        pass_payloads=pass_payloads,
    )
    gate_summary = _evaluate_gate(
        comparison_rows,
        consensus_vs_human=consensus_vs_human,
        persisted_vs_human=persisted_vs_human,
    )
    report = build_report(
        manifest=manifest,
        flip_summary=flip_summary,
        confidence_summary=confidence_summary,
        irr_summary=irr_summary,
        joined_results=joined_results,
        comparison_rows=comparison_rows,
        gate_summary=gate_summary,
        pass_provenance=pass_provenance,
        pass_similarity=pass_similarity,
        bundle_status=bundle_status,
    )
    return (
        report,
        consensus_frame,
        joined_results,
        comparison_rows,
        flip_summary,
        confidence_summary,
        irr_summary,
        gate_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize twinkl-754 consensus re-judging results."
    )
    parser.add_argument(
        "--bundle-dir",
        default="logs/exports/twinkl_754",
        help="Directory produced by twinkl_754_prepare_consensus.py.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional report output path. Defaults to <bundle-dir>/consensus_rejudging_report.md",
    )
    parser.add_argument(
        "--consensus-output",
        default="logs/judge_labels/consensus_labels.parquet",
        help="Consensus parquet output path.",
    )
    parser.add_argument(
        "--annotations-dir",
        default="logs/annotations",
        help="Directory containing the three human annotation parquet files.",
    )
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    report_path = (
        Path(args.output)
        if args.output
        else bundle_dir / "consensus_rejudging_report.md"
    )
    (
        report,
        consensus_frame,
        joined_results,
        comparison_rows,
        flip_summary,
        confidence_summary,
        irr_summary,
        _gate_summary,
    ) = summarize_bundle(
        bundle_dir,
        annotations_dir=Path(args.annotations_dir),
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    joined_results.write_csv(bundle_dir / "joined_results.csv")
    comparison_rows.write_csv(bundle_dir / "comparison_rows.csv")
    flip_summary.write_csv(bundle_dir / "flip_summary.csv")
    confidence_summary.write_csv(bundle_dir / "confidence_summary.csv")
    irr_summary.write_csv(bundle_dir / "irr_summary.csv")
    consolidate_consensus_labels(
        consensus_frame,
        output_path=Path(args.consensus_output),
    )
    bundle_status = load_bundle_status(bundle_dir)
    bundle_status.update(
        {
            "status": "summarized",
        }
    )
    write_bundle_status(bundle_dir, bundle_status)

    print(f"Wrote report: {report_path}")
    print(f"Wrote consensus parquet: {Path(args.consensus_output)}")


if __name__ == "__main__":
    main()
