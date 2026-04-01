#!/usr/bin/env python3
"""Summarize twinkl-754 consensus re-judging results."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
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
LOW_CONFIDENCE_TIERS = ("bare_majority", "no_majority")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42
N_PASSES = 5
STABILITY_METRICS = (
    "n_entries",
    "n_non_neutral_entries",
    "non_unanimous_rate_all",
    "mean_vote_entropy_all",
    "polarity_flip_rate_all",
    "low_confidence_non_neutral_ratio",
    "non_unanimous_rate_non_neutral",
    "mean_vote_entropy_non_neutral",
    "polarity_flip_rate_non_neutral",
)
PERSONA_AGG_COLUMNS = (
    "n_entries",
    "n_non_neutral_entries",
    "non_unanimous_count_all",
    "vote_entropy_sum_all",
    "polarity_flip_count_all",
    "low_confidence_non_neutral_count",
    "non_unanimous_count_non_neutral",
    "vote_entropy_sum_non_neutral",
    "polarity_flip_count_non_neutral",
)


def _load_manifest(bundle_dir: Path) -> pl.DataFrame:
    manifest_path = bundle_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return pl.read_csv(manifest_path).sort(["persona_id", "t_index"])


def _manifest_entry_ids(manifest: pl.DataFrame) -> list[str]:
    return manifest["entry_id"].to_list()


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


def _empty_alignment_wide_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "persona_id": pl.Utf8,
            "t_index": pl.Int64,
            **{f"alignment_{value_name}": pl.Int64 for value_name in SCHWARTZ_VALUE_ORDER},
        }
    )


def _load_human_benchmark(
    annotations_dir: Path,
    *,
    full_corpus_entry_count: int,
) -> tuple[pl.DataFrame, dict[str, int]]:
    parquet_files = sorted(annotations_dir.glob("*.parquet"))
    summary = {
        "annotator_file_count": len(parquet_files),
        "union_entry_count": 0,
        "union_persona_count": 0,
        "strict_overlap_entry_count": 0,
        "strict_overlap_persona_count": 0,
        "single_annotated_entry_count": 0,
        "excluded_full_corpus_entry_count": full_corpus_entry_count,
    }
    if not parquet_files:
        return _empty_alignment_wide_frame(), summary

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
    union = annotations.select(["persona_id", "t_index"]).unique()
    strict_overlap = grouped.filter(pl.col("annotator_count") == expected_annotators)

    summary.update(
        {
            "union_entry_count": int(union.height),
            "union_persona_count": int(union.select("persona_id").n_unique()),
            "strict_overlap_entry_count": int(strict_overlap.height),
            "strict_overlap_persona_count": int(
                strict_overlap.select("persona_id").n_unique()
            ),
            "single_annotated_entry_count": int(
                grouped.filter(pl.col("annotator_count") == 1).height
            ),
            "excluded_full_corpus_entry_count": max(
                0,
                full_corpus_entry_count - int(strict_overlap.height),
            ),
        }
    )
    if strict_overlap.height == 0:
        return _empty_alignment_wide_frame(), summary

    majority_wide = strict_overlap.select(
        ["persona_id", "t_index"]
        + [
            pl.col(f"votes_{value_name}")
            .list.sort()
            .list.get(expected_annotators // 2)
            .alias(f"alignment_{value_name}")
            for value_name in SCHWARTZ_VALUE_ORDER
        ]
    ).sort(["persona_id", "t_index"])
    return majority_wide, summary


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
                "decision_role": "informational",
            }
        )
        rows.append(
            {
                "metric": "consensus_vs_persisted_cohen_kappa",
                "dimension": value_name,
                "value": consensus_vs_persisted.get(value_name),
                "reference": None,
                "decision_role": "informational",
            }
        )
        rows.append(
            {
                "metric": "consensus_vs_human_cohen_kappa",
                "dimension": value_name,
                "value": consensus_vs_human.get(value_name),
                "reference": persisted_vs_human.get(value_name),
                "decision_role": "advisory_only",
            }
        )
        rows.append(
            {
                "metric": "persisted_vs_human_cohen_kappa",
                "dimension": value_name,
                "value": persisted_vs_human.get(value_name),
                "reference": None,
                "decision_role": "advisory_only",
            }
        )

    rows.append(
        {
            "metric": "fleiss_n_shared",
            "dimension": "aggregate",
            "value": fleiss.get("n_shared"),
            "reference": None,
            "decision_role": "informational",
        }
    )
    return (
        pl.DataFrame(rows),
        fleiss,
        consensus_vs_human,
        persisted_vs_human,
    )


def _vote_entropy(votes: list[int]) -> float:
    counts = np.asarray(
        [
            sum(vote == -1 for vote in votes),
            sum(vote == 0 for vote in votes),
            sum(vote == 1 for vote in votes),
        ],
        dtype=np.float64,
    )
    probabilities = counts[counts > 0] / float(len(votes))
    return float(-(probabilities * np.log2(probabilities)).sum())


def _ratio_array(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    missing_if_zero: bool = False,
) -> np.ndarray:
    result = np.full(
        numerator.shape,
        np.nan if missing_if_zero else 0.0,
        dtype=np.float64,
    )
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _compute_metric_arrays(total_matrix: np.ndarray) -> dict[str, np.ndarray]:
    n_entries = total_matrix[:, 0]
    n_non_neutral_entries = total_matrix[:, 1]
    return {
        "n_entries": n_entries.astype(np.float64),
        "n_non_neutral_entries": n_non_neutral_entries.astype(np.float64),
        "non_unanimous_rate_all": _ratio_array(total_matrix[:, 2], n_entries),
        "mean_vote_entropy_all": _ratio_array(total_matrix[:, 3], n_entries),
        "polarity_flip_rate_all": _ratio_array(total_matrix[:, 4], n_entries),
        "low_confidence_non_neutral_ratio": _ratio_array(
            total_matrix[:, 5],
            n_non_neutral_entries,
            missing_if_zero=True,
        ),
        "non_unanimous_rate_non_neutral": _ratio_array(
            total_matrix[:, 6],
            n_non_neutral_entries,
            missing_if_zero=True,
        ),
        "mean_vote_entropy_non_neutral": _ratio_array(
            total_matrix[:, 7],
            n_non_neutral_entries,
            missing_if_zero=True,
        ),
        "polarity_flip_rate_non_neutral": _ratio_array(
            total_matrix[:, 8],
            n_non_neutral_entries,
            missing_if_zero=True,
        ),
    }


def _bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan"), float("nan")
    lower, upper = np.quantile(finite_values, [0.025, 0.975])
    return float(lower), float(upper)


def _build_stability_summary(comparison_rows: pl.DataFrame) -> pl.DataFrame:
    feature_rows: list[dict] = []
    for row in comparison_rows.sort(["dimension", "persona_id", "t_index"]).to_dicts():
        votes = [int(row[f"pass_{pass_index}_label"]) for pass_index in range(1, N_PASSES + 1)]
        non_neutral = int(row["consensus_label"]) != 0
        low_confidence = row["confidence_tier"] in LOW_CONFIDENCE_TIERS
        has_positive = any(vote == 1 for vote in votes)
        has_negative = any(vote == -1 for vote in votes)
        feature_rows.append(
            {
                "dimension": row["dimension"],
                "persona_id": row["persona_id"],
                "non_neutral": non_neutral,
                "low_confidence_non_neutral": non_neutral and low_confidence,
                "non_unanimous": len(set(votes)) > 1,
                "vote_entropy": _vote_entropy(votes),
                "polarity_flip": has_positive and has_negative,
            }
        )

    feature_frame = pl.DataFrame(feature_rows)
    persona_aggregates = feature_frame.group_by(["dimension", "persona_id"]).agg(
        [
            pl.len().alias("n_entries"),
            pl.col("non_neutral").sum().alias("n_non_neutral_entries"),
            pl.col("non_unanimous").sum().alias("non_unanimous_count_all"),
            pl.col("vote_entropy").sum().alias("vote_entropy_sum_all"),
            pl.col("polarity_flip").sum().alias("polarity_flip_count_all"),
            pl.col("low_confidence_non_neutral").sum().alias(
                "low_confidence_non_neutral_count"
            ),
            (pl.col("non_unanimous") & pl.col("non_neutral")).sum().alias(
                "non_unanimous_count_non_neutral"
            ),
            pl.when(pl.col("non_neutral"))
            .then(pl.col("vote_entropy"))
            .otherwise(0.0)
            .sum()
            .alias("vote_entropy_sum_non_neutral"),
            (pl.col("polarity_flip") & pl.col("non_neutral")).sum().alias(
                "polarity_flip_count_non_neutral"
            ),
        ]
    )

    persona_ids = sorted(persona_aggregates["persona_id"].unique().to_list())
    if not persona_ids:
        return pl.DataFrame(
            schema={
                "dimension": pl.Utf8,
                "difficulty_rank": pl.Int64,
                **{
                    f"{metric}_{suffix}": pl.Float64
                    for metric in STABILITY_METRICS
                    for suffix in ("point", "ci_lo", "ci_hi")
                },
            }
        )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0,
        len(persona_ids),
        size=(BOOTSTRAP_REPLICATES, len(persona_ids)),
    )

    rows: list[dict] = []
    for value_name in SCHWARTZ_VALUE_ORDER:
        dimension_aggregates = (
            pl.DataFrame({"persona_id": persona_ids})
            .join(
                persona_aggregates.filter(pl.col("dimension") == value_name),
                on="persona_id",
                how="left",
            )
            .with_columns(
                [pl.col(column).fill_null(0).alias(column) for column in PERSONA_AGG_COLUMNS]
            )
        )
        totals = (
            dimension_aggregates.select(list(PERSONA_AGG_COLUMNS))
            .to_numpy()
            .astype(np.float64)
        )
        observed_totals = totals.sum(axis=0, keepdims=True)
        bootstrap_totals = totals[bootstrap_indices].sum(axis=1)

        observed_metrics = _compute_metric_arrays(observed_totals)
        bootstrap_metrics = _compute_metric_arrays(bootstrap_totals)

        summary_row: dict[str, object] = {"dimension": value_name}
        for metric_name in STABILITY_METRICS:
            summary_row[f"{metric_name}_point"] = float(observed_metrics[metric_name][0])
            ci_lo, ci_hi = _bootstrap_interval(bootstrap_metrics[metric_name])
            summary_row[f"{metric_name}_ci_lo"] = ci_lo
            summary_row[f"{metric_name}_ci_hi"] = ci_hi
        rows.append(summary_row)

    ranked_rows = sorted(
        rows,
        key=lambda row: (
            not math.isfinite(float(row["mean_vote_entropy_non_neutral_point"])),
            -float(row["mean_vote_entropy_non_neutral_point"])
            if math.isfinite(float(row["mean_vote_entropy_non_neutral_point"]))
            else 0.0,
            str(row["dimension"]),
        ),
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["difficulty_rank"] = rank

    return pl.DataFrame(ranked_rows)


def _evaluate_gate(stability_summary: pl.DataFrame) -> dict:
    hard_rows: list[dict] = []
    stability_gate_passed = True

    for value_name in HARD_DIMENSIONS:
        row = stability_summary.filter(pl.col("dimension") == value_name).to_dicts()[0]
        ci_hi = float(row["low_confidence_non_neutral_ratio_ci_hi"])
        passes = math.isfinite(ci_hi) and ci_hi < 0.5
        stability_gate_passed = stability_gate_passed and passes
        hard_rows.append(
            {
                "dimension": value_name,
                "n_non_neutral_entries": int(round(float(row["n_non_neutral_entries_point"]))),
                "low_confidence_non_neutral_ratio_point": float(
                    row["low_confidence_non_neutral_ratio_point"]
                ),
                "low_confidence_non_neutral_ratio_ci_lo": float(
                    row["low_confidence_non_neutral_ratio_ci_lo"]
                ),
                "low_confidence_non_neutral_ratio_ci_hi": ci_hi,
                "non_unanimous_rate_non_neutral_point": float(
                    row["non_unanimous_rate_non_neutral_point"]
                ),
                "non_unanimous_rate_non_neutral_ci_lo": float(
                    row["non_unanimous_rate_non_neutral_ci_lo"]
                ),
                "non_unanimous_rate_non_neutral_ci_hi": float(
                    row["non_unanimous_rate_non_neutral_ci_hi"]
                ),
                "mean_vote_entropy_non_neutral_point": float(
                    row["mean_vote_entropy_non_neutral_point"]
                ),
                "mean_vote_entropy_non_neutral_ci_lo": float(
                    row["mean_vote_entropy_non_neutral_ci_lo"]
                ),
                "mean_vote_entropy_non_neutral_ci_hi": float(
                    row["mean_vote_entropy_non_neutral_ci_hi"]
                ),
                "polarity_flip_rate_non_neutral_point": float(
                    row["polarity_flip_rate_non_neutral_point"]
                ),
                "polarity_flip_rate_non_neutral_ci_lo": float(
                    row["polarity_flip_rate_non_neutral_ci_lo"]
                ),
                "polarity_flip_rate_non_neutral_ci_hi": float(
                    row["polarity_flip_rate_non_neutral_ci_hi"]
                ),
                "passes": passes,
            }
        )

    recommendation = (
        "Eligible for retrain comparison under full-corpus stability criteria."
        if stability_gate_passed
        else "Hold retrain until full-corpus stability improves."
    )
    return {
        "stability_gate_passed": stability_gate_passed,
        "overall_passed": stability_gate_passed,
        "hard_dimension_rows": hard_rows,
        "recommendation": recommendation,
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


def _fmt_count(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int | float):
        if math.isnan(float(value)):
            return "N/A"
        return str(int(round(float(value))))
    return str(value)


def _fmt_interval(point: object, ci_lo: object, ci_hi: object) -> str:
    return f"{_fmt_float(point)} [{_fmt_float(ci_lo)}, {_fmt_float(ci_hi)}]"


def build_report(
    *,
    manifest: pl.DataFrame,
    flip_summary: pl.DataFrame,
    confidence_summary: pl.DataFrame,
    irr_summary: pl.DataFrame,
    stability_summary: pl.DataFrame,
    joined_results: pl.DataFrame,
    comparison_rows: pl.DataFrame,
    gate_summary: dict,
    human_benchmark_summary: dict[str, int],
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
            f"- Stability bootstrap: `{BOOTSTRAP_REPLICATES}` persona-cluster resamples, seed `{BOOTSTRAP_SEED}`",
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
            "## 3. Human-Overlap Benchmark (Advisory)",
            "",
            (
                "These kappas are a limited non-expert human-overlap benchmark. "
                "They are advisory only and do not act as the hard retrain gate."
            ),
            "",
            f"- Annotator files loaded: `{human_benchmark_summary['annotator_file_count']}`",
            (
                "- Union coverage: "
                f"`{human_benchmark_summary['union_entry_count']}` unique annotated entries across "
                f"`{human_benchmark_summary['union_persona_count']}` personas"
            ),
            (
                "- Strict 3-way overlap used for comparison: "
                f"`{human_benchmark_summary['strict_overlap_entry_count']}` entries across "
                f"`{human_benchmark_summary['strict_overlap_persona_count']}` personas"
            ),
            (
                "- Singly annotated entries excluded from majority aggregation: "
                f"`{human_benchmark_summary['single_annotated_entry_count']}`"
            ),
            (
                "- Full-corpus entries outside the overlap excluded from the human benchmark: "
                f"`{human_benchmark_summary['excluded_full_corpus_entry_count']}`"
            ),
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
            [
                "Dimension",
                "Consensus vs human overlap (advisory)",
                "Persisted vs human overlap (advisory)",
            ],
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
            "## 6. Full-Corpus Stability Gate",
            "",
            (
                "This is the hard retrain gate. It uses full-corpus stability only: "
                "the upper 95% CI of `low_confidence_non_neutral_ratio` must stay below "
                "`0.5` for `security`, `hedonism`, and `stimulation`."
            ),
            "",
            (
                "The human-overlap benchmark above remains advisory and limited-sample; "
                "it is not used in this go/no-go decision."
            ),
            "",
            f"- Full-corpus stability gate passed: `{gate_summary['stability_gate_passed']}`",
            f"- Retrain readiness summary: `{gate_summary['recommendation']}`",
            "",
        ]
    )
    hard_rows = [
        [
            row["dimension"],
            str(int(row["n_non_neutral_entries"])),
            _fmt_interval(
                row["low_confidence_non_neutral_ratio_point"],
                row["low_confidence_non_neutral_ratio_ci_lo"],
                row["low_confidence_non_neutral_ratio_ci_hi"],
            ),
            _fmt_interval(
                row["mean_vote_entropy_non_neutral_point"],
                row["mean_vote_entropy_non_neutral_ci_lo"],
                row["mean_vote_entropy_non_neutral_ci_hi"],
            ),
            str(row["passes"]),
        ]
        for row in gate_summary["hard_dimension_rows"]
    ]
    lines.append(
        _format_table(
            [
                "Dimension",
                "Non-neutral labels",
                "Low-confidence ratio (95% CI)",
                "Mean vote entropy (95% CI)",
                "Passes",
            ],
            hard_rows,
        )
    )
    lines.extend(
        [
            "",
            "## 7. Per-Dimension Stability",
            "",
            (
                "Dimensions are ranked by `mean_vote_entropy_non_neutral` point estimate "
                "from highest to lowest. The 95% CIs below show uncertainty around the "
                "estimate; they are not the difficulty ranking itself."
            ),
            "",
        ]
    )
    stability_rows = []
    for row in stability_summary.sort("difficulty_rank").to_dicts():
        stability_rows.append(
            [
                str(int(row["difficulty_rank"])),
                row["dimension"],
                _fmt_count(row["n_non_neutral_entries_point"]),
                _fmt_interval(
                    row["mean_vote_entropy_non_neutral_point"],
                    row["mean_vote_entropy_non_neutral_ci_lo"],
                    row["mean_vote_entropy_non_neutral_ci_hi"],
                ),
                _fmt_interval(
                    row["non_unanimous_rate_non_neutral_point"],
                    row["non_unanimous_rate_non_neutral_ci_lo"],
                    row["non_unanimous_rate_non_neutral_ci_hi"],
                ),
                _fmt_interval(
                    row["polarity_flip_rate_non_neutral_point"],
                    row["polarity_flip_rate_non_neutral_ci_lo"],
                    row["polarity_flip_rate_non_neutral_ci_hi"],
                ),
                _fmt_interval(
                    row["low_confidence_non_neutral_ratio_point"],
                    row["low_confidence_non_neutral_ratio_ci_lo"],
                    row["low_confidence_non_neutral_ratio_ci_hi"],
                ),
            ]
        )
    lines.append(
        _format_table(
            [
                "Rank",
                "Dimension",
                "Non-neutral labels",
                "Mean vote entropy (95% CI)",
                "Non-unanimous rate (95% CI)",
                "Polarity-flip rate (95% CI)",
                "Low-confidence ratio (95% CI)",
            ],
            stability_rows,
        )
    )
    lines.extend(["", "Hard-dimension callouts:", ""])
    for row in gate_summary["hard_dimension_rows"]:
        lines.append(
            (
                f"- `{row['dimension']}`: low-confidence ratio "
                f"`{_fmt_interval(row['low_confidence_non_neutral_ratio_point'], row['low_confidence_non_neutral_ratio_ci_lo'], row['low_confidence_non_neutral_ratio_ci_hi'])}`; "
                f"mean vote entropy "
                f"`{_fmt_interval(row['mean_vote_entropy_non_neutral_point'], row['mean_vote_entropy_non_neutral_ci_lo'], row['mean_vote_entropy_non_neutral_ci_hi'])}`; "
                f"gate pass=`{row['passes']}`"
            )
        )
    lines.extend(
        [
            "",
            "## 8. Hard-Dimension Deep Dive",
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
                    "Full-corpus stability first: "
                    f"`{hard_gate_row['n_non_neutral_entries']}` non-neutral labels, "
                    "mean vote entropy "
                    f"`{_fmt_interval(hard_gate_row['mean_vote_entropy_non_neutral_point'], hard_gate_row['mean_vote_entropy_non_neutral_ci_lo'], hard_gate_row['mean_vote_entropy_non_neutral_ci_hi'])}`, "
                    "non-unanimous rate "
                    f"`{_fmt_interval(hard_gate_row['non_unanimous_rate_non_neutral_point'], hard_gate_row['non_unanimous_rate_non_neutral_ci_lo'], hard_gate_row['non_unanimous_rate_non_neutral_ci_hi'])}`, "
                    "polarity-flip rate "
                    f"`{_fmt_interval(hard_gate_row['polarity_flip_rate_non_neutral_point'], hard_gate_row['polarity_flip_rate_non_neutral_ci_lo'], hard_gate_row['polarity_flip_rate_non_neutral_ci_hi'])}`, "
                    "and low-confidence ratio "
                    f"`{_fmt_interval(hard_gate_row['low_confidence_non_neutral_ratio_point'], hard_gate_row['low_confidence_non_neutral_ratio_ci_lo'], hard_gate_row['low_confidence_non_neutral_ratio_ci_hi'])}` "
                    f"(gate pass=`{hard_gate_row['passes']}`)."
                ),
                "",
                (
                    "Secondary advisory benchmark: persisted-vs-consensus kappa "
                    f"`{_fmt_float(persisted_kappa['value'])}`; "
                    "consensus-vs-human overlap kappa "
                    f"`{_fmt_float(human_kappa['value'])}` on the strict "
                    f"`{human_benchmark_summary['strict_overlap_entry_count']}`-entry advisory subset."
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
            "## 9. Rationale Source Summary",
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
            "## 10. Label Migration Summary",
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

    lines.extend(
        [
            "",
            "## 11. Recommendation",
            "",
            gate_summary["recommendation"],
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
    stability_summary = _build_stability_summary(comparison_rows)

    persisted_wide = _build_persisted_wide(manifest)
    consensus_wide = _build_consensus_wide(consensus_frame)
    human_majority_wide, human_benchmark_summary = _load_human_benchmark(
        annotations_dir or (ROOT / "logs" / "annotations"),
        full_corpus_entry_count=manifest.height,
    )
    irr_summary, _fleiss, _consensus_vs_human, _persisted_vs_human = _build_irr_summary(
        persisted_wide=persisted_wide,
        consensus_wide=consensus_wide,
        human_majority_wide=human_majority_wide,
        pass_payloads=pass_payloads,
    )
    gate_summary = _evaluate_gate(stability_summary)
    report = build_report(
        manifest=manifest,
        flip_summary=flip_summary,
        confidence_summary=confidence_summary,
        irr_summary=irr_summary,
        stability_summary=stability_summary,
        joined_results=joined_results,
        comparison_rows=comparison_rows,
        gate_summary=gate_summary,
        human_benchmark_summary=human_benchmark_summary,
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
        stability_summary,
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
        stability_summary,
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
    stability_summary.write_csv(bundle_dir / "stability_summary.csv")
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
