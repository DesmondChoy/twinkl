#!/usr/bin/env python3
"""Validate shard results, merge pass files, and persist provenance."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judge.consensus_utils import (
    aggregate_rationale_coverage,
    compute_score_only_hash,
    count_entry_vector_differences,
    hash_file,
    load_bundle_status,
    load_expected_ids,
    load_manifest_ids,
    read_jsonl,
    validate_result_rows_against_ids,
    write_bundle_status,
    write_jsonl,
)

N_PASSES = 5


def _load_shard_manifest(bundle_dir: Path) -> pl.DataFrame:
    shard_manifest_path = bundle_dir / "shard_manifest.csv"
    if not shard_manifest_path.exists():
        raise FileNotFoundError(f"Missing shard manifest: {shard_manifest_path}")
    return pl.read_csv(shard_manifest_path).sort(["pass_index", "shard_id"])


def _load_attempt_overrides(
    attempt_manifest_path: Path | None,
) -> dict[tuple[int, str], dict[str, object]]:
    if attempt_manifest_path is None:
        return {}
    if not attempt_manifest_path.exists():
        raise FileNotFoundError(f"Missing attempt manifest: {attempt_manifest_path}")

    overrides: dict[tuple[int, str], dict[str, object]] = {}
    with attempt_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {"pass_index", "shard_id", "attempt"}
        if not required_fields.issubset(reader.fieldnames or []):
            raise ValueError(
                "Attempt manifest must contain pass_index, shard_id, and attempt columns."
            )
        for row in reader:
            key = (int(row["pass_index"]), row["shard_id"])
            overrides[key] = {
                "attempt": int(row["attempt"]),
                "worker_model": row.get("worker_model") or None,
            }
    return overrides


def _completion_timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def merge_pass_results(
    bundle_dir: Path,
    *,
    worker_model: str = "gpt-5.4",
    attempt_manifest_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    manifest_path = bundle_dir / "manifest.csv"
    valid_manifest_ids = load_manifest_ids(manifest_path)
    shard_manifest = _load_shard_manifest(bundle_dir)
    attempt_overrides = _load_attempt_overrides(attempt_manifest_path)

    provenance_dir = bundle_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    shard_rows: list[dict] = []
    pass_rows: list[dict] = []
    pass_payloads: dict[int, list[dict]] = {}

    for pass_index in range(1, N_PASSES + 1):
        pass_name = f"pass_{pass_index}"
        pass_prompt_path = bundle_dir / "prompts" / f"{pass_name}.jsonl"
        pass_expected_ids = load_expected_ids(pass_prompt_path)
        pass_row_payloads: list[dict] = []
        pass_completion_timestamps: list[str] = []
        pass_attempts: list[int] = []
        pass_worker_models: set[str] = set()

        for shard_row in shard_manifest.filter(pl.col("pass_index") == pass_index).to_dicts():
            prompt_path = Path(shard_row["prompt_path"])
            result_path = Path(shard_row["result_path"])
            expected_entry_ids = load_expected_ids(prompt_path)
            normalized_rows, coverage = validate_result_rows_against_ids(
                read_jsonl(result_path),
                valid_manifest_ids=valid_manifest_ids,
                expected_entry_ids=expected_entry_ids,
            )

            override = attempt_overrides.get((pass_index, shard_row["shard_id"]), {})
            attempt = int(override.get("attempt") or 1)
            shard_worker_model = str(override.get("worker_model") or worker_model)
            completion_timestamp = _completion_timestamp(result_path)

            shard_rows.append(
                {
                    "pass_index": pass_index,
                    "shard_id": shard_row["shard_id"],
                    "attempt": attempt,
                    "worker_model": shard_worker_model,
                    "completion_timestamp": completion_timestamp,
                    "prompt_sha256": hash_file(prompt_path),
                    "raw_result_sha256": hash_file(result_path),
                    "score_only_sha256": compute_score_only_hash(normalized_rows),
                    "non_zero_score_count": coverage.non_zero_score_count,
                    "non_zero_rationale_count": coverage.non_zero_rationale_count,
                    "non_zero_rationale_coverage": coverage.coverage,
                }
            )
            pass_row_payloads.extend(normalized_rows)
            pass_completion_timestamps.append(completion_timestamp)
            pass_attempts.append(attempt)
            pass_worker_models.add(shard_worker_model)

        expected_order = {entry_id: index for index, entry_id in enumerate(pass_expected_ids)}
        observed_ids = [row["entry_id"] for row in pass_row_payloads]
        if set(observed_ids) != set(pass_expected_ids):
            missing = sorted(set(pass_expected_ids) - set(observed_ids))
            extra = sorted(set(observed_ids) - set(pass_expected_ids))
            raise ValueError(
                f"{pass_name} merged shards do not match pass prompt. "
                f"Missing={missing[:5]}, extra={extra[:5]}"
            )

        ordered_rows = sorted(
            pass_row_payloads,
            key=lambda row: expected_order[row["entry_id"]],
        )
        pass_results_path = bundle_dir / "results" / f"{pass_name}_results.jsonl"
        write_jsonl(pass_results_path, ordered_rows)

        pass_coverage = aggregate_rationale_coverage(ordered_rows)
        pass_rows.append(
            {
                "pass_index": pass_index,
                "pass_name": pass_name,
                "attempt": max(pass_attempts) if pass_attempts else 1,
                "worker_model": ",".join(sorted(pass_worker_models)) or worker_model,
                "completion_timestamp": max(pass_completion_timestamps)
                if pass_completion_timestamps
                else "",
                "prompt_sha256": hash_file(pass_prompt_path),
                "raw_result_sha256": hash_file(pass_results_path),
                "score_only_sha256": compute_score_only_hash(ordered_rows),
                "non_zero_score_count": pass_coverage.non_zero_score_count,
                "non_zero_rationale_count": pass_coverage.non_zero_rationale_count,
                "non_zero_rationale_coverage": pass_coverage.coverage,
            }
        )
        pass_payloads[pass_index] = ordered_rows

    similarity_rows: list[dict] = []
    for left_pass in range(1, N_PASSES + 1):
        left_row = next(row for row in pass_rows if row["pass_index"] == left_pass)
        left_payloads = pass_payloads[left_pass]
        for right_pass in range(left_pass + 1, N_PASSES + 1):
            right_row = next(row for row in pass_rows if row["pass_index"] == right_pass)
            right_payloads = pass_payloads[right_pass]
            differing_entry_vectors = count_entry_vector_differences(
                left_payloads,
                right_payloads,
            )
            similarity_rows.append(
                {
                    "left_pass_index": left_pass,
                    "right_pass_index": right_pass,
                    "left_pass_name": f"pass_{left_pass}",
                    "right_pass_name": f"pass_{right_pass}",
                    "raw_hash_match": (
                        left_row["raw_result_sha256"] == right_row["raw_result_sha256"]
                    ),
                    "score_hash_match": (
                        left_row["score_only_sha256"] == right_row["score_only_sha256"]
                    ),
                    "differing_entry_vectors": differing_entry_vectors,
                    "identical_entry_vectors": len(pass_payloads[left_pass])
                    - differing_entry_vectors,
                }
            )

    shard_provenance = pl.DataFrame(shard_rows).sort(["pass_index", "shard_id"])
    pass_provenance = pl.DataFrame(pass_rows).sort("pass_index")
    pass_similarity = pl.DataFrame(similarity_rows).sort(
        ["left_pass_index", "right_pass_index"]
    )

    shard_provenance.write_csv(provenance_dir / "shard_provenance.csv")
    pass_provenance.write_csv(provenance_dir / "pass_provenance.csv")
    pass_similarity.write_csv(provenance_dir / "pass_similarity.csv")

    low_coverage = pass_provenance.filter(
        pl.col("non_zero_rationale_coverage") < 1.0
    )
    if low_coverage.height > 0:
        failing_passes = ", ".join(low_coverage["pass_name"].to_list())
        raise ValueError(
            "All non-zero scores must have rationales before consensus aggregation. "
            f"Failing passes: {failing_passes}"
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
            "Duplicate pass outputs detected before summarization: "
            + formatted_pairs
        )

    bundle_status = load_bundle_status(bundle_dir)
    bundle_status.update(
        {
            "status": "merged_validated",
            "invalidated": False,
            "warning": "",
            "merged_at": datetime.now(tz=UTC).isoformat(),
        }
    )
    write_bundle_status(bundle_dir, bundle_status)

    return shard_provenance, pass_provenance, pass_similarity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate shard results, merge pass files, and persist provenance."
    )
    parser.add_argument(
        "--bundle-dir",
        default="logs/exports/twinkl_754",
        help="Directory produced by twinkl_754_prepare_consensus.py.",
    )
    parser.add_argument(
        "--worker-model",
        default="gpt-5.4",
        help="Default worker model to record when no attempt manifest is provided.",
    )
    parser.add_argument(
        "--attempt-manifest",
        default="",
        help=(
            "Optional CSV with pass_index, shard_id, attempt, and worker_model columns "
            "for accepted shard results."
        ),
    )
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    shard_provenance, pass_provenance, pass_similarity = merge_pass_results(
        bundle_dir,
        worker_model=args.worker_model,
        attempt_manifest_path=Path(args.attempt_manifest)
        if args.attempt_manifest
        else None,
    )

    print(
        "Wrote shard provenance: "
        f"{bundle_dir / 'provenance' / 'shard_provenance.csv'}"
    )
    print(
        "Wrote pass provenance: "
        f"{bundle_dir / 'provenance' / 'pass_provenance.csv'}"
    )
    print(
        "Wrote pass similarity: "
        f"{bundle_dir / 'provenance' / 'pass_similarity.csv'}"
    )
    print(f"Validated shards: {shard_provenance.height}")
    print(f"Merged passes: {pass_provenance.height}")
    print(f"Compared pass pairs: {pass_similarity.height}")


if __name__ == "__main__":
    main()
