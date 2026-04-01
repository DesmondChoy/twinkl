#!/usr/bin/env python3
"""Validate twinkl-754 shard or pass result files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.judge.consensus_utils import (
    RationaleCoverage,
    load_expected_ids,
    load_manifest_ids,
    read_jsonl,
    validate_result_rows_against_ids,
    write_jsonl,
)


def validate_result_rows(
    rows: list[dict],
    *,
    valid_manifest_ids: set[str],
    expected_entry_ids: list[str],
) -> list[dict]:
    normalized_rows, _ = validate_result_rows_with_stats(
        rows,
        valid_manifest_ids=valid_manifest_ids,
        expected_entry_ids=expected_entry_ids,
    )
    return normalized_rows


def validate_result_rows_with_stats(
    rows: list[dict],
    *,
    valid_manifest_ids: set[str],
    expected_entry_ids: list[str],
) -> tuple[list[dict], RationaleCoverage]:
    return validate_result_rows_against_ids(
        rows,
        valid_manifest_ids=valid_manifest_ids,
        expected_entry_ids=expected_entry_ids,
    )


def validate_result_file(
    *,
    manifest_path: Path,
    expected_jsonl_path: Path,
    results_path: Path,
) -> list[dict]:
    valid_manifest_ids = load_manifest_ids(manifest_path)
    expected_entry_ids = load_expected_ids(expected_jsonl_path)
    rows = read_jsonl(results_path)
    return validate_result_rows(
        rows,
        valid_manifest_ids=valid_manifest_ids,
        expected_entry_ids=expected_entry_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate twinkl-754 shard or pass result files."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to logs/exports/twinkl_754/manifest.csv.",
    )
    parser.add_argument(
        "--expected-jsonl",
        required=True,
        help="Prompt JSONL file that defines the exact expected entry_ids.",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Result JSONL file to validate.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the normalized validated JSONL.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    expected_jsonl_path = Path(args.expected_jsonl)
    results_path = Path(args.results)
    normalized_rows, coverage = validate_result_rows_with_stats(
        read_jsonl(results_path),
        valid_manifest_ids=load_manifest_ids(manifest_path),
        expected_entry_ids=load_expected_ids(expected_jsonl_path),
    )
    if args.output:
        write_jsonl(Path(args.output), normalized_rows)

    print(f"Validated rows: {len(normalized_rows)}")
    print(f"Expected source: {args.expected_jsonl}")
    print(f"Results file: {args.results}")
    print(
        "Non-zero rationale coverage: "
        f"{coverage.non_zero_rationale_count}/{coverage.non_zero_score_count}"
    )


if __name__ == "__main__":
    main()
