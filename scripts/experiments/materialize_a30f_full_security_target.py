#!/usr/bin/env python3
# ruff: noqa: E402
"""Materialize the receipt-bound full-corpus a30f Security target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vif.security_target import (
    write_full_corpus_security_target_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-labels",
        type=Path,
        default=Path("logs/judge_labels/judge_labels.parquet"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "logs/exports/twinkl_a30f_active_critic_state_full_v1/active_critic_state_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("logs/exports/twinkl_a30f_active_critic_state_full_v1/results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/exports/twinkl_a30f_security_target_full_v1"),
    )
    args = parser.parse_args()
    target, labels, summary = write_full_corpus_security_target_artifacts(
        base_labels_path=args.base_labels,
        active_state_manifest_path=args.manifest,
        review_pass_paths={
            index: args.results_dir / f"pass_{index}_results.jsonl"
            for index in (1, 2, 3)
        },
        tiebreak_results_path=args.results_dir / "tiebreak_results.jsonl",
        output_dir=args.output_dir,
    )
    print(f"Wrote target: {target}")
    print(f"Wrote labels: {labels}")
    print(f"Wrote summary: {summary}")


if __name__ == "__main__":
    main()
