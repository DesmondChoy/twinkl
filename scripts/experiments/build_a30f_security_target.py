#!/usr/bin/env python3
"""Materialize the twinkl-a30f sampled Security target audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vif.security_target import write_security_target_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--joined-results",
        type=Path,
        default=Path("logs/exports/twinkl_747/joined_results.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/experiments/artifacts/security_target_twinkl_a30f_20260711"),
    )
    args = parser.parse_args()
    target_path, summary_path = write_security_target_artifacts(
        joined_results_path=args.joined_results,
        output_dir=args.output_dir,
    )
    print(f"Wrote {target_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
