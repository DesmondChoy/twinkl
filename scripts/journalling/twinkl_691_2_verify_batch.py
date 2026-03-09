#!/usr/bin/env python3
"""Verify the manually generated twinkl-691.2 targeted synthetic batch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

_DIR = Path.cwd()
while _DIR != _DIR.parent:
    if (_DIR / "src").is_dir() and (_DIR / "pyproject.toml").is_file():
        os.chdir(_DIR)
        break
    _DIR = _DIR.parent
sys.path.insert(0, os.getcwd())

from src.synthetic.batch_verification import (
    load_baseline_persona_ids,
    load_yaml_file,
    render_spot_check_report,
    verify_targeted_batch,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the manually generated targeted synthetic batch for twinkl-691.2."
    )
    parser.add_argument(
        "--config",
        default="config/experiments/vif/twinkl_691_2.yaml",
        help="Path to the twinkl-691.2 experiment config.",
    )
    args = parser.parse_args()

    config = load_yaml_file(args.config)
    snapshot_path = config["baseline"]["snapshot_path"]
    verification = config["verification"]

    summary = verify_targeted_batch(
        baseline_persona_ids=load_baseline_persona_ids(snapshot_path),
        registry_path=verification["registry_path"],
        synthetic_dir=verification["synthetic_dir"],
        required_targets=list(verification["required_targets"]),
        expected_new_persona_count=int(verification["expected_new_persona_count"]),
        expected_min_personas_per_target=int(
            verification["expected_min_personas_per_target"]
        ),
        min_entries=int(verification["min_entries"]),
        max_entries=int(verification["max_entries"]),
        require_unsettled_entries=bool(verification["require_unsettled_entries"]),
        require_non_unsettled_entries=bool(
            verification.get("require_non_unsettled_entries", False)
        ),
        required_core_value_pairs=dict(
            verification.get("required_core_value_pairs", {})
        ),
    )

    summary_path = Path(verification["summary_output_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    spot_check_path = Path(verification["spot_check_output_path"])
    spot_check_path.parent.mkdir(parents=True, exist_ok=True)
    spot_check_path.write_text(
        render_spot_check_report(
            summary,
            title=verification.get("report_title", "twinkl-691.2 Generation Spot Check"),
        ),
        encoding="utf-8",
    )

    print(f"Accepted: {'yes' if summary['accepted'] else 'no'}")
    print(f"New personas: {len(summary['new_persona_ids'])}")
    print(f"Target counts: {summary['target_counts']}")
    print(f"Summary: {summary_path}")
    print(f"Spot check: {spot_check_path}")

    if summary["failures"]:
        print("\nFailures:")
        for failure in summary["failures"]:
            print(f"- {failure}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
