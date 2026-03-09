#!/usr/bin/env python3
"""Prepare the frozen-holdout baseline snapshot for twinkl-691.2."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_DIR = Path.cwd()
while _DIR != _DIR.parent:
    if (_DIR / "src").is_dir() and (_DIR / "pyproject.toml").is_file():
        os.chdir(_DIR)
        break
    _DIR = _DIR.parent
sys.path.insert(0, os.getcwd())

from src.synthetic.batch_preparation import (
    prepare_baseline_artifacts,
    write_yaml_payload,
)
from src.synthetic.batch_verification import load_yaml_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen-holdout baseline snapshot for twinkl-691.2."
    )
    parser.add_argument(
        "--config",
        default="config/experiments/vif/twinkl_691_2.yaml",
        help="Path to the twinkl-691.2 experiment config.",
    )
    args = parser.parse_args()

    config = load_yaml_file(args.config)
    baseline = config["baseline"]
    snapshot_payload, holdout_payload, wrote_holdout = prepare_baseline_artifacts(baseline)
    snapshot_path = write_yaml_payload(baseline["snapshot_path"], snapshot_payload)
    holdout_path = Path(baseline["holdout_manifest_path"])
    if wrote_holdout:
        holdout_path = write_yaml_payload(holdout_path, holdout_payload)

    print(f"Wrote snapshot: {snapshot_path}")
    print(
        f"{'Wrote' if wrote_holdout else 'Reused'} holdout manifest: {holdout_path}"
    )
    print(
        "Split sizes: "
        f"train={len(holdout_payload['train_persona_ids'])}, "
        f"val={len(holdout_payload['val_persona_ids'])}, "
        f"test={len(holdout_payload['test_persona_ids'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
