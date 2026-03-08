#!/usr/bin/env python3
"""Freeze the pre-generation holdout and baseline snapshot for twinkl-681.5."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yaml

_DIR = Path.cwd()
while _DIR != _DIR.parent:
    if (_DIR / "src").is_dir() and (_DIR / "pyproject.toml").is_file():
        os.chdir(_DIR)
        break
    _DIR = _DIR.parent
sys.path.insert(0, os.getcwd())

from src.synthetic.batch_verification import load_yaml_file
from src.vif.dataset import load_all_data, split_by_persona


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the pre-generation holdout and baseline snapshot for twinkl-681.5."
    )
    parser.add_argument(
        "--config",
        default="config/experiments/vif/twinkl_681_5.yaml",
        help="Path to the twinkl-681.5 experiment config.",
    )
    args = parser.parse_args()

    config = load_yaml_file(args.config)
    baseline = config["baseline"]

    labels_df, entries_df = load_all_data(
        baseline["labels_path"],
        baseline["wrangled_dir"],
    )
    train_df, val_df, test_df = split_by_persona(
        labels_df,
        entries_df,
        train_ratio=float(baseline["train_ratio"]),
        val_ratio=float(baseline["val_ratio"]),
        seed=int(baseline["split_seed"]),
    )

    registry = pl.read_parquet(baseline["registry_path"])
    snapshot_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_persona_count": int(registry.height),
        "registry_persona_ids": sorted(registry.get_column("persona_id").to_list()),
        "synthetic_persona_file_count": len(
            list(Path(baseline["synthetic_dir"]).glob("persona_*.md"))
        ),
    }

    holdout_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_seed": int(baseline["split_seed"]),
        "train_ratio": float(baseline["train_ratio"]),
        "val_ratio": float(baseline["val_ratio"]),
        "source_persona_count": int(registry.height),
        "train_persona_ids": _sorted_persona_ids(train_df),
        "val_persona_ids": _sorted_persona_ids(val_df),
        "test_persona_ids": _sorted_persona_ids(test_df),
    }

    snapshot_path = Path(baseline["snapshot_path"])
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        yaml.safe_dump(snapshot_payload, sort_keys=False),
        encoding="utf-8",
    )

    holdout_path = Path(baseline["holdout_manifest_path"])
    holdout_path.parent.mkdir(parents=True, exist_ok=True)
    holdout_path.write_text(
        yaml.safe_dump(holdout_payload, sort_keys=False),
        encoding="utf-8",
    )

    print(f"Wrote snapshot: {snapshot_path}")
    print(f"Wrote holdout manifest: {holdout_path}")
    print(
        "Split sizes: "
        f"train={len(holdout_payload['train_persona_ids'])}, "
        f"val={len(holdout_payload['val_persona_ids'])}, "
        f"test={len(holdout_payload['test_persona_ids'])}"
    )
    return 0


def _sorted_persona_ids(df: pl.DataFrame) -> list[str]:
    return sorted(df.select("persona_id").unique().to_series().to_list())


if __name__ == "__main__":
    raise SystemExit(main())
