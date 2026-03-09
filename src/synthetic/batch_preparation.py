"""Helpers for preparing targeted synthetic-batch baselines."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import yaml

from src.vif.dataset import load_all_data, split_by_persona
from src.vif.holdout import load_holdout_manifest


def build_snapshot_payload(
    *,
    registry_path: str | Path,
    synthetic_dir: str | Path,
) -> dict:
    """Build a pre-generation snapshot payload from the current workspace state."""
    registry = pl.read_parquet(registry_path)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_persona_count": int(registry.height),
        "registry_persona_ids": sorted(registry.get_column("persona_id").to_list()),
        "synthetic_persona_file_count": len(list(Path(synthetic_dir).glob("persona_*.md"))),
    }


def build_holdout_payload(
    *,
    labels_path: str | Path,
    wrangled_dir: str | Path,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    source_persona_count: int,
) -> dict:
    """Build a new holdout manifest payload from the current labeled data."""
    labels_df, entries_df = load_all_data(labels_path, wrangled_dir)
    train_df, val_df, test_df = split_by_persona(
        labels_df,
        entries_df,
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        seed=int(split_seed),
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_seed": int(split_seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "source_persona_count": int(source_persona_count),
        "train_persona_ids": _sorted_persona_ids(train_df),
        "val_persona_ids": _sorted_persona_ids(val_df),
        "test_persona_ids": _sorted_persona_ids(test_df),
    }


def prepare_baseline_artifacts(baseline: dict) -> tuple[dict, dict, bool]:
    """Prepare snapshot data and either reuse or rebuild the holdout manifest."""
    snapshot_payload = build_snapshot_payload(
        registry_path=baseline["registry_path"],
        synthetic_dir=baseline["synthetic_dir"],
    )
    reuse_existing_holdout = bool(baseline.get("reuse_existing_holdout", False))
    holdout_manifest_path = baseline["holdout_manifest_path"]
    if reuse_existing_holdout:
        holdout_payload = load_holdout_manifest(holdout_manifest_path)
        return snapshot_payload, holdout_payload, False

    holdout_payload = build_holdout_payload(
        labels_path=baseline["labels_path"],
        wrangled_dir=baseline["wrangled_dir"],
        split_seed=int(baseline["split_seed"]),
        train_ratio=float(baseline["train_ratio"]),
        val_ratio=float(baseline["val_ratio"]),
        source_persona_count=int(snapshot_payload["registry_persona_count"]),
    )
    return snapshot_payload, holdout_payload, True


def write_yaml_payload(path: str | Path, payload: dict) -> Path:
    """Write a YAML payload to disk, creating parent directories when needed."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path


def _sorted_persona_ids(df: pl.DataFrame) -> list[str]:
    return sorted(df.select("persona_id").unique().to_series().to_list())
