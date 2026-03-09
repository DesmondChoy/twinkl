"""Tests for targeted synthetic-batch baseline preparation helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.synthetic.batch_preparation import (
    build_holdout_payload,
    build_snapshot_payload,
    prepare_baseline_artifacts,
    write_yaml_payload,
)


def test_build_snapshot_payload_reads_registry_and_synthetic_dir(tmp_path: Path):
    registry_path = tmp_path / "personas.parquet"
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    (synthetic_dir / "persona_aaa11111.md").write_text("one", encoding="utf-8")
    (synthetic_dir / "persona_bbb22222.md").write_text("two", encoding="utf-8")

    pl.DataFrame(
        {
            "persona_id": ["bbb22222", "aaa11111"],
            "name": ["B", "A"],
        }
    ).write_parquet(registry_path)

    payload = build_snapshot_payload(
        registry_path=registry_path,
        synthetic_dir=synthetic_dir,
    )

    assert payload["registry_persona_count"] == 2
    assert payload["registry_persona_ids"] == ["aaa11111", "bbb22222"]
    assert payload["synthetic_persona_file_count"] == 2


@patch("src.synthetic.batch_preparation.split_by_persona")
@patch("src.synthetic.batch_preparation.load_all_data")
def test_build_holdout_payload_uses_splitter(mock_load_all_data, mock_split_by_persona):
    mock_load_all_data.return_value = (pl.DataFrame({"dummy": [1]}), pl.DataFrame({"dummy": [2]}))
    mock_split_by_persona.return_value = (
        pl.DataFrame({"persona_id": ["train_b", "train_a"]}),
        pl.DataFrame({"persona_id": ["val_a"]}),
        pl.DataFrame({"persona_id": ["test_a"]}),
    )

    payload = build_holdout_payload(
        labels_path="logs/judge_labels/judge_labels.parquet",
        wrangled_dir="logs/wrangled",
        split_seed=2025,
        train_ratio=0.7,
        val_ratio=0.15,
        source_persona_count=10,
    )

    mock_load_all_data.assert_called_once_with(
        "logs/judge_labels/judge_labels.parquet",
        "logs/wrangled",
    )
    mock_split_by_persona.assert_called_once()
    assert payload["train_persona_ids"] == ["train_a", "train_b"]
    assert payload["val_persona_ids"] == ["val_a"]
    assert payload["test_persona_ids"] == ["test_a"]
    assert payload["source_persona_count"] == 10


def test_prepare_baseline_artifacts_reuses_existing_holdout(tmp_path: Path):
    registry_path = tmp_path / "personas.parquet"
    synthetic_dir = tmp_path / "synthetic"
    holdout_path = tmp_path / "holdout.yaml"
    synthetic_dir.mkdir()
    (synthetic_dir / "persona_aaa11111.md").write_text("one", encoding="utf-8")

    pl.DataFrame({"persona_id": ["aaa11111"], "name": ["A"]}).write_parquet(registry_path)
    holdout_path.write_text(
        "\n".join(
            [
                "train_persona_ids:",
                "- train_a",
                "val_persona_ids:",
                "- val_a",
                "test_persona_ids:",
                "- test_a",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot_payload, holdout_payload, wrote_holdout = prepare_baseline_artifacts(
        {
            "registry_path": registry_path,
            "synthetic_dir": synthetic_dir,
            "reuse_existing_holdout": True,
            "holdout_manifest_path": holdout_path,
        }
    )

    assert snapshot_payload["registry_persona_ids"] == ["aaa11111"]
    assert holdout_payload["val_persona_ids"] == ["val_a"]
    assert wrote_holdout is False


def test_write_yaml_payload_creates_parent_dirs(tmp_path: Path):
    output_path = tmp_path / "nested" / "payload.yaml"

    written = write_yaml_payload(output_path, {"answer": 42})

    assert written == output_path
    assert output_path.exists()
    assert "answer: 42" in output_path.read_text(encoding="utf-8")
