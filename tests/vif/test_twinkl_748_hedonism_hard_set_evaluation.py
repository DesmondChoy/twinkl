"""Focused tests for the twinkl-748 Hedonism-only evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
import yaml

from scripts.experiments.evaluate_twinkl_748_hedonism_hard_set import (
    FAMILIES,
    TARGET_VERSION,
    evaluate,
    load_frozen_hard_set,
    resolve_checkpoints,
    sha256_file,
)
from src.models.judge import SCHWARTZ_VALUE_ORDER


class _StateEncoder:
    input_entry_count = 1
    state_dim = 266

    def build_state_vector(self, *, texts, dates, core_values):  # noqa: ARG002
        state = np.zeros(266, dtype=np.float32)
        state[0] = 1.0 if "enjoy" in texts[0] else -1.0
        return state


class _Model:
    def eval(self):
        return self

    def predict_logits_and_probabilities(self, states):
        batch = states.shape[0]
        probabilities = torch.full((batch, 10, 3), 1 / 3, dtype=torch.float32)
        dim = SCHWARTZ_VALUE_ORDER.index("hedonism")
        positive = states[:, 0] > 0
        probabilities[positive, dim] = torch.tensor([0.02, 0.03, 0.95])
        probabilities[~positive, dim] = torch.tensor([0.95, 0.03, 0.02])
        return torch.log(probabilities), probabilities

    def predict_with_uncertainty(self, states, n_samples):  # noqa: ARG002
        _logits, probabilities = self.predict_logits_and_probabilities(states)
        classes = probabilities.argmax(dim=-1).float() - 1
        return classes, torch.zeros_like(classes)


def _training_config() -> dict:
    return {
        "encoder_model": "test-encoder",
        "trust_remote_code": False,
        "truncate_dim": 256,
        "text_prefix": "classification: ",
        "prompt_name": None,
        "prompt": None,
        "window_size": 1,
        "history_pooling": None,
        "history_window_size": None,
        "history_summary_dim": None,
    }


def _frozen(path: Path) -> Path:
    rows = []
    for pair_index in range(2):
        for sign, word in [(-1, "skip"), (1, "enjoy")]:
            rows.append(
                {
                    "review_pair_id": f"pair_{pair_index}",
                    "review_entry_id": f"entry_{pair_index}_{sign}",
                    "source_pair_id": f"source_{pair_index}",
                    "source_variant_id": f"variant_{pair_index}_{sign}",
                    "family": "test_family",
                    "core_values": ["Hedonism"],
                    "journal_entry": f"I {word} the afternoon.",
                    "hedonism_target": sign,
                }
            )
    pl.DataFrame(rows).write_parquet(path)
    summary = {
        "target_version": TARGET_VERSION,
        "frozen_hard_set_sha256": sha256_file(path),
    }
    summary_path = path.parent / "review_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest = {
        "target_version": TARGET_VERSION,
        "materialization_complete": True,
        "review_summary_sha256": sha256_file(summary_path),
    }
    (path.parent / "audit_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return path


def _runs(root: Path) -> None:
    run_dir = root / "logs/experiments/runs"
    checkpoint_dir = root / "checkpoints"
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir()
    for run_ids in FAMILIES.values():
        for run_id in run_ids:
            checkpoint = checkpoint_dir / f"run_{run_id:03d}.pt"
            checkpoint.write_bytes(f"checkpoint-{run_id}".encode())
            manifest = {"artifacts": {"checkpoint": str(checkpoint.relative_to(root))}}
            (run_dir / f"run_{run_id:03d}_BalancedSoftmax.yaml").write_text(
                yaml.safe_dump(manifest), encoding="utf-8"
            )


def test_frozen_loader_rejects_unpaired_target(tmp_path: Path):
    path = _frozen(tmp_path / "frozen.parquet")
    frame = pl.read_parquet(path).with_columns(pl.lit(1).alias("hedonism_target"))
    frame.write_parquet(path)

    with pytest.raises(ValueError, match=r"one -1 and one \+1"):
        load_frozen_hard_set(path)


def test_evaluate_rejects_frozen_set_that_drifted_after_review(
    tmp_path: Path,
):
    path = _frozen(tmp_path / "frozen.parquet")
    frame = pl.read_parquet(path).with_columns(
        pl.when(pl.col("review_entry_id") == "entry_0_-1")
        .then(pl.lit("Changed after review."))
        .otherwise(pl.col("journal_entry"))
        .alias("journal_entry")
    )
    frame.write_parquet(path)

    with pytest.raises(ValueError, match="does not match review summary"):
        evaluate(frozen_path=path, output_dir=tmp_path / "evaluation", root=tmp_path)


def test_resolve_checkpoints_reads_all_six_manifests(tmp_path: Path):
    _runs(tmp_path)

    resolved = resolve_checkpoints(root=tmp_path)

    assert [row["run_id"] for row in resolved] == [19, 20, 21, 34, 35, 36]
    assert all(len(row["checkpoint_sha256"]) == 64 for row in resolved)


def test_evaluate_scores_hedonism_pairs_and_preserves_hashes(
    tmp_path: Path, monkeypatch
):
    frozen = _frozen(tmp_path / "frozen.parquet")
    _runs(tmp_path)

    def fake_bundle(checkpoint_path, *, config_path, device):  # noqa: ARG001
        runtime = {
            "encoder": {
                "model_name": "test-encoder",
                "truncate_dim": 256,
                "text_prefix": "classification: ",
            },
            "state_encoder": {"window_size": 1},
        }
        return (
            _Model(),
            _StateEncoder(),
            runtime,
            {"epoch": 3, "training_config": _training_config()},
            "cpu",
        )

    def fake_checkpoint_model(run, *, device):  # noqa: ARG001
        return _Model(), {"epoch": 3, "training_config": _training_config()}

    monkeypatch.setattr(
        "scripts.experiments.evaluate_twinkl_748_hedonism_hard_set.load_runtime_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "scripts.experiments.evaluate_twinkl_748_hedonism_hard_set.load_checkpoint_model",
        fake_checkpoint_model,
    )
    output = tmp_path / "evaluation"

    result = evaluate(
        frozen_path=frozen,
        output_dir=output,
        base_seed=748,
        mc_samples=2,
        root=tmp_path,
    )

    assert len(result["run_summaries"]) == 6
    assert len(result["family_summaries"]) == 2
    assert all(row["exact_accuracy"] == 1 for row in result["run_summaries"])
    assert all(
        row["strict_both_members_correct_pair_rate"] == 1
        and row["pair_directional_accuracy"] == 1
        for row in result["run_summaries"]
    )
    assert result["family_summaries"][0]["qwk_secondary"]["median"] == 1
    assert len(result["frozen_hard_set_sha256"]) == 64
    assert (output / "hedonism_predictions.parquet").is_file()
    assert (output / "evaluation_summary.json").is_file()
    assert (output / "evaluation_report.md").is_file()
    predictions = pl.read_parquet(output / "hedonism_predictions.parquet")
    assert predictions.height == 24
    assert set(predictions["hedonism_target"].unique()) == {-1, 1}
    assert "security_target" not in predictions.columns

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluate(frozen_path=frozen, output_dir=output, mc_samples=2, root=tmp_path)
