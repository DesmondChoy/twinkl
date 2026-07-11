"""Tests for runtime VIF inference helpers."""

from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.vif.critic import CriticMLP
from src.vif.dataset import VIFDataset
from src.vif.runtime import (
    _build_state_matrix,
    _resolve_runtime_config,
    aggregate_timeline_by_week,
    persist_runtime_artifacts,
    predict_persona_timeline,
)
from src.vif.state_encoder import StateEncoder


class _ContentAwareMockTextEncoder:
    embedding_dim = 8
    model_name = "content-aware-mock"

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.encode_batch(texts)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:  # noqa: ARG002
        rows = []
        for text in texts:
            values = [float((len(text) + idx) % 11) / 10.0 for idx in range(self.embedding_dim)]
            rows.append(values)
        if not rows:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)


def _write_wrangled_persona(path: Path) -> None:
    path.write_text(
        """# Persona deadbeef: Casey

## Profile
- **Persona ID:** deadbeef
- **Name:** Casey
- **Age:** 25-34
- **Profession:** Engineer
- **Culture:** Singaporean
- **Core Values:** Self-Direction, Benevolence
- **Bio:** Test persona for runtime inference.

---

## Entry 0 - 2025-01-01

Skipped the gym and doomscrolled after work.

---

## Entry 1 - 2025-01-03

Called my sister and helped a teammate debug a release issue.

---

## Entry 2 - 2025-01-10

Protected two hours for my side project before dinner.

---

## Entry 3 - 2025-01-12

Stayed up late to finish slides and ignored everyone else.

---
"""
    )


def test_predict_persona_timeline_and_aggregate_weekly(tmp_path, monkeypatch):
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir()
    _write_wrangled_persona(wrangled_dir / "persona_deadbeef.md")

    monkeypatch.setattr(
        "src.vif.runtime.create_encoder",
        lambda _config: _ContentAwareMockTextEncoder(),
    )

    torch.manual_seed(7)
    model = CriticMLP(input_dim=18, hidden_dim=6, output_dim=10, dropout=0.1)
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "training_config": {
                "encoder": {
                    "type": "sbert",
                    "model_name": "content-aware-mock",
                    "truncate_dim": None,
                    "text_prefix": "",
                    "prompt_name": None,
                    "prompt": None,
                },
                "state_encoder": {"window_size": 1},
                "mc_dropout": {"n_samples": 4},
            },
        },
        checkpoint_path,
    )

    timeline_df, metadata = predict_persona_timeline(
        persona_id="deadbeef",
        checkpoint_path=checkpoint_path,
        wrangled_dir=wrangled_dir,
        config_path=None,
        batch_size=2,
    )

    assert timeline_df.height == 4
    assert metadata["window_size"] == 1
    assert metadata["history_pooling"] == "none"
    assert metadata["n_mc_samples"] == 4
    assert set(["alignment_self_direction", "uncertainty_self_direction", "overall_mean"]).issubset(
        set(timeline_df.columns)
    )
    assert timeline_df["uncertainty_vector"].list.len().to_list() == [10, 10, 10, 10]
    assert timeline_df["alignment_vector"].list.len().to_list() == [10, 10, 10, 10]

    weekly_df = aggregate_timeline_by_week(timeline_df)
    assert weekly_df.height == 2
    assert weekly_df["n_entries"].to_list() == [2, 2]
    assert weekly_df["week_start"].to_list() == ["2024-12-30", "2025-01-06"]
    assert weekly_df["alignment_vector"].list.len().to_list() == [10, 10]

    artifact_paths = persist_runtime_artifacts(
        timeline_df=timeline_df,
        weekly_df=weekly_df,
        output_dir=tmp_path / "artifacts",
    )
    assert pl.read_parquet(artifact_paths["timeline_path"]).height == 4
    assert pl.read_parquet(artifact_paths["weekly_path"]).height == 2


def test_runtime_restores_flat_compact_history_config(tmp_path, monkeypatch):
    wrangled_dir = tmp_path / "wrangled"
    wrangled_dir.mkdir()
    _write_wrangled_persona(wrangled_dir / "persona_deadbeef.md")
    monkeypatch.setattr(
        "src.vif.runtime.create_encoder",
        lambda _config: _ContentAwareMockTextEncoder(),
    )

    model = CriticMLP(input_dim=23, hidden_dim=6, output_dim=10, dropout=0.1)
    checkpoint_path = tmp_path / "compact.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model.get_config(),
            "training_config": {
                "encoder_model": "content-aware-mock",
                "truncate_dim": None,
                "window_size": 1,
                "history_pooling": "mean",
                "history_window_size": 3,
                "history_summary_dim": 4,
                "mc_dropout": {"n_samples": 2},
            },
        },
        checkpoint_path,
    )

    timeline_df, metadata = predict_persona_timeline(
        persona_id="deadbeef",
        checkpoint_path=checkpoint_path,
        wrangled_dir=wrangled_dir,
        config_path=None,
        n_mc_samples=2,
    )

    assert timeline_df.height == 4
    assert metadata["history_pooling"] == "mean"
    assert metadata["history_window_size"] == 3
    assert metadata["history_summary_dim"] == 4


def test_flat_runtime_config_preserves_explicit_null_encoder_fields():
    config = _resolve_runtime_config(
        {
            "training_config": {
                "encoder_model": "custom-encoder",
                "truncate_dim": None,
                "prompt_name": None,
                "prompt": None,
                "window_size": 1,
            }
        },
        config_path=None,
    )

    assert config["encoder"]["truncate_dim"] is None
    assert config["encoder"]["prompt_name"] is None
    assert config["encoder"]["prompt"] is None


def test_legacy_checkpoint_disables_new_history_default(tmp_path):
    config_path = tmp_path / "future-default.yaml"
    config_path.write_text(
        """state_encoder:
  window_size: 1
  history_pooling: mean
  history_window_size: 3
  history_summary_dim: 4
""",
        encoding="utf-8",
    )

    for training_config in [
        {"window_size": 1},
        {"state_encoder": {"window_size": 1}},
    ]:
        config = _resolve_runtime_config(
            {"training_config": training_config},
            config_path=config_path,
        )
        assert config["state_encoder"]["history_pooling"] == "none"


def test_compact_history_dataset_runtime_state_parity():
    encoder = _ContentAwareMockTextEncoder()
    state_encoder = StateEncoder(
        encoder,
        window_size=1,
        history_pooling="mean",
        history_window_size=2,
        history_summary_dim=4,
    )
    entries = [
        {
            "persona_id": "deadbeef",
            "t_index": t_index,
            "date": date,
            "initial_entry": text,
            "nudge_text": None,
            "response_text": None,
            "core_values": ["Security"],
            "alignment_vector": [0.0] * 10,
        }
        for t_index, date, text in [
            (0, "2025-01-01", "first"),
            (2, "2025-01-03", "second"),
            (5, "2025-01-08", "third"),
        ]
    ]
    runtime_states = _build_state_matrix(
        entries,
        state_encoder=state_encoder,
        core_values=["Security"],
    )
    dataset = VIFDataset(
        pl.DataFrame(entries),
        state_encoder,
        cache_embeddings=False,
    )
    dataset_states = np.stack([dataset[index][0].numpy() for index in range(3)])

    np.testing.assert_allclose(dataset_states, runtime_states, atol=1e-6)
