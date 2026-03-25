"""Tests for runtime VIF inference helpers."""

from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.vif.critic import CriticMLP
from src.vif.runtime import (
    aggregate_timeline_by_week,
    persist_runtime_artifacts,
    predict_persona_timeline,
)


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
