"""Integrity checks for cached target-evaluation evidence."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from src.vif.drift_scoring import (
    _cached_evidence_is_valid,
    _sha256_file,
    score_mlp_cases,
)


def _write_cache(
    output: Path,
    provenance_path: Path,
    digest_path: Path,
    evidence: pl.DataFrame,
    provenance: dict,
) -> None:
    evidence.write_parquet(output)
    provenance_path.write_text(
        json.dumps(provenance, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest_path.write_text(_sha256_file(output) + "\n", encoding="utf-8")


def test_cached_evidence_requires_an_untampered_complete_parquet(tmp_path: Path):
    output = tmp_path / "evidence.parquet"
    provenance_path = output.with_suffix(".provenance.json")
    digest_path = output.with_suffix(".sha256")
    provenance = {
        "schema_version": 1,
        "arm_id": "run_020",
        "checkpoint_sha256": "checkpoint",
        "cases_sha256": "cases",
        "expected_coordinate_count": 1,
    }
    expected_metadata = {("p1", "security", 0): "2026-01-01"}
    evidence = pl.DataFrame(
        {
            "source": ["run_020"],
            "persona_id": ["p1"],
            "dimension": ["security"],
            "t_index": [0],
            "date": ["2026-01-01"],
            "p_minus1": [0.75],
            "uncertainty": [0.10],
            "predicted_class": [-1],
            "evidence_kind": ["soft_probability"],
        }
    )
    _write_cache(output, provenance_path, digest_path, evidence, provenance)

    assert _cached_evidence_is_valid(
        output,
        provenance_path,
        digest_path,
        provenance,
        expected_metadata,
    )

    evidence.with_columns(pl.lit(0.25).alias("p_minus1")).write_parquet(output)
    assert not _cached_evidence_is_valid(
        output,
        provenance_path,
        digest_path,
        provenance,
        expected_metadata,
    )

    malformed = evidence.drop("date")
    _write_cache(output, provenance_path, digest_path, malformed, provenance)
    assert not _cached_evidence_is_valid(
        output,
        provenance_path,
        digest_path,
        provenance,
        expected_metadata,
    )

    duplicated = pl.concat([evidence, evidence])
    _write_cache(output, provenance_path, digest_path, duplicated, provenance)
    assert not _cached_evidence_is_valid(
        output,
        provenance_path,
        digest_path,
        provenance,
        expected_metadata,
    )


def test_score_mlp_cases_rejects_non_positive_mc_sample_count(tmp_path: Path):
    with pytest.raises(ValueError, match="mc_samples must be positive"):
        score_mlp_cases(
            cases=[
                {
                    "persona_id": "p1",
                    "core_values": ["security"],
                    "entries": [
                        {
                            "t_index": 0,
                            "date": "2026-01-01",
                            "initial_entry": "Entry",
                        }
                    ],
                }
            ],
            checkpoint_path=tmp_path / "missing.pt",
            arm_id="run_020",
            output_path=tmp_path / "evidence.parquet",
            mc_samples=0,
        )


def test_score_mlp_cases_accepts_runtime_string_device_and_persists_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import numpy as np
    import torch

    class DummyTextEncoder:
        embedding_dim = 16

        def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
            return [np.zeros(16, dtype=np.float32) for _ in texts]

    class DummyStateEncoder:
        window_size = 1
        text_encoder = DummyTextEncoder()

        def concatenate_entry_text(
            self, initial_entry: str | None, nudge: str | None, response: str | None
        ) -> str:
            return initial_entry or "text"

        def build_state_vector_from_embeddings(
            self,
            embeddings: list[np.ndarray],
            dates: list[str | None],
            core_values: list[str],
        ) -> np.ndarray:
            return np.zeros(26, dtype=np.float32)

    class DummyModel(torch.nn.Module):
        def predict_with_uncertainty(
            self, x: torch.Tensor, n_samples: int = 5
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch_size = x.shape[0]
            return torch.zeros(batch_size, 10), torch.full((batch_size, 10), 0.1)

        def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            # shape (batch_size, 10, 3)
            probs = torch.zeros(batch_size, 10, 3)
            probs[:, :, 0] = 0.8  # p_minus1
            probs[:, :, 1] = 0.1
            probs[:, :, 2] = 0.1
            return probs

    dummy_checkpoint = tmp_path / "dummy_checkpoint.pt"
    dummy_checkpoint.write_bytes(b"dummy")

    dummy_model = DummyModel()
    dummy_state_encoder = DummyStateEncoder()

    monkeypatch.setattr(
        "src.vif.drift_scoring.load_runtime_bundle",
        lambda checkpoint_path: (
            dummy_model,
            dummy_state_encoder,
            {},
            {},
            "cpu",
        ),
    )

    cases = [
        {
            "persona_id": "p_test",
            "core_values": ["security"],
            "entries": [
                {
                    "t_index": 0,
                    "date": "2026-01-01",
                    "initial_entry": "Felt insecure today.",
                }
            ],
        }
    ]
    output_path = tmp_path / "scored_evidence.parquet"

    result = score_mlp_cases(
        cases=cases,
        checkpoint_path=dummy_checkpoint,
        arm_id="run_dummy",
        output_path=output_path,
        mc_seed=42,
        mc_samples=5,
    )

    assert result.height == 1
    assert result["persona_id"][0] == "p_test"
    assert result["dimension"][0] == "security"
    assert result["p_minus1"][0] == pytest.approx(0.8)
    assert result["predicted_class"][0] == -1
    assert output_path.exists()
    assert output_path.with_suffix(".provenance.json").exists()
    assert output_path.with_suffix(".sha256").exists()
