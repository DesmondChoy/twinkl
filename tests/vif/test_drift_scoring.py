"""Integrity checks for cached target-evaluation evidence."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src.vif.drift_scoring import _cached_evidence_is_valid, _sha256_file


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
