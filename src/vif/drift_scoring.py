"""Shared scoring helpers for target-based VIF drift evaluations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.drift_benchmark import normalize_value_name
from src.vif.runtime import load_runtime_bundle


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cases_sha256(cases: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        cases,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_metadata(cases: list[dict[str, Any]]) -> dict[tuple[str, str, int], str]:
    metadata = {}
    for case in cases:
        persona_id = str(case["persona_id"])
        dimensions = {normalize_value_name(str(value)) for value in case["core_values"]}
        for entry in case["entries"]:
            for dimension in dimensions:
                coordinate = (persona_id, dimension, int(entry["t_index"]))
                if coordinate in metadata:
                    raise ValueError(f"Duplicate locked score coordinate {coordinate}")
                metadata[coordinate] = str(entry["date"])
    return metadata


def _cached_evidence_is_valid(
    output: Path,
    provenance_path: Path,
    digest_path: Path,
    expected_provenance: dict[str, Any],
    expected_metadata: dict[tuple[str, str, int], str],
) -> bool:
    if (
        not output.is_file()
        or not provenance_path.is_file()
        or not digest_path.is_file()
    ):
        return False
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        evidence = pl.read_parquet(output)
        if provenance != expected_provenance:
            return False
        if digest_path.read_text(encoding="utf-8").strip() != _sha256_file(output):
            return False
        required_columns = {
            "source",
            "persona_id",
            "dimension",
            "t_index",
            "date",
            "p_minus1",
            "uncertainty",
            "predicted_class",
            "evidence_kind",
        }
        if not required_columns.issubset(evidence.columns):
            return False
        if evidence.height != len(expected_metadata):
            return False
        observed_metadata = {
            (
                str(row["persona_id"]),
                str(row["dimension"]),
                int(row["t_index"]),
            ): str(row["date"])
            for row in evidence.select(
                "persona_id", "dimension", "t_index", "date"
            ).to_dicts()
        }
        if observed_metadata != expected_metadata:
            return False
        if evidence["source"].unique().to_list() != [expected_provenance["arm_id"]]:
            return False
        if evidence["evidence_kind"].unique().to_list() != ["soft_probability"]:
            return False
        if not set(evidence["predicted_class"].unique().to_list()).issubset({-1, 0, 1}):
            return False
        for column in ("p_minus1", "uncertainty"):
            if not np.isfinite(evidence[column].to_numpy()).all():
                return False
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        pl.exceptions.PolarsError,
    ):
        return False
    return True


def score_mlp_cases(
    *,
    cases: list[dict[str, Any]],
    checkpoint_path: str | Path,
    arm_id: str,
    output_path: str | Path,
    mc_seed: int | None = None,
    mc_samples: int = 50,
) -> pl.DataFrame:
    """Score full trajectories with the production-style VIF state encoder.

    This intentionally contains no labels or target logic. It is reusable for a
    separately reviewed target without creating a second inference path.
    """
    output = Path(output_path)
    if not cases:
        raise ValueError("Cannot score an empty case list")
    if mc_samples <= 0:
        raise ValueError("mc_samples must be positive")
    checkpoint = Path(checkpoint_path)
    provenance_path = output.with_suffix(".provenance.json")
    digest_path = output.with_suffix(".sha256")
    expected_metadata = _expected_metadata(cases)
    expected_provenance = {
        "schema_version": 1,
        "arm_id": arm_id,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "cases_sha256": _cases_sha256(cases),
        "expected_coordinate_count": len(expected_metadata),
        "mc_seed": mc_seed,
        "mc_samples": mc_samples,
    }
    if _cached_evidence_is_valid(
        output,
        provenance_path,
        digest_path,
        expected_provenance,
        expected_metadata,
    ):
        return pl.read_parquet(output)

    model, state_encoder, _config, _checkpoint, device = load_runtime_bundle(checkpoint)
    flattened: list[tuple[dict[str, Any], dict[str, Any]]] = []
    texts = []
    seen_keys = set()
    for case in cases:
        persona_id = str(case.get("persona_id"))
        core_values = case.get("core_values")
        if not isinstance(core_values, (list, tuple)) or not core_values:
            raise ValueError(f"{persona_id} has no declared core values")
        entries = case.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{persona_id} has no entries")
        for entry in entries:
            key = (persona_id, int(entry["t_index"]))
            if key in seen_keys:
                raise ValueError(f"Duplicate score coordinate {key}")
            seen_keys.add(key)
            text = state_encoder.concatenate_entry_text(
                entry.get("initial_entry"),
                entry.get("nudge_text"),
                entry.get("response_text"),
            )
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing journal text for {key}")
            flattened.append((case, entry))
            texts.append(text)

    embeddings = state_encoder.text_encoder.encode_batch(texts)
    embedding_by_key = {
        (str(case["persona_id"]), int(entry["t_index"])): embeddings[index]
        for index, (case, entry) in enumerate(flattened)
    }
    states = []
    for case, entry in flattened:
        persona_id = str(case["persona_id"])
        t_index = int(entry["t_index"])
        case_entries = {int(row["t_index"]): row for row in case["entries"]}
        window_embeddings = []
        window_dates = []
        for offset in range(state_encoder.window_size):
            previous_index = t_index - offset
            if previous_index in case_entries:
                window_embeddings.append(embedding_by_key[(persona_id, previous_index)])
                window_dates.append(str(case_entries[previous_index]["date"]))
            else:
                window_embeddings.append(
                    np.zeros(state_encoder.text_encoder.embedding_dim, dtype=np.float32)
                )
                window_dates.append(None)
        states.append(
            state_encoder.build_state_vector_from_embeddings(
                embeddings=window_embeddings,
                dates=window_dates,
                core_values=list(case["core_values"]),
            )
        )

    state_tensor = torch.from_numpy(np.stack(states).astype(np.float32)).to(device)
    fork_devices = []
    if device.type == "cuda":
        fork_devices.append(
            device.index if device.index is not None else torch.cuda.current_device()
        )
    with torch.random.fork_rng(devices=fork_devices):
        if mc_seed is not None:
            torch.manual_seed(mc_seed)
        with torch.no_grad():
            _means, uncertainties = model.predict_with_uncertainty(
                state_tensor, n_samples=mc_samples
            )
            if not hasattr(model, "predict_probabilities"):
                raise ValueError(f"{arm_id} is not an ordinal probability model")
            probabilities = model.predict_probabilities(state_tensor)
    probabilities_np = probabilities.detach().cpu().numpy()
    uncertainties_np = uncertainties.detach().cpu().numpy()
    predicted_classes = probabilities_np.argmax(axis=-1) - 1

    rows = []
    for row_index, (case, entry) in enumerate(flattened):
        core_dimensions = {
            normalize_value_name(str(value)) for value in case["core_values"]
        }
        for dim_index, dimension in enumerate(SCHWARTZ_VALUE_ORDER):
            if dimension not in core_dimensions:
                continue
            rows.append(
                {
                    "source": arm_id,
                    "persona_id": str(case["persona_id"]),
                    "dimension": dimension,
                    "t_index": int(entry["t_index"]),
                    "date": str(entry["date"]),
                    "p_minus1": float(probabilities_np[row_index, dim_index, 0]),
                    "uncertainty": float(uncertainties_np[row_index, dim_index]),
                    "predicted_class": int(predicted_classes[row_index, dim_index]),
                    "evidence_kind": "soft_probability",
                }
            )
    result = pl.DataFrame(rows)
    observed_metadata = {
        (str(row["persona_id"]), str(row["dimension"]), int(row["t_index"])): str(
            row["date"]
        )
        for row in result.select(
            "persona_id", "dimension", "t_index", "date"
        ).to_dicts()
    }
    if observed_metadata != expected_metadata:
        raise ValueError("Scorer output does not cover the locked case coordinates")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output)
    provenance_path.write_text(
        json.dumps(expected_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest_path.write_text(_sha256_file(output) + "\n", encoding="utf-8")
    return result
