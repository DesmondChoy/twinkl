#!/usr/bin/env python3
"""Score historical and repaired Security models under both target lenses."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.judge import SCHWARTZ_VALUE_ORDER  # noqa: E402
from src.vif.eval import (  # noqa: E402
    compute_calibration_summary,
    compute_hedging_per_dimension,
    compute_qwk_per_dimension,
    compute_recall_per_class,
    discretize_predictions,
)

OUTPUT_KEYS = ["persona_id", "t_index", "date", "dimension"]
REQUIRED_OUTPUT_COLUMNS = {
    *OUTPUT_KEYS,
    "split",
    "mean_prediction",
    "uncertainty",
}
NEARBY_SECURITY_DIMENSIONS = ("security", "conformity", "tradition")


def evaluate_comparison(
    *,
    historical_outputs: pl.DataFrame,
    repaired_outputs: pl.DataFrame,
    historical_labels: pl.DataFrame,
    repaired_labels: pl.DataFrame,
    split: str,
) -> dict[str, Any]:
    """Return a two-model by two-target-lens comparison."""
    historical_frame = _validate_output_frame(
        historical_outputs, split=split, arm="historical_model"
    )
    repaired_frame = _validate_output_frame(
        repaired_outputs, split=split, arm="repaired_model"
    )

    historical_samples, historical_predictions, historical_uncertainties = (
        _output_matrices(historical_frame)
    )
    repaired_samples, repaired_predictions, repaired_uncertainties = _output_matrices(
        repaired_frame
    )
    if historical_samples != repaired_samples:
        raise ValueError(
            "Historical and repaired model outputs cover different samples."
        )

    historical_targets = _label_matrix(historical_labels, historical_samples)
    repaired_targets = _label_matrix(repaired_labels, historical_samples)
    security_index = SCHWARTZ_VALUE_ORDER.index("security")
    nonsecurity_indices = [
        index for index in range(len(SCHWARTZ_VALUE_ORDER)) if index != security_index
    ]
    if not np.array_equal(
        historical_targets[:, nonsecurity_indices],
        repaired_targets[:, nonsecurity_indices],
    ):
        raise ValueError("Repaired labels changed a non-Security target.")

    changed_security_count = int(
        np.count_nonzero(
            historical_targets[:, security_index]
            != repaired_targets[:, security_index]
        )
    )
    arms = {
        "historical_model": (historical_predictions, historical_uncertainties),
        "repaired_model": (repaired_predictions, repaired_uncertainties),
    }
    lenses = {
        "historical_labels": historical_targets,
        "repaired_labels": repaired_targets,
    }
    cells = []
    for model_arm, (predictions, uncertainties) in arms.items():
        for target_lens, targets in lenses.items():
            cells.append(
                {
                    "model_arm": model_arm,
                    "target_lens": target_lens,
                    "split": split,
                    "n_samples": len(historical_samples),
                    "metrics": _compute_metrics(predictions, targets, uncertainties),
                }
            )

    return {
        "comparison": "twinkl-a30f_security_target_2x2",
        "split": split,
        "sample_count": len(historical_samples),
        "changed_security_target_count": changed_security_count,
        "cells": cells,
    }


def write_comparison_artifacts(result: dict[str, Any], output_dir: str | Path) -> None:
    """Write JSON, parquet, and Markdown summaries without overwriting a run."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)

    serializable = _json_safe(result)
    (output / "comparison_summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = []
    for cell in serializable["cells"]:
        metrics = cell["metrics"]
        rows.append(
            {
                "model_arm": cell["model_arm"],
                "target_lens": cell["target_lens"],
                "split": cell["split"],
                "n_samples": cell["n_samples"],
                "qwk_mean": metrics["qwk_mean"],
                "security_qwk": metrics["security_qwk"],
                "security_recall_minus1": metrics["security_recall"]["minus1"],
                "security_recall_zero": metrics["security_recall"]["zero"],
                "security_recall_plus1": metrics["security_recall"]["plus1"],
                "recall_minus1": metrics["recall_mean"]["minus1"],
                "minority_recall_mean": metrics["minority_recall_mean"],
                "hedging_mean": metrics["hedging_mean"],
                "calibration_global": metrics["calibration_global"],
                "conformity_qwk": metrics["qwk_per_dimension"]["conformity"],
                "tradition_qwk": metrics["qwk_per_dimension"]["tradition"],
            }
        )
    pl.DataFrame(rows).write_parquet(output / "comparison_cells.parquet")
    (output / "comparison_report.md").write_text(
        _format_markdown(serializable), encoding="utf-8"
    )


def _validate_output_frame(
    frame: pl.DataFrame, *, split: str, arm: str
) -> pl.DataFrame:
    missing = REQUIRED_OUTPUT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{arm} outputs are missing columns: {sorted(missing)}")
    selected = frame.filter(pl.col("split") == split)
    if selected.is_empty():
        raise ValueError(f"{arm} outputs contain no rows for split {split!r}.")
    if selected.select(OUTPUT_KEYS).is_duplicated().any():
        raise ValueError(f"{arm} outputs contain duplicate sample-dimension rows.")
    unknown_dimensions = set(selected["dimension"].unique()) - set(
        SCHWARTZ_VALUE_ORDER
    )
    if unknown_dimensions:
        raise ValueError(
            f"{arm} outputs contain unknown dimensions: {sorted(unknown_dimensions)}"
        )
    return selected


def _output_matrices(
    frame: pl.DataFrame,
) -> tuple[list[tuple[str, int, str]], np.ndarray, np.ndarray]:
    rows = frame.select(
        OUTPUT_KEYS + ["mean_prediction", "uncertainty"]
    ).to_dicts()
    by_sample: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        sample = (str(row["persona_id"]), int(row["t_index"]), str(row["date"]))
        by_sample.setdefault(sample, {})[str(row["dimension"])] = row

    samples = sorted(by_sample)
    predictions = np.empty((len(samples), len(SCHWARTZ_VALUE_ORDER)), dtype=float)
    uncertainties = np.empty_like(predictions)
    expected_dimensions = set(SCHWARTZ_VALUE_ORDER)
    for sample_index, sample in enumerate(samples):
        dimensions = by_sample[sample]
        if set(dimensions) != expected_dimensions:
            missing = sorted(expected_dimensions - set(dimensions))
            raise ValueError(
                f"Output sample {sample[:2]} is missing dimensions: {missing}"
            )
        for dim_index, dimension in enumerate(SCHWARTZ_VALUE_ORDER):
            predictions[sample_index, dim_index] = float(
                dimensions[dimension]["mean_prediction"]
            )
            uncertainties[sample_index, dim_index] = float(
                dimensions[dimension]["uncertainty"]
            )
    if not np.isfinite(predictions).all() or not np.isfinite(uncertainties).all():
        raise ValueError("Output predictions and uncertainties must be finite.")
    return samples, predictions, uncertainties


def _label_matrix(
    labels: pl.DataFrame,
    samples: list[tuple[str, int, str]],
) -> np.ndarray:
    required = {"persona_id", "t_index", "date", "alignment_vector"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label artifact is missing columns: {sorted(missing)}")
    if labels.select(["persona_id", "t_index"]).is_duplicated().any():
        raise ValueError("Label artifact contains duplicate entry coordinates.")

    lookup = {
        (str(row["persona_id"]), int(row["t_index"])): row
        for row in labels.select(required).to_dicts()
    }
    targets = np.empty((len(samples), len(SCHWARTZ_VALUE_ORDER)), dtype=int)
    for sample_index, (persona_id, t_index, date) in enumerate(samples):
        row = lookup.get((persona_id, t_index))
        if row is None:
            raise ValueError(
                f"Label artifact is missing sample {(persona_id, t_index)}."
            )
        if str(row["date"]) != date:
            raise ValueError(f"Label date mismatch for sample {(persona_id, t_index)}.")
        vector = row["alignment_vector"]
        if vector is None or len(vector) != len(SCHWARTZ_VALUE_ORDER):
            raise ValueError(
                f"Invalid alignment_vector for sample {(persona_id, t_index)}."
            )
        if any(value not in {-1, 0, 1} for value in vector):
            raise ValueError(
                f"Non-ordinal alignment_vector for sample {(persona_id, t_index)}."
            )
        targets[sample_index] = vector
    return targets


def _compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
) -> dict[str, Any]:
    qwk_per_dimension = compute_qwk_per_dimension(predictions, targets)
    recall = compute_recall_per_class(predictions, targets)
    hedging = compute_hedging_per_dimension(predictions)
    calibration = compute_calibration_summary(predictions, targets, uncertainties)
    pred_classes = discretize_predictions(predictions)
    confusion = {}
    for dimension in NEARBY_SECURITY_DIMENSIONS:
        index = SCHWARTZ_VALUE_ORDER.index(dimension)
        confusion[dimension] = confusion_matrix(
            targets[:, index], pred_classes[:, index], labels=[-1, 0, 1]
        ).tolist()

    finite_qwk = [value for value in qwk_per_dimension.values() if np.isfinite(value)]
    minority_values = [
        recall["mean"]["minus1"],
        recall["mean"]["plus1"],
    ]
    finite_minority = [value for value in minority_values if np.isfinite(value)]
    return {
        "qwk_mean": float(np.mean(finite_qwk)) if finite_qwk else float("nan"),
        "qwk_per_dimension": qwk_per_dimension,
        "security_qwk": qwk_per_dimension["security"],
        "security_recall": recall["per_dim"]["security"],
        "recall_mean": recall["mean"],
        "minority_recall_mean": (
            float(np.mean(finite_minority)) if finite_minority else float("nan")
        ),
        "hedging_mean": float(np.mean(list(hedging.values()))),
        "hedging_per_dimension": hedging,
        "calibration_global": calibration["error_uncertainty_correlation"],
        "calibration_per_dimension": calibration["per_dim"],
        "confusion_labels": [-1, 0, 1],
        "confusion_matrices": confusion,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _format_metric(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _format_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# twinkl-a30f Security target comparison",
        "",
        f"Split: `{result['split']}`. Samples: {result['sample_count']}. "
        f"Changed Security targets: {result['changed_security_target_count']}.",
        "",
        "| model | target lens | QWK | Security QWK | Security recall -1 | "
        "hedging | calibration |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cell in result["cells"]:
        metrics = cell["metrics"]
        lines.append(
            "| {model} | {lens} | {qwk} | {security_qwk} | {security_recall} | "
            "{hedging} | {calibration} |".format(
                model=cell["model_arm"],
                lens=cell["target_lens"],
                qwk=_format_metric(metrics["qwk_mean"]),
                security_qwk=_format_metric(metrics["security_qwk"]),
                security_recall=_format_metric(
                    metrics["security_recall"]["minus1"]
                ),
                hedging=_format_metric(metrics["hedging_mean"]),
                calibration=_format_metric(metrics["calibration_global"]),
            )
        )
    lines.extend(
        [
            "",
            "Each model is scored against both label regimes. Values across target "
            "lenses are diagnostic and must not be treated as one leaderboard.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-outputs", type=Path, required=True)
    parser.add_argument("--repaired-outputs", type=Path, required=True)
    parser.add_argument("--historical-labels", type=Path, required=True)
    parser.add_argument("--repaired-labels", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = evaluate_comparison(
        historical_outputs=pl.read_parquet(args.historical_outputs),
        repaired_outputs=pl.read_parquet(args.repaired_outputs),
        historical_labels=pl.read_parquet(args.historical_labels),
        repaired_labels=pl.read_parquet(args.repaired_labels),
        split=args.split,
    )
    write_comparison_artifacts(result, args.output_dir)
    print(f"Wrote {args.output_dir / 'comparison_summary.json'}")
    print(f"Wrote {args.output_dir / 'comparison_cells.parquet'}")
    print(f"Wrote {args.output_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
