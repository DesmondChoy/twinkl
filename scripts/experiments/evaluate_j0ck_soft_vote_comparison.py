#!/usr/bin/env python3
"""Score the twinkl-j0ck hard/soft paired runs against vote distributions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.judge import SCHWARTZ_VALUE_ORDER  # noqa: E402
from src.vif.holdout import load_fixed_holdout_ids  # noqa: E402


def distribution_scores(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_entropy_bits: np.ndarray,
) -> dict[str, float | int | None]:
    """Return proper scores and entropy agreement for three-class targets."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    target_entropy_bits = np.asarray(target_entropy_bits, dtype=np.float64)
    if probabilities.shape != targets.shape or probabilities.ndim != 2:
        raise ValueError("probabilities and targets must share shape (n, classes).")
    if probabilities.shape[1] != 3 or target_entropy_bits.shape != (len(targets),):
        raise ValueError("Expected three classes and one entropy value per target.")

    clipped = np.clip(probabilities, 1e-12, 1.0)
    nll = -np.sum(targets * np.log(clipped), axis=1)
    brier = np.sum((probabilities - targets) ** 2, axis=1)
    predicted_entropy = -np.sum(clipped * np.log2(clipped), axis=1)
    if (
        len(targets) > 1
        and float(np.std(predicted_entropy)) > 0.0
        and float(np.std(target_entropy_bits)) > 0.0
    ):
        entropy_correlation = float(
            np.corrcoef(predicted_entropy, target_entropy_bits)[0, 1]
        )
    else:
        entropy_correlation = None
    return {
        "n": int(len(targets)),
        "soft_nll": float(nll.mean()),
        "multiclass_brier": float(brier.mean()),
        "entropy_mae_bits": float(
            np.abs(predicted_entropy - target_entropy_bits).mean()
        ),
        "entropy_correlation": entropy_correlation,
    }


def build_long_targets(labels: pl.DataFrame) -> pl.DataFrame:
    """Expand the value-major target artifact to entry-by-dimension rows."""
    rows: list[dict] = []
    for dimension in SCHWARTZ_VALUE_ORDER:
        probability_column = f"vote_probability_{dimension}"
        entropy_column = f"vote_entropy_bits_{dimension}"
        if dimension == "security":
            tier_column = "security_decision_method"
        else:
            tier_column = f"confidence_{dimension}"
        required = {
            "persona_id",
            "t_index",
            probability_column,
            entropy_column,
            tier_column,
        }
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(f"Target artifact is missing columns: {sorted(missing)}")
        for row in labels.select(sorted(required)).iter_rows(named=True):
            rows.append(
                {
                    "persona_id": row["persona_id"],
                    "t_index": row["t_index"],
                    "dimension": dimension,
                    "vote_distribution": row[probability_column],
                    "target_entropy_bits": row[entropy_column],
                    "confidence_tier": row[tier_column],
                }
            )
    return pl.DataFrame(rows)


def _score_frame(frame: pl.DataFrame) -> dict[str, float | int | None]:
    return distribution_scores(
        np.asarray(frame["class_probabilities"].to_list(), dtype=np.float64),
        np.asarray(frame["vote_distribution"].to_list(), dtype=np.float64),
        frame["target_entropy_bits"].to_numpy(),
    )


def _training_class_priors(run: dict, labels: pl.DataFrame) -> np.ndarray:
    data_config = run["config"]["data"]
    val_ids, test_ids = load_fixed_holdout_ids(
        data_config["fixed_holdout_manifest_path"]
    )
    train = labels.filter(
        ~pl.col("persona_id").is_in(set(val_ids) | set(test_ids))
    )
    mode = run["config"]["training"]["training_target_mode"]
    if mode == "vote_distribution":
        values = np.asarray(
            train[data_config["soft_target_column"]].to_list(), dtype=np.float64
        ).reshape(-1, len(SCHWARTZ_VALUE_ORDER), 3)
        counts = values.sum(axis=0)
    else:
        values = np.asarray(
            train[data_config["hard_target_column"]].to_list(), dtype=np.int64
        )
        counts = np.stack(
            [(values == label).sum(axis=0) for label in (-1, 0, 1)], axis=1
        ).astype(np.float64)
    return counts / counts.sum(axis=1, keepdims=True)


def score_run(run_path: Path, targets: pl.DataFrame, labels: pl.DataFrame) -> dict:
    run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
    outputs = pl.read_parquet(run["artifacts"]["test_outputs"])
    joined = outputs.join(
        targets,
        on=["persona_id", "t_index", "dimension"],
        how="inner",
        validate="1:1",
    )
    if joined.height != outputs.height:
        raise ValueError(
            f"{run_path.name} joined {joined.height}/{outputs.height} test rows."
        )

    priors = _training_class_priors(run, labels)
    probabilities = np.asarray(
        joined["class_probabilities"].to_list(), dtype=np.float64
    )
    dimension_indices = np.asarray(
        [SCHWARTZ_VALUE_ORDER.index(value) for value in joined["dimension"]]
    )
    adjusted_probabilities = probabilities * priors[dimension_indices]
    adjusted_probabilities /= adjusted_probabilities.sum(axis=1, keepdims=True)
    loss_space_frame = joined.with_columns(
        pl.Series("class_probabilities", adjusted_probabilities.tolist())
    )

    per_dimension = {
        dimension: _score_frame(joined.filter(pl.col("dimension") == dimension))
        for dimension in SCHWARTZ_VALUE_ORDER
    }
    per_confidence = {
        str(tier): _score_frame(joined.filter(pl.col("confidence_tier") == tier))
        for tier in sorted(joined["confidence_tier"].unique().to_list())
    }
    evaluation = run["evaluation"]
    return {
        "run_id": run["metadata"]["run_id"],
        "run_path": str(run_path),
        "training_target_mode": run["config"]["training"]["training_target_mode"],
        "model_seed": int(run["config"]["training"]["seed"]),
        "hard_metrics": {
            key: float(evaluation[key])
            for key in (
                "qwk_mean",
                "recall_minus1",
                "minority_recall_mean",
                "hedging_mean",
                "calibration_global",
            )
        },
        "distribution_metrics": _score_frame(joined),
        "loss_space_distribution_metrics": _score_frame(loss_space_frame),
        "per_dimension": per_dimension,
        "per_confidence": per_confidence,
    }


def summarize_families(run_scores: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for mode in ("hard", "vote_distribution"):
        group = [row for row in run_scores if row["training_target_mode"] == mode]
        if not group:
            continue
        metrics: dict[str, float] = {}
        for section in (
            "hard_metrics",
            "distribution_metrics",
            "loss_space_distribution_metrics",
        ):
            for key in group[0][section]:
                if key == "n":
                    continue
                values = [row[section][key] for row in group]
                metric_name = f"{section}.{key}"
                metrics[f"median_{metric_name}"] = float(np.median(values))
                metrics[f"min_{metric_name}"] = float(np.min(values))
                metrics[f"max_{metric_name}"] = float(np.max(values))
        summary[mode] = {"run_ids": [row["run_id"] for row in group], **metrics}

    paired: list[dict] = []
    by_key = {
        (row["training_target_mode"], row["model_seed"]): row
        for row in run_scores
    }
    seeds = sorted({row["model_seed"] for row in run_scores})
    expected_keys = {
        (mode, seed)
        for seed in seeds
        for mode in ("hard", "vote_distribution")
    }
    missing_pairs = sorted(expected_keys - set(by_key))
    if missing_pairs:
        raise ValueError(
            "Runs must contain one hard and one vote_distribution arm per seed; "
            f"missing {missing_pairs}."
        )
    for seed in seeds:
        hard = by_key[("hard", seed)]
        soft = by_key[("vote_distribution", seed)]
        deltas = {}
        for section in (
            "hard_metrics",
            "distribution_metrics",
            "loss_space_distribution_metrics",
        ):
            for key in hard[section]:
                if key == "n":
                    continue
                deltas[f"{section}.{key}"] = soft[section][key] - hard[section][key]
        paired.append({"model_seed": seed, "soft_minus_hard": deltas})
    return {"families": summary, "paired_seed_deltas": paired}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(
            "logs/exports/twinkl_j0ck_soft_vote_target_v1/"
            "hybrid_soft_vote_labels.parquet"
        ),
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=[
            Path(f"logs/experiments/runs/run_{run_id:03d}_BalancedSoftmax.yaml")
            for run_id in range(63, 69)
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/exports/twinkl_j0ck_soft_vote_target_v1/"
            "paired_comparison_summary.json"
        ),
    )
    args = parser.parse_args()

    labels = pl.read_parquet(args.target)
    targets = build_long_targets(labels)
    run_scores = [score_run(path, targets, labels) for path in args.runs]
    payload = {
        "target_path": str(args.target),
        "class_order": [-1, 0, 1],
        "value_order": list(SCHWARTZ_VALUE_ORDER),
        "run_scores": run_scores,
        **summarize_families(run_scores),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote paired comparison: {args.output}")


if __name__ == "__main__":
    main()
