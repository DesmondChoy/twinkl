"""Validation-only post-hoc boundary optimization for ordinal VIF runs.

Consumes selected validation/test output artifacts from existing ordinal runs,
searches reversible decode policies on validation only, and emits one untouched
final test evaluation per tuned model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import median

import numpy as np
import polars as pl
import yaml
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.dataset import load_all_data, split_by_persona
from src.vif.eval import (
    compute_accuracy_per_dimension,
    compute_calibration_summary,
    compute_circumplex_diagnostics,
    compute_hedging_per_dimension,
    compute_mae_per_dimension,
    compute_qwk_nan_dims_count,
    compute_qwk_per_dimension,
    compute_recall_per_class,
    compute_spearman_per_dimension,
    discretize_predictions,
)
from src.vif.class_balance import class_counts_to_priors, compute_ordinal_class_counts

NUM_DIMS = len(SCHWARTZ_VALUE_ORDER)
NUM_CLASSES = 3
CLASS_VALUES = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
CLASS_LABELS = np.array([-1, 0, 1], dtype=np.int64)
DIM_TO_INDEX = {name: idx for idx, name in enumerate(SCHWARTZ_VALUE_ORDER)}
STANDARD_SOFTMAX_BRANCH = "standard_menon"
EFFECTIVE_PRIOR_SOFTMAX_BRANCH = "effective_prior"
SOFTMAX_BRANCH_LABELS = {
    "baseline": "Untouched baseline",
    STANDARD_SOFTMAX_BRANCH: "Best standard Menon branch",
    EFFECTIVE_PRIOR_SOFTMAX_BRANCH: "Effective-prior + per-dimension tau",
}

DEFAULT_CONFIG = {
    "runs_dir": "logs/experiments/runs",
    "labels_path": "logs/judge_labels/judge_labels.parquet",
    "wrangled_dir": "logs/wrangled",
    "run_ids": ["run_016", "run_017", "run_018"],
    "models": ["CDWCE_a3", "SoftOrdinal", "CORN"],
    "artifact_root": "logs/experiments/artifacts",
    "artifact_run_prefix": "posthoc_twinkl_681_3",
    "report_path": "logs/experiments/reports/experiment_review_2026-03-07_twinkl_681_3.md",
    "report_title": None,
    "report_scope_note": None,
    "recommended_model_label": "Recommended softmax base for `twinkl-681.4`",
    "summary_model_order": None,
    "selection_policy": {
        "name": "recall_minus1_then_qwk_guarded",
        "max_qwk_drop": 0.03,
        "require_non_negative_calibration": True,
        "rank_order": [
            "recall_minus1",
            "qwk_mean",
            "calibration_global",
            "decision_neutral_rate",
        ],
    },
    "softmax_logit_adjustment": {
        "enabled": True,
        "tau_grid": [0.0, 0.3, 0.5, 0.7, 1.0],
        "target_models": ["CDWCE_a3", "SoftOrdinal"],
        "prior_source": "train_split",
        "effective_prior_branch": {
            "enabled": False,
            "prior_source": "validation_posteriors",
            "tau_mode": "per_dimension",
            "tau_grid": [0.0, 0.3, 0.5, 0.7, 1.0],
            "prior_estimation": {
                "method": "mean_posterior",
                "eps": 1e-9,
            },
        },
    },
    "corn_threshold_policy": {
        "enabled": True,
        "target_models": ["CORN"],
        "margin_grid": [0.0, 0.05, 0.10, 0.15, 0.20],
        "search_shared_policy_first": True,
        "allow_per_dimension_override": True,
    },
}


@dataclass(frozen=True)
class RunSpec:
    """Resolved source run metadata for one active frontier model."""

    run_id: str
    model_name: str
    run_path: Path
    checkpoint_path: Path
    validation_outputs_path: Path
    test_outputs_path: Path
    split_seed: int
    train_ratio: float
    val_ratio: float


@dataclass
class ArtifactBundle:
    """Wide + long representations of an exported ordinal output artifact."""

    data_frame: pl.DataFrame
    metadata_rows: list[dict]
    split: str
    model_name: str
    targets: np.ndarray
    baseline_predicted_classes: np.ndarray
    baseline_mean_predictions: np.ndarray
    uncertainties: np.ndarray
    raw_logits: np.ndarray
    class_probabilities: np.ndarray

    @property
    def n_samples(self) -> int:
        return self.targets.shape[0]


def _deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path | None) -> dict:
    """Load YAML config and deep-merge into defaults."""
    config = deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config

    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    payload = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    _deep_update(config, payload)
    return config


def _resolve_path(root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _softmax_branch_specs_for_model(model_name: str, config: dict) -> list[dict]:
    softmax_config = config["softmax_logit_adjustment"]
    branches: list[dict] = []

    standard_targets = set(softmax_config.get("target_models", []))
    if softmax_config.get("enabled", True) and model_name in standard_targets:
        branches.append(
            {
                "branch_name": STANDARD_SOFTMAX_BRANCH,
                "prior_source": softmax_config.get("prior_source", "train_split"),
                "tau_mode": "shared",
                "tau_grid": [float(value) for value in softmax_config.get("tau_grid", [])],
                "prior_estimation": None,
            }
        )

    effective_cfg = softmax_config.get("effective_prior_branch", {})
    effective_targets = set(effective_cfg.get("target_models", softmax_config.get("target_models", [])))
    if effective_cfg.get("enabled", False) and model_name in effective_targets:
        branches.append(
            {
                "branch_name": EFFECTIVE_PRIOR_SOFTMAX_BRANCH,
                "prior_source": effective_cfg.get("prior_source", "validation_posteriors"),
                "tau_mode": effective_cfg.get("tau_mode", "per_dimension"),
                "tau_grid": [float(value) for value in effective_cfg.get("tau_grid", softmax_config.get("tau_grid", []))],
                "prior_estimation": deepcopy(
                    effective_cfg.get(
                        "prior_estimation",
                        {"method": "mean_posterior", "eps": 1e-9},
                    )
                ),
            }
        )

    return branches


def _find_run_path(runs_dir: Path, run_id: str, model_name: str) -> Path:
    path = runs_dir / f"{run_id}_{model_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Run YAML not found: {path}")
    return path


def load_run_specs(config: dict, repo_root: Path) -> list[RunSpec]:
    """Resolve run YAMLs + source artifact paths for the requested frontier."""
    runs_dir = _resolve_path(repo_root, config["runs_dir"])
    run_specs: list[RunSpec] = []

    for run_id in config["run_ids"]:
        for model_name in config["models"]:
            run_path = _find_run_path(runs_dir, run_id, model_name)
            run_data = yaml.safe_load(run_path.read_text(encoding="utf-8"))
            artifacts = run_data.get("artifacts", {})
            run_specs.append(
                RunSpec(
                    run_id=run_id,
                    model_name=model_name,
                    run_path=run_path,
                    checkpoint_path=_resolve_path(repo_root, artifacts["checkpoint"]),
                    validation_outputs_path=_resolve_path(repo_root, artifacts["validation_outputs"]),
                    test_outputs_path=_resolve_path(repo_root, artifacts["test_outputs"]),
                    split_seed=int(run_data["config"]["data"]["split_seed"]),
                    train_ratio=float(run_data["config"]["data"]["train_ratio"]),
                    val_ratio=float(run_data["config"]["data"]["val_ratio"]),
                )
            )

    return run_specs


def _validate_sorted_artifact(df: pl.DataFrame, artifact_path: Path) -> pl.DataFrame:
    required_columns = {
        "persona_id",
        "t_index",
        "date",
        "split",
        "model_name",
        "dimension",
        "target",
        "predicted_class",
        "mean_prediction",
        "uncertainty",
        "raw_logits",
        "class_probabilities",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Artifact {artifact_path} is missing required columns: {sorted(missing)}")

    df = df.with_columns(
        pl.col("dimension").replace_strict(DIM_TO_INDEX, return_dtype=pl.Int64).alias("_dim_index")
    ).sort(["persona_id", "t_index", "date", "_dim_index"])

    sample_sizes = (
        df.group_by(["persona_id", "t_index", "date"], maintain_order=True)
        .len()
        .get_column("len")
        .to_list()
    )
    if not sample_sizes or any(count != NUM_DIMS for count in sample_sizes):
        raise ValueError(
            f"Artifact {artifact_path} does not contain exactly {NUM_DIMS} dimension rows per sample."
        )

    expected_dimension_order = SCHWARTZ_VALUE_ORDER * (df.height // NUM_DIMS)
    actual_dimension_order = df.get_column("dimension").to_list()
    if actual_dimension_order != expected_dimension_order:
        raise ValueError(
            f"Artifact {artifact_path} does not align to the expected Schwartz dimension order."
        )

    return df


def load_artifact_bundle(artifact_path: str | Path) -> ArtifactBundle:
    """Load a saved ordinal output artifact into wide arrays + long rows."""
    artifact_path = Path(artifact_path)
    df = _validate_sorted_artifact(
        pl.read_parquet(artifact_path),
        artifact_path,
    )

    metadata_columns = ["persona_id", "t_index", "date", "split", "model_name"]
    metadata_rows = []
    rows = df.select(metadata_columns).to_dicts()
    for row_idx in range(0, len(rows), NUM_DIMS):
        metadata_rows.append(rows[row_idx])

    row_count = df.height
    n_samples = row_count // NUM_DIMS
    split = metadata_rows[0]["split"] if metadata_rows else "unknown"
    model_name = metadata_rows[0]["model_name"] if metadata_rows else "unknown"

    raw_logits_rows = np.asarray(df.get_column("raw_logits").to_list(), dtype=np.float64)
    probabilities_rows = np.asarray(df.get_column("class_probabilities").to_list(), dtype=np.float64)

    return ArtifactBundle(
        data_frame=df.drop("_dim_index"),
        metadata_rows=metadata_rows,
        split=split,
        model_name=model_name,
        targets=np.asarray(df.get_column("target").to_list(), dtype=np.int64).reshape(n_samples, NUM_DIMS),
        baseline_predicted_classes=np.asarray(
            df.get_column("predicted_class").to_list(), dtype=np.int64
        ).reshape(n_samples, NUM_DIMS),
        baseline_mean_predictions=np.asarray(
            df.get_column("mean_prediction").to_list(), dtype=np.float64
        ).reshape(n_samples, NUM_DIMS),
        uncertainties=np.asarray(
            df.get_column("uncertainty").to_list(), dtype=np.float64
        ).reshape(n_samples, NUM_DIMS),
        raw_logits=raw_logits_rows.reshape(n_samples, NUM_DIMS, -1),
        class_probabilities=probabilities_rows.reshape(n_samples, NUM_DIMS, NUM_CLASSES),
    )


def _expected_scores(probabilities: np.ndarray) -> np.ndarray:
    return probabilities @ CLASS_VALUES


def _nanmean_or_nan(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def _compute_score_based_policy_metrics(
    *,
    targets: np.ndarray,
    score_predictions: np.ndarray,
    uncertainties: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict:
    """Match frontier run metrics by scoring thresholded continuous predictions."""
    predicted_classes = discretize_predictions(score_predictions)
    qwk_per_dim = compute_qwk_per_dimension(score_predictions, targets)
    recall_per_class = compute_recall_per_class(score_predictions, targets)
    qwk_mean = _nanmean_or_nan(list(qwk_per_dim.values()))
    calibration = compute_calibration_summary(score_predictions, targets, uncertainties)
    qwk_nan_dims_count = compute_qwk_nan_dims_count(qwk_per_dim)
    decision_neutral_rate = float((predicted_classes == 0).mean())
    hedging_per_dim = compute_hedging_per_dimension(score_predictions)
    mae_per_dim = compute_mae_per_dimension(score_predictions, targets)
    accuracy_per_dim = compute_accuracy_per_dimension(score_predictions, targets)
    spearman_per_dim = compute_spearman_per_dimension(score_predictions, targets)
    circumplex = compute_circumplex_diagnostics(score_predictions, probabilities=probabilities)

    return {
        "predictions": score_predictions,
        "targets": targets.astype(np.int64),
        "expected_scores": score_predictions,
        "predicted_classes": predicted_classes.astype(np.int64),
        "uncertainties": uncertainties,
        "mae_per_dim": mae_per_dim,
        "mae_mean": float(np.mean(list(mae_per_dim.values()))),
        "accuracy_per_dim": accuracy_per_dim,
        "accuracy_mean": float(np.mean(list(accuracy_per_dim.values()))),
        "qwk_per_dim": qwk_per_dim,
        "qwk_mean": qwk_mean,
        "qwk_nan_dims_count": qwk_nan_dims_count,
        "recall_per_class": recall_per_class,
        "recall_minus1": float(recall_per_class["mean"]["minus1"]),
        "recall_zero": float(recall_per_class["mean"]["zero"]),
        "recall_plus1": float(recall_per_class["mean"]["plus1"]),
        "minority_recall_mean": _nanmean_or_nan(
            [recall_per_class["mean"]["minus1"], recall_per_class["mean"]["plus1"]]
        ),
        "decision_neutral_rate": decision_neutral_rate,
        "hedging_per_dim": hedging_per_dim,
        "hedging_mean": float(np.mean(list(hedging_per_dim.values()))),
        "calibration": calibration,
        "spearman_per_dim": spearman_per_dim,
        "spearman_mean": _nanmean_or_nan(list(spearman_per_dim.values())),
        "circumplex": circumplex,
    }


def _compute_policy_metrics(
    *,
    targets: np.ndarray,
    predicted_classes: np.ndarray,
    expected_scores: np.ndarray,
    uncertainties: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict:
    predictions_for_discrete_metrics = predicted_classes.astype(np.float64)
    qwk_per_dim = compute_qwk_per_dimension(predictions_for_discrete_metrics, targets)
    recall_per_class = compute_recall_per_class(predictions_for_discrete_metrics, targets)
    qwk_mean = _nanmean_or_nan(list(qwk_per_dim.values()))
    calibration = compute_calibration_summary(expected_scores, targets, uncertainties)
    qwk_nan_dims_count = compute_qwk_nan_dims_count(qwk_per_dim)
    decision_neutral_rate = float((predicted_classes == 0).mean())
    hedging_per_dim = compute_hedging_per_dimension(expected_scores)
    mae_per_dim = compute_mae_per_dimension(predictions_for_discrete_metrics, targets)
    accuracy_per_dim = compute_accuracy_per_dimension(predictions_for_discrete_metrics, targets)
    spearman_per_dim = compute_spearman_per_dimension(expected_scores, targets)
    circumplex = compute_circumplex_diagnostics(expected_scores, probabilities=probabilities)

    return {
        "predictions": predictions_for_discrete_metrics,
        "targets": targets.astype(np.int64),
        "expected_scores": expected_scores,
        "predicted_classes": predicted_classes.astype(np.int64),
        "uncertainties": uncertainties,
        "mae_per_dim": mae_per_dim,
        "mae_mean": float(np.mean(list(mae_per_dim.values()))),
        "accuracy_per_dim": accuracy_per_dim,
        "accuracy_mean": float(np.mean(list(accuracy_per_dim.values()))),
        "qwk_per_dim": qwk_per_dim,
        "qwk_mean": qwk_mean,
        "qwk_nan_dims_count": qwk_nan_dims_count,
        "recall_per_class": recall_per_class,
        "recall_minus1": float(recall_per_class["mean"]["minus1"]),
        "recall_zero": float(recall_per_class["mean"]["zero"]),
        "recall_plus1": float(recall_per_class["mean"]["plus1"]),
        "minority_recall_mean": _nanmean_or_nan(
            [recall_per_class["mean"]["minus1"], recall_per_class["mean"]["plus1"]]
        ),
        "decision_neutral_rate": decision_neutral_rate,
        "hedging_per_dim": hedging_per_dim,
        "hedging_mean": float(np.mean(list(hedging_per_dim.values()))),
        "calibration": calibration,
        "spearman_per_dim": spearman_per_dim,
        "spearman_mean": _nanmean_or_nan(list(spearman_per_dim.values())),
        "circumplex": circumplex,
    }


def _rank_metric_value(metric_name: str, metrics: dict) -> float:
    metric_lookup = {
        "recall_minus1": float(metrics["recall_minus1"]),
        "qwk_mean": float(metrics["qwk_mean"]),
        "calibration_global": float(metrics["calibration"]["error_uncertainty_correlation"]),
        "decision_neutral_rate": -float(metrics["decision_neutral_rate"]),
    }
    if metric_name not in metric_lookup:
        raise ValueError(f"Unsupported selection rank metric: {metric_name}")

    value = metric_lookup[metric_name]
    return value if np.isfinite(value) else float("-inf")


def _policy_rank_key(metrics: dict, config: dict) -> list[float]:
    rank_order = config["selection_policy"].get(
        "rank_order",
        ["recall_minus1", "qwk_mean", "calibration_global", "decision_neutral_rate"],
    )
    return [_rank_metric_value(metric_name, metrics) for metric_name in rank_order]


def _is_candidate_eligible(candidate_metrics: dict, baseline_metrics: dict, config: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    qwk_mean = float(candidate_metrics["qwk_mean"])
    calibration_global = float(candidate_metrics["calibration"]["error_uncertainty_correlation"])
    qwk_nan_dims_count = int(candidate_metrics["qwk_nan_dims_count"])
    qwk_delta = qwk_mean - float(baseline_metrics["qwk_mean"])

    if not np.isfinite(qwk_mean):
        reasons.append("qwk_mean_non_finite")
    if qwk_nan_dims_count > 0:
        reasons.append("qwk_nan_dims_present")
    if qwk_delta < -float(config["selection_policy"]["max_qwk_drop"]):
        reasons.append("qwk_drop_exceeds_guard")
    if (
        config["selection_policy"].get("require_non_negative_calibration", True)
        and not np.isfinite(calibration_global)
    ):
        reasons.append("non_finite_calibration")
    if (
        config["selection_policy"].get("require_non_negative_calibration", True)
        and np.isfinite(calibration_global)
        and calibration_global < 0
    ):
        reasons.append("negative_calibration")

    return len(reasons) == 0, reasons


def _build_candidate_record(
    *,
    policy_name: str,
    policy_family: str,
    policy_payload: dict,
    baseline_metrics: dict,
    candidate_metrics: dict,
    config: dict,
    split: str,
) -> dict:
    eligible, ineligible_reasons = _is_candidate_eligible(
        candidate_metrics,
        baseline_metrics,
        config=config,
    )
    qwk_delta = float(candidate_metrics["qwk_mean"] - baseline_metrics["qwk_mean"])
    selection_score = {
        "rank_key": _policy_rank_key(candidate_metrics, config),
        "baseline_qwk_mean": float(baseline_metrics["qwk_mean"]),
        "tuned_qwk_mean": float(candidate_metrics["qwk_mean"]),
        "qwk_delta": qwk_delta,
    }
    return {
        "split": split,
        "policy_name": policy_name,
        "policy_family": policy_family,
        "policy_payload": policy_payload,
        "selection_score": selection_score,
        "eligible": eligible,
        "ineligible_reasons": ineligible_reasons,
        "metrics": candidate_metrics,
    }


def _baseline_candidate_record(
    *,
    policy_name: str,
    policy_family: str,
    policy_payload: dict,
    artifact: ArtifactBundle,
    config: dict,
) -> dict:
    baseline_metrics = _compute_policy_metrics(
        targets=artifact.targets,
        predicted_classes=artifact.baseline_predicted_classes,
        expected_scores=artifact.baseline_mean_predictions,
        uncertainties=artifact.uncertainties,
        probabilities=artifact.class_probabilities,
    )
    return _build_candidate_record(
        policy_name=policy_name,
        policy_family=policy_family,
        policy_payload=policy_payload,
        baseline_metrics=baseline_metrics,
        candidate_metrics=baseline_metrics,
        config=config,
        split=artifact.split,
    )


def _candidate_sort_tuple(candidate_record: dict) -> tuple[float, float, float, float]:
    rank_key = candidate_record["selection_score"]["rank_key"]
    return (
        rank_key[0],
        rank_key[1],
        rank_key[2],
        rank_key[3],
    )


def _pick_best_candidate(
    *,
    candidate_records: list[dict],
    baseline_policy_name: str,
) -> dict:
    eligible = [record for record in candidate_records if record["eligible"]]
    if eligible:
        return max(eligible, key=_candidate_sort_tuple)

    for record in candidate_records:
        if record["policy_name"] == baseline_policy_name:
            return record

    raise ValueError("No baseline candidate found during post-hoc candidate selection.")


def _policy_family_for_model(model_name: str, config: dict) -> str:
    softmax_config = config["softmax_logit_adjustment"]
    corn_config = config["corn_threshold_policy"]

    if _softmax_branch_specs_for_model(model_name, config):
        return "softmax"
    if corn_config.get("enabled", True) and model_name in corn_config["target_models"]:
        return "corn"

    raise ValueError(
        f"Model {model_name} is not enabled for any post-hoc policy family in the provided config."
    )


def _softmax_logit_adjustment(
    raw_logits: np.ndarray,
    priors: np.ndarray,
    tau: float | np.ndarray,
) -> np.ndarray:
    raw_logits = raw_logits.astype(np.float64, copy=False)
    if raw_logits.ndim != 3 or raw_logits.shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"Softmax logits must have shape (n_samples, n_dims, {NUM_CLASSES}); got {raw_logits.shape}."
        )

    n_dims = raw_logits.shape[1]
    priors = np.asarray(priors, dtype=np.float64)
    if priors.shape == (n_dims, NUM_CLASSES):
        priors = priors.reshape(1, n_dims, NUM_CLASSES)
    if priors.shape != (1, n_dims, NUM_CLASSES):
        raise ValueError(
            f"Softmax priors must have shape ({n_dims}, {NUM_CLASSES}); got {priors.shape}."
        )

    tau_value = np.asarray(tau, dtype=np.float64)
    if tau_value.ndim == 0:
        tau_broadcast: float | np.ndarray = float(tau_value)
    elif tau_value.shape == (n_dims,):
        tau_broadcast = tau_value.reshape(1, n_dims, 1)
    else:
        raise ValueError(f"Softmax tau must be scalar or shape ({n_dims},); got {tau_value.shape}.")

    if np.allclose(tau_value, 0.0):
        adjusted_logits = raw_logits.astype(np.float64, copy=True)
    else:
        adjusted_logits = raw_logits - (tau_broadcast * np.log(priors))

    shifted = adjusted_logits - adjusted_logits.max(axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)


def _softmax_tau_mode(tau: float | np.ndarray) -> str:
    tau_value = np.asarray(tau, dtype=np.float64)
    return "shared" if tau_value.ndim == 0 else "per_dimension"


def _tau_is_zero(tau: float | np.ndarray) -> bool:
    tau_value = np.asarray(tau, dtype=np.float64)
    return bool(np.allclose(tau_value, 0.0))


def _corn_margin_predictions(
    probabilities: np.ndarray,
    *,
    minus_margins: np.ndarray,
    plus_margins: np.ndarray,
) -> np.ndarray:
    if minus_margins.shape != (NUM_DIMS,) or plus_margins.shape != (NUM_DIMS,):
        raise ValueError("CORN margin arrays must have one value per Schwartz dimension.")

    prob_minus = probabilities[:, :, 0]
    prob_zero = probabilities[:, :, 1]
    prob_plus = probabilities[:, :, 2]

    predicted = np.zeros(prob_minus.shape, dtype=np.int64)

    minus_mask = (prob_minus >= (prob_zero - minus_margins.reshape(1, -1))) & (prob_minus >= prob_plus)
    plus_mask = (prob_plus > prob_minus) & (prob_plus >= (prob_zero - plus_margins.reshape(1, -1)))

    predicted[minus_mask] = -1
    predicted[plus_mask] = 1
    return predicted


def _candidate_frame_row(
    *,
    source_run: RunSpec,
    split: str,
    policy_name: str,
    policy_family: str,
    candidate_record: dict,
) -> dict:
    metrics = candidate_record["metrics"]
    calibration_global = float(metrics["calibration"]["error_uncertainty_correlation"])
    circumplex_summary = metrics["circumplex"]["summary"]
    policy_payload = candidate_record["policy_payload"]
    tau_mode = policy_payload.get("tau_mode", "")
    per_dimension_tau = policy_payload.get("per_dimension_tau")
    return {
        "run_id": source_run.run_id,
        "model_name": source_run.model_name,
        "split": split,
        "policy_name": policy_name,
        "policy_family": policy_family,
        "branch_name": policy_payload.get("branch_name", ""),
        "prior_source": policy_payload.get("prior_source", ""),
        "prior_estimation_method": (policy_payload.get("prior_estimation") or {}).get("method", ""),
        "tau_mode": tau_mode,
        "shared_tau": float(policy_payload["tau"]) if tau_mode == "shared" and "tau" in policy_payload else float("nan"),
        "per_dimension_tau": (
            yaml.safe_dump(per_dimension_tau, sort_keys=True).strip() if per_dimension_tau is not None else ""
        ),
        "eligible": bool(candidate_record["eligible"]),
        "ineligible_reasons": ",".join(candidate_record["ineligible_reasons"]),
        "recall_minus1": float(metrics["recall_minus1"]),
        "qwk_mean": float(metrics["qwk_mean"]),
        "qwk_nan_dims_count": int(metrics["qwk_nan_dims_count"]),
        "calibration_global": calibration_global,
        "decision_neutral_rate": float(metrics["decision_neutral_rate"]),
        "hedging_mean": float(metrics["hedging_mean"]),
        "minority_recall_mean": float(metrics["minority_recall_mean"]),
        "opposite_violation_mean": float(circumplex_summary["opposite_violation_mean"]),
        "adjacent_support_mean": float(circumplex_summary["adjacent_support_mean"]),
        "selection_rank_0": float(candidate_record["selection_score"]["rank_key"][0]),
        "selection_rank_1": float(candidate_record["selection_score"]["rank_key"][1]),
        "selection_rank_2": float(candidate_record["selection_score"]["rank_key"][2]),
        "selection_rank_3": float(candidate_record["selection_score"]["rank_key"][3]),
        "qwk_delta": float(candidate_record["selection_score"]["qwk_delta"]),
        "policy_payload": yaml.safe_dump(policy_payload, sort_keys=True).strip(),
    }


def compute_class_priors(train_df: pl.DataFrame) -> np.ndarray:
    """Compute per-dimension class priors ordered as [-1, 0, +1]."""
    targets = np.asarray(train_df.get_column("alignment_vector").to_list(), dtype=np.int64)
    counts = compute_ordinal_class_counts(targets)
    return class_counts_to_priors(counts, eps=1e-9)


@lru_cache(maxsize=8)
def reconstruct_train_priors(
    *,
    labels_path: str,
    wrangled_dir: str,
    train_ratio: float,
    val_ratio: float,
    split_seed: int,
) -> np.ndarray:
    """Recompute deterministic train priors for the corrected split."""
    labels_df, entries_df = load_all_data(labels_path, wrangled_dir)
    train_df, _val_df, _test_df = split_by_persona(
        labels_df,
        entries_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=split_seed,
    )
    return compute_class_priors(train_df)


def estimate_effective_priors_from_validation_posteriors(
    artifact: ArtifactBundle,
    *,
    eps: float = 1e-9,
) -> np.ndarray:
    """Estimate per-dimension effective priors from validation posteriors only."""
    probabilities = np.asarray(artifact.class_probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[1:] != (NUM_DIMS, NUM_CLASSES):
        raise ValueError(
            "Validation posterior probabilities must have shape "
            f"(n_samples, {NUM_DIMS}, {NUM_CLASSES}); got {probabilities.shape}."
        )

    priors = probabilities.mean(axis=0)
    priors = np.clip(priors, float(eps), None)
    priors /= priors.sum(axis=1, keepdims=True)
    return priors


def _artifact_output_frame(
    *,
    metadata_rows: list[dict],
    predicted_classes: np.ndarray,
    expected_scores: np.ndarray,
    uncertainties: np.ndarray,
    raw_logits: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    split: str,
    model_name: str,
    policy_name: str,
) -> pl.DataFrame:
    rows: list[dict] = []
    for sample_idx, metadata in enumerate(metadata_rows):
        for dim_idx, dimension in enumerate(SCHWARTZ_VALUE_ORDER):
            rows.append(
                {
                    "persona_id": metadata["persona_id"],
                    "t_index": int(metadata["t_index"]),
                    "date": metadata["date"],
                    "split": split,
                    "model_name": model_name,
                    "policy_name": policy_name,
                    "dimension": dimension,
                    "target": int(targets[sample_idx, dim_idx]),
                    "predicted_class": int(predicted_classes[sample_idx, dim_idx]),
                    "mean_prediction": float(expected_scores[sample_idx, dim_idx]),
                    "uncertainty": float(uncertainties[sample_idx, dim_idx]),
                    "raw_logits": raw_logits[sample_idx, dim_idx].tolist(),
                    "class_probabilities": probabilities[sample_idx, dim_idx].tolist(),
                }
            )
    return pl.DataFrame(rows)


def _compact_metrics(metrics: dict) -> dict:
    circumplex_summary = metrics["circumplex"]["summary"]
    return {
        "qwk_mean": float(metrics["qwk_mean"]),
        "recall_minus1": float(metrics["recall_minus1"]),
        "minority_recall_mean": float(metrics["minority_recall_mean"]),
        "hedging_mean": float(metrics["hedging_mean"]),
        "decision_neutral_rate": float(metrics["decision_neutral_rate"]),
        "calibration_global": float(metrics["calibration"]["error_uncertainty_correlation"]),
        "circumplex_summary": {
            "opposite_violation_mean": float(circumplex_summary["opposite_violation_mean"]),
            "adjacent_support_mean": float(circumplex_summary["adjacent_support_mean"]),
        },
    }


def _format_run_scope(run_ids: list[str]) -> str:
    if not run_ids:
        return "configured frontier"

    ordered = sorted(set(run_ids))
    if len(ordered) == 1:
        return ordered[0]

    run_numbers = []
    for run_id in ordered:
        try:
            run_numbers.append(int(run_id.split("_", maxsplit=1)[1]))
        except (IndexError, ValueError):
            return ", ".join(ordered)

    is_contiguous = all(
        current == previous + 1
        for previous, current in zip(run_numbers, run_numbers[1:], strict=False)
    )
    if is_contiguous:
        return f"{ordered[0]}-{ordered[-1]}"
    return ", ".join(ordered)


def _evaluate_softmax_tau_candidate(
    *,
    artifact: ArtifactBundle,
    priors: np.ndarray,
    tau: float | np.ndarray,
    baseline_metrics: dict,
    config: dict,
    branch_name: str = STANDARD_SOFTMAX_BRANCH,
    prior_source: str = "train_split",
    prior_estimation: dict | None = None,
) -> dict:
    adjusted_probs = _softmax_logit_adjustment(
        artifact.raw_logits,
        priors,
        tau,
    )
    adjusted_expected_scores = _expected_scores(adjusted_probs)
    tau_mode = _softmax_tau_mode(tau)

    # Preserve the saved MC-dropout score surface for tau=0 so the baseline
    # candidate stays comparable to the frontier run YAML metrics.
    score_predictions = (
        artifact.baseline_mean_predictions if _tau_is_zero(tau) else adjusted_expected_scores
    )
    candidate_metrics = _compute_score_based_policy_metrics(
        targets=artifact.targets,
        score_predictions=score_predictions,
        uncertainties=artifact.uncertainties,
        probabilities=adjusted_probs,
    )
    policy_payload = {
        "type": "logit_adjustment",
        "branch_name": branch_name,
        "tau_mode": tau_mode,
        "prior_source": prior_source,
        "class_order": CLASS_LABELS.tolist(),
        "per_dimension_priors": {
            dimension: [float(value) for value in priors[idx].tolist()]
            for idx, dimension in enumerate(SCHWARTZ_VALUE_ORDER)
        },
    }
    if prior_estimation is not None:
        policy_payload["prior_estimation"] = deepcopy(prior_estimation)

    if tau_mode == "shared":
        tau_scalar = float(np.asarray(tau, dtype=np.float64))
        policy_name = (
            f"logit_adjustment_tau_{tau_scalar:.2f}"
            if branch_name == STANDARD_SOFTMAX_BRANCH
            else f"{branch_name}_tau_{tau_scalar:.2f}"
        )
        policy_payload["tau"] = tau_scalar
    else:
        tau_values = np.asarray(tau, dtype=np.float64)
        policy_name = f"{branch_name}_per_dimension_tau"
        policy_payload["per_dimension_tau"] = {
            dimension: float(tau_values[idx]) for idx, dimension in enumerate(SCHWARTZ_VALUE_ORDER)
        }

    candidate_record = _build_candidate_record(
        policy_name=policy_name,
        policy_family="softmax_logit_adjustment",
        policy_payload=policy_payload,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        config=config,
        split=artifact.split,
    )
    candidate_record["probabilities"] = adjusted_probs
    candidate_record["expected_scores"] = score_predictions
    candidate_record["predicted_classes"] = candidate_metrics["predicted_classes"]
    return candidate_record


def _resolve_softmax_branch_priors(
    *,
    branch_spec: dict,
    artifact: ArtifactBundle,
    train_priors: np.ndarray,
) -> np.ndarray:
    prior_source = branch_spec["prior_source"]
    if prior_source == "train_split":
        return train_priors
    if prior_source == "validation_posteriors":
        prior_estimation = branch_spec.get("prior_estimation") or {}
        method = prior_estimation.get("method", "mean_posterior")
        if method != "mean_posterior":
            raise ValueError(f"Unsupported effective-prior estimation method: {method}")
        return estimate_effective_priors_from_validation_posteriors(
            artifact,
            eps=float(prior_estimation.get("eps", 1e-9)),
        )

    raise ValueError(f"Unsupported softmax prior source: {prior_source}")


def _best_per_dimension_softmax_tau_policy(
    *,
    artifact: ArtifactBundle,
    priors: np.ndarray,
    baseline_metrics: dict,
    config: dict,
    branch_spec: dict,
) -> dict:
    tau_grid = [float(value) for value in branch_spec["tau_grid"]]
    tau_values = np.zeros(NUM_DIMS, dtype=np.float64)
    targets = artifact.targets

    for dim_idx, dimension in enumerate(SCHWARTZ_VALUE_ORDER):
        best_score: tuple[float, float, float] | None = None
        best_tau = 0.0
        baseline_qwk = float(baseline_metrics["qwk_per_dim"][dimension])
        target_dim = targets[:, dim_idx]

        for tau in tau_grid:
            adjusted_probs = _softmax_logit_adjustment(
                artifact.raw_logits[:, dim_idx : dim_idx + 1, :],
                priors[dim_idx : dim_idx + 1, :],
                float(tau),
            )[:, 0, :]
            score_predictions = (
                artifact.baseline_mean_predictions[:, dim_idx]
                if tau == 0.0
                else adjusted_probs @ CLASS_VALUES
            )
            predicted_classes = discretize_predictions(score_predictions.reshape(-1, 1)).reshape(-1)

            if len(np.unique(predicted_classes)) < 2 or len(np.unique(target_dim)) < 2:
                continue

            qwk = float(
                cohen_kappa_score(
                    target_dim.astype(int),
                    predicted_classes.astype(int),
                    weights="quadratic",
                    labels=[-1, 0, 1],
                )
            )
            if np.isnan(qwk):
                continue
            if np.isfinite(baseline_qwk) and qwk < baseline_qwk - float(config["selection_policy"]["max_qwk_drop"]):
                continue

            cm = confusion_matrix(
                target_dim.astype(int),
                predicted_classes.astype(int),
                labels=[-1, 0, 1],
            )
            minus_row = cm[0].sum()
            minus_recall = float(cm[0, 0] / minus_row) if minus_row > 0 else float("nan")
            score = (minus_recall, qwk, -float((predicted_classes == 0).mean()))
            if best_score is None or score > best_score:
                best_score = score
                best_tau = float(tau)

        tau_values[dim_idx] = best_tau

    return _evaluate_softmax_tau_candidate(
        artifact=artifact,
        priors=priors,
        tau=tau_values,
        baseline_metrics=baseline_metrics,
        config=config,
        branch_name=branch_spec["branch_name"],
        prior_source=branch_spec["prior_source"],
        prior_estimation=branch_spec.get("prior_estimation"),
    )


def _softmax_candidate_records(
    *,
    artifact: ArtifactBundle,
    train_priors: np.ndarray,
    config: dict,
) -> list[dict]:
    baseline_metrics = _compute_score_based_policy_metrics(
        targets=artifact.targets,
        score_predictions=artifact.baseline_mean_predictions,
        uncertainties=artifact.uncertainties,
        probabilities=artifact.class_probabilities,
    )
    records: list[dict] = []
    for branch_spec in _softmax_branch_specs_for_model(artifact.model_name, config):
        priors = _resolve_softmax_branch_priors(
            branch_spec=branch_spec,
            artifact=artifact,
            train_priors=train_priors,
        )
        tau_mode = branch_spec["tau_mode"]
        if tau_mode == "shared":
            for tau in branch_spec["tau_grid"]:
                records.append(
                    _evaluate_softmax_tau_candidate(
                        artifact=artifact,
                        priors=priors,
                        tau=float(tau),
                        baseline_metrics=baseline_metrics,
                        config=config,
                        branch_name=branch_spec["branch_name"],
                        prior_source=branch_spec["prior_source"],
                        prior_estimation=branch_spec.get("prior_estimation"),
                    )
                )
        elif tau_mode == "per_dimension":
            records.append(
                _best_per_dimension_softmax_tau_policy(
                    artifact=artifact,
                    priors=priors,
                    baseline_metrics=baseline_metrics,
                    config=config,
                    branch_spec=branch_spec,
                )
            )
        else:
            raise ValueError(f"Unsupported softmax tau mode: {tau_mode}")

    return records


def _softmax_branch_name(candidate_record: dict) -> str:
    return candidate_record["policy_payload"].get("branch_name", STANDARD_SOFTMAX_BRANCH)


def _select_softmax_branch_candidate(
    *,
    candidate_records: list[dict],
    branch_name: str,
    baseline_candidate: dict,
) -> tuple[dict, bool]:
    branch_records = [record for record in candidate_records if _softmax_branch_name(record) == branch_name]
    if not branch_records:
        return baseline_candidate, True

    eligible = [record for record in branch_records if record["eligible"]]
    if eligible:
        return max(eligible, key=_candidate_sort_tuple), False

    return baseline_candidate, True


def _apply_selected_softmax_policy(
    *,
    artifact: ArtifactBundle,
    baseline_metrics: dict,
    config: dict,
    selected_policy: dict,
) -> dict:
    policy_payload = selected_policy["policy_payload"]
    priors = np.array(
        [policy_payload["per_dimension_priors"][dimension] for dimension in SCHWARTZ_VALUE_ORDER],
        dtype=np.float64,
    )
    tau_mode = policy_payload.get("tau_mode", "shared")
    if tau_mode == "shared":
        tau: float | np.ndarray = float(policy_payload["tau"])
    else:
        tau = np.array(
            [float(policy_payload["per_dimension_tau"][dimension]) for dimension in SCHWARTZ_VALUE_ORDER],
            dtype=np.float64,
        )

    return _evaluate_softmax_tau_candidate(
        artifact=artifact,
        priors=priors,
        tau=tau,
        baseline_metrics=baseline_metrics,
        config=config,
        branch_name=policy_payload.get("branch_name", STANDARD_SOFTMAX_BRANCH),
        prior_source=policy_payload.get("prior_source", "train_split"),
        prior_estimation=policy_payload.get("prior_estimation"),
    )


def _shared_margin_candidates(
    *,
    artifact: ArtifactBundle,
    baseline_metrics: dict,
    config: dict,
) -> list[dict]:
    margin_grid = config["corn_threshold_policy"]["margin_grid"]
    records: list[dict] = []
    for minus_margin in margin_grid:
        for plus_margin in margin_grid:
            minus_margins = np.full(NUM_DIMS, float(minus_margin), dtype=np.float64)
            plus_margins = np.full(NUM_DIMS, float(plus_margin), dtype=np.float64)
            predicted_classes = _corn_margin_predictions(
                artifact.class_probabilities,
                minus_margins=minus_margins,
                plus_margins=plus_margins,
            )
            candidate_metrics = _compute_policy_metrics(
                targets=artifact.targets,
                predicted_classes=predicted_classes,
                expected_scores=artifact.baseline_mean_predictions,
                uncertainties=artifact.uncertainties,
                probabilities=artifact.class_probabilities,
            )
            candidate_record = _build_candidate_record(
                policy_name=f"corn_shared_margin_m{minus_margin:.2f}_p{plus_margin:.2f}",
                policy_family="corn_margin_threshold",
                policy_payload={
                    "type": "corn_probability_margin",
                    "shared_minus_margin": float(minus_margin),
                    "shared_plus_margin": float(plus_margin),
                },
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
                config=config,
                split=artifact.split,
            )
            candidate_record["probabilities"] = artifact.class_probabilities
            candidate_record["expected_scores"] = artifact.baseline_mean_predictions
            candidate_record["predicted_classes"] = predicted_classes
            records.append(candidate_record)
    return records


def _best_per_dimension_margin_policy(
    *,
    artifact: ArtifactBundle,
    baseline_metrics: dict,
    config: dict,
) -> dict:
    margin_grid = config["corn_threshold_policy"]["margin_grid"]
    minus_margins = np.zeros(NUM_DIMS, dtype=np.float64)
    plus_margins = np.zeros(NUM_DIMS, dtype=np.float64)

    prob_minus = artifact.class_probabilities[:, :, 0]
    prob_zero = artifact.class_probabilities[:, :, 1]
    prob_plus = artifact.class_probabilities[:, :, 2]
    targets = artifact.targets

    for dim_idx, dimension in enumerate(SCHWARTZ_VALUE_ORDER):
        best_score = None
        best_pair = (0.0, 0.0)
        target_dim = targets[:, dim_idx]
        baseline_qwk = baseline_metrics["qwk_per_dim"][dimension]

        for minus_margin in margin_grid:
            for plus_margin in margin_grid:
                pred_dim = np.zeros_like(target_dim)
                minus_mask = (prob_minus[:, dim_idx] >= (prob_zero[:, dim_idx] - minus_margin)) & (
                    prob_minus[:, dim_idx] >= prob_plus[:, dim_idx]
                )
                plus_mask = (prob_plus[:, dim_idx] > prob_minus[:, dim_idx]) & (
                    prob_plus[:, dim_idx] >= (prob_zero[:, dim_idx] - plus_margin)
                )
                pred_dim[minus_mask] = -1
                pred_dim[plus_mask] = 1

                if len(np.unique(pred_dim)) < 2 or len(np.unique(target_dim)) < 2:
                    continue
                qwk = float(
                    cohen_kappa_score(
                        target_dim.astype(int),
                        pred_dim.astype(int),
                        weights="quadratic",
                        labels=[-1, 0, 1],
                    )
                )
                if np.isnan(qwk):
                    continue
                if qwk < baseline_qwk - float(config["selection_policy"]["max_qwk_drop"]):
                    continue
                cm = confusion_matrix(
                    target_dim.astype(int),
                    pred_dim.astype(int),
                    labels=[-1, 0, 1],
                )
                minus_row = cm[0].sum()
                minus_recall = float(cm[0, 0] / minus_row) if minus_row > 0 else float("nan")
                score = (
                    minus_recall,
                    float(qwk),
                    -float((pred_dim == 0).mean()),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (float(minus_margin), float(plus_margin))

        minus_margins[dim_idx], plus_margins[dim_idx] = best_pair

    predicted_classes = _corn_margin_predictions(
        artifact.class_probabilities,
        minus_margins=minus_margins,
        plus_margins=plus_margins,
    )
    candidate_metrics = _compute_policy_metrics(
        targets=artifact.targets,
        predicted_classes=predicted_classes,
        expected_scores=artifact.baseline_mean_predictions,
        uncertainties=artifact.uncertainties,
        probabilities=artifact.class_probabilities,
    )
    candidate_record = _build_candidate_record(
        policy_name="corn_per_dimension_margin",
        policy_family="corn_margin_threshold",
        policy_payload={
            "type": "corn_probability_margin",
            "per_dimension_minus_margin": {
                dimension: float(minus_margins[idx]) for idx, dimension in enumerate(SCHWARTZ_VALUE_ORDER)
            },
            "per_dimension_plus_margin": {
                dimension: float(plus_margins[idx]) for idx, dimension in enumerate(SCHWARTZ_VALUE_ORDER)
            },
        },
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        config=config,
        split=artifact.split,
    )
    candidate_record["probabilities"] = artifact.class_probabilities
    candidate_record["expected_scores"] = artifact.baseline_mean_predictions
    candidate_record["predicted_classes"] = predicted_classes
    return candidate_record


def _evaluate_corn_policy(
    *,
    artifact: ArtifactBundle,
    policy_name: str,
    minus_margins: np.ndarray,
    plus_margins: np.ndarray,
    baseline_metrics: dict,
    config: dict,
    policy_payload: dict,
) -> dict:
    predicted_classes = _corn_margin_predictions(
        artifact.class_probabilities,
        minus_margins=minus_margins,
        plus_margins=plus_margins,
    )
    candidate_metrics = _compute_policy_metrics(
        targets=artifact.targets,
        predicted_classes=predicted_classes,
        expected_scores=artifact.baseline_mean_predictions,
        uncertainties=artifact.uncertainties,
        probabilities=artifact.class_probabilities,
    )
    candidate_record = _build_candidate_record(
        policy_name=policy_name,
        policy_family="corn_margin_threshold",
        policy_payload=policy_payload,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        config=config,
        split=artifact.split,
    )
    candidate_record["probabilities"] = artifact.class_probabilities
    candidate_record["expected_scores"] = artifact.baseline_mean_predictions
    candidate_record["predicted_classes"] = predicted_classes
    return candidate_record


def _corn_candidate_records(artifact: ArtifactBundle, config: dict) -> list[dict]:
    baseline_metrics = _compute_policy_metrics(
        targets=artifact.targets,
        predicted_classes=artifact.baseline_predicted_classes,
        expected_scores=artifact.baseline_mean_predictions,
        uncertainties=artifact.uncertainties,
        probabilities=artifact.class_probabilities,
    )
    records = [
        _baseline_candidate_record(
            policy_name="artifact_argmax",
            policy_family="corn_margin_threshold",
            policy_payload={"type": "artifact_argmax"},
            artifact=artifact,
            config=config,
        )
    ]
    records[0]["probabilities"] = artifact.class_probabilities
    records[0]["expected_scores"] = artifact.baseline_mean_predictions
    records[0]["predicted_classes"] = artifact.baseline_predicted_classes

    records.extend(
        _shared_margin_candidates(
            artifact=artifact,
            baseline_metrics=baseline_metrics,
            config=config,
        )
    )
    if config["corn_threshold_policy"]["allow_per_dimension_override"]:
        records.append(
            _best_per_dimension_margin_policy(
                artifact=artifact,
                baseline_metrics=baseline_metrics,
                config=config,
            )
        )
    return records


def _apply_selected_corn_policy(
    *,
    artifact: ArtifactBundle,
    baseline_metrics: dict,
    config: dict,
    selected_policy: dict,
) -> dict:
    policy_payload = selected_policy["policy_payload"]
    if policy_payload["type"] == "artifact_argmax":
        candidate_record = _baseline_candidate_record(
            policy_name="artifact_argmax",
            policy_family="corn_margin_threshold",
            policy_payload=policy_payload,
            artifact=artifact,
            config=config,
        )
        candidate_record["probabilities"] = artifact.class_probabilities
        candidate_record["expected_scores"] = artifact.baseline_mean_predictions
        candidate_record["predicted_classes"] = artifact.baseline_predicted_classes
        return candidate_record

    if "shared_minus_margin" in policy_payload:
        minus_margins = np.full(NUM_DIMS, float(policy_payload["shared_minus_margin"]), dtype=np.float64)
        plus_margins = np.full(NUM_DIMS, float(policy_payload["shared_plus_margin"]), dtype=np.float64)
    else:
        minus_margins = np.array(
            [float(policy_payload["per_dimension_minus_margin"][dimension]) for dimension in SCHWARTZ_VALUE_ORDER],
            dtype=np.float64,
        )
        plus_margins = np.array(
            [float(policy_payload["per_dimension_plus_margin"][dimension]) for dimension in SCHWARTZ_VALUE_ORDER],
            dtype=np.float64,
        )

    return _evaluate_corn_policy(
        artifact=artifact,
        policy_name=selected_policy["policy_name"],
        minus_margins=minus_margins,
        plus_margins=plus_margins,
        baseline_metrics=baseline_metrics,
        config=config,
        policy_payload=policy_payload,
    )


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _iqr(values: list[float]) -> float:
    if not values:
        return float("nan")
    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    return q3 - q1


def _summary_metrics(records: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record["model_name"]]["qwk_mean"].append(record["test_metrics"]["qwk_mean"])
        grouped[record["model_name"]]["recall_minus1"].append(record["test_metrics"]["recall_minus1"])
        grouped[record["model_name"]]["minority_recall_mean"].append(
            record["test_metrics"]["minority_recall_mean"]
        )
        grouped[record["model_name"]]["hedging_mean"].append(record["test_metrics"]["hedging_mean"])
        grouped[record["model_name"]]["decision_neutral_rate"].append(
            record["test_metrics"]["decision_neutral_rate"]
        )
        grouped[record["model_name"]]["calibration_global"].append(
            record["test_metrics"]["calibration_global"]
        )
        grouped[record["model_name"]]["opposite_violation_mean"].append(
            record["test_metrics"]["circumplex_summary"]["opposite_violation_mean"]
        )
        grouped[record["model_name"]]["adjacent_support_mean"].append(
            record["test_metrics"]["circumplex_summary"]["adjacent_support_mean"]
        )

    summary = {}
    for model_name, metrics in grouped.items():
        summary[model_name] = {
            f"{metric_name}_median": float(median(metric_values))
            for metric_name, metric_values in metrics.items()
        }
        summary[model_name].update(
            {
                f"{metric_name}_iqr": float(_iqr(metric_values))
                for metric_name, metric_values in metrics.items()
            }
        )
    return summary


def _family_delta_summary(records: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        family = record["selected_policy"]["policy_family"]
        baseline = record["baseline_test_metrics"]
        tuned = record["test_metrics"]
        grouped[family]["recall_minus1_delta"].append(tuned["recall_minus1"] - baseline["recall_minus1"])
        grouped[family]["qwk_mean_delta"].append(tuned["qwk_mean"] - baseline["qwk_mean"])
        grouped[family]["hedging_mean_delta"].append(tuned["hedging_mean"] - baseline["hedging_mean"])
        grouped[family]["opposite_violation_mean_delta"].append(
            tuned["circumplex_summary"]["opposite_violation_mean"]
            - baseline["circumplex_summary"]["opposite_violation_mean"]
        )
        grouped[family]["adjacent_support_mean_delta"].append(
            tuned["circumplex_summary"]["adjacent_support_mean"]
            - baseline["circumplex_summary"]["adjacent_support_mean"]
        )

    return {
        family: {
            "median_recall_minus1_delta": float(median(values["recall_minus1_delta"])),
            "median_qwk_mean_delta": float(median(values["qwk_mean_delta"])),
            "median_hedging_mean_delta": float(median(values["hedging_mean_delta"])),
            "median_opposite_violation_mean_delta": float(
                median(values["opposite_violation_mean_delta"])
            ),
            "median_adjacent_support_mean_delta": float(
                median(values["adjacent_support_mean_delta"])
            ),
        }
        for family, values in grouped.items()
    }


def _softmax_branch_delta_summary(records: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        branch_comparison = record.get("softmax_branch_comparison")
        if not branch_comparison:
            continue

        baseline_metrics = branch_comparison["baseline"]["test_metrics"]
        for branch_name in (STANDARD_SOFTMAX_BRANCH, EFFECTIVE_PRIOR_SOFTMAX_BRANCH):
            branch_metrics = branch_comparison.get(branch_name, {}).get("test_metrics")
            if branch_metrics is None:
                continue

            grouped[branch_name]["recall_minus1_delta"].append(
                branch_metrics["recall_minus1"] - baseline_metrics["recall_minus1"]
            )
            grouped[branch_name]["qwk_mean_delta"].append(
                branch_metrics["qwk_mean"] - baseline_metrics["qwk_mean"]
            )
            grouped[branch_name]["hedging_mean_delta"].append(
                branch_metrics["hedging_mean"] - baseline_metrics["hedging_mean"]
            )
            grouped[branch_name]["opposite_violation_mean_delta"].append(
                branch_metrics["circumplex_summary"]["opposite_violation_mean"]
                - baseline_metrics["circumplex_summary"]["opposite_violation_mean"]
            )
            grouped[branch_name]["adjacent_support_mean_delta"].append(
                branch_metrics["circumplex_summary"]["adjacent_support_mean"]
                - baseline_metrics["circumplex_summary"]["adjacent_support_mean"]
            )

    return {
        branch_name: {
            "median_recall_minus1_delta": float(median(values["recall_minus1_delta"])),
            "median_qwk_mean_delta": float(median(values["qwk_mean_delta"])),
            "median_hedging_mean_delta": float(median(values["hedging_mean_delta"])),
            "median_opposite_violation_mean_delta": float(
                median(values["opposite_violation_mean_delta"])
            ),
            "median_adjacent_support_mean_delta": float(
                median(values["adjacent_support_mean_delta"])
            ),
        }
        for branch_name, values in grouped.items()
    }


def _best_softmax_family(
    summary_by_model: dict[str, dict[str, float]],
    config: dict,
) -> str | None:
    softmax_targets = set(config["softmax_logit_adjustment"].get("target_models", []))
    effective_cfg = config["softmax_logit_adjustment"].get("effective_prior_branch", {})
    softmax_targets.update(effective_cfg.get("target_models", []))
    softmax_models = [name for name in summary_by_model if name in softmax_targets]
    if not softmax_models:
        return None
    return max(
        softmax_models,
        key=lambda model_name: (
            summary_by_model[model_name]["recall_minus1_median"],
            summary_by_model[model_name]["qwk_mean_median"],
            summary_by_model[model_name]["calibration_global_median"],
            -summary_by_model[model_name]["decision_neutral_rate_median"],
        ),
    )


def _softmax_branch_comparison_entry(
    *,
    branch_key: str,
    validation_candidate: dict,
    test_candidate: dict,
    used_baseline_fallback: bool,
) -> dict:
    return {
        "label": SOFTMAX_BRANCH_LABELS[branch_key],
        "selected_policy": {
            "policy_name": validation_candidate["policy_name"],
            "policy_family": validation_candidate["policy_family"],
            "policy_payload": deepcopy(validation_candidate["policy_payload"]),
        },
        "selection_score": deepcopy(validation_candidate["selection_score"]),
        "validation_metrics": _compact_metrics(validation_candidate["metrics"]),
        "test_metrics": _compact_metrics(test_candidate["metrics"]),
        "used_baseline_fallback": bool(used_baseline_fallback),
    }


def _render_report(summary: dict, config: dict) -> str:
    tuned_records = summary["tuned_runs"]
    summary_by_model = summary["summary_by_model"]
    family_summary = summary["family_delta_summary"]
    softmax_branch_summary = summary.get("softmax_branch_delta_summary", {})
    best_softmax_family = summary["recommended_softmax_base"]
    best_softmax_family_str = best_softmax_family or "N/A"
    generated_date = str(summary["generated_at"])[:10]
    run_ids = sorted({record["run_id"] for record in tuned_records})
    run_scope = _format_run_scope(run_ids)
    report_title = config.get("report_title") or (
        f"Experiment Review — {generated_date} — twinkl-681.3 post-hoc boundary optimization"
    )
    report_scope_note = config.get("report_scope_note")
    model_order = config.get("summary_model_order") or config["models"]
    recommended_model_label = config.get(
        "recommended_model_label",
        "Recommended softmax base for `twinkl-681.4`",
    )
    has_softmax_branch_comparison = any(record.get("softmax_branch_comparison") for record in tuned_records)
    standard_branch_delta = softmax_branch_summary.get(STANDARD_SOFTMAX_BRANCH, {})
    effective_branch_delta = softmax_branch_summary.get(EFFECTIVE_PRIOR_SOFTMAX_BRANCH, {})

    if has_softmax_branch_comparison:
        standard_recall_delta = standard_branch_delta.get("median_recall_minus1_delta", float("nan"))
        standard_qwk_delta = standard_branch_delta.get("median_qwk_mean_delta", float("nan"))
        effective_recall_delta = effective_branch_delta.get("median_recall_minus1_delta", float("nan"))
        effective_qwk_delta = effective_branch_delta.get("median_qwk_mean_delta", float("nan"))
        if (
            np.isfinite(effective_recall_delta)
            and np.isfinite(effective_qwk_delta)
            and np.isfinite(standard_recall_delta)
            and np.isfinite(standard_qwk_delta)
            and effective_recall_delta > standard_recall_delta
            and effective_qwk_delta >= standard_qwk_delta
        ):
            conclusion_text = (
                "The effective-prior + per-dimension tau branch is the stronger post-hoc variant on the guarded "
                "median test comparison, so the current frontier still has some post-hoc headroom."
            )
        else:
            conclusion_text = (
                "The effective-prior + per-dimension tau branch did not beat the standard Menon control on the "
                "guarded median test comparison, and neither branch produced a clean enough package to justify a "
                "frontier change. Treat this as evidence that the current post-hoc line is likely exhausted for the "
                "active frontier."
            )
    else:
        conclusion_text = (
            f"`{best_softmax_family_str}` is the strongest softmax-family model under the configured recall-first guarded selector. "
            "The comparison above also shows whether softmax logit adjustment or CORN boundary tuning extracted more validation-disciplined recall gains from the configured frontier."
            if best_softmax_family is not None
            else "No softmax-family model qualified for recommendation under the configured selector."
        )

    lines = [
        f"# {report_title}",
        "",
        "## Scope",
        "",
        f"Validation-only tuning on the corrected-split frontier `{run_scope}` using existing selected-checkpoint artifacts only. No retraining was performed.",
        "",
    ]
    if report_scope_note:
        lines.extend([report_scope_note, ""])

    lines.extend(
        [
            "## Test Summary",
            "",
            "| Run | Model | Selected policy | Test QWK | Test recall_-1 | Test MinR | Test hedging | Test neutral rate | Test calibration | OppV | AdjS |",
            "|-----|-------|-----------------|---------:|---------------:|----------:|-------------:|------------------:|-----------------:|-----:|-----:|",
        ]
    )

    for record in tuned_records:
        test_metrics = record["test_metrics"]
        lines.append(
            "| "
            f"{record['run_id']} | {record['model_name']} | {record['selected_policy']['policy_name']} | "
            f"{test_metrics['qwk_mean']:.3f} | {test_metrics['recall_minus1']:.3f} | "
            f"{test_metrics['minority_recall_mean']:.3f} | {test_metrics['hedging_mean']:.3f} | "
            f"{test_metrics['decision_neutral_rate']:.3f} | "
            f"{test_metrics['calibration_global']:.3f} | "
            f"{test_metrics['circumplex_summary']['opposite_violation_mean']:.3f} | "
            f"{test_metrics['circumplex_summary']['adjacent_support_mean']:.3f} |"
        )

    if has_softmax_branch_comparison:
        lines.extend(
            [
                "",
                "## Branch Comparison",
                "",
                "| Run | Variant | Validation policy | Test QWK | Test recall_-1 | Test MinR | Test hedging | Test calibration | OppV | AdjS |",
                "|-----|---------|-------------------|---------:|---------------:|----------:|-------------:|-----------------:|-----:|-----:|",
            ]
        )
        for record in tuned_records:
            branch_comparison = record.get("softmax_branch_comparison") or {}
            for branch_key in ("baseline", STANDARD_SOFTMAX_BRANCH, EFFECTIVE_PRIOR_SOFTMAX_BRANCH):
                if branch_key not in branch_comparison:
                    continue
                entry = branch_comparison[branch_key]
                test_metrics = entry["test_metrics"]
                policy_name = entry["selected_policy"]["policy_name"]
                if entry.get("used_baseline_fallback"):
                    policy_name = f"{policy_name} (baseline fallback)"
                lines.append(
                    "| "
                    f"{record['run_id']} | {entry['label']} | {policy_name} | "
                    f"{test_metrics['qwk_mean']:.3f} | {test_metrics['recall_minus1']:.3f} | "
                    f"{test_metrics['minority_recall_mean']:.3f} | {test_metrics['hedging_mean']:.3f} | "
                    f"{test_metrics['calibration_global']:.3f} | "
                    f"{test_metrics['circumplex_summary']['opposite_violation_mean']:.3f} | "
                    f"{test_metrics['circumplex_summary']['adjacent_support_mean']:.3f} |"
                )

    lines.extend(
        [
            "",
            "## Median / IQR by Family",
            "",
            "| Model | Median QWK | IQR QWK | Median recall_-1 | IQR recall_-1 | Median MinR | Median hedging | IQR hedging | Median neutral rate | Median calibration | Median OppV | IQR OppV | Median AdjS | IQR AdjS |",
            "|-------|-----------:|--------:|-----------------:|--------------:|------------:|---------------:|------------:|--------------------:|-------------------:|------------:|---------:|------------:|---------:|",
        ]
    )

    for model_name in model_order:
        if model_name not in summary_by_model:
            continue
        stats = summary_by_model[model_name]
        lines.append(
            "| "
            f"{model_name} | {stats['qwk_mean_median']:.3f} | {stats['qwk_mean_iqr']:.3f} | "
            f"{stats['recall_minus1_median']:.3f} | {stats['recall_minus1_iqr']:.3f} | "
            f"{stats['minority_recall_mean_median']:.3f} | {stats['hedging_mean_median']:.3f} | "
            f"{stats['hedging_mean_iqr']:.3f} | {stats['decision_neutral_rate_median']:.3f} | "
            f"{stats['calibration_global_median']:.3f} | "
            f"{stats['opposite_violation_mean_median']:.3f} | {stats['opposite_violation_mean_iqr']:.3f} | "
            f"{stats['adjacent_support_mean_median']:.3f} | {stats['adjacent_support_mean_iqr']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Policy-Level Takeaways",
            "",
            f"- Softmax logit adjustment median delta: recall_-1 {family_summary.get('softmax_logit_adjustment', {}).get('median_recall_minus1_delta', float('nan')):.3f}, "
            f"QWK {family_summary.get('softmax_logit_adjustment', {}).get('median_qwk_mean_delta', float('nan')):.3f}, "
            f"hedging {family_summary.get('softmax_logit_adjustment', {}).get('median_hedging_mean_delta', float('nan')):.3f}, "
            f"OppV {family_summary.get('softmax_logit_adjustment', {}).get('median_opposite_violation_mean_delta', float('nan')):.3f}, "
            f"AdjS {family_summary.get('softmax_logit_adjustment', {}).get('median_adjacent_support_mean_delta', float('nan')):.3f}.",
            f"- CORN threshold tuning median delta: recall_-1 {family_summary.get('corn_margin_threshold', {}).get('median_recall_minus1_delta', float('nan')):.3f}, "
            f"QWK {family_summary.get('corn_margin_threshold', {}).get('median_qwk_mean_delta', float('nan')):.3f}, "
            f"hedging {family_summary.get('corn_margin_threshold', {}).get('median_hedging_mean_delta', float('nan')):.3f}, "
            f"OppV {family_summary.get('corn_margin_threshold', {}).get('median_opposite_violation_mean_delta', float('nan')):.3f}, "
            f"AdjS {family_summary.get('corn_margin_threshold', {}).get('median_adjacent_support_mean_delta', float('nan')):.3f}.",
            f"- Standard Menon median delta vs baseline: recall_-1 {softmax_branch_summary.get(STANDARD_SOFTMAX_BRANCH, {}).get('median_recall_minus1_delta', float('nan')):.3f}, "
            f"QWK {softmax_branch_summary.get(STANDARD_SOFTMAX_BRANCH, {}).get('median_qwk_mean_delta', float('nan')):.3f}, "
            f"hedging {softmax_branch_summary.get(STANDARD_SOFTMAX_BRANCH, {}).get('median_hedging_mean_delta', float('nan')):.3f}, "
            f"OppV {softmax_branch_summary.get(STANDARD_SOFTMAX_BRANCH, {}).get('median_opposite_violation_mean_delta', float('nan')):.3f}, "
            f"AdjS {softmax_branch_summary.get(STANDARD_SOFTMAX_BRANCH, {}).get('median_adjacent_support_mean_delta', float('nan')):.3f}.",
            f"- Effective-prior + per-dimension tau median delta vs baseline: recall_-1 {softmax_branch_summary.get(EFFECTIVE_PRIOR_SOFTMAX_BRANCH, {}).get('median_recall_minus1_delta', float('nan')):.3f}, "
            f"QWK {softmax_branch_summary.get(EFFECTIVE_PRIOR_SOFTMAX_BRANCH, {}).get('median_qwk_mean_delta', float('nan')):.3f}, "
            f"hedging {softmax_branch_summary.get(EFFECTIVE_PRIOR_SOFTMAX_BRANCH, {}).get('median_hedging_mean_delta', float('nan')):.3f}, "
            f"OppV {softmax_branch_summary.get(EFFECTIVE_PRIOR_SOFTMAX_BRANCH, {}).get('median_opposite_violation_mean_delta', float('nan')):.3f}, "
            f"AdjS {softmax_branch_summary.get(EFFECTIVE_PRIOR_SOFTMAX_BRANCH, {}).get('median_adjacent_support_mean_delta', float('nan')):.3f}.",
            f"- {recommended_model_label}: `{best_softmax_family_str}`.",
            "",
            "## Conclusion",
            "",
            conclusion_text,
            "",
        ]
    )

    return "\n".join(lines)


def run_posthoc(config: dict, *, repo_root: Path | None = None) -> dict:
    """Execute validation-only post-hoc tuning across the configured frontier."""
    repo_root = repo_root or Path.cwd()
    labels_path = _resolve_path(repo_root, config["labels_path"])
    wrangled_dir = _resolve_path(repo_root, config["wrangled_dir"])
    run_specs = load_run_specs(config, repo_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_run_prefix = config.get("artifact_run_prefix", "posthoc_twinkl_681_3")
    output_root = _resolve_path(repo_root, config["artifact_root"]) / f"{artifact_run_prefix}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    tuned_runs: list[dict] = []

    for source_run in run_specs:
        validation_bundle = load_artifact_bundle(source_run.validation_outputs_path)
        test_bundle = load_artifact_bundle(source_run.test_outputs_path)
        policy_family = _policy_family_for_model(source_run.model_name, config)
        softmax_branch_comparison = None

        if policy_family == "softmax":
            train_priors = reconstruct_train_priors(
                labels_path=str(labels_path),
                wrangled_dir=str(wrangled_dir),
                train_ratio=source_run.train_ratio,
                val_ratio=source_run.val_ratio,
                split_seed=source_run.split_seed,
            )
            validation_candidates = _softmax_candidate_records(
                artifact=validation_bundle,
                train_priors=train_priors,
                config=config,
            )
            baseline_test_metrics = _compute_score_based_policy_metrics(
                targets=test_bundle.targets,
                score_predictions=test_bundle.baseline_mean_predictions,
                uncertainties=test_bundle.uncertainties,
                probabilities=test_bundle.class_probabilities,
            )
            baseline_policy_name = "logit_adjustment_tau_0.00"
        elif policy_family == "corn":
            validation_candidates = _corn_candidate_records(validation_bundle, config)
            baseline_test_metrics = _compute_policy_metrics(
                targets=test_bundle.targets,
                predicted_classes=test_bundle.baseline_predicted_classes,
                expected_scores=test_bundle.baseline_mean_predictions,
                uncertainties=test_bundle.uncertainties,
                probabilities=test_bundle.class_probabilities,
            )
            baseline_policy_name = "artifact_argmax"
        else:
            raise ValueError(f"Unsupported post-hoc policy family: {policy_family}")

        baseline_validation_candidate = next(
            record for record in validation_candidates if record["policy_name"] == baseline_policy_name
        )

        if policy_family == "softmax":
            standard_validation_candidate, standard_used_baseline = _select_softmax_branch_candidate(
                candidate_records=validation_candidates,
                branch_name=STANDARD_SOFTMAX_BRANCH,
                baseline_candidate=baseline_validation_candidate,
            )
            effective_branch_present = any(
                _softmax_branch_name(record) == EFFECTIVE_PRIOR_SOFTMAX_BRANCH
                for record in validation_candidates
            )
            effective_validation_candidate = baseline_validation_candidate
            effective_used_baseline = True
            if effective_branch_present:
                effective_validation_candidate, effective_used_baseline = _select_softmax_branch_candidate(
                    candidate_records=validation_candidates,
                    branch_name=EFFECTIVE_PRIOR_SOFTMAX_BRANCH,
                    baseline_candidate=baseline_validation_candidate,
                )

            branch_selection_candidates = [baseline_validation_candidate, standard_validation_candidate]
            if effective_branch_present:
                branch_selection_candidates.append(effective_validation_candidate)
            selected_validation_candidate = _pick_best_candidate(
                candidate_records=branch_selection_candidates,
                baseline_policy_name=baseline_policy_name,
            )

            baseline_test_candidate = _apply_selected_softmax_policy(
                artifact=test_bundle,
                baseline_metrics=baseline_test_metrics,
                config=config,
                selected_policy=baseline_validation_candidate,
            )
            standard_test_candidate = _apply_selected_softmax_policy(
                artifact=test_bundle,
                baseline_metrics=baseline_test_metrics,
                config=config,
                selected_policy=standard_validation_candidate,
            )
            effective_test_candidate = None
            if effective_branch_present:
                effective_test_candidate = _apply_selected_softmax_policy(
                    artifact=test_bundle,
                    baseline_metrics=baseline_test_metrics,
                    config=config,
                    selected_policy=effective_validation_candidate,
                )
            selected_test_candidate = _apply_selected_softmax_policy(
                artifact=test_bundle,
                baseline_metrics=baseline_test_metrics,
                config=config,
                selected_policy=selected_validation_candidate,
            )

            softmax_branch_comparison = {
                "baseline": _softmax_branch_comparison_entry(
                    branch_key="baseline",
                    validation_candidate=baseline_validation_candidate,
                    test_candidate=baseline_test_candidate,
                    used_baseline_fallback=False,
                ),
                STANDARD_SOFTMAX_BRANCH: _softmax_branch_comparison_entry(
                    branch_key=STANDARD_SOFTMAX_BRANCH,
                    validation_candidate=standard_validation_candidate,
                    test_candidate=standard_test_candidate,
                    used_baseline_fallback=standard_used_baseline,
                ),
            }
            if effective_branch_present and effective_test_candidate is not None:
                softmax_branch_comparison[EFFECTIVE_PRIOR_SOFTMAX_BRANCH] = _softmax_branch_comparison_entry(
                    branch_key=EFFECTIVE_PRIOR_SOFTMAX_BRANCH,
                    validation_candidate=effective_validation_candidate,
                    test_candidate=effective_test_candidate,
                    used_baseline_fallback=effective_used_baseline,
                )
        else:
            selected_validation_candidate = _pick_best_candidate(
                candidate_records=validation_candidates,
                baseline_policy_name=baseline_policy_name,
            )
            selected_test_candidate = _apply_selected_corn_policy(
                artifact=test_bundle,
                baseline_metrics=baseline_test_metrics,
                config=config,
                selected_policy=selected_validation_candidate,
            )
            baseline_test_candidate = _baseline_candidate_record(
                policy_name="artifact_argmax",
                policy_family="corn_margin_threshold",
                policy_payload={"type": "artifact_argmax"},
                artifact=test_bundle,
                config=config,
            )
            baseline_test_candidate["probabilities"] = test_bundle.class_probabilities
            baseline_test_candidate["expected_scores"] = test_bundle.baseline_mean_predictions
            baseline_test_candidate["predicted_classes"] = test_bundle.baseline_predicted_classes

        run_output_dir = output_root / f"{source_run.run_id}_{source_run.model_name}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        validation_sweep_df = pl.DataFrame(
            [
                _candidate_frame_row(
                    source_run=source_run,
                    split=validation_bundle.split,
                    policy_name=record["policy_name"],
                    policy_family=record["policy_family"],
                    candidate_record=record,
                )
                for record in validation_candidates
            ]
        )
        validation_sweep_path = run_output_dir / "validation_sweep.parquet"
        validation_sweep_df.write_parquet(validation_sweep_path)

        tuned_validation_df = _artifact_output_frame(
            metadata_rows=validation_bundle.metadata_rows,
            predicted_classes=selected_validation_candidate["predicted_classes"],
            expected_scores=selected_validation_candidate["expected_scores"],
            uncertainties=validation_bundle.uncertainties,
            raw_logits=validation_bundle.raw_logits,
            probabilities=selected_validation_candidate["probabilities"],
            targets=validation_bundle.targets,
            split=validation_bundle.split,
            model_name=source_run.model_name,
            policy_name=selected_validation_candidate["policy_name"],
        )
        tuned_test_df = _artifact_output_frame(
            metadata_rows=test_bundle.metadata_rows,
            predicted_classes=selected_test_candidate["predicted_classes"],
            expected_scores=selected_test_candidate["expected_scores"],
            uncertainties=test_bundle.uncertainties,
            raw_logits=test_bundle.raw_logits,
            probabilities=selected_test_candidate["probabilities"],
            targets=test_bundle.targets,
            split=test_bundle.split,
            model_name=source_run.model_name,
            policy_name=selected_test_candidate["policy_name"],
        )
        tuned_validation_path = run_output_dir / "tuned_validation_outputs.parquet"
        tuned_test_path = run_output_dir / "tuned_test_outputs.parquet"
        tuned_validation_df.write_parquet(tuned_validation_path)
        tuned_test_df.write_parquet(tuned_test_path)

        selected_policy = {
            "run_id": source_run.run_id,
            "model_name": source_run.model_name,
            "selection_policy": config["selection_policy"],
            "selection_source": "validation_only_recall_first_guarded",
            "selected_policy": {
                "policy_name": selected_validation_candidate["policy_name"],
                "policy_family": selected_validation_candidate["policy_family"],
                "policy_payload": selected_validation_candidate["policy_payload"],
            },
            "selection_score": selected_validation_candidate["selection_score"],
            "baseline_validation_metrics": _compact_metrics(baseline_validation_candidate["metrics"]),
            "selected_validation_metrics": _compact_metrics(selected_validation_candidate["metrics"]),
            "untouched_test_metrics": {
                "baseline": _compact_metrics(baseline_test_candidate["metrics"]),
                "selected": _compact_metrics(selected_test_candidate["metrics"]),
            },
            "circumplex": {
                "validation": {
                    "baseline": deepcopy(baseline_validation_candidate["metrics"]["circumplex"]),
                    "selected": deepcopy(selected_validation_candidate["metrics"]["circumplex"]),
                },
                "test": {
                    "baseline": deepcopy(baseline_test_candidate["metrics"]["circumplex"]),
                    "selected": deepcopy(selected_test_candidate["metrics"]["circumplex"]),
                },
            },
            "artifacts": {
                "source_validation_outputs": str(source_run.validation_outputs_path),
                "source_test_outputs": str(source_run.test_outputs_path),
                "validation_sweep": str(validation_sweep_path),
                "tuned_validation_outputs": str(tuned_validation_path),
                "tuned_test_outputs": str(tuned_test_path),
            },
        }
        if selected_validation_candidate["policy_family"] == "softmax_logit_adjustment":
            selected_policy["logit_policy"] = deepcopy(selected_validation_candidate["policy_payload"])
        else:
            selected_policy["threshold_policy"] = deepcopy(selected_validation_candidate["policy_payload"])
        if softmax_branch_comparison is not None:
            selected_policy["softmax_branch_comparison"] = deepcopy(softmax_branch_comparison)
        selected_policy_path = run_output_dir / "selected_policy.yaml"
        _write_yaml(selected_policy_path, selected_policy)

        metrics_summary = {
            "run_id": source_run.run_id,
            "model_name": source_run.model_name,
            "baseline_validation_metrics": selected_policy["baseline_validation_metrics"],
            "selected_validation_metrics": selected_policy["selected_validation_metrics"],
            "baseline_test_metrics": selected_policy["untouched_test_metrics"]["baseline"],
            "selected_test_metrics": selected_policy["untouched_test_metrics"]["selected"],
        }
        if softmax_branch_comparison is not None:
            metrics_summary["softmax_branch_comparison"] = deepcopy(softmax_branch_comparison)
        metrics_summary_path = run_output_dir / "metrics_summary.yaml"
        _write_yaml(metrics_summary_path, metrics_summary)

        tuned_runs.append(
            {
                "run_id": source_run.run_id,
                "model_name": source_run.model_name,
                "selected_policy": selected_policy["selected_policy"],
                "selection_score": selected_policy["selection_score"],
                "baseline_validation_metrics": selected_policy["baseline_validation_metrics"],
                "validation_metrics": selected_policy["selected_validation_metrics"],
                "baseline_test_metrics": selected_policy["untouched_test_metrics"]["baseline"],
                "test_metrics": selected_policy["untouched_test_metrics"]["selected"],
                "artifact_dir": str(run_output_dir),
                "selected_policy_path": str(selected_policy_path),
                "metrics_summary_path": str(metrics_summary_path),
                "softmax_branch_comparison": deepcopy(softmax_branch_comparison),
            }
        )

    summary_by_model = _summary_metrics(tuned_runs)
    family_delta_summary = _family_delta_summary(tuned_runs)
    softmax_branch_delta_summary = _softmax_branch_delta_summary(tuned_runs)
    recommended_softmax_base = _best_softmax_family(summary_by_model, config)
    summary = {
        "generated_at": datetime.now().isoformat(),
        "output_root": str(output_root),
        "selection_policy": config["selection_policy"],
        "tuned_runs": tuned_runs,
        "summary_by_model": summary_by_model,
        "family_delta_summary": family_delta_summary,
        "softmax_branch_delta_summary": softmax_branch_delta_summary,
        "recommended_softmax_base": recommended_softmax_base,
    }

    summary_path = output_root / "summary.yaml"
    _write_yaml(summary_path, summary)

    report_body = _render_report(summary, config)
    report_output_path = _resolve_path(repo_root, config["report_path"])
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(report_body, encoding="utf-8")

    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_output_path)
    return summary


def _print_summary(summary: dict) -> None:
    print(f"Post-hoc artifact root: {summary['output_root']}")
    print(f"Summary YAML: {summary['summary_path']}")
    print(f"Report:       {summary['report_path']}")
    print()
    print(
        f"{'Run':<10s} {'Model':<18s} {'Selected policy':<30s} "
        f"{'Test QWK':>10s} {'Test R-1':>10s} {'Test MinR':>12s}"
    )
    print("-" * 98)
    for record in summary["tuned_runs"]:
        print(
            f"{record['run_id']:<10s} {record['model_name']:<18s} "
            f"{record['selected_policy']['policy_name']:<30s} "
            f"{record['test_metrics']['qwk_mean']:>10.3f} "
            f"{record['test_metrics']['recall_minus1']:>10.3f} "
            f"{record['test_metrics']['minority_recall_mean']:>12.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run validation-only post-hoc boundary optimization for ordinal VIF artifacts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiments/vif/twinkl_681_3.yaml",
        help="YAML config for the post-hoc run.",
    )
    args = parser.parse_args()

    summary = run_posthoc(load_config(args.config))
    _print_summary(summary)


if __name__ == "__main__":
    main()
