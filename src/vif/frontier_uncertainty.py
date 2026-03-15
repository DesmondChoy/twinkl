"""Persona-cluster uncertainty review for frontier VIF checkpoints.

Uses saved test-output artifacts only. The main entrypoint computes persona-
cluster bootstrap BCa confidence intervals for active frontier families,
stratified cluster-permutation tests for hard dimensions, and a reusable report
artifact that future experiment reviews can reference.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import yaml
from scipy.stats import norm

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.eval import (
    compute_accuracy_per_dimension,
    compute_calibration_summary,
    compute_hedging_per_dimension,
    compute_mae_per_dimension,
    compute_qwk_per_dimension,
    compute_recall_per_class,
)
from src.vif.posthoc import load_artifact_bundle

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_ORDER = ("qwk_mean", "recall_minus1", "minority_recall_mean")

DEFAULT_CONFIG = {
    "runs_dir": "logs/experiments/runs",
    "artifact_root": "logs/experiments/artifacts",
    "artifact_run_prefix": "frontier_uncertainty_twinkl_730",
    "report_path": "logs/experiments/reports/experiment_review_2026-03-14_twinkl_730.md",
    "summary_path": None,
    "confidence_level": 0.95,
    "n_bootstrap": 1000,
    "n_permutations": 1000,
    "random_seed": 20260314,
    "baseline_family_key": "default_balanced_softmax",
    "hard_dimensions": ["hedonism", "security"],
    "auto_append_lowest_qwk_dimensions": 1,
    "families": [
        {
            "key": "default_balanced_softmax",
            "label": "Current default BalancedSoftmax",
            "model_name": "BalancedSoftmax",
            "run_ids": ["run_019", "run_020", "run_021"],
        },
        {
            "key": "weighted_balanced_softmax",
            "label": "BalancedSoftmax + dimweight",
            "model_name": "BalancedSoftmax",
            "run_ids": ["run_034", "run_035", "run_036"],
        },
        {
            "key": "circreg_recall_floor",
            "label": "BalancedSoftmax + circreg + recall floor",
            "model_name": "BalancedSoftmax",
            "run_ids": ["run_031", "run_032", "run_033"],
        },
        {
            "key": "targeted_batch",
            "label": "BalancedSoftmax + targeted batch",
            "model_name": "BalancedSoftmax",
            "run_ids": ["run_022", "run_023", "run_024"],
        },
        {
            "key": "hedonism_security_lift",
            "label": "BalancedSoftmax + hedonism/security lift",
            "model_name": "BalancedSoftmax",
            "run_ids": ["run_025", "run_026", "run_027"],
        },
    ],
}


@dataclass(frozen=True)
class FamilySpec:
    key: str
    label: str
    model_name: str
    run_ids: tuple[str, ...]


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    model_name: str
    family_key: str
    family_label: str
    run_path: Path
    test_outputs_path: Path


@dataclass
class LoadedRun:
    spec: RunSpec
    metadata_rows: list[dict]
    targets: np.ndarray
    score_predictions: np.ndarray
    uncertainties: np.ndarray
    observed_metrics: dict


def _deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _repo_root() -> Path:
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "src").is_dir() and (current / "pyproject.toml").is_file():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate repo root containing src/ and pyproject.toml")


def _resolve_path(root: Path, raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_config(config_path: str | Path | None) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config

    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    payload = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    _deep_update(config, payload)
    return config


def _family_specs_from_config(config: dict) -> list[FamilySpec]:
    family_specs: list[FamilySpec] = []
    seen_keys: set[str] = set()
    seen_runs: set[tuple[str, str]] = set()

    for raw_family in config["families"]:
        family = FamilySpec(
            key=str(raw_family["key"]),
            label=str(raw_family["label"]),
            model_name=str(raw_family["model_name"]),
            run_ids=tuple(str(run_id) for run_id in raw_family["run_ids"]),
        )
        if family.key in seen_keys:
            raise ValueError(f"Duplicate family key configured: {family.key}")
        seen_keys.add(family.key)

        if not family.run_ids:
            raise ValueError(f"Family {family.key} has no configured run_ids")
        for run_id in family.run_ids:
            run_key = (run_id, family.model_name)
            if run_key in seen_runs:
                raise ValueError(
                    f"Run {run_id} / model {family.model_name} appears in multiple families"
                )
            seen_runs.add(run_key)
        family_specs.append(family)

    baseline_key = str(config["baseline_family_key"])
    if baseline_key not in seen_keys:
        raise ValueError(f"baseline_family_key {baseline_key} is not present in families config")

    return family_specs


def _find_run_path(runs_dir: Path, run_id: str, model_name: str) -> Path:
    path = runs_dir / f"{run_id}_{model_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Run YAML not found: {path}")
    return path


def resolve_run_specs(config: dict, repo_root: Path) -> list[RunSpec]:
    runs_dir = _resolve_path(repo_root, config["runs_dir"])
    assert runs_dir is not None

    run_specs: list[RunSpec] = []
    for family in _family_specs_from_config(config):
        for run_id in family.run_ids:
            run_path = _find_run_path(runs_dir, run_id, family.model_name)
            run_data = yaml.safe_load(run_path.read_text(encoding="utf-8"))
            artifacts = run_data.get("artifacts", {})
            if "test_outputs" not in artifacts:
                raise KeyError(f"Run YAML {run_path} is missing artifacts.test_outputs")
            run_specs.append(
                RunSpec(
                    run_id=run_id,
                    model_name=family.model_name,
                    family_key=family.key,
                    family_label=family.label,
                    run_path=run_path,
                    test_outputs_path=_resolve_path(repo_root, artifacts["test_outputs"]),
                )
            )

    return run_specs


def _nanmean_or_nan(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def _nanmedian_or_nan(values: list[float]) -> float:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan")
    return float(np.median(finite_values))


def compute_score_metrics(
    *,
    targets: np.ndarray,
    score_predictions: np.ndarray,
    uncertainties: np.ndarray,
) -> dict:
    qwk_per_dim = compute_qwk_per_dimension(score_predictions, targets)
    recall_per_class = compute_recall_per_class(score_predictions, targets)
    calibration = compute_calibration_summary(score_predictions, targets, uncertainties)
    hedging_per_dim = compute_hedging_per_dimension(score_predictions)
    mae_per_dim = compute_mae_per_dimension(score_predictions, targets)
    accuracy_per_dim = compute_accuracy_per_dimension(score_predictions, targets)

    return {
        "qwk_mean": _nanmean_or_nan(list(qwk_per_dim.values())),
        "recall_minus1": float(recall_per_class["mean"]["minus1"]),
        "minority_recall_mean": _nanmean_or_nan(
            [recall_per_class["mean"]["minus1"], recall_per_class["mean"]["plus1"]]
        ),
        "calibration_global": float(calibration["error_uncertainty_correlation"]),
        "qwk_per_dim": qwk_per_dim,
        "recall_per_class": recall_per_class,
        "hedging_mean": float(np.mean(list(hedging_per_dim.values()))),
        "mae_mean": float(np.mean(list(mae_per_dim.values()))),
        "accuracy_mean": float(np.mean(list(accuracy_per_dim.values()))),
    }


def load_runs(run_specs: list[RunSpec]) -> list[LoadedRun]:
    loaded_runs: list[LoadedRun] = []
    reference_manifest: list[tuple[str, int, str]] | None = None

    for spec in run_specs:
        bundle = load_artifact_bundle(spec.test_outputs_path)
        manifest = [
            (str(row["persona_id"]), int(row["t_index"]), str(row["date"]))
            for row in bundle.metadata_rows
        ]
        if reference_manifest is None:
            reference_manifest = manifest
        elif manifest != reference_manifest:
            raise ValueError(
                f"Saved test artifacts do not share the same holdout ordering: {spec.test_outputs_path}"
            )

        observed_metrics = compute_score_metrics(
            targets=bundle.targets,
            score_predictions=bundle.baseline_mean_predictions,
            uncertainties=bundle.uncertainties,
        )
        loaded_runs.append(
            LoadedRun(
                spec=spec,
                metadata_rows=bundle.metadata_rows,
                targets=bundle.targets,
                score_predictions=bundle.baseline_mean_predictions,
                uncertainties=bundle.uncertainties,
                observed_metrics=observed_metrics,
            )
        )

    return loaded_runs


def build_persona_clusters(
    metadata_rows: list[dict],
) -> tuple[list[str], dict[str, np.ndarray], dict[int, list[str]]]:
    persona_order: list[str] = []
    persona_to_indices: dict[str, list[int]] = defaultdict(list)

    for sample_idx, row in enumerate(metadata_rows):
        persona_id = str(row["persona_id"])
        if persona_id not in persona_to_indices:
            persona_order.append(persona_id)
        persona_to_indices[persona_id].append(sample_idx)

    persona_to_arrays = {
        persona_id: np.asarray(indices, dtype=np.int64)
        for persona_id, indices in persona_to_indices.items()
    }
    size_groups: dict[int, list[str]] = defaultdict(list)
    for persona_id in persona_order:
        size_groups[len(persona_to_arrays[persona_id])].append(persona_id)

    return persona_order, persona_to_arrays, dict(size_groups)


def sample_persona_cluster_indices(
    persona_order: list[str],
    persona_to_indices: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    sampled_personas = rng.choice(persona_order, size=len(persona_order), replace=True)
    return np.concatenate([persona_to_indices[persona_id] for persona_id in sampled_personas])


def build_stratified_target_permutation_indices(
    persona_order: list[str],
    persona_to_indices: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    groups: dict[int, list[str]] = defaultdict(list)
    for persona_id in persona_order:
        groups[len(persona_to_indices[persona_id])].append(persona_id)

    target_mapping: dict[str, str] = {}
    for group_personas in groups.values():
        shuffled = list(group_personas)
        rng.shuffle(shuffled)
        for destination_persona, source_persona in zip(group_personas, shuffled, strict=True):
            target_mapping[destination_persona] = source_persona

    return np.concatenate(
        [persona_to_indices[target_mapping[persona_id]] for persona_id in persona_order]
    )


def summarize_family_metrics(
    run_metrics: dict[str, dict],
    family_specs: list[FamilySpec],
) -> dict[str, dict]:
    family_metrics: dict[str, dict] = {}

    for family in family_specs:
        family_run_metrics = [run_metrics[run_id] for run_id in family.run_ids]
        family_metrics[family.key] = {
            metric_name: _nanmedian_or_nan(
                [float(run_metric[metric_name]) for run_metric in family_run_metrics]
            )
            for metric_name in METRIC_ORDER
        }
        family_metrics[family.key]["qwk_per_dim"] = {
            dimension: _nanmedian_or_nan(
                [float(run_metric["qwk_per_dim"][dimension]) for run_metric in family_run_metrics]
            )
            for dimension in SCHWARTZ_VALUE_ORDER
        }

    return family_metrics


def evaluate_subset(
    *,
    loaded_runs: list[LoadedRun],
    family_specs: list[FamilySpec],
    baseline_family_key: str,
    sample_indices: np.ndarray,
) -> dict:
    run_metrics: dict[str, dict] = {}
    for loaded_run in loaded_runs:
        run_metrics[loaded_run.spec.run_id] = compute_score_metrics(
            targets=loaded_run.targets[sample_indices],
            score_predictions=loaded_run.score_predictions[sample_indices],
            uncertainties=loaded_run.uncertainties[sample_indices],
        )

    family_metrics = summarize_family_metrics(run_metrics, family_specs)
    baseline_metrics = family_metrics[baseline_family_key]
    family_deltas = {}
    for family in family_specs:
        if family.key == baseline_family_key:
            continue
        family_deltas[family.key] = {
            metric_name: float(family_metrics[family.key][metric_name] - baseline_metrics[metric_name])
            for metric_name in METRIC_ORDER
        }

    return {
        "run_metrics": run_metrics,
        "family_metrics": family_metrics,
        "family_deltas": family_deltas,
    }


def flatten_summary(summary: dict) -> dict[str, float]:
    flat: dict[str, float] = {}

    for run_id, metrics in summary["run_metrics"].items():
        for metric_name in METRIC_ORDER:
            flat[f"run_metric::{run_id}::{metric_name}"] = float(metrics[metric_name])

    for family_key, metrics in summary["family_metrics"].items():
        for metric_name in METRIC_ORDER:
            flat[f"family_metric::{family_key}::{metric_name}"] = float(metrics[metric_name])
        for dimension, qwk_value in metrics["qwk_per_dim"].items():
            flat[f"family_qwk::{family_key}::{dimension}"] = float(qwk_value)

    for family_key, metrics in summary["family_deltas"].items():
        for metric_name, value in metrics.items():
            flat[f"family_delta::{family_key}::{metric_name}"] = float(value)

    return flat


def _distribution_map(records: list[dict[str, float]]) -> dict[str, np.ndarray]:
    values_by_key: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for key, value in record.items():
            values_by_key[key].append(float(value))
    return {
        key: np.asarray(values, dtype=np.float64)
        for key, values in values_by_key.items()
    }


def bca_confidence_interval(
    observed: float,
    bootstrap_values: np.ndarray,
    jackknife_values: np.ndarray,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float | str]:
    alpha = 1.0 - float(confidence_level)
    finite_boot = np.asarray(bootstrap_values[np.isfinite(bootstrap_values)], dtype=np.float64)
    finite_jack = np.asarray(jackknife_values[np.isfinite(jackknife_values)], dtype=np.float64)

    if not np.isfinite(observed) or finite_boot.size < 10:
        return {
            "estimate": float(observed),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "method": "insufficient",
            "bootstrap_n": int(finite_boot.size),
        }

    if finite_jack.size < 3:
        lower, upper = np.quantile(finite_boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        return {
            "estimate": float(observed),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "method": "percentile",
            "bootstrap_n": int(finite_boot.size),
        }

    prop_less = (
        np.count_nonzero(finite_boot < observed)
        + 0.5 * np.count_nonzero(finite_boot == observed)
    ) / finite_boot.size
    eps = 0.5 / finite_boot.size
    prop_less = float(np.clip(prop_less, eps, 1.0 - eps))
    z0 = float(norm.ppf(prop_less))

    jack_mean = float(np.mean(finite_jack))
    diffs = jack_mean - finite_jack
    denom = 6.0 * float(np.sum(diffs**2) ** 1.5)
    if denom <= 0.0 or not np.isfinite(denom):
        lower, upper = np.quantile(finite_boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        return {
            "estimate": float(observed),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "method": "percentile",
            "bootstrap_n": int(finite_boot.size),
        }

    acceleration = float(np.sum(diffs**3) / denom)
    z_low = float(norm.ppf(alpha / 2.0))
    z_high = float(norm.ppf(1.0 - alpha / 2.0))

    def adjusted_quantile(z_alpha: float) -> float:
        denom_inner = 1.0 - acceleration * (z0 + z_alpha)
        if abs(denom_inner) < 1e-12 or not np.isfinite(denom_inner):
            return float("nan")
        adjusted = norm.cdf(z0 + (z0 + z_alpha) / denom_inner)
        return float(np.clip(adjusted, 0.0, 1.0))

    q_low = adjusted_quantile(z_low)
    q_high = adjusted_quantile(z_high)
    if not np.isfinite(q_low) or not np.isfinite(q_high):
        lower, upper = np.quantile(finite_boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        return {
            "estimate": float(observed),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "method": "percentile",
            "bootstrap_n": int(finite_boot.size),
        }

    lower = float(np.quantile(finite_boot, q_low))
    upper = float(np.quantile(finite_boot, q_high))
    return {
        "estimate": float(observed),
        "ci_lower": lower,
        "ci_upper": upper,
        "method": "bca",
        "bootstrap_n": int(finite_boot.size),
    }


def _family_lookup(family_specs: list[FamilySpec]) -> dict[str, FamilySpec]:
    return {family.key: family for family in family_specs}


def _run_lookup(loaded_runs: list[LoadedRun]) -> dict[str, LoadedRun]:
    return {loaded_run.spec.run_id: loaded_run for loaded_run in loaded_runs}


def resolve_hard_dimensions(
    *,
    observed_summary: dict,
    config: dict,
    baseline_family_key: str,
) -> list[str]:
    hard_dimensions = list(dict.fromkeys(str(dim) for dim in config.get("hard_dimensions", [])))
    baseline_qwk = observed_summary["family_metrics"][baseline_family_key]["qwk_per_dim"]
    auto_append = int(config.get("auto_append_lowest_qwk_dimensions", 0))

    if auto_append > 0:
        ranked = sorted(
            baseline_qwk.items(),
            key=lambda item: (
                float("inf") if not np.isfinite(item[1]) else item[1],
                item[0],
            ),
        )
        for dimension, _value in ranked[:auto_append]:
            if dimension not in hard_dimensions:
                hard_dimensions.append(dimension)

    invalid = [dimension for dimension in hard_dimensions if dimension not in SCHWARTZ_VALUE_ORDER]
    if invalid:
        raise ValueError(f"Configured hard dimensions are invalid: {invalid}")
    return hard_dimensions


def _parse_stat_key(key: str) -> tuple[str, str, str]:
    parts = key.split("::")
    if len(parts) != 3:
        raise ValueError(f"Malformed statistic key: {key}")
    return parts[0], parts[1], parts[2]


def _interval_rows(
    *,
    observed: dict[str, float],
    bootstrap_dist: dict[str, np.ndarray],
    jackknife_dist: dict[str, np.ndarray],
    confidence_level: float,
    family_specs: list[FamilySpec],
    loaded_runs: list[LoadedRun],
    baseline_family_key: str,
) -> dict[str, pl.DataFrame]:
    family_lookup = _family_lookup(family_specs)
    run_lookup = _run_lookup(loaded_runs)

    checkpoint_rows: list[dict] = []
    family_rows: list[dict] = []
    dimension_rows: list[dict] = []
    delta_rows: list[dict] = []

    for key, observed_value in observed.items():
        interval = bca_confidence_interval(
            observed_value,
            bootstrap_dist.get(key, np.asarray([], dtype=np.float64)),
            jackknife_dist.get(key, np.asarray([], dtype=np.float64)),
            confidence_level=confidence_level,
        )
        stat_kind, entity_key, metric_name = _parse_stat_key(key)
        row = {
            "estimate": float(interval["estimate"]),
            "ci_lower": float(interval["ci_lower"]),
            "ci_upper": float(interval["ci_upper"]),
            "method": str(interval["method"]),
            "bootstrap_n": int(interval["bootstrap_n"]),
        }

        if stat_kind == "run_metric":
            loaded_run = run_lookup[entity_key]
            row.update(
                {
                    "run_id": entity_key,
                    "family_key": loaded_run.spec.family_key,
                    "family_label": loaded_run.spec.family_label,
                    "metric": metric_name,
                }
            )
            checkpoint_rows.append(row)
            continue

        if stat_kind == "family_metric":
            family = family_lookup[entity_key]
            row.update(
                {
                    "family_key": entity_key,
                    "family_label": family.label,
                    "metric": metric_name,
                }
            )
            family_rows.append(row)
            continue

        if stat_kind == "family_qwk":
            family = family_lookup[entity_key]
            row.update(
                {
                    "family_key": entity_key,
                    "family_label": family.label,
                    "dimension": metric_name,
                    "metric": "qwk",
                }
            )
            dimension_rows.append(row)
            continue

        if stat_kind == "family_delta":
            family = family_lookup[entity_key]
            row.update(
                {
                    "family_key": entity_key,
                    "family_label": family.label,
                    "baseline_family_key": baseline_family_key,
                    "baseline_family_label": family_lookup[baseline_family_key].label,
                    "metric": metric_name,
                    "verdict": classify_delta(row["ci_lower"], row["ci_upper"]),
                }
            )
            delta_rows.append(row)
            continue

        raise ValueError(f"Unsupported statistic kind: {stat_kind}")

    return {
        "checkpoint_metric_cis": pl.DataFrame(checkpoint_rows),
        "family_metric_cis": pl.DataFrame(family_rows),
        "dimension_qwk_cis": pl.DataFrame(dimension_rows),
        "family_delta_cis": pl.DataFrame(delta_rows),
    }


def classify_delta(ci_lower: float, ci_upper: float) -> str:
    if np.isfinite(ci_lower) and ci_lower > 0:
        return "distinguishable_gain"
    if np.isfinite(ci_upper) and ci_upper < 0:
        return "distinguishable_regression"
    return "likely_noise"


def run_hard_dimension_permutation_tests(
    *,
    loaded_runs: list[LoadedRun],
    family_specs: list[FamilySpec],
    persona_order: list[str],
    persona_to_indices: dict[str, np.ndarray],
    hard_dimensions: list[str],
    n_permutations: int,
    seed: int,
) -> pl.DataFrame:
    observed_run_metrics = {
        loaded_run.spec.run_id: loaded_run.observed_metrics for loaded_run in loaded_runs
    }
    observed_family_metrics = summarize_family_metrics(observed_run_metrics, family_specs)

    null_distributions: dict[tuple[str, str], list[float]] = defaultdict(list)
    rng = np.random.default_rng(seed)

    for _ in range(int(n_permutations)):
        permuted_target_indices = build_stratified_target_permutation_indices(
            persona_order,
            persona_to_indices,
            rng,
        )

        run_qwk_by_run: dict[str, dict[str, float]] = {}
        for loaded_run in loaded_runs:
            permuted_targets = loaded_run.targets[permuted_target_indices]
            qwk_per_dim = compute_qwk_per_dimension(
                loaded_run.score_predictions,
                permuted_targets,
            )
            run_qwk_by_run[loaded_run.spec.run_id] = qwk_per_dim

        for family in family_specs:
            for dimension in hard_dimensions:
                family_value = _nanmedian_or_nan(
                    [run_qwk_by_run[run_id][dimension] for run_id in family.run_ids]
                )
                null_distributions[(family.key, dimension)].append(family_value)

    rows: list[dict] = []
    for family in family_specs:
        for dimension in hard_dimensions:
            observed_qwk = float(observed_family_metrics[family.key]["qwk_per_dim"][dimension])
            null_values = np.asarray(
                null_distributions[(family.key, dimension)],
                dtype=np.float64,
            )
            finite_null = null_values[np.isfinite(null_values)]
            if finite_null.size == 0:
                p_value = float("nan")
                null_q95 = float("nan")
            else:
                p_value = float(
                    (1 + np.count_nonzero(finite_null >= observed_qwk))
                    / (finite_null.size + 1)
                )
                null_q95 = float(np.quantile(finite_null, 0.95))
            rows.append(
                {
                    "family_key": family.key,
                    "family_label": family.label,
                    "dimension": dimension,
                    "observed_qwk": observed_qwk,
                    "null_q95": null_q95,
                    "p_value": p_value,
                    "significant_above_chance": bool(
                        np.isfinite(p_value) and p_value < 0.05 and observed_qwk > 0
                    ),
                    "n_permutations": int(n_permutations),
                    "permutation_method": "persona_cluster_permutation_stratified_by_sequence_length",
                }
            )

    return pl.DataFrame(rows)


def _write_frame(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.write_parquet(path)
        return
    if path.suffix == ".csv":
        df.write_csv(path)
        return
    raise ValueError(f"Unsupported frame output path: {path}")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _format_interval(estimate: float, ci_lower: float, ci_upper: float) -> str:
    if not np.isfinite(estimate):
        return "NaN"
    if not np.isfinite(ci_lower) or not np.isfinite(ci_upper):
        return f"{estimate:.3f}"
    return f"{estimate:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _metric_row_lookup(df: pl.DataFrame, entity_key: str, metric_name: str) -> dict:
    row = df.filter(
        (pl.col("family_key") == entity_key) & (pl.col("metric") == metric_name)
    ).to_dicts()
    if not row:
        raise KeyError(f"Missing row for family={entity_key} metric={metric_name}")
    return row[0]


def _checkpoint_row_lookup(df: pl.DataFrame, run_id: str, metric_name: str) -> dict:
    row = df.filter(
        (pl.col("run_id") == run_id) & (pl.col("metric") == metric_name)
    ).to_dicts()
    if not row:
        raise KeyError(f"Missing row for run_id={run_id} metric={metric_name}")
    return row[0]


def _dimension_row_lookup(df: pl.DataFrame, family_key: str, dimension: str) -> dict:
    row = df.filter(
        (pl.col("family_key") == family_key) & (pl.col("dimension") == dimension)
    ).to_dicts()
    if not row:
        raise KeyError(f"Missing row for family={family_key} dimension={dimension}")
    return row[0]


def _delta_row_lookup(df: pl.DataFrame, family_key: str, metric_name: str) -> dict:
    row = df.filter(
        (pl.col("family_key") == family_key) & (pl.col("metric") == metric_name)
    ).to_dicts()
    if not row:
        raise KeyError(f"Missing delta row for family={family_key} metric={metric_name}")
    return row[0]


def plot_family_deltas(
    *,
    delta_df: pl.DataFrame,
    family_specs: list[FamilySpec],
    baseline_family_key: str,
    output_path: Path,
) -> None:
    challenger_families = [family for family in family_specs if family.key != baseline_family_key]
    metrics = list(METRIC_ORDER)
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5), sharey=True)
    if len(metrics) == 1:
        axes = [axes]

    y_positions = np.arange(len(challenger_families))
    for axis, metric_name in zip(axes, metrics, strict=True):
        for row_idx, family in enumerate(challenger_families):
            row = _delta_row_lookup(delta_df, family.key, metric_name)
            estimate = float(row["estimate"])
            ci_lower = float(row["ci_lower"])
            ci_upper = float(row["ci_upper"])
            left = estimate - ci_lower if np.isfinite(ci_lower) else 0.0
            right = ci_upper - estimate if np.isfinite(ci_upper) else 0.0
            axis.errorbar(
                x=estimate,
                y=row_idx,
                xerr=np.array([[left], [right]]),
                fmt="o",
                color="#1f4d75",
                ecolor="#7aa3c8",
                capsize=4,
            )
        axis.axvline(0.0, color="#999999", linestyle="--", linewidth=1)
        axis.set_title(metric_name.replace("_", " "))
        axis.set_yticks(y_positions, [family.label for family in challenger_families])
        axis.grid(axis="x", alpha=0.25)

    fig.suptitle("Family-Median Deltas vs Incumbent (95% BCa CI)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hard_dimension_qwk(
    *,
    dimension_df: pl.DataFrame,
    hard_dimensions: list[str],
    family_specs: list[FamilySpec],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(hard_dimensions), 1, figsize=(11, 3.5 * len(hard_dimensions)))
    if len(hard_dimensions) == 1:
        axes = [axes]

    y_positions = np.arange(len(family_specs))
    for axis, dimension in zip(axes, hard_dimensions, strict=True):
        for row_idx, family in enumerate(family_specs):
            row = _dimension_row_lookup(dimension_df, family.key, dimension)
            estimate = float(row["estimate"])
            ci_lower = float(row["ci_lower"])
            ci_upper = float(row["ci_upper"])
            left = estimate - ci_lower if np.isfinite(ci_lower) else 0.0
            right = ci_upper - estimate if np.isfinite(ci_upper) else 0.0
            axis.errorbar(
                x=estimate,
                y=row_idx,
                xerr=np.array([[left], [right]]),
                fmt="o",
                color="#8c3b2b",
                ecolor="#d5a195",
                capsize=4,
            )
        axis.axvline(0.0, color="#999999", linestyle="--", linewidth=1)
        axis.set_title(f"{dimension} family-median QWK")
        axis.set_yticks(y_positions, [family.label for family in family_specs])
        axis.grid(axis="x", alpha=0.25)

    fig.suptitle("Hard-Dimension Family-Median QWK (95% BCa CI)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report(
    *,
    config: dict,
    family_specs: list[FamilySpec],
    hard_dimensions: list[str],
    interval_tables: dict[str, pl.DataFrame],
    significance_df: pl.DataFrame,
    artifact_root: Path,
    generated_at: datetime,
) -> str:
    family_df = interval_tables["family_metric_cis"]
    delta_df = interval_tables["family_delta_cis"]
    checkpoint_df = interval_tables["checkpoint_metric_cis"]
    dimension_df = interval_tables["dimension_qwk_cis"]
    baseline_key = str(config["baseline_family_key"])

    weighted_qwk_delta = _delta_row_lookup(delta_df, "weighted_balanced_softmax", "qwk_mean")
    weighted_recall_delta = _delta_row_lookup(
        delta_df,
        "weighted_balanced_softmax",
        "recall_minus1",
    )
    weighted_minr_delta = _delta_row_lookup(
        delta_df,
        "weighted_balanced_softmax",
        "minority_recall_mean",
    )

    stimulation_rows = (
        significance_df.filter(pl.col("dimension") == "stimulation")
        .sort("family_key")
        .to_dicts()
    )
    stimulation_sig_count = sum(
        1 for row in stimulation_rows if bool(row["significant_above_chance"])
    )
    total_family_count = len(family_specs)

    checkpoint_rows = []
    for loaded_family in family_specs:
        for run_id in loaded_family.run_ids:
            qwk = _checkpoint_row_lookup(checkpoint_df, run_id, "qwk_mean")
            recall = _checkpoint_row_lookup(checkpoint_df, run_id, "recall_minus1")
            minority = _checkpoint_row_lookup(checkpoint_df, run_id, "minority_recall_mean")
            checkpoint_rows.append(
                [
                    run_id,
                    loaded_family.label,
                    _format_interval(qwk["estimate"], qwk["ci_lower"], qwk["ci_upper"]),
                    _format_interval(
                        recall["estimate"],
                        recall["ci_lower"],
                        recall["ci_upper"],
                    ),
                    _format_interval(
                        minority["estimate"],
                        minority["ci_lower"],
                        minority["ci_upper"],
                    ),
                ]
            )

    family_rows = []
    for family in family_specs:
        qwk = _metric_row_lookup(family_df, family.key, "qwk_mean")
        recall = _metric_row_lookup(family_df, family.key, "recall_minus1")
        minority = _metric_row_lookup(family_df, family.key, "minority_recall_mean")
        family_rows.append(
            [
                family.label,
                _format_interval(qwk["estimate"], qwk["ci_lower"], qwk["ci_upper"]),
                _format_interval(
                    recall["estimate"],
                    recall["ci_lower"],
                    recall["ci_upper"],
                ),
                _format_interval(
                    minority["estimate"],
                    minority["ci_lower"],
                    minority["ci_upper"],
                ),
            ]
        )

    delta_rows = []
    for family in family_specs:
        if family.key == baseline_key:
            continue
        qwk = _delta_row_lookup(delta_df, family.key, "qwk_mean")
        recall = _delta_row_lookup(delta_df, family.key, "recall_minus1")
        minority = _delta_row_lookup(delta_df, family.key, "minority_recall_mean")
        delta_rows.append(
            [
                family.label,
                _format_interval(qwk["estimate"], qwk["ci_lower"], qwk["ci_upper"]),
                qwk["verdict"],
                _format_interval(
                    recall["estimate"],
                    recall["ci_lower"],
                    recall["ci_upper"],
                ),
                recall["verdict"],
                _format_interval(
                    minority["estimate"],
                    minority["ci_lower"],
                    minority["ci_upper"],
                ),
                minority["verdict"],
            ]
        )

    hard_dimension_rows = []
    for family in family_specs:
        for dimension in hard_dimensions:
            qwk_row = _dimension_row_lookup(dimension_df, family.key, dimension)
            sig_row = significance_df.filter(
                (pl.col("family_key") == family.key) & (pl.col("dimension") == dimension)
            ).to_dicts()[0]
            hard_dimension_rows.append(
                [
                    family.label,
                    dimension,
                    _format_interval(
                        qwk_row["estimate"],
                        qwk_row["ci_lower"],
                        qwk_row["ci_upper"],
                    ),
                    f"{float(sig_row['p_value']):.3f}"
                    if np.isfinite(sig_row["p_value"])
                    else "NaN",
                    "yes" if bool(sig_row["significant_above_chance"]) else "no",
                ]
            )

    lines = [
        f"# Experiment Review — {generated_at.date()} — `twinkl-730` frontier uncertainty",
        "",
        "This review quantifies evaluation uncertainty for the active corrected-split",
        "BalancedSoftmax frontier using saved test-output artifacts only. The core",
        "question is whether the apparent family-level differences driving recent",
        "promotion decisions are distinguishable from holdout noise once the fixed",
        "27-persona frozen test set is treated as clustered data.",
        "",
        "## Method",
        "",
        f"- Families reviewed: {', '.join(family.label for family in family_specs)}.",
        f"- Bootstrap: {int(config['n_bootstrap'])} persona-cluster resamples with 95% BCa confidence intervals.",
        "- Cluster definition: resample entire persona trajectories so all within-persona time steps stay together.",
        f"- Hard-dimension significance: {int(config['n_permutations'])} stratified persona-cluster permutations, preserving trajectory length while breaking prediction/target pairing.",
        f"- Hard dimensions: {', '.join(hard_dimensions)}.",
        "",
        "## Findings",
        "",
        f"1. The weighted reference branch still has the strongest tail package, but its QWK change versus the incumbent remains `{weighted_qwk_delta['verdict']}` at {_format_interval(weighted_qwk_delta['estimate'], weighted_qwk_delta['ci_lower'], weighted_qwk_delta['ci_upper'])}. That means the point-estimate drop should not be treated as a certain regression on its own.",
        f"2. The same weighted branch shows a `{weighted_recall_delta['verdict']}` `recall_-1` delta of {_format_interval(weighted_recall_delta['estimate'], weighted_recall_delta['ci_lower'], weighted_recall_delta['ci_upper'])} and a `{weighted_minr_delta['verdict']}` minority-recall delta of {_format_interval(weighted_minr_delta['estimate'], weighted_minr_delta['ci_lower'], weighted_minr_delta['ci_upper'])}. Tail-sensitive gains are real only when those delta intervals stay cleanly above zero.",
        f"3. Hard-dimension significance is uneven. `stimulation` is above chance in {stimulation_sig_count}/{total_family_count} reviewed families under the stratified permutation test, so it remains the weakest discriminator. `hedonism` and `security` should only influence promotion calls when both their own QWK is above chance and the challenger-vs-incumbent family delta is not just bootstrap noise.",
        "",
        "## Checkpoint Intervals",
        "",
        _markdown_table(
            ["Run", "Family", "QWK", "recall_-1", "MinR"],
            checkpoint_rows,
        ),
        "",
        "## Family Intervals",
        "",
        _markdown_table(
            ["Family", "Family-median QWK", "Family-median recall_-1", "Family-median MinR"],
            family_rows,
        ),
        "",
        "## Family Deltas vs Incumbent",
        "",
        _markdown_table(
            [
                "Challenger family",
                "Delta QWK",
                "Verdict",
                "Delta recall_-1",
                "Verdict",
                "Delta MinR",
                "Verdict",
            ],
            delta_rows,
        ),
        "",
        "## Hard-Dimension QWK and Chance Tests",
        "",
        _markdown_table(
            ["Family", "Dimension", "Family-median QWK", "Permutation p", "Above chance?"],
            hard_dimension_rows,
        ),
        "",
        "## Promotion Gate Recommendation",
        "",
        "1. Default replacement decisions should require family-median delta intervals, not point medians alone. Treat any 95% BCa delta interval spanning zero as unresolved noise.",
        "2. For a tail-first promotion, require the challenger's 95% BCa lower bound to stay above zero on both `recall_-1` and minority recall, while `qwk_mean` must not show a materially negative interval. As a practical review guardrail, treat `qwk_mean` lower bounds below `-0.010` as a meaningful regression risk.",
        "3. Treat `hedonism`, `security`, and other hard-dimension QWK numbers as tie-breakers only after two conditions hold: the family-level hard-dimension QWK is above chance under permutation (`p < 0.05`), and the challenger's family-level delta is not classified as bootstrap noise.",
        "",
        "## Artifacts",
        "",
        f"- Artifact root: `{artifact_root}`",
        f"- Family delta plot: `{artifact_root / 'plots' / 'family_delta_intervals.png'}`",
        f"- Hard-dimension QWK plot: `{artifact_root / 'plots' / 'hard_dimension_qwk.png'}`",
    ]
    return "\n".join(lines) + "\n"


def _summary_payload(
    *,
    config: dict,
    generated_at: datetime,
    artifact_root: Path,
    report_path: Path,
    hard_dimensions: list[str],
    interval_tables: dict[str, pl.DataFrame],
    significance_df: pl.DataFrame,
) -> dict:
    weighted_delta_df = interval_tables["family_delta_cis"]
    weighted_recall = _delta_row_lookup(
        weighted_delta_df,
        "weighted_balanced_softmax",
        "recall_minus1",
    )
    weighted_qwk = _delta_row_lookup(
        weighted_delta_df,
        "weighted_balanced_softmax",
        "qwk_mean",
    )

    return {
        "generated_at": generated_at.isoformat(),
        "artifact_root": str(artifact_root),
        "report_path": str(report_path),
        "confidence_level": float(config["confidence_level"]),
        "n_bootstrap": int(config["n_bootstrap"]),
        "n_permutations": int(config["n_permutations"]),
        "hard_dimensions": hard_dimensions,
        "weighted_branch_summary": {
            "qwk_delta": {
                "estimate": float(weighted_qwk["estimate"]),
                "ci_lower": float(weighted_qwk["ci_lower"]),
                "ci_upper": float(weighted_qwk["ci_upper"]),
                "verdict": str(weighted_qwk["verdict"]),
            },
            "recall_minus1_delta": {
                "estimate": float(weighted_recall["estimate"]),
                "ci_lower": float(weighted_recall["ci_lower"]),
                "ci_upper": float(weighted_recall["ci_upper"]),
                "verdict": str(weighted_recall["verdict"]),
            },
        },
        "files": {
            "checkpoint_metric_cis": str(artifact_root / "checkpoint_metric_cis.parquet"),
            "family_metric_cis": str(artifact_root / "family_metric_cis.parquet"),
            "dimension_qwk_cis": str(artifact_root / "dimension_qwk_cis.parquet"),
            "family_delta_cis": str(artifact_root / "family_delta_cis.parquet"),
            "hard_dimension_significance": str(artifact_root / "hard_dimension_significance.parquet"),
            "family_delta_plot": str(artifact_root / "plots" / "family_delta_intervals.png"),
            "hard_dimension_plot": str(artifact_root / "plots" / "hard_dimension_qwk.png"),
        },
        "significance_summary": significance_df.to_dicts(),
    }


def run_frontier_uncertainty(config: dict) -> dict:
    repo_root = _repo_root()
    family_specs = _family_specs_from_config(config)
    run_specs = resolve_run_specs(config, repo_root)
    loaded_runs = load_runs(run_specs)

    persona_order, persona_to_indices, _size_groups = build_persona_clusters(
        loaded_runs[0].metadata_rows
    )
    all_sample_indices = np.arange(len(loaded_runs[0].metadata_rows), dtype=np.int64)
    observed_summary = evaluate_subset(
        loaded_runs=loaded_runs,
        family_specs=family_specs,
        baseline_family_key=str(config["baseline_family_key"]),
        sample_indices=all_sample_indices,
    )
    observed_flat = flatten_summary(observed_summary)
    hard_dimensions = resolve_hard_dimensions(
        observed_summary=observed_summary,
        config=config,
        baseline_family_key=str(config["baseline_family_key"]),
    )

    rng = np.random.default_rng(int(config["random_seed"]))
    bootstrap_records: list[dict[str, float]] = []
    for _ in range(int(config["n_bootstrap"])):
        sample_indices = sample_persona_cluster_indices(
            persona_order,
            persona_to_indices,
            rng,
        )
        bootstrap_records.append(
            flatten_summary(
                evaluate_subset(
                    loaded_runs=loaded_runs,
                    family_specs=family_specs,
                    baseline_family_key=str(config["baseline_family_key"]),
                    sample_indices=sample_indices,
                )
            )
        )

    jackknife_records: list[dict[str, float]] = []
    for excluded_persona in persona_order:
        keep_indices = np.concatenate(
            [
                persona_to_indices[persona_id]
                for persona_id in persona_order
                if persona_id != excluded_persona
            ]
        )
        jackknife_records.append(
            flatten_summary(
                evaluate_subset(
                    loaded_runs=loaded_runs,
                    family_specs=family_specs,
                    baseline_family_key=str(config["baseline_family_key"]),
                    sample_indices=keep_indices,
                )
            )
        )

    bootstrap_dist = _distribution_map(bootstrap_records)
    jackknife_dist = _distribution_map(jackknife_records)
    interval_tables = _interval_rows(
        observed=observed_flat,
        bootstrap_dist=bootstrap_dist,
        jackknife_dist=jackknife_dist,
        confidence_level=float(config["confidence_level"]),
        family_specs=family_specs,
        loaded_runs=loaded_runs,
        baseline_family_key=str(config["baseline_family_key"]),
    )

    significance_df = run_hard_dimension_permutation_tests(
        loaded_runs=loaded_runs,
        family_specs=family_specs,
        persona_order=persona_order,
        persona_to_indices=persona_to_indices,
        hard_dimensions=hard_dimensions,
        n_permutations=int(config["n_permutations"]),
        seed=int(config["random_seed"]) + 1,
    )

    generated_at = datetime.now()
    artifact_root = (
        _resolve_path(
            repo_root,
            config["artifact_root"],
        )
        / f"{config['artifact_run_prefix']}_{generated_at.strftime('%Y%m%d_%H%M%S')}"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    for stem, frame in interval_tables.items():
        _write_frame(frame, artifact_root / f"{stem}.parquet")
        _write_frame(frame, artifact_root / f"{stem}.csv")

    _write_frame(significance_df, artifact_root / "hard_dimension_significance.parquet")
    _write_frame(significance_df, artifact_root / "hard_dimension_significance.csv")

    plots_dir = artifact_root / "plots"
    plot_family_deltas(
        delta_df=interval_tables["family_delta_cis"],
        family_specs=family_specs,
        baseline_family_key=str(config["baseline_family_key"]),
        output_path=plots_dir / "family_delta_intervals.png",
    )
    plot_hard_dimension_qwk(
        dimension_df=interval_tables["dimension_qwk_cis"],
        hard_dimensions=hard_dimensions,
        family_specs=family_specs,
        output_path=plots_dir / "hard_dimension_qwk.png",
    )

    report_path = _resolve_path(repo_root, config["report_path"])
    assert report_path is not None
    report_body = render_report(
        config=config,
        family_specs=family_specs,
        hard_dimensions=hard_dimensions,
        interval_tables=interval_tables,
        significance_df=significance_df,
        artifact_root=artifact_root,
        generated_at=generated_at,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_body, encoding="utf-8")

    summary_payload = _summary_payload(
        config=config,
        generated_at=generated_at,
        artifact_root=artifact_root,
        report_path=report_path,
        hard_dimensions=hard_dimensions,
        interval_tables=interval_tables,
        significance_df=significance_df,
    )
    summary_path = artifact_root / "summary.yaml"
    _write_yaml(summary_path, summary_payload)

    configured_summary_path = _resolve_path(repo_root, config.get("summary_path"))
    if configured_summary_path is not None:
        _write_yaml(configured_summary_path, summary_payload)

    return {
        "generated_at": generated_at.isoformat(),
        "artifact_root": str(artifact_root),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "hard_dimensions": hard_dimensions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute persona-cluster frontier uncertainty intervals from saved artifacts."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config overriding the default twinkl-730 review settings.",
    )
    args = parser.parse_args()

    summary = run_frontier_uncertainty(load_config(args.config))
    print(f"Frontier uncertainty artifact root: {summary['artifact_root']}")
    print(f"Report: {summary['report_path']}")
    print(f"Summary: {summary['summary_path']}")
    print(f"Hard dimensions: {', '.join(summary['hard_dimensions'])}")


if __name__ == "__main__":
    main()
