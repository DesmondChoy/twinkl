"""No-new-data VIF frontier policy search over persisted checkpoint artifacts.

The script selects a small seed-aligned ensemble/routing policy on validation
outputs only, then evaluates the selected policy once on the fixed test outputs.
It never reads consensus-label runs for promotion and never changes holdouts.
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import yaml

_bootstrap_dir = Path.cwd().resolve()
while _bootstrap_dir != _bootstrap_dir.parent:
    if (_bootstrap_dir / "src").is_dir() and (_bootstrap_dir / "pyproject.toml").is_file():
        sys.path.insert(0, str(_bootstrap_dir))
        break
    _bootstrap_dir = _bootstrap_dir.parent

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.eval import discretize_predictions
from src.vif.posthoc import _compute_score_based_policy_metrics, load_artifact_bundle


CLASS_VALUES = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
HARD_DIMS = ("hedonism", "security", "stimulation")

RUN_GROUPS = {
    "incumbent": {
        11: "logs/experiments/runs/run_019_BalancedSoftmax.yaml",
        22: "logs/experiments/runs/run_020_BalancedSoftmax.yaml",
        33: "logs/experiments/runs/run_021_BalancedSoftmax.yaml",
    },
    "dimweight": {
        11: "logs/experiments/runs/run_034_BalancedSoftmax.yaml",
        22: "logs/experiments/runs/run_035_BalancedSoftmax.yaml",
        33: "logs/experiments/runs/run_036_BalancedSoftmax.yaml",
    },
    "qwen": {
        11: "logs/experiments/runs/run_042_BalancedSoftmax.yaml",
        22: "logs/experiments/runs/run_043_BalancedSoftmax.yaml",
        33: "logs/experiments/runs/run_044_BalancedSoftmax.yaml",
    },
    "twostage": {
        11: "logs/experiments/runs/run_045_TwoStageBalancedSoftmax.yaml",
        22: "logs/experiments/runs/run_046_TwoStageBalancedSoftmax.yaml",
        33: "logs/experiments/runs/run_047_TwoStageBalancedSoftmax.yaml",
    },
}

PROMOTION_FLOORS = {
    "qwk_mean": 0.400,
    "recall_minus1": 0.400,
    "minority_recall_mean": 0.480,
    "hedging_mean": 0.580,
    "calibration_global": 0.720,
}


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    payload: dict
    apply: Callable[[dict[str, object]], tuple[np.ndarray, np.ndarray, np.ndarray]]


def _repo_root() -> Path:
    path = Path.cwd().resolve()
    while path != path.parent:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path
        path = path.parent
    raise RuntimeError("Could not locate repo root.")


def _load_run_yaml(repo_root: Path, run_yaml: str) -> dict:
    path = repo_root / run_yaml
    if not path.is_file():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["_run_yaml_path"] = str(path.relative_to(repo_root))
    return data


def _load_bundles(repo_root: Path) -> dict[str, dict[int, dict[str, object]]]:
    bundles: dict[str, dict[int, dict[str, object]]] = {}
    for family, seed_map in RUN_GROUPS.items():
        bundles[family] = {}
        for seed, run_yaml in seed_map.items():
            run_data = _load_run_yaml(repo_root, run_yaml)
            artifacts = run_data["artifacts"]
            bundles[family][seed] = {
                "run_id": run_data["metadata"]["run_id"],
                "run_yaml": run_data["_run_yaml_path"],
                "validation": load_artifact_bundle(repo_root / artifacts["validation_outputs"]),
                "test": load_artifact_bundle(repo_root / artifacts["test_outputs"]),
            }
    _assert_aligned_manifests(bundles)
    return bundles


def _manifest_frame(bundle: object) -> pl.DataFrame:
    return bundle.data_frame.select(["persona_id", "t_index", "date", "dimension", "target"])


def _assert_aligned_manifests(bundles: dict[str, dict[int, dict[str, object]]]) -> None:
    reference: dict[str, pl.DataFrame] = {}
    reference_name: dict[str, str] = {}
    for family, seed_map in bundles.items():
        for seed, split_map in seed_map.items():
            for split_name in ("validation", "test"):
                frame = _manifest_frame(split_map[split_name])
                key = f"{split_name}"
                label = f"{family}/seed{seed}/{split_name}"
                if key not in reference:
                    reference[key] = frame
                    reference_name[key] = label
                    continue
                if not frame.equals(reference[key]):
                    raise ValueError(
                        "Artifact manifests do not align: "
                        f"{label} differs from {reference_name[key]}."
                    )


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if math.isclose(temperature, 1.0):
        return probabilities
    adjusted = np.power(np.clip(probabilities, 1e-12, 1.0), 1.0 / temperature)
    return adjusted / adjusted.sum(axis=-1, keepdims=True)


def _combine_probabilities(
    family_bundles: dict[str, object],
    weights: dict[str, float],
    *,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    families = [family for family, weight in weights.items() if weight > 0]
    if len(families) == 1 and math.isclose(weights[families[0]], 1.0) and math.isclose(temperature, 1.0):
        bundle = family_bundles[families[0]]
        return (
            bundle.class_probabilities,
            bundle.baseline_mean_predictions,
            bundle.uncertainties,
        )

    probs = np.zeros_like(family_bundles[families[0]].class_probabilities, dtype=np.float64)
    expected_by_family = []
    uncertainty_var = np.zeros_like(family_bundles[families[0]].uncertainties, dtype=np.float64)
    for family, weight in weights.items():
        if weight <= 0:
            continue
        bundle = family_bundles[family]
        probs += weight * bundle.class_probabilities
        expected_by_family.append((weight, bundle.baseline_mean_predictions))
        uncertainty_var += weight * np.square(bundle.uncertainties)

    probs = _temperature_scale(probs, temperature)
    expected = probs @ CLASS_VALUES
    disagreement_var = np.zeros_like(expected, dtype=np.float64)
    for weight, family_expected in expected_by_family:
        disagreement_var += weight * np.square(family_expected - expected)
    uncertainty = np.sqrt(np.maximum(uncertainty_var + disagreement_var, 0.0))
    return probs, expected, uncertainty


def _compute_metrics(
    *,
    targets: np.ndarray,
    probabilities: np.ndarray,
    expected: np.ndarray,
    uncertainties: np.ndarray,
) -> dict:
    metrics = _compute_score_based_policy_metrics(
        targets=targets,
        score_predictions=expected,
        uncertainties=uncertainties,
        probabilities=probabilities,
    )
    metrics["calibration_global"] = float(
        metrics["calibration"]["error_uncertainty_correlation"]
    )
    return metrics


def _metric_row(policy: Policy, seed: int, split: str, metrics: dict) -> dict:
    return {
        "policy": policy.name,
        "policy_kind": policy.kind,
        "seed": seed,
        "split": split,
        "qwk_mean": float(metrics["qwk_mean"]),
        "recall_minus1": float(metrics["recall_minus1"]),
        "minority_recall_mean": float(metrics["minority_recall_mean"]),
        "hedging_mean": float(metrics["hedging_mean"]),
        "calibration_global": float(metrics["calibration_global"]),
        "accuracy_mean": float(metrics["accuracy_mean"]),
        "mae_mean": float(metrics["mae_mean"]),
        "decision_neutral_rate": float(metrics["decision_neutral_rate"]),
        "hedonism_qwk": float(metrics["qwk_per_dim"]["hedonism"]),
        "security_qwk": float(metrics["qwk_per_dim"]["security"]),
        "stimulation_qwk": float(metrics["qwk_per_dim"]["stimulation"]),
        "hedonism_active_recall": _active_recall(metrics, "hedonism"),
        "security_active_recall": _active_recall(metrics, "security"),
        "stimulation_active_recall": _active_recall(metrics, "stimulation"),
    }


def _active_recall(metrics: dict, dimension: str) -> float:
    recalls = metrics["recall_per_class"]["per_dim"][dimension]
    values = [recalls["minus1"], recalls["plus1"]]
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _median(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def _family_summary(rows: list[dict], policy_name: str, split: str) -> dict:
    policy_rows = [
        row for row in rows if row["policy"] == policy_name and row["split"] == split
    ]
    if not policy_rows:
        raise ValueError(f"No rows for policy={policy_name}, split={split}")
    summary = {
        "policy": policy_name,
        "split": split,
        "qwk_mean": _median([row["qwk_mean"] for row in policy_rows]),
        "recall_minus1": _median([row["recall_minus1"] for row in policy_rows]),
        "minority_recall_mean": _median([row["minority_recall_mean"] for row in policy_rows]),
        "hedging_mean": _median([row["hedging_mean"] for row in policy_rows]),
        "calibration_global": _median([row["calibration_global"] for row in policy_rows]),
        "accuracy_mean": _median([row["accuracy_mean"] for row in policy_rows]),
        "mae_mean": _median([row["mae_mean"] for row in policy_rows]),
        "decision_neutral_rate": _median([row["decision_neutral_rate"] for row in policy_rows]),
    }
    for dim in HARD_DIMS:
        summary[f"{dim}_qwk"] = _median([row[f"{dim}_qwk"] for row in policy_rows])
        summary[f"{dim}_active_recall"] = _median(
            [row[f"{dim}_active_recall"] for row in policy_rows]
        )
    summary.update(_floor_flags(summary))
    return summary


def _floor_flags(summary: dict) -> dict:
    return {
        "qwk_floor": summary["qwk_mean"] >= PROMOTION_FLOORS["qwk_mean"],
        "recall_minus1_floor": summary["recall_minus1"] >= PROMOTION_FLOORS["recall_minus1"],
        "minority_recall_floor": summary["minority_recall_mean"] >= PROMOTION_FLOORS["minority_recall_mean"],
        "hedging_floor": summary["hedging_mean"] <= PROMOTION_FLOORS["hedging_mean"],
        "calibration_floor": summary["calibration_global"] >= PROMOTION_FLOORS["calibration_global"],
    }


def _hard_guardrail_flags(candidate: dict, baseline: dict) -> dict:
    regressions = {}
    improvements = {}
    for dim in HARD_DIMS:
        regressions[dim] = candidate[f"{dim}_qwk"] - baseline[f"{dim}_qwk"]
        improvements[dim] = max(
            candidate[f"{dim}_qwk"] - baseline[f"{dim}_qwk"],
            candidate[f"{dim}_active_recall"] - baseline[f"{dim}_active_recall"],
        )
    return {
        "hard_qwk_deltas": regressions,
        "hard_improvement_deltas": improvements,
        "hard_no_material_regression": all(delta >= -0.02 for delta in regressions.values()),
        "hard_has_required_gain": any(delta >= 0.05 for delta in improvements.values()),
    }


def _selection_score(summary: dict, hard_flags: dict) -> tuple:
    margins = [
        summary["qwk_mean"] - PROMOTION_FLOORS["qwk_mean"],
        summary["recall_minus1"] - PROMOTION_FLOORS["recall_minus1"],
        summary["minority_recall_mean"] - PROMOTION_FLOORS["minority_recall_mean"],
        PROMOTION_FLOORS["hedging_mean"] - summary["hedging_mean"],
        summary["calibration_global"] - PROMOTION_FLOORS["calibration_global"],
    ]
    floor_count = int(sum(_floor_flags(summary).values()))
    return (
        floor_count,
        int(hard_flags["hard_no_material_regression"]),
        int(hard_flags["hard_has_required_gain"]),
        min(margins),
        summary["qwk_mean"],
        summary["recall_minus1"],
        summary["minority_recall_mean"],
        -summary["hedging_mean"],
        summary["calibration_global"],
    )


def _simplex_weights(families: list[str], step: float) -> list[dict[str, float]]:
    units = int(round(1.0 / step))
    weights = []
    for combo in itertools.product(range(units + 1), repeat=len(families)):
        if sum(combo) != units:
            continue
        weights.append(
            {
                family: value / units
                for family, value in zip(families, combo, strict=True)
            }
        )
    return weights


def _make_weight_policy(weights: dict[str, float], temperature: float) -> Policy:
    label = ",".join(f"{family}={weight:.2f}" for family, weight in weights.items() if weight > 0)
    name = f"ensemble[{label};temp={temperature:.2f}]"

    def apply(family_bundles: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _combine_probabilities(family_bundles, weights, temperature=temperature)

    return Policy(
        name=name,
        kind="probability_ensemble",
        payload={"weights": weights, "temperature": temperature},
        apply=apply,
    )


def _dimension_score(metrics: dict, dim: str, objective: str) -> float:
    qwk = float(metrics["qwk_per_dim"][dim])
    active = _active_recall(metrics, dim)
    hedge = float(metrics["hedging_per_dim"][dim])
    cal = float(metrics["calibration"]["per_dim"][dim])
    values = {
        "qwk": qwk,
        "active": active,
        "balanced": qwk + (0.50 * active) - (0.20 * hedge) + (0.10 * cal),
    }
    value = values[objective]
    return value if np.isfinite(value) else float("-inf")


def _select_router_map(
    val_metrics_by_family_seed: dict[str, dict[int, dict]],
    objective: str,
) -> dict[str, str]:
    router: dict[str, str] = {}
    for dim in SCHWARTZ_VALUE_ORDER:
        scores = {}
        for family, seed_metrics in val_metrics_by_family_seed.items():
            scores[family] = _median(
                [_dimension_score(metrics, dim, objective) for metrics in seed_metrics.values()]
            )
        router[dim] = max(scores, key=scores.get)
    return router


def _make_router_policy(router: dict[str, str], objective: str) -> Policy:
    dim_to_index = {dim: idx for idx, dim in enumerate(SCHWARTZ_VALUE_ORDER)}
    name = "router[" + objective + ";" + ",".join(
        f"{dim}={router[dim]}" for dim in SCHWARTZ_VALUE_ORDER
    ) + "]"

    def apply(family_bundles: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        first = family_bundles[next(iter(family_bundles))]
        probabilities = np.zeros_like(first.class_probabilities)
        expected = np.zeros_like(first.baseline_mean_predictions)
        uncertainties = np.zeros_like(first.uncertainties)
        for dim, family in router.items():
            idx = dim_to_index[dim]
            bundle = family_bundles[family]
            probabilities[:, idx, :] = bundle.class_probabilities[:, idx, :]
            expected[:, idx] = bundle.baseline_mean_predictions[:, idx]
            uncertainties[:, idx] = bundle.uncertainties[:, idx]
        return probabilities, expected, uncertainties

    return Policy(
        name=name,
        kind="dimension_router",
        payload={"objective": objective, "dimension_family": router},
        apply=apply,
    )


def _build_policies(
    val_metrics_by_family_seed: dict[str, dict[int, dict]],
    *,
    weight_step: float,
    temperatures: list[float],
) -> list[Policy]:
    families = list(RUN_GROUPS)
    policies = []
    for weights in _simplex_weights(families, weight_step):
        for temperature in temperatures:
            policies.append(_make_weight_policy(weights, temperature))
    for objective in ("qwk", "active", "balanced"):
        router = _select_router_map(val_metrics_by_family_seed, objective)
        policies.append(_make_router_policy(router, objective))
    return policies


def _evaluate_policy(
    policy: Policy,
    bundles: dict[str, dict[int, dict[str, object]]],
    split: str,
) -> tuple[list[dict], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    rows = []
    outputs = {}
    for seed in (11, 22, 33):
        family_bundles = {
            family: seed_map[seed][split]
            for family, seed_map in bundles.items()
        }
        probabilities, expected, uncertainties = policy.apply(family_bundles)
        metrics = _compute_metrics(
            targets=family_bundles["incumbent"].targets,
            probabilities=probabilities,
            expected=expected,
            uncertainties=uncertainties,
        )
        rows.append(_metric_row(policy, seed, split, metrics))
        outputs[seed] = (probabilities, expected, uncertainties)
    return rows, outputs


def _evaluate_family_metrics(
    bundles: dict[str, dict[int, dict[str, object]]],
) -> tuple[list[dict], dict[str, dict[int, dict]]]:
    rows = []
    val_metrics_by_family_seed: dict[str, dict[int, dict]] = {}
    for family in RUN_GROUPS:
        weights = {name: 1.0 if name == family else 0.0 for name in RUN_GROUPS}
        policy = _make_weight_policy(weights, 1.0)
        val_metrics_by_family_seed[family] = {}
        for split in ("validation", "test"):
            split_rows, _ = _evaluate_policy(policy, bundles, split)
            rows.extend(split_rows)
            if split == "validation":
                for row in split_rows:
                    seed = int(row["seed"])
                    family_bundles = {
                        fam: seed_map[seed][split]
                        for fam, seed_map in bundles.items()
                    }
                    probabilities, expected, uncertainties = policy.apply(family_bundles)
                    val_metrics_by_family_seed[family][seed] = _compute_metrics(
                        targets=family_bundles["incumbent"].targets,
                        probabilities=probabilities,
                        expected=expected,
                        uncertainties=uncertainties,
                    )
    return rows, val_metrics_by_family_seed


def _oracle_validation_summary(bundles: dict[str, dict[int, dict[str, object]]]) -> dict:
    rows = []
    for seed in (11, 22, 33):
        incumbent = bundles["incumbent"][seed]["validation"]
        family_bundles = {
            family: seed_map[seed]["validation"]
            for family, seed_map in bundles.items()
        }
        family_predictions = {
            family: discretize_predictions(bundle.baseline_mean_predictions)
            for family, bundle in family_bundles.items()
        }
        targets = incumbent.targets.astype(int)
        chosen = np.zeros_like(targets, dtype=np.float64)
        probabilities = np.zeros_like(incumbent.class_probabilities, dtype=np.float64)
        uncertainties = np.zeros_like(incumbent.uncertainties, dtype=np.float64)
        for sample_idx in range(targets.shape[0]):
            for dim_idx in range(targets.shape[1]):
                target = targets[sample_idx, dim_idx]
                best_family = min(
                    RUN_GROUPS,
                    key=lambda family: abs(family_predictions[family][sample_idx, dim_idx] - target),
                )
                chosen[sample_idx, dim_idx] = family_predictions[best_family][sample_idx, dim_idx]
                probabilities[sample_idx, dim_idx, :] = family_bundles[best_family].class_probabilities[
                    sample_idx, dim_idx, :
                ]
                uncertainties[sample_idx, dim_idx] = family_bundles[best_family].uncertainties[
                    sample_idx, dim_idx
                ]
        metrics = _compute_metrics(
            targets=targets,
            probabilities=probabilities,
            expected=chosen,
            uncertainties=uncertainties,
        )
        rows.append(_metric_row(
            Policy("validation_oracle_per_cell", "oracle", {}, lambda _: None),
            seed,
            "validation",
            metrics,
        ))
    return _family_summary(rows, "validation_oracle_per_cell", "validation")


def _write_output_frame(
    output_path: Path,
    canonical_bundle: object,
    *,
    policy_name: str,
    probabilities: np.ndarray,
    expected: np.ndarray,
    uncertainties: np.ndarray,
) -> None:
    predicted = discretize_predictions(expected)
    raw_logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    frame = canonical_bundle.data_frame.with_columns(
        pl.lit(policy_name).alias("model_name"),
        pl.Series("predicted_class", predicted.reshape(-1).astype(int)),
        pl.Series("mean_prediction", expected.reshape(-1).astype(float)),
        pl.Series("uncertainty", uncertainties.reshape(-1).astype(float)),
        pl.Series("raw_logits", raw_logits.reshape(-1, 3).tolist()),
        pl.Series("class_probabilities", probabilities.reshape(-1, 3).tolist()),
    )
    frame.write_parquet(output_path)


def _write_report(
    output_dir: Path,
    selected_policy: Policy,
    validation_summary_rows: list[dict],
    test_summary_rows: list[dict],
    oracle_summary: dict,
    baseline_validation: dict,
    baseline_test: dict,
    selected_validation: dict,
    selected_test: dict,
) -> None:
    validation_hard = _hard_guardrail_flags(selected_validation, baseline_validation)
    test_hard = _hard_guardrail_flags(selected_test, baseline_test)
    lines = [
        "# No-New-Data VIF Policy Search",
        "",
        f"Selected policy: `{selected_policy.name}`",
        "",
        "## Validation Selection",
        "",
        "| Policy | QWK | recall_-1 | MinR | Hedge | Cal | floors | hard ok | hard gain |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in validation_summary_rows[:15]:
        floors = int(sum(_floor_flags(row).values()))
        hard = _hard_guardrail_flags(row, baseline_validation)
        lines.append(
            f"| `{row['policy']}` | {row['qwk_mean']:.3f} | {row['recall_minus1']:.3f} | "
            f"{row['minority_recall_mean']:.3f} | {row['hedging_mean']:.3f} | "
            f"{row['calibration_global']:.3f} | {floors}/5 | "
            f"{hard['hard_no_material_regression']} | {hard['hard_has_required_gain']} |"
        )
    lines.extend(
        [
            "",
            "## Validation Oracle Upper Bound",
            "",
            "| QWK | recall_-1 | MinR | Hedge | Cal |",
            "|---:|---:|---:|---:|---:|",
            (
                f"| {oracle_summary['qwk_mean']:.3f} | {oracle_summary['recall_minus1']:.3f} | "
                f"{oracle_summary['minority_recall_mean']:.3f} | "
                f"{oracle_summary['hedging_mean']:.3f} | {oracle_summary['calibration_global']:.3f} |"
            ),
            "",
            "The oracle is per-cell and label-aware, so it is not deployable. It is only a complementarity ceiling.",
            "",
            "## Fixed Test Evaluation",
            "",
            "| Policy | QWK | recall_-1 | MinR | Hedge | Cal | floors | hard ok | hard gain |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in test_summary_rows:
        floors = int(sum(_floor_flags(row).values()))
        hard = _hard_guardrail_flags(row, baseline_test)
        lines.append(
            f"| `{row['policy']}` | {row['qwk_mean']:.3f} | {row['recall_minus1']:.3f} | "
            f"{row['minority_recall_mean']:.3f} | {row['hedging_mean']:.3f} | "
            f"{row['calibration_global']:.3f} | {floors}/5 | "
            f"{hard['hard_no_material_regression']} | {hard['hard_has_required_gain']} |"
        )
    lines.extend(
        [
            "",
            "## Selected Hard-Dimension Test Deltas vs Incumbent",
            "",
            "| Dimension | QWK delta | active-recall delta |",
            "|---|---:|---:|",
        ]
    )
    for dim in HARD_DIMS:
        lines.append(
            f"| {dim} | {test_hard['hard_qwk_deltas'][dim]:+.3f} | "
            f"{selected_test[f'{dim}_active_recall'] - baseline_test[f'{dim}_active_recall']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            "- Validation hard guardrail: "
            f"{validation_hard['hard_no_material_regression']} / "
            f"{validation_hard['hard_has_required_gain']}",
            "- Test hard guardrail: "
            f"{test_hard['hard_no_material_regression']} / {test_hard['hard_has_required_gain']}",
            "- Promotion floors met on test: "
            f"{int(sum(_floor_flags(selected_test).values()))}/5",
        ]
    )
    (output_dir / "policy_search_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact output directory. Defaults to a timestamped no-new-data artifact path.",
    )
    parser.add_argument("--weight-step", type=float, default=0.25)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.75, 0.85, 1.0, 1.15],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = repo_root / "logs/experiments/artifacts" / f"no_new_data_vif_policy_twinkl_lct3_{stamp}"
    elif not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bundles = _load_bundles(repo_root)
    baseline_rows, val_metrics_by_family_seed = _evaluate_family_metrics(bundles)
    policies = _build_policies(
        val_metrics_by_family_seed,
        weight_step=args.weight_step,
        temperatures=args.temperatures,
    )

    all_validation_rows = list(baseline_rows)
    policy_payloads = {}
    for policy in policies:
        rows, _ = _evaluate_policy(policy, bundles, "validation")
        all_validation_rows.extend(rows)
        policy_payloads[policy.name] = {
            "kind": policy.kind,
            "payload": policy.payload,
        }

    baseline_validation = _family_summary(all_validation_rows, _make_weight_policy(
        {name: 1.0 if name == "incumbent" else 0.0 for name in RUN_GROUPS},
        1.0,
    ).name, "validation")

    validation_summaries = []
    seen = set()
    for row in all_validation_rows:
        key = row["policy"]
        if key in seen:
            continue
        seen.add(key)
        summary = _family_summary(all_validation_rows, key, "validation")
        hard = _hard_guardrail_flags(summary, baseline_validation)
        summary["selection_score"] = _selection_score(summary, hard)
        validation_summaries.append(summary)
    validation_summaries.sort(key=lambda item: item["selection_score"], reverse=True)

    selected_name = validation_summaries[0]["policy"]
    selected_policy = next(policy for policy in policies if policy.name == selected_name)

    selected_validation_rows, selected_validation_outputs = _evaluate_policy(
        selected_policy,
        bundles,
        "validation",
    )
    selected_test_rows, selected_test_outputs = _evaluate_policy(selected_policy, bundles, "test")

    test_rows = [row for row in baseline_rows if row["split"] == "test"] + selected_test_rows
    baseline_policy_name = baseline_validation["policy"]
    baseline_test = _family_summary(test_rows, baseline_policy_name, "test")
    selected_validation = _family_summary(
        selected_validation_rows,
        selected_name,
        "validation",
    )
    selected_test = _family_summary(selected_test_rows, selected_name, "test")

    oracle_summary = _oracle_validation_summary(bundles)

    validation_frame = pl.DataFrame(all_validation_rows)
    validation_frame.write_csv(output_dir / "validation_policy_rows.csv")
    validation_summary_rows = [
        {key: value for key, value in row.items() if key != "selection_score"}
        for row in validation_summaries
    ]
    pl.DataFrame(validation_summary_rows).write_csv(
        output_dir / "validation_policy_summary.csv"
    )
    pl.DataFrame(test_rows).write_csv(output_dir / "test_policy_rows.csv")
    test_summaries = [
        _family_summary(test_rows, policy_name, "test")
        for policy_name in sorted({row["policy"] for row in test_rows})
    ]
    pl.DataFrame(test_summaries).write_csv(output_dir / "test_policy_summary.csv")

    selected_payload = {
        "selected_policy": selected_name,
        "policy": policy_payloads[selected_name],
        "promotion_floors": PROMOTION_FLOORS,
        "validation_summary": selected_validation,
        "test_summary": selected_test,
        "baseline_validation_summary": baseline_validation,
        "baseline_test_summary": baseline_test,
        "validation_oracle_summary": oracle_summary,
        "hard_guardrail_validation": _hard_guardrail_flags(selected_validation, baseline_validation),
        "hard_guardrail_test": _hard_guardrail_flags(selected_test, baseline_test),
        "run_groups": RUN_GROUPS,
    }
    (output_dir / "selected_policy.yaml").write_text(
        yaml.safe_dump(selected_payload, sort_keys=False),
        encoding="utf-8",
    )

    for seed, (probabilities, expected, uncertainties) in selected_validation_outputs.items():
        _write_output_frame(
            output_dir / f"selected_validation_seed{seed}.parquet",
            bundles["incumbent"][seed]["validation"],
            policy_name=selected_name,
            probabilities=probabilities,
            expected=expected,
            uncertainties=uncertainties,
        )
    for seed, (probabilities, expected, uncertainties) in selected_test_outputs.items():
        _write_output_frame(
            output_dir / f"selected_test_seed{seed}.parquet",
            bundles["incumbent"][seed]["test"],
            policy_name=selected_name,
            probabilities=probabilities,
            expected=expected,
            uncertainties=uncertainties,
        )

    _write_report(
        output_dir,
        selected_policy,
        validation_summaries,
        test_summaries,
        oracle_summary,
        baseline_validation,
        baseline_test,
        selected_validation,
        selected_test,
    )

    try:
        artifact_dir_label = output_dir.relative_to(repo_root)
    except ValueError:
        artifact_dir_label = output_dir
    print(f"artifact_dir: {artifact_dir_label}")
    print(f"selected_policy: {selected_name}")
    print(
        "validation: "
        f"qwk={selected_validation['qwk_mean']:.3f} "
        f"recall_-1={selected_validation['recall_minus1']:.3f} "
        f"minr={selected_validation['minority_recall_mean']:.3f} "
        f"hedge={selected_validation['hedging_mean']:.3f} "
        f"cal={selected_validation['calibration_global']:.3f}"
    )
    print(
        "test: "
        f"qwk={selected_test['qwk_mean']:.3f} "
        f"recall_-1={selected_test['recall_minus1']:.3f} "
        f"minr={selected_test['minority_recall_mean']:.3f} "
        f"hedge={selected_test['hedging_mean']:.3f} "
        f"cal={selected_test['calibration_global']:.3f}"
    )


if __name__ == "__main__":
    main()
