"""Evaluation metrics for VIF Critic model.

This module provides functions to evaluate model performance on alignment
prediction, including per-dimension MSE, Spearman correlation, and
uncertainty calibration checks.

Metrics:
- MSE: Mean squared error per Schwartz dimension
- Spearman: Rank correlation per dimension (captures ordering quality)
- Calibration: Check if uncertainty estimates are well-calibrated

Usage:
    from src.vif.eval import evaluate_model, compute_mse_per_dimension

    # Full evaluation
    results = evaluate_model(model, test_loader)
    print(f"Mean MSE: {results['mse_mean']:.4f}")
    print(f"Mean Spearman: {results['spearman_mean']:.4f}")

    # Per-dimension analysis
    mse_per_dim = compute_mse_per_dimension(predictions, targets)
"""

from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.models.judge import SCHWARTZ_VALUE_ORDER


def discretize_predictions(values: np.ndarray) -> np.ndarray:
    """Convert continuous predictions to discrete classes {-1, 0, +1}.

    Uses explicit thresholds to avoid numpy's bankers rounding
    (round-half-to-even), which biases ±0.5 toward 0.

    Args:
        values: Array of continuous predictions

    Returns:
        Integer array with values in {-1, 0, +1}
    """
    classes = np.zeros_like(values, dtype=int)
    classes[values < -0.5] = -1
    classes[values > 0.5] = 1
    return classes


def _nanmean_or_nan(values: list[float]) -> float:
    """Return NaN immediately if all values are NaN, avoiding numpy warning."""
    if all(np.isnan(v) for v in values):
        return float("nan")
    return float(np.nanmean(values))


def _is_ordinal_model(model) -> bool:
    """Return True when the model exposes the ordinal prediction interface."""
    return hasattr(model, "predict") and hasattr(model, "_variant_name")


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Spearman correlation or NaN when either side is constant."""
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    corr, _ = stats.spearmanr(x, y)
    return float(corr)


def _metric_sort_value(value: float, *, higher_is_better: bool) -> float:
    """Normalize values into a descending-friendly comparison scalar."""
    if not np.isfinite(value):
        return float("-inf")
    return value if higher_is_better else -value


def compute_mse_per_dimension(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute MSE for each Schwartz value dimension.

    Args:
        predictions: (n_samples, 10) array of predicted alignment scores
        targets: (n_samples, 10) array of true alignment scores

    Returns:
        Dict mapping dimension name to MSE value
    """
    mse_per_dim = {}

    for i, dim_name in enumerate(SCHWARTZ_VALUE_ORDER):
        pred_dim = predictions[:, i]
        target_dim = targets[:, i]
        mse = np.mean((pred_dim - target_dim) ** 2)
        mse_per_dim[dim_name] = float(mse)

    return mse_per_dim


def compute_spearman_per_dimension(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute Spearman correlation for each Schwartz value dimension.

    Spearman correlation measures how well the model preserves the ranking
    of alignment scores, which is useful even when exact values differ.

    Args:
        predictions: (n_samples, 10) array of predicted alignment scores
        targets: (n_samples, 10) array of true alignment scores

    Returns:
        Dict mapping dimension name to Spearman correlation coefficient.
        Returns NaN for dimensions with constant values (can't compute correlation).
    """
    spearman_per_dim = {}

    for i, dim_name in enumerate(SCHWARTZ_VALUE_ORDER):
        pred_dim = predictions[:, i]
        target_dim = targets[:, i]

        # Check for constant values (correlation undefined)
        # Use small threshold instead of exact 0 for floating point robustness
        if np.std(target_dim) < 1e-8 or np.std(pred_dim) < 1e-8:
            spearman_per_dim[dim_name] = float("nan")
            continue

        corr, _ = stats.spearmanr(pred_dim, target_dim)
        spearman_per_dim[dim_name] = float(corr)

    return spearman_per_dim


def compute_mae_per_dimension(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute MAE for each Schwartz value dimension.

    For ordinal outputs {-1, 0, 1}, MAE directly measures average ordinal
    distance: off-by-1 = 1, off-by-2 = 2.

    Args:
        predictions: (n_samples, 10) array of predicted alignment scores
        targets: (n_samples, 10) array of true alignment scores

    Returns:
        Dict mapping dimension name to MAE value
    """
    mae_per_dim = {}

    for i, dim_name in enumerate(SCHWARTZ_VALUE_ORDER):
        pred_dim = predictions[:, i]
        target_dim = targets[:, i]
        mae = np.mean(np.abs(pred_dim - target_dim))
        mae_per_dim[dim_name] = float(mae)

    return mae_per_dim


def compute_qwk_per_dimension(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute Quadratic Weighted Kappa for each Schwartz value dimension.

    QWK is the standard metric for ordinal classification. It measures
    agreement between predicted and true classes, adjusted for chance,
    with quadratic penalty for larger ordinal distances.

    Args:
        predictions: (n_samples, 10) array of predicted alignment scores
        targets: (n_samples, 10) array of true alignment scores

    Returns:
        Dict mapping dimension name to QWK value.
        Returns NaN for dimensions with constant predictions or targets.
    """
    qwk_per_dim = {}

    for i, dim_name in enumerate(SCHWARTZ_VALUE_ORDER):
        pred_dim = discretize_predictions(predictions[:, i])
        target_dim = targets[:, i].astype(int)

        # QWK undefined if either rater is constant
        if len(np.unique(pred_dim)) < 2 or len(np.unique(target_dim)) < 2:
            qwk_per_dim[dim_name] = float("nan")
            continue

        qwk = cohen_kappa_score(target_dim, pred_dim, weights="quadratic", labels=[-1, 0, 1])
        qwk_per_dim[dim_name] = float(qwk)

    return qwk_per_dim


def compute_accuracy_per_dimension(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Compute classification accuracy per dimension.

    Since targets are discrete {-1, 0, 1}, we can measure how often
    predictions round to the correct class.

    Args:
        predictions: (n_samples, 10) array of predicted alignment scores
        targets: (n_samples, 10) array of true alignment scores (discrete)

    Returns:
        Dict mapping dimension name to accuracy (fraction correct)
    """
    acc_per_dim = {}

    for i, dim_name in enumerate(SCHWARTZ_VALUE_ORDER):
        pred_dim = predictions[:, i]
        target_dim = targets[:, i]

        # Discretize predictions to nearest class
        pred_classes = discretize_predictions(pred_dim)

        # Compute accuracy
        correct = (pred_classes == target_dim).mean()
        acc_per_dim[dim_name] = float(correct)

    return acc_per_dim


def compute_recall_per_class(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Compute per-class recall summaries for ordinal predictions.

    Returns:
        Dict with:
        - per_dim: {dimension: {minus1, zero, plus1}}
        - mean: {minus1, zero, plus1}
    """
    pred_classes = discretize_predictions(predictions)
    target_classes = targets.astype(int)

    class_key_by_index = {
        0: "minus1",
        1: "zero",
        2: "plus1",
    }
    recall_lists = {
        "minus1": [],
        "zero": [],
        "plus1": [],
    }
    recall_per_dim = {}

    for i, dim_name in enumerate(SCHWARTZ_VALUE_ORDER):
        cm = confusion_matrix(target_classes[:, i], pred_classes[:, i], labels=[-1, 0, 1])
        dim_recalls = {}
        for class_index, key in class_key_by_index.items():
            row_sum = cm[class_index].sum()
            recall = cm[class_index, class_index] / row_sum if row_sum > 0 else float("nan")
            dim_recalls[key] = float(recall)
            recall_lists[key].append(float(recall))
        recall_per_dim[dim_name] = dim_recalls

    recall_mean = {
        key: _nanmean_or_nan(values)
        for key, values in recall_lists.items()
    }

    return {
        "per_dim": recall_per_dim,
        "mean": recall_mean,
    }


def compute_hedging_per_dimension(
    predictions: np.ndarray,
    *,
    lower: float = -0.3,
    upper: float = 0.3,
) -> dict[str, float]:
    """Compute the fraction of near-neutral predictions for each dimension."""
    hedging_per_dim = {}

    for i, dim_name in enumerate(SCHWARTZ_VALUE_ORDER):
        pred_dim = predictions[:, i]
        hedging = ((pred_dim > lower) & (pred_dim < upper)).mean()
        hedging_per_dim[dim_name] = float(hedging)

    return hedging_per_dim


def compute_calibration_summary(
    predictions: np.ndarray,
    targets: np.ndarray,
    uncertainties: np.ndarray,
) -> dict:
    """Compute global and per-dimension uncertainty calibration diagnostics."""
    errors = np.abs(predictions - targets)
    flat_unc = uncertainties.flatten()
    flat_err = errors.flatten()
    cal_corr = _spearman_correlation(flat_unc, flat_err)

    if np.isnan(cal_corr):
        cal_quality = "unknown"
    elif cal_corr >= 0.3:
        cal_quality = "good"
    elif cal_corr >= 0.1:
        cal_quality = "marginal"
    elif cal_corr >= 0.0:
        cal_quality = "poor"
    else:
        cal_quality = "negative"

    per_dim = {}
    positive_count = 0
    for i, dim_name in enumerate(SCHWARTZ_VALUE_ORDER):
        corr = _spearman_correlation(uncertainties[:, i], errors[:, i])
        per_dim[dim_name] = corr
        if np.isfinite(corr) and corr > 0:
            positive_count += 1

    return {
        "error_uncertainty_correlation": cal_corr,
        "mean_uncertainty": float(uncertainties.mean()),
        "quality": cal_quality,
        "per_dim": per_dim,
        "positive_count": positive_count,
    }


def compute_qwk_nan_dims_count(qwk_per_dim: dict[str, float]) -> int:
    """Count how many dimensions produced NaN QWK."""
    return int(sum(np.isnan(value) for value in qwk_per_dim.values()))


DEFAULT_ORDINAL_SELECTION_POLICY = {
    "name": "qwk_then_recall_guarded",
    "rank_order": [
        "qwk_mean",
        "recall_minus1",
        "calibration_global",
        "hedging_mean",
        "val_loss",
    ],
    "guardrails": {
        "qwk_mean_must_be_finite": True,
        "qwk_nan_dims_count_must_equal_zero": True,
        "calibration_global_must_be_non_negative": True,
    },
    "fallback": "best_finite_qwk",
}


def ordinal_selection_policy_summary() -> dict:
    """Return a copy of the default ordinal checkpoint selection policy."""
    return {
        "name": DEFAULT_ORDINAL_SELECTION_POLICY["name"],
        "rank_order": list(DEFAULT_ORDINAL_SELECTION_POLICY["rank_order"]),
        "guardrails": dict(DEFAULT_ORDINAL_SELECTION_POLICY["guardrails"]),
        "fallback": DEFAULT_ORDINAL_SELECTION_POLICY["fallback"],
    }


def build_ordinal_selection_candidate(
    *,
    epoch: int,
    val_loss: float,
    eval_result: dict,
) -> dict:
    """Build a checkpoint-selection candidate from validation metrics."""
    qwk_mean = float(eval_result.get("qwk_mean", float("nan")))
    recall_minus1 = float(eval_result.get("recall_minus1", float("nan")))
    calibration_global = float(
        eval_result.get("calibration", {}).get("error_uncertainty_correlation", float("nan"))
    )
    hedging_mean = float(eval_result.get("hedging_mean", float("nan")))
    qwk_nan_dims_count = int(eval_result.get("qwk_nan_dims_count", 0))

    ineligible_reasons = []
    if not np.isfinite(qwk_mean):
        ineligible_reasons.append("qwk_mean_non_finite")
    if qwk_nan_dims_count > 0:
        ineligible_reasons.append("qwk_nan_dims_present")
    if np.isfinite(calibration_global) and calibration_global < 0:
        ineligible_reasons.append("negative_calibration")

    return {
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "qwk_mean": qwk_mean,
        "recall_minus1": recall_minus1,
        "calibration_global": calibration_global,
        "hedging_mean": hedging_mean,
        "qwk_nan_dims_count": qwk_nan_dims_count,
        "eligible": len(ineligible_reasons) == 0,
        "ineligible_reasons": ineligible_reasons,
    }


def ordinal_candidate_sort_key(candidate: dict) -> tuple[float, float, float, float, float]:
    """Return the descending sort key for ordinal checkpoint candidates."""
    return (
        _metric_sort_value(candidate.get("qwk_mean", float("nan")), higher_is_better=True),
        _metric_sort_value(candidate.get("recall_minus1", float("nan")), higher_is_better=True),
        _metric_sort_value(candidate.get("calibration_global", float("nan")), higher_is_better=True),
        _metric_sort_value(candidate.get("hedging_mean", float("nan")), higher_is_better=False),
        _metric_sort_value(candidate.get("val_loss", float("nan")), higher_is_better=False),
    )


def is_better_ordinal_candidate(candidate: dict, incumbent: dict | None) -> bool:
    """Return True when the candidate outranks the current incumbent."""
    if incumbent is None:
        return True
    return ordinal_candidate_sort_key(candidate) > ordinal_candidate_sort_key(incumbent)


def _metadata_rows_from_dataloader(
    dataloader: torch.utils.data.DataLoader,
) -> list[dict]:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None or not hasattr(dataset, "get_all_metadata"):
        raise ValueError(
            "Dataloader dataset must expose get_all_metadata() for artifact export."
        )
    metadata_rows = dataset.get_all_metadata()
    if len(metadata_rows) != len(dataset):
        raise ValueError(
            "Dataset metadata length does not match dataset length for artifact export."
        )
    return metadata_rows


def export_ordinal_output_artifact(
    results: dict,
    dataloader: torch.utils.data.DataLoader,
    output_path: str | Path,
    *,
    split: str,
    model_name: str,
) -> str:
    """Export per-sample ordinal logits/probabilities for later analysis."""
    raw_logits = results.get("raw_logits")
    probabilities = results.get("probabilities")
    if raw_logits is None or probabilities is None:
        raise ValueError(
            "Ordinal artifact export requires raw_logits and probabilities in the evaluation results."
        )

    metadata_rows = _metadata_rows_from_dataloader(dataloader)
    predictions = results["predictions"]
    uncertainties = results["uncertainties"]
    targets = results["targets"]
    predicted_classes = probabilities.argmax(axis=-1) - 1

    if len(metadata_rows) != predictions.shape[0]:
        raise ValueError(
            "Number of metadata rows does not match evaluated samples for artifact export."
        )

    rows = []
    for sample_idx, metadata in enumerate(metadata_rows):
        base_row = {
            "persona_id": metadata["persona_id"],
            "t_index": int(metadata["t_index"]),
            "date": metadata["date"],
            "split": split,
            "model_name": model_name,
        }
        for dim_idx, dim_name in enumerate(SCHWARTZ_VALUE_ORDER):
            rows.append(
                {
                    **base_row,
                    "dimension": dim_name,
                    "target": int(targets[sample_idx, dim_idx]),
                    "predicted_class": int(predicted_classes[sample_idx, dim_idx]),
                    "mean_prediction": float(predictions[sample_idx, dim_idx]),
                    "uncertainty": float(uncertainties[sample_idx, dim_idx]),
                    "raw_logits": raw_logits[sample_idx, dim_idx].tolist(),
                    "class_probabilities": probabilities[sample_idx, dim_idx].tolist(),
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(output_path)
    return str(output_path)


def evaluate_with_uncertainty(
    model,
    dataloader: torch.utils.data.DataLoader,
    n_mc_samples: int = 50,
    device: str = "cpu",
    include_ordinal_metrics: bool = False,
    include_raw_outputs: bool = False,
) -> dict:
    """Evaluate model with MC Dropout uncertainty estimation.

    Runs the model with MC Dropout to get both predictions and uncertainty
    estimates, then computes metrics and checks calibration. Works with
    both CriticMLP and ordinal critic models (all now have
    predict_with_uncertainty via OrdinalCriticBase).

    Args:
        model: CriticMLP or ordinal critic model
        dataloader: DataLoader with test/validation data
        n_mc_samples: Number of MC Dropout samples
        device: Device to run evaluation on
        include_ordinal_metrics: If True, also compute MAE and QWK
            in addition to always-computed MSE
        include_raw_outputs: If True for ordinal models, also capture
            deterministic raw logits and class probabilities.

    Returns:
        Dict with predictions, uncertainties, targets, metrics, and calibration.
    """
    model.to(device)
    model.eval()

    all_means = []
    all_stds = []
    all_targets = []
    all_raw_logits = []
    all_probabilities = []
    is_ordinal_model = _is_ordinal_model(model)

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)

        mean, std = model.predict_with_uncertainty(batch_x, n_samples=n_mc_samples)

        all_means.append(mean.cpu().numpy())
        all_stds.append(std.cpu().numpy())
        all_targets.append(batch_y.numpy())

        if include_raw_outputs:
            if not is_ordinal_model or not hasattr(model, "predict_logits_and_probabilities"):
                raise ValueError(
                    "include_raw_outputs=True requires an ordinal model with "
                    "predict_logits_and_probabilities()."
                )
            with torch.no_grad():
                raw_logits, probabilities = model.predict_logits_and_probabilities(batch_x)
            all_raw_logits.append(raw_logits.detach().cpu().numpy())
            all_probabilities.append(probabilities.detach().cpu().numpy())

    if not all_means:
        raise ValueError(
            "Evaluation dataloader produced zero batches. "
            "Check split ratios and dataset size."
        )

    predictions = np.concatenate(all_means, axis=0)
    uncertainties = np.concatenate(all_stds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    spearman_per_dim = compute_spearman_per_dimension(predictions, targets)
    accuracy_per_dim = compute_accuracy_per_dimension(predictions, targets)
    mse_per_dim = compute_mse_per_dimension(predictions, targets)

    calibration = compute_calibration_summary(predictions, targets, uncertainties)

    results = {
        "predictions": predictions,
        "uncertainties": uncertainties,
        "targets": targets,
        "spearman_per_dim": spearman_per_dim,
        "spearman_mean": _nanmean_or_nan(list(spearman_per_dim.values())),
        "accuracy_per_dim": accuracy_per_dim,
        "accuracy_mean": float(np.mean(list(accuracy_per_dim.values()))),
        "mse_per_dim": mse_per_dim,
        "mse_mean": float(np.mean(list(mse_per_dim.values()))),
        "calibration": calibration,
    }

    if include_ordinal_metrics:
        mae_per_dim = compute_mae_per_dimension(predictions, targets)
        qwk_per_dim = compute_qwk_per_dimension(predictions, targets)
        recall_per_class = compute_recall_per_class(predictions, targets)
        hedging_per_dim = compute_hedging_per_dimension(predictions)
        results["mae_per_dim"] = mae_per_dim
        results["mae_mean"] = float(np.mean(list(mae_per_dim.values())))
        results["qwk_per_dim"] = qwk_per_dim
        results["qwk_mean"] = _nanmean_or_nan(list(qwk_per_dim.values()))
        results["qwk_nan_dims_count"] = compute_qwk_nan_dims_count(qwk_per_dim)
        results["recall_per_class"] = recall_per_class
        results["recall_minus1"] = recall_per_class["mean"]["minus1"]
        results["recall_zero"] = recall_per_class["mean"]["zero"]
        results["recall_plus1"] = recall_per_class["mean"]["plus1"]
        results["minority_recall_mean"] = _nanmean_or_nan(
            [
                recall_per_class["mean"]["minus1"],
                recall_per_class["mean"]["plus1"],
            ]
        )
        results["hedging_per_dim"] = hedging_per_dim
        results["hedging_mean"] = float(np.mean(list(hedging_per_dim.values())))

    if include_raw_outputs:
        results["raw_logits"] = np.concatenate(all_raw_logits, axis=0)
        results["probabilities"] = np.concatenate(all_probabilities, axis=0)

    return results


def evaluate_model(
    model,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    include_ordinal_metrics: bool = False,
) -> dict:
    """Evaluate model without uncertainty estimation (faster).

    Automatically detects ordinal models (those with a predict() method)
    and uses predict() to get class labels instead of forward() which
    returns raw logits for ordinal variants.

    Args:
        model: CriticMLP or ordinal critic model
        dataloader: DataLoader with test/validation data
        device: Device to run evaluation on
        include_ordinal_metrics: If True, also compute MAE and QWK
            in addition to always-computed MSE

    Returns:
        Dict with error metrics, Spearman, and accuracy metrics.
        MSE is always included. MAE/QWK are included when
        include_ordinal_metrics=True.
    """
    model.to(device)
    model.eval()

    # Use predict() for ordinal models, forward() for CriticMLP
    use_predict = hasattr(model, "predict") and hasattr(model, "_variant_name")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            pred = model.predict(batch_x) if use_predict else model(batch_x)

            all_predictions.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())

    if not all_predictions:
        raise ValueError(
            "Evaluation dataloader produced zero batches. "
            "Check split ratios and dataset size."
        )

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    spearman_per_dim = compute_spearman_per_dimension(predictions, targets)
    accuracy_per_dim = compute_accuracy_per_dimension(predictions, targets)
    mse_per_dim = compute_mse_per_dimension(predictions, targets)

    results = {
        "predictions": predictions,
        "targets": targets,
        "spearman_per_dim": spearman_per_dim,
        "spearman_mean": _nanmean_or_nan(list(spearman_per_dim.values())),
        "accuracy_per_dim": accuracy_per_dim,
        "accuracy_mean": float(np.mean(list(accuracy_per_dim.values()))),
        "mse_per_dim": mse_per_dim,
        "mse_mean": float(np.mean(list(mse_per_dim.values()))),
    }

    if include_ordinal_metrics:
        mae_per_dim = compute_mae_per_dimension(predictions, targets)
        qwk_per_dim = compute_qwk_per_dimension(predictions, targets)
        results["mae_per_dim"] = mae_per_dim
        results["mae_mean"] = float(np.mean(list(mae_per_dim.values())))
        results["qwk_per_dim"] = qwk_per_dim
        results["qwk_mean"] = _nanmean_or_nan(list(qwk_per_dim.values()))

    return results


def format_results_table(results: dict) -> str:
    """Format evaluation results as a readable table.

    Automatically detects which error metric is present (MAE or MSE)
    and whether QWK is included.

    Args:
        results: Dict from evaluate_model or evaluate_with_uncertainty

    Returns:
        Formatted string table
    """
    has_mae = "mae_per_dim" in results
    has_qwk = "qwk_per_dim" in results
    err_key = "mae_per_dim" if has_mae else "mse_per_dim"
    err_label = "MAE" if has_mae else "MSE"
    err_mean_key = "mae_mean" if has_mae else "mse_mean"

    header = f"{'Dimension':<20} {err_label:>10} {'Spearman':>10} {'Accuracy':>10}"
    if has_qwk:
        header += f" {'QWK':>10}"

    lines = []
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for dim_name in SCHWARTZ_VALUE_ORDER:
        err = results[err_key][dim_name]
        spearman = results["spearman_per_dim"][dim_name]
        accuracy = results["accuracy_per_dim"][dim_name]

        spearman_str = f"{spearman:.3f}" if not np.isnan(spearman) else "N/A"

        row = f"{dim_name:<20} {err:>10.4f} {spearman_str:>10} {accuracy:>10.2%}"
        if has_qwk:
            qwk = results["qwk_per_dim"][dim_name]
            qwk_str = f"{qwk:.3f}" if not np.isnan(qwk) else "N/A"
            row += f" {qwk_str:>10}"
        lines.append(row)

    lines.append("-" * len(header))
    spearman_mean = results["spearman_mean"]
    spearman_mean_str = f"{spearman_mean:.3f}" if not np.isnan(spearman_mean) else "N/A"
    mean_row = (
        f"{'MEAN':<20} {results[err_mean_key]:>10.4f} "
        f"{spearman_mean_str:>10} {results['accuracy_mean']:>10.2%}"
    )
    if has_qwk:
        qwk_mean = results["qwk_mean"]
        qwk_mean_str = f"{qwk_mean:.3f}" if not np.isnan(qwk_mean) else "N/A"
        mean_row += f" {qwk_mean_str:>10}"
    lines.append(mean_row)
    lines.append("=" * len(header))

    if "calibration" in results:
        cal_corr = results["calibration"]["error_uncertainty_correlation"]
        cal_corr_str = f"{cal_corr:.3f}" if not np.isnan(cal_corr) else "N/A"
        lines.append("\nCalibration:")
        lines.append(f"  Error-uncertainty correlation: {cal_corr_str}")
        lines.append(f"  Mean uncertainty: {results['calibration']['mean_uncertainty']:.4f}")
        if "quality" in results["calibration"]:
            quality = results["calibration"]["quality"]
            if cal_corr < 0 if not np.isnan(cal_corr) else False:
                lines.append(
                    "  WARNING: Negative calibration -- uncertainty "
                    "ANTI-correlates with error"
                )
            else:
                lines.append(f"  Quality: {quality}")
        if "positive_count" in results["calibration"]:
            lines.append(f"  Positive dims: {results['calibration']['positive_count']}/10")

    if "recall_per_class" in results:
        recall_mean = results["recall_per_class"]["mean"]
        lines.append("\nRecall per class:")
        lines.append(
            "  -1: "
            f"{recall_mean['minus1']:.2%} | "
            f"0: {recall_mean['zero']:.2%} | "
            f"+1: {recall_mean['plus1']:.2%}"
        )

    if "hedging_mean" in results:
        lines.append(f"\nHedging mean: {results['hedging_mean']:.2%}")

    if "qwk_nan_dims_count" in results:
        lines.append(f"QWK NaN dims: {results['qwk_nan_dims_count']}")

    return "\n".join(lines)
