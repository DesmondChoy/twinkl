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

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import cohen_kappa_score

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
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification accuracy per dimension.

    Since targets are discrete {-1, 0, 1}, we can measure how often
    predictions round to the correct class.

    Args:
        predictions: (n_samples, 10) array of predicted alignment scores
        targets: (n_samples, 10) array of true alignment scores (discrete)
        threshold: Distance from target to count as correct (default: 0.5)

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


def evaluate_with_uncertainty(
    model,
    dataloader: torch.utils.data.DataLoader,
    n_mc_samples: int = 50,
    device: str = "cpu",
    include_ordinal_metrics: bool = False,
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

    Returns:
        Dict with predictions, uncertainties, targets, metrics, and calibration.
    """
    model.to(device)
    model.eval()

    all_means = []
    all_stds = []
    all_targets = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)

        mean, std = model.predict_with_uncertainty(batch_x, n_samples=n_mc_samples)

        all_means.append(mean.cpu().numpy())
        all_stds.append(std.cpu().numpy())
        all_targets.append(batch_y.numpy())

    predictions = np.concatenate(all_means, axis=0)
    uncertainties = np.concatenate(all_stds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute metrics
    spearman_per_dim = compute_spearman_per_dimension(predictions, targets)
    accuracy_per_dim = compute_accuracy_per_dimension(predictions, targets)

    # Calibration check: higher uncertainty should correlate with higher error
    errors = np.abs(predictions - targets)
    error_uncertainty_corr = stats.spearmanr(
        uncertainties.flatten(),
        errors.flatten(),
    )[0]

    results = {
        "predictions": predictions,
        "uncertainties": uncertainties,
        "targets": targets,
        "spearman_per_dim": spearman_per_dim,
        "spearman_mean": float(np.nanmean(list(spearman_per_dim.values()))),
        "accuracy_per_dim": accuracy_per_dim,
        "accuracy_mean": float(np.mean(list(accuracy_per_dim.values()))),
        "calibration": {
            "error_uncertainty_correlation": float(error_uncertainty_corr),
            "mean_uncertainty": float(uncertainties.mean()),
        },
    }

    if include_ordinal_metrics:
        mae_per_dim = compute_mae_per_dimension(predictions, targets)
        qwk_per_dim = compute_qwk_per_dimension(predictions, targets)
        results["mae_per_dim"] = mae_per_dim
        results["mae_mean"] = float(np.mean(list(mae_per_dim.values())))
        results["qwk_per_dim"] = qwk_per_dim
        results["qwk_mean"] = float(np.nanmean(list(qwk_per_dim.values())))
    else:
        mse_per_dim = compute_mse_per_dimension(predictions, targets)
        results["mse_per_dim"] = mse_per_dim
        results["mse_mean"] = float(np.mean(list(mse_per_dim.values())))

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
            (replaces MSE with MAE in results)

    Returns:
        Dict with error metrics, Spearman, and accuracy metrics.
        When include_ordinal_metrics=True, uses MAE/QWK instead of MSE.
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

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    spearman_per_dim = compute_spearman_per_dimension(predictions, targets)
    accuracy_per_dim = compute_accuracy_per_dimension(predictions, targets)

    results = {
        "predictions": predictions,
        "targets": targets,
        "spearman_per_dim": spearman_per_dim,
        "spearman_mean": float(np.nanmean(list(spearman_per_dim.values()))),
        "accuracy_per_dim": accuracy_per_dim,
        "accuracy_mean": float(np.mean(list(accuracy_per_dim.values()))),
    }

    if include_ordinal_metrics:
        mae_per_dim = compute_mae_per_dimension(predictions, targets)
        qwk_per_dim = compute_qwk_per_dimension(predictions, targets)
        results["mae_per_dim"] = mae_per_dim
        results["mae_mean"] = float(np.mean(list(mae_per_dim.values())))
        results["qwk_per_dim"] = qwk_per_dim
        results["qwk_mean"] = float(np.nanmean(list(qwk_per_dim.values())))
    else:
        mse_per_dim = compute_mse_per_dimension(predictions, targets)
        results["mse_per_dim"] = mse_per_dim
        results["mse_mean"] = float(np.mean(list(mse_per_dim.values())))

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
    mean_row = (
        f"{'MEAN':<20} {results[err_mean_key]:>10.4f} "
        f"{results['spearman_mean']:>10.3f} {results['accuracy_mean']:>10.2%}"
    )
    if has_qwk:
        mean_row += f" {results['qwk_mean']:>10.3f}"
    lines.append(mean_row)
    lines.append("=" * len(header))

    if "calibration" in results:
        lines.append("\nCalibration:")
        lines.append(f"  Error-uncertainty correlation: {results['calibration']['error_uncertainty_correlation']:.3f}")
        lines.append(f"  Mean uncertainty: {results['calibration']['mean_uncertainty']:.4f}")

    return "\n".join(lines)
