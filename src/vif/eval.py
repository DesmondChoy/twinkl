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

from src.models.judge import SCHWARTZ_VALUE_ORDER


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
        if np.std(target_dim) == 0 or np.std(pred_dim) == 0:
            spearman_per_dim[dim_name] = float("nan")
            continue

        corr, _ = stats.spearmanr(pred_dim, target_dim)
        spearman_per_dim[dim_name] = float(corr)

    return spearman_per_dim


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

        # Round predictions to nearest class
        pred_classes = np.round(pred_dim).clip(-1, 1)

        # Compute accuracy
        correct = (pred_classes == target_dim).mean()
        acc_per_dim[dim_name] = float(correct)

    return acc_per_dim


def evaluate_with_uncertainty(
    model,
    dataloader: torch.utils.data.DataLoader,
    n_mc_samples: int = 50,
    device: str = "cpu",
) -> dict:
    """Evaluate model with MC Dropout uncertainty estimation.

    Runs the model with MC Dropout to get both predictions and uncertainty
    estimates, then computes metrics and checks calibration.

    Args:
        model: CriticMLP model
        dataloader: DataLoader with test/validation data
        n_mc_samples: Number of MC Dropout samples
        device: Device to run evaluation on

    Returns:
        Dict with keys:
        - predictions: (n_samples, 10) mean predictions
        - uncertainties: (n_samples, 10) std estimates
        - targets: (n_samples, 10) true values
        - mse_per_dim: Per-dimension MSE
        - spearman_per_dim: Per-dimension Spearman correlation
        - calibration: Calibration statistics
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
    mse_per_dim = compute_mse_per_dimension(predictions, targets)
    spearman_per_dim = compute_spearman_per_dimension(predictions, targets)
    accuracy_per_dim = compute_accuracy_per_dimension(predictions, targets)

    # Calibration check: higher uncertainty should correlate with higher error
    errors = np.abs(predictions - targets)
    error_uncertainty_corr = stats.spearmanr(
        uncertainties.flatten(),
        errors.flatten(),
    )[0]

    return {
        "predictions": predictions,
        "uncertainties": uncertainties,
        "targets": targets,
        "mse_per_dim": mse_per_dim,
        "mse_mean": float(np.mean(list(mse_per_dim.values()))),
        "spearman_per_dim": spearman_per_dim,
        "spearman_mean": float(np.nanmean(list(spearman_per_dim.values()))),
        "accuracy_per_dim": accuracy_per_dim,
        "accuracy_mean": float(np.mean(list(accuracy_per_dim.values()))),
        "calibration": {
            "error_uncertainty_correlation": float(error_uncertainty_corr),
            "mean_uncertainty": float(uncertainties.mean()),
        },
    }


def evaluate_model(
    model,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> dict:
    """Evaluate model without uncertainty estimation (faster).

    Args:
        model: CriticMLP model
        dataloader: DataLoader with test/validation data
        device: Device to run evaluation on

    Returns:
        Dict with MSE, Spearman, and accuracy metrics
    """
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            pred = model(batch_x)

            all_predictions.append(pred.cpu().numpy())
            all_targets.append(batch_y.numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mse_per_dim = compute_mse_per_dimension(predictions, targets)
    spearman_per_dim = compute_spearman_per_dimension(predictions, targets)
    accuracy_per_dim = compute_accuracy_per_dimension(predictions, targets)

    return {
        "predictions": predictions,
        "targets": targets,
        "mse_per_dim": mse_per_dim,
        "mse_mean": float(np.mean(list(mse_per_dim.values()))),
        "spearman_per_dim": spearman_per_dim,
        "spearman_mean": float(np.nanmean(list(spearman_per_dim.values()))),
        "accuracy_per_dim": accuracy_per_dim,
        "accuracy_mean": float(np.mean(list(accuracy_per_dim.values()))),
    }


def format_results_table(results: dict) -> str:
    """Format evaluation results as a readable table.

    Args:
        results: Dict from evaluate_model or evaluate_with_uncertainty

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"{'Dimension':<20} {'MSE':>10} {'Spearman':>10} {'Accuracy':>10}")
    lines.append("-" * 70)

    for dim_name in SCHWARTZ_VALUE_ORDER:
        mse = results["mse_per_dim"][dim_name]
        spearman = results["spearman_per_dim"][dim_name]
        accuracy = results["accuracy_per_dim"][dim_name]

        spearman_str = f"{spearman:.3f}" if not np.isnan(spearman) else "N/A"

        lines.append(f"{dim_name:<20} {mse:>10.4f} {spearman_str:>10} {accuracy:>10.2%}")

    lines.append("-" * 70)
    lines.append(
        f"{'MEAN':<20} {results['mse_mean']:>10.4f} "
        f"{results['spearman_mean']:>10.3f} {results['accuracy_mean']:>10.2%}"
    )
    lines.append("=" * 70)

    if "calibration" in results:
        lines.append("\nCalibration:")
        lines.append(f"  Error-uncertainty correlation: {results['calibration']['error_uncertainty_correlation']:.3f}")
        lines.append(f"  Mean uncertainty: {results['calibration']['mean_uncertainty']:.4f}")

    return "\n".join(lines)
