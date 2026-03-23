"""Training script for VIF Critic model.

This script provides a CLI entry point for training the CriticMLP model
on labeled journal entry data. Supports configuration via YAML file and
command-line overrides for ablation studies.

Usage:
    # Default training
    python -m src.vif.train

    # With config file
    python -m src.vif.train --config config/vif.yaml

    # Quick test run
    python -m src.vif.train --epochs 5 --batch-size 8

    # Ablation study with different encoder
    python -m src.vif.train --encoder-model all-mpnet-base-v2
"""

import argparse
import copy
import json
import math
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.vif.critic import CriticMLP
from src.vif.dataset import create_dataloaders
from src.vif.encoders import create_encoder
from src.vif.eval import evaluate_with_uncertainty, format_results_table
from src.vif.lr_finder import run_lr_finder
from src.vif.state_encoder import StateEncoder


class NonFiniteLossError(ValueError):
    """Raised when training or validation encounters a non-finite loss."""

    def __init__(
        self,
        *,
        phase: str,
        epoch: int,
        batch_index: int,
        loss_name: str,
        loss_value: str,
    ) -> None:
        self.phase = phase
        self.epoch = epoch
        self.batch_index = batch_index
        self.loss_name = loss_name
        self.loss_value = loss_value
        super().__init__(
            "Non-finite "
            f"{loss_name} detected during {phase} at epoch {epoch}, "
            f"batch {batch_index}: {loss_value}"
        )

    def to_metadata(self, *, best_checkpoint_path: str, best_checkpoint_exists: bool) -> dict:
        """Return a JSON-serializable termination payload."""
        return {
            "status": "failed_non_finite_loss",
            "phase": self.phase,
            "epoch": self.epoch,
            "batch_index": self.batch_index,
            "loss_name": self.loss_name,
            "loss_value": self.loss_value,
            "best_checkpoint_path": best_checkpoint_path,
            "best_checkpoint_exists": best_checkpoint_exists,
        }


def load_config(config_path: str | Path | None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML, or None for defaults

    Returns:
        Configuration dict
    """
    # Fallback defaults mirror config/vif.yaml. Both are preliminary and
    # subject to revision through ongoing model ablation studies.
    default_config = {
        "encoder": {
            "type": "sbert",
            "model_name": "nomic-ai/nomic-embed-text-v1.5",
            "trust_remote_code": True,
            "truncate_dim": 256,
            "text_prefix": "classification: ",
            "prompt_name": None,
            "prompt": None,
        },
        "state_encoder": {"window_size": 1},
        "model": {"hidden_dim": 64, "dropout": 0.3, "output_dim": 10},
        "training": {
            "seed": 2025,
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "class_balance_source": "train_split_per_dimension",
            "grad_clip": 1.0,
            "gradient_logging": {
                "enabled": True,
                "sample_every_batches": 1,
            },
            "scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "patience": 10,
                "min_lr": 0.00001,
            },
            "early_stopping": {"patience": 20, "min_delta": 0.001},
            "lr_finder": {
                "enabled": True,
                "start_lr": 0.0000001,
                "end_lr": 1.0,
                "num_iter": 200,
                "output_path": None,
            },
            "dimension_weighting": {
                "enabled": True,
                "mode": "inverse_loss",
                "temperature": 0.5,
                "ema_alpha": 0.3,
                "warmup_epochs": 1,
                "eps": 0.000001,
                "min_weight": 0.5,
                "max_weight": 1.5,
            },
            "circumplex_regularizer": {
                "enabled": False,
                "opposite_weight": 0.0,
                "adjacent_weight": 0.0,
            },
            "ldam_drw": {
                "max_m": 0.5,
                "scale": 30.0,
                "drw_start_epoch": 50,
                "beta": 0.9999,
            },
            "slace": {
                "alpha": 1.0,
            },
        },
        "data": {
            "labels_path": "logs/judge_labels/judge_labels.parquet",
            "wrangled_dir": "logs/wrangled",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "seed": 2025,
            "split_seed": 2025,
            "fixed_holdout_manifest_path": None,
        },
        "mc_dropout": {"n_samples": 50},
        "output": {"checkpoint_dir": "models/vif", "log_dir": "logs/vif_training"},
    }

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                file_config = yaml.safe_load(f)
            # Deep merge file config into defaults
            _deep_update(default_config, file_config)

    return default_config


def resolve_split_seed(config: dict) -> int:
    """Return the persona split seed with legacy fallback support."""
    data_config = config.get("data", {})
    return int(data_config.get("split_seed", data_config.get("seed", 42)))


def resolve_training_seed(config: dict) -> int:
    """Return the model/training seed with split-seed fallback support."""
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    return int(
        training_config.get(
            "seed",
            data_config.get("split_seed", data_config.get("seed", 42)),
        )
    )


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update a dict with another dict."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_gradient_config(training_config: dict) -> dict:
    """Resolve effective gradient clipping and telemetry settings."""
    grad_clip_raw = training_config.get("grad_clip", 1.0)
    grad_clip = None if grad_clip_raw is None else float(grad_clip_raw)
    if grad_clip is not None and grad_clip <= 0:
        grad_clip = None

    gradient_logging = training_config.get("gradient_logging", {})
    gradient_logging_enabled = bool(gradient_logging.get("enabled", True))
    gradient_log_every = int(gradient_logging.get("sample_every_batches", 1))
    if gradient_log_every <= 0:
        raise ValueError("training.gradient_logging.sample_every_batches must be >= 1")

    return {
        "grad_clip": grad_clip,
        "gradient_logging_enabled": gradient_logging_enabled,
        "gradient_log_every": gradient_log_every,
    }


def _format_non_finite_value(value: float) -> str:
    """Return a stable JSON/log-friendly representation of a scalar."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return str(float(value))


def _ensure_finite_loss(
    loss: torch.Tensor,
    *,
    phase: str,
    epoch: int,
    batch_index: int,
    loss_name: str = "mse_loss",
) -> float:
    """Return the scalar loss value or raise if it is NaN/Inf."""
    loss_value = float(loss.item())
    if math.isfinite(loss_value):
        return loss_value

    raise NonFiniteLossError(
        phase=phase,
        epoch=epoch,
        batch_index=batch_index,
        loss_name=loss_name,
        loss_value=_format_non_finite_value(loss_value),
    )


def _find_non_finite_tensor_path(value: object, *, path: str) -> str | None:
    """Return the first tensor path that contains NaN/Inf values."""
    if isinstance(value, torch.Tensor):
        if bool(torch.isfinite(value).all()):
            return None
        return path

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = _find_non_finite_tensor_path(child, path=f"{path}.{key}")
            if child_path is not None:
                return child_path
        return None

    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_path = _find_non_finite_tensor_path(child, path=f"{path}[{index}]")
            if child_path is not None:
                return child_path

    return None


def _compute_total_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """Return the total L2 norm across all parameter gradients."""
    grad_norms = [
        torch.linalg.vector_norm(parameter.grad.detach(), ord=2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grad_norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(grad_norms), ord=2).item())


def _empty_gradient_metrics(*, grad_clip_enabled: bool) -> dict:
    """Return a normalized empty gradient-metrics payload."""
    return {
        "grad_norm_mean": None,
        "grad_norm_max": None,
        "grad_batches_tracked": 0,
        "grad_clipped_fraction": 0.0 if grad_clip_enabled else None,
    }


def _coerce_train_epoch_result(epoch_result: float | tuple[float, dict], *, grad_clip_enabled: bool) -> tuple[float, dict]:
    """Normalize train_epoch() results for callers and tests."""
    if not isinstance(epoch_result, tuple):
        return float(epoch_result), _empty_gradient_metrics(grad_clip_enabled=grad_clip_enabled)

    train_loss, gradient_metrics = epoch_result
    gradient_metrics = gradient_metrics or {}

    return float(train_loss), {
        "grad_norm_mean": (
            float(gradient_metrics["grad_norm_mean"])
            if gradient_metrics.get("grad_norm_mean") is not None
            else None
        ),
        "grad_norm_max": (
            float(gradient_metrics["grad_norm_max"])
            if gradient_metrics.get("grad_norm_max") is not None
            else None
        ),
        "grad_batches_tracked": int(gradient_metrics.get("grad_batches_tracked", 0)),
        "grad_clipped_fraction": (
            float(gradient_metrics["grad_clipped_fraction"])
            if gradient_metrics.get("grad_clipped_fraction") is not None
            else (0.0 if grad_clip_enabled else None)
        ),
    }


def _summarize_gradient_history(history: dict, *, gradient_config: dict) -> dict:
    """Build run-level gradient diagnostics from epoch history."""
    grad_norm_means = [value for value in history.get("grad_norm_mean", []) if value is not None]
    grad_norm_maxes = [value for value in history.get("grad_norm_max", []) if value is not None]
    clipped_fractions = [
        value for value in history.get("grad_clipped_fraction", [])
        if value is not None
    ]

    return {
        "gradient_config": {
            "grad_clip": gradient_config["grad_clip"],
            "gradient_logging_enabled": gradient_config["gradient_logging_enabled"],
            "sample_every_batches": gradient_config["gradient_log_every"],
        },
        "gradient_summary": {
            "epochs_with_gradient_samples": len(grad_norm_means),
            "total_gradient_batches_tracked": int(sum(history.get("grad_batches_tracked", []))),
            "grad_norm_mean_over_epochs": (
                float(np.mean(grad_norm_means)) if grad_norm_means else None
            ),
            "grad_norm_max_over_epochs": (
                float(max(grad_norm_maxes)) if grad_norm_maxes else None
            ),
            "grad_clipped_fraction_mean": (
                float(np.mean(clipped_fractions)) if clipped_fractions else None
            ),
            "grad_clipped_fraction_max": (
                float(max(clipped_fractions)) if clipped_fractions else None
            ),
        },
    }


def resolve_training_curve_paths(output_dir: str | Path) -> tuple[Path, Path]:
    """Return standard artifact paths for training-curve exports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "training_curves.png", output_dir / "training_curves.json"


def save_training_curve_artifacts(
    history: dict,
    best_epoch: int | None,
    output_dir: str | Path,
) -> dict[str, str]:
    """Persist training-curve artifacts for quick review and structured reuse."""
    plot_path, history_path = resolve_training_curve_paths(output_dir)
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    curve_history = {
        "epochs": epochs,
        "train_loss": history.get("train_loss", []),
        "val_loss": history.get("val_loss", []),
        "train_val_gap": history.get("train_val_gap", []),
        "learning_rate": history.get("learning_rate", []),
        "best_epoch": None if best_epoch is None else best_epoch + 1,
    }
    with open(history_path, "w") as f:
        json.dump(curve_history, f, indent=2)

    fig, (loss_ax, lr_ax) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    if epochs:
        loss_ax.plot(epochs, history.get("train_loss", []), label="Train", alpha=0.85)
        loss_ax.plot(epochs, history.get("val_loss", []), label="Val", alpha=0.85)
        if best_epoch is not None and 0 <= best_epoch < len(epochs):
            loss_ax.axvline(
                best_epoch + 1,
                color="green",
                linestyle=":",
                alpha=0.6,
                label=f"Best ({best_epoch + 1})",
            )
        lr_ax.plot(epochs, history.get("learning_rate", []), color="tab:orange")
    else:
        loss_ax.text(0.5, 0.5, "No training history", ha="center", va="center")
        lr_ax.text(0.5, 0.5, "No LR history", ha="center", va="center")

    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training Curves")
    handles, labels = loss_ax.get_legend_handles_labels()
    if handles:
        loss_ax.legend(handles, labels)
    loss_ax.grid(True, alpha=0.3)

    lr_ax.set_xlabel("Epoch")
    lr_ax.set_ylabel("Learning Rate")
    lr_ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "plot_path": str(plot_path),
        "history_path": str(history_path),
    }


def _summarize_gap_history(history: dict, *, best_epoch: int | None) -> dict:
    """Build run-level train/val gap diagnostics from recorded history."""
    train_val_gap = [float(value) for value in history.get("train_val_gap", [])]
    total_epochs = len(history.get("train_loss", []))
    gap_at_best = None
    if best_epoch is not None and best_epoch < len(train_val_gap):
        gap_at_best = float(train_val_gap[best_epoch])

    return {
        "best_epoch": None if best_epoch is None else best_epoch + 1,
        "total_epochs": total_epochs,
        "gap_at_best": gap_at_best,
        "gap_at_final": float(train_val_gap[-1]) if train_val_gap else None,
        "max_gap": float(max(train_val_gap)) if train_val_gap else None,
        "min_gap": float(min(train_val_gap)) if train_val_gap else None,
    }


def _summarize_training_dynamics(
    history: dict,
    *,
    gradient_config: dict,
    best_epoch: int | None,
    curve_artifacts: dict[str, str],
    termination: dict | None = None,
) -> dict:
    """Assemble additive training_dynamics payload for training_log.json."""
    training_dynamics = _summarize_gradient_history(
        history,
        gradient_config=gradient_config,
    )
    training_dynamics["gap_summary"] = _summarize_gap_history(
        history,
        best_epoch=best_epoch,
    )
    training_dynamics["curve_artifacts"] = curve_artifacts
    if termination is not None:
        training_dynamics["termination"] = termination
    return training_dynamics


def _build_test_metrics_payload(test_results: dict | None) -> dict | None:
    """Normalize optional final evaluation metrics for training_log.json."""
    if test_results is None:
        return None

    return {
        "mse_mean": test_results["mse_mean"],
        "spearman_mean": test_results["spearman_mean"],
        "accuracy_mean": test_results["accuracy_mean"],
        "mse_per_dim": test_results["mse_per_dim"],
        "spearman_per_dim": test_results["spearman_per_dim"],
        "accuracy_per_dim": test_results["accuracy_per_dim"],
    }


def _persist_training_log(
    *,
    output_dir: Path,
    history: dict,
    best_epoch: int | None,
    best_val_loss: float,
    training_time: float,
    gradient_config: dict,
    lr_finder_result: dict,
    run_config: dict,
    test_results: dict | None = None,
    termination: dict | None = None,
) -> tuple[Path, dict]:
    """Write training_log.json and return its path plus training dynamics."""
    log_path = output_dir / "training_log.json"
    curve_artifacts = save_training_curve_artifacts(
        history,
        best_epoch,
        output_dir,
    )
    training_dynamics = _summarize_training_dynamics(
        history,
        gradient_config=gradient_config,
        best_epoch=best_epoch,
        curve_artifacts=curve_artifacts,
        termination=termination,
    )
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "training_time_seconds": training_time,
        "epochs_completed": len(history["train_loss"]),
        "best_val_loss": float(best_val_loss) if math.isfinite(best_val_loss) else None,
        "history": history,
        "training_dynamics": training_dynamics,
        "test_metrics": _build_test_metrics_payload(test_results),
        "lr_finder": lr_finder_result,
        "config": run_config,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_path, training_dynamics


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    *,
    epoch: int = 1,
    grad_clip: float | None = None,
    gradient_logging_enabled: bool = True,
    gradient_log_every: int = 1,
) -> tuple[float, dict]:
    """Train for one epoch.

    Args:
        model: CriticMLP model
        dataloader: Training DataLoader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on
        epoch: 1-indexed epoch number for diagnostics
        grad_clip: Total gradient-norm clipping threshold, or None to disable
        gradient_logging_enabled: Whether to record gradient telemetry
        gradient_log_every: Sample gradient telemetry every N batches

    Returns:
        Tuple of average training loss and epoch-level gradient diagnostics
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    grad_norm_total = 0.0
    grad_norm_max = None
    grad_batches_tracked = 0
    clipped_batches = 0
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    for batch_idx, (batch_x, batch_y) in enumerate(dataloader, start=1):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss_value = _ensure_finite_loss(
            loss,
            phase="train",
            epoch=epoch,
            batch_index=batch_idx,
        )
        loss.backward()

        sampled_for_logging = (
            gradient_logging_enabled and (batch_idx - 1) % gradient_log_every == 0
        )
        grad_norm = None
        if grad_clip is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=grad_clip)
            )
            if grad_norm > grad_clip:
                clipped_batches += 1
        elif sampled_for_logging:
            grad_norm = _compute_total_grad_norm(parameters)

        if sampled_for_logging:
            if grad_norm is None:
                grad_norm = _compute_total_grad_norm(parameters)
            grad_norm_total += grad_norm
            grad_norm_max = grad_norm if grad_norm_max is None else max(grad_norm_max, grad_norm)
            grad_batches_tracked += 1

        optimizer.step()

        total_loss += loss_value
        n_batches += 1

    if n_batches == 0:
        raise ValueError(
            "Training dataloader produced zero batches. "
            "Check split ratios and dataset size."
        )

    gradient_metrics = {
        "grad_norm_mean": (
            grad_norm_total / grad_batches_tracked if grad_batches_tracked else None
        ),
        "grad_norm_max": grad_norm_max,
        "grad_batches_tracked": grad_batches_tracked,
        "grad_clipped_fraction": (
            clipped_batches / n_batches if grad_clip is not None else None
        ),
    }
    return total_loss / n_batches, gradient_metrics


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    *,
    epoch: int = 1,
) -> float:
    """Validate the model.

    Args:
        model: CriticMLP model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Device to validate on
        epoch: 1-indexed epoch number for diagnostics

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader, start=1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss_value = _ensure_finite_loss(
                loss,
                phase="val",
                epoch=epoch,
                batch_index=batch_idx,
            )

            total_loss += loss_value
            n_batches += 1

    if n_batches == 0:
        raise ValueError(
            "Validation dataloader produced zero batches. "
            "Check split ratios and dataset size."
        )

    return total_loss / n_batches


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    config: dict,
    output_dir: Path,
    filename: str = "best_model.pt",
):
    """Save model checkpoint.

    Args:
        model: CriticMLP model
        optimizer: Optimizer instance
        epoch: Current epoch
        val_loss: Validation loss
        config: Training configuration
        output_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "model_config": model.get_config(),
        "training_config": config,
    }

    val_loss_value = float(val_loss)
    if not math.isfinite(val_loss_value):
        raise ValueError(
            "Refusing to save checkpoint "
            f"{filename!r} with non-finite val_loss={_format_non_finite_value(val_loss_value)}"
        )

    non_finite_tensor_path = _find_non_finite_tensor_path(checkpoint, path="checkpoint")
    if non_finite_tensor_path is not None:
        raise ValueError(
            "Refusing to save checkpoint "
            f"{filename!r} because {non_finite_tensor_path} contains non-finite values"
        )

    torch.save(checkpoint, output_dir / filename)

    # Also save config as JSON for easy inspection
    config_path = output_dir / filename.replace(".pt", "_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "epoch": epoch,
                "val_loss": val_loss_value,
                "model_config": model.get_config(),
                "training_config": config,
            },
            f,
            indent=2,
        )


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[CriticMLP, dict]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Tuple of (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = CriticMLP.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model, checkpoint


def train(config: dict, verbose: bool = True) -> dict:
    """Main training function.

    Args:
        config: Training configuration dict
        verbose: Whether to print progress

    Returns:
        Dict with training history and final metrics
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")

    # Create encoder and state encoder
    if verbose:
        print(f"Loading encoder: {config['encoder']['model_name']}...")
    text_encoder = create_encoder(config["encoder"])
    state_encoder = StateEncoder(
        text_encoder,
        window_size=config["state_encoder"]["window_size"],
    )

    if verbose:
        print(f"State dimension: {state_encoder.state_dim}")

    # Create dataloaders
    if verbose:
        print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        state_encoder,
        batch_size=config["training"]["batch_size"],
        seed=resolve_split_seed(config),
        labels_path=config["data"]["labels_path"],
        wrangled_dir=config["data"]["wrangled_dir"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        fixed_holdout_manifest_path=config["data"].get("fixed_holdout_manifest_path"),
    )

    split_sizes = {
        "train": len(train_loader.dataset),
        "val": len(val_loader.dataset),
        "test": len(test_loader.dataset),
    }
    empty_splits = [name for name, size in split_sizes.items() if size == 0]
    if empty_splits:
        joined = ", ".join(empty_splits)
        raise ValueError(
            f"Empty dataset split(s): {joined}. "
            "Adjust train/val ratios or provide more persona data."
        )

    if verbose:
        print(
            f"Split ratios: train={config['data']['train_ratio']:.0%}, "
            f"val={config['data']['val_ratio']:.0%}, "
            f"test={1 - config['data']['train_ratio'] - config['data']['val_ratio']:.0%}"
        )
        print(f"Train: {len(train_loader.dataset)} samples")
        print(f"Val: {len(val_loader.dataset)} samples")
        print(f"Test: {len(test_loader.dataset)} samples")

    # Create model
    model = CriticMLP(
        input_dim=state_encoder.state_dim,
        hidden_dim=config["model"]["hidden_dim"],
        dropout=config["model"]["dropout"],
        output_dim=config["model"]["output_dim"],
    )
    model.to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

    # Setup training
    criterion = nn.MSELoss()
    output_dir = Path(config["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    gradient_config = _resolve_gradient_config(config["training"])

    configured_learning_rate = config["training"]["learning_rate"]
    lr_finder_config = config["training"].get("lr_finder", {})
    lr_finder_enabled = lr_finder_config.get("enabled", True)
    start_lr = lr_finder_config.get("start_lr")
    end_lr = lr_finder_config.get("end_lr")
    num_iter = lr_finder_config.get("num_iter")
    scheduler_min_lr = config["training"]["scheduler"].get("min_lr")

    missing = []
    if configured_learning_rate is None:
        missing.append("training.learning_rate")
    if lr_finder_enabled and start_lr is None:
        missing.append("training.lr_finder.start_lr")
    if lr_finder_enabled and end_lr is None:
        missing.append("training.lr_finder.end_lr")
    if lr_finder_enabled and num_iter is None:
        missing.append("training.lr_finder.num_iter")
    if scheduler_min_lr is None:
        missing.append("training.scheduler.min_lr")
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Missing required LR configuration keys: {missing_str}. "
            "Set them in config/vif.yaml or via CLI overrides."
        )

    if lr_finder_enabled:
        if verbose:
            print("\nRunning default LR finder pass...")

        lr_finder_result = run_lr_finder(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            configured_learning_rate=configured_learning_rate,
            weight_decay=config["training"]["weight_decay"],
            device=device,
            output_dir=output_dir,
            output_plot_path=lr_finder_config.get("output_path"),
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=int(num_iter),
            max_selected_lr=lr_finder_config.get("max_selected_lr"),
        )
        selected_learning_rate = lr_finder_result["lr_selected"]

        if verbose:
            steep = lr_finder_result["suggestions"]["lr_steep"]
            valley = lr_finder_result["suggestions"]["lr_valley"]
            print(
                "LR finder suggestions: "
                f"lr_steep={steep if steep is not None else 'N/A'}, "
                f"lr_valley={valley if valley is not None else 'N/A'}"
            )
            print(
                f"LR selection source: {lr_finder_result.get('lr_selected_source', 'unknown')}"
            )
            if lr_finder_result.get("fallback_reason"):
                print(f"LR selection note: {lr_finder_result['fallback_reason']}")
            print(f"Using training learning rate: {selected_learning_rate:.6f}")
            print(f"LR finder plot: {lr_finder_result['artifacts']['plot_path']}")
    else:
        selected_learning_rate = configured_learning_rate
        lr_finder_result = {
            "enabled": False,
            "params": None,
            "suggestions": {"lr_steep": None, "lr_valley": None},
            "lr_selected": selected_learning_rate,
            "lr_selected_source": "configured_learning_rate",
            "configured_learning_rate": configured_learning_rate,
            "fallback_reason": "lr_finder_disabled",
            "history_points": 0,
            "artifacts": {
                "plot_path": None,
                "history_path": None,
            },
        }
        if verbose:
            print("\nLR finder disabled; using configured learning rate directly.")

    optimizer = AdamW(
        model.parameters(),
        lr=selected_learning_rate,
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["scheduler"]["factor"],
        patience=config["training"]["scheduler"]["patience"],
        min_lr=float(scheduler_min_lr),
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_val_gap": [],
        "learning_rate": [],
        "grad_norm_mean": [],
        "grad_norm_max": [],
        "grad_batches_tracked": [],
        "grad_clipped_fraction": [],
    }

    best_val_loss = float("inf")
    best_epoch = None
    epochs_without_improvement = 0
    early_stop_patience = config["training"]["early_stopping"]["patience"]
    early_stop_delta = config["training"]["early_stopping"]["min_delta"]

    run_config = copy.deepcopy(config)
    run_config["training"]["learning_rate_configured"] = configured_learning_rate
    run_config["training"]["learning_rate"] = selected_learning_rate
    run_config["training"]["grad_clip"] = config["training"].get("grad_clip", 1.0)
    run_config["training"].setdefault("gradient_logging", {})
    run_config["training"]["gradient_logging"]["enabled"] = (
        config["training"].get("gradient_logging", {}).get("enabled", True)
    )
    run_config["training"]["gradient_logging"]["sample_every_batches"] = (
        config["training"].get("gradient_logging", {}).get("sample_every_batches", 1)
    )

    # Training loop
    if verbose:
        print("\nStarting training...")
    start_time = time.time()

    try:
        for epoch in range(config["training"]["epochs"]):
            # Train
            train_epoch_result = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                epoch=epoch + 1,
                grad_clip=gradient_config["grad_clip"],
                gradient_logging_enabled=gradient_config["gradient_logging_enabled"],
                gradient_log_every=gradient_config["gradient_log_every"],
            )
            train_loss, gradient_metrics = _coerce_train_epoch_result(
                train_epoch_result,
                grad_clip_enabled=gradient_config["grad_clip"] is not None,
            )

            # Validate
            val_loss = validate(
                model,
                val_loader,
                criterion,
                device,
                epoch=epoch + 1,
            )

            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_val_gap"].append(val_loss - train_loss)
            history["learning_rate"].append(current_lr)
            history["grad_norm_mean"].append(gradient_metrics["grad_norm_mean"])
            history["grad_norm_max"].append(gradient_metrics["grad_norm_max"])
            history["grad_batches_tracked"].append(gradient_metrics["grad_batches_tracked"])
            history["grad_clipped_fraction"].append(gradient_metrics["grad_clipped_fraction"])

            # Check for improvement
            if val_loss < best_val_loss - early_stop_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint(model, optimizer, epoch, val_loss, run_config, output_dir)
                if verbose:
                    print(
                        f"Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                        f"lr={current_lr:.6f} [BEST]"
                    )
            else:
                epochs_without_improvement += 1
                if verbose and epoch % 10 == 0:
                    print(
                        f"Epoch {epoch+1:3d}: train={train_loss:.4f}, val={val_loss:.4f}, "
                        f"lr={current_lr:.6f}"
                    )

            # Early stopping
            if epochs_without_improvement >= early_stop_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
    except NonFiniteLossError as exc:
        training_time = time.time() - start_time
        best_checkpoint_path = output_dir / "best_model.pt"
        termination = exc.to_metadata(
            best_checkpoint_path=str(best_checkpoint_path),
            best_checkpoint_exists=best_checkpoint_path.exists(),
        )
        log_path, _ = _persist_training_log(
            output_dir=output_dir,
            history=history,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            training_time=training_time,
            gradient_config=gradient_config,
            lr_finder_result=lr_finder_result,
            run_config=run_config,
            termination=termination,
        )
        if verbose:
            print(f"\nTraining terminated early: {exc}")
            print(f"Training log saved to: {log_path}")
        raise

    training_time = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {training_time:.1f}s")

    # Load best model for final evaluation
    model, _ = load_checkpoint(output_dir / "best_model.pt", device)

    # Final evaluation on test set
    if verbose:
        print("\nEvaluating on test set...")
    test_results = evaluate_with_uncertainty(
        model,
        test_loader,
        n_mc_samples=config["mc_dropout"]["n_samples"],
        device=device,
    )

    if verbose:
        print("\nTest Results:")
        print(format_results_table(test_results))

    log_path, training_dynamics = _persist_training_log(
        output_dir=output_dir,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        training_time=training_time,
        gradient_config=gradient_config,
        lr_finder_result=lr_finder_result,
        run_config=run_config,
        test_results=test_results,
    )

    if verbose:
        print(f"\nCheckpoint saved to: {output_dir / 'best_model.pt'}")
        print(f"Training log saved to: {log_path}")

    return {
        "history": history,
        "test_results": test_results,
        "best_val_loss": best_val_loss,
        "training_time": training_time,
        "lr_finder": lr_finder_result,
        "learning_rate_configured": configured_learning_rate,
        "learning_rate_applied": selected_learning_rate,
        "training_dynamics": training_dynamics,
        "log_path": str(log_path),
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train VIF Critic model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.vif.train
    python -m src.vif.train --config config/vif.yaml
    python -m src.vif.train --epochs 5 --batch-size 8
    python -m src.vif.train --encoder-model all-mpnet-base-v2
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/vif.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override learning rate",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        help="Clip total gradient L2 norm at this value (<=0 disables clipping)",
    )
    parser.add_argument(
        "--no-log-gradients",
        action="store_true",
        help="Disable gradient telemetry in training_log.json",
    )
    parser.add_argument(
        "--grad-log-every",
        type=int,
        help="Record gradient telemetry every N training batches",
    )
    parser.add_argument(
        "--encoder-model",
        type=str,
        help="Override encoder model name",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        help="Override hidden dimension",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--lr-find-output-path",
        type=str,
        help="Optional path to save LR finder plot (history JSON shares same stem)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.grad_clip is not None:
        config["training"]["grad_clip"] = args.grad_clip
    if args.no_log_gradients:
        config["training"].setdefault("gradient_logging", {})
        config["training"]["gradient_logging"]["enabled"] = False
    if args.grad_log_every is not None:
        config["training"].setdefault("gradient_logging", {})
        config["training"]["gradient_logging"]["sample_every_batches"] = args.grad_log_every
    if args.encoder_model is not None:
        config["encoder"]["model_name"] = args.encoder_model
    if args.hidden_dim is not None:
        config["model"]["hidden_dim"] = args.hidden_dim
    if args.seed is not None:
        config["data"]["seed"] = args.seed
        config["data"]["split_seed"] = args.seed
        config["training"]["seed"] = args.seed
    if args.lr_find_output_path is not None:
        config["training"].setdefault("lr_finder", {})
        config["training"]["lr_finder"]["output_path"] = args.lr_find_output_path

    # Set random seeds
    training_seed = resolve_training_seed(config)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)

    # Train
    results = train(config, verbose=not args.quiet)

    if not args.quiet:
        print(f"Configured LR: {results['learning_rate_configured']:.6f}")
        print(f"Applied LR:    {results['learning_rate_applied']:.6f}")
        print(f"LR plot:       {results['lr_finder']['artifacts']['plot_path']}")

    print(f"\nFinal test MSE: {results['test_results']['mse_mean']:.4f}")
    print(f"Final test Spearman: {results['test_results']['spearman_mean']:.4f}")
    print(f"Final test Accuracy: {results['test_results']['accuracy_mean']:.2%}")


if __name__ == "__main__":
    main()
