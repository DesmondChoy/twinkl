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
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.vif.critic import CriticMLP
from src.vif.dataset import create_dataloaders
from src.vif.encoders import create_encoder
from src.vif.eval import evaluate_model, evaluate_with_uncertainty, format_results_table
from src.vif.state_encoder import StateEncoder


def load_config(config_path: str | Path | None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config YAML, or None for defaults

    Returns:
        Configuration dict
    """
    default_config = {
        "encoder": {
            "type": "sbert",
            "model_name": "nomic-ai/nomic-embed-text-v1.5",
            "trust_remote_code": True,
            "truncate_dim": 256,
            "text_prefix": "classification: ",
        },
        "state_encoder": {"window_size": 1, "ema_alpha": 0.3},
        "model": {"hidden_dim": 32, "dropout": 0.3, "output_dim": 10},
        "training": {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "scheduler": {"type": "reduce_on_plateau", "factor": 0.5, "patience": 10, "min_lr": 1e-5},
            "early_stopping": {"patience": 20, "min_delta": 0.001},
        },
        "data": {
            "labels_path": "logs/judge_labels/judge_labels.parquet",
            "wrangled_dir": "logs/wrangled",
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "seed": 42,
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


def _deep_update(base: dict, update: dict) -> dict:
    """Recursively update a dict with another dict."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch.

    Args:
        model: CriticMLP model
        dataloader: Training DataLoader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device to train on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """Validate the model.

    Args:
        model: CriticMLP model
        dataloader: Validation DataLoader
        criterion: Loss function
        device: Device to validate on

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)

            total_loss += loss.item()
            n_batches += 1

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

    torch.save(checkpoint, output_dir / filename)

    # Also save config as JSON for easy inspection
    config_path = output_dir / filename.replace(".pt", "_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "epoch": epoch,
                "val_loss": val_loss,
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
        ema_alpha=config["state_encoder"]["ema_alpha"],
    )

    if verbose:
        print(f"State dimension: {state_encoder.state_dim}")

    # Create dataloaders
    if verbose:
        print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        state_encoder,
        batch_size=config["training"]["batch_size"],
        seed=config["data"]["seed"],
        labels_path=config["data"]["labels_path"],
        wrangled_dir=config["data"]["wrangled_dir"],
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
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
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["scheduler"]["factor"],
        patience=config["training"]["scheduler"]["patience"],
        min_lr=config["training"]["scheduler"]["min_lr"],
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stop_patience = config["training"]["early_stopping"]["patience"]
    early_stop_delta = config["training"]["early_stopping"]["min_delta"]

    output_dir = Path(config["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    if verbose:
        print("\nStarting training...")
    start_time = time.time()

    for epoch in range(config["training"]["epochs"]):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        # Check for improvement
        if val_loss < best_val_loss - early_stop_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_loss, config, output_dir)
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

    # Save training log
    log_path = output_dir / "training_log.json"
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "training_time_seconds": training_time,
        "epochs_completed": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "history": history,
        "test_metrics": {
            "mse_mean": test_results["mse_mean"],
            "spearman_mean": test_results["spearman_mean"],
            "accuracy_mean": test_results["accuracy_mean"],
            "mse_per_dim": test_results["mse_per_dim"],
            "spearman_per_dim": test_results["spearman_per_dim"],
            "accuracy_per_dim": test_results["accuracy_per_dim"],
        },
        "config": config,
    }
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    if verbose:
        print(f"\nCheckpoint saved to: {output_dir / 'best_model.pt'}")
        print(f"Training log saved to: {log_path}")

    return {
        "history": history,
        "test_results": test_results,
        "best_val_loss": best_val_loss,
        "training_time": training_time,
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
    if args.encoder_model is not None:
        config["encoder"]["model_name"] = args.encoder_model
    if args.hidden_dim is not None:
        config["model"]["hidden_dim"] = args.hidden_dim
    if args.seed is not None:
        config["data"]["seed"] = args.seed

    # Set random seeds
    np.random.seed(config["data"]["seed"])
    torch.manual_seed(config["data"]["seed"])

    # Train
    results = train(config, verbose=not args.quiet)

    print(f"\nFinal test MSE: {results['test_results']['mse_mean']:.4f}")
    print(f"Final test Spearman: {results['test_results']['spearman_mean']:.4f}")
    print(f"Final test Accuracy: {results['test_results']['accuracy_mean']:.2%}")


if __name__ == "__main__":
    main()
