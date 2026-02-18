"""Training script for VIF CriticBNN (Bayesian Neural Network) model.

Trains the BNN critic with variational inference. Loss = MSE + KL/batch_size.

Usage:
    python -m src.vif.train_bnn
    python -m src.vif.train_bnn --config config/vif.yaml --epochs 50
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

from src.vif.critic_bnn import CriticBNN, get_kl_loss
from src.vif.dataset import create_dataloaders
from src.vif.encoders import create_encoder
from src.vif.eval import evaluate_with_uncertainty, format_results_table
from src.vif.state_encoder import StateEncoder
from src.vif.train import load_config, _deep_update, save_checkpoint


def train_epoch_bnn(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch with MSE + KL loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_size = batch_x.size(0)

        optimizer.zero_grad()
        predictions = model(batch_x)
        mse_loss = criterion(predictions, batch_y)
        kl_loss = get_kl_loss(model)
        loss = mse_loss + kl_loss / batch_size
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
    """Validate with MSE only (no KL during validation)."""
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


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[CriticBNN, dict]:
    """Load CriticBNN from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = CriticBNN.from_config(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    return model, checkpoint


def train(config: dict, verbose: bool = True) -> dict:
    """Main training function for CriticBNN."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")

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

    bnn_cfg = config.get("bnn", {})
    model = CriticBNN(
        input_dim=state_encoder.state_dim,
        hidden_dim=config["model"]["hidden_dim"],
        output_dim=config["model"]["output_dim"],
        prior_mean=bnn_cfg.get("prior_mean", 0.0),
        prior_variance=bnn_cfg.get("prior_variance", 1.0),
        posterior_rho_init=bnn_cfg.get("posterior_rho_init", -3.0),
    )
    model.to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")

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

    history = {"train_loss": [], "val_loss": [], "learning_rate": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stop_patience = config["training"]["early_stopping"]["patience"]
    early_stop_delta = config["training"]["early_stopping"]["min_delta"]

    output_dir = Path(config["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\nStarting BNN training (MSE + KL loss)...")
    start_time = time.time()

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_epoch_bnn(
            model, train_loader, optimizer, criterion, device
        )
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

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

        if epochs_without_improvement >= early_stop_patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {training_time:.1f}s")

    model, _ = load_checkpoint(output_dir / "best_model.pt", device)

    if verbose:
        print("\nEvaluating on test set...")
    n_samples = config.get("mc_dropout", {}).get("n_samples", 50)
    test_results = evaluate_with_uncertainty(
        model, test_loader, n_mc_samples=n_samples, device=device
    )

    if verbose:
        print("\nTest Results:")
        print(format_results_table(test_results))

    log_path = output_dir / "training_log.json"
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "CriticBNN",
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
    parser = argparse.ArgumentParser(
        description="Train VIF CriticBNN (Bayesian Neural Network)"
    )
    parser.add_argument("--config", type=str, default="config/vif.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--encoder-model", type=str)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()
    config = load_config(args.config)

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

    np.random.seed(config["data"]["seed"])
    torch.manual_seed(config["data"]["seed"])

    results = train(config, verbose=not args.quiet)

    print(f"\nFinal test MSE: {results['test_results']['mse_mean']:.4f}")
    print(f"Final test Spearman: {results['test_results']['spearman_mean']:.4f}")
    print(f"Final test Accuracy: {results['test_results']['accuracy_mean']:.2%}")


if __name__ == "__main__":
    main()
