"""CriticBNN model for VIF alignment prediction.

This module implements a Bayesian Neural Network (BNN) critic that predicts
per-dimension alignment scores from state vectors, with principled uncertainty
estimation via variational inference.

Uses Intel Labs bayesian-torch library:
https://github.com/IntelLabs/bayesian-torch

Architecture:
- Input: State vector (text embeddings + time gaps + history + profile)
- Hidden: 2 layers with Bayesian Linear (Reparameterization) + LayerNorm + GELU
- Output: 10-dim alignment predictions in [-1, 1] via Bayesian Linear + Tanh

Uncertainty is estimated by running multiple forward passes; each pass samples
from the learned weight posterior. Mean and std of samples give prediction and
uncertainty.

Training requires adding KL divergence to the loss:
    mse_loss = F.mse_loss(predictions, targets)
    kl_loss = get_kl_loss(model)
    loss = mse_loss + kl_loss / batch_size

Usage:
    from src.vif.critic_bnn import CriticBNN, get_kl_loss

    model = CriticBNN(input_dim=1174, hidden_dim=256)

    # Training (include KL in loss)
    pred = model(batch_x)
    mse = F.mse_loss(pred, batch_y)
    loss = mse + get_kl_loss(model) / batch_size

    # Standard prediction
    predictions = model(state_batch)

    # Prediction with uncertainty
    mean, std = model.predict_with_uncertainty(state_batch, n_samples=50)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bayesian_torch.layers import LinearReparameterization
from bayesian_torch.models.dnn_to_bnn import get_kl_loss as _get_kl_loss


def get_kl_loss(model: nn.Module) -> torch.Tensor:
    """Compute KL divergence between prior and posterior for all BNN layers.

    Use this during training to add the variational inference regularizer:
        loss = mse_loss + get_kl_loss(model) / batch_size

    Args:
        model: CriticBNN or any model containing bayesian_torch layers

    Returns:
        Scalar tensor with total KL loss
    """
    return _get_kl_loss(model)


class CriticBNN(nn.Module):
    """Bayesian Neural Network critic for predicting Schwartz value alignment scores.

    Replaces deterministic Linear layers with Bayesian (variational) layers.
    Weights are represented as distributions; forward pass samples from the
    posterior. Uncertainty estimates come from variance across MC samples.

    Architecture:
        Input → BayesianLinear → LayerNorm → GELU
              → BayesianLinear → LayerNorm → GELU
              → BayesianLinear → Tanh → Output

    Prior: N(0, 1) on weights. Posterior: learned mean and log-scale (reparameterized).

    Example:
        model = CriticBNN(input_dim=1174)

        # Training
        pred = model(batch)
        loss = F.mse_loss(pred, targets) + get_kl_loss(model) / batch_size

        # Inference with uncertainty
        model.eval()
        mean, std = model.predict_with_uncertainty(batch)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 10,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        posterior_rho_init: float = -3.0,
    ):
        """Initialize the CriticBNN.

        Args:
            input_dim: Dimension of input state vector
            hidden_dim: Dimension of hidden layers (default: 256)
            output_dim: Number of output dimensions (default: 10 for Schwartz values)
            prior_mean: Mean of weight prior (default: 0.0)
            prior_variance: Variance of weight prior (default: 1.0)
            posterior_rho_init: Initial log-scale for posterior (default: -3.0)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_rho_init = posterior_rho_init

        bnn_kw = {
            "prior_mean": prior_mean,
            "prior_variance": prior_variance,
            "posterior_mu_init": 0.0,
            "posterior_rho_init": posterior_rho_init,
            "bias": True,
        }

        # Bayesian hidden layers
        self.fc1 = LinearReparameterization(input_dim, hidden_dim, **bnn_kw)
        self.ln1 = nn.LayerNorm(hidden_dim)

        self.fc2 = LinearReparameterization(hidden_dim, hidden_dim, **bnn_kw)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Bayesian output layer
        self.fc_out = LinearReparameterization(hidden_dim, output_dim, **bnn_kw)

        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (single sample from weight posterior).

        Bayesian layers return (output, kl); we use the output and ignore kl here.
        KL is accumulated via get_kl_loss(model) during training.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim) with values in [-1, 1]
        """
        x, _ = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu(x)

        x, _ = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu(x)

        x, _ = self.fc_out(x)
        x = self.tanh(x)

        return x

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with MC sampling over weight posterior.

        Runs multiple forward passes; each samples weights from the variational
        posterior. Mean and std of samples give prediction and uncertainty.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            n_samples: Number of MC samples (default: 50)

        Returns:
            Tuple of (mean, std) where:
            - mean: (batch_size, output_dim) mean predictions
            - std: (batch_size, output_dim) standard deviation (uncertainty)
        """
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                samples.append(pred)

        samples = torch.stack(samples, dim=0)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        return mean, std

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "prior_mean": self.prior_mean,
            "prior_variance": self.prior_variance,
            "posterior_rho_init": self.posterior_rho_init,
        }

    @classmethod
    def from_config(cls, config: dict) -> CriticBNN:
        """Create model from configuration dict."""
        return cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            prior_mean=config.get("prior_mean", 0.0),
            prior_variance=config.get("prior_variance", 1.0),
            posterior_rho_init=config.get("posterior_rho_init", -3.0),
        )
