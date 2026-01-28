"""CriticMLP model for VIF alignment prediction.

This module implements the MLP critic model that predicts per-dimension
alignment scores from state vectors, with MC Dropout for uncertainty estimation.

Architecture:
- Input: State vector (text embeddings + time gaps + history + profile)
- Hidden: 2 layers with LayerNorm + GELU + Dropout
- Output: 10-dim alignment predictions in [-1, 1]

MC Dropout enables uncertainty estimation by keeping dropout active during
inference and running multiple forward passes.

Usage:
    from src.vif import CriticMLP

    model = CriticMLP(input_dim=1174, hidden_dim=256, dropout=0.2)

    # Standard prediction
    predictions = model(state_batch)

    # Prediction with uncertainty
    mean, std = model.predict_with_uncertainty(state_batch, n_samples=50)
"""

import torch
import torch.nn as nn


class CriticMLP(nn.Module):
    """MLP critic for predicting Schwartz value alignment scores.

    This model takes a state vector (combining text embeddings, temporal
    features, history statistics, and user profile) and predicts alignment
    scores for each of the 10 Schwartz value dimensions.

    The model uses MC Dropout for uncertainty estimation: by keeping dropout
    active during inference and running multiple forward passes, we can
    estimate the model's uncertainty about its predictions.

    Architecture:
        Input → Linear → LayerNorm → GELU → Dropout
              → Linear → LayerNorm → GELU → Dropout
              → Linear → Tanh → Output

    The Tanh activation constrains outputs to [-1, 1], matching the
    alignment score range.

    Example:
        model = CriticMLP(input_dim=1174)  # MiniLM embeddings

        # Training
        predictions = model(batch)
        loss = F.mse_loss(predictions, targets)

        # Inference with uncertainty
        model.eval()  # Note: MC dropout still works due to explicit enable_dropout
        mean, std = model.predict_with_uncertainty(batch)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 10,
        dropout: float = 0.2,
    ):
        """Initialize the CriticMLP.

        Args:
            input_dim: Dimension of input state vector
            hidden_dim: Dimension of hidden layers (default: 256)
            output_dim: Number of output dimensions (default: 10 for Schwartz values)
            dropout: Dropout probability for MC Dropout (default: 0.2)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

        # Activation functions
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim) with values in [-1, 1]
        """
        # Layer 1
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        # Output
        x = self.fc_out(x)
        x = self.tanh(x)

        return x

    def enable_dropout(self):
        """Enable dropout layers for MC Dropout inference.

        This is called before running multiple forward passes for
        uncertainty estimation. Even in eval mode, this ensures
        dropout remains active.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with MC Dropout uncertainty estimation.

        Runs multiple forward passes with dropout enabled to estimate
        prediction uncertainty. The mean of samples gives the prediction,
        while the standard deviation indicates uncertainty.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            n_samples: Number of MC Dropout samples (default: 50)

        Returns:
            Tuple of (mean, std) where:
            - mean: (batch_size, output_dim) mean predictions
            - std: (batch_size, output_dim) standard deviation (uncertainty)
        """
        # Enable dropout for MC sampling
        self.enable_dropout()

        # Collect samples
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                samples.append(pred)

        # Stack samples: (n_samples, batch_size, output_dim)
        samples = torch.stack(samples, dim=0)

        # Compute statistics
        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        return mean, std

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout_p,
        }

    @classmethod
    def from_config(cls, config: dict) -> "CriticMLP":
        """Create model from configuration dict."""
        return cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            dropout=config["dropout"],
        )
