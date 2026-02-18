"""CriticMLP model for VIF alignment prediction.

This module implements the MLP critic model that predicts per-dimension
alignment scores from state vectors, with MC Dropout for uncertainty estimation.

Architecture:
- Input: State vector (text embeddings + time gaps + history + profile)
- Hidden: 2 layers with LayerNorm + GELU + Dropout
- Output: 10-dim alignment predictions in [-1, 1]

MC Dropout enables uncertainty estimation by keeping dropout active during
inference and running multiple forward passes.

When heteroscedastic=True, the model also predicts per-dimension log-variance,
enabling learned aleatoric uncertainty via Beta-NLL loss.

Usage:
    from src.vif import CriticMLP

    model = CriticMLP(input_dim=state_encoder.state_dim)

    # Standard prediction
    predictions = model(state_batch)

    # Prediction with uncertainty
    mean, std = model.predict_with_uncertainty(state_batch, n_samples=50)

    # Heteroscedastic model (learned aleatoric uncertainty)
    model = CriticMLP(input_dim=state_encoder.state_dim, heteroscedastic=True)
    mean, log_var = model.forward_with_log_var(state_batch)
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

    When heteroscedastic=True, the output layer produces both mean predictions
    and log-variance estimates. The variance head enables the model to learn
    *where* it is uncertain (aleatoric uncertainty), complementing the
    epistemic uncertainty from MC Dropout.

    Architecture:
        Input → Linear → LayerNorm → GELU → Dropout
              → Linear → LayerNorm → GELU → Dropout
              → Linear → Tanh (mean) / unbounded (log_var) → Output

    The Tanh activation constrains mean outputs to [-1, 1], matching the
    alignment score range. Log-variance is left unbounded so the model can
    express arbitrary confidence levels.

    Example:
        model = CriticMLP(input_dim=state_encoder.state_dim)

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
        heteroscedastic: bool = False,
    ):
        """Initialize the CriticMLP.

        Args:
            input_dim: Dimension of input state vector
            hidden_dim: Dimension of hidden layers (default: 256)
            output_dim: Number of output dimensions (default: 10 for Schwartz values)
            dropout: Dropout probability for MC Dropout (default: 0.2)
            heteroscedastic: If True, predict per-dimension log-variance alongside
                mean, enabling learned aleatoric uncertainty (default: False)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout
        self.heteroscedastic = heteroscedastic

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        # Output layer: 2x width when heteroscedastic (mean + log_var)
        out_features = output_dim * 2 if heteroscedastic else output_dim
        self.fc_out = nn.Linear(hidden_dim, out_features)

        # Activation functions
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Shared hidden layers (everything before the output head).

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Hidden representation of shape (batch_size, hidden_dim)
        """
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning mean predictions only.

        For heteroscedastic models, this discards the log-variance portion,
        keeping backward compatibility with all eval code that expects
        (batch_size, output_dim) output.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim) with values in [-1, 1]
        """
        x = self._backbone(x)
        x = self.fc_out(x)

        if self.heteroscedastic:
            return self.tanh(x[:, : self.output_dim])

        return self.tanh(x)

    def forward_with_log_var(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both mean and log-variance.

        Only meaningful for heteroscedastic models. Used during training
        with Beta-NLL loss.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Tuple of (mean, log_var) where:
            - mean: (batch_size, output_dim) with values in [-1, 1] via tanh
            - log_var: (batch_size, output_dim) unbounded log-variance
        """
        x = self._backbone(x)
        x = self.fc_out(x)

        mean = self.tanh(x[:, : self.output_dim])
        log_var = x[:, self.output_dim :]

        return mean, log_var

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

        For heteroscedastic models, combines two uncertainty sources:
        - Aleatoric: learned per-sample variance from the variance head
        - Epistemic: disagreement between MC Dropout samples
        Total uncertainty = sqrt(aleatoric^2 + epistemic^2)

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            n_samples: Number of MC Dropout samples (default: 50)

        Returns:
            Tuple of (mean, std) where:
            - mean: (batch_size, output_dim) mean predictions
            - std: (batch_size, output_dim) total uncertainty
        """
        self.enable_dropout()

        mean_samples = []
        log_var_samples = [] if self.heteroscedastic else None

        with torch.no_grad():
            for _ in range(n_samples):
                if self.heteroscedastic:
                    m, lv = self.forward_with_log_var(x)
                    mean_samples.append(m)
                    log_var_samples.append(lv)
                else:
                    pred = self.forward(x)
                    mean_samples.append(pred)

        # Stack: (n_samples, batch_size, output_dim)
        mean_stack = torch.stack(mean_samples, dim=0)

        # Epistemic uncertainty: std of mean predictions across MC samples
        epistemic = mean_stack.std(dim=0)
        mean = mean_stack.mean(dim=0)

        if self.heteroscedastic:
            # Aleatoric uncertainty: mean of exp(log_var) across MC samples
            log_var_stack = torch.stack(log_var_samples, dim=0)
            aleatoric_var = torch.exp(log_var_stack).mean(dim=0)
            aleatoric = aleatoric_var.sqrt()

            # Total = sqrt(aleatoric^2 + epistemic^2)
            total_std = (aleatoric**2 + epistemic**2).sqrt()
            return mean, total_std

        return mean, epistemic

    def get_config(self) -> dict:
        """Get model configuration for serialization."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout_p,
            "heteroscedastic": self.heteroscedastic,
        }

    @classmethod
    def from_config(cls, config: dict) -> "CriticMLP":
        """Create model from configuration dict."""
        return cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            dropout=config["dropout"],
            heteroscedastic=config.get("heteroscedastic", False),
        )
