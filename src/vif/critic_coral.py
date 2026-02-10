"""CriticMLPCORAL model for VIF alignment prediction.

Uses CORAL (Consistent Rank Logits via cumulative probabilities) from
coral-pytorch to predict per-dimension alignment as ordinal classes {-1, 0, +1}.

Architecture:
- Same backbone as CriticMLP (Linear → LayerNorm → GELU → Dropout × 2)
- Output layer: Linear(hidden_dim, 10 * 2) = 20 logits (K-1 per dimension for K=3)
- No Tanh; raw logits for coral_loss (binary CE on cumulative probabilities)

CORAL vs CORN:
- CORAL: shared weights, per-class biases, binary CE on P(Y > k)
- CORN: conditional probabilities P(Y = k | Y >= k), rank-consistency by construction

Training: coral_loss(logits, levels, importance_weights)
Prediction: sigmoid(logits) → cumulative probs → class labels

Usage:
    from src.vif.critic_coral import CriticMLPCORAL, coral_loss_multi

    model = CriticMLPCORAL(input_dim=1174, hidden_dim=256)

    # Training
    logits = model(batch_x)
    loss = coral_loss_multi(logits, batch_y)

    # Prediction (returns -1, 0, 1)
    pred = model.predict(batch_x)
"""

import torch
import torch.nn as nn

from coral_pytorch.losses import coral_loss as _coral_loss

# Number of ordinal classes per dimension: -1, 0, +1 → K=3
NUM_CLASSES = 3
# Logits per dimension: K-1 = 2
LOGITS_PER_DIM = NUM_CLASSES - 1
# Total output dims: 10 Schwartz dimensions × 2 logits
NUM_DIMS = 10
OUTPUT_LOGITS = NUM_DIMS * LOGITS_PER_DIM  # 20


def alignment_to_levels(y: torch.Tensor) -> torch.Tensor:
    """Map alignment labels {-1, 0, 1} to CORAL levels.

    CORAL expects levels as binary indicators: for class c (0-indexed),
    levels[k] = 1 if c > k, else 0.

    Examples (K=3):
        class 0 (-1) → [0, 0]
        class 1 ( 0) → [1, 0]
        class 2 (+1) → [1, 1]

    Args:
        y: (batch, 10) with values in {-1, 0, 1}

    Returns:
        (batch, 10, 2) binary level indicators
    """
    # Map {-1, 0, 1} → {0, 1, 2}
    classes = (y.long() + 1).clamp(0, NUM_CLASSES - 1)  # (batch, 10)

    batch, dims = classes.shape
    # Create level matrix: levels[b, d, k] = 1 if class[b,d] > k
    thresholds = torch.arange(LOGITS_PER_DIM, device=y.device).unsqueeze(0).unsqueeze(0)
    levels = (classes.unsqueeze(-1) > thresholds).float()

    return levels  # (batch, 10, 2)


def coral_loss_multi(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """CORAL loss for multi-output ordinal regression.

    logits: (batch, 20) raw logits
    y: (batch, 10) with values in {-1, 0, 1}

    Flattens to (batch*10, 2) logits and (batch*10, 2) levels for coral_loss.
    """
    batch = logits.size(0)
    logits_flat = logits.view(batch * NUM_DIMS, LOGITS_PER_DIM)  # (B*10, 2)
    levels = alignment_to_levels(y)  # (batch, 10, 2)
    levels_flat = levels.view(batch * NUM_DIMS, LOGITS_PER_DIM)  # (B*10, 2)
    return _coral_loss(logits_flat, levels_flat, num_classes)


class CriticMLPCORAL(nn.Module):
    """MLP critic with CORAL output for ordinal alignment prediction.

    Output layer has 20 neurons (10 dims × 2 logits) for CORAL.
    No Tanh; raw logits for coral_loss.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = NUM_DIMS,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        # CORAL: K-1 = 2 neurons per dimension, 10 dimensions → 20
        self.fc_out = nn.Linear(hidden_dim, OUTPUT_LOGITS)

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (batch, 20) for coral_loss."""
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        return self.fc_out(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted alignment scores (batch, 10) in {-1, 0, 1}.

        Applies sigmoid to get cumulative probabilities, then converts to
        class labels by counting how many thresholds are exceeded.
        """
        logits = self.forward(x)  # (batch, 20)
        batch = logits.size(0)
        logits_per_dim = logits.view(batch, NUM_DIMS, LOGITS_PER_DIM)  # (batch, 10, 2)

        # sigmoid → cumulative probs P(Y > k)
        cum_probs = torch.sigmoid(logits_per_dim)  # (batch, 10, 2)

        # Class = number of thresholds exceeded (round at 0.5)
        # class 0 if both < 0.5, class 1 if first >= 0.5 but second < 0.5, etc.
        classes = (cum_probs > 0.5).sum(dim=-1)  # (batch, 10) in {0, 1, 2}

        # Map {0, 1, 2} → {-1, 0, 1}
        return classes.float() - 1.0
