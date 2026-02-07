"""CriticMLPCORN model for VIF alignment prediction.

Uses CORN (Consistent Rank Logits for Ordinal Regression) from coral-pytorch
to predict per-dimension alignment scores as ordinal classes {-1, 0, +1}.

Architecture:
- Same backbone as CriticMLP (Linear → LayerNorm → GELU → Dropout × 2)
- Output layer: Linear(hidden_dim, 10 * 2) = 20 logits (K-1 per dimension for K=3 classes)
- No Tanh; raw logits for CORN loss

Training: corn_loss(logits, labels_0indexed, num_classes=3)
Prediction: corn_label_from_logits → map 0→-1, 1→0, 2→1

Usage:
    from src.vif.critic_corn import CriticMLPCORN

    model = CriticMLPCORN(input_dim=1174, hidden_dim=256)

    # Training
    logits = model(batch_x)
    loss = corn_loss(logits, labels_0indexed, num_classes=3)

    # Prediction (returns -1, 0, 1)
    pred = model.predict(batch_x)
"""

import torch
import torch.nn as nn

from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import corn_loss as _corn_loss


# Number of ordinal classes per dimension: -1, 0, +1 → K=3
NUM_CLASSES = 3
# Logits per dimension: K-1 = 2
LOGITS_PER_DIM = NUM_CLASSES - 1
# Total output dims: 10 Schwartz dimensions × 2 logits
NUM_DIMS = 10
OUTPUT_LOGITS = NUM_DIMS * LOGITS_PER_DIM  # 20


def alignment_to_corn_labels(y: torch.Tensor) -> torch.Tensor:
    """Map alignment labels {-1, 0, 1} to CORN class indices {0, 1, 2}."""
    # y: (batch, 10)
    return (y.long() + 1).clamp(0, NUM_CLASSES - 1)


def corn_labels_to_alignment(labels: torch.Tensor) -> torch.Tensor:
    """Map CORN class indices {0, 1, 2} back to alignment {-1, 0, 1}."""
    return (labels.float() - 1).clamp(-1.0, 1.0)


def corn_loss_multi(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """CORN loss for multi-output ordinal regression.

    logits: (batch, 10 * 2) = (batch, 20)
    y: (batch, 10) with values in {-1, 0, 1}

    Flattens to (batch*10, 2) and (batch*10,) for corn_loss.
    """
    batch = logits.size(0)
    logits_flat = logits.view(batch * NUM_DIMS, LOGITS_PER_DIM)
    y_corn = alignment_to_corn_labels(y)  # (batch, 10)
    y_flat = y_corn.view(batch * NUM_DIMS)
    return _corn_loss(logits_flat, y_flat, num_classes)


class CriticMLPCORN(nn.Module):
    """MLP critic with CORN output for ordinal alignment prediction.

    Output layer has 20 neurons (10 dims × 2 logits) for CORN.
    No Tanh; raw logits for corn_loss.
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

        # CORN: K-1 = 2 neurons per dimension, 10 dimensions → 20
        self.fc_out = nn.Linear(hidden_dim, OUTPUT_LOGITS)

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (batch, 20) for corn_loss."""
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
        """Return predicted alignment scores (batch, 10) in {-1, 0, 1}."""
        logits = self.forward(x)  # (batch, 20)
        batch = logits.size(0)
        logits_per_dim = logits.view(batch, NUM_DIMS, LOGITS_PER_DIM)

        preds = []
        for d in range(NUM_DIMS):
            ld = corn_label_from_logits(logits_per_dim[:, d, :])  # (batch,)
            preds.append(corn_labels_to_alignment(ld))
        return torch.stack(preds, dim=1)
