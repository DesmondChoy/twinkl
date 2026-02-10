"""CriticMLPSoftOrdinal model for VIF alignment prediction.

Uses soft ordinal labels with KL divergence loss. Instead of hard labels
{-1, 0, 1}, converts targets to probability distributions with ordinal
smoothing, then trains with KL divergence between predicted softmax and
soft target distributions.

Ordinal smoothing spreads probability mass to adjacent classes based on
ordinal distance, so the model learns that -1→0 is a smaller mistake
than -1→+1.

Architecture:
- Same backbone as CriticMLP (Linear → LayerNorm → GELU → Dropout × 2)
- Output layer: Linear(hidden_dim, 10 * 3) = 30 logits (3 classes per dimension)
- Softmax over 3 classes per dimension for prediction

Training: KL divergence between softmax(logits) and soft ordinal targets
Prediction: argmax of softmax → map {0,1,2} → {-1,0,1}

Usage:
    from src.vif.critic_soft_ordinal import CriticMLPSoftOrdinal, soft_ordinal_loss_multi

    model = CriticMLPSoftOrdinal(input_dim=1174, hidden_dim=256)

    # Training
    logits = model(batch_x)
    loss = soft_ordinal_loss_multi(logits, batch_y, smoothing=0.15)

    # Prediction (returns -1, 0, 1)
    pred = model.predict(batch_x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Number of ordinal classes per dimension: -1, 0, +1 → K=3
NUM_CLASSES = 3
NUM_DIMS = 10
OUTPUT_LOGITS = NUM_DIMS * NUM_CLASSES  # 30


def make_soft_ordinal_targets(
    y: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    smoothing: float = 0.15,
) -> torch.Tensor:
    """Convert hard labels to soft ordinal distributions.

    Spreads probability mass to adjacent classes proportional to ordinal
    proximity. Closer classes get more mass than distant ones.

    For smoothing=0.15 and 3 classes:
        class 0 (-1) → [0.85, 0.15, 0.00]  (normalized)
        class 1 ( 0) → [0.15, 0.85, 0.15]  (normalized)
        class 2 (+1) → [0.00, 0.15, 0.85]  (normalized)

    Args:
        y: (batch, 10) with values in {-1, 0, 1}
        num_classes: number of ordinal classes
        smoothing: probability mass to spread per ordinal step

    Returns:
        (batch, 10, 3) soft probability distributions
    """
    # Map {-1, 0, 1} → {0, 1, 2}
    classes = (y.long() + 1).clamp(0, num_classes - 1)  # (batch, 10)
    batch, dims = classes.shape

    # Start with one-hot
    soft = F.one_hot(classes, num_classes).float()  # (batch, 10, 3)

    # Spread mass to neighbors based on ordinal distance
    for step in range(1, num_classes):
        weight = smoothing ** step
        # Shift right (spread to higher classes)
        soft[:, :, step:] += weight * F.one_hot(classes, num_classes).float()[:, :, :-step]
        # Shift left (spread to lower classes)
        soft[:, :, :-step] += weight * F.one_hot(classes, num_classes).float()[:, :, step:]

    # Normalize to valid probability distribution
    soft = soft / soft.sum(dim=-1, keepdim=True)

    return soft


def soft_ordinal_loss_multi(
    logits: torch.Tensor,
    y: torch.Tensor,
    smoothing: float = 0.15,
) -> torch.Tensor:
    """KL divergence loss with soft ordinal targets.

    logits: (batch, 30) raw logits
    y: (batch, 10) with values in {-1, 0, 1}
    """
    batch = logits.size(0)
    logits_3d = logits.view(batch, NUM_DIMS, NUM_CLASSES)  # (batch, 10, 3)

    # Soft targets
    soft_targets = make_soft_ordinal_targets(y, smoothing=smoothing)  # (batch, 10, 3)

    # KL divergence: sum over classes, mean over batch and dimensions
    log_probs = F.log_softmax(logits_3d, dim=-1)
    kl = F.kl_div(log_probs, soft_targets, reduction="batchmean")

    return kl


class CriticMLPSoftOrdinal(nn.Module):
    """MLP critic with soft ordinal classification output.

    Output layer has 30 neurons (10 dims × 3 classes).
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

        # 3 logits per dimension, 10 dimensions → 30
        self.fc_out = nn.Linear(hidden_dim, OUTPUT_LOGITS)

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (batch, 30)."""
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
        logits = self.forward(x)  # (batch, 30)
        batch = logits.size(0)
        logits_3d = logits.view(batch, NUM_DIMS, NUM_CLASSES)  # (batch, 10, 3)

        # Argmax over 3 classes → {0, 1, 2}, then map to {-1, 0, 1}
        classes = logits_3d.argmax(dim=-1)  # (batch, 10)
        return classes.float() - 1.0
