"""CriticMLPEMD model for VIF alignment prediction.

Uses Squared Earth Mover Distance loss (Hou et al. 2017) for ordinal
classification. Penalizes predictions proportional to ordinal distance
between predicted and true class distributions.

Uses squared L2 between CDFs (not L1) for gradient quality:
    EMD²(p, q) = sum_k (CDF_p(k) - CDF_q(k))²

L1 EMD gives constant ±1 gradients (via sign function), causing the
optimizer to overshoot. L2 gives gradients proportional to error magnitude.

Architecture:
- Same backbone as CriticMLP (Linear → LayerNorm → GELU → Dropout × 2)
- Output layer: Linear(hidden_dim, 10 * 3) = 30 logits (3 classes per dimension)
- Softmax over 3 classes per dimension

Training: EMD between softmax(logits) and one-hot targets
Prediction: argmax of softmax → map {0,1,2} → {-1,0,1}

Usage:
    from src.vif.critic_emd import CriticMLPEMD, emd_loss_multi

    model = CriticMLPEMD(input_dim=1174, hidden_dim=256)

    # Training
    logits = model(batch_x)
    loss = emd_loss_multi(logits, batch_y)

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


def emd_loss_multi(
    logits: torch.Tensor,
    y: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Squared Earth Mover Distance loss for multi-output ordinal classification.

    Uses squared L2 distance between CDFs (Hou et al. 2017), which provides
    gradients proportional to error magnitude — unlike L1 EMD where the
    gradient is always ±1 regardless of how far off the prediction is.

    EMD² = sum_k (CDF_pred(k) - CDF_target(k))²

    logits: (batch, 30) raw logits
    y: (batch, 10) with values in {-1, 0, 1}
    """
    batch = logits.size(0)
    logits_3d = logits.view(batch, NUM_DIMS, num_classes)  # (batch, 10, 3)

    # Predicted probabilities via softmax
    pred_probs = F.softmax(logits_3d, dim=-1)  # (batch, 10, 3)

    # Target one-hot
    classes = (y.long() + 1).clamp(0, num_classes - 1)  # (batch, 10)
    target_probs = F.one_hot(classes, num_classes).float()  # (batch, 10, 3)

    # CDF: cumulative sum along class dimension
    pred_cdf = pred_probs.cumsum(dim=-1)  # (batch, 10, 3)
    target_cdf = target_probs.cumsum(dim=-1)  # (batch, 10, 3)

    # Squared EMD = mean of squared L2 distance between CDFs
    # Exclude last CDF value (always 1.0) since it adds no information
    emd_sq = ((pred_cdf[:, :, :-1] - target_cdf[:, :, :-1]) ** 2).sum(dim=-1)  # (batch, 10)

    return emd_sq.mean()


class CriticMLPEMD(nn.Module):
    """MLP critic with EMD loss for ordinal alignment prediction.

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
