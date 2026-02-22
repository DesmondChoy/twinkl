"""Ordinal critic models for VIF alignment prediction.

This module contains the shared base class and all four ordinal critic variants:
CORAL, CORN, EMD, and SoftOrdinal. Each predicts per-dimension alignment as
ordinal classes {-1, 0, +1} across 10 Schwartz value dimensions.

Shared architecture (OrdinalCriticBase):
    Input → Linear → LayerNorm → GELU → Dropout
          → Linear → LayerNorm → GELU → Dropout
          → Linear → raw logits (no activation)

Variants differ only in output layer size, loss function, and prediction decoding:
- CORAL: K-1=2 logits/dim, binary CE on cumulative P(Y > k)
- CORN:  K-1=2 logits/dim, conditional P(Y = k | Y >= k)
- EMD:   K=3 logits/dim, squared Earth Mover Distance between CDFs
- SoftOrdinal: K=3 logits/dim, KL divergence with smoothed ordinal targets

Usage:
    from src.vif.critic_ordinal import CriticMLPCORAL, coral_loss_multi
    from src.vif.critic_ordinal import OrdinalCriticBase  # dispatch via from_config
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coral_pytorch.dataset import corn_label_from_logits
from coral_pytorch.losses import coral_loss as _coral_loss
from coral_pytorch.losses import corn_loss as _corn_loss

# ─── Shared constants ─────────────────────────────────────────────────────────

# Number of ordinal classes per dimension: -1, 0, +1 → K=3
NUM_CLASSES = 3
# CORAL/CORN logits per dimension: K-1 = 2
LOGITS_PER_DIM = NUM_CLASSES - 1
# Schwartz value dimensions
NUM_DIMS = 10


# ─── OrdinalCriticBase ────────────────────────────────────────────────────────


class OrdinalCriticBase(ABC, nn.Module):
    """Abstract base for ordinal critic MLP variants.

    Provides the identical 2-layer backbone used by CORAL, CORN, EMD, and
    SoftOrdinal models. Subclasses only need to specify the output logit
    count and implement predict().
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = NUM_DIMS,
        dropout: float = 0.2,
        output_logits: int = 20,
    ):
        # ABC doesn't need __init__, but nn.Module does
        nn.Module.__init__(self)

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

        # Output layer — size depends on variant
        self.fc_out = nn.Linear(hidden_dim, output_logits)

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shared forward pass: 2-layer MLP → raw logits."""
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        x = self.dropout2(x)

        return self.fc_out(x)

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to alignment scores (batch, 10) in {-1, 0, 1}.

        Each subclass implements its own decoding logic (sigmoid thresholds
        for CORAL, conditional probabilities for CORN, argmax for EMD/SoftOrdinal).
        """
        ...

    @abstractmethod
    def _variant_name(self) -> str:
        """String identifier for this variant (e.g. 'coral', 'corn')."""
        ...

    def enable_dropout(self):
        """Enable dropout layers for MC Dropout inference."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout uncertainty estimation over predict() calls.

        Returns:
            Tuple of (mean, std) each shaped (batch_size, 10)
        """
        # Save original dropout training state so we can restore it after MC sampling
        dropout_states = {
            m: m.training for m in self.modules() if isinstance(m, nn.Dropout)
        }
        self.enable_dropout()

        try:
            samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    pred = self.predict(x)
                    samples.append(pred)

            samples = torch.stack(samples, dim=0)  # (n_samples, batch, 10)
            mean = samples.mean(dim=0)
            std = samples.std(dim=0)

            return mean, std
        finally:
            for m, was_training in dropout_states.items():
                m.training = was_training

    def get_config(self) -> dict:
        """Serialize model configuration."""
        return {
            "variant": self._variant_name(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout_p,
        }

    @classmethod
    def from_config(cls, config: dict) -> "OrdinalCriticBase":
        """Create model from configuration dict.

        When called on OrdinalCriticBase directly, dispatches to the
        correct subclass based on config['variant']. When called on a
        specific subclass, creates that subclass directly.
        """
        if cls is OrdinalCriticBase:
            registry = {
                "coral": CriticMLPCORAL,
                "corn": CriticMLPCORN,
                "emd": CriticMLPEMD,
                "soft_ordinal": CriticMLPSoftOrdinal,
            }
            variant = config["variant"]
            if variant not in registry:
                raise ValueError(f"Unknown variant: {variant}. Expected one of {list(registry.keys())}")
            return registry[variant].from_config(config)

        return cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            dropout=config["dropout"],
        )


# ─── CORAL ─────────────────────────────────────────────────────────────────────
#
# Consistent Rank Logits via cumulative probabilities (coral-pytorch).
# K-1=2 logits per dimension, binary CE on P(Y > k).

# Total output logits for CORAL/CORN: 10 dims × 2 logits
_CORAL_OUTPUT_LOGITS = NUM_DIMS * LOGITS_PER_DIM  # 20


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


def compute_coral_importance_weights(
    class_counts: np.ndarray,
    method: str = "inverse_freq",
) -> torch.Tensor:
    """Compute per-dimension importance weights for CORAL's binary thresholds.

    CORAL's K-1=2 binary thresholds each split ordinal classes into two groups.
    When the minority side of a split is rare, the model learns to ignore it.
    Importance weights scale binary CE at each threshold inversely to the
    minority side's frequency, forcing the model to attend to rare boundaries.

    Args:
        class_counts: (n_dims, 3) array — columns are counts for [-1, 0, +1].
        method: "inverse_freq" (1/count) or "inverse_sqrt" (1/sqrt(count)).

    Returns:
        (n_dims, 2) tensor — importance weights, normalized so mean=1 per dim.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    n_dims = counts.shape[0]
    weights = np.zeros((n_dims, 2), dtype=np.float64)

    for d in range(n_dims):
        # Threshold 0: separates class -1 from {0, +1} → minority is n(-1)
        # Threshold 1: separates {-1, 0} from +1 → minority is n(+1)
        minority = np.array([counts[d, 0], counts[d, 2]])
        minority = np.clip(minority, 1.0, None)  # avoid division by zero

        if method == "inverse_freq":
            raw = 1.0 / minority
        elif method == "inverse_sqrt":
            raw = 1.0 / np.sqrt(minority)
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'inverse_freq' or 'inverse_sqrt'.")

        # Normalize so mean(w[d,:]) = 1.0
        raw *= 2.0 / raw.sum()
        weights[d] = raw

    return torch.tensor(weights, dtype=torch.float32)


def coral_loss_multi(
    logits: torch.Tensor,
    y: torch.Tensor,
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
    return _coral_loss(logits_flat, levels_flat)


def coral_loss_multi_weighted(
    logits: torch.Tensor,
    y: torch.Tensor,
    importance_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted CORAL loss with per-dimension importance weights.

    Unlike coral_loss_multi which flattens all dimensions into one call,
    this loops over dimensions so each gets its own importance_weights pair.
    This is necessary because class distributions vary dramatically across
    dimensions (e.g., stimulation 3.5% class -1 vs self_direction 14%).

    Args:
        logits: (batch, 20) raw logits from CriticMLPCORAL.
        y: (batch, 10) with values in {-1, 0, 1}.
        importance_weights: (10, 2) per-dimension threshold weights from
            compute_coral_importance_weights().

    Returns:
        Scalar loss, mean over dimensions (each already batch-reduced).

    Raises:
        ValueError: If importance_weights shape is not (NUM_DIMS, LOGITS_PER_DIM).
    """
    if importance_weights.shape != (NUM_DIMS, LOGITS_PER_DIM):
        raise ValueError(
            f"importance_weights shape must be ({NUM_DIMS}, {LOGITS_PER_DIM}), "
            f"got {tuple(importance_weights.shape)}"
        )

    batch = logits.size(0)
    logits_3d = logits.view(batch, NUM_DIMS, LOGITS_PER_DIM)  # (B, 10, 2)
    levels = alignment_to_levels(y)  # (B, 10, 2)

    # Move weights to same device as logits if needed
    if importance_weights.device != logits.device:
        importance_weights = importance_weights.to(logits.device)

    dim_losses = []
    for d in range(NUM_DIMS):
        loss_d = _coral_loss(
            logits_3d[:, d, :],
            levels[:, d, :],
            importance_weights=importance_weights[d],
        )
        dim_losses.append(loss_d)

    return torch.stack(dim_losses).mean()


class CriticMLPCORAL(OrdinalCriticBase):
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
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            output_logits=_CORAL_OUTPUT_LOGITS,
        )

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

    def _variant_name(self) -> str:
        return "coral"


# ─── CORN ──────────────────────────────────────────────────────────────────────
#
# Consistent Rank Logits for Ordinal Regression (coral-pytorch).
# Conditional probabilities P(Y = k | Y >= k), rank-consistent by construction.

_CORN_OUTPUT_LOGITS = NUM_DIMS * LOGITS_PER_DIM  # 20


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


class CriticMLPCORN(OrdinalCriticBase):
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
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            output_logits=_CORN_OUTPUT_LOGITS,
        )

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

    def _variant_name(self) -> str:
        return "corn"


# ─── EMD ───────────────────────────────────────────────────────────────────────
#
# Squared Earth Mover Distance loss (Hou et al. 2017).
# K=3 logits per dimension, softmax → CDF comparison.

_EMD_OUTPUT_LOGITS = NUM_DIMS * NUM_CLASSES  # 30


def emd_loss_multi(
    logits: torch.Tensor,
    y: torch.Tensor,
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
    logits_3d = logits.view(batch, NUM_DIMS, NUM_CLASSES)  # (batch, 10, 3)

    # Predicted probabilities via softmax
    pred_probs = F.softmax(logits_3d, dim=-1)  # (batch, 10, 3)

    # Target one-hot
    classes = (y.long() + 1).clamp(0, NUM_CLASSES - 1)  # (batch, 10)
    target_probs = F.one_hot(classes, NUM_CLASSES).float()  # (batch, 10, 3)

    # CDF: cumulative sum along class dimension
    pred_cdf = pred_probs.cumsum(dim=-1)  # (batch, 10, 3)
    target_cdf = target_probs.cumsum(dim=-1)  # (batch, 10, 3)

    # Squared EMD = mean of squared L2 distance between CDFs
    # Exclude last CDF value (always 1.0) since it adds no information
    emd_sq = ((pred_cdf[:, :, :-1] - target_cdf[:, :, :-1]) ** 2).sum(dim=-1)  # (batch, 10)

    return emd_sq.mean()


class CriticMLPEMD(OrdinalCriticBase):
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
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            output_logits=_EMD_OUTPUT_LOGITS,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted alignment scores (batch, 10) in {-1, 0, 1}."""
        logits = self.forward(x)  # (batch, 30)
        batch = logits.size(0)
        logits_3d = logits.view(batch, NUM_DIMS, NUM_CLASSES)  # (batch, 10, 3)

        # Argmax over 3 classes → {0, 1, 2}, then map to {-1, 0, 1}
        classes = logits_3d.argmax(dim=-1)  # (batch, 10)
        return classes.float() - 1.0

    def _variant_name(self) -> str:
        return "emd"


# ─── SoftOrdinal ──────────────────────────────────────────────────────────────
#
# Soft ordinal labels with KL divergence loss. Converts hard labels to
# probability distributions with ordinal smoothing.

_SOFT_ORDINAL_OUTPUT_LOGITS = NUM_DIMS * NUM_CLASSES  # 30


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
    # batchmean reduction sums over classes and averages over batch, but sums over
    # other dimensions (here, the 10 value dimensions). We divide by NUM_DIMS
    # to get the mean over both batch and dimensions, matching the scale of
    # other losses like EMD and CORAL.
    kl = F.kl_div(log_probs, soft_targets, reduction="batchmean")

    return kl / NUM_DIMS


class CriticMLPSoftOrdinal(OrdinalCriticBase):
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
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout=dropout,
            output_logits=_SOFT_ORDINAL_OUTPUT_LOGITS,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted alignment scores (batch, 10) in {-1, 0, 1}."""
        logits = self.forward(x)  # (batch, 30)
        batch = logits.size(0)
        logits_3d = logits.view(batch, NUM_DIMS, NUM_CLASSES)  # (batch, 10, 3)

        # Argmax over 3 classes → {0, 1, 2}, then map to {-1, 0, 1}
        classes = logits_3d.argmax(dim=-1)  # (batch, 10)
        return classes.float() - 1.0

    def _variant_name(self) -> str:
        return "soft_ordinal"
