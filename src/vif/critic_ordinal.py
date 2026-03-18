"""Ordinal critic models for VIF alignment prediction.

This module contains the shared base class and all eight ordinal critic
variants: CORAL, CORN, EMD, SoftOrdinal, CDW-CE, Balanced Softmax,
LDAM-DRW, and SLACE. Each predicts per-dimension alignment as ordinal
classes {-1, 0, +1} across 10 Schwartz value dimensions.

Shared architecture (OrdinalCriticBase):
    Input → Linear → LayerNorm → GELU → Dropout
          → Linear → LayerNorm → GELU → Dropout
          → Linear → raw logits (no activation)

Variants differ only in output layer size, loss function, and prediction decoding:
- CORAL: K-1=2 logits/dim, binary CE on cumulative P(Y > k)
- CORN:  K-1=2 logits/dim, conditional P(Y = k | Y >= k)
- EMD:   K=3 logits/dim, squared Earth Mover Distance between CDFs
- SoftOrdinal: K=3 logits/dim, KL divergence with smoothed ordinal targets
- CDW-CE: K=3 logits/dim, class-distance weighted cross-entropy over non-target probs
- Balanced Softmax: K=3 logits/dim, cross-entropy with train-prior logit correction
- LDAM-DRW: K=3 logits/dim, label-distribution-aware margins with deferred re-weighting
- SLACE: K=3 logits/dim, soft-label accumulated cross-entropy

Usage:
    from src.vif.critic_ordinal import CriticMLPCORAL, coral_loss_multi
    from src.vif.critic_ordinal import OrdinalCriticBase  # dispatch via from_config
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coral_pytorch.losses import coral_loss as _coral_loss
from coral_pytorch.losses import corn_loss as _corn_loss
from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.class_balance import compute_effective_number_weights, compute_ldam_margins
from src.vif.eval import CIRCUMPLEX_ADJACENT_PAIRS, CIRCUMPLEX_OPPOSITE_PAIRS

# ─── Shared constants ─────────────────────────────────────────────────────────

# Number of ordinal classes per dimension: -1, 0, +1 → K=3
NUM_CLASSES = 3
# CORAL/CORN logits per dimension: K-1 = 2
LOGITS_PER_DIM = NUM_CLASSES - 1
# Schwartz value dimensions
NUM_DIMS = 10
_DIMENSION_INDEX = {name: idx for idx, name in enumerate(SCHWARTZ_VALUE_ORDER)}
_SLACE_CLASS_POSITIONS = torch.arange(NUM_CLASSES, dtype=torch.float32)
_SLACE_PROX_DOM = (
    (
        torch.abs(_SLACE_CLASS_POSITIONS.view(1, 1, -1) - _SLACE_CLASS_POSITIONS.view(-1, 1, 1))
        <= torch.abs(_SLACE_CLASS_POSITIONS.view(1, -1, 1) - _SLACE_CLASS_POSITIONS.view(-1, 1, 1))
    )
    .float()
)


# ─── OrdinalCriticBase ────────────────────────────────────────────────────────


class OrdinalCriticBase(ABC, nn.Module):
    """Abstract base for ordinal critic MLP variants.

    Provides the identical 2-layer backbone used by CORAL, CORN, EMD,
    SoftOrdinal, CDW-CE, Balanced Softmax, LDAM-DRW, and SLACE models.
    Subclasses only need to specify the output logit count and how to
    reconstruct 3-class probabilities from their raw logits.
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
    def logits_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits shaped as (batch, 10, k)."""
        ...

    @abstractmethod
    def probabilities_from_logits(self, logits_per_dim: torch.Tensor) -> torch.Tensor:
        """Return normalized class probabilities shaped as (batch, 10, 3)."""
        ...

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return normalized 3-class probabilities for each value dimension."""
        logits_per_dim = self.logits_per_dim(x)
        return self.probabilities_from_logits(logits_per_dim)

    def predict_logits_and_probabilities(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return model-native logits and normalized class probabilities."""
        logits_per_dim = self.logits_per_dim(x)
        probabilities = self.probabilities_from_logits(logits_per_dim)
        return logits_per_dim, probabilities

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to alignment scores (batch, 10) in {-1, 0, 1}."""
        probabilities = self.predict_probabilities(x)
        classes = probabilities.argmax(dim=-1)
        return classes.float() - 1.0

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
                "cdw_ce": CriticMLPCDWCE,
                "balanced_softmax": CriticMLPBalancedSoftmax,
                "ldam_drw": CriticMLPLDAMDRW,
                "slace": CriticMLPSLACE,
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

    def logits_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw CORAL logits as (batch, 10, 2)."""
        logits = self.forward(x)  # (batch, 20)
        batch = logits.size(0)
        return logits.view(batch, NUM_DIMS, LOGITS_PER_DIM)  # (batch, 10, 2)

    def probabilities_from_logits(self, logits_per_dim: torch.Tensor) -> torch.Tensor:
        """Reconstruct class probabilities from CORAL cumulative thresholds."""
        cum_probs = torch.sigmoid(logits_per_dim.float())
        # Enforce rank consistency when the independent sigmoid heads disagree.
        lower_threshold = cum_probs[:, :, 0]
        upper_threshold = torch.minimum(lower_threshold, cum_probs[:, :, 1])

        prob_minus1 = 1.0 - lower_threshold
        prob_zero = lower_threshold - upper_threshold
        prob_plus1 = upper_threshold

        probabilities = torch.stack(
            [prob_minus1, prob_zero, prob_plus1],
            dim=-1,
        ).clamp(min=0.0)
        denom = probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probabilities / denom

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
    """Map CORN class indices {0, 1, 2} back to alignment labels {-1, 0, 1}."""
    return labels.to(dtype=torch.float32) - 1.0


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

    def logits_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw CORN logits as (batch, 10, 2)."""
        logits = self.forward(x)  # (batch, 20)
        batch = logits.size(0)
        return logits.view(batch, NUM_DIMS, LOGITS_PER_DIM)

    def probabilities_from_logits(self, logits_per_dim: torch.Tensor) -> torch.Tensor:
        """Reconstruct class probabilities from CORN conditional thresholds."""
        conditional_probs = torch.sigmoid(logits_per_dim.float())
        first_threshold = conditional_probs[:, :, 0]
        second_threshold = conditional_probs[:, :, 1]

        prob_minus1 = 1.0 - first_threshold
        prob_zero = first_threshold * (1.0 - second_threshold)
        prob_plus1 = first_threshold * second_threshold

        probabilities = torch.stack(
            [prob_minus1, prob_zero, prob_plus1],
            dim=-1,
        )
        denom = probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probabilities / denom

    def _variant_name(self) -> str:
        return "corn"


# ─── EMD ───────────────────────────────────────────────────────────────────────
#
# Squared Earth Mover Distance loss (Hou et al. 2017).
# K=3 logits per dimension, softmax → CDF comparison.

_EMD_OUTPUT_LOGITS = NUM_DIMS * NUM_CLASSES  # 30


def _validate_softmax_loss_inputs(
    logits: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate shared (batch, 30) softmax-head loss inputs."""
    if logits.dim() != 2:
        raise ValueError(
            f"logits must be 2D (batch, {NUM_DIMS * NUM_CLASSES}), got {tuple(logits.shape)}"
        )
    if y.dim() != 2:
        raise ValueError(f"y must be 2D (batch, {NUM_DIMS}), got {tuple(y.shape)}")
    if logits.size(1) != NUM_DIMS * NUM_CLASSES:
        raise ValueError(
            f"logits second dimension must be {NUM_DIMS * NUM_CLASSES}, got {logits.size(1)}"
        )
    if y.size(1) != NUM_DIMS:
        raise ValueError(f"y second dimension must be {NUM_DIMS}, got {y.size(1)}")
    if logits.size(0) != y.size(0):
        raise ValueError(
            f"batch sizes must match between logits and y, got {logits.size(0)} and {y.size(0)}"
        )

    return logits, y


def _prepare_class_stat_tensor(
    class_stat: torch.Tensor | np.ndarray,
    *,
    name: str,
    logits: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Move per-dimension class statistics onto the logits device/dtype."""
    if isinstance(class_stat, torch.Tensor):
        stat = class_stat.detach().to(device=logits.device, dtype=torch.float32)
    else:
        stat = torch.as_tensor(class_stat, device=logits.device, dtype=torch.float32)
    if stat.shape != (NUM_DIMS, NUM_CLASSES):
        raise ValueError(
            f"{name} must have shape ({NUM_DIMS}, {NUM_CLASSES}), got {tuple(stat.shape)}"
        )
    return stat.clamp_min(float(eps))


def _validate_non_negative_loss_weight(value: float, *, name: str) -> float:
    """Return a finite non-negative loss weight for optional regularizers."""
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _validate_dimension_weighting_mode(mode: str) -> str:
    mode = str(mode)
    if mode != "inverse_loss":
        raise ValueError(f"dimension_weighting_mode must be 'inverse_loss', got {mode!r}")
    return mode


def _validate_positive_float(value: float, *, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def validate_dimension_weighting_config(
    *,
    mode: str,
    temperature: float,
    ema_alpha: float,
    warmup_epochs: int,
    eps: float,
    min_weight: float,
    max_weight: float,
) -> dict[str, float | int | str]:
    """Validate EMA-smoothed per-dimension weighting config."""
    resolved_mode = _validate_dimension_weighting_mode(mode)
    resolved_temperature = _validate_positive_float(
        temperature,
        name="dimension_weighting_temperature",
    )
    resolved_ema_alpha = float(ema_alpha)
    if not 0.0 < resolved_ema_alpha <= 1.0:
        raise ValueError(
            f"dimension_weighting_ema_alpha must be in (0, 1], got {resolved_ema_alpha}"
        )
    resolved_warmup_epochs = int(warmup_epochs)
    if resolved_warmup_epochs < 0:
        raise ValueError(
            f"dimension_weighting_warmup_epochs must be >= 0, got {resolved_warmup_epochs}"
        )
    resolved_eps = _validate_positive_float(
        eps,
        name="dimension_weighting_eps",
    )
    resolved_min_weight = _validate_positive_float(
        min_weight,
        name="dimension_weighting_min",
    )
    resolved_max_weight = _validate_positive_float(
        max_weight,
        name="dimension_weighting_max",
    )
    if resolved_min_weight > resolved_max_weight:
        raise ValueError(
            "dimension_weighting_min must be <= dimension_weighting_max, "
            f"got {resolved_min_weight} > {resolved_max_weight}"
        )
    return {
        "mode": resolved_mode,
        "temperature": resolved_temperature,
        "ema_alpha": resolved_ema_alpha,
        "warmup_epochs": resolved_warmup_epochs,
        "eps": resolved_eps,
        "min_weight": resolved_min_weight,
        "max_weight": resolved_max_weight,
    }


def _prepare_dimension_weights(
    dimension_weights: torch.Tensor | np.ndarray,
    *,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Move per-dimension weights onto the logits device/dtype."""
    if isinstance(dimension_weights, torch.Tensor):
        weights = dimension_weights.detach().to(device=logits.device, dtype=torch.float32)
    else:
        weights = torch.as_tensor(dimension_weights, device=logits.device, dtype=torch.float32)
    if weights.shape != (NUM_DIMS,):
        raise ValueError(f"dimension_weights must have shape ({NUM_DIMS},), got {tuple(weights.shape)}")
    if not torch.isfinite(weights).all():
        raise ValueError("dimension_weights must be finite")
    if torch.any(weights <= 0):
        raise ValueError("dimension_weights must be strictly positive")
    return weights


def _mean_pair_probability(
    probabilities: torch.Tensor,
    pairs: tuple[tuple[str, str], ...],
    *,
    class_index_pairs: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Average pairwise class co-activation probabilities across circumplex pairs."""
    pair_scores = []
    for left_name, right_name in pairs:
        left_idx = _DIMENSION_INDEX[left_name]
        right_idx = _DIMENSION_INDEX[right_name]
        score = probabilities.new_zeros(())
        for left_class_idx, right_class_idx in class_index_pairs:
            score = score + (
                probabilities[:, left_idx, left_class_idx]
                * probabilities[:, right_idx, right_class_idx]
            ).mean()
        pair_scores.append(score)

    if not pair_scores:
        return probabilities.new_zeros(())
    return torch.stack(pair_scores).mean()


def _circumplex_probability_regularizer(
    probabilities: torch.Tensor,
    *,
    opposite_weight: float,
    adjacent_weight: float,
) -> torch.Tensor:
    """Return the soft circumplex regularizer from model probabilities."""
    if opposite_weight == 0.0 and adjacent_weight == 0.0:
        return probabilities.new_zeros(())

    opposite_term = _mean_pair_probability(
        probabilities,
        CIRCUMPLEX_OPPOSITE_PAIRS,
        class_index_pairs=((0, 0), (2, 2)),
    )
    adjacent_term = _mean_pair_probability(
        probabilities,
        CIRCUMPLEX_ADJACENT_PAIRS,
        class_index_pairs=((2, 2),),
    )
    return (opposite_weight * opposite_term) - (adjacent_weight * adjacent_term)


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
    logits, y = _validate_softmax_loss_inputs(logits, y)
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

    def logits_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw EMD logits as (batch, 10, 3)."""
        logits = self.forward(x)  # (batch, 30)
        batch = logits.size(0)
        return logits.view(batch, NUM_DIMS, NUM_CLASSES)

    def probabilities_from_logits(self, logits_per_dim: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits_per_dim.float(), dim=-1)

    def _variant_name(self) -> str:
        return "emd"


# ─── CDW-CE ────────────────────────────────────────────────────────────────────
#
# Class Distance Weighted Cross Entropy (Polat et al. 2025).
# K=3 logits per dimension, softmax probabilities weighted by |i-c|^alpha.

_CDW_CE_OUTPUT_LOGITS = NUM_DIMS * NUM_CLASSES  # 30


def cdw_ce_loss_multi(
    logits: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 2.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """CDW-CE loss for multi-output ordinal classification.

    Implements the non-margin CDW-CE objective:
        L = -sum_i log(1 - p_i) * |i - c|^alpha
    where p_i are softmax probabilities and c is the target class index.

    logits: (batch, 30) raw logits
    y: (batch, 10) with values in {-1, 0, 1}
    """
    logits, y = _validate_softmax_loss_inputs(logits, y)
    batch = logits.size(0)
    logits_3d = logits.view(batch, NUM_DIMS, NUM_CLASSES)  # (batch, 10, 3)

    # Compute probabilities in fp32 for numerical stability when logits are fp16/bf16.
    probs = F.softmax(logits_3d.float(), dim=-1)  # (batch, 10, 3)
    log_one_minus_p = torch.log1p(-probs.clamp(max=1.0 - eps))  # (batch, 10, 3)

    # Map {-1, 0, 1} → {0, 1, 2}
    classes = (y.long() + 1).clamp(0, NUM_CLASSES - 1)  # (batch, 10)
    class_idx = torch.arange(NUM_CLASSES, device=logits.device, dtype=probs.dtype).view(1, 1, -1)
    distance = (class_idx - classes.unsqueeze(-1).to(probs.dtype)).abs()  # (batch, 10, 3)
    weights = distance.pow(alpha)  # (batch, 10, 3)

    loss = -(weights * log_one_minus_p).sum(dim=-1)  # (batch, 10)
    return loss.mean()


class CriticMLPCDWCE(OrdinalCriticBase):
    """MLP critic with CDW-CE output for ordinal alignment prediction.

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
            output_logits=_CDW_CE_OUTPUT_LOGITS,
        )

    def logits_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw CDW-CE logits as (batch, 10, 3)."""
        logits = self.forward(x)  # (batch, 30)
        batch = logits.size(0)
        return logits.view(batch, NUM_DIMS, NUM_CLASSES)

    def probabilities_from_logits(self, logits_per_dim: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits_per_dim.float(), dim=-1)

    def _variant_name(self) -> str:
        return "cdw_ce"


# ─── Balanced Softmax ─────────────────────────────────────────────────────────
#
# Cross-entropy with train-prior correction in the softmax denominator.


def balanced_softmax_ce_per_dimension(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    class_priors: torch.Tensor | np.ndarray,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return per-dimension BalancedSoftmax CE means with shape ``(10,)``."""
    logits, y = _validate_softmax_loss_inputs(logits, y)
    priors = _prepare_class_stat_tensor(
        class_priors,
        name="class_priors",
        logits=logits,
        eps=eps,
    )

    batch = logits.size(0)
    logits_3d = logits.view(batch, NUM_DIMS, NUM_CLASSES).float()
    adjusted_logits = logits_3d + torch.log(priors).view(1, NUM_DIMS, NUM_CLASSES)
    classes = (y.long() + 1).clamp(0, NUM_CLASSES - 1)
    per_example_loss = F.cross_entropy(
        adjusted_logits.view(batch * NUM_DIMS, NUM_CLASSES),
        classes.view(batch * NUM_DIMS),
        reduction="none",
    )
    return per_example_loss.view(batch, NUM_DIMS).mean(dim=0)


def compute_inverse_loss_dimension_weights(
    loss_ema: torch.Tensor | np.ndarray,
    *,
    mode: str = "inverse_loss",
    temperature: float = 0.5,
    eps: float = 1e-6,
    min_weight: float = 0.5,
    max_weight: float = 1.5,
) -> torch.Tensor:
    """Convert a per-dimension EMA loss vector into clipped inverse-loss weights."""
    _validate_dimension_weighting_mode(mode)
    resolved_temperature = _validate_positive_float(
        temperature,
        name="dimension_weighting_temperature",
    )
    resolved_eps = _validate_positive_float(
        eps,
        name="dimension_weighting_eps",
    )
    resolved_min_weight = _validate_positive_float(
        min_weight,
        name="dimension_weighting_min",
    )
    resolved_max_weight = _validate_positive_float(
        max_weight,
        name="dimension_weighting_max",
    )
    if resolved_min_weight > resolved_max_weight:
        raise ValueError(
            "dimension_weighting_min must be <= dimension_weighting_max, "
            f"got {resolved_min_weight} > {resolved_max_weight}"
        )

    if isinstance(loss_ema, torch.Tensor):
        ema = loss_ema.detach().to(dtype=torch.float32)
    else:
        ema = torch.as_tensor(loss_ema, dtype=torch.float32)
    if ema.shape != (NUM_DIMS,):
        raise ValueError(f"loss_ema must have shape ({NUM_DIMS},), got {tuple(ema.shape)}")
    if not torch.isfinite(ema).all():
        raise ValueError("loss_ema must be finite")
    if torch.any(ema < 0):
        raise ValueError("loss_ema must be non-negative")

    raw_weights = (ema + resolved_eps).pow(-resolved_temperature)
    normalized_weights = raw_weights / raw_weights.mean()
    return normalized_weights.clamp(min=resolved_min_weight, max=resolved_max_weight)


def balanced_softmax_loss_multi(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    class_priors: torch.Tensor | np.ndarray,
    dimension_weights: torch.Tensor | np.ndarray | None = None,
    circumplex_regularizer_opposite_weight: float = 0.0,
    circumplex_regularizer_adjacent_weight: float = 0.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Balanced Softmax loss for multi-output ordinal classification.

    Optionally adds a soft circumplex prior in probability space to discourage
    same-sign activation on opposing value pairs and reward positive
    co-activation on adjacent pairs.
    """
    opposite_weight = _validate_non_negative_loss_weight(
        circumplex_regularizer_opposite_weight,
        name="circumplex_regularizer_opposite_weight",
    )
    adjacent_weight = _validate_non_negative_loss_weight(
        circumplex_regularizer_adjacent_weight,
        name="circumplex_regularizer_adjacent_weight",
    )
    logits, y = _validate_softmax_loss_inputs(logits, y)
    ce_per_dim = balanced_softmax_ce_per_dimension(
        logits,
        y,
        class_priors=class_priors,
        eps=eps,
    )
    if dimension_weights is None:
        ce_loss = ce_per_dim.mean()
    else:
        weights = _prepare_dimension_weights(
            dimension_weights,
            logits=logits,
        )
        ce_loss = (ce_per_dim * weights).mean()
    if opposite_weight == 0.0 and adjacent_weight == 0.0:
        return ce_loss

    batch = logits.size(0)
    probabilities = F.softmax(logits.view(batch, NUM_DIMS, NUM_CLASSES).float(), dim=-1)
    circumplex_regularizer = _circumplex_probability_regularizer(
        probabilities,
        opposite_weight=opposite_weight,
        adjacent_weight=adjacent_weight,
    )
    return ce_loss + circumplex_regularizer


class CriticMLPBalancedSoftmax(OrdinalCriticBase):
    """MLP critic with a 3-class softmax head for Balanced Softmax training."""

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

    def logits_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        batch = logits.size(0)
        return logits.view(batch, NUM_DIMS, NUM_CLASSES)

    def probabilities_from_logits(self, logits_per_dim: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits_per_dim.float(), dim=-1)

    def _variant_name(self) -> str:
        return "balanced_softmax"


# ─── LDAM-DRW ────────────────────────────────────────────────────────────────
#
# Label-distribution-aware margins with deferred re-weighting.


def ldam_drw_loss_multi(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    class_counts: torch.Tensor | np.ndarray,
    epoch: int,
    drw_start_epoch: int = 50,
    max_m: float = 0.5,
    scale: float = 30.0,
    beta: float = 0.9999,
    eps: float = 1e-12,
) -> torch.Tensor:
    """LDAM loss with deferred re-weighting for multi-output ordinal classification."""
    logits, y = _validate_softmax_loss_inputs(logits, y)
    counts = _prepare_class_stat_tensor(
        class_counts,
        name="class_counts",
        logits=logits,
        eps=eps,
    )

    margins = torch.as_tensor(
        compute_ldam_margins(counts.detach().cpu().numpy(), max_m=max_m),
        device=logits.device,
        dtype=torch.float32,
    )
    weights = torch.as_tensor(
        compute_effective_number_weights(counts.detach().cpu().numpy(), beta=beta),
        device=logits.device,
        dtype=torch.float32,
    )

    batch = logits.size(0)
    logits_3d = logits.view(batch, NUM_DIMS, NUM_CLASSES).float()
    classes = (y.long() + 1).clamp(0, NUM_CLASSES - 1)

    target_margins = margins.gather(dim=1, index=classes.transpose(0, 1)).transpose(0, 1)
    adjusted_logits = logits_3d.clone()
    adjusted_logits.scatter_add_(
        dim=2,
        index=classes.unsqueeze(-1),
        src=-target_margins.unsqueeze(-1),
    )
    adjusted_logits = adjusted_logits * float(scale)

    class_weight = None
    if int(epoch) >= int(drw_start_epoch):
        class_weight = weights.gather(dim=1, index=classes.transpose(0, 1)).transpose(0, 1)
        class_weight = class_weight.reshape(batch * NUM_DIMS)

    return F.cross_entropy(
        adjusted_logits.view(batch * NUM_DIMS, NUM_CLASSES),
        classes.view(batch * NUM_DIMS),
        weight=None,
        reduction="none",
    ).mul(class_weight if class_weight is not None else 1.0).mean()


class CriticMLPLDAMDRW(OrdinalCriticBase):
    """MLP critic with a 3-class softmax head for LDAM-DRW training."""

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

    def logits_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        batch = logits.size(0)
        return logits.view(batch, NUM_DIMS, NUM_CLASSES)

    def probabilities_from_logits(self, logits_per_dim: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits_per_dim.float(), dim=-1)

    def _variant_name(self) -> str:
        return "ldam_drw"


# ─── SLACE ───────────────────────────────────────────────────────────────────
#
# Soft Labels Accumulating Cross Entropy (Nachmani et al. 2025).

_SLACE_OUTPUT_LOGITS = NUM_DIMS * NUM_CLASSES  # 30


def make_slace_targets(
    y: torch.Tensor,
    *,
    alpha: float = 1.0,
    num_classes: int = NUM_CLASSES,
) -> torch.Tensor:
    """Convert hard labels to the soft ordinal targets used by SLACE."""
    resolved_alpha = _validate_positive_float(alpha, name="slace_alpha")
    classes = (y.long() + 1).clamp(0, num_classes - 1)
    class_positions = torch.arange(
        num_classes,
        device=y.device,
        dtype=torch.float32,
    ).view(1, 1, -1)
    distances = (
        class_positions
        - classes.unsqueeze(-1).to(torch.float32)
    ).abs()
    return F.softmax(-resolved_alpha * distances, dim=-1)


def slace_loss_multi(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 1.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """SLACE loss for multi-output ordinal classification."""
    logits, y = _validate_softmax_loss_inputs(logits, y)
    resolved_eps = _validate_positive_float(eps, name="slace_eps")

    batch = logits.size(0)
    logits_flat = logits.view(batch * NUM_DIMS, NUM_CLASSES).float()
    probabilities = F.softmax(logits_flat, dim=-1)
    classes = (y.long() + 1).clamp(0, NUM_CLASSES - 1).view(batch * NUM_DIMS)
    soft_targets = make_slace_targets(
        y,
        alpha=alpha,
        num_classes=NUM_CLASSES,
    ).view(batch * NUM_DIMS, NUM_CLASSES)

    prox_dom = _SLACE_PROX_DOM.to(device=probabilities.device, dtype=probabilities.dtype)
    accumulating_softmax = torch.matmul(
        prox_dom[classes],
        probabilities.unsqueeze(-1),
    ).squeeze(-1)

    per_sample_loss = -torch.sum(
        soft_targets * torch.log(accumulating_softmax.clamp_min(resolved_eps)),
        dim=-1,
    )
    return per_sample_loss.mean()


class CriticMLPSLACE(OrdinalCriticBase):
    """MLP critic with a 3-class softmax head for SLACE training."""

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
            output_logits=_SLACE_OUTPUT_LOGITS,
        )

    def logits_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        batch = logits.size(0)
        return logits.view(batch, NUM_DIMS, NUM_CLASSES)

    def probabilities_from_logits(self, logits_per_dim: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits_per_dim.float(), dim=-1)

    def _variant_name(self) -> str:
        return "slace"


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

    def logits_per_dim(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw SoftOrdinal logits as (batch, 10, 3)."""
        logits = self.forward(x)  # (batch, 30)
        batch = logits.size(0)
        return logits.view(batch, NUM_DIMS, NUM_CLASSES)

    def probabilities_from_logits(self, logits_per_dim: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits_per_dim.float(), dim=-1)

    def _variant_name(self) -> str:
        return "soft_ordinal"
