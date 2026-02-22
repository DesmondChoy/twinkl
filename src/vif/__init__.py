"""VIF (Value Identity Function) Critic training module.

This module implements the MLP-based critic model that predicts per-dimension
alignment scores from journal entries with MC Dropout uncertainty estimation.

Usage:
    from src.vif import CriticMLP, StateEncoder, SBERTEncoder
    from src.vif.dataset import VIFDataset, load_all_data, split_by_persona
    from src.vif.eval import compute_mse_per_dimension, compute_spearman_per_dimension

Example:
    # Create components (hyperparameters are preliminary; see config/vif.yaml)
    encoder = create_encoder(config["encoder"])
    state_encoder = StateEncoder(encoder, **config["state_encoder"])

    # Load data
    dataset = VIFDataset(state_encoder, split="train")

    # Train model
    model = CriticMLP(input_dim=state_encoder.state_dim)
"""

from src.vif.encoders import SBERTEncoder, TextEncoder, create_encoder
from src.vif.state_encoder import StateEncoder
from src.vif.critic import CriticMLP
from src.vif.critic_bnn import CriticBNN, get_kl_loss
from src.vif.critic_ordinal import (
    OrdinalCriticBase,
    CriticMLPCORAL,
    coral_loss_multi,
    CriticMLPCORN,
    corn_loss_multi,
    CriticMLPEMD,
    emd_loss_multi,
    CriticMLPSoftOrdinal,
    soft_ordinal_loss_multi,
)

__all__ = [
    "TextEncoder",
    "SBERTEncoder",
    "create_encoder",
    "StateEncoder",
    "CriticMLP",
    "CriticBNN",
    "OrdinalCriticBase",
    "get_kl_loss",
    "CriticMLPCORN",
    "corn_loss_multi",
    "CriticMLPCORAL",
    "coral_loss_multi",
    "CriticMLPSoftOrdinal",
    "soft_ordinal_loss_multi",
    "CriticMLPEMD",
    "emd_loss_multi",
]
