"""VIF (Value Identity Function) Critic training module.

This module implements the MLP-based critic model that predicts per-dimension
alignment scores from journal entries with MC Dropout uncertainty estimation.

Usage:
    from src.vif import CriticMLP, StateEncoder, SBERTEncoder
    from src.vif.dataset import VIFDataset, load_all_data, split_by_persona
    from src.vif.eval import compute_mse_per_dimension, compute_spearman_per_dimension

Example:
    # Create components
    encoder = SBERTEncoder("all-MiniLM-L6-v2")
    state_encoder = StateEncoder(encoder)

    # Load data
    dataset = VIFDataset(state_encoder, split="train")

    # Train model
    model = CriticMLP(input_dim=state_encoder.state_dim)
"""

from src.vif.encoders import SBERTEncoder, TextEncoder, create_encoder
from src.vif.state_encoder import StateEncoder
from src.vif.critic import CriticMLP
from src.vif.critic_bnn import CriticBNN, get_kl_loss
from src.vif.critic_corn import CriticMLPCORN, corn_loss_multi

__all__ = [
    "TextEncoder",
    "SBERTEncoder",
    "create_encoder",
    "StateEncoder",
    "CriticMLP",
    "CriticBNN",
    "get_kl_loss",
    "CriticMLPCORN",
    "corn_loss_multi",
]
