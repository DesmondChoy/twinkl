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
from src.vif.runtime import (
    aggregate_timeline_by_week,
    load_runtime_bundle,
    persist_runtime_artifacts,
    predict_persona_timeline,
)
from src.vif.evolution import (
    DimensionEvolutionSignal,
    EvolutionDetectionResult,
    ProfileUpdateSuggestion,
    classify_weekly_evolution,
)
from src.vif.drift import detect_weekly_drift
from src.vif.critic_ordinal import (
    OrdinalCriticBase,
    CriticMLPCORAL,
    coral_loss_multi,
    compute_coral_importance_weights,
    coral_loss_multi_weighted,
    CriticMLPCORN,
    corn_loss_multi,
    CriticMLPEMD,
    emd_loss_multi,
    CriticMLPCDWCE,
    cdw_ce_loss_multi,
    CriticMLPBalancedSoftmax,
    balanced_softmax_loss_multi,
    CriticMLPTwoStageBalancedSoftmax,
    two_stage_balanced_softmax_loss_multi,
    CriticMLPLDAMDRW,
    ldam_drw_loss_multi,
    CriticMLPSLACE,
    make_slace_targets,
    slace_loss_multi,
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
    "load_runtime_bundle",
    "predict_persona_timeline",
    "aggregate_timeline_by_week",
    "persist_runtime_artifacts",
    "DimensionEvolutionSignal",
    "EvolutionDetectionResult",
    "ProfileUpdateSuggestion",
    "classify_weekly_evolution",
    "detect_weekly_drift",
    "get_kl_loss",
    "CriticMLPCORN",
    "corn_loss_multi",
    "CriticMLPCORAL",
    "coral_loss_multi",
    "compute_coral_importance_weights",
    "coral_loss_multi_weighted",
    "CriticMLPSoftOrdinal",
    "soft_ordinal_loss_multi",
    "CriticMLPEMD",
    "emd_loss_multi",
    "CriticMLPCDWCE",
    "cdw_ce_loss_multi",
    "CriticMLPBalancedSoftmax",
    "balanced_softmax_loss_multi",
    "CriticMLPTwoStageBalancedSoftmax",
    "two_stage_balanced_softmax_loss_multi",
    "CriticMLPLDAMDRW",
    "ldam_drw_loss_multi",
    "CriticMLPSLACE",
    "make_slace_targets",
    "slace_loss_multi",
]
