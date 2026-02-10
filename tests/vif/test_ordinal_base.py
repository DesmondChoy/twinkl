"""Tests for OrdinalCriticBase and all ordinal model variants.

Parametrized over all 4 models to verify the shared interface contract.
"""

import torch
import pytest

from src.vif.critic_ordinal import (
    CriticMLPCORAL,
    CriticMLPCORN,
    CriticMLPEMD,
    CriticMLPSoftOrdinal,
    OrdinalCriticBase,
)

INPUT_DIM = 100
HIDDEN_DIM = 64
BATCH_SIZE = 4

ALL_MODELS = [CriticMLPCORAL, CriticMLPCORN, CriticMLPEMD, CriticMLPSoftOrdinal]


@pytest.fixture(params=ALL_MODELS, ids=lambda c: c.__name__)
def model(request):
    """Fixture that yields each ordinal model variant."""
    cls = request.param
    return cls(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)


@pytest.fixture
def sample_input():
    """Random input batch."""
    return torch.randn(BATCH_SIZE, INPUT_DIM)


class TestForward:
    """Tests for the shared forward() method."""

    def test_output_shape(self, model, sample_input):
        """Forward should return logits with correct shape."""
        logits = model(sample_input)
        assert logits.dim() == 2
        assert logits.shape[0] == BATCH_SIZE

    def test_output_is_raw_logits(self, model, sample_input):
        """Output should be unbounded raw logits (not sigmoid/softmax)."""
        logits = model(sample_input)
        # Raw logits can exceed [0, 1] range
        assert logits.min().item() < 0 or logits.max().item() > 1


class TestPredict:
    """Tests for the predict() method."""

    def test_output_shape(self, model, sample_input):
        """Predict should return (batch, 10) tensor."""
        pred = model.predict(sample_input)
        assert pred.shape == (BATCH_SIZE, 10)

    def test_values_in_valid_range(self, model, sample_input):
        """Predictions should be in {-1, 0, 1}."""
        pred = model.predict(sample_input)
        valid_values = {-1.0, 0.0, 1.0}
        unique_vals = set(pred.unique().tolist())
        assert unique_vals.issubset(valid_values)


class TestConfigRoundTrip:
    """Tests for get_config()/from_config() serialization."""

    def test_config_contains_variant(self, model):
        """Config should include a variant key."""
        config = model.get_config()
        assert "variant" in config
        assert config["variant"] in {"coral", "corn", "emd", "soft_ordinal"}

    def test_config_round_trip(self, model, sample_input):
        """Model created from config should produce same-shape output."""
        config = model.get_config()
        model2 = type(model).from_config(config)

        logits1 = model(sample_input)
        logits2 = model2(sample_input)
        assert logits1.shape == logits2.shape

    def test_base_class_dispatch(self, model):
        """OrdinalCriticBase.from_config() should dispatch to correct subclass."""
        config = model.get_config()
        model2 = OrdinalCriticBase.from_config(config)
        assert type(model2) == type(model)


class TestStateDictRoundTrip:
    """Tests for checkpoint compatibility."""

    def test_state_dict_keys_match(self, model):
        """State dict keys should match between original and from_config model."""
        config = model.get_config()
        model2 = type(model).from_config(config)
        assert set(model.state_dict().keys()) == set(model2.state_dict().keys())

    def test_load_state_dict(self, model, sample_input):
        """Loading state dict should reproduce identical outputs."""
        config = model.get_config()
        model2 = type(model).from_config(config)
        model2.load_state_dict(model.state_dict())

        model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = model(sample_input)
            out2 = model2(sample_input)
        assert torch.allclose(out1, out2)


class TestEnableDropout:
    """Tests for MC Dropout support."""

    def test_enable_dropout_sets_train_mode(self, model):
        """enable_dropout() should put Dropout layers in train mode."""
        model.eval()
        model.enable_dropout()

        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                assert module.training


class TestPredictWithUncertainty:
    """Tests for MC Dropout uncertainty estimation."""

    def test_output_shapes(self, model, sample_input):
        """Should return mean and std with shape (batch, 10)."""
        mean, std = model.predict_with_uncertainty(sample_input, n_samples=5)
        assert mean.shape == (BATCH_SIZE, 10)
        assert std.shape == (BATCH_SIZE, 10)

    def test_std_non_negative(self, model, sample_input):
        """Standard deviation should be non-negative."""
        _, std = model.predict_with_uncertainty(sample_input, n_samples=5)
        assert (std >= 0).all()

    def test_mean_in_valid_range(self, model, sample_input):
        """Mean predictions should be in [-1, 1] (average of {-1,0,1} values)."""
        mean, _ = model.predict_with_uncertainty(sample_input, n_samples=10)
        assert mean.min().item() >= -1.0
        assert mean.max().item() <= 1.0
