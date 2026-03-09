"""Tests for OrdinalCriticBase and all ordinal model variants.

Parametrized over all 7 models to verify the shared interface contract.
"""

import torch
import pytest

from src.vif.critic_ordinal import (
    CriticMLPBalancedSoftmax,
    CriticMLPCORAL,
    CriticMLPCORN,
    CriticMLPCDWCE,
    CriticMLPEMD,
    CriticMLPLDAMDRW,
    CriticMLPSoftOrdinal,
    OrdinalCriticBase,
)

INPUT_DIM = 100
HIDDEN_DIM = 64
BATCH_SIZE = 4

ALL_MODELS = [
    CriticMLPCORAL,
    CriticMLPCORN,
    CriticMLPEMD,
    CriticMLPCDWCE,
    CriticMLPBalancedSoftmax,
    CriticMLPLDAMDRW,
    CriticMLPSoftOrdinal,
]

THRESHOLD_MODELS = [CriticMLPCORAL, CriticMLPCORN]
SOFTMAX_MODELS = [
    CriticMLPEMD,
    CriticMLPCDWCE,
    CriticMLPBalancedSoftmax,
    CriticMLPLDAMDRW,
    CriticMLPSoftOrdinal,
]


def _constant_forward(flat_logits: torch.Tensor):
    def forward(x: torch.Tensor) -> torch.Tensor:
        return flat_logits.unsqueeze(0).repeat(x.size(0), 1)

    return forward


def _threshold_family_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    per_dim_logits = torch.tensor(
        [[-6.0, -6.0], [6.0, -6.0], [6.0, 6.0]] + [[-6.0, -6.0]] * 7,
        dtype=torch.float32,
    )
    expected_classes = torch.tensor([0, 1, 2] + [0] * 7, dtype=torch.long)
    return per_dim_logits.reshape(-1), expected_classes


def _softmax_family_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    per_dim_logits = torch.tensor(
        [[6.0, -2.0, -3.0], [-3.0, 6.0, -2.0], [-3.0, -2.0, 6.0]] + [[6.0, -2.0, -3.0]] * 7,
        dtype=torch.float32,
    )
    expected_classes = torch.tensor([0, 1, 2] + [0] * 7, dtype=torch.long)
    return per_dim_logits.reshape(-1), expected_classes


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

    @pytest.mark.parametrize("model_cls", THRESHOLD_MODELS, ids=lambda c: c.__name__)
    def test_threshold_family_predict_decodes_known_logits(self, model_cls, sample_input):
        """CORAL/CORN decoders should map known logits to all three classes."""
        model = model_cls(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0)
        flat_logits, expected_classes = _threshold_family_fixture()
        model.forward = _constant_forward(flat_logits)

        pred = model.predict(sample_input)
        expected = (expected_classes.float() - 1.0).unsqueeze(0).repeat(BATCH_SIZE, 1)

        assert torch.equal(pred, expected)

    @pytest.mark.parametrize("model_cls", SOFTMAX_MODELS, ids=lambda c: c.__name__)
    def test_softmax_family_predict_decodes_known_logits(self, model_cls, sample_input):
        """3-logit decoders should map known logits to all three classes."""
        model = model_cls(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0)
        flat_logits, expected_classes = _softmax_family_fixture()
        model.forward = _constant_forward(flat_logits)

        pred = model.predict(sample_input)
        expected = (expected_classes.float() - 1.0).unsqueeze(0).repeat(BATCH_SIZE, 1)

        assert torch.equal(pred, expected)


class TestOrdinalOutputs:
    """Tests for standardized logits/probability export helpers."""

    def test_logits_per_dim_shape(self, model, sample_input):
        logits = model.logits_per_dim(sample_input)
        assert logits.shape[0] == BATCH_SIZE
        assert logits.shape[1] == 10
        assert logits.shape[2] in {2, 3}

    def test_predict_probabilities_shape(self, model, sample_input):
        probabilities = model.predict_probabilities(sample_input)
        assert probabilities.shape == (BATCH_SIZE, 10, 3)

    def test_predict_probabilities_sum_to_one(self, model, sample_input):
        probabilities = model.predict_probabilities(sample_input)
        probs_sum = probabilities.sum(dim=-1)
        assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)

    def test_predict_logits_and_probabilities(self, model, sample_input):
        logits, probabilities = model.predict_logits_and_probabilities(sample_input)
        assert logits.shape[0] == BATCH_SIZE
        assert logits.shape[1] == 10
        assert probabilities.shape == (BATCH_SIZE, 10, 3)

    @pytest.mark.parametrize("model_cls", THRESHOLD_MODELS, ids=lambda c: c.__name__)
    def test_threshold_family_probabilities_match_known_classes(self, model_cls, sample_input):
        model = model_cls(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0)
        flat_logits, expected_classes = _threshold_family_fixture()
        model.forward = _constant_forward(flat_logits)

        probabilities = model.predict_probabilities(sample_input)
        expected = expected_classes.unsqueeze(0).repeat(BATCH_SIZE, 1)

        assert torch.equal(probabilities.argmax(dim=-1), expected)

    @pytest.mark.parametrize("model_cls", SOFTMAX_MODELS, ids=lambda c: c.__name__)
    def test_softmax_family_probabilities_match_known_classes(self, model_cls, sample_input):
        model = model_cls(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, dropout=0.0)
        flat_logits, expected_classes = _softmax_family_fixture()
        model.forward = _constant_forward(flat_logits)

        probabilities = model.predict_probabilities(sample_input)
        expected = expected_classes.unsqueeze(0).repeat(BATCH_SIZE, 1)

        assert torch.equal(probabilities.argmax(dim=-1), expected)


class TestConfigRoundTrip:
    """Tests for get_config()/from_config() serialization."""

    def test_config_contains_variant(self, model):
        """Config should include a variant key."""
        config = model.get_config()
        assert "variant" in config
        assert config["variant"] in {
            "coral",
            "corn",
            "emd",
            "cdw_ce",
            "balanced_softmax",
            "ldam_drw",
            "soft_ordinal",
        }

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
