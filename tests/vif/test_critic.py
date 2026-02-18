"""Tests for CriticMLP model."""

import torch
import pytest

from src.vif.critic import CriticMLP


class TestCriticMLP:
    """Core contract tests for CriticMLP."""

    @pytest.fixture
    def model(self):
        return CriticMLP(input_dim=20, hidden_dim=32, output_dim=10, dropout=0.2)

    @pytest.fixture
    def batch(self):
        return torch.randn(4, 20)

    def test_forward_shape(self, model, batch):
        """forward() should return (batch_size, output_dim)."""
        out = model(batch)
        assert out.shape == (4, 10)

    def test_output_in_tanh_range(self, model, batch):
        """All outputs should be in [-1, 1] due to tanh activation."""
        out = model(batch)
        assert (out >= -1.0).all()
        assert (out <= 1.0).all()

    def test_predict_with_uncertainty_shape(self, model, batch):
        """predict_with_uncertainty should return (batch, 10) mean and std."""
        mean, std = model.predict_with_uncertainty(batch, n_samples=5)
        assert mean.shape == (4, 10)
        assert std.shape == (4, 10)

    def test_uncertainty_non_negative(self, model, batch):
        """Standard deviation should be non-negative."""
        _, std = model.predict_with_uncertainty(batch, n_samples=5)
        assert (std >= 0).all()

    def test_get_config_from_config_roundtrip(self, model):
        """Config serialization should preserve all parameters."""
        config = model.get_config()
        assert config == {
            "input_dim": 20,
            "hidden_dim": 32,
            "output_dim": 10,
            "dropout": 0.2,
        }

        restored = CriticMLP.from_config(config)
        assert restored.input_dim == 20
        assert restored.hidden_dim == 32
        assert restored.output_dim == 10
        assert restored.dropout_p == 0.2
        assert restored.fc_out.out_features == 10

    def test_output_dim_matches_fc_out(self, model):
        """fc_out layer should have output_dim output features."""
        assert model.fc_out.out_features == 10

    def test_enable_dropout_in_eval_mode(self, model, batch):
        """enable_dropout should keep dropout active even in eval mode."""
        model.eval()
        model.enable_dropout()
        # Dropout modules should be in training mode
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                assert module.training
