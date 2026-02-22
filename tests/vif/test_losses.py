"""Tests for ordinal loss functions."""

import numpy as np
import torch
import pytest


class TestCoralLoss:
    """Tests for CORAL loss and helper functions."""

    def test_alignment_to_levels_known_values(self):
        """Verify level encoding for each class: -1→[0,0], 0→[1,0], 1→[1,1]."""
        from src.vif.critic_ordinal import alignment_to_levels

        y = torch.tensor([[-1.0, 0.0, 1.0]])  # (1, 3)
        levels = alignment_to_levels(y)  # (1, 3, 2)

        assert levels.shape == (1, 3, 2)
        # class 0 (-1) → [0, 0]
        assert levels[0, 0].tolist() == [0.0, 0.0]
        # class 1 (0) → [1, 0]
        assert levels[0, 1].tolist() == [1.0, 0.0]
        # class 2 (+1) → [1, 1]
        assert levels[0, 2].tolist() == [1.0, 1.0]

    def test_coral_loss_scalar_output(self):
        """Loss should be a scalar tensor."""
        from src.vif.critic_ordinal import coral_loss_multi

        logits = torch.randn(4, 20)
        y = torch.randint(-1, 2, (4, 10)).float()
        loss = coral_loss_multi(logits, y)

        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # positive

    def test_coral_loss_gradient_flow(self):
        """Loss should allow gradient backpropagation."""
        from src.vif.critic_ordinal import coral_loss_multi

        logits = torch.randn(4, 20, requires_grad=True)
        y = torch.randint(-1, 2, (4, 10)).float()
        loss = coral_loss_multi(logits, y)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == (4, 20)
        assert not torch.all(logits.grad == 0)


class TestCornLoss:
    """Tests for CORN loss and helper functions."""

    def test_alignment_to_corn_labels(self):
        """Verify {-1,0,1} → {0,1,2} mapping."""
        from src.vif.critic_ordinal import alignment_to_corn_labels

        y = torch.tensor([[-1.0, 0.0, 1.0]])
        labels = alignment_to_corn_labels(y)
        assert labels.tolist() == [[0, 1, 2]]

    def test_corn_labels_to_alignment(self):
        """Verify {0,1,2} → {-1,0,1} mapping."""
        from src.vif.critic_ordinal import corn_labels_to_alignment

        labels = torch.tensor([0, 1, 2])
        alignment = corn_labels_to_alignment(labels)
        assert alignment.tolist() == [-1.0, 0.0, 1.0]

    def test_corn_loss_scalar_output(self):
        """Loss should be a scalar tensor."""
        from src.vif.critic_ordinal import corn_loss_multi

        logits = torch.randn(4, 20)
        y = torch.randint(-1, 2, (4, 10)).float()
        loss = corn_loss_multi(logits, y)

        assert loss.dim() == 0

    def test_corn_loss_gradient_flow(self):
        """Loss should allow gradient backpropagation."""
        from src.vif.critic_ordinal import corn_loss_multi

        logits = torch.randn(4, 20, requires_grad=True)
        y = torch.randint(-1, 2, (4, 10)).float()
        loss = corn_loss_multi(logits, y)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


class TestEmdLoss:
    """Tests for EMD loss."""

    def test_emd_loss_scalar_output(self):
        """Loss should be a scalar tensor."""
        from src.vif.critic_ordinal import emd_loss_multi

        logits = torch.randn(4, 30)
        y = torch.randint(-1, 2, (4, 10)).float()
        loss = emd_loss_multi(logits, y)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_emd_loss_perfect_prediction(self):
        """EMD should be zero when predicted distribution matches target."""
        from src.vif.critic_ordinal import emd_loss_multi

        # Create logits that will produce one-hot after softmax
        # For class 1 (alignment=0): [small, large, small] per dim
        batch = 4
        logits = torch.zeros(batch, 30)
        for d in range(10):
            logits[:, d * 3 + 1] = 100.0  # Strong preference for class 1
        y = torch.zeros(batch, 10)  # All neutral → class 1

        loss = emd_loss_multi(logits, y)
        assert loss.item() < 1e-4

    def test_emd_loss_gradient_flow(self):
        """Loss should allow gradient backpropagation."""
        from src.vif.critic_ordinal import emd_loss_multi

        logits = torch.randn(4, 30, requires_grad=True)
        y = torch.randint(-1, 2, (4, 10)).float()
        loss = emd_loss_multi(logits, y)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_emd_loss_ordinal_sensitivity(self):
        """Off-by-2 error should produce larger loss than off-by-1."""
        from src.vif.critic_ordinal import emd_loss_multi

        batch = 8
        y = -torch.ones(batch, 10)  # true class: -1 (class index 0)

        # Predict class 1 (off-by-1): strong logits for middle class
        logits_off1 = torch.zeros(batch, 30)
        for d in range(10):
            logits_off1[:, d * 3 + 1] = 100.0

        # Predict class 2 (off-by-2): strong logits for last class
        logits_off2 = torch.zeros(batch, 30)
        for d in range(10):
            logits_off2[:, d * 3 + 2] = 100.0

        loss_off1 = emd_loss_multi(logits_off1, y).item()
        loss_off2 = emd_loss_multi(logits_off2, y).item()

        assert loss_off2 > loss_off1


class TestSoftOrdinalLoss:
    """Tests for Soft Ordinal loss."""

    def test_make_soft_ordinal_targets_shape(self):
        """Soft targets should have correct shape."""
        from src.vif.critic_ordinal import make_soft_ordinal_targets

        y = torch.randint(-1, 2, (4, 10)).float()
        soft = make_soft_ordinal_targets(y)
        assert soft.shape == (4, 10, 3)

    def test_make_soft_ordinal_targets_valid_distribution(self):
        """Soft targets should sum to 1 along class axis."""
        from src.vif.critic_ordinal import make_soft_ordinal_targets

        y = torch.randint(-1, 2, (4, 10)).float()
        soft = make_soft_ordinal_targets(y)
        sums = soft.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_make_soft_ordinal_targets_peak_at_true_class(self):
        """Highest probability should be at the true class."""
        from src.vif.critic_ordinal import make_soft_ordinal_targets

        y = torch.tensor([[0.0]])  # class 1
        soft = make_soft_ordinal_targets(y)  # (1, 1, 3)
        assert soft[0, 0].argmax().item() == 1

    def test_soft_ordinal_loss_scalar_output(self):
        """Loss should be a scalar tensor."""
        from src.vif.critic_ordinal import soft_ordinal_loss_multi

        logits = torch.randn(4, 30)
        y = torch.randint(-1, 2, (4, 10)).float()
        loss = soft_ordinal_loss_multi(logits, y)

        assert loss.dim() == 0

    def test_soft_ordinal_loss_gradient_flow(self):
        """Loss should allow gradient backpropagation."""
        from src.vif.critic_ordinal import soft_ordinal_loss_multi

        logits = torch.randn(4, 30, requires_grad=True)
        y = torch.randint(-1, 2, (4, 10)).float()
        loss = soft_ordinal_loss_multi(logits, y)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


class TestCoralLossWeighted:
    """Tests for weighted CORAL loss and importance weight computation."""

    def test_uniform_weights_match_unweighted(self):
        """Uniform weights (all 1.0) should produce the same loss as coral_loss_multi.

        This is the critical correctness test: validates that the per-dimension
        loop restructuring in coral_loss_multi_weighted is equivalent to the
        flattened approach in coral_loss_multi when no reweighting is applied.
        """
        from src.vif.critic_ordinal import coral_loss_multi, coral_loss_multi_weighted

        torch.manual_seed(42)
        logits = torch.randn(8, 20)
        y = torch.randint(-1, 2, (8, 10)).float()
        uniform_weights = torch.ones(10, 2)

        loss_unweighted = coral_loss_multi(logits, y)
        loss_weighted = coral_loss_multi_weighted(logits, y, uniform_weights)

        assert torch.allclose(loss_unweighted, loss_weighted, atol=1e-5), (
            f"Uniform-weighted loss {loss_weighted.item():.6f} != "
            f"unweighted loss {loss_unweighted.item():.6f}"
        )

    def test_weighted_loss_scalar_output(self):
        """Weighted loss should return a positive scalar tensor."""
        from src.vif.critic_ordinal import coral_loss_multi_weighted

        logits = torch.randn(4, 20)
        y = torch.randint(-1, 2, (4, 10)).float()
        weights = torch.rand(10, 2) + 0.1  # positive weights

        loss = coral_loss_multi_weighted(logits, y, weights)

        assert loss.dim() == 0
        assert loss.item() > 0

    def test_weighted_loss_gradient_flow(self):
        """Gradients should propagate through weighted loss."""
        from src.vif.critic_ordinal import coral_loss_multi_weighted

        logits = torch.randn(4, 20, requires_grad=True)
        y = torch.randint(-1, 2, (4, 10)).float()
        weights = torch.ones(10, 2)

        loss = coral_loss_multi_weighted(logits, y, weights)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == (4, 20)
        assert not torch.all(logits.grad == 0)

    def test_nonuniform_weights_change_loss(self):
        """Non-uniform weights should produce a different loss than uniform."""
        from src.vif.critic_ordinal import coral_loss_multi_weighted

        torch.manual_seed(42)
        logits = torch.randn(8, 20)
        y = torch.randint(-1, 2, (8, 10)).float()

        uniform = torch.ones(10, 2)
        nonuniform = torch.ones(10, 2)
        nonuniform[0, 0] = 5.0  # heavily weight threshold 0 of first dim

        loss_uniform = coral_loss_multi_weighted(logits, y, uniform)
        loss_nonuniform = coral_loss_multi_weighted(logits, y, nonuniform)

        assert not torch.allclose(loss_uniform, loss_nonuniform, atol=1e-6)

    def test_compute_importance_weights_shape(self):
        """compute_coral_importance_weights should return (n_dims, 2) tensor."""
        from src.vif.critic_ordinal import compute_coral_importance_weights

        counts = np.array([[50, 800, 150]] * 10)  # 10 dims
        weights = compute_coral_importance_weights(counts)

        assert weights.shape == (10, 2)
        assert weights.dtype == torch.float32

    def test_compute_importance_weights_normalized(self):
        """Per-dimension weights should have mean=1.0."""
        from src.vif.critic_ordinal import compute_coral_importance_weights

        counts = np.array([
            [51, 1282, 127],   # stimulation-like
            [204, 859, 397],   # self_direction-like
            [56, 1246, 158],   # universalism-like
        ])
        weights = compute_coral_importance_weights(counts)

        for d in range(3):
            dim_mean = weights[d].mean().item()
            assert abs(dim_mean - 1.0) < 1e-5, (
                f"Dim {d} mean weight {dim_mean:.6f} != 1.0"
            )

    def test_compute_importance_weights_methods(self):
        """Both 'inverse_freq' and 'inverse_sqrt' should return valid results."""
        from src.vif.critic_ordinal import compute_coral_importance_weights

        counts = np.array([[50, 800, 150]] * 5)

        for method in ["inverse_freq", "inverse_sqrt"]:
            weights = compute_coral_importance_weights(counts, method=method)
            assert weights.shape == (5, 2)
            assert torch.all(weights > 0)
            # Each dim's mean should be 1.0
            for d in range(5):
                assert abs(weights[d].mean().item() - 1.0) < 1e-5

    def test_weighted_loss_rejects_wrong_shape(self):
        """Wrong importance_weights shape should raise ValueError."""
        from src.vif.critic_ordinal import coral_loss_multi_weighted

        logits = torch.randn(4, 20)
        y = torch.randint(-1, 2, (4, 10)).float()

        with pytest.raises(ValueError, match="importance_weights shape"):
            coral_loss_multi_weighted(logits, y, torch.ones(5, 2))

        with pytest.raises(ValueError, match="importance_weights shape"):
            coral_loss_multi_weighted(logits, y, torch.ones(10, 3))
