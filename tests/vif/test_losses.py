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


class TestCDWCELoss:
    """Tests for CDW-CE loss."""

    def test_cdw_ce_loss_scalar_output(self):
        """Loss should be a non-negative scalar tensor."""
        from src.vif.critic_ordinal import cdw_ce_loss_multi

        logits = torch.randn(4, 30)
        y = torch.randint(-1, 2, (4, 10)).float()
        loss = cdw_ce_loss_multi(logits, y, alpha=2)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_cdw_ce_loss_perfect_prediction(self):
        """CDW-CE should be near zero for near-perfect one-hot predictions."""
        from src.vif.critic_ordinal import cdw_ce_loss_multi

        batch = 4
        logits = torch.zeros(batch, 30)
        for d in range(10):
            logits[:, d * 3 + 1] = 100.0  # true class is 0 => class index 1
        y = torch.zeros(batch, 10)

        loss = cdw_ce_loss_multi(logits, y, alpha=2)
        assert loss.item() < 1e-4

    def test_cdw_ce_loss_gradient_flow(self):
        """Loss should allow gradient backpropagation."""
        from src.vif.critic_ordinal import cdw_ce_loss_multi

        logits = torch.randn(4, 30, requires_grad=True)
        y = torch.randint(-1, 2, (4, 10)).float()
        loss = cdw_ce_loss_multi(logits, y, alpha=2)
        loss.backward()

        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_cdw_ce_loss_ordinal_sensitivity(self):
        """Off-by-2 error should produce larger loss than off-by-1."""
        from src.vif.critic_ordinal import cdw_ce_loss_multi

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

        loss_off1 = cdw_ce_loss_multi(logits_off1, y, alpha=2).item()
        loss_off2 = cdw_ce_loss_multi(logits_off2, y, alpha=2).item()

        assert loss_off2 > loss_off1

    def test_cdw_ce_alpha_penalizes_far_errors_more(self):
        """Higher alpha should increase penalty for distant misclassification."""
        from src.vif.critic_ordinal import cdw_ce_loss_multi

        batch = 8
        y = -torch.ones(batch, 10)  # true class: -1 (class index 0)
        logits = torch.zeros(batch, 30)
        for d in range(10):
            logits[:, d * 3 + 2] = 100.0  # off-by-2 prediction

        loss_a2 = cdw_ce_loss_multi(logits, y, alpha=2).item()
        loss_a5 = cdw_ce_loss_multi(logits, y, alpha=5).item()

        assert loss_a5 > loss_a2

    def test_cdw_ce_rejects_wrong_shapes(self):
        """Shape mismatches should raise ValueError with actionable messages."""
        from src.vif.critic_ordinal import cdw_ce_loss_multi

        y = torch.randint(-1, 2, (4, 10)).float()
        with pytest.raises(ValueError, match="logits must be 2D"):
            cdw_ce_loss_multi(torch.randn(4, 10, 3), y)

        with pytest.raises(ValueError, match="logits second dimension must be 30"):
            cdw_ce_loss_multi(torch.randn(4, 29), y)

        with pytest.raises(ValueError, match="y second dimension must be 10"):
            cdw_ce_loss_multi(torch.randn(4, 30), torch.randint(-1, 2, (4, 9)).float())

        with pytest.raises(ValueError, match="batch sizes must match"):
            cdw_ce_loss_multi(torch.randn(4, 30), torch.randint(-1, 2, (5, 10)).float())


class TestBalancedSoftmaxLoss:
    """Tests for Balanced Softmax loss."""

    @staticmethod
    def _class_priors() -> torch.Tensor:
        return torch.tensor([[0.1, 0.8, 0.1]] * 10, dtype=torch.float32)

    @staticmethod
    def _uniform_priors() -> torch.Tensor:
        return torch.full((10, 3), 1.0 / 3.0, dtype=torch.float32)

    @staticmethod
    def _uniform_weights() -> torch.Tensor:
        return torch.ones(10, dtype=torch.float32)

    @staticmethod
    def _neutral_logits(batch_size: int = 1, confidence: float = 8.0) -> torch.Tensor:
        logits = torch.zeros(batch_size, 30, dtype=torch.float32)
        logits[:, 1::3] = confidence
        return logits

    @staticmethod
    def _neutral_targets(batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(batch_size, 10, dtype=torch.float32)

    def test_balanced_softmax_loss_scalar_output(self):
        """Loss should be a non-negative scalar tensor."""
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        logits = torch.randn(4, 30)
        y = torch.randint(-1, 2, (4, 10)).float()
        class_priors = self._class_priors()

        loss = balanced_softmax_loss_multi(logits, y, class_priors=class_priors)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_balanced_softmax_loss_gradient_flow(self):
        """Loss should allow gradient backpropagation."""
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        logits = torch.randn(4, 30, requires_grad=True)
        y = torch.randint(-1, 2, (4, 10)).float()
        class_priors = self._class_priors()

        loss = balanced_softmax_loss_multi(logits, y, class_priors=class_priors)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == (4, 30)
        assert not torch.all(logits.grad == 0)

    def test_balanced_softmax_zero_weight_regularizer_matches_legacy_path(self):
        """Explicit zero regularizer weights should preserve legacy loss exactly."""
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        torch.manual_seed(42)
        logits = torch.randn(4, 30)
        y = torch.randint(-1, 2, (4, 10)).float()
        class_priors = self._class_priors()

        legacy_loss = balanced_softmax_loss_multi(logits, y, class_priors=class_priors)
        zero_weight_loss = balanced_softmax_loss_multi(
            logits,
            y,
            class_priors=class_priors,
            circumplex_regularizer_opposite_weight=0.0,
            circumplex_regularizer_adjacent_weight=0.0,
        )

        assert torch.equal(legacy_loss, zero_weight_loss)

    def test_uniform_dimension_weights_match_unweighted_loss(self):
        """Uniform dimension weights should preserve the unweighted CE path."""
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        torch.manual_seed(123)
        logits = torch.randn(4, 30)
        y = torch.randint(-1, 2, (4, 10)).float()
        class_priors = self._class_priors()

        unweighted_loss = balanced_softmax_loss_multi(
            logits,
            y,
            class_priors=class_priors,
        )
        weighted_loss = balanced_softmax_loss_multi(
            logits,
            y,
            class_priors=class_priors,
            dimension_weights=self._uniform_weights(),
        )

        torch.testing.assert_close(weighted_loss, unweighted_loss)

    def test_balanced_softmax_dimension_weights_change_loss(self):
        """Non-uniform dimension weights should change the CE contribution."""
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        logits = self._neutral_logits()
        targets = self._neutral_targets()
        # Make the first dimension a high-confidence error while the rest stay correct.
        logits[:, 0:3] = torch.tensor([0.0, 0.0, 8.0])
        targets[:, 0] = -1.0
        dimension_weights = torch.tensor([2.0] + [0.5] * 9, dtype=torch.float32)

        unweighted_loss = balanced_softmax_loss_multi(
            logits,
            targets,
            class_priors=self._uniform_priors(),
        )
        weighted_loss = balanced_softmax_loss_multi(
            logits,
            targets,
            class_priors=self._uniform_priors(),
            dimension_weights=dimension_weights,
        )

        assert weighted_loss.item() > unweighted_loss.item()

    def test_balanced_softmax_circumplex_regularizer_gradient_flow(self):
        """Circumplex-regularized loss should still backpropagate gradients."""
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        logits = torch.randn(4, 30, requires_grad=True)
        y = torch.randint(-1, 2, (4, 10)).float()

        loss = balanced_softmax_loss_multi(
            logits,
            y,
            class_priors=self._class_priors(),
            circumplex_regularizer_opposite_weight=0.5,
            circumplex_regularizer_adjacent_weight=0.1,
        )
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == (4, 30)
        assert not torch.all(logits.grad == 0)

    def test_balanced_softmax_rare_class_increases_penalty(self):
        """Rare-class targets should incur a larger penalty than uniform priors."""
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        logits = torch.zeros(1, 30)
        y = -torch.ones(1, 10)
        uniform_priors = self._uniform_priors()
        imbalanced_priors = torch.tensor([[0.05, 0.90, 0.05]] * 10, dtype=torch.float32)

        loss_uniform = balanced_softmax_loss_multi(logits, y, class_priors=uniform_priors)
        loss_imbalanced = balanced_softmax_loss_multi(logits, y, class_priors=imbalanced_priors)

        assert loss_imbalanced.item() > loss_uniform.item()

    def test_balanced_softmax_regularizer_penalizes_opposite_pair_same_sign_activation(self):
        """Confident same-sign activation on opposing pairs should increase loss."""
        from src.models.judge import SCHWARTZ_VALUE_ORDER
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        security_idx = SCHWARTZ_VALUE_ORDER.index("security")
        self_direction_idx = SCHWARTZ_VALUE_ORDER.index("self_direction")

        logits_benign = self._neutral_logits()
        targets_benign = self._neutral_targets()
        logits_benign[:, security_idx * 3 : (security_idx + 1) * 3] = torch.tensor([0.0, 0.0, 8.0])
        logits_benign[:, self_direction_idx * 3 : (self_direction_idx + 1) * 3] = torch.tensor([0.0, 8.0, 0.0])
        targets_benign[:, security_idx] = 1.0

        logits_opposite = logits_benign.clone()
        targets_opposite = targets_benign.clone()
        logits_opposite[:, self_direction_idx * 3 : (self_direction_idx + 1) * 3] = torch.tensor([0.0, 0.0, 8.0])
        targets_opposite[:, self_direction_idx] = 1.0

        loss_benign = balanced_softmax_loss_multi(
            logits_benign,
            targets_benign,
            class_priors=self._uniform_priors(),
            circumplex_regularizer_opposite_weight=0.5,
        )
        loss_opposite = balanced_softmax_loss_multi(
            logits_opposite,
            targets_opposite,
            class_priors=self._uniform_priors(),
            circumplex_regularizer_opposite_weight=0.5,
        )

        assert loss_opposite.item() > loss_benign.item()

    def test_balanced_softmax_regularizer_rewards_adjacent_positive_support(self):
        """Positive adjacent-pair support should reduce the total loss."""
        from src.models.judge import SCHWARTZ_VALUE_ORDER
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        self_direction_idx = SCHWARTZ_VALUE_ORDER.index("self_direction")
        stimulation_idx = SCHWARTZ_VALUE_ORDER.index("stimulation")

        logits_isolated = self._neutral_logits()
        targets_isolated = self._neutral_targets()
        logits_isolated[:, self_direction_idx * 3 : (self_direction_idx + 1) * 3] = torch.tensor([0.0, 0.0, 8.0])
        logits_isolated[:, stimulation_idx * 3 : (stimulation_idx + 1) * 3] = torch.tensor([0.0, 8.0, 0.0])
        targets_isolated[:, self_direction_idx] = 1.0

        logits_adjacent = logits_isolated.clone()
        targets_adjacent = targets_isolated.clone()
        logits_adjacent[:, stimulation_idx * 3 : (stimulation_idx + 1) * 3] = torch.tensor([0.0, 0.0, 8.0])
        targets_adjacent[:, stimulation_idx] = 1.0

        loss_isolated = balanced_softmax_loss_multi(
            logits_isolated,
            targets_isolated,
            class_priors=self._uniform_priors(),
            circumplex_regularizer_adjacent_weight=0.1,
        )
        loss_adjacent = balanced_softmax_loss_multi(
            logits_adjacent,
            targets_adjacent,
            class_priors=self._uniform_priors(),
            circumplex_regularizer_adjacent_weight=0.1,
        )

        assert loss_adjacent.item() < loss_isolated.item()

    def test_inverse_loss_dimension_weights_downweight_higher_losses(self):
        """Higher EMA loss should map to a smaller inverse-loss weight."""
        from src.vif.critic_ordinal import compute_inverse_loss_dimension_weights

        loss_ema = torch.tensor([4.0, 0.25] + [1.0] * 8, dtype=torch.float32)

        weights = compute_inverse_loss_dimension_weights(loss_ema, temperature=0.5)

        assert weights.shape == (10,)
        assert weights[1].item() > weights[0].item()

    def test_inverse_loss_dimension_weights_respect_clamps(self):
        """Configured min/max clamps should bound the resulting weights."""
        from src.vif.critic_ordinal import compute_inverse_loss_dimension_weights

        loss_ema = torch.tensor([0.0, 100.0] + [1.0] * 8, dtype=torch.float32)

        weights = compute_inverse_loss_dimension_weights(
            loss_ema,
            temperature=1.0,
            eps=1e-6,
            min_weight=0.8,
            max_weight=1.2,
        )

        assert torch.all(weights >= 0.8)
        assert torch.all(weights <= 1.2)

    def test_balanced_softmax_rejects_wrong_shapes(self):
        """Shape mismatches should raise ValueError."""
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        y = torch.randint(-1, 2, (4, 10)).float()
        class_priors = self._class_priors()

        with pytest.raises(ValueError):
            balanced_softmax_loss_multi(torch.randn(4, 10, 3), y, class_priors=class_priors)

        with pytest.raises(ValueError):
            balanced_softmax_loss_multi(torch.randn(4, 30), torch.randint(-1, 2, (4, 9)).float(), class_priors=class_priors)

        with pytest.raises(ValueError):
            balanced_softmax_loss_multi(torch.randn(4, 30), y, class_priors=torch.ones(9, 3))

        with pytest.raises(ValueError, match="dimension_weights"):
            balanced_softmax_loss_multi(
                torch.randn(4, 30),
                y,
                class_priors=class_priors,
                dimension_weights=torch.ones(9),
            )

    def test_balanced_softmax_rejects_non_positive_dimension_weights(self):
        """Dimension weights must be finite and strictly positive."""
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        logits = torch.randn(4, 30)
        y = torch.randint(-1, 2, (4, 10)).float()
        class_priors = self._class_priors()

        with pytest.raises(ValueError, match="strictly positive"):
            balanced_softmax_loss_multi(
                logits,
                y,
                class_priors=class_priors,
                dimension_weights=torch.tensor([0.0] + [1.0] * 9),
            )

        with pytest.raises(ValueError, match="finite"):
            balanced_softmax_loss_multi(
                logits,
                y,
                class_priors=class_priors,
                dimension_weights=torch.tensor([float("inf")] + [1.0] * 9),
            )

    def test_balanced_softmax_rejects_negative_regularizer_weights(self):
        """Circumplex regularizer weights must be non-negative."""
        from src.vif.critic_ordinal import balanced_softmax_loss_multi

        logits = torch.randn(4, 30)
        y = torch.randint(-1, 2, (4, 10)).float()

        with pytest.raises(ValueError, match="circumplex_regularizer_opposite_weight"):
            balanced_softmax_loss_multi(
                logits,
                y,
                class_priors=self._class_priors(),
                circumplex_regularizer_opposite_weight=-0.1,
            )

        with pytest.raises(ValueError, match="circumplex_regularizer_adjacent_weight"):
            balanced_softmax_loss_multi(
                logits,
                y,
                class_priors=self._class_priors(),
                circumplex_regularizer_adjacent_weight=-0.1,
            )

    def test_validate_dimension_weighting_config_rejects_invalid_inputs(self):
        """Dimension-weighting config should reject invalid schedule parameters."""
        from src.vif.critic_ordinal import validate_dimension_weighting_config

        with pytest.raises(ValueError, match="dimension_weighting_mode"):
            validate_dimension_weighting_config(
                mode="unknown",
                temperature=0.5,
                ema_alpha=0.3,
                warmup_epochs=1,
                eps=1e-6,
                min_weight=0.5,
                max_weight=1.5,
            )

        with pytest.raises(ValueError, match="dimension_weighting_temperature"):
            validate_dimension_weighting_config(
                mode="inverse_loss",
                temperature=0.0,
                ema_alpha=0.3,
                warmup_epochs=1,
                eps=1e-6,
                min_weight=0.5,
                max_weight=1.5,
            )

        with pytest.raises(ValueError, match="dimension_weighting_ema_alpha"):
            validate_dimension_weighting_config(
                mode="inverse_loss",
                temperature=0.5,
                ema_alpha=1.1,
                warmup_epochs=1,
                eps=1e-6,
                min_weight=0.5,
                max_weight=1.5,
            )

        with pytest.raises(ValueError, match="dimension_weighting_warmup_epochs"):
            validate_dimension_weighting_config(
                mode="inverse_loss",
                temperature=0.5,
                ema_alpha=0.3,
                warmup_epochs=-1,
                eps=1e-6,
                min_weight=0.5,
                max_weight=1.5,
            )

        with pytest.raises(ValueError, match="dimension_weighting_eps"):
            validate_dimension_weighting_config(
                mode="inverse_loss",
                temperature=0.5,
                ema_alpha=0.3,
                warmup_epochs=1,
                eps=0.0,
                min_weight=0.5,
                max_weight=1.5,
            )

        with pytest.raises(ValueError, match="dimension_weighting_min"):
            validate_dimension_weighting_config(
                mode="inverse_loss",
                temperature=0.5,
                ema_alpha=0.3,
                warmup_epochs=1,
                eps=1e-6,
                min_weight=0.0,
                max_weight=1.5,
            )

        with pytest.raises(ValueError, match="dimension_weighting_min must be <="):
            validate_dimension_weighting_config(
                mode="inverse_loss",
                temperature=0.5,
                ema_alpha=0.3,
                warmup_epochs=1,
                eps=1e-6,
                min_weight=1.6,
                max_weight=1.5,
            )


class TestLDAMDRWLoss:
    """Tests for LDAM-DRW loss."""

    @staticmethod
    def _class_counts() -> torch.Tensor:
        return torch.tensor(
            [
                [8.0, 96.0, 12.0],
            ] * 10,
            dtype=torch.float32,
        )

    def test_ldam_drw_loss_scalar_output(self):
        """Loss should be a non-negative scalar tensor."""
        from src.vif.critic_ordinal import ldam_drw_loss_multi

        logits = torch.randn(4, 30)
        y = torch.randint(-1, 2, (4, 10)).float()

        loss = ldam_drw_loss_multi(logits, y, class_counts=self._class_counts(), epoch=0)

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_ldam_drw_loss_gradient_flow(self):
        """Loss should allow gradient backpropagation."""
        from src.vif.critic_ordinal import ldam_drw_loss_multi

        logits = torch.randn(4, 30, requires_grad=True)
        y = torch.randint(-1, 2, (4, 10)).float()

        loss = ldam_drw_loss_multi(logits, y, class_counts=self._class_counts(), epoch=0)
        loss.backward()

        assert logits.grad is not None
        assert logits.grad.shape == (4, 30)
        assert not torch.all(logits.grad == 0)

    def test_ldam_drw_margin_increases_rare_class_penalty(self):
        """Minority-class targets should be penalized more than majority targets."""
        from src.vif.critic_ordinal import ldam_drw_loss_multi

        logits = torch.zeros(1, 30)
        rare_target = -torch.ones(1, 10)
        majority_target = torch.zeros(1, 10)

        loss_rare = ldam_drw_loss_multi(logits, rare_target, class_counts=self._class_counts(), epoch=0)
        loss_majority = ldam_drw_loss_multi(logits, majority_target, class_counts=self._class_counts(), epoch=0)

        assert loss_rare.item() > loss_majority.item()

    def test_ldam_drw_deferred_reweighting_changes_loss_after_threshold(self):
        """DRW should change the loss once the deferred-weighting epoch is reached."""
        from src.vif.critic_ordinal import ldam_drw_loss_multi

        logits = torch.zeros(1, 30)
        y = -torch.ones(1, 10)

        loss_before_drw = ldam_drw_loss_multi(
            logits,
            y,
            class_counts=self._class_counts(),
            epoch=10,
            drw_start_epoch=50,
        )
        loss_after_drw = ldam_drw_loss_multi(
            logits,
            y,
            class_counts=self._class_counts(),
            epoch=75,
            drw_start_epoch=50,
        )

        assert loss_after_drw.item() != pytest.approx(loss_before_drw.item())

    def test_ldam_drw_rejects_wrong_shapes(self):
        """Shape mismatches should raise ValueError."""
        from src.vif.critic_ordinal import ldam_drw_loss_multi

        y = torch.randint(-1, 2, (4, 10)).float()

        with pytest.raises(ValueError):
            ldam_drw_loss_multi(torch.randn(4, 10, 3), y, class_counts=self._class_counts(), epoch=0)

        with pytest.raises(ValueError):
            ldam_drw_loss_multi(torch.randn(4, 30), torch.randint(-1, 2, (4, 9)).float(), class_counts=self._class_counts(), epoch=0)

        with pytest.raises(ValueError):
            ldam_drw_loss_multi(torch.randn(4, 30), y, class_counts=torch.ones(9, 3), epoch=0)


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
