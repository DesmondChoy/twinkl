"""Tests for evaluation metrics (MAE, QWK)."""

import numpy as np
import pytest


class TestMAE:
    """Tests for compute_mae_per_dimension."""

    def test_perfect_predictions(self):
        """MAE should be 0 for perfect predictions."""
        from src.vif.eval import compute_mae_per_dimension

        targets = np.array([[1, 0, -1, 1, 0, -1, 1, 0, -1, 1]] * 10, dtype=float)
        predictions = targets.copy()

        mae = compute_mae_per_dimension(predictions, targets)
        for v in mae.values():
            assert v == 0.0

    def test_known_mae_value(self):
        """MAE should match hand-computed value."""
        from src.vif.eval import compute_mae_per_dimension

        # 4 samples, all error in dim 0: predictions off by 1 in 2 samples
        targets = np.zeros((4, 10))
        predictions = np.zeros((4, 10))
        predictions[0, 0] = 1.0  # error = 1
        predictions[1, 0] = -1.0  # error = 1

        mae = compute_mae_per_dimension(predictions, targets)
        from src.models.judge import SCHWARTZ_VALUE_ORDER

        first_dim = SCHWARTZ_VALUE_ORDER[0]
        assert mae[first_dim] == pytest.approx(0.5)  # 2/4 = 0.5


class TestQWK:
    """Tests for compute_qwk_per_dimension."""

    def test_perfect_agreement(self):
        """QWK should be 1.0 for perfect agreement (when variability exists)."""
        from src.vif.eval import compute_qwk_per_dimension

        # Need variety in both predictions and targets for QWK to be defined
        targets = np.array([[-1, 0, 1, -1, 0, 1, -1, 0, 1, -1]] * 10, dtype=float)
        predictions = targets.copy()

        qwk = compute_qwk_per_dimension(predictions, targets)
        for dim, v in qwk.items():
            if not np.isnan(v):
                assert v == pytest.approx(1.0, abs=1e-6)

    def test_constant_predictions_returns_nan(self):
        """QWK should be NaN when predictions are constant."""
        from src.vif.eval import compute_qwk_per_dimension

        targets = np.array([[-1, 0, 1, -1, 0, 1, -1, 0, 1, -1]] * 10, dtype=float)
        predictions = np.zeros_like(targets)  # constant 0

        qwk = compute_qwk_per_dimension(predictions, targets)
        from src.models.judge import SCHWARTZ_VALUE_ORDER

        # Dims where targets vary but predictions are constant → NaN
        first_dim = SCHWARTZ_VALUE_ORDER[0]
        assert np.isnan(qwk[first_dim])
