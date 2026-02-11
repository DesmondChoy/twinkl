"""Tests for VIF evaluation metrics."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.judge import SCHWARTZ_VALUE_ORDER


# ── Mock models ───────────────────────────────────────────────────────────────


class MockCriticMLP:
    """Mock CriticMLP: has __call__ and predict_with_uncertainty, but NOT predict/_variant_name."""

    def __init__(self, fixed_output: torch.Tensor):
        self._fixed = fixed_output  # (10,) — broadcast to batch

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        return self._fixed.unsqueeze(0).expand(batch, -1)

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self(x)
        std = torch.full_like(mean, 0.1)
        return mean, std


class MockOrdinalCritic:
    """Mock ordinal critic: has predict, _variant_name, and predict_with_uncertainty."""

    def __init__(self, fixed_output: torch.Tensor):
        self._fixed = fixed_output

    def to(self, device):
        return self

    def eval(self):
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        return self._fixed.unsqueeze(0).expand(batch, -1)

    def _variant_name(self) -> str:
        return "mock_ordinal"

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 50
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.predict(x)
        std = torch.full_like(mean, 0.05)
        return mean, std


def _make_dataloader(
    inputs: np.ndarray, targets: np.ndarray, batch_size: int = 32
) -> DataLoader:
    """Wrap numpy arrays into a TensorDataset + DataLoader."""
    x = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


# ── Metric tests ──────────────────────────────────────────────────────────────


class TestDiscretize:
    """Tests for discretize_predictions."""

    def test_clear_classes(self):
        """Values clearly in each class should map correctly."""
        from src.vif.eval import discretize_predictions

        values = np.array([-0.9, -0.6, 0.0, 0.3, 0.6, 0.9])
        result = discretize_predictions(values)
        np.testing.assert_array_equal(result, [-1, -1, 0, 0, 1, 1])

    def test_boundary_stays_neutral(self):
        """Exactly ±0.5 should be classified as neutral (0)."""
        from src.vif.eval import discretize_predictions

        values = np.array([-0.5, 0.5])
        result = discretize_predictions(values)
        np.testing.assert_array_equal(result, [0, 0])

    def test_just_past_boundary(self):
        """Values just past ±0.5 should be classified as -1/+1."""
        from src.vif.eval import discretize_predictions

        values = np.array([-0.500001, 0.500001])
        result = discretize_predictions(values)
        np.testing.assert_array_equal(result, [-1, 1])

    def test_2d_array(self):
        """Should work on 2D arrays (n_samples, n_dims)."""
        from src.vif.eval import discretize_predictions

        values = np.array([[0.0, 0.8], [-0.7, 0.3]])
        result = discretize_predictions(values)
        np.testing.assert_array_equal(result, [[0, 1], [-1, 0]])


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

        # Dims where targets vary but predictions are constant → NaN
        first_dim = SCHWARTZ_VALUE_ORDER[0]
        assert np.isnan(qwk[first_dim])


class TestMSE:
    """Tests for compute_mse_per_dimension."""

    def test_perfect_predictions(self):
        """MSE should be 0 for perfect predictions."""
        from src.vif.eval import compute_mse_per_dimension

        targets = np.array([[1, 0, -1, 1, 0, -1, 1, 0, -1, 1]] * 10, dtype=float)
        predictions = targets.copy()

        mse = compute_mse_per_dimension(predictions, targets)
        for v in mse.values():
            assert v == 0.0

    def test_known_mse_value(self):
        """MSE should match hand-computed value.

        4 samples, dim 0: predictions off by 1 in 2 samples.
        MSE = (1^2 + 1^2 + 0 + 0) / 4 = 0.5
        """
        from src.vif.eval import compute_mse_per_dimension

        targets = np.zeros((4, 10))
        predictions = np.zeros((4, 10))
        predictions[0, 0] = 1.0  # error^2 = 1
        predictions[1, 0] = -1.0  # error^2 = 1

        mse = compute_mse_per_dimension(predictions, targets)

        first_dim = SCHWARTZ_VALUE_ORDER[0]
        assert mse[first_dim] == pytest.approx(0.5)


class TestSpearman:
    """Tests for compute_spearman_per_dimension."""

    def test_perfect_positive_correlation(self):
        """Spearman should be 1.0 for perfectly monotonic predictions."""
        from src.vif.eval import compute_spearman_per_dimension

        # 10 samples with strictly increasing values in every dimension
        targets = np.tile(np.arange(10, dtype=float).reshape(-1, 1), (1, 10))
        predictions = targets.copy()

        sp = compute_spearman_per_dimension(predictions, targets)
        for v in sp.values():
            assert v == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        """Spearman should be -1.0 for perfectly reversed predictions."""
        from src.vif.eval import compute_spearman_per_dimension

        targets = np.tile(np.arange(10, dtype=float).reshape(-1, 1), (1, 10))
        predictions = targets[::-1].copy()  # reverse the order

        sp = compute_spearman_per_dimension(predictions, targets)
        for v in sp.values():
            assert v == pytest.approx(-1.0)

    def test_constant_targets_returns_nan(self):
        """Spearman should be NaN when targets are constant."""
        from src.vif.eval import compute_spearman_per_dimension

        targets = np.zeros((10, 10))
        predictions = np.random.default_rng(42).standard_normal((10, 10))

        sp = compute_spearman_per_dimension(predictions, targets)
        for v in sp.values():
            assert np.isnan(v)

    def test_constant_predictions_returns_nan(self):
        """Spearman should be NaN when predictions are constant."""
        from src.vif.eval import compute_spearman_per_dimension

        targets = np.tile(np.arange(10, dtype=float).reshape(-1, 1), (1, 10))
        predictions = np.zeros((10, 10))

        sp = compute_spearman_per_dimension(predictions, targets)
        for v in sp.values():
            assert np.isnan(v)

    def test_returns_all_ten_dimensions(self):
        """Output dict should have exactly the 10 Schwartz dimensions."""
        from src.vif.eval import compute_spearman_per_dimension

        targets = np.tile(np.arange(10, dtype=float).reshape(-1, 1), (1, 10))
        predictions = targets.copy()

        sp = compute_spearman_per_dimension(predictions, targets)
        assert list(sp.keys()) == SCHWARTZ_VALUE_ORDER


class TestAccuracy:
    """Tests for compute_accuracy_per_dimension."""

    def test_perfect_predictions(self):
        """Accuracy should be 1.0 for perfect predictions."""
        from src.vif.eval import compute_accuracy_per_dimension

        targets = np.array([[-1, 0, 1, -1, 0, 1, -1, 0, 1, -1]] * 10, dtype=float)
        predictions = targets.copy()

        acc = compute_accuracy_per_dimension(predictions, targets)
        for v in acc.values():
            assert v == pytest.approx(1.0)

    def test_threshold_behavior(self):
        """Predictions within [-0.5, 0.5] should discretize to 0.

        0.4 → 0 (correct when target=0), 0.6 → 1 (correct when target=1).
        """
        from src.vif.eval import compute_accuracy_per_dimension

        # All targets are 0; predictions are 0.4 → class 0 → correct
        targets = np.zeros((4, 10))
        predictions = np.full((4, 10), 0.4)

        acc = compute_accuracy_per_dimension(predictions, targets)
        for v in acc.values():
            assert v == pytest.approx(1.0)

    def test_boundary_at_half(self):
        """Exactly 0.5 stays neutral; >0.5 goes to +1.

        This documents the intentional threshold behavior: values at exactly
        ±0.5 are classified as neutral, avoiding numpy's bankers rounding.
        """
        from src.vif.eval import compute_accuracy_per_dimension

        # 0.5 → class 0 (neutral), target is 0 → correct
        targets = np.zeros((4, 10))
        predictions = np.full((4, 10), 0.5)

        acc = compute_accuracy_per_dimension(predictions, targets)
        for v in acc.values():
            assert v == pytest.approx(1.0)

        # 0.51 → class +1, target is 1 → correct
        targets_pos = np.ones((4, 10))
        predictions_pos = np.full((4, 10), 0.51)

        acc_pos = compute_accuracy_per_dimension(predictions_pos, targets_pos)
        for v in acc_pos.values():
            assert v == pytest.approx(1.0)

    def test_extreme_predictions_clamp_to_valid_range(self):
        """Predictions far outside [-1, 1] should still map to -1 or +1.

        1.6 > 0.5 → class +1 (matches target 1).
        """
        from src.vif.eval import compute_accuracy_per_dimension

        targets = np.ones((4, 10))
        predictions = np.full((4, 10), 1.6)

        acc = compute_accuracy_per_dimension(predictions, targets)
        for v in acc.values():
            assert v == pytest.approx(1.0)

    def test_all_wrong(self):
        """Accuracy should be 0.0 when all predictions are wrong."""
        from src.vif.eval import compute_accuracy_per_dimension

        targets = np.ones((10, 10))  # all 1
        predictions = np.full((10, 10), -0.6)  # < -0.5 → class -1

        acc = compute_accuracy_per_dimension(predictions, targets)
        for v in acc.values():
            assert v == pytest.approx(0.0)


# ── Integration tests ─────────────────────────────────────────────────────────


class TestEvaluateModel:
    """Tests for evaluate_model (no uncertainty)."""

    def test_non_ordinal_returns_mse_keys(self):
        """CriticMLP path should produce MSE-based keys, not MAE/QWK."""
        from src.vif.eval import evaluate_model

        fixed = torch.zeros(10)
        model = MockCriticMLP(fixed)
        targets = np.zeros((8, 10))
        dl = _make_dataloader(np.random.default_rng(0).standard_normal((8, 20)), targets)

        results = evaluate_model(model, dl)

        assert "mse_per_dim" in results
        assert "mse_mean" in results
        assert "mae_per_dim" not in results
        assert "qwk_per_dim" not in results

    def test_ordinal_returns_mae_qwk_keys(self):
        """Ordinal model with include_ordinal_metrics should produce MAE/QWK keys."""
        from src.vif.eval import evaluate_model

        fixed = torch.zeros(10)
        model = MockOrdinalCritic(fixed)
        targets = np.zeros((8, 10))
        dl = _make_dataloader(np.random.default_rng(0).standard_normal((8, 20)), targets)

        results = evaluate_model(model, dl, include_ordinal_metrics=True)

        assert "mae_per_dim" in results
        assert "mae_mean" in results
        assert "qwk_per_dim" in results
        assert "qwk_mean" in results
        assert "mse_per_dim" not in results

    def test_perfect_non_ordinal_predictions(self):
        """MSE should be 0 when model outputs exactly match targets."""
        from src.vif.eval import evaluate_model

        target_row = [0.5, -0.3, 0.1, 0.0, 0.7, -0.5, 0.2, 0.8, -0.1, 0.4]
        fixed = torch.tensor(target_row, dtype=torch.float32)
        model = MockCriticMLP(fixed)
        targets = np.tile(target_row, (8, 1))
        dl = _make_dataloader(np.random.default_rng(0).standard_normal((8, 20)), targets)

        results = evaluate_model(model, dl)

        assert results["mse_mean"] == pytest.approx(0.0, abs=1e-6)

    def test_ordinal_without_include_flag_uses_mse(self):
        """Ordinal model without include_ordinal_metrics flag should still use MSE."""
        from src.vif.eval import evaluate_model

        fixed = torch.zeros(10)
        model = MockOrdinalCritic(fixed)
        targets = np.zeros((8, 10))
        dl = _make_dataloader(np.random.default_rng(0).standard_normal((8, 20)), targets)

        # include_ordinal_metrics defaults to False → MSE path
        results = evaluate_model(model, dl)

        assert "mse_per_dim" in results
        assert "mae_per_dim" not in results


class TestEvaluateWithUncertainty:
    """Tests for evaluate_with_uncertainty (MC Dropout path)."""

    def test_returns_uncertainty_keys(self):
        """Result should contain predictions, uncertainties, and calibration."""
        from src.vif.eval import evaluate_with_uncertainty

        fixed = torch.zeros(10)
        model = MockCriticMLP(fixed)
        targets = np.zeros((8, 10))
        dl = _make_dataloader(np.random.default_rng(0).standard_normal((8, 20)), targets)

        results = evaluate_with_uncertainty(model, dl)

        assert "predictions" in results
        assert "uncertainties" in results
        assert "calibration" in results
        assert "error_uncertainty_correlation" in results["calibration"]
        assert "mean_uncertainty" in results["calibration"]

    def test_non_ordinal_computes_mse(self):
        """Without include_ordinal_metrics, should compute MSE (not MAE)."""
        from src.vif.eval import evaluate_with_uncertainty

        fixed = torch.zeros(10)
        model = MockCriticMLP(fixed)
        targets = np.zeros((8, 10))
        dl = _make_dataloader(np.random.default_rng(0).standard_normal((8, 20)), targets)

        results = evaluate_with_uncertainty(model, dl, include_ordinal_metrics=False)

        assert "mse_per_dim" in results
        assert "mae_per_dim" not in results

    def test_ordinal_computes_mae_qwk(self):
        """With include_ordinal_metrics=True, should compute MAE/QWK instead of MSE."""
        from src.vif.eval import evaluate_with_uncertainty

        fixed = torch.zeros(10)
        model = MockOrdinalCritic(fixed)
        targets = np.zeros((8, 10))
        dl = _make_dataloader(np.random.default_rng(0).standard_normal((8, 20)), targets)

        results = evaluate_with_uncertainty(
            model, dl, include_ordinal_metrics=True
        )

        assert "mae_per_dim" in results
        assert "qwk_per_dim" in results
        assert "mse_per_dim" not in results

    def test_mean_uncertainty_positive(self):
        """Mean uncertainty from mocks (fixed std=0.1) should be positive."""
        from src.vif.eval import evaluate_with_uncertainty

        fixed = torch.zeros(10)
        model = MockCriticMLP(fixed)
        targets = np.zeros((8, 10))
        dl = _make_dataloader(np.random.default_rng(0).standard_normal((8, 20)), targets)

        results = evaluate_with_uncertainty(model, dl)

        assert results["calibration"]["mean_uncertainty"] > 0


class TestFormatResultsTable:
    """Tests for format_results_table."""

    def _base_results(self, *, use_mae: bool = False) -> dict:
        """Build a minimal results dict for format_results_table."""
        zero_dims = {d: 0.0 for d in SCHWARTZ_VALUE_ORDER}
        nan_spearman = {d: float("nan") for d in SCHWARTZ_VALUE_ORDER}
        results = {
            "spearman_per_dim": nan_spearman,
            "spearman_mean": float("nan"),
            "accuracy_per_dim": {d: 1.0 for d in SCHWARTZ_VALUE_ORDER},
            "accuracy_mean": 1.0,
        }
        if use_mae:
            results["mae_per_dim"] = zero_dims
            results["mae_mean"] = 0.0
        else:
            results["mse_per_dim"] = zero_dims
            results["mse_mean"] = 0.0
        return results

    def test_mse_format(self):
        """MSE results should produce a table with 'MSE' header."""
        from src.vif.eval import format_results_table

        table = format_results_table(self._base_results(use_mae=False))

        assert "MSE" in table
        assert "MAE" not in table

    def test_mae_with_qwk_format(self):
        """MAE results with QWK should include both columns."""
        from src.vif.eval import format_results_table

        results = self._base_results(use_mae=True)
        results["qwk_per_dim"] = {d: 0.5 for d in SCHWARTZ_VALUE_ORDER}
        results["qwk_mean"] = 0.5

        table = format_results_table(results)

        assert "MAE" in table
        assert "QWK" in table

    def test_calibration_section_included(self):
        """When calibration data is present, the table should include it."""
        from src.vif.eval import format_results_table

        results = self._base_results()
        results["calibration"] = {
            "error_uncertainty_correlation": 0.42,
            "mean_uncertainty": 0.1234,
        }

        table = format_results_table(results)

        assert "Calibration" in table
        assert "0.42" in table

    def test_nan_spearman_shows_na(self):
        """NaN Spearman values should render as 'N/A' in the table."""
        from src.vif.eval import format_results_table

        table = format_results_table(self._base_results())

        assert "N/A" in table
        # Every dimension row should show N/A for its Spearman column
        for dim in SCHWARTZ_VALUE_ORDER:
            # Find the line containing this dimension
            dim_line = [line for line in table.split("\n") if dim in line]
            assert len(dim_line) == 1
            assert "N/A" in dim_line[0]
