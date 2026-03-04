"""Tests for LR finder helpers."""

from pathlib import Path

import numpy as np

from src.vif.lr_finder import compute_lr_suggestions, resolve_lr_finder_paths


def test_compute_lr_suggestions_returns_expected_keys():
    lrs = np.geomspace(1e-6, 1e-1, 20)
    losses = np.array(
        [6.0, 5.5, 5.1, 4.6, 4.0, 3.5, 3.0, 2.6, 2.3, 2.1, 2.0, 2.1, 2.4, 2.7, 3.1, 3.6, 4.2, 5.0, 5.8, 6.5]
    )

    suggestions = compute_lr_suggestions(lrs, losses)

    assert suggestions["lr_steep"] is not None
    assert suggestions["lr_valley"] is not None
    assert suggestions["n_valid_points"] == 20
    assert 1e-6 <= suggestions["lr_valley"] <= 1e-1
    assert suggestions["valley_strategy"] in {"valley_over_10", "fallback_lr_steep"}


def test_compute_lr_suggestions_filters_non_finite_values():
    lrs = np.geomspace(1e-5, 1e-1, 8)
    losses = np.array([5.0, np.nan, 3.5, 2.4, np.inf, 2.8, 3.2, 4.1])

    suggestions = compute_lr_suggestions(lrs, losses)

    assert suggestions["n_valid_points"] == 6
    assert suggestions["lr_steep"] is not None
    assert suggestions["lr_valley"] is not None


def test_compute_lr_suggestions_handles_insufficient_points():
    suggestions = compute_lr_suggestions([1e-4, 1e-3], [1.0, 0.9])

    assert suggestions["n_valid_points"] == 2
    assert suggestions["lr_steep"] is None
    assert suggestions["lr_valley"] is None
    assert suggestions["valley_strategy"] is None


def test_compute_lr_suggestions_monotonic_curve_falls_back_to_lr_steep():
    lrs = np.geomspace(1e-7, 1.0, 200)
    losses = np.linspace(1.0, 0.01, 200)  # almost perfectly monotonic downward

    suggestions = compute_lr_suggestions(lrs, losses)

    assert suggestions["is_mostly_monotonic"] is True
    assert suggestions["valley_strategy"] == "fallback_lr_steep"
    assert suggestions["lr_valley"] == suggestions["lr_steep"]


def test_resolve_lr_finder_paths_uses_defaults(tmp_path):
    plot_path, history_path = resolve_lr_finder_paths(tmp_path)

    assert plot_path == Path(tmp_path) / "lr_find_loss_vs_lr.png"
    assert history_path == Path(tmp_path) / "lr_find_history.json"


def test_resolve_lr_finder_paths_honors_override(tmp_path):
    custom_plot = Path(tmp_path) / "custom" / "lr_plot.png"

    plot_path, history_path = resolve_lr_finder_paths(tmp_path, custom_plot)

    assert plot_path == custom_plot
    assert history_path == custom_plot.with_suffix(".json")
    assert plot_path.parent.exists()
