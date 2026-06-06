"""Tests for recall-aware checkpoint replay helpers."""

import polars as pl

from scripts.experiments.replay_recall_aware_checkpoint_selection import (
    label_regime_for_run,
    select_current_policy,
    select_recall_window_policy,
)


def _trace(rows: list[dict]) -> pl.DataFrame:
    defaults = {
        "calibration_global": 0.5,
        "hedging_mean": 0.6,
        "val_loss": 0.5,
        "qwk_nan_dims_count": 0,
        "eligible": True,
        "ineligible_reasons": "",
    }
    return pl.DataFrame([{**defaults, **row} for row in rows])


def test_current_policy_prioritizes_qwk_before_recall():
    trace = _trace(
        [
            {"epoch": 0, "qwk_mean": 0.50, "recall_minus1": 0.20},
            {"epoch": 1, "qwk_mean": 0.49, "recall_minus1": 0.80},
        ]
    )

    selected = select_current_policy(trace)

    assert selected["epoch"] == 0


def test_recall_window_policy_picks_high_recall_epoch_inside_qwk_window():
    trace = _trace(
        [
            {"epoch": 0, "qwk_mean": 0.50, "recall_minus1": 0.20},
            {"epoch": 1, "qwk_mean": 0.49, "recall_minus1": 0.80},
            {"epoch": 2, "qwk_mean": 0.47, "recall_minus1": 0.95},
        ]
    )

    selected = select_recall_window_policy(trace, qwk_window=0.02)

    assert selected["epoch"] == 1


def test_recall_window_policy_excludes_epoch_below_qwk_window():
    trace = _trace(
        [
            {"epoch": 0, "qwk_mean": 0.50, "recall_minus1": 0.20},
            {"epoch": 1, "qwk_mean": 0.47, "recall_minus1": 0.95},
        ]
    )

    selected = select_recall_window_policy(trace, qwk_window=0.02)

    assert selected["epoch"] == 0


def test_recall_window_policy_tiebreaks_on_lower_hedging_then_calibration_then_epoch():
    trace = _trace(
        [
            {
                "epoch": 0,
                "qwk_mean": 0.50,
                "recall_minus1": 0.80,
                "hedging_mean": 0.60,
                "calibration_global": 0.9,
            },
            {
                "epoch": 1,
                "qwk_mean": 0.49,
                "recall_minus1": 0.80,
                "hedging_mean": 0.55,
                "calibration_global": 0.1,
            },
            {
                "epoch": 2,
                "qwk_mean": 0.49,
                "recall_minus1": 0.80,
                "hedging_mean": 0.55,
                "calibration_global": 0.7,
            },
        ]
    )

    selected = select_recall_window_policy(trace, qwk_window=0.02)

    assert selected["epoch"] == 2


def test_label_regime_uses_consensus_run_ids_when_yaml_lacks_label_path():
    run_data = {
        "metadata": {"run_id": "run_048"},
        "config": {"data": {}},
    }

    assert label_regime_for_run(run_data) == "consensus"
