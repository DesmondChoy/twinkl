"""Tests for per-epoch training trace parquet helpers."""

import polars as pl

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.training_traces import (
    build_dimension_weight_trace_frame,
    build_selection_trace_frame,
)


def test_build_selection_trace_frame_includes_train_loss_and_lr():
    candidate_trace = [
        {
            "epoch": 0,
            "val_loss": 1.2,
            "qwk_mean": 0.31,
            "recall_minus1": 0.22,
            "calibration_global": 0.18,
            "hedging_mean": 0.41,
            "qwk_nan_dims_count": 0,
            "eligible": True,
            "ineligible_reasons": [],
        }
    ]
    history = {
        "train_loss": [1.0],
        "lr": [0.0015],
    }

    trace_df = build_selection_trace_frame(candidate_trace, history)

    assert trace_df.shape == (1, 11)
    assert trace_df.row(0, named=True) == {
        "epoch": 0,
        "train_loss": 1.0,
        "val_loss": 1.2,
        "lr": 0.0015,
        "qwk_mean": 0.31,
        "recall_minus1": 0.22,
        "calibration_global": 0.18,
        "hedging_mean": 0.41,
        "qwk_nan_dims_count": 0,
        "eligible": True,
        "ineligible_reasons": "",
    }


def test_build_selection_trace_frame_empty_schema_is_stable():
    trace_df = build_selection_trace_frame([], {})

    assert trace_df.schema == {
        "epoch": pl.Int64,
        "train_loss": pl.Float64,
        "val_loss": pl.Float64,
        "lr": pl.Float64,
        "qwk_mean": pl.Float64,
        "recall_minus1": pl.Float64,
        "calibration_global": pl.Float64,
        "hedging_mean": pl.Float64,
        "qwk_nan_dims_count": pl.Int64,
        "eligible": pl.Boolean,
        "ineligible_reasons": pl.Utf8,
    }


def test_build_dimension_weight_trace_frame_marks_selected_epoch():
    epoch_diagnostics = [
        {
            "epoch": 0,
            "train_ce_mean": [0.1] * 10,
            "train_ce_ema": [0.1] * 10,
            "applied_weight": [1.0] * 10,
            "val_qwk_per_dim": {dim_name: 0.2 for dim_name in SCHWARTZ_VALUE_ORDER},
            "val_accuracy_per_dim": {dim_name: 0.7 for dim_name in SCHWARTZ_VALUE_ORDER},
            "val_hedging_per_dim": {dim_name: 0.5 for dim_name in SCHWARTZ_VALUE_ORDER},
            "val_recall_per_class_per_dim": {
                dim_name: {"minus1": 0.1, "zero": 0.8, "plus1": 0.2}
                for dim_name in SCHWARTZ_VALUE_ORDER
            },
            "eligible": False,
        },
        {
            "epoch": 1,
            "train_ce_mean": [0.2] * 10,
            "train_ce_ema": [0.15] * 10,
            "applied_weight": [1.1] * 10,
            "val_qwk_per_dim": {dim_name: 0.3 for dim_name in SCHWARTZ_VALUE_ORDER},
            "val_accuracy_per_dim": {dim_name: 0.75 for dim_name in SCHWARTZ_VALUE_ORDER},
            "val_hedging_per_dim": {dim_name: 0.45 for dim_name in SCHWARTZ_VALUE_ORDER},
            "val_recall_per_class_per_dim": {
                dim_name: {"minus1": 0.2, "zero": 0.75, "plus1": 0.3}
                for dim_name in SCHWARTZ_VALUE_ORDER
            },
            "eligible": True,
        },
    ]

    trace_df = build_dimension_weight_trace_frame(epoch_diagnostics, selected_epoch=1)

    assert trace_df.shape == (20, 13)
    epoch_zero_rows = trace_df.filter(pl.col("epoch") == 0)
    epoch_one_rows = trace_df.filter(pl.col("epoch") == 1)

    assert epoch_zero_rows.select("selected").to_series().to_list() == [False] * 10
    assert epoch_one_rows.select("selected").to_series().to_list() == [True] * 10
    assert epoch_zero_rows.select("eligible").to_series().to_list() == [False] * 10
    assert epoch_one_rows.select("eligible").to_series().to_list() == [True] * 10
    assert epoch_one_rows.select("applied_weight").to_series().to_list() == [1.1] * 10
