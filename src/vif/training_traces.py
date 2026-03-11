"""Helpers for serializing per-epoch VIF training diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from src.models.judge import SCHWARTZ_VALUE_ORDER


_SELECTION_TRACE_SCHEMA = {
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

_DIMENSION_WEIGHT_TRACE_SCHEMA = {
    "epoch": pl.Int64,
    "dimension": pl.Utf8,
    "train_ce_mean": pl.Float64,
    "train_ce_ema": pl.Float64,
    "applied_weight": pl.Float64,
    "val_qwk": pl.Float64,
    "val_accuracy": pl.Float64,
    "val_hedging": pl.Float64,
    "val_recall_minus1": pl.Float64,
    "val_recall_zero": pl.Float64,
    "val_recall_plus1": pl.Float64,
    "eligible": pl.Boolean,
    "selected": pl.Boolean,
}


def _history_value(values: Sequence[float] | None, idx: int) -> float:
    if values is None or idx >= len(values):
        return float("nan")
    return float(values[idx])


def _vector_by_dimension(
    values: Sequence[float],
    *,
    field_name: str,
) -> dict[str, float]:
    if len(values) != len(SCHWARTZ_VALUE_ORDER):
        raise ValueError(
            f"{field_name} must have length {len(SCHWARTZ_VALUE_ORDER)}, got {len(values)}"
        )
    return {
        dim_name: float(values[dim_idx])
        for dim_idx, dim_name in enumerate(SCHWARTZ_VALUE_ORDER)
    }


def _mapping_by_dimension(
    values: Mapping[str, float],
    *,
    field_name: str,
) -> dict[str, float]:
    missing = [dim_name for dim_name in SCHWARTZ_VALUE_ORDER if dim_name not in values]
    if missing:
        raise ValueError(f"{field_name} missing dimensions: {missing}")
    return {
        dim_name: float(values[dim_name])
        for dim_name in SCHWARTZ_VALUE_ORDER
    }


def build_selection_trace_frame(candidate_trace: list[dict], history: dict | None = None) -> pl.DataFrame:
    """Build the per-epoch checkpoint selection trace parquet."""
    history = history or {}
    train_losses = history.get("train_loss")
    learning_rates = history.get("lr") or history.get("learning_rate")

    rows = [
        {
            "epoch": int(candidate["epoch"]),
            "train_loss": _history_value(train_losses, idx),
            "val_loss": float(candidate["val_loss"]),
            "lr": _history_value(learning_rates, idx),
            "qwk_mean": float(candidate["qwk_mean"]),
            "recall_minus1": float(candidate["recall_minus1"]),
            "calibration_global": float(candidate["calibration_global"]),
            "hedging_mean": float(candidate["hedging_mean"]),
            "qwk_nan_dims_count": int(candidate["qwk_nan_dims_count"]),
            "eligible": bool(candidate["eligible"]),
            "ineligible_reasons": ",".join(candidate["ineligible_reasons"]),
        }
        for idx, candidate in enumerate(candidate_trace)
    ]
    if rows:
        return pl.DataFrame(rows)
    return pl.DataFrame(schema=_SELECTION_TRACE_SCHEMA)


def build_dimension_weight_trace_frame(
    epoch_diagnostics: list[dict],
    *,
    selected_epoch: int | None = None,
) -> pl.DataFrame:
    """Build the per-epoch, per-dimension weighting trace parquet."""
    rows = []
    for epoch_diag in epoch_diagnostics:
        epoch = int(epoch_diag["epoch"])
        train_ce_mean = _vector_by_dimension(
            epoch_diag["train_ce_mean"],
            field_name="train_ce_mean",
        )
        train_ce_ema = _vector_by_dimension(
            epoch_diag["train_ce_ema"],
            field_name="train_ce_ema",
        )
        applied_weight = _vector_by_dimension(
            epoch_diag["applied_weight"],
            field_name="applied_weight",
        )
        val_qwk = _mapping_by_dimension(
            epoch_diag["val_qwk_per_dim"],
            field_name="val_qwk_per_dim",
        )
        val_accuracy = _mapping_by_dimension(
            epoch_diag["val_accuracy_per_dim"],
            field_name="val_accuracy_per_dim",
        )
        val_hedging = _mapping_by_dimension(
            epoch_diag["val_hedging_per_dim"],
            field_name="val_hedging_per_dim",
        )
        recall_per_dim = epoch_diag["val_recall_per_class_per_dim"]
        missing_recall = [
            dim_name for dim_name in SCHWARTZ_VALUE_ORDER if dim_name not in recall_per_dim
        ]
        if missing_recall:
            raise ValueError(
                "val_recall_per_class_per_dim missing dimensions: "
                f"{missing_recall}"
            )
        eligible = bool(epoch_diag["eligible"])
        selected = selected_epoch is not None and epoch == int(selected_epoch)

        for dim_name in SCHWARTZ_VALUE_ORDER:
            dim_recall = recall_per_dim[dim_name]
            rows.append(
                {
                    "epoch": epoch,
                    "dimension": dim_name,
                    "train_ce_mean": train_ce_mean[dim_name],
                    "train_ce_ema": train_ce_ema[dim_name],
                    "applied_weight": applied_weight[dim_name],
                    "val_qwk": val_qwk[dim_name],
                    "val_accuracy": val_accuracy[dim_name],
                    "val_hedging": val_hedging[dim_name],
                    "val_recall_minus1": float(dim_recall["minus1"]),
                    "val_recall_zero": float(dim_recall["zero"]),
                    "val_recall_plus1": float(dim_recall["plus1"]),
                    "eligible": eligible,
                    "selected": selected,
                }
            )

    if rows:
        return pl.DataFrame(rows)
    return pl.DataFrame(schema=_DIMENSION_WEIGHT_TRACE_SCHEMA)
