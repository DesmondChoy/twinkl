"""Learning-rate finder helpers for VIF critic training.

This module wraps ``torch-lr-finder`` behind a small API that:
1. Runs a range test with clean model/optimizer reset semantics.
2. Derives ``lr_steep`` and ``lr_valley`` style suggestions.
3. Persists plot/history artifacts for reproducibility and notebook display.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch_lr_finder import LRFinder


def resolve_lr_finder_paths(
    output_dir: Path,
    output_plot_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve default or user-provided artifact paths.

    Args:
        output_dir: Training checkpoint directory.
        output_plot_path: Optional override for LR finder plot path.

    Returns:
        Tuple of ``(plot_path, history_path)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_plot_path is None:
        plot_path = output_dir / "lr_find_loss_vs_lr.png"
        history_path = output_dir / "lr_find_history.json"
    else:
        plot_path = Path(output_plot_path)
        history_path = plot_path.with_suffix(".json")
        plot_path.parent.mkdir(parents=True, exist_ok=True)

    return plot_path, history_path


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Apply centered moving average with edge-value padding."""
    if window <= 1 or values.size < window:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def compute_lr_suggestions(
    learning_rates: list[float] | np.ndarray,
    losses: list[float] | np.ndarray,
    valley_tail_exclusion: float = 0.10,
    valley_tail_guard: float = 0.05,
    monotonic_drop_threshold: float = 0.90,
) -> dict:
    """Compute ``lr_steep`` and ``lr_valley`` from LR range-test history.

    ``lr_steep`` is the learning rate at the steepest downward slope on
    smoothed loss-vs-log10(lr). ``lr_valley`` is derived from the smoothed
    minimum (divided by 10 and clipped to observed LR range), with safety
    fallbacks:
    1. ignore the final ``valley_tail_exclusion`` region when searching
    2. if minimum still lands in final ``valley_tail_guard`` region, use
       ``lr_steep`` instead
    3. if the curve is mostly monotonic decreasing (>= threshold), use
       ``lr_steep`` instead
    """
    lr_arr = np.asarray(learning_rates, dtype=np.float64)
    loss_arr = np.asarray(losses, dtype=np.float64)

    valid_mask = np.isfinite(lr_arr) & np.isfinite(loss_arr) & (lr_arr > 0)
    lr_valid = lr_arr[valid_mask]
    loss_valid = loss_arr[valid_mask]

    if lr_valid.size < 3:
        return {
            "lr_steep": None,
            "lr_valley": None,
            "n_valid_points": int(lr_valid.size),
            "valley_idx": None,
            "valley_strategy": None,
            "valley_in_tail": None,
            "smoothed_drop_fraction": None,
            "is_mostly_monotonic": None,
        }

    smoothing_window = max(3, min(11, int(lr_valid.size * 0.1)))
    if smoothing_window % 2 == 0:
        smoothing_window += 1

    smoothed_loss = _moving_average(loss_valid, smoothing_window)
    log_lr = np.log10(lr_valid)

    slopes = np.gradient(smoothed_loss, log_lr)
    n_points = int(lr_valid.size)

    # Ignore extreme LR edges when finding steepest descent; endpoints are
    # often noisy and can force unrealistically tiny/large picks.
    slope_edge_guard = int(np.floor(n_points * 0.05))
    slope_edge_guard = min(max(slope_edge_guard, 1), max(n_points - 2, 1))
    slope_start = slope_edge_guard
    slope_end = max(slope_start + 1, n_points - slope_edge_guard)
    slope_slice = slice(slope_start, slope_end)
    steep_idx = int(np.argmin(slopes[slope_slice]) + slope_start)

    search_end = int(np.floor(n_points * (1.0 - valley_tail_exclusion)))
    search_end = min(max(search_end, 3), n_points)
    search_slice = slice(0, search_end)
    valley_idx = int(np.argmin(smoothed_loss[search_slice]))

    tail_start = int(np.floor(n_points * (1.0 - valley_tail_guard)))
    valley_in_tail = valley_idx >= tail_start

    lr_steep = float(lr_valid[steep_idx])
    smoothed_drop_fraction = float(np.mean(np.diff(smoothed_loss) < 0.0))
    is_mostly_monotonic = smoothed_drop_fraction >= monotonic_drop_threshold

    if valley_in_tail or is_mostly_monotonic:
        lr_valley = lr_steep
        valley_strategy = "fallback_lr_steep"
    else:
        lr_valley_raw = float(lr_valid[valley_idx] / 10.0)
        lr_valley = float(np.clip(lr_valley_raw, lr_valid.min(), lr_valid.max()))
        valley_strategy = "valley_over_10"

    return {
        "lr_steep": lr_steep,
        "lr_valley": lr_valley,
        "n_valid_points": n_points,
        "valley_idx": valley_idx,
        "valley_strategy": valley_strategy,
        "valley_in_tail": valley_in_tail,
        "smoothed_drop_fraction": smoothed_drop_fraction,
        "is_mostly_monotonic": is_mostly_monotonic,
    }


def _save_lr_history(
    history_path: Path,
    learning_rates: list[float],
    losses: list[float],
    suggestions: dict,
) -> None:
    payload = {
        "learning_rate": learning_rates,
        "loss": losses,
        "suggestions": suggestions,
    }
    with open(history_path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_lr_plot(
    plot_path: Path,
    learning_rates: list[float],
    losses: list[float],
    suggestions: dict,
) -> None:
    lr_arr = np.asarray(learning_rates, dtype=np.float64)
    loss_arr = np.asarray(losses, dtype=np.float64)
    mask = np.isfinite(lr_arr) & np.isfinite(loss_arr) & (lr_arr > 0)
    lr_arr = lr_arr[mask]
    loss_arr = loss_arr[mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    if lr_arr.size == 0:
        ax.text(
            0.5,
            0.5,
            "No valid LR finder points recorded.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        ax.plot(lr_arr, loss_arr, color="#1f77b4", linewidth=1.8)
        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss")
        ax.set_title("LR Finder: Loss vs Learning Rate")
        ax.grid(alpha=0.25)

        lr_steep = suggestions.get("lr_steep")
        if lr_steep is not None:
            ax.axvline(lr_steep, color="#ff7f0e", linestyle="--", label=f"lr_steep={lr_steep:.2e}")

        lr_valley = suggestions.get("lr_valley")
        if lr_valley is not None:
            ax.axvline(lr_valley, color="#2ca02c", linestyle="--", label=f"lr_valley={lr_valley:.2e}")

        if lr_steep is not None or lr_valley is not None:
            ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def run_lr_finder(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    configured_learning_rate: float,
    weight_decay: float,
    device: str,
    output_dir: Path,
    output_plot_path: str | Path | None = None,
    start_lr: float | None = None,
    end_lr: float | None = None,
    num_iter: int | None = None,
    max_selected_lr: float | None = None,
) -> dict:
    """Run LR finder and return metadata for training/logging.

    The selected learning rate follows ``lr_valley`` by default, with fallback
    to ``configured_learning_rate`` when suggestions are unavailable.
    """
    if configured_learning_rate is None:
        raise ValueError("configured_learning_rate must be provided")
    if start_lr is None or end_lr is None or num_iter is None:
        raise ValueError("start_lr, end_lr, and num_iter must all be provided")

    start_lr = float(start_lr)
    end_lr = float(end_lr)
    num_iter = int(num_iter)
    configured_learning_rate = float(configured_learning_rate)

    plot_path, history_path = resolve_lr_finder_paths(output_dir, output_plot_path)
    lr_finder = None
    error_message: str | None = None
    learning_rates: list[float] = []
    losses: list[float] = []

    optimizer = AdamW(
        model.parameters(),
        lr=configured_learning_rate,
        weight_decay=weight_decay,
    )

    try:
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(
            train_loader,
            start_lr=start_lr,
            end_lr=end_lr,
            num_iter=num_iter,
            step_mode="exp",
        )
        history = lr_finder.history or {}
        learning_rates = [float(x) for x in history.get("lr", [])]
        losses = [float(x) for x in history.get("loss", [])]
    except Exception as exc:  # pragma: no cover - exercised in integration usage
        error_message = str(exc)
    finally:
        if lr_finder is not None:
            try:
                lr_finder.reset()
            except Exception:
                # Keep workflow non-fatal; fallback path handles missing suggestions.
                pass

    suggestions = compute_lr_suggestions(learning_rates, losses)
    lr_valley = suggestions.get("lr_valley")
    if lr_valley is None:
        selected_lr = float(configured_learning_rate)
        fallback_reason = "lr_valley_unavailable"
        selected_lr_source = "configured_learning_rate"
    else:
        selected_lr = float(lr_valley)
        fallback_reason = None
        selected_lr_source = "lr_valley"

    if error_message and not learning_rates:
        fallback_reason = f"lr_finder_error: {error_message}"

    if max_selected_lr is not None:
        max_selected_lr = float(max_selected_lr)
        if selected_lr > max_selected_lr:
            selected_lr = max_selected_lr
            selected_lr_source = "max_selected_lr_cap"
            if fallback_reason:
                fallback_reason = f"{fallback_reason}; clipped_to_max_selected_lr"
            else:
                fallback_reason = "clipped_to_max_selected_lr"

    _save_lr_history(history_path, learning_rates, losses, suggestions)
    _save_lr_plot(plot_path, learning_rates, losses, suggestions)

    return {
        "enabled": True,
        "params": {
            "start_lr": start_lr,
            "end_lr": end_lr,
            "num_iter": num_iter,
            "max_selected_lr": float(max_selected_lr) if max_selected_lr is not None else None,
        },
        "suggestions": {
            "lr_steep": suggestions.get("lr_steep"),
            "lr_valley": suggestions.get("lr_valley"),
            "valley_strategy": suggestions.get("valley_strategy"),
            "valley_in_tail": suggestions.get("valley_in_tail"),
            "smoothed_drop_fraction": suggestions.get("smoothed_drop_fraction"),
            "is_mostly_monotonic": suggestions.get("is_mostly_monotonic"),
        },
        "lr_selected": selected_lr,
        "lr_selected_source": selected_lr_source,
        "configured_learning_rate": configured_learning_rate,
        "fallback_reason": fallback_reason,
        "history_points": len(learning_rates),
        "artifacts": {
            "plot_path": str(plot_path),
            "history_path": str(history_path),
        },
    }
