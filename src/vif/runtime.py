"""Runtime helpers for VIF inference over wrangled journal timelines.

This module bridges trained Critic checkpoints to Coach-facing runtime artifacts.
It rebuilds state vectors from wrangled journal history, runs MC uncertainty
inference, and emits per-entry plus per-week signal tables that downstream
drift/coach modules can consume.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.critic import CriticMLP
from src.vif.critic_bnn import CriticBNN
from src.vif.critic_ordinal import OrdinalCriticBase
from src.vif.encoders import create_encoder
from src.vif.state_encoder import StateEncoder
from src.vif.train import load_config as load_train_config
from src.wrangling.parse_wrangled_data import parse_wrangled_file


ALIGNMENT_COLUMNS = [f"alignment_{dim}" for dim in SCHWARTZ_VALUE_ORDER]
UNCERTAINTY_COLUMNS = [f"uncertainty_{dim}" for dim in SCHWARTZ_VALUE_ORDER]
PROFILE_WEIGHT_COLUMNS = [f"profile_weight_{dim}" for dim in SCHWARTZ_VALUE_ORDER]


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively update nested dicts without mutating the caller's inputs."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _checkpoint_to_runtime_overrides(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Extract runtime-relevant config hints from checkpoint metadata."""
    overrides: dict[str, Any] = {}

    training_config = checkpoint.get("training_config")
    if isinstance(training_config, dict):
        _deep_update(overrides, training_config)

    training_metadata = checkpoint.get("training_metadata")
    if isinstance(training_metadata, dict):
        encoder_overrides = {
            "model_name": training_metadata.get("encoder_model"),
            "truncate_dim": training_metadata.get("truncate_dim"),
            "text_prefix": training_metadata.get("text_prefix"),
            "prompt_name": training_metadata.get("prompt_name"),
            "prompt": training_metadata.get("prompt"),
        }
        state_overrides = {
            "window_size": training_metadata.get("window_size"),
        }
        mc_overrides = {
            "n_samples": training_metadata.get("mc_dropout_samples"),
        }

        encoder_clean = {k: v for k, v in encoder_overrides.items() if v is not None}
        state_clean = {k: v for k, v in state_overrides.items() if v is not None}
        mc_clean = {k: v for k, v in mc_overrides.items() if v is not None}

        if encoder_clean:
            overrides.setdefault("encoder", {})
            overrides["encoder"].update(encoder_clean)
        if state_clean:
            overrides.setdefault("state_encoder", {})
            overrides["state_encoder"].update(state_clean)
        if mc_clean:
            overrides.setdefault("mc_dropout", {})
            overrides["mc_dropout"].update(mc_clean)

    return overrides


def _resolve_runtime_config(
    checkpoint: dict[str, Any],
    config_path: str | Path | None = "config/vif.yaml",
) -> dict[str, Any]:
    """Resolve runtime config from repo defaults plus checkpoint metadata."""
    config = load_train_config(config_path)
    overrides = _checkpoint_to_runtime_overrides(checkpoint)
    if overrides:
        _deep_update(config, overrides)
    return config


def _instantiate_model(model_config: dict[str, Any]) -> torch.nn.Module:
    """Instantiate the appropriate Critic class from checkpoint config."""
    if "variant" in model_config:
        return OrdinalCriticBase.from_config(model_config)
    if any(key in model_config for key in ("prior_mean", "prior_variance", "posterior_rho_init")):
        return CriticBNN.from_config(model_config)
    return CriticMLP.from_config(model_config)


def load_runtime_bundle(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = "config/vif.yaml",
    device: str | None = None,
) -> tuple[torch.nn.Module, StateEncoder, dict[str, Any], dict[str, Any], str]:
    """Load a trained Critic plus the encoder/state config needed for runtime."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    model = _instantiate_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    runtime_config = _resolve_runtime_config(checkpoint, config_path=config_path)
    text_encoder = create_encoder(runtime_config["encoder"])
    state_encoder = StateEncoder(
        text_encoder,
        window_size=runtime_config["state_encoder"]["window_size"],
    )
    return model, state_encoder, runtime_config, checkpoint, resolved_device


def _load_persona_entries(
    persona_id: str,
    wrangled_dir: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load one persona's wrangled journal history."""
    wrangled_file = Path(wrangled_dir) / f"persona_{persona_id}.md"
    if not wrangled_file.exists():
        raise FileNotFoundError(f"Wrangled file not found: {wrangled_file}")

    profile, entries, _warnings = parse_wrangled_file(wrangled_file)
    entries = sorted(entries, key=lambda row: int(row["t_index"]))
    return profile, entries


def _build_state_matrix(
    entries: list[dict[str, Any]],
    *,
    state_encoder: StateEncoder,
    core_values: list[str],
) -> np.ndarray:
    """Construct one state vector per journal entry."""
    states: list[np.ndarray] = []
    for index, _entry in enumerate(entries):
        texts: list[str] = []
        dates: list[str] = []
        for offset in range(state_encoder.window_size):
            target_index = index - offset
            if target_index < 0:
                continue
            previous = entries[target_index]
            texts.append(
                state_encoder.concatenate_entry_text(
                    previous.get("initial_entry"),
                    previous.get("nudge_text"),
                    previous.get("response_text"),
                )
            )
            dates.append(previous["date"])
        states.append(
            state_encoder.build_state_vector(
                texts=texts,
                dates=dates,
                core_values=core_values,
            )
        )
    return np.stack(states).astype(np.float32)


def predict_persona_timeline(
    *,
    persona_id: str,
    checkpoint_path: str | Path,
    wrangled_dir: str | Path = "logs/wrangled",
    config_path: str | Path | None = "config/vif.yaml",
    n_mc_samples: int | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Predict per-entry VIF means and uncertainties for one persona timeline."""
    model, state_encoder, runtime_config, _checkpoint, resolved_device = load_runtime_bundle(
        checkpoint_path,
        config_path=config_path,
        device=device,
    )

    profile, entries = _load_persona_entries(persona_id, wrangled_dir)
    if not entries:
        raise ValueError(f"No journal entries found for persona_id={persona_id}")

    states = _build_state_matrix(
        entries,
        state_encoder=state_encoder,
        core_values=profile.get("core_values") or [],
    )
    state_tensor = torch.from_numpy(states).to(resolved_device)

    sample_count = int(
        n_mc_samples
        if n_mc_samples is not None
        else runtime_config.get("mc_dropout", {}).get("n_samples", 50)
    )

    mean_batches: list[np.ndarray] = []
    std_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(entries), batch_size):
            batch = state_tensor[start : start + batch_size]
            mean, std = model.predict_with_uncertainty(batch, n_samples=sample_count)
            mean_batches.append(mean.detach().cpu().numpy().astype(np.float32))
            std_batches.append(std.detach().cpu().numpy().astype(np.float32))

    mean_array = np.concatenate(mean_batches, axis=0)
    std_array = np.concatenate(std_batches, axis=0)
    profile_weights = state_encoder.parse_core_values_to_weights(profile.get("core_values") or [])

    rows: list[dict[str, Any]] = []
    for entry, mean_row, std_row in zip(entries, mean_array, std_array):
        row: dict[str, Any] = {
            "persona_id": persona_id,
            "persona_name": profile.get("name"),
            "date": entry["date"],
            "t_index": int(entry["t_index"]),
            "initial_entry": entry.get("initial_entry"),
            "nudge_text": entry.get("nudge_text"),
            "response_text": entry.get("response_text"),
            "has_nudge": bool(entry.get("has_nudge")),
            "has_response": bool(entry.get("has_response")),
            "core_values": profile.get("core_values") or [],
            "alignment_vector": mean_row.tolist(),
            "uncertainty_vector": std_row.tolist(),
            "overall_mean": float(np.dot(profile_weights, mean_row)),
            "overall_uncertainty": float(np.dot(profile_weights, std_row)),
        }
        for dim, mean_value, std_value, weight in zip(
            SCHWARTZ_VALUE_ORDER,
            mean_row.tolist(),
            std_row.tolist(),
            profile_weights.tolist(),
        ):
            row[f"alignment_{dim}"] = float(mean_value)
            row[f"uncertainty_{dim}"] = float(std_value)
            row[f"profile_weight_{dim}"] = float(weight)
        rows.append(row)

    return pl.DataFrame(rows), {
        "persona_id": persona_id,
        "persona_name": profile.get("name"),
        "core_values": profile.get("core_values") or [],
        "window_size": state_encoder.window_size,
        "n_mc_samples": sample_count,
        "device": resolved_device,
        "checkpoint_path": str(checkpoint_path),
    }


def aggregate_timeline_by_week(timeline_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-entry timeline predictions into weekly VIF signals."""
    if timeline_df.is_empty():
        raise ValueError("timeline_df must contain at least one row")

    annotated = timeline_df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("_entry_date")
    ).with_columns(
        pl.col("_entry_date").dt.truncate("1w").alias("_week_start")
    ).with_columns(
        (pl.col("_week_start") + timedelta(days=6)).alias("_week_end")
    )

    agg_exprs: list[pl.Expr] = [
        pl.col("persona_name").first().alias("persona_name"),
        pl.col("core_values").first().alias("core_values"),
        pl.len().alias("n_entries"),
        pl.col("overall_mean").mean().alias("overall_mean"),
        pl.col("overall_uncertainty").mean().alias("overall_uncertainty"),
    ]
    for col_name in ALIGNMENT_COLUMNS + UNCERTAINTY_COLUMNS + PROFILE_WEIGHT_COLUMNS:
        agg_exprs.append(pl.col(col_name).mean().alias(col_name))

    weekly = annotated.group_by(["persona_id", "_week_start", "_week_end"]).agg(
        agg_exprs
    ).sort(["persona_id", "_week_start"])

    weekly = weekly.with_columns(
        [
            pl.col("_week_start").dt.strftime("%Y-%m-%d").alias("week_start"),
            pl.col("_week_end").dt.strftime("%Y-%m-%d").alias("week_end"),
            pl.concat_list(ALIGNMENT_COLUMNS).alias("alignment_vector"),
            pl.concat_list(UNCERTAINTY_COLUMNS).alias("uncertainty_vector"),
        ]
    )

    return weekly.select(
        [
            "persona_id",
            "persona_name",
            "week_start",
            "week_end",
            "n_entries",
            "core_values",
            "alignment_vector",
            "uncertainty_vector",
            *ALIGNMENT_COLUMNS,
            *UNCERTAINTY_COLUMNS,
            *PROFILE_WEIGHT_COLUMNS,
            "overall_mean",
            "overall_uncertainty",
        ]
    )


def persist_runtime_artifacts(
    *,
    timeline_df: pl.DataFrame,
    output_dir: str | Path,
    weekly_df: pl.DataFrame | None = None,
    timeline_filename: str = "vif_timeline.parquet",
    weekly_filename: str = "vif_weekly.parquet",
) -> dict[str, str]:
    """Persist timeline and optional weekly signal tables to parquet files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timeline_path = output_path / timeline_filename
    timeline_df.write_parquet(timeline_path)

    artifact_paths = {"timeline_path": str(timeline_path)}
    if weekly_df is not None:
        weekly_path = output_path / weekly_filename
        weekly_df.write_parquet(weekly_path)
        artifact_paths["weekly_path"] = str(weekly_path)

    return artifact_paths
