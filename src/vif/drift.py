"""Uncertainty-gated drift detection on top of weekly VIF signals."""

from __future__ import annotations

import polars as pl

from src.coach.schemas import DriftDetectionResult
from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.evolution import EvolutionDetectionResult, classify_weekly_evolution


def _build_dimension_signals(
    target_row: dict,
    evolution_result: EvolutionDetectionResult,
    *,
    rut_dims: set[str],
    crash_dims: set[str],
    evolution_dims: set[str],
    uncertainty_threshold: float,
) -> list[DriftDetectionResult.DimensionSignal]:
    signals: list[DriftDetectionResult.DimensionSignal] = []
    evolution_by_dim = {row.dimension: row for row in evolution_result.dimensions}

    for dim in SCHWARTZ_VALUE_ORDER:
        evo_row = evolution_by_dim[dim]
        trigger = None
        if dim in crash_dims:
            trigger = "crash"
        elif dim in rut_dims:
            trigger = "rut"
        elif dim in evolution_dims:
            trigger = "evolution"
        elif float(target_row[f"uncertainty_{dim}"]) >= uncertainty_threshold:
            trigger = "high_uncertainty"

        signals.append(
            DriftDetectionResult.DimensionSignal(
                dimension=dim,
                classification=evo_row.classification,
                mean_alignment=float(target_row[f"alignment_{dim}"]),
                mean_uncertainty=float(target_row[f"uncertainty_{dim}"]),
                trigger=trigger,
                residual=evo_row.residual,
                volatility=evo_row.volatility,
            )
        )

    return signals


def detect_weekly_drift(
    weekly_df: pl.DataFrame,
    *,
    target_week_end: str | None = None,
    evolution_result: EvolutionDetectionResult | None = None,
    lookback_weeks: int = 6,
    min_rut_weeks: int = 3,
    rut_threshold: float = -0.4,
    crash_delta: float = 0.5,
    uncertainty_threshold: float = 0.3,
    min_profile_weight: float = 0.15,
    residual_threshold: float = 0.4,
    volatility_threshold: float = 0.5,
    blend_rate: float = 0.3,
    importance_floor: float = 0.15,
) -> DriftDetectionResult:
    """Resolve weekly coach mode from aggregated VIF signals."""
    if weekly_df.is_empty():
        raise ValueError("weekly_df must contain at least one weekly signal row")

    persona_ids = weekly_df["persona_id"].unique().to_list()
    if len(persona_ids) != 1:
        raise ValueError(f"detect_weekly_drift expects one persona, found {len(persona_ids)}")

    ordered = weekly_df.sort("week_end")
    if target_week_end is not None:
        ordered = ordered.filter(pl.col("week_end") <= target_week_end)
    if ordered.is_empty():
        raise ValueError("No weekly rows remain after applying target_week_end filter")

    target_row = ordered.row(-1, named=True)
    previous_row = ordered.row(-2, named=True) if ordered.height >= 2 else None

    if evolution_result is None:
        evolution_result = classify_weekly_evolution(
            ordered,
            target_week_end=target_row["week_end"],
            lookback_weeks=lookback_weeks,
            min_weeks=max(min_rut_weeks, 3),
            residual_threshold=residual_threshold,
            volatility_threshold=volatility_threshold,
            blend_rate=blend_rate,
            importance_floor=importance_floor,
        )

    overall_mean = float(target_row["overall_mean"])
    overall_uncertainty = float(target_row["overall_uncertainty"])

    if overall_uncertainty >= uncertainty_threshold:
        dimension_signals = _build_dimension_signals(
            target_row,
            evolution_result,
            rut_dims=set(),
            crash_dims=set(),
            evolution_dims=set(),
            uncertainty_threshold=uncertainty_threshold,
        )
        return DriftDetectionResult(
            response_mode="high_uncertainty",
            rationale=(
                "The target week carries elevated model uncertainty, so the Coach "
                "should avoid a confident drift critique."
            ),
            reasons=["overall_uncertainty_above_threshold"],
            source="drift_detector",
            trigger_type="high_uncertainty",
            week_start=str(target_row["week_start"]),
            week_end=str(target_row["week_end"]),
            overall_mean=overall_mean,
            overall_uncertainty=overall_uncertainty,
            triggered_dimensions=[
                signal.dimension
                for signal in dimension_signals
                if signal.trigger == "high_uncertainty"
            ],
            dimension_signals=dimension_signals,
        )

    profile_weights = {
        dim: float(target_row[f"profile_weight_{dim}"]) for dim in SCHWARTZ_VALUE_ORDER
    }
    evolution_dims = {
        row.dimension
        for row in evolution_result.dimensions
        if row.classification == "evolution" and row.weight >= min_profile_weight
    }

    rut_dims: set[str] = set()
    if ordered.height >= min_rut_weeks:
        recent = ordered.tail(min_rut_weeks)
        for dim in SCHWARTZ_VALUE_ORDER:
            if profile_weights[dim] < min_profile_weight or dim in evolution_dims:
                continue
            recent_scores = recent[f"alignment_{dim}"].to_list()
            recent_uncertainty = recent[f"uncertainty_{dim}"].to_list()
            if all(
                float(score) < rut_threshold and float(sig) < uncertainty_threshold
                for score, sig in zip(recent_scores, recent_uncertainty)
            ):
                rut_dims.add(dim)

    crash_dims: set[str] = set()
    if previous_row is not None:
        scalar_drop = float(previous_row["overall_mean"]) - overall_mean
        if scalar_drop > crash_delta:
            for dim in SCHWARTZ_VALUE_ORDER:
                if profile_weights[dim] < min_profile_weight or dim in evolution_dims:
                    continue
                dim_drop = float(previous_row[f"alignment_{dim}"]) - float(target_row[f"alignment_{dim}"])
                if dim_drop > 0:
                    crash_dims.add(dim)

    dimension_signals = _build_dimension_signals(
        target_row,
        evolution_result,
        rut_dims=rut_dims,
        crash_dims=crash_dims,
        evolution_dims=evolution_dims,
        uncertainty_threshold=uncertainty_threshold,
    )

    if crash_dims:
        return DriftDetectionResult(
            response_mode="crash",
            rationale=(
                "The profile-weighted weekly alignment dropped sharply compared with the "
                "previous week on important dimensions."
            ),
            reasons=["weekly_scalar_drop_above_threshold", *sorted(crash_dims)],
            source="drift_detector",
            trigger_type="crash",
            week_start=str(target_row["week_start"]),
            week_end=str(target_row["week_end"]),
            overall_mean=overall_mean,
            overall_uncertainty=overall_uncertainty,
            triggered_dimensions=sorted(crash_dims),
            dimension_signals=dimension_signals,
        )

    if rut_dims:
        return DriftDetectionResult(
            response_mode="rut",
            rationale=(
                "A declared core value stayed below the rut threshold for multiple "
                "low-uncertainty weeks."
            ),
            reasons=["consecutive_low_core_value_weeks", *sorted(rut_dims)],
            source="drift_detector",
            trigger_type="rut",
            week_start=str(target_row["week_start"]),
            week_end=str(target_row["week_end"]),
            overall_mean=overall_mean,
            overall_uncertainty=overall_uncertainty,
            triggered_dimensions=sorted(rut_dims),
            dimension_signals=dimension_signals,
        )

    if evolution_dims:
        profile_update = None
        if evolution_result.suggested_profile is not None:
            profile_update = evolution_result.suggested_profile.suggested_profile
        return DriftDetectionResult(
            response_mode="evolution",
            rationale=(
                "Recent divergence looks sustained and low-volatility, which suggests "
                "a genuine shift in priorities rather than behavioral drift."
            ),
            reasons=["low_volatility_directional_shift", *sorted(evolution_dims)],
            source="drift_detector",
            trigger_type="evolution",
            week_start=str(target_row["week_start"]),
            week_end=str(target_row["week_end"]),
            overall_mean=overall_mean,
            overall_uncertainty=overall_uncertainty,
            triggered_dimensions=sorted(evolution_dims),
            dimension_signals=dimension_signals,
            profile_update=profile_update,
        )

    strongest_dims = sorted(
        SCHWARTZ_VALUE_ORDER,
        key=lambda dim: (abs(float(target_row[f"alignment_{dim}"])), profile_weights[dim]),
        reverse=True,
    )
    return DriftDetectionResult(
        response_mode="stable",
        rationale=(
            "No uncertainty-gated crash, rut, or evolution trigger fired for the "
            "target week."
        ),
        reasons=["no_trigger_fired"],
        source="drift_detector",
        trigger_type="stable",
        week_start=str(target_row["week_start"]),
        week_end=str(target_row["week_end"]),
        overall_mean=overall_mean,
        overall_uncertainty=overall_uncertainty,
        triggered_dimensions=strongest_dims[:2],
        dimension_signals=dimension_signals,
    )
