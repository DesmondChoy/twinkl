"""Fallback mode logic for weekly Coach digests.

This module keeps weekly response-mode heuristics isolated from the main
digest-building code so they can be evolved independently as the Coach
contract becomes more sophisticated.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.coach.schemas import CoachResponseMode, JournalHistoryEntry

ACUTE_DISTRESS_PATTERNS = (
    "died",
    "death",
    "dead",
    "grief",
    "grieving",
    "screamed",
    "crying",
    "stopped shaking",
    "unbearable",
    "funeral",
    "held her while she screamed",
)

BACKGROUND_STRAIN_PATTERNS = (
    "tired",
    "habit more than intention",
    "holding all the pieces",
    "too busy bracing",
    "quiet it will be",
    "memorizing",
    "guilty",
    "tight in my chest",
    "could not quite find it",
    "cold so i boiled water",
    "sat with her for an hour",
    "i am tired in a way",
)


@dataclass(frozen=True)
class WeeklyModeDecision:
    """Resolved weekly Coach mode plus short provenance metadata."""

    response_mode: CoachResponseMode
    mode_source: str
    mode_rationale: str


@dataclass(frozen=True)
class WeeklyModeSignals:
    """Signals used by the fallback mode router."""

    overall_mean: float
    top_tensions: list[str]
    top_strengths: list[str]
    core_values: list[str]
    window_entries: list[JournalHistoryEntry]
    has_mixed_core_polarity: bool


def has_acute_distress_context(window_entries: list[JournalHistoryEntry]) -> bool:
    """Detect highly sensitive weeks where value-specific critique is brittle."""
    combined = "\n".join(entry.content.lower() for entry in window_entries)
    return any(pattern in combined for pattern in ACUTE_DISTRESS_PATTERNS)


def has_background_strain_context(window_entries: list[JournalHistoryEntry]) -> bool:
    """Detect softer burden/strain weeks that are not cleanly stable."""
    combined = "\n".join(entry.content.lower() for entry in window_entries)
    return any(pattern in combined for pattern in BACKGROUND_STRAIN_PATTERNS)


def infer_response_mode(signals: WeeklyModeSignals) -> WeeklyModeDecision:
    """Assign a conservative fallback mode until drift detection is wired."""
    if has_acute_distress_context(signals.window_entries):
        return WeeklyModeDecision(
            response_mode="high_uncertainty",
            mode_source="fallback_heuristic",
            mode_rationale=(
                "Acute distress or grief markers were detected in the week, so the "
                "digest falls back to a presence-oriented mode instead of forcing a "
                "confident critique."
            ),
        )

    critical_core_tensions = [dim for dim in signals.top_tensions if dim in signals.core_values]
    if signals.overall_mean < -0.15 and critical_core_tensions:
        return WeeklyModeDecision(
            response_mode="rut",
            mode_source="fallback_heuristic",
            mode_rationale=(
                "Weekly aggregate is negative and a declared core value appears among "
                "the main tensions."
            ),
        )

    if (
        signals.top_strengths
        and (signals.top_tensions or signals.has_mixed_core_polarity)
        and -0.15 <= signals.overall_mean <= 0.2
    ):
        return WeeklyModeDecision(
            response_mode="mixed_state",
            mode_source="fallback_heuristic",
            mode_rationale=(
                "The week contains genuinely mixed signals, including both supportive "
                "and straining evidence, without collapsing into a single dominant direction."
            ),
        )

    if (
        not signals.top_tensions
        and signals.top_strengths
        and has_background_strain_context(signals.window_entries)
    ):
        return WeeklyModeDecision(
            response_mode="background_strain",
            mode_source="fallback_heuristic",
            mode_rationale=(
                "The week trends positive overall, but softer burden or transition "
                "signals suggest background strain rather than clean stability."
            ),
        )

    return WeeklyModeDecision(
        response_mode="stable",
        mode_source="fallback_heuristic",
        mode_rationale=(
            "Drift detection is not wired yet, so the digest defaults to a stable "
            "reflective mode unless there is clear weekly strain on core values."
        ),
    )
