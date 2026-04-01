"""Coach-layer package."""

from src.coach.mode_logic import WeeklyModeDecision, WeeklyModeSignals
from src.coach.schemas import (
    CoachNarrative,
    DriftDetectionResult,
    DigestValidation,
    DimensionDigest,
    EvidenceSnippet,
    JournalHistoryEntry,
    ValidationCheck,
    WeeklyDigest,
)


def run_weekly_coach_cycle(*args, **kwargs):
    """Lazily import the runtime bridge to avoid package cycles."""
    from src.coach.runtime import run_weekly_coach_cycle as _run_weekly_coach_cycle

    return _run_weekly_coach_cycle(*args, **kwargs)

__all__ = [
    "CoachNarrative",
    "DriftDetectionResult",
    "DigestValidation",
    "DimensionDigest",
    "EvidenceSnippet",
    "JournalHistoryEntry",
    "run_weekly_coach_cycle",
    "ValidationCheck",
    "WeeklyModeDecision",
    "WeeklyModeSignals",
    "WeeklyDigest",
]
