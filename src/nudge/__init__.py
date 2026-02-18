"""Nudge decision and generation pipeline.

Public API for determining whether to nudge a journal entry,
generating nudge text, and generating persona responses.
"""

from src.nudge.decision import (
    decide_nudge,
    format_previous_entries,
    should_suppress_nudge,
)
from src.nudge.generation import (
    count_words,
    generate_nudge_response,
    generate_nudge_text,
    select_response_mode,
    weighted_choice,
)
from src.nudge.schemas import (
    NUDGE_DECISION_RESPONSE_FORMAT,
    NUDGE_RESPONSE_FORMAT,
    NUDGE_RESPONSE_RESPONSE_FORMAT,
)

__all__ = [
    # decision
    "should_suppress_nudge",
    "format_previous_entries",
    "decide_nudge",
    # generation
    "count_words",
    "weighted_choice",
    "select_response_mode",
    "generate_nudge_text",
    "generate_nudge_response",
    # schemas
    "NUDGE_DECISION_RESPONSE_FORMAT",
    "NUDGE_RESPONSE_FORMAT",
    "NUDGE_RESPONSE_RESPONSE_FORMAT",
]
