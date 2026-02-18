"""Pydantic models for the nudge decision and generation pipeline.

These models define the data structures for nudge classification,
nudge text generation, and conversational journal turns.

Usage:
    from src.models.nudge import NudgeCategory, NudgeResult, JournalTurn

    result = NudgeResult(
        nudge_text="What made today feel off?",
        nudge_category="clarification",
        trigger_reason="Entry too vague to understand",
    )
"""

from pydantic import BaseModel, Field
from typing import Literal

NudgeCategory = Literal["clarification", "elaboration", "tension_surfacing"]

NUDGE_CATEGORIES: list[str] = ["clarification", "elaboration", "tension_surfacing"]


class NudgeResult(BaseModel):
    """Generated nudge with metadata."""

    nudge_text: str
    nudge_category: NudgeCategory
    trigger_reason: str
    was_responded_to: bool = False


class JournalTurn(BaseModel):
    """A single turn in the conversation (entry or response)."""

    date: str
    content: str
    turn_type: Literal["initial_entry", "nudge_response"]
    responding_to_nudge: str | None = None
