"""Core nudge decision logic.

Separates the deterministic anti-annoyance check from the LLM-based
semantic classification. The LLM dependency is injected as a callable
to keep this module testable without mocking external clients.
"""

import json
from typing import Callable, Awaitable

from prompts import nudge_decision_prompt
from src.models.nudge import NUDGE_CATEGORIES, NudgeCategory
from src.nudge.schemas import NUDGE_DECISION_RESPONSE_FORMAT

# Type alias for the injected LLM callable.
# Signature: (prompt: str, response_format: dict | None) -> str | None
LLMCompleteFn = Callable[[str, dict | None], Awaitable[str | None]]


def should_suppress_nudge(
    previous_entries_have_nudge: list[bool],
    window_size: int = 3,
    max_nudges: int = 2,
) -> bool:
    """Pure anti-annoyance check: cap nudges within a sliding window.

    Returns True if the nudge should be suppressed (too many recent nudges).

    Args:
        previous_entries_have_nudge: Boolean list where True means that
            entry had a nudge. Ordered oldest-to-newest.
        window_size: Number of recent entries to consider.
        max_nudges: Maximum allowed nudges within the window.
    """
    recent = previous_entries_have_nudge[-window_size:]
    return sum(recent) >= max_nudges


def format_previous_entries(
    entries: list[dict] | None,
    max_entries: int = 3,
) -> list[dict] | None:
    """Format previous entries for LLM context, stripping synthetic metadata.

    Only passes `date` and `content` to the LLM prompt — never tone,
    verbosity, reflection_mode, or other generation metadata. This
    enforces the no-metadata-leakage invariant.

    Args:
        entries: List of dicts that may contain arbitrary keys.
        max_entries: Maximum number of entries to include.

    Returns:
        Sanitized list of {date, content} dicts, or None if empty.
    """
    if not entries:
        return None

    sanitized = [
        {"date": e["date"], "content": e["content"]}
        for e in entries[-max_entries:]
    ]
    return sanitized if sanitized else None


def _normalize_category(raw_decision: str) -> NudgeCategory | None:
    """Normalize LLM output to a valid NudgeCategory.

    Handles case insensitivity, whitespace, and invalid values.
    Returns None for "no_nudge" or unrecognized values.
    """
    cleaned = raw_decision.strip().lower().replace(" ", "_")
    if cleaned == "no_nudge":
        return None
    if cleaned in NUDGE_CATEGORIES:
        return cleaned  # type: ignore[return-value]
    return None


async def decide_nudge(
    entry_content: str,
    entry_date: str,
    previous_entries: list[dict] | None,
    llm_complete: LLMCompleteFn,
) -> tuple[bool, NudgeCategory | None, str | None]:
    """LLM-based nudge decision.

    Calls the LLM to classify whether the entry warrants a nudge
    and which category it falls into.

    Args:
        entry_content: The journal entry text.
        entry_date: The entry date string.
        previous_entries: Sanitized list of {date, content} dicts.
        llm_complete: Injected async LLM callable.

    Returns:
        Tuple of (should_nudge, nudge_category, trigger_reason).
    """
    prompt = nudge_decision_prompt.render(
        entry_content=entry_content,
        entry_date=entry_date,
        previous_entries=previous_entries,
    )

    raw_json = await llm_complete(prompt, NUDGE_DECISION_RESPONSE_FORMAT)
    if not raw_json:
        return False, None, None

    data = json.loads(raw_json)
    decision = data.get("decision", "no_nudge")
    reason = data.get("reason", "")

    category = _normalize_category(decision)
    if category is None:
        return False, None, None

    return True, category, reason
