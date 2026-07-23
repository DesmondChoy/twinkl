"""Live nudge decision and generation for the demo app's real user journal.

Reuses src.nudge.decision and src.nudge.generation exactly as built for the
synthetic persona pipeline — same prompts, same word-count contract. This
module deliberately calls only decide_nudge() and generate_nudge_text():
src.nudge.generation.generate_nudge_response() fabricates a synthetic
persona's reply and must never be used to write words into a real user's own
response.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.nudge.decision import (
    decide_nudge,
    format_previous_entries,
    should_suppress_nudge,
)
from src.nudge.generation import LLMCompleteFn, generate_nudge_text

NUDGE_CONFIG_PATH = Path("config/synthetic_data.yaml")
_DEFAULT_NUDGE_CONFIG = {"min_words": 2, "max_words": 12}

_nudge_config: dict[str, Any] | None = None


def _get_nudge_config(path: Path = NUDGE_CONFIG_PATH) -> dict[str, Any]:
    """Load {"nudge": {...}} from the synthetic-data config, cached.

    Only the nudge subsection is used; the rest of that config (persona
    generation, value labels, etc.) is out of scope here.
    """
    global _nudge_config
    if _nudge_config is None:
        try:
            with path.open() as handle:
                full_config = yaml.safe_load(handle) or {}
            nudge_section = full_config.get("nudge") or _DEFAULT_NUDGE_CONFIG
        except FileNotFoundError:
            nudge_section = _DEFAULT_NUDGE_CONFIG
        _nudge_config = {"nudge": nudge_section}
    return _nudge_config


async def check_for_nudge(
    *,
    entry_text: str,
    entry_date: str,
    previous_entries: list[dict[str, Any]],
    llm_complete: LLMCompleteFn,
) -> str | None:
    """Decide whether this entry warrants a nudge, and if so, generate it.

    previous_entries are journal entry dicts (date/initial_entry/nudge_text/
    response_text, oldest first); this sanitizes them to {date, content}
    before they ever reach a prompt, and applies the same anti-annoyance cap
    (should_suppress_nudge) used in synthetic generation before spending a
    call on the LLM decision.

    Returns the nudge question text, or None if no nudge should be shown
    (including on any decision/generation failure — nudging always degrades
    to "no nudge" rather than blocking the entry from saving).
    """
    previous_had_nudge = [bool(entry.get("nudge_text")) for entry in previous_entries]
    if should_suppress_nudge(previous_had_nudge):
        return None

    sanitized_previous = format_previous_entries(
        [
            {"date": entry["date"], "content": entry.get("initial_entry") or ""}
            for entry in previous_entries
        ]
    )

    should_nudge, category, _reason = await decide_nudge(
        entry_text, entry_date, sanitized_previous, llm_complete
    )
    if not should_nudge or category is None:
        return None

    nudge_text, _prompt = await generate_nudge_text(
        entry_text,
        entry_date,
        category,
        sanitized_previous,
        _get_nudge_config(),
        llm_complete,
    )
    return nudge_text
