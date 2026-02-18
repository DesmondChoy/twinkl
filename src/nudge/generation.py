"""Nudge text and response generation.

Generates the actual nudge text and optional persona responses using
injected LLM callables. Keeps generation concerns separate from
decision logic.
"""

import json
import random
from typing import Awaitable, Callable

from prompts import nudge_generation_prompt, nudge_response_prompt
from src.models.nudge import JournalTurn, NudgeCategory
from src.nudge.schemas import NUDGE_RESPONSE_FORMAT, NUDGE_RESPONSE_RESPONSE_FORMAT

LLMCompleteFn = Callable[[str, dict | None], Awaitable[str | None]]


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def weighted_choice(weights: dict[str, float]) -> str:
    """Make a weighted random choice from a dict of {option: weight}."""
    if not weights:
        raise ValueError("weights must contain at least one option")

    options = list(weights.keys())
    try:
        probs = [float(weight) for weight in weights.values()]
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be numeric") from exc

    if any(weight < 0 for weight in probs):
        raise ValueError("weights must be non-negative")

    total = sum(probs)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")

    probs = [p / total for p in probs]
    return random.choices(options, weights=probs, k=1)[0]


def select_response_mode(config: dict) -> str:
    """Select a response mode based on configured weights."""
    modes = config.get("nudge", {}).get("response_modes", [])
    if not modes:
        return "Answering directly"

    weights = {
        mode_cfg["mode"]: mode_cfg.get("weight", 0.0)
        for mode_cfg in modes
        if "mode" in mode_cfg
    }
    if not weights:
        return "Answering directly"

    try:
        return weighted_choice(weights)
    except ValueError:
        if "Answering directly" in weights:
            return "Answering directly"
        return next(iter(weights))


def _safe_load_json_object(raw_json: str) -> dict | None:
    """Parse a JSON object safely, returning None on malformed payloads."""
    try:
        data = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


async def generate_nudge_text(
    entry_content: str,
    entry_date: str,
    category: NudgeCategory,
    previous_entries: list[dict] | None,
    config: dict,
    llm_complete: LLMCompleteFn,
    max_attempts: int = 2,
) -> tuple[str | None, str]:
    """Generate nudge text for the given entry and category.

    Args:
        entry_content: The journal entry text.
        entry_date: The entry date string.
        category: The nudge category to generate for.
        previous_entries: Sanitized list of {date, content} dicts.
        config: Full config dict (reads nudge.min_words/max_words).
        llm_complete: Injected async LLM callable.
        max_attempts: Max generation retries for word-count validation.

    Returns:
        Tuple of (nudge_text or None, prompt used).
    """
    nudge_config = config["nudge"]

    prompt = nudge_generation_prompt.render(
        entry_content=entry_content,
        entry_date=entry_date,
        nudge_category=category,
        previous_entries=previous_entries,
        min_words=nudge_config["min_words"],
        max_words=nudge_config["max_words"],
    )

    for _ in range(max_attempts):
        raw_json = await llm_complete(prompt, NUDGE_RESPONSE_FORMAT)
        if not raw_json:
            continue

        data = _safe_load_json_object(raw_json)
        if data is None:
            continue

        nudge_text_raw = data.get("nudge_text", "")
        if not isinstance(nudge_text_raw, str):
            continue

        nudge_text = nudge_text_raw.strip()

        word_count = count_words(nudge_text)
        if (
            word_count < nudge_config["min_words"]
            or word_count > nudge_config["max_words"]
        ):
            continue

        return nudge_text, prompt

    return None, prompt


async def generate_nudge_response(
    persona_name: str,
    persona_age: str,
    persona_profession: str,
    persona_culture: str,
    persona_bio: str,
    entry_content: str,
    entry_date: str,
    nudge_text: str,
    config: dict,
    llm_complete: LLMCompleteFn,
    max_attempts: int = 2,
) -> tuple[JournalTurn | None, str, str]:
    """Generate a persona's response to a nudge.

    Args:
        persona_name: Persona's name.
        persona_age: Persona's age range.
        persona_profession: Persona's profession.
        persona_culture: Persona's cultural background.
        persona_bio: Persona's biography.
        entry_content: The journal entry text.
        entry_date: The entry date string.
        nudge_text: The nudge text to respond to.
        config: Full config dict (reads nudge.response_modes).
        llm_complete: Injected async LLM callable.
        max_attempts: Max generation retries.

    Returns:
        Tuple of (JournalTurn or None, prompt used, response_mode).
    """
    response_mode = select_response_mode(config)

    if response_mode == "Deflecting/redirecting":
        min_words, max_words = 5, 30
    elif response_mode == "Revealing deeper thought":
        min_words, max_words = 20, 80
    else:  # Answering directly
        min_words, max_words = 15, 60

    prompt = nudge_response_prompt.render(
        name=persona_name,
        age=persona_age,
        profession=persona_profession,
        culture=persona_culture,
        bio=persona_bio,
        entry_content=entry_content,
        nudge_text=nudge_text,
        response_mode=response_mode,
        min_words=min_words,
        max_words=max_words,
    )

    for _ in range(max_attempts):
        raw_json = await llm_complete(prompt, NUDGE_RESPONSE_RESPONSE_FORMAT)
        if not raw_json:
            continue

        data = _safe_load_json_object(raw_json)
        if data is None:
            continue

        content_raw = data.get("content", "")
        if not isinstance(content_raw, str):
            continue

        content = content_raw.strip()

        if content:
            turn = JournalTurn(
                date=entry_date,
                content=content,
                turn_type="nudge_response",
                responding_to_nudge=nudge_text,
            )
            return turn, prompt, response_mode

    return None, prompt, response_mode
