"""Nudge text and response generation.

Generates the actual nudge text and optional persona responses using
injected LLM callables. Keeps generation concerns separate from
decision logic.
"""

import json
import random
from typing import Callable, Awaitable

from prompts import nudge_generation_prompt, nudge_response_prompt
from src.models.nudge import JournalTurn, NudgeCategory
from src.nudge.schemas import NUDGE_RESPONSE_FORMAT, NUDGE_RESPONSE_RESPONSE_FORMAT

LLMCompleteFn = Callable[[str, dict | None], Awaitable[str | None]]


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def weighted_choice(weights: dict[str, float]) -> str:
    """Make a weighted random choice from a dict of {option: weight}."""
    options = list(weights.keys())
    probs = list(weights.values())
    total = sum(probs)
    probs = [p / total for p in probs]
    return random.choices(options, weights=probs, k=1)[0]


def select_response_mode(config: dict) -> str:
    """Select a response mode based on configured weights."""
    modes = config["nudge"]["response_modes"]
    weights = {m["mode"]: m["weight"] for m in modes}
    return weighted_choice(weights)


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

        data = json.loads(raw_json)
        nudge_text = data.get("nudge_text", "").strip()

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

        data = json.loads(raw_json)
        content = data.get("content", "").strip()

        if content:
            turn = JournalTurn(
                date=entry_date,
                content=content,
                turn_type="nudge_response",
                responding_to_nudge=nudge_text,
            )
            return turn, prompt, response_mode

    return None, prompt, response_mode
