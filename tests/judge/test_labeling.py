"""Tests for judge labeling helpers."""

from __future__ import annotations

import pytest

from src.judge.labeling import (
    SCHWARTZ_VALUE_DISPLAY,
    build_value_rubric_context,
    build_session_content,
    judge_session,
    load_schwartz_values,
    render_judge_prompt,
)
from src.models.judge import SCHWARTZ_VALUE_ORDER


def test_build_value_rubric_context_contains_expected_sections():
    schwartz_config = load_schwartz_values("config/schwartz_values.yaml")
    rubric = build_value_rubric_context(schwartz_config)

    for key in SCHWARTZ_VALUE_ORDER:
        assert f"### {SCHWARTZ_VALUE_DISPLAY[key]}" in rubric

    assert rubric.count("### ") == len(SCHWARTZ_VALUE_ORDER)
    assert rubric.count("**Core Motivation:**") == len(SCHWARTZ_VALUE_ORDER)
    assert rubric.count("**Key Behaviors (Aligned):**") == len(SCHWARTZ_VALUE_ORDER)
    assert rubric.count("**Key Behaviors (Misaligned):**") == len(SCHWARTZ_VALUE_ORDER)


def test_build_session_content_matches_wrangled_format():
    rendered = build_session_content(
        "I stayed late to finish the launch deck.",
        "What made you keep going?",
        "I did not want to hand it off.",
    )

    assert rendered == (
        "I stayed late to finish the launch deck.\n\n"
        '**Nudge:** "What made you keep going?"\n\n'
        "**Response:** I did not want to hand it off."
    )


def test_render_judge_prompt_omits_previous_entries_when_none():
    schwartz_config = load_schwartz_values("config/schwartz_values.yaml")

    prompt = render_judge_prompt(
        session_content="Routine day.",
        entry_date="2025-01-10",
        persona_name="Test User",
        persona_age="25-34",
        persona_profession="Teacher",
        persona_culture="East Asian",
        persona_core_values=["Security"],
        persona_bio="Prefers predictable routines.",
        schwartz_config=schwartz_config,
        previous_entries=None,
    )

    assert "## Recent Entries (for context)" not in prompt
    assert "- **Core Values (from profile):** Security" in prompt
    assert "- **Bio:** Prefers predictable routines." in prompt


@pytest.mark.asyncio
async def test_judge_session_parses_valid_scores():
    schwartz_config = load_schwartz_values("config/schwartz_values.yaml")

    async def fake_llm_complete(prompt: str, response_format: dict | None) -> str:
        assert "Journal Session to Evaluate" in prompt
        assert response_format is not None
        return (
            '{"scores": {"self_direction": 0, "stimulation": 0, "hedonism": 0, '
            '"achievement": 1, "power": 0, "security": 0, "conformity": 0, '
            '"tradition": 0, "benevolence": 0, "universalism": 0}, '
            '"rationales": {"achievement": "Worked late to finish a difficult deliverable."}}'
        )

    scores, rationales, _ = await judge_session(
        session_content="I stayed late to finish the launch deck.",
        entry_date="2025-01-10",
        persona_name="Test User",
        persona_age="25-34",
        persona_profession="Product Manager",
        persona_culture="North American",
        persona_core_values=["Achievement"],
        persona_bio="Focused on shipping ambitious product milestones.",
        schwartz_config=schwartz_config,
        llm_complete=fake_llm_complete,
        previous_entries=None,
    )

    assert scores is not None
    assert scores.achievement == 1
    assert rationales == {
        "achievement": "Worked late to finish a difficult deliverable."
    }


@pytest.mark.asyncio
async def test_judge_session_handles_invalid_payload():
    schwartz_config = load_schwartz_values("config/schwartz_values.yaml")

    async def fake_llm_complete(_: str, __: dict | None) -> str:
        return "not-json"

    scores, rationales, _ = await judge_session(
        session_content="Routine day.",
        entry_date="2025-01-10",
        persona_name="Test User",
        persona_age="25-34",
        persona_profession="Teacher",
        persona_culture="East Asian",
        persona_core_values=["Security"],
        persona_bio="Prefers predictable routines.",
        schwartz_config=schwartz_config,
        llm_complete=fake_llm_complete,
        previous_entries=None,
    )

    assert scores is None
    assert rationales is None
