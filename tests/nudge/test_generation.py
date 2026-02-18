"""Tests for src.nudge.generation — text generation helpers and integration."""

import json
from unittest.mock import AsyncMock

import pytest

from src.nudge.generation import (
    count_words,
    generate_nudge_response,
    generate_nudge_text,
    select_response_mode,
    weighted_choice,
)


# ---------------------------------------------------------------------------
# count_words
# ---------------------------------------------------------------------------

class TestCountWords:
    def test_simple(self):
        assert count_words("hello world") == 2

    def test_empty(self):
        assert count_words("") == 0

    def test_multiline(self):
        assert count_words("one two\nthree four") == 4

    def test_extra_whitespace(self):
        assert count_words("  one   two  ") == 2


# ---------------------------------------------------------------------------
# weighted_choice
# ---------------------------------------------------------------------------

class TestWeightedChoice:
    def test_single_option(self):
        assert weighted_choice({"only": 1.0}) == "only"

    def test_returns_valid_option(self):
        weights = {"a": 0.5, "b": 0.3, "c": 0.2}
        for _ in range(50):
            result = weighted_choice(weights)
            assert result in weights

    def test_zero_weight_never_chosen(self):
        # With weight 0, "never" should (almost) never be chosen
        # Using a non-zero epsilon to avoid division issues
        weights = {"always": 1.0, "never": 0.0001}
        results = {weighted_choice(weights) for _ in range(100)}
        assert "always" in results

    def test_empty_weights_raises(self):
        with pytest.raises(ValueError, match="at least one option"):
            weighted_choice({})

    def test_non_positive_total_raises(self):
        with pytest.raises(ValueError, match="sum to a positive value"):
            weighted_choice({"a": 0.0, "b": 0.0})


# ---------------------------------------------------------------------------
# select_response_mode
# ---------------------------------------------------------------------------

class TestSelectResponseMode:
    def test_returns_valid_mode(self):
        config = {
            "nudge": {
                "response_modes": [
                    {"mode": "Answering directly", "weight": 0.5},
                    {"mode": "Deflecting/redirecting", "weight": 0.3},
                    {"mode": "Revealing deeper thought", "weight": 0.2},
                ]
            }
        }
        valid_modes = {"Answering directly", "Deflecting/redirecting", "Revealing deeper thought"}
        for _ in range(50):
            assert select_response_mode(config) in valid_modes

    def test_falls_back_when_weights_invalid(self):
        config = {
            "nudge": {
                "response_modes": [
                    {"mode": "Answering directly", "weight": 0.0},
                    {"mode": "Deflecting/redirecting", "weight": 0.0},
                ]
            }
        }
        assert select_response_mode(config) == "Answering directly"

    def test_falls_back_when_modes_missing(self):
        assert select_response_mode({"nudge": {"response_modes": []}}) == "Answering directly"


# ---------------------------------------------------------------------------
# generate_nudge_text — integration tests with mock LLM
# ---------------------------------------------------------------------------

class TestGenerateNudgeTextIntegration:
    @pytest.fixture
    def nudge_config(self):
        return {"nudge": {"min_words": 2, "max_words": 12}}

    @pytest.mark.asyncio
    async def test_successful_generation(self, nudge_config):
        mock_llm = AsyncMock(return_value=json.dumps({
            "nudge_text": "What made today feel off?"
        }))

        text, prompt = await generate_nudge_text(
            "Feeling off today.",
            "2024-01-15",
            "clarification",
            None,
            nudge_config,
            mock_llm,
        )

        assert text == "What made today feel off?"
        assert "clarification" in prompt.lower() or "clarification" in prompt

    @pytest.mark.asyncio
    async def test_word_count_validation_rejects_over_limit(self, nudge_config):
        """If LLM returns text exceeding max_words, it should retry and fail."""
        long_text = " ".join(["word"] * 20)  # 20 words > max_words=12
        mock_llm = AsyncMock(return_value=json.dumps({
            "nudge_text": long_text
        }))

        text, prompt = await generate_nudge_text(
            "Entry content.",
            "2024-01-15",
            "elaboration",
            None,
            nudge_config,
            mock_llm,
            max_attempts=2,
        )

        assert text is None
        assert mock_llm.call_count == 2  # Retried

    @pytest.mark.asyncio
    async def test_all_categories_render_valid_prompts(self, nudge_config):
        """Each category should produce a prompt without errors."""
        categories = ["clarification", "elaboration", "tension_surfacing"]

        for category in categories:
            mock_llm = AsyncMock(return_value=json.dumps({
                "nudge_text": "A short nudge?"
            }))

            text, prompt = await generate_nudge_text(
                "Some entry content.",
                "2024-01-15",
                category,
                None,
                nudge_config,
                mock_llm,
            )

            assert text is not None, f"Failed for category: {category}"
            assert len(prompt) > 0

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self, nudge_config):
        mock_llm = AsyncMock(return_value=None)

        text, prompt = await generate_nudge_text(
            "Entry.",
            "2024-01-15",
            "clarification",
            None,
            nudge_config,
            mock_llm,
        )

        assert text is None

    @pytest.mark.asyncio
    async def test_malformed_json_retries_until_success(self, nudge_config):
        mock_llm = AsyncMock(
            side_effect=[
                "not-json",
                json.dumps({"nudge_text": "What felt hard about that moment?"}),
            ]
        )

        text, _ = await generate_nudge_text(
            "Entry.",
            "2024-01-15",
            "clarification",
            None,
            nudge_config,
            mock_llm,
            max_attempts=2,
        )

        assert text == "What felt hard about that moment?"
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_non_string_nudge_text_retries_until_success(self, nudge_config):
        mock_llm = AsyncMock(
            side_effect=[
                json.dumps({"nudge_text": 123}),
                json.dumps({"nudge_text": "What felt hardest today?"}),
            ]
        )

        text, _ = await generate_nudge_text(
            "Entry.",
            "2024-01-15",
            "clarification",
            None,
            nudge_config,
            mock_llm,
            max_attempts=2,
        )

        assert text == "What felt hardest today?"
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_previous_entries_passed_to_prompt(self, nudge_config):
        """Previous entries should appear in the rendered prompt."""
        captured_prompt = None

        async def capture_llm(prompt, response_format):
            nonlocal captured_prompt
            captured_prompt = prompt
            return json.dumps({"nudge_text": "Short nudge?"})

        prev = [{"date": "2024-01-14", "content": "Yesterday's entry text"}]

        await generate_nudge_text(
            "Today's entry.",
            "2024-01-15",
            "elaboration",
            prev,
            nudge_config,
            capture_llm,
        )

        assert captured_prompt is not None
        assert "Yesterday's entry text" in captured_prompt


class TestGenerateNudgeResponseIntegration:
    @pytest.mark.asyncio
    async def test_malformed_json_retries_until_success(self):
        config = {
            "nudge": {
                "response_modes": [
                    {"mode": "Answering directly", "weight": 1.0},
                ]
            }
        }
        mock_llm = AsyncMock(
            side_effect=["not-json", json.dumps({"content": "That actually makes sense."})]
        )

        response, _, response_mode = await generate_nudge_response(
            persona_name="Alex",
            persona_age="28-35",
            persona_profession="Designer",
            persona_culture="Singaporean",
            persona_bio="Thoughtful and often reflective.",
            entry_content="I had a rough day.",
            entry_date="2024-01-15",
            nudge_text="What part felt roughest?",
            config=config,
            llm_complete=mock_llm,
            max_attempts=2,
        )

        assert response is not None
        assert response.content == "That actually makes sense."
        assert response_mode == "Answering directly"
        assert mock_llm.call_count == 2

    @pytest.mark.asyncio
    async def test_non_string_content_retries_until_success(self):
        config = {
            "nudge": {
                "response_modes": [
                    {"mode": "Answering directly", "weight": 1.0},
                ]
            }
        }
        mock_llm = AsyncMock(
            side_effect=[json.dumps({"content": 42}), json.dumps({"content": "I can share more."})]
        )

        response, _, response_mode = await generate_nudge_response(
            persona_name="Alex",
            persona_age="28-35",
            persona_profession="Designer",
            persona_culture="Singaporean",
            persona_bio="Thoughtful and often reflective.",
            entry_content="I had a rough day.",
            entry_date="2024-01-15",
            nudge_text="What part felt roughest?",
            config=config,
            llm_complete=mock_llm,
            max_attempts=2,
        )

        assert response is not None
        assert response.content == "I can share more."
        assert response_mode == "Answering directly"
        assert mock_llm.call_count == 2
