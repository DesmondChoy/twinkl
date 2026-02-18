"""Tests for src.nudge.decision — anti-annoyance, formatting, and LLM decision."""

import json
from unittest.mock import AsyncMock

import pytest

from src.nudge.decision import (
    _normalize_category,
    decide_nudge,
    format_previous_entries,
    should_suppress_nudge,
)


# ---------------------------------------------------------------------------
# should_suppress_nudge — pure anti-annoyance cap
# ---------------------------------------------------------------------------

class TestShouldSuppressNudge:
    """2-per-3-entry sliding-window cap."""

    def test_empty_history(self):
        assert should_suppress_nudge([]) is False

    def test_below_cap(self):
        assert should_suppress_nudge([True, False, False]) is False

    def test_at_cap_suppresses(self):
        assert should_suppress_nudge([True, True, False]) is True

    def test_above_cap_suppresses(self):
        assert should_suppress_nudge([True, True, True]) is True

    def test_window_boundary_only_recent_matter(self):
        # Old nudges outside window should not count
        assert should_suppress_nudge([True, True, True, False, False, False]) is False

    def test_custom_window_and_max(self):
        # window_size=2, max_nudges=1 → suppress if 1 nudge in last 2
        assert should_suppress_nudge([False, True], window_size=2, max_nudges=1) is True
        assert should_suppress_nudge([False, False], window_size=2, max_nudges=1) is False

    def test_single_entry_no_nudge(self):
        assert should_suppress_nudge([False]) is False

    def test_single_entry_with_nudge_default_cap(self):
        # 1 nudge < max_nudges=2, so not suppressed
        assert should_suppress_nudge([True]) is False


# ---------------------------------------------------------------------------
# format_previous_entries — metadata stripping
# ---------------------------------------------------------------------------

class TestFormatPreviousEntries:
    """Ensures only date + content pass through (no metadata leakage)."""

    def test_none_input(self):
        assert format_previous_entries(None) is None

    def test_empty_list(self):
        assert format_previous_entries([]) is None

    def test_strips_metadata(self):
        entries = [
            {
                "date": "2024-01-01",
                "content": "hello",
                "tone": "reflective",
                "verbosity": "short",
                "reflection_mode": "narrative",
            }
        ]
        result = format_previous_entries(entries)
        assert result == [{"date": "2024-01-01", "content": "hello"}]

    def test_truncates_to_max_entries(self):
        entries = [
            {"date": f"2024-01-0{i}", "content": f"entry {i}"}
            for i in range(1, 6)
        ]
        result = format_previous_entries(entries, max_entries=2)
        assert len(result) == 2
        # Should keep the last 2 (most recent)
        assert result[0]["date"] == "2024-01-04"
        assert result[1]["date"] == "2024-01-05"

    def test_preserves_only_date_and_content(self):
        entries = [
            {"date": "2024-01-01", "content": "text", "extra_field": "should be gone"}
        ]
        result = format_previous_entries(entries)
        assert "extra_field" not in result[0]
        assert set(result[0].keys()) == {"date", "content"}


# ---------------------------------------------------------------------------
# _normalize_category — LLM output edge cases
# ---------------------------------------------------------------------------

class TestNormalizeCategory:
    def test_valid_categories(self):
        assert _normalize_category("clarification") == "clarification"
        assert _normalize_category("elaboration") == "elaboration"
        assert _normalize_category("tension_surfacing") == "tension_surfacing"

    def test_case_insensitivity(self):
        assert _normalize_category("Clarification") == "clarification"
        assert _normalize_category("ELABORATION") == "elaboration"
        assert _normalize_category("Tension_Surfacing") == "tension_surfacing"

    def test_whitespace_handling(self):
        assert _normalize_category("  clarification  ") == "clarification"
        assert _normalize_category("tension surfacing") == "tension_surfacing"

    def test_no_nudge_returns_none(self):
        assert _normalize_category("no_nudge") is None
        assert _normalize_category("No_Nudge") is None

    def test_invalid_value_returns_none(self):
        assert _normalize_category("invalid") is None
        assert _normalize_category("") is None
        assert _normalize_category("grounding") is None


# ---------------------------------------------------------------------------
# decide_nudge — async LLM-based decision
# ---------------------------------------------------------------------------

class TestDecideNudge:
    @pytest.mark.asyncio
    async def test_clarification_decision(self):
        mock_llm = AsyncMock(return_value=json.dumps({
            "decision": "clarification",
            "reason": "Entry is vague",
        }))

        should, category, reason = await decide_nudge(
            "Feeling off today.", "2024-01-15", None, mock_llm
        )

        assert should is True
        assert category == "clarification"
        assert reason == "Entry is vague"

    @pytest.mark.asyncio
    async def test_no_nudge_decision(self):
        mock_llm = AsyncMock(return_value=json.dumps({
            "decision": "no_nudge",
            "reason": "Entry is complete",
        }))

        should, category, reason = await decide_nudge(
            "Had a great day at work.", "2024-01-15", None, mock_llm
        )

        assert should is False
        assert category is None
        assert reason is None

    @pytest.mark.asyncio
    async def test_llm_failure_returns_no_nudge(self):
        mock_llm = AsyncMock(return_value=None)

        should, category, reason = await decide_nudge(
            "Some entry.", "2024-01-15", None, mock_llm
        )

        assert should is False
        assert category is None
        assert reason is None

    @pytest.mark.asyncio
    async def test_metadata_not_leaked_into_prompt(self):
        """Verify the LLM prompt does not contain synthetic metadata fields."""
        captured_prompt = None

        async def capture_llm(prompt, response_format):
            nonlocal captured_prompt
            captured_prompt = prompt
            return json.dumps({"decision": "no_nudge", "reason": ""})

        prev = [{"date": "2024-01-14", "content": "Previous entry"}]

        await decide_nudge("Entry text", "2024-01-15", prev, capture_llm)

        assert captured_prompt is not None
        # These metadata fields must never appear in the prompt
        for forbidden in ["tone", "verbosity", "reflection_mode"]:
            assert forbidden not in captured_prompt.lower(), (
                f"Metadata field '{forbidden}' leaked into LLM prompt"
            )

    @pytest.mark.asyncio
    async def test_tension_surfacing_decision(self):
        mock_llm = AsyncMock(return_value=json.dumps({
            "decision": "tension_surfacing",
            "reason": "Hints at unresolved conflict",
        }))

        should, category, reason = await decide_nudge(
            "It was fine, I guess.", "2024-01-15", None, mock_llm
        )

        assert should is True
        assert category == "tension_surfacing"
        assert reason == "Hints at unresolved conflict"
