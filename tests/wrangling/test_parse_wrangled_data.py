"""Tests for wrangled file parser and VIF dataset integration."""

from unittest.mock import patch

import pytest

from src.wrangling.parse_wrangled_data import (
    parse_wrangled_entries,
    parse_wrangled_file,
    parse_wrangled_persona_profile,
)

# --- Sample wrangled markdown for testing ---

SAMPLE_WRANGLED_CONTENT = """\
# Persona abc12345: Test User

## Profile
- **Persona ID:** abc12345
- **Name:** Test User
- **Age:** 25-34
- **Profession:** Engineer
- **Culture:** Western European
- **Core Values:** Security, Benevolence
- **Bio:** Test user bio that spans a single line.

---

## Entry 0 - 2025-01-01

Had a productive day at work. Finished the API integration ahead of schedule and felt good about the progress.

**Nudge:** "What made this project feel different from others?"

**Response:** I think it was having clear requirements from the start. Usually I spend half my time guessing what people want.

---

## Entry 1 - 2025-01-05

Quiet weekend. Spent most of it reading and cooking. Did not check email once.

---

## Entry 2 - 2025-01-10

Difficult conversation with my manager about the team restructuring. She wants me to lead the new platform team but I am not sure I want the extra responsibility.

**Nudge:** "What would taking on the role mean for you beyond the extra work?"

**Response:** Recognition, I suppose. And a chance to shape something from the ground up. But also more meetings, more politics. I keep going back and forth.

---
"""


class TestParseWrangledPersonaProfile:
    """Tests for parse_wrangled_persona_profile."""

    def test_extracts_all_profile_fields(self):
        profile = parse_wrangled_persona_profile(SAMPLE_WRANGLED_CONTENT)

        assert profile["name"] == "Test User"
        assert profile["age"] == "25-34"
        assert profile["profession"] == "Engineer"
        assert profile["culture"] == "Western European"
        assert profile["core_values"] == ["Security", "Benevolence"]
        assert profile["bio"] == "Test user bio that spans a single line."

    def test_missing_fields_return_none(self):
        minimal = "# Persona abc12345: Someone\n\nNo profile fields here."
        profile = parse_wrangled_persona_profile(minimal)

        assert profile["name"] == "Someone"
        assert profile["age"] is None
        assert profile["profession"] is None
        assert profile["core_values"] == []


class TestParseWrangledEntries:
    """Tests for parse_wrangled_entries."""

    def test_parses_entry_with_nudge_and_response(self):
        entries, warnings = parse_wrangled_entries(SAMPLE_WRANGLED_CONTENT)

        assert len(warnings) == 0
        assert len(entries) == 3

        # Entry 0: has nudge + response
        e0 = entries[0]
        assert e0["t_index"] == 0
        assert e0["date"] == "2025-01-01"
        assert "productive day at work" in e0["initial_entry"]
        assert e0["has_nudge"] is True
        assert "What made this project feel different" in e0["nudge_text"]
        assert e0["has_response"] is True
        assert "clear requirements" in e0["response_text"]

    def test_parses_entry_without_nudge(self):
        entries, warnings = parse_wrangled_entries(SAMPLE_WRANGLED_CONTENT)

        # Entry 1: no nudge, no response
        e1 = entries[1]
        assert e1["t_index"] == 1
        assert e1["date"] == "2025-01-05"
        assert "Quiet weekend" in e1["initial_entry"]
        assert e1["has_nudge"] is False
        assert e1["nudge_text"] is None
        assert e1["has_response"] is False
        assert e1["response_text"] is None

    def test_all_fields_non_null_for_nudged_entry(self):
        """Regression: the old parser returned None for all text fields on wrangled format."""
        entries, _ = parse_wrangled_entries(SAMPLE_WRANGLED_CONTENT)

        for entry in entries:
            assert entry["initial_entry"] is not None, (
                f"Entry {entry['t_index']} has null initial_entry"
            )
            assert len(entry["initial_entry"]) >= 10, (
                f"Entry {entry['t_index']} initial_entry too short"
            )

    def test_warns_on_short_entry(self):
        """Entries with very short text are skipped with a warning."""
        content = """\
## Entry 0 - 2025-01-01

Short.

---
"""
        entries, warnings = parse_wrangled_entries(content, "test.md")

        assert len(entries) == 0
        assert len(warnings) == 1
        assert "too short" in warnings[0].message


class TestParseWrangledFile:
    """Tests for parse_wrangled_file."""

    def test_returns_3_tuple(self, tmp_path):
        """parse_wrangled_file returns (profile, entries, warnings) — not 2-tuple."""
        filepath = tmp_path / "persona_abc12345.md"
        filepath.write_text(SAMPLE_WRANGLED_CONTENT)

        result = parse_wrangled_file(filepath)

        assert isinstance(result, tuple)
        assert len(result) == 3

        profile, entries, warnings = result
        assert profile["persona_id"] == "abc12345"
        assert profile["name"] == "Test User"
        assert len(entries) == 3
        assert len(warnings) == 0

    def test_extracts_persona_id_from_filename(self, tmp_path):
        filepath = tmp_path / "persona_deadbeef.md"
        filepath.write_text(SAMPLE_WRANGLED_CONTENT)

        profile, _, _ = parse_wrangled_file(filepath)
        assert profile["persona_id"] == "deadbeef"


class TestDatasetLoadEntries:
    """Tests for src.vif.dataset.load_entries."""

    def test_raises_if_parser_returns_all_null_initial_entries(self, tmp_path):
        """Guard rail: all-null parser output is rejected immediately."""
        from src.vif.dataset import load_entries

        filepath = tmp_path / "persona_abc12345.md"
        filepath.write_text(SAMPLE_WRANGLED_CONTENT)

        with patch("src.vif.dataset.parse_wrangled_file") as mock_parse:
            mock_parse.return_value = (
                {"persona_id": "abc12345", "name": "Test", "core_values": ["Security"]},
                [
                    {
                        "t_index": 0,
                        "date": "2025-01-01",
                        "initial_entry": None,
                        "nudge_text": None,
                        "response_text": None,
                    },
                    {
                        "t_index": 1,
                        "date": "2025-01-02",
                        "initial_entry": None,
                        "nudge_text": None,
                        "response_text": None,
                    },
                ],
                [],  # warnings
            )

            with pytest.raises(ValueError, match="All initial_entry values are null"):
                load_entries(tmp_path)
            mock_parse.assert_called_once_with(filepath)

    def test_succeeds_with_valid_wrangled_files(self, tmp_path):
        """load_entries successfully parses valid wrangled-format files."""
        from src.vif.dataset import load_entries

        filepath = tmp_path / "persona_abc12345.md"
        filepath.write_text(SAMPLE_WRANGLED_CONTENT)

        df = load_entries(tmp_path)

        assert len(df) == 3
        assert "initial_entry" in df.columns
        assert "nudge_text" in df.columns
        assert "response_text" in df.columns

        # Verify text content is actually populated (not None/empty)
        initial_entries = df["initial_entry"].to_list()
        assert all(entry is not None for entry in initial_entries)
        assert all(len(entry) > 0 for entry in initial_entries)

        # Verify persona fields
        assert df["persona_id"][0] == "abc12345"
        assert df["persona_name"][0] == "Test User"
