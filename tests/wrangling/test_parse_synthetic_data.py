"""Tests for synthetic data parser (parse_synthetic_data.py)."""

import pytest

from src.wrangling.parse_synthetic_data import (
    extract_persona_id,
    parse_entries,
    parse_persona_profile,
    write_wrangled_markdown,
)

# --- Module-level markdown constants ---

PERSONA_HEADER = """\
# Persona a3f8b2c1: Test Person

- Age: 30-39
- Profession: Software Engineer
- Culture: East Asian
- Core Values: Power, Achievement
- Bio: A driven professional focused on career growth.

---

"""

TRIGGER_ENTRY = """\
## Entry 1 - 2025-01-15

### Initial Entry
**Tone:** reflective

Today I spent extra hours polishing the presentation for the board meeting tomorrow.

### Nudge
**Trigger:** career milestone

"Have you considered what success looks like beyond the next promotion?"

### Response
**Mode:** thoughtful

That is a good question. I have been so focused on the next rung that I have not thought about the bigger picture.

---
"""

CATEGORY_ENTRY = """\
## Entry 1 - 2025-02-10

### Initial Entry
**Tone:** contemplative

I stayed late again even though nobody asked me to.

### Nudge
**Category:** values-reflection

Sometimes the things we chase hardest are the things we need least.

### Response
**Mode:** defensive

I disagree. Hard work always pays off eventually.

---
"""

TYPE_ENTRY = """\
## Entry 1 - 2025-03-05

### Initial Entry
**Tone:** exhausted

Long week. I feel burnt out but cannot stop thinking about work.

### Nudge
**Type:** gentle-reminder

> Remember that rest is not the opposite of productivity.

### Response
**Mode:** reluctant

Maybe. I will try to take a break this weekend.

---
"""

NO_NUDGE_ENTRY = """\
## Entry 2 - 2025-01-20

### Initial Entry
**Tone:** neutral

Routine day at the office. Nothing remarkable happened.

*(No nudge for this entry)*

---
"""

NO_RESPONSE_ENTRY = """\
## Entry 3 - 2025-01-25

### Initial Entry
**Tone:** tired

Long day. Just want to rest.

### Nudge
**Trigger:** self-care

"What does rest mean to you?"

*(No response - persona did not reply to nudge)*

---
"""

# Raw trigger-format nudge with no response section and no explicit marker.
TRIGGER_NO_RESPONSE_NO_MARKER_ENTRY = """\
## Entry 4 - 2025-02-01

### Initial Entry
**Tone:** reflective

I keep circling around the same unresolved decision.

### Nudge
**Trigger:** unresolved tension

"What happens if you decide now?"

---
"""

# Raw category-format nudge with no response marker (regression case).
CATEGORY_NO_RESPONSE_ENTRY = """\
## Entry 4 - 2025-02-02

### Initial Entry
**Tone:** uneasy

I keep postponing the call because I don't know what to say.

### Nudge
**Category:** tension_surfacing

What would it cost you to ask for help right now?

*(No response - persona did not reply to nudge)*

---
"""

# Raw entry with non-canonical no-response marker.
NO_RESPONSE_RECORDED_ENTRY = """\
## Entry 5 - 2025-02-03

### Initial Entry
**Tone:** unsettled

I am avoiding this conversation again.

### Nudge
**Trigger:** avoidance

"What are you protecting by waiting?"

*(No response recorded)*

---
"""

# Raw entry with response header variant and no mode line.
RESPONSE_HEADER_VARIANT_ENTRY = """\
## Entry 6 - 2025-02-04

### Initial Entry
**Tone:** neutral

I noticed I was more irritable than usual.

### Nudge
**Trigger:** reflection

"What was different today?"

### Response (Deflecting/redirecting)
Nothing. Just a long day.

---
"""

# Raw entry with "Response Mode" metadata key.
RESPONSE_MODE_VARIANT_ENTRY = """\
## Entry 7 - 2025-02-05

### Initial Entry
**Tone:** tense

I snapped at my colleague in the meeting.

### Nudge
**Trigger:** accountability

"What did you want him to hear?"

### Response
**Response Mode**: Emotional/Venting

That I am carrying too much and pretending I am fine.

---
"""

# Raw entry where nudge is explicitly suppressed.
NO_NUDGE_SUPPRESSED_ENTRY = """\
## Entry 8 - 2025-02-06

### Initial Entry
**Tone:** defensive

I was still thinking about yesterday and did not want another prompt.

### Nudge
**Category:** Silence

`[No nudge generated - Same-day follow-up, allowing space for continued reflection]`

### Response
`[N/A - No nudge provided]`

---
"""

# Wrangled-format entry with nested quotes inside the nudge text.
WRANGLED_NESTED_QUOTE_ENTRY = """\
## Entry 0 - 2025-03-10

I am trying to make sense of what happened in that meeting.

**Nudge:** "What does \"keeping score\" mean to you right now?"

---
"""

# Full persona with trigger format and multiple entries
FULL_TRIGGER_PERSONA = PERSONA_HEADER + TRIGGER_ENTRY + NO_NUDGE_ENTRY + NO_RESPONSE_ENTRY

# Full persona with category format
FULL_CATEGORY_PERSONA = PERSONA_HEADER + CATEGORY_ENTRY

# Full persona with type format
FULL_TYPE_PERSONA = PERSONA_HEADER + TYPE_ENTRY


class TestParsePersonaProfile:

    def test_extracts_name_from_header(self):
        profile = parse_persona_profile(FULL_TRIGGER_PERSONA)
        assert profile["name"] == "Test Person"

    def test_extracts_all_profile_fields(self):
        profile = parse_persona_profile(FULL_TRIGGER_PERSONA)
        assert profile["age"] == "30-39"
        assert profile["profession"] == "Software Engineer"
        assert profile["culture"] == "East Asian"
        assert profile["bio"] == "A driven professional focused on career growth."

    def test_core_values_comma_separated(self):
        profile = parse_persona_profile(FULL_TRIGGER_PERSONA)
        assert profile["core_values"] == ["Power", "Achievement"]


class TestMetadataStripping:

    def test_initial_entry_excludes_tone(self):
        entries = parse_entries(FULL_TRIGGER_PERSONA)
        for entry in entries:
            if entry["initial_entry"]:
                assert "Tone" not in entry["initial_entry"]
                assert "Verbosity" not in entry["initial_entry"]

    def test_response_excludes_mode(self):
        entries = parse_entries(FULL_TRIGGER_PERSONA)
        entry_with_response = entries[0]
        assert entry_with_response["has_response"] is True
        assert "Mode" not in entry_with_response["response_text"]

    def test_nudge_excludes_trigger_metadata(self):
        entries = parse_entries(FULL_TRIGGER_PERSONA)
        entry_with_nudge = entries[0]
        assert entry_with_nudge["has_nudge"] is True
        assert "Trigger" not in entry_with_nudge["nudge_text"]

    def test_nudge_excludes_category_metadata(self):
        entries = parse_entries(FULL_CATEGORY_PERSONA)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert "Category" not in entry["nudge_text"]


class TestNudgeFormats:

    def test_trigger_format_quoted_text(self):
        entries = parse_entries(PERSONA_HEADER + TRIGGER_ENTRY)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert entry["nudge_text"] == "Have you considered what success looks like beyond the next promotion?"

    def test_category_format_unquoted_text(self):
        entries = parse_entries(FULL_CATEGORY_PERSONA)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert entry["nudge_text"] == "Sometimes the things we chase hardest are the things we need least."

    def test_type_format_blockquote(self):
        entries = parse_entries(FULL_TYPE_PERSONA)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert entry["nudge_text"] == "Remember that rest is not the opposite of productivity."

    def test_no_nudge_marker(self):
        entries = parse_entries(PERSONA_HEADER + NO_NUDGE_ENTRY)
        entry = entries[0]
        assert entry["has_nudge"] is False
        assert entry["nudge_text"] is None
        assert entry["has_response"] is False
        assert entry["response_text"] is None

    def test_no_response_marker(self):
        entries = parse_entries(PERSONA_HEADER + NO_RESPONSE_ENTRY)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert entry["nudge_text"] is not None
        assert entry["has_response"] is False
        assert entry["response_text"] is None

    def test_noncanonical_no_response_marker(self):
        entries = parse_entries(PERSONA_HEADER + NO_RESPONSE_RECORDED_ENTRY)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert entry["has_response"] is False
        assert entry["response_text"] is None

    def test_category_nudge_excludes_no_response_marker(self):
        entries = parse_entries(PERSONA_HEADER + CATEGORY_NO_RESPONSE_ENTRY)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert entry["nudge_text"] == "What would it cost you to ask for help right now?"
        assert entry["has_response"] is False
        assert entry["response_text"] is None

    def test_trigger_nudge_without_response_marker_still_parses(self):
        entries = parse_entries(PERSONA_HEADER + TRIGGER_NO_RESPONSE_NO_MARKER_ENTRY)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert entry["nudge_text"] == "What happens if you decide now?"
        assert entry["has_response"] is False

    def test_response_header_parenthetical_parses_without_mode_line(self):
        entries = parse_entries(PERSONA_HEADER + RESPONSE_HEADER_VARIANT_ENTRY)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert entry["has_response"] is True
        assert entry["response_text"] == "Nothing. Just a long day."

    def test_response_mode_metadata_key_variant(self):
        entries = parse_entries(PERSONA_HEADER + RESPONSE_MODE_VARIANT_ENTRY)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert entry["has_response"] is True
        assert "Response Mode" not in entry["response_text"]
        assert entry["response_text"] == "That I am carrying too much and pretending I am fine."

    def test_no_nudge_suppressed_placeholder(self):
        entries = parse_entries(PERSONA_HEADER + NO_NUDGE_SUPPRESSED_ENTRY)
        entry = entries[0]
        assert entry["has_nudge"] is False
        assert entry["nudge_text"] is None
        assert entry["has_response"] is False
        assert entry["response_text"] is None

    def test_wrangled_nudge_preserves_nested_quotes(self):
        entries = parse_entries(PERSONA_HEADER + WRANGLED_NESTED_QUOTE_ENTRY)
        entry = entries[0]
        assert entry["has_nudge"] is True
        assert entry["nudge_text"] == 'What does "keeping score" mean to you right now?'

    def test_nudge_section_without_format_raises(self):
        malformed = PERSONA_HEADER + """\
## Entry 1 - 2025-04-01

### Initial Entry
**Tone:** neutral

Just a regular day.

### Nudge

This nudge has no format marker at all.

### Response
**Mode:** confused

I am not sure what to make of this.

---
"""
        with pytest.raises(ValueError, match="Found '### Nudge' section but missing"):
            parse_entries(malformed)


class TestParseEntries:

    def test_auto_detects_raw_format(self):
        entries = parse_entries(FULL_TRIGGER_PERSONA)
        # If raw format is detected, initial_entry should be clean (no "### Initial Entry" text)
        entry = entries[0]
        assert entry["initial_entry"] is not None
        assert "### Initial Entry" not in entry["initial_entry"]

    def test_t_index_zero_based(self):
        entries = parse_entries(FULL_TRIGGER_PERSONA)
        assert len(entries) == 3
        assert [e["t_index"] for e in entries] == [0, 1, 2]

    def test_entry_with_all_sections(self):
        entries = parse_entries(PERSONA_HEADER + TRIGGER_ENTRY)
        entry = entries[0]
        assert entry["date"] == "2025-01-15"
        assert entry["initial_entry"] is not None
        assert len(entry["initial_entry"]) > 0
        assert entry["nudge_text"] is not None
        assert len(entry["nudge_text"]) > 0
        assert entry["response_text"] is not None
        assert len(entry["response_text"]) > 0
        assert entry["has_nudge"] is True
        assert entry["has_response"] is True


class TestWriteWrangledMarkdown:

    def test_writes_output_files(self, tmp_path):
        input_dir = tmp_path / "synthetic"
        output_dir = tmp_path / "wrangled"
        input_dir.mkdir()

        (input_dir / "persona_a3f8b2c1.md").write_text(FULL_TRIGGER_PERSONA)

        written = write_wrangled_markdown(input_dir, output_dir, update_registry=False)

        assert len(written) == 1
        assert (output_dir / "persona_a3f8b2c1.md").exists()

    def test_skips_existing_files(self, tmp_path):
        input_dir = tmp_path / "synthetic"
        output_dir = tmp_path / "wrangled"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "persona_a3f8b2c1.md").write_text(FULL_TRIGGER_PERSONA)
        (output_dir / "persona_a3f8b2c1.md").write_text("already exists")

        written = write_wrangled_markdown(input_dir, output_dir, update_registry=False)

        assert len(written) == 0
        # Verify original content not overwritten
        assert (output_dir / "persona_a3f8b2c1.md").read_text() == "already exists"

    def test_output_clean_of_metadata(self, tmp_path):
        input_dir = tmp_path / "synthetic"
        output_dir = tmp_path / "wrangled"
        input_dir.mkdir()

        (input_dir / "persona_a3f8b2c1.md").write_text(FULL_TRIGGER_PERSONA)

        write_wrangled_markdown(input_dir, output_dir, update_registry=False)

        output_content = (output_dir / "persona_a3f8b2c1.md").read_text()
        # Generation metadata should be stripped
        assert "**Tone:**" not in output_content
        assert "**Tone**:" not in output_content
        assert "**Mode:**" not in output_content
        assert "**Mode**:" not in output_content
        assert "**Trigger:**" not in output_content
        assert "**Trigger**:" not in output_content
        assert "**Category:**" not in output_content
        assert "**Category**:" not in output_content
        assert "**Type:**" not in output_content
        assert "**Type**:" not in output_content


class TestExtractPersonaId:

    def test_uuid_format(self):
        assert extract_persona_id("persona_a3f8b2c1.md") == "a3f8b2c1"

    def test_numeric_format(self):
        assert extract_persona_id("persona_001.md") == "001"

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError, match="Cannot extract persona ID from filename"):
            extract_persona_id("not_a_persona.md")
