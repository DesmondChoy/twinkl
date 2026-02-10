"""Parse wrangled persona markdown files into structured data.

This module provides shared parsing functions for wrangled-format markdown files.
The wrangled format is the cleaned output from the synthetic data pipeline,
used by both the VIF training pipeline and the annotation tool.

Wrangled format characteristics:
- Entry text directly after date header (no ### Initial Entry section)
- **Nudge:** and **Response:** inline markers
- Profile metadata in ## Profile section with **Field:** Value format

Usage:
    from src.wrangling.parse_wrangled_data import parse_wrangled_file

    profile, entries, warnings = parse_wrangled_file(Path("logs/wrangled/persona_abc123.md"))
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParseWarning:
    """Warning about a parsing issue that was handled gracefully."""

    file: str
    entry_index: int | None
    message: str


def parse_wrangled_persona_profile(content: str) -> dict:
    """Extract persona profile from wrangled markdown content.

    Args:
        content: Full markdown file content

    Returns:
        Dict with name, age, profession, culture, core_values, bio
    """
    profile = {}

    # Extract name from title: "# Persona a3f8b2c1: [Name]"
    name_match = re.search(r"^# Persona [a-f0-9]+: (.+)$", content, re.MULTILINE)
    profile["name"] = name_match.group(1).strip() if name_match else None

    # Extract profile fields (format: - **Field:** Value)
    def extract_field(field_name: str) -> str | None:
        pattern = rf"- \*\*{field_name}:\*\* (.+?)(?:\n|$)"
        match = re.search(pattern, content)
        return match.group(1).strip() if match else None

    profile["age"] = extract_field("Age")
    profile["profession"] = extract_field("Profession")
    profile["culture"] = extract_field("Culture")

    # Core values
    core_values_str = extract_field("Core Values")
    if core_values_str:
        profile["core_values"] = [v.strip() for v in core_values_str.split(",")]
    else:
        profile["core_values"] = []

    # Bio
    bio_match = re.search(r"- \*\*Bio:\*\* (.+?)(?=\n---|\n## |\Z)", content, re.DOTALL)
    profile["bio"] = bio_match.group(1).strip() if bio_match else None

    return profile


def parse_wrangled_entries(
    content: str, filename: str = "unknown"
) -> tuple[list[dict], list[ParseWarning]]:
    """Extract all journal entries from wrangled markdown content with validation.

    Wrangled format:
        ## Entry N - YYYY-MM-DD

        Entry text here...

        **Nudge:** "nudge text"

        **Response:** response text

    Args:
        content: Full markdown file content
        filename: Name of the file being parsed (for warnings)

    Returns:
        Tuple of (list of entry dicts, list of warnings for skipped/problematic entries)
    """
    entries = []
    warnings = []

    # Split by entry headers: "## Entry N - YYYY-MM-DD"
    entry_pattern = r"## Entry (\d+) - (\d{4}-\d{2}-\d{2})"
    parts = re.split(entry_pattern, content)

    # parts alternates: [preamble, idx1, date1, content1, idx2, date2, content2, ...]
    for i in range(1, len(parts), 3):
        if i + 2 >= len(parts):
            break

        t_index = int(parts[i])
        date = parts[i + 1]
        entry_content = parts[i + 2]

        entry = {
            "t_index": t_index,
            "date": date,
            "initial_entry": None,
            "nudge_text": None,
            "response_text": None,
            "has_nudge": False,
            "has_response": False,
        }

        # Parse the entry content
        # Entry text is everything before **Nudge:** or end of section
        nudge_marker = entry_content.find("**Nudge:**")
        if nudge_marker != -1:
            entry["initial_entry"] = entry_content[:nudge_marker].strip()

            # Extract nudge text (quoted) - use greedy match for nested quotes
            # e.g., "What's the "something else" you're hinting at?"
            nudge_match = re.search(r'\*\*Nudge:\*\* "(.+)"', entry_content)
            if nudge_match:
                entry["nudge_text"] = nudge_match.group(1).strip()
                entry["has_nudge"] = True

            # Extract response text
            response_match = re.search(
                r"\*\*Response:\*\* (.+?)(?=\n---|\n## |\Z)", entry_content, re.DOTALL
            )
            if response_match:
                entry["response_text"] = response_match.group(1).strip()
                entry["has_response"] = True
        else:
            # No nudge - entry text is everything until end
            entry["initial_entry"] = entry_content.strip().rstrip("-").strip()

        # Validation: check for required fields
        if not entry["date"]:
            warnings.append(
                ParseWarning(
                    file=filename,
                    entry_index=t_index,
                    message=f"Entry {t_index} missing date",
                )
            )
            continue  # Skip this entry

        if not entry["initial_entry"] or len(entry["initial_entry"]) < 10:
            warnings.append(
                ParseWarning(
                    file=filename,
                    entry_index=t_index,
                    message=f"Entry {t_index} has empty or too short initial_entry",
                )
            )
            continue  # Skip this entry

        entries.append(entry)

    return entries, warnings


def parse_wrangled_file(
    filepath: Path,
) -> tuple[dict, list[dict], list[ParseWarning]]:
    """Parse a single wrangled persona markdown file with validation.

    Args:
        filepath: Path to persona_*.md file

    Returns:
        Tuple of (profile_dict, list_of_entry_dicts, list_of_warnings)
    """
    content = filepath.read_text()

    # Extract persona ID from filename
    id_match = re.search(r"persona_([a-f0-9]+)\.md", filepath.name)
    persona_id = id_match.group(1) if id_match else filepath.stem

    profile = parse_wrangled_persona_profile(content)
    profile["persona_id"] = persona_id

    entries, warnings = parse_wrangled_entries(content, filepath.name)

    return profile, entries, warnings
