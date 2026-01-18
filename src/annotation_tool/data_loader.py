"""Load journal entries from wrangled files for annotation.

This module provides functions to load entries from wrangled persona files,
ordered by persona then t_index for sequential annotation.

The wrangled format is simpler than synthetic format:
- Entry text directly after date header (no ### Initial Entry section)
- **Nudge:** and **Response:** markers

Usage:
    from src.annotation_tool.data_loader import load_entries, get_ordered_entries

    df = load_entries()
    entries = get_ordered_entries(df)
"""

import re
from pathlib import Path

import polars as pl


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


def parse_wrangled_entries(content: str) -> list[dict]:
    """Extract all journal entries from wrangled markdown content.

    Wrangled format:
        ## Entry N - YYYY-MM-DD

        Entry text here...

        **Nudge:** "nudge text"

        **Response:** response text

    Args:
        content: Full markdown file content

    Returns:
        List of entry dicts with date, initial_entry, nudge_text, response_text
    """
    entries = []

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

            # Extract nudge text (quoted)
            nudge_match = re.search(r'\*\*Nudge:\*\* "([^"]+)"', entry_content)
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

        entries.append(entry)

    return entries


def parse_wrangled_file(filepath: Path) -> tuple[dict, list[dict]]:
    """Parse a single wrangled persona markdown file.

    Args:
        filepath: Path to persona_*.md file

    Returns:
        Tuple of (profile_dict, list_of_entry_dicts)
    """
    content = filepath.read_text()

    # Extract persona ID from filename
    id_match = re.search(r"persona_([a-f0-9]+)\.md", filepath.name)
    persona_id = id_match.group(1) if id_match else filepath.stem

    profile = parse_wrangled_persona_profile(content)
    profile["persona_id"] = persona_id

    entries = parse_wrangled_entries(content)

    return profile, entries


def load_entries(wrangled_dir: str | Path = "logs/wrangled") -> pl.DataFrame:
    """Load all entries from wrangled persona files.

    Args:
        wrangled_dir: Path to the wrangled files directory

    Returns:
        Polars DataFrame with one row per entry, columns for persona and entry fields.
    """
    wrangled_path = Path(wrangled_dir)
    persona_files = sorted(wrangled_path.glob("persona_*.md"))

    if not persona_files:
        raise FileNotFoundError(f"No persona_*.md files found in {wrangled_dir}")

    rows = []
    for filepath in persona_files:
        profile, entries = parse_wrangled_file(filepath)

        for entry in entries:
            row = {
                # Persona fields
                "persona_id": profile["persona_id"],
                "persona_name": profile["name"],
                "persona_age": profile["age"],
                "persona_profession": profile["profession"],
                "persona_culture": profile["culture"],
                "persona_core_values": profile["core_values"],
                "persona_bio": profile["bio"],
                # Entry fields
                "t_index": entry["t_index"],
                "date": entry["date"],
                "initial_entry": entry["initial_entry"],
                "nudge_text": entry["nudge_text"],
                "response_text": entry["response_text"],
                "has_nudge": entry["has_nudge"],
                "has_response": entry["has_response"],
            }
            rows.append(row)

    return pl.DataFrame(rows)


def get_ordered_entries(df: pl.DataFrame) -> list[dict]:
    """Get entries sorted by persona_id then t_index.

    Args:
        df: DataFrame from load_entries()

    Returns:
        List of entry dicts, each containing both persona context and entry content.
        Ordered by persona_id, then by t_index within each persona.
    """
    sorted_df = df.sort(["persona_id", "t_index"])
    return sorted_df.to_dicts()


def get_total_entries(wrangled_dir: str | Path = "logs/wrangled") -> int:
    """Get the total number of entries across all personas.

    Args:
        wrangled_dir: Path to the wrangled files directory

    Returns:
        Total entry count
    """
    df = load_entries(wrangled_dir)
    return len(df)
