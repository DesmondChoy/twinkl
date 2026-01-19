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
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass
class ParseWarning:
    """Warning about a parsing issue that was handled gracefully."""

    file: str
    entry_index: int | None
    message: str


@dataclass
class LoadResult:
    """Result of loading entries with any warnings.

    Attributes:
        df: Polars DataFrame with loaded entries
        warnings: List of non-fatal warnings encountered during parsing
        skipped_count: Total number of entries that were skipped
    """

    df: pl.DataFrame
    warnings: list[ParseWarning] = field(default_factory=list)
    skipped_count: int = 0


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


def load_entries(wrangled_dir: str | Path = "logs/wrangled") -> pl.DataFrame:
    """Load all entries from wrangled persona files.

    Args:
        wrangled_dir: Path to the wrangled files directory

    Returns:
        Polars DataFrame with one row per entry, columns for persona and entry fields.

    Raises:
        FileNotFoundError: If no persona files are found in the directory.

    Note:
        Use load_entries_with_warnings() if you need access to parsing warnings.
    """
    result = load_entries_with_warnings(wrangled_dir)
    return result.df


def load_entries_with_warnings(wrangled_dir: str | Path = "logs/wrangled") -> LoadResult:
    """Load all entries from wrangled persona files with warning collection.

    Args:
        wrangled_dir: Path to the wrangled files directory

    Returns:
        LoadResult containing the DataFrame and any parsing warnings.

    Raises:
        FileNotFoundError: If no persona files are found in the directory.
    """
    wrangled_path = Path(wrangled_dir)
    persona_files = sorted(wrangled_path.glob("persona_*.md"))

    if not persona_files:
        raise FileNotFoundError(f"No persona_*.md files found in {wrangled_dir}")

    rows = []
    all_warnings = []
    skipped_count = 0

    for filepath in persona_files:
        try:
            profile, entries, warnings = parse_wrangled_file(filepath)
            all_warnings.extend(warnings)
            skipped_count += len(warnings)  # Each warning represents a skipped entry

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
        except Exception as e:
            # File-level error - add warning and continue
            all_warnings.append(
                ParseWarning(
                    file=filepath.name,
                    entry_index=None,
                    message=f"Failed to parse file: {type(e).__name__}: {e}",
                )
            )

    if not rows:
        raise FileNotFoundError(
            f"No valid entries found in {wrangled_dir}. "
            f"{len(all_warnings)} files had parsing errors."
        )

    return LoadResult(
        df=pl.DataFrame(rows),
        warnings=all_warnings,
        skipped_count=skipped_count,
    )


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
