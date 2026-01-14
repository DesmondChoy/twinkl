"""Parse synthetic persona markdown files into clean format for Judge labeling.

This module extracts structured data from persona_*.md files, stripping
generation metadata (tone, verbosity, etc.) and outputting clean markdown
optimized for LLM consumption in the Judge labeling pipeline.

Usage (CLI - outputs markdown):
    python -m src.wrangling.parse_synthetic_data logs/synthetic_data/2026-01-09_09-37-09

    Output: logs/wrangled/<timestamp>/persona_*.md (one file per persona)

Usage (library - returns DataFrame):
    from src.wrangling.parse_synthetic_data import parse_synthetic_data_run
    df = parse_synthetic_data_run("logs/synthetic_data/2026-01-09_09-37-09")
"""

import re
from pathlib import Path

import polars as pl


def parse_persona_profile(content: str) -> dict:
    """Extract persona profile from markdown content.

    Args:
        content: Full markdown file content

    Returns:
        Dict with name, age, profession, culture, core_values, bio
    """
    profile = {}

    # Extract name from title: "# Persona XXX: [Name]"
    name_match = re.search(r"^# Persona \d+: (.+)$", content, re.MULTILINE)
    profile["name"] = name_match.group(1).strip() if name_match else None

    # Extract profile fields
    profile["age"] = _extract_field(content, "Age")
    profile["profession"] = _extract_field(content, "Profession")
    profile["culture"] = _extract_field(content, "Culture")

    # Core values can be comma-separated
    core_values_str = _extract_field(content, "Core Values")
    if core_values_str:
        profile["core_values"] = [v.strip() for v in core_values_str.split(",")]
    else:
        profile["core_values"] = []

    # Bio is multi-line, extract everything after "- Bio: " until next section
    bio_match = re.search(r"- Bio: (.+?)(?=\n---|\n## |\Z)", content, re.DOTALL)
    profile["bio"] = bio_match.group(1).strip() if bio_match else None

    return profile


def _extract_field(content: str, field_name: str) -> str | None:
    """Extract a simple field value from markdown."""
    pattern = rf"- {field_name}: (.+?)(?:\n|$)"
    match = re.search(pattern, content)
    return match.group(1).strip() if match else None


def parse_entries(content: str) -> list[dict]:
    """Extract all journal entries from markdown content.

    Args:
        content: Full markdown file content

    Returns:
        List of entry dicts with date, initial_entry, nudge_text, response_text, flags
    """
    entries = []

    # Split by entry headers: "## Entry N - [Date]"
    entry_pattern = r"## Entry \d+ - (\d{4}-\d{2}-\d{2})"
    entry_splits = re.split(entry_pattern, content)

    # entry_splits alternates: [preamble, date1, content1, date2, content2, ...]
    for i in range(1, len(entry_splits), 2):
        date = entry_splits[i]
        entry_content = entry_splits[i + 1] if i + 1 < len(entry_splits) else ""

        entry = parse_single_entry(date, entry_content)
        entry["t_index"] = len(entries)  # 0-based index
        entries.append(entry)

    return entries


def parse_single_entry(date: str, content: str) -> dict:
    """Parse a single entry's content.

    Args:
        date: Entry date string (YYYY-MM-DD)
        content: Entry markdown content (everything after date header)

    Returns:
        Dict with date, initial_entry, nudge_text, response_text, has_nudge, has_response
    """
    entry = {
        "date": date,
        "initial_entry": None,
        "nudge_text": None,
        "response_text": None,
        "has_nudge": False,
        "has_response": False,
    }

    # Check for no-nudge marker
    has_no_nudge_marker = "*(No nudge for this entry)*" in content

    # Check for no-response marker
    has_no_response_marker = "*(No response - persona did not reply to nudge)*" in content

    # --- Extract Initial Entry ---
    # Pattern: After "### Initial Entry" and metadata line, before "### Nudge" or markers
    initial_match = re.search(
        r"### Initial Entry\s*\n"
        r"\*\*Tone\*\*:.*?\n\n"  # Skip the metadata line
        r"(.+?)"  # Capture the content
        r"(?=\n### Nudge|\n\*\(No nudge|\n---|\Z)",  # Stop before nudge section or markers
        content,
        re.DOTALL,
    )
    if initial_match:
        entry["initial_entry"] = initial_match.group(1).strip()

    # --- Extract Nudge Text ---
    if not has_no_nudge_marker:
        # Look for quoted text after "### Nudge" section
        # Pattern: ### Nudge (category)\n**Trigger**: ...\n"quoted text"
        nudge_match = re.search(
            r'### Nudge.*?\n'
            r'\*\*Trigger\*\*:.*?\n'
            r'"([^"]+)"',  # Capture the quoted nudge text
            content,
            re.DOTALL,
        )
        if nudge_match:
            entry["nudge_text"] = nudge_match.group(1).strip()
            entry["has_nudge"] = True

    # --- Extract Response Text ---
    if entry["has_nudge"] and not has_no_response_marker:
        # Look for content after "### Response" and mode line
        response_match = re.search(
            r"### Response\s*\n"
            r"\*\*Mode\*\*:.*?\n"  # Skip the mode line
            r"(.+?)"  # Capture the response content
            r"(?=\n---|\n## |\Z)",  # Stop before section break or next entry
            content,
            re.DOTALL,
        )
        if response_match:
            entry["response_text"] = response_match.group(1).strip()
            entry["has_response"] = True

    return entry


# --- Markdown Formatting Functions ---


def format_entry_markdown(entry: dict) -> str:
    """Format a single entry as clean markdown for Judge consumption.

    Minimal format: only include content that exists. Absence of nudge/response
    sections is self-evident — no markers needed.

    Args:
        entry: Dict with t_index, date, initial_entry, nudge_text, response_text,
               has_nudge, has_response

    Returns:
        Formatted entry markdown (without trailing separators)
    """
    lines = []

    # Entry header with t_index for consistent referencing
    lines.append(f"## Entry {entry['t_index']} - {entry['date']}")
    lines.append("")

    # Entry content (no label needed — it's obvious this is the entry)
    lines.append(entry["initial_entry"] or "")

    # Only add nudge/response sections if they exist
    if entry["has_nudge"]:
        lines.append("")
        lines.append(f'**Nudge:** "{entry["nudge_text"]}"')

        if entry["has_response"]:
            lines.append("")
            lines.append(f"**Response:** {entry['response_text']}")

    # No markers for absence — silence is the signal

    return "\n".join(lines)


def format_persona_markdown(profile: dict, entries: list[dict]) -> str:
    """Format a persona's data as clean markdown for Judge consumption.

    Strips generation metadata (tone, verbosity, etc.) and keeps only
    the content needed for Judge labeling.

    Args:
        profile: Dict with persona_id, name, age, profession, culture, core_values, bio
        entries: List of entry dicts from parse_entries()

    Returns:
        Formatted markdown string for the entire persona
    """
    lines = []

    # Header
    lines.append(f"# Persona {profile['persona_id']:03d}: {profile['name']}")
    lines.append("")

    # Profile section
    lines.append("## Profile")
    lines.append(f"- **Name:** {profile['name']}")
    lines.append(f"- **Age:** {profile['age']}")
    lines.append(f"- **Profession:** {profile['profession']}")
    lines.append(f"- **Culture:** {profile['culture']}")
    lines.append(f"- **Core Values:** {', '.join(profile['core_values'])}")
    lines.append(f"- **Bio:** {profile['bio']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Entries
    for entry in entries:
        lines.append(format_entry_markdown(entry))
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def write_wrangled_markdown(run_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Write wrangled persona files as clean markdown.

    Args:
        run_dir: Path to logs/synthetic_data/<timestamp>/ directory
        output_dir: Path to output directory (e.g., logs/wrangled/<timestamp>/)

    Returns:
        List of paths to written markdown files
    """
    run_path = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    persona_files = sorted(run_path.glob("persona_*.md"))

    if not persona_files:
        raise FileNotFoundError(f"No persona_*.md files found in {run_dir}")

    written_files = []
    for filepath in persona_files:
        profile, entries = parse_persona_file(filepath)
        markdown = format_persona_markdown(profile, entries)

        output_file = output_path / filepath.name
        output_file.write_text(markdown)
        written_files.append(output_file)

    return written_files


def parse_persona_file(filepath: Path) -> tuple[dict, list[dict]]:
    """Parse a single persona markdown file.

    Args:
        filepath: Path to persona_*.md file

    Returns:
        Tuple of (profile_dict, list_of_entry_dicts)
    """
    content = filepath.read_text()

    # Extract persona ID from filename: persona_001.md -> 1
    id_match = re.search(r"persona_(\d+)\.md", filepath.name)
    persona_id = int(id_match.group(1)) if id_match else 0

    profile = parse_persona_profile(content)
    profile["persona_id"] = persona_id

    entries = parse_entries(content)

    return profile, entries


def parse_synthetic_data_run(run_dir: str | Path) -> pl.DataFrame:
    """Parse all persona files in a synthetic data run directory.

    Args:
        run_dir: Path to logs/synthetic_data/<timestamp>/ directory

    Returns:
        Polars DataFrame with one row per entry, columns for persona and entry fields
    """
    run_path = Path(run_dir)
    persona_files = sorted(run_path.glob("persona_*.md"))

    if not persona_files:
        raise FileNotFoundError(f"No persona_*.md files found in {run_dir}")

    rows = []
    for filepath in persona_files:
        profile, entries = parse_persona_file(filepath)

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

    df = pl.DataFrame(rows)
    return df


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.wrangling.parse_synthetic_data <run_directory>")
        print("Example: python -m src.wrangling.parse_synthetic_data logs/synthetic_data/2026-01-09_09-37-09")
        sys.exit(1)

    run_dir = Path(sys.argv[1])

    # Extract timestamp folder name and build output path in logs/wrangled/
    # e.g., logs/synthetic_data/2026-01-09_09-37-09 → logs/wrangled/2026-01-09_09-37-09
    timestamp_folder = run_dir.name
    output_dir = Path("logs/wrangled") / timestamp_folder

    print(f"Parsing synthetic data from: {run_dir}")

    # Write clean markdown files (one per persona)
    written_files = write_wrangled_markdown(run_dir, output_dir)

    # Also generate DataFrame for summary stats
    df = parse_synthetic_data_run(run_dir)

    print(f"\nWrangled {len(written_files)} personas with {len(df)} total entries")
    print(f"\nOutput files:")
    for f in written_files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
