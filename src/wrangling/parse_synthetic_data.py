"""Parse synthetic persona markdown files into clean format for Judge labeling.

This module extracts structured data from persona_*.md files, stripping
generation metadata (tone, verbosity, etc.) and outputting clean markdown
optimized for LLM consumption in the Judge labeling pipeline.

Supports two filename formats:
- UUID-based (new): persona_a3f8b2c1.md (8-char hex UUID)
- Numeric (legacy): persona_001.md (3-digit number)

Usage (CLI - outputs markdown):
    # Process all synthetic data files (flat directory)
    python -m src.wrangling.parse_synthetic_data

    # Process specific file(s)
    python -m src.wrangling.parse_synthetic_data logs/synthetic_data/persona_a3f8b2c1.md

    Output: logs/wrangled/persona_*.md (flat directory, same filename)

Usage (library - returns DataFrame):
    from src.wrangling.parse_synthetic_data import parse_synthetic_data_dir
    df = parse_synthetic_data_dir("logs/synthetic_data")
"""

import re
from pathlib import Path

import polars as pl


def parse_persona_profile(content: str) -> dict:
    """Extract persona profile from markdown content.

    Supports two header formats:
    - UUID-based (new): "# Persona a3f8b2c1: [Name]"
    - Numeric (legacy): "# Persona 001: [Name]"

    Args:
        content: Full markdown file content

    Returns:
        Dict with name, age, profession, culture, core_values, bio
    """
    profile = {}

    # Extract name from title - supports both UUID and numeric formats
    # UUID format: "# Persona a3f8b2c1: [Name]"
    # Numeric format: "# Persona 001: [Name]"
    name_match = re.search(r"^# Persona [a-f0-9]+: (.+)$", content, re.MULTILINE)
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
    # Handles optional bold formatting: - Bio: or - **Bio:**
    bio_match = re.search(r"- (?:\*\*)?Bio:(?:\*\*)?\s*(.+?)(?=\n---|\n## |\Z)", content, re.DOTALL)
    profile["bio"] = bio_match.group(1).strip() if bio_match else None

    return profile


def _extract_field(content: str, field_name: str) -> str | None:
    """Extract a simple field value from markdown.

    Handles optional bold formatting around field name:
    - Age: 25-34           (no bold)
    - **Age:** 45-54       (bold with colon inside)
    """
    pattern = rf"- (?:\*\*)?{field_name}:(?:\*\*)?\s*(.+?)(?:\n|$)"
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
    # Handles colon inside OR outside bold: **Tone**: or **Tone:**
    initial_match = re.search(
        r"### Initial Entry\s*\n"
        r"\*\*Tone(?:\*\*:|:\*\*).*?\n\n"  # Skip metadata line (colon inside or outside)
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
        # Handles colon inside OR outside bold: **Trigger**: or **Trigger:**
        nudge_match = re.search(
            r'### Nudge.*?\n'
            r'\*\*Trigger(?:\*\*:|:\*\*).*?\n'
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
        # Handles colon inside OR outside bold: **Mode**: or **Mode:**
        response_match = re.search(
            r"### Response\s*\n"
            r"\*\*Mode(?:\*\*:|:\*\*).*?\n"  # Skip mode line (colon inside or outside)
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
        profile: Dict with persona_id (str UUID or int), name, age, profession,
                 culture, core_values, bio
        entries: List of entry dicts from parse_entries()

    Returns:
        Formatted markdown string for the entire persona
    """
    lines = []

    # Header - handle both string UUID and numeric persona_id
    persona_id = profile["persona_id"]
    if isinstance(persona_id, int):
        # Legacy numeric format
        lines.append(f"# Persona {persona_id:03d}: {profile['name']}")
    else:
        # UUID format
        lines.append(f"# Persona {persona_id}: {profile['name']}")
    lines.append("")

    # Profile section
    lines.append("## Profile")
    lines.append(f"- **Persona ID:** {persona_id}")
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


def write_wrangled_markdown(
    input_dir: str | Path,
    output_dir: str | Path,
    update_registry: bool = True,
) -> list[Path]:
    """Write wrangled persona files as clean markdown.

    Args:
        input_dir: Path to logs/synthetic_data/ directory (flat)
        output_dir: Path to output directory (e.g., logs/wrangled/)
        update_registry: If True, mark personas as wrangled in registry

    Returns:
        List of paths to written markdown files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    persona_files = sorted(input_path.glob("persona_*.md"))

    if not persona_files:
        raise FileNotFoundError(f"No persona_*.md files found in {input_dir}")

    written_files = []
    for filepath in persona_files:
        profile, entries = parse_persona_file(filepath)
        markdown = format_persona_markdown(profile, entries)

        output_file = output_path / filepath.name
        output_file.write_text(markdown)
        written_files.append(output_file)

        # Update registry to mark as wrangled
        if update_registry:
            try:
                from src.registry import update_stage

                update_stage(profile["persona_id"], "wrangled")
            except (ImportError, ValueError) as e:
                # Registry not available or persona not registered
                # (could be legacy data without registry)
                print(f"  Note: Could not update registry for {profile['persona_id']}: {e}")

    return written_files


def extract_persona_id(filename: str) -> str:
    """Extract persona ID from filename.

    Supports two formats:
    - UUID-based (new): persona_a3f8b2c1.md → "a3f8b2c1"
    - Numeric (legacy): persona_001.md → "001"

    Args:
        filename: Just the filename (not full path)

    Returns:
        Persona ID as string (UUID or zero-padded number)

    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    # Try UUID format first (8 hex chars)
    uuid_match = re.search(r"persona_([a-f0-9]{8})\.md", filename)
    if uuid_match:
        return uuid_match.group(1)

    # Fall back to numeric format
    num_match = re.search(r"persona_(\d+)\.md", filename)
    if num_match:
        return num_match.group(1)

    raise ValueError(f"Cannot extract persona ID from filename: {filename}")


def parse_persona_file(filepath: Path) -> tuple[dict, list[dict]]:
    """Parse a single persona markdown file.

    Supports both UUID-based and numeric filename formats.

    Args:
        filepath: Path to persona_*.md file

    Returns:
        Tuple of (profile_dict, list_of_entry_dicts)
    """
    content = filepath.read_text()

    # Extract persona ID from filename (supports both formats)
    persona_id = extract_persona_id(filepath.name)

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


def parse_synthetic_data_dir(data_dir: str | Path) -> pl.DataFrame:
    """Parse all persona files in a synthetic data directory (flat structure).

    Args:
        data_dir: Path to logs/synthetic_data/ directory (flat, no timestamp subfolder)

    Returns:
        Polars DataFrame with one row per entry, columns for persona and entry fields
    """
    data_path = Path(data_dir)
    persona_files = sorted(data_path.glob("persona_*.md"))

    if not persona_files:
        raise FileNotFoundError(f"No persona_*.md files found in {data_dir}")

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
    """CLI entry point.

    Usage:
        # Process all files in logs/synthetic_data/ (flat directory)
        python -m src.wrangling.parse_synthetic_data

        # Process specific directory
        python -m src.wrangling.parse_synthetic_data logs/synthetic_data

        # Process specific file(s) - pass full paths
        python -m src.wrangling.parse_synthetic_data logs/synthetic_data/persona_a3f8b2c1.md
    """
    import sys

    # Default directories (flat structure)
    input_dir = Path("logs/synthetic_data")
    output_dir = Path("logs/wrangled")

    if len(sys.argv) >= 2:
        arg = Path(sys.argv[1])
        if arg.is_file():
            # Single file mode - process just this file
            input_dir = arg.parent
            # For single file, we'll filter to just that file later
            single_file = arg
        else:
            input_dir = arg
            single_file = None
    else:
        single_file = None

    print(f"Parsing synthetic data from: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Write clean markdown files
    written_files = write_wrangled_markdown(input_dir, output_dir)

    # Generate DataFrame for summary stats
    df = parse_synthetic_data_dir(input_dir)

    print(f"\nWrangled {len(written_files)} personas with {len(df)} total entries")
    print(f"\nOutput files:")
    for f in written_files:
        print(f"  {f}")

    # Show registry status if available
    try:
        from src.registry import get_status

        status = get_status()
        print(f"\nRegistry status:")
        print(f"  Total personas: {status['total']}")
        print(f"  Wrangled: {status['wrangled']}")
        print(f"  Pending labeling: {status['pending_labeling']}")
    except (ImportError, FileNotFoundError):
        pass


if __name__ == "__main__":
    main()
