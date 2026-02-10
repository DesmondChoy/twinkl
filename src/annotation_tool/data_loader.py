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

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from src.wrangling.parse_wrangled_data import (
    ParseWarning,
    parse_wrangled_file,
)


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


def load_entries_with_warnings(
    wrangled_dir: str | Path = "logs/wrangled",
    registry_path: str | Path = "logs/registry/personas.parquet",
) -> LoadResult:
    """Load all entries from wrangled persona files with warning collection.

    Args:
        wrangled_dir: Path to the wrangled files directory
        registry_path: Path to the persona registry (for annotation_order)

    Returns:
        LoadResult containing the DataFrame and any parsing warnings.
        The DataFrame includes an 'annotation_order' column if available
        in the registry.

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

    df = pl.DataFrame(rows)

    # Try to merge annotation_order from registry
    registry_path = Path(registry_path)
    if registry_path.exists():
        try:
            registry_df = pl.read_parquet(registry_path)
            if "annotation_order" in registry_df.columns:
                # Join annotation_order by persona_id
                df = df.join(
                    registry_df.select(["persona_id", "annotation_order"]),
                    on="persona_id",
                    how="left",
                )
        except Exception:
            # Registry read failed, proceed without annotation_order
            pass

    return LoadResult(
        df=df,
        warnings=all_warnings,
        skipped_count=skipped_count,
    )


def get_ordered_entries(df: pl.DataFrame) -> list[dict]:
    """Get entries sorted by annotation_order (if available) then t_index.

    If annotation_order is not present or is null, falls back to persona_id sorting.
    This allows the annotation tool to display personas in a custom order set in
    the registry, enabling prioritization of specific personas (e.g., those with
    underrepresented values like Stimulation, Power, Security).

    Args:
        df: DataFrame from load_entries()

    Returns:
        List of entry dicts, each containing both persona context and entry content.
        Ordered by annotation_order (or persona_id as fallback), then by t_index.
    """
    # Check if annotation_order column exists and has non-null values
    if "annotation_order" in df.columns:
        # Use annotation_order as primary sort, falling back to persona_id for nulls
        sorted_df = df.sort(
            [
                pl.col("annotation_order").fill_null(999999),  # Nulls sort last
                "persona_id",  # Secondary sort for same annotation_order or nulls
                "t_index",
            ]
        )
    else:
        # Fallback to original behavior
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
