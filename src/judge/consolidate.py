"""Consolidate judge label JSON files into a single parquet file.

This module validates and merges the JSON output files from judge subagents
into a single parquet file for downstream processing.

Usage (CLI):
    python -m src.judge.consolidate logs/judge_labels/2026-01-15_10-30-00

Usage (library):
    from src.judge.consolidate import consolidate_judge_labels

    df, errors = consolidate_judge_labels(
        labels_dir="logs/judge_labels/2026-01-15_10-30-00",
        output_path="logs/judge_labels/2026-01-15_10-30-00/judge_labels.parquet"
    )
"""

import json
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from src.models.judge import SCHWARTZ_VALUE_ORDER, PersonaLabels


def consolidate_judge_labels(
    labels_dir: str | Path,
    output_path: str | Path | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """Consolidate persona JSON files into a single parquet file.

    Args:
        labels_dir: Directory containing persona_*_labels.json files
        output_path: Path to write parquet file. If None, writes to
                     labels_dir/judge_labels.parquet

    Returns:
        Tuple of (DataFrame with all labels, list of validation error messages)
    """
    labels_path = Path(labels_dir)

    if output_path is None:
        output_path = labels_path / "judge_labels.parquet"
    else:
        output_path = Path(output_path)

    # Find all persona label files
    json_files = sorted(labels_path.glob("persona_*_labels.json"))

    if not json_files:
        raise FileNotFoundError(f"No persona_*_labels.json files found in {labels_dir}")

    rows: list[dict] = []
    errors: list[str] = []

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Validate with Pydantic
            validated = PersonaLabels.model_validate(data)

            # Extract rows
            for entry_label in validated.labels:
                row = {
                    "persona_id": validated.persona_id,
                    "t_index": entry_label.t_index,
                    "date": entry_label.date,
                    "alignment_vector": entry_label.scores.to_vector(),
                }

                # Add individual alignment columns
                for value_name in SCHWARTZ_VALUE_ORDER:
                    row[f"alignment_{value_name}"] = getattr(
                        entry_label.scores, value_name
                    )

                rows.append(row)

        except json.JSONDecodeError as e:
            errors.append(f"{json_file.name}: Invalid JSON - {e}")
        except ValidationError as e:
            errors.append(f"{json_file.name}: Validation failed - {e}")

    if not rows:
        raise ValueError(f"No valid labels found. Errors: {errors}")

    # Create DataFrame
    df = pl.DataFrame(rows)

    # Write parquet
    df.write_parquet(output_path)

    return df, errors


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.judge.consolidate <labels_directory>")
        print(
            "Example: python -m src.judge.consolidate logs/judge_labels/2026-01-15_10-30-00"
        )
        sys.exit(1)

    labels_dir = Path(sys.argv[1])

    print(f"Consolidating labels from: {labels_dir}")

    try:
        df, errors = consolidate_judge_labels(labels_dir)

        print(f"\nConsolidated {len(df)} entries from {df['persona_id'].n_unique()} personas")
        print(f"Output: {labels_dir / 'judge_labels.parquet'}")

        if errors:
            print(f"\nWarnings ({len(errors)} files had issues):")
            for error in errors:
                print(f"  - {error}")

        print("\nSchema:")
        print(df.schema)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
