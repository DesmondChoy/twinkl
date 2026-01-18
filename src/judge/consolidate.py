"""Consolidate judge label JSON files into a single parquet file.

This module validates and merges the JSON output files from judge subagents
into a single parquet file for downstream processing. Also updates the
central registry to mark personas as labeled.

Usage (CLI):
    # Process all files in logs/judge_labels/ (default)
    python -m src.judge.consolidate

    # Process specific directory
    python -m src.judge.consolidate logs/judge_labels

Usage (library):
    from src.judge.consolidate import consolidate_judge_labels

    df, errors = consolidate_judge_labels(
        labels_dir="logs/judge_labels",
        output_path="logs/judge_labels/judge_labels.parquet"
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
    update_registry: bool = True,
) -> tuple[pl.DataFrame, list[str]]:
    """Consolidate persona JSON files into a single parquet file.

    Args:
        labels_dir: Directory containing persona_*_labels.json files
        output_path: Path to write parquet file. If None, writes to
                     labels_dir/judge_labels.parquet
        update_registry: If True, mark personas as labeled in registry

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
    validated_persona_ids: list[str] = []

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            # Validate with Pydantic
            validated = PersonaLabels.model_validate(data)

            # Track validated persona for registry update
            validated_persona_ids.append(validated.persona_id)

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

                # Add rationales as JSON string (None if no rationales)
                row["rationales_json"] = (
                    json.dumps(entry_label.rationales)
                    if entry_label.rationales
                    else None
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

    # Update registry for validated personas
    if update_registry:
        try:
            from src.registry import update_stage

            for persona_id in validated_persona_ids:
                try:
                    update_stage(persona_id, "labeled")
                except ValueError as e:
                    # Persona not in registry (could be legacy data)
                    errors.append(f"Registry: {persona_id} - {e}")
        except ImportError:
            errors.append("Registry module not available")

    return df, errors


def main():
    """CLI entry point.

    Usage:
        # Process all files in logs/judge_labels/ (default)
        python -m src.judge.consolidate

        # Process specific directory
        python -m src.judge.consolidate logs/judge_labels
    """
    import sys

    # Default to logs/judge_labels/ (flat structure)
    labels_dir = Path("logs/judge_labels")

    if len(sys.argv) >= 2:
        labels_dir = Path(sys.argv[1])

    print(f"Consolidating labels from: {labels_dir}")

    try:
        df, errors = consolidate_judge_labels(labels_dir)

        print(f"\nConsolidated {len(df)} entries from {df['persona_id'].n_unique()} personas")
        print(f"Output: {labels_dir / 'judge_labels.parquet'}")

        if errors:
            print(f"\nWarnings ({len(errors)} issues):")
            for error in errors:
                print(f"  - {error}")

        print("\nSchema:")
        print(df.schema)

        # Show registry status if available
        try:
            from src.registry import get_status

            status = get_status()
            print(f"\nRegistry status:")
            print(f"  Total personas: {status['total']}")
            print(f"  Labeled: {status['labeled']}")
            print(f"  Pending labeling: {status['pending_labeling']}")
        except (ImportError, FileNotFoundError):
            pass

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
