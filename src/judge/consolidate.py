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
from copy import deepcopy
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from src.models.judge import SCHWARTZ_VALUE_ORDER, PersonaLabels


def _normalize_legacy_entry_schema(entry: dict) -> tuple[dict, bool]:
    """Normalize legacy entry shape to current schema.

    Legacy judge outputs may use:
    - "rationale": str (single explanation for all non-zero scores)

    Current schema expects:
    - "rationales": dict[str, str]

    Returns:
        Tuple of (normalized_entry, changed_flag)
    """
    normalized = deepcopy(entry)

    # Already in current schema, nothing to do.
    if normalized.get("rationales"):
        normalized.pop("rationale", None)
        return normalized, False

    rationale_text = normalized.get("rationale")
    if not isinstance(rationale_text, str) or not rationale_text.strip():
        normalized.pop("rationale", None)
        return normalized, False

    scores = normalized.get("scores", {})
    if not isinstance(scores, dict):
        normalized.pop("rationale", None)
        return normalized, False

    # Broadcast the legacy single rationale string to each non-zero dimension.
    rationales = {
        key: rationale_text.strip()
        for key in SCHWARTZ_VALUE_ORDER
        if key in scores and scores[key] != 0
    }

    normalized["rationales"] = rationales if rationales else None
    normalized.pop("rationale", None)
    return normalized, True


def normalize_legacy_persona_schema(data: dict) -> tuple[dict, int]:
    """Normalize one persona label payload to current schema."""
    normalized = deepcopy(data)
    labels = normalized.get("labels", [])

    migrated_entries = 0
    if isinstance(labels, list):
        normalized_labels = []
        for label in labels:
            if isinstance(label, dict):
                migrated_label, changed = _normalize_legacy_entry_schema(label)
                migrated_entries += int(changed)
                normalized_labels.append(migrated_label)
            else:
                normalized_labels.append(label)
        normalized["labels"] = normalized_labels

    return normalized, migrated_entries


def migrate_legacy_judge_rationales(
    labels_dir: str | Path,
    write_changes: bool = True,
) -> tuple[int, int]:
    """Normalize legacy `rationale` fields in judge JSON files.

    Args:
        labels_dir: Directory containing persona_*_labels.json files
        write_changes: If True, persist normalized JSON files in place

    Returns:
        Tuple of (files_migrated, entries_migrated)
    """
    labels_path = Path(labels_dir)
    json_files = sorted(labels_path.glob("persona_*_labels.json"))

    files_migrated = 0
    entries_migrated = 0

    for json_file in json_files:
        with open(json_file) as f:
            data = json.load(f)

        normalized, migrated_entries = normalize_legacy_persona_schema(data)
        if migrated_entries <= 0:
            continue

        files_migrated += 1
        entries_migrated += migrated_entries

        if write_changes:
            with open(json_file, "w") as f:
                json.dump(normalized, f, indent=2)
                f.write("\n")

    return files_migrated, entries_migrated


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

            data, migrated_entries = normalize_legacy_persona_schema(data)
            if migrated_entries > 0:
                errors.append(
                    f"{json_file.name}: Migrated {migrated_entries} legacy rationale entr"
                    f"{'y' if migrated_entries == 1 else 'ies'} to `rationales` schema in-memory"
                )

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
