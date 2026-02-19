#!/usr/bin/env python3
"""One-time migration for legacy judge rationale schema.

This script normalizes legacy judge JSON labels that use a single `rationale`
string to the current `rationales` dict format and optionally rebuilds the
consolidated parquet output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a standalone script from repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.judge.consolidate import (
    consolidate_judge_labels,
    migrate_legacy_judge_rationales,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize legacy judge label schema (`rationale`) to current "
            "`rationales` format."
        )
    )
    parser.add_argument(
        "labels_dir",
        nargs="?",
        default="logs/judge_labels",
        help="Directory containing persona_*_labels.json files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report migration counts without writing JSON files",
    )
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Skip parquet rebuild after migration",
    )
    args = parser.parse_args()

    labels_dir = Path(args.labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")

    files_migrated, entries_migrated = migrate_legacy_judge_rationales(
        labels_dir=labels_dir,
        write_changes=not args.dry_run,
    )

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"[{mode}] Migrated {entries_migrated} entries across {files_migrated} files")

    if not args.skip_consolidate:
        if args.dry_run:
            print("Skipped parquet rebuild in dry-run mode")
            return

        df, errors = consolidate_judge_labels(
            labels_dir=labels_dir,
            output_path=labels_dir / "judge_labels.parquet",
            update_registry=False,
        )
        print(
            f"Rebuilt parquet: {labels_dir / 'judge_labels.parquet'} "
            f"({len(df)} rows, {df['persona_id'].n_unique()} personas)"
        )
        if errors:
            print(f"Warnings: {len(errors)}")
            for warning in errors[:10]:
                print(f"  - {warning}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
