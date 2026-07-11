#!/usr/bin/env python3
"""Build evidence-only paired-review packets for the v8pb drift target."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vif.dataset import load_entries  # noqa: E402
from src.vif.drift_target import (  # noqa: E402
    build_full_trajectory_cases,
    validate_target_manifest,
    write_review_bundle,
)

DEFAULT_MANIFEST = Path("config/evals/drift_v1_student_visible_v1.yaml")
DEFAULT_OUTPUT_ROOT = Path(
    "logs/experiments/artifacts/drift_target_twinkl_v8pb_20260711"
)


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _load_manifest(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def build_bundle(args: argparse.Namespace) -> dict:
    manifest_path = _rooted(args.manifest)
    manifest = _load_manifest(manifest_path)
    source = manifest["source"]
    registry_path = _rooted(Path(source["registry_path"]))
    entries_dir = _rooted(Path(source["wrangled_dir"]))
    consensus_path = _rooted(Path(source["consensus_labels_path"]))
    original_holdout_path = _rooted(Path(source["original_holdout_manifest"]))

    registry = pl.read_parquet(registry_path)
    entries = load_entries(entries_dir)
    _validated_manifest, split = validate_target_manifest(
        manifest_path, registry, entries
    )
    if args.cohort == "development":
        persona_ids = split.development_persona_ids
    elif args.cohort == "promotion":
        persona_ids = split.promotion_persona_ids
    else:
        raise ValueError(f"Unsupported cohort {args.cohort!r}")
    cases = build_full_trajectory_cases(
        entries, registry, persona_ids, split=args.cohort
    )

    output_dir = _rooted(args.output_dir) / args.cohort
    return write_review_bundle(
        output_dir=output_dir,
        root=ROOT,
        source_paths={
            "target_manifest": manifest_path,
            "registry": registry_path,
            "original_holdout_manifest": original_holdout_path,
            "consensus_labels_provenance": consensus_path,
        },
        cases=cases,
        split=args.cohort,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort", choices=("development", "promotion"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_bundle(args)
    print(
        f"Built {result['split']} review bundle: {result['case_count']} cases, "
        f"{result['entry_count']} displayed entries"
    )


if __name__ == "__main__":
    main()
