#!/usr/bin/env python3
"""Materialize a non-destructive v8pb target variant from paired reviews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vif.dataset import load_entries  # noqa: E402
from src.vif.drift_target import (  # noqa: E402
    apply_student_visible_target,
    build_full_trajectory_cases,
    build_paired_target_delta,
    finalize_audit_manifest,
    parse_aware_timestamp,
    sha256_file,
    sha256_json,
    target_split_sha256,
    validate_audit_manifest,
    validate_promotion_threshold_receipt,
    validate_review_bundle_cases,
    validate_target_manifest,
)

DEFAULT_MANIFEST = Path("config/evals/drift_v1_student_visible_v1.yaml")


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_manifest(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def materialize(args: argparse.Namespace) -> dict:
    manifest_path = _rooted(args.manifest)
    manifest = _read_manifest(manifest_path)
    source = manifest["source"]
    registry_path = _rooted(Path(source["registry_path"]))
    labels_path = _rooted(Path(source["consensus_labels_path"]))
    entries = load_entries(_rooted(Path(source["wrangled_dir"])))
    registry = pl.read_parquet(registry_path)
    _validated_manifest, split = validate_target_manifest(
        manifest_path, registry, entries
    )
    persona_ids = (
        split.development_persona_ids
        if args.cohort == "development"
        else split.promotion_persona_ids
    )
    expected_cases = build_full_trajectory_cases(
        entries,
        registry,
        persona_ids,
        split=args.cohort,
    )

    reviewer_packet_dir = _rooted(args.reviewer_packet_dir)
    parent_control_dir = _rooted(args.parent_control_dir)
    packet_path = reviewer_packet_dir / "blind_packet.json"
    schema_path = reviewer_packet_dir / "response_schema.json"
    key_path = parent_control_dir / "reconciliation_key.json"
    audit_manifest_path = parent_control_dir / "audit_manifest.json"
    packet = _read_json(packet_path)
    key = _read_json(key_path)
    audit_manifest = validate_audit_manifest(
        audit_manifest_path,
        root=ROOT,
        packet_path=packet_path,
        key_path=key_path,
        schema_path=schema_path,
        expected_cases=expected_cases,
        split=args.cohort,
    )
    validate_review_bundle_cases(
        packet=packet,
        reconciliation_key=key,
        expected_cases=expected_cases,
        split=args.cohort,
    )
    reviewer_a_path = _rooted(args.reviewer_a)
    reviewer_b_path = _rooted(args.reviewer_b)
    if reviewer_a_path.resolve() == reviewer_b_path.resolve():
        raise ValueError("Paired reviews must use distinct response files")
    reviewer_a = _read_json(reviewer_a_path)
    reviewer_b = _read_json(reviewer_b_path)
    if args.cohort == "promotion":
        threshold_receipt = validate_promotion_threshold_receipt(
            audit_manifest,
            root=ROOT,
        )
        receipt_time = parse_aware_timestamp(
            threshold_receipt["recorded_at"],
            field_name="threshold receipt",
        )
        review_times = [
            parse_aware_timestamp(
                reviewer["reviewed_at"],
                field_name="promotion reviewer submission",
            )
            for reviewer in (reviewer_a, reviewer_b)
        ]
        if receipt_time >= min(review_times):
            raise ValueError(
                "Promotion threshold receipt must be recorded before the first "
                "review submission"
            )
    base_labels = pl.read_parquet(labels_path)
    delta, summary = build_paired_target_delta(
        packet=packet,
        reconciliation_key=key,
        reviewer_a=reviewer_a,
        reviewer_b=reviewer_b,
        base_labels_df=base_labels,
        registry_df=registry,
        split=args.cohort,
        packet_sha256=sha256_file(packet_path),
    )
    variant = apply_student_visible_target(
        base_labels,
        delta,
        registry,
        retired_persona_ids=split.retired_persona_ids,
    )

    output_dir = parent_control_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_path = output_dir / "student_visible_target_delta.parquet"
    variant_path = output_dir / "student_visible_target_variant.parquet"
    delta.write_parquet(delta_path)
    variant.write_parquet(variant_path)
    finalize_audit_manifest(
        audit_manifest_path,
        root=ROOT,
        reviewer_a_path=reviewer_a_path,
        reviewer_a=reviewer_a,
        reviewer_b_path=reviewer_b_path,
        reviewer_b=reviewer_b,
    )
    summary_path = output_dir / "target_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                **summary,
                "packet_sha256": sha256_file(packet_path),
                "reconciliation_key_sha256": sha256_file(key_path),
                "reviewer_a_sha256": sha256_file(reviewer_a_path),
                "reviewer_b_sha256": sha256_file(reviewer_b_path),
                "target_manifest_sha256": sha256_file(manifest_path),
                "target_split_sha256": target_split_sha256(split),
                "review_cases_sha256": sha256_json(expected_cases),
                "target_delta_sha256": sha256_file(delta_path),
                "target_variant_sha256": sha256_file(variant_path),
                "audit_manifest_sha256": sha256_file(audit_manifest_path),
                "expected_case_count": len(expected_cases),
                "expected_entry_count": sum(
                    len(case["entries"]) for case in expected_cases
                ),
                "materialization_complete": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cohort", choices=("development", "promotion"), required=True)
    parser.add_argument("--reviewer-packet-dir", type=Path, required=True)
    parser.add_argument("--parent-control-dir", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    return parser


def main() -> None:
    summary = materialize(build_parser().parse_args())
    print(
        f"Materialized {summary['split']} target delta: {summary['entry_count']} "
        "entries; qualification agreement="
        f"{summary['qualification_agreement_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
