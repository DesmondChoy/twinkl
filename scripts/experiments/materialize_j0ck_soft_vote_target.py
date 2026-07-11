#!/usr/bin/env python3
"""Materialize the current-contract twinkl-j0ck hybrid soft-vote target."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vif.security_target import build_full_corpus_security_target  # noqa: E402
from src.vif.soft_vote_target import (  # noqa: E402
    SOFT_TARGET_CLASS_ORDER,
    SOFT_TARGET_REGIME,
    SOFT_TARGET_VALUE_ORDER,
    build_hybrid_soft_vote_target,
    read_jsonl,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--twinkl-754-dir",
        type=Path,
        default=Path("logs/exports/twinkl_754"),
    )
    parser.add_argument(
        "--consensus-labels",
        type=Path,
        default=Path("logs/judge_labels/consensus_labels.parquet"),
    )
    parser.add_argument(
        "--base-labels",
        type=Path,
        default=Path("logs/judge_labels/judge_labels.parquet"),
    )
    parser.add_argument(
        "--security-manifest",
        type=Path,
        default=Path(
            "logs/exports/twinkl_a30f_active_critic_state_full_v1/"
            "active_critic_state_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--security-results-dir",
        type=Path,
        default=Path(
            "logs/exports/twinkl_a30f_active_critic_state_full_v1/results"
        ),
    )
    parser.add_argument(
        "--repaired-security-labels",
        type=Path,
        default=Path(
            "logs/exports/twinkl_a30f_security_target_full_v1/"
            "security_repaired_labels.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "logs/exports/twinkl_j0ck_soft_vote_target_v1/"
            "hybrid_soft_vote_labels.parquet"
        ),
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=None,
        help="Defaults to target_provenance.json beside --output.",
    )
    args = parser.parse_args()

    manifest_path = args.twinkl_754_dir / "manifest.csv"
    twinkl_pass_paths = {
        index: args.twinkl_754_dir / "results" / f"pass_{index}_results.jsonl"
        for index in range(1, 6)
    }
    security_pass_paths = {
        index: args.security_results_dir / f"pass_{index}_results.jsonl"
        for index in (1, 2, 3)
    }
    security_tiebreak_path = args.security_results_dir / "tiebreak_results.jsonl"
    base_labels = pl.read_parquet(args.base_labels)
    security_target, rebuilt_repaired_labels = build_full_corpus_security_target(
        base_labels,
        active_state_manifest=read_jsonl(args.security_manifest),
        review_passes={
            index: read_jsonl(path)
            for index, path in security_pass_paths.items()
        },
        tiebreak_results=read_jsonl(security_tiebreak_path),
    )
    committed_repaired_labels = pl.read_parquet(args.repaired_security_labels)
    if not rebuilt_repaired_labels.equals(committed_repaired_labels, null_equal=True):
        raise ValueError(
            "Rebuilt Security labels do not match the committed repaired artifact."
        )

    target = build_hybrid_soft_vote_target(
        twinkl_754_manifest=pl.read_csv(manifest_path),
        twinkl_754_vote_passes={
            index: read_jsonl(path)
            for index, path in twinkl_pass_paths.items()
        },
        consensus_labels=pl.read_parquet(args.consensus_labels),
        repaired_security_labels=committed_repaired_labels,
        security_vote_target=security_target,
    )
    if args.output.exists():
        existing_target = pl.read_parquet(args.output)
        if not existing_target.equals(target, null_equal=True):
            raise FileExistsError(
                f"Existing target does not match rebuilt evidence: {args.output}"
            )
        action = "Verified"
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        target.write_parquet(args.output)
        action = "Wrote"

    provenance_output = args.provenance_output or (
        args.output.parent / "target_provenance.json"
    )
    input_paths = {
        "twinkl_754_manifest": manifest_path,
        **{
            f"twinkl_754_pass_{index}": path
            for index, path in twinkl_pass_paths.items()
        },
        "twinkl_754_consensus_labels": args.consensus_labels,
        "historical_base_labels": args.base_labels,
        "security_active_state_manifest": args.security_manifest,
        **{
            f"security_review_pass_{index}": path
            for index, path in security_pass_paths.items()
        },
        "security_tiebreak_results": security_tiebreak_path,
        "security_repaired_labels": args.repaired_security_labels,
    }
    provenance = {
        "schema_version": 1,
        "target_regime": SOFT_TARGET_REGIME,
        "class_order": list(SOFT_TARGET_CLASS_ORDER),
        "value_order": list(SOFT_TARGET_VALUE_ORDER),
        "input_files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
        "output": {
            "path": str(args.output),
            "sha256": _sha256(args.output),
            "row_count": target.height,
            "soft_target_length": len(SOFT_TARGET_CLASS_ORDER)
            * len(SOFT_TARGET_VALUE_ORDER),
        },
    }
    serialized = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    if provenance_output.exists():
        if provenance_output.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"Existing provenance does not match rebuilt evidence: {provenance_output}"
            )
    else:
        provenance_output.write_text(serialized, encoding="utf-8")

    print(
        f"{action} {target.height} hybrid soft-vote rows: {args.output}\n"
        f"Verified provenance: {provenance_output}"
    )


if __name__ == "__main__":
    main()
