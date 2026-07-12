#!/usr/bin/env python3
"""Prepare and reconcile the twinkl-748 Hedonism hard-set review bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vif.hedonism_hard_set import (  # noqa: E402
    build_review_bundle,
    load_candidate_spec,
    materialize_reviewed_hard_set,
)

DEFAULT_SPEC = Path("config/evals/twinkl_748_hedonism_hard_set_v1.yaml")


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate the candidate spec")
    validate.add_argument("--spec", type=Path, default=DEFAULT_SPEC)

    prepare = commands.add_parser("prepare", help="Write a blinded review bundle")
    prepare.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--random-seed", type=int, default=748)

    materialize = commands.add_parser(
        "materialize", help="Reconcile two completed reviewer responses"
    )
    materialize.add_argument("--bundle-dir", type=Path, required=True)
    materialize.add_argument("--reviewer-a", type=Path, required=True)
    materialize.add_argument("--reviewer-b", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        spec = load_candidate_spec(_rooted(args.spec))
        print(
            json.dumps(
                {
                    "target_version": spec["target_version"],
                    "pair_count": len(spec["pairs"]),
                    "entry_count": 2 * len(spec["pairs"]),
                    "families": sorted({pair["family"] for pair in spec["pairs"]}),
                },
                indent=2,
            )
        )
        return
    if args.command == "prepare":
        manifest = build_review_bundle(
            spec_path=_rooted(args.spec),
            output_dir=_rooted(args.output_dir),
            root=ROOT,
            random_seed=args.random_seed,
        )
        print(json.dumps(manifest, indent=2))
        return
    summary = materialize_reviewed_hard_set(
        bundle_dir=_rooted(args.bundle_dir),
        reviewer_a_path=_rooted(args.reviewer_a),
        reviewer_b_path=_rooted(args.reviewer_b),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
