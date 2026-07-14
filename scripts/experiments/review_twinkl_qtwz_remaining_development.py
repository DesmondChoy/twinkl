#!/usr/bin/env python3
"""Prepare the blinded review of Twinkl's remaining development cases."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vif.dataset import load_entries  # noqa: E402
from src.vif.drift_candidate_review import (  # noqa: E402
    ReviewProtocol,
    artifact_hashes,
    build_blind_shard,
    build_review_response_schema,
    sha256_file,
    sha256_json,
    shard_review_cases,
)
from src.vif.drift_target import (  # noqa: E402
    TargetSplit,
    build_full_trajectory_cases,
    derive_target_split,
)

DEFAULT_CONFIG = Path("config/evals/twinkl_qtwz_complete_development_review_v1.yaml")
DEFAULT_OUTPUT = Path(
    "logs/experiments/artifacts/twinkl_qtwz_complete_development_review_20260714"
)


def _rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    if payload.get("schema_version") != 1:
        raise ValueError("Review config must use schema_version: 1")
    if payload.get("analysis_role") != "development_only":
        raise ValueError("Remaining-case review must be development-only")
    return payload


def _protocol(config: dict[str, Any]) -> ReviewProtocol:
    review = config["review"]
    return ReviewProtocol(
        cohort_version=str(config["cohort_version"]),
        review_schema_version=str(review["review_schema_version"]),
        review_prompt_version=str(review["reviewer_prompt_version"]),
        adjudication_schema_version=str(review["adjudication_schema_version"]),
        adjudication_prompt_version=str(review["adjudication_prompt_version"]),
    )


def _historical_split_map(split: TargetSplit) -> dict[str, str]:
    groups = {
        "training": split.training_persona_ids,
        "development": split.development_persona_ids,
        "retired": split.retired_persona_ids,
        "former_promotion": split.promotion_persona_ids,
    }
    result: dict[str, str] = {}
    for split_name, persona_ids in groups.items():
        for persona_id in persona_ids:
            if persona_id in result:
                raise ValueError(
                    f"Persona appears in two historical splits: {persona_id}"
                )
            result[persona_id] = split_name
    return result


def _source_files(paths: dict[str, Path]) -> list[Path]:
    files = [path for path in paths.values() if path.is_file()]
    wrangled_files = sorted(paths["wrangled_dir"].glob("persona_*.md"))
    if not wrangled_files:
        raise FileNotFoundError("No wrangled persona files found")
    return [*files, *wrangled_files]


def _validate_expected_counts(
    *,
    config: dict[str, Any],
    reviewed_union: list[dict[str, Any]],
    selected: pl.DataFrame,
) -> None:
    expected = config["expected_counts"]
    actual = {
        "reviewed_union_cases": len(reviewed_union),
        "remaining_cases": selected.height,
        "remaining_personas": selected["persona_id"].n_unique(),
        "remaining_entry_core_value_combinations": int(
            selected["trajectory_length"].sum()
        ),
    }
    for key, value in actual.items():
        if int(expected[key]) != value:
            raise ValueError(
                f"expected_counts.{key}={expected[key]!r}; live value is {value}"
            )
    for field in ("historical_splits", "dimensions"):
        column = "historical_split" if field == "historical_splits" else "dimension"
        observed = {
            str(row[column]): int(row["len"])
            for row in selected.group_by(column).len().to_dicts()
        }
        configured = {str(key): int(value) for key, value in expected[field].items()}
        if observed != configured:
            raise ValueError(f"expected_counts.{field} does not match live cases")


def _load_live(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    ReviewProtocol,
    pl.DataFrame,
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Path],
]:
    config = _read_config(config_path)
    protocol = _protocol(config)
    source = config["source"]
    paths = {
        "registry_path": _rooted(source["registry_path"]),
        "wrangled_dir": _rooted(source["wrangled_dir"]),
        "original_holdout_manifest": _rooted(source["original_holdout_manifest"]),
        "reviewed_union_path": _rooted(source["reviewed_union_path"]),
    }
    if sha256_file(paths["reviewed_union_path"]) != source["reviewed_union_sha256"]:
        raise ValueError("Frozen 106-case reviewed union changed")

    reviewed_union = _read_json(paths["reviewed_union_path"])
    if not isinstance(reviewed_union, list):
        raise ValueError("Reviewed union must be a JSON array")
    reviewed_ids = [str(case["canonical_case_id"]) for case in reviewed_union]
    if len(reviewed_ids) != len(set(reviewed_ids)):
        raise ValueError("Reviewed union contains duplicate cases")

    registry = pl.read_parquet(paths["registry_path"])
    entries = load_entries(paths["wrangled_dir"])
    split = derive_target_split(registry, paths["original_holdout_manifest"])
    split_map = _historical_split_map(split)
    all_persona_ids = tuple(sorted(str(value) for value in registry["persona_id"]))
    all_cases = build_full_trajectory_cases(
        entries,
        registry,
        all_persona_ids,
        split="development_only",
    )
    all_cases_by_id = {
        f"{case['persona_id']}:{case['dimension']}": case for case in all_cases
    }
    if len(all_cases_by_id) != len(all_cases):
        raise ValueError("Full development population contains duplicate cases")
    reviewed_id_set = set(reviewed_ids)
    if not reviewed_id_set.issubset(all_cases_by_id):
        raise ValueError("Reviewed union contains a case outside the live population")

    remaining_ids = sorted(set(all_cases_by_id) - reviewed_id_set)
    rows = []
    remaining_cases_by_id = {}
    for case_id in remaining_ids:
        case = all_cases_by_id[case_id]
        remaining_cases_by_id[case_id] = case
        rows.append(
            {
                "canonical_case_id": case_id,
                "persona_id": str(case["persona_id"]),
                "dimension": str(case["dimension"]),
                "historical_split": split_map[str(case["persona_id"])],
                "analysis_role": "development_only",
                "cohort_role": "previously_unreviewed",
                "trajectory_length": len(case["entries"]),
                "case_content_sha256": sha256_json(case),
            }
        )
    selected = pl.DataFrame(rows).sort("canonical_case_id")
    _validate_expected_counts(
        config=config,
        reviewed_union=reviewed_union,
        selected=selected,
    )
    return (
        config,
        protocol,
        selected,
        reviewed_union,
        remaining_cases_by_id,
        paths,
    )


def prepare(args: argparse.Namespace) -> None:
    config_path = _rooted(args.config)
    output = _rooted(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen review root: {output}")
    config, protocol, selected, reviewed_union, cases_by_id, paths = _load_live(
        config_path
    )

    parent_dir = output / "parent_control"
    parent_dir.mkdir(parents=True)
    selected_path = parent_dir / "selected_cases.parquet"
    cases_path = parent_dir / "cohort_cases.json"
    selected.write_parquet(selected_path)
    cases_path.write_text(
        json.dumps([cases_by_id[key] for key in sorted(cases_by_id)], indent=2) + "\n",
        encoding="utf-8",
    )

    cohort_sha256 = sha256_json(
        {
            "cohort_version": protocol.cohort_version,
            "reviewed_union_sha256": config["source"]["reviewed_union_sha256"],
            "selected": selected.to_dicts(),
        }
    )
    review = config["review"]
    shards = shard_review_cases(
        selected,
        max_cases=int(review["max_cases_per_shard"]),
        max_entries=int(review["max_entries_per_shard"]),
    )
    if len(shards) != int(config["expected_counts"]["shards"]):
        raise ValueError("Frozen shard count does not match expected_counts.shards")

    response_schema_path = output / "response_schema.json"
    response_schema_path.write_text(
        json.dumps(build_review_response_schema(protocol), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    response_schema_sha256 = sha256_file(response_schema_path)
    shard_records = []
    for shard_id, case_ids in shards.items():
        packet_dir = output / "reviewer_packets" / shard_id
        packet_dir.mkdir(parents=True)
        packet, key = build_blind_shard(
            shard_id=shard_id,
            case_ids=case_ids,
            cases_by_id=cases_by_id,
            protocol=protocol,
        )
        packet_path = packet_dir / "blind_packet.json"
        key_path = parent_dir / f"{shard_id}_reconciliation_key.json"
        packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
        key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
        shard_records.append(
            {
                "shard_id": shard_id,
                "case_count": len(case_ids),
                "entry_count": int(
                    selected.filter(pl.col("canonical_case_id").is_in(case_ids))[
                        "trajectory_length"
                    ].sum()
                ),
                "case_ids_sha256": sha256_json(case_ids),
                "packet_path": _manifest_path(packet_path),
                "packet_sha256": sha256_file(packet_path),
                "key_path": _manifest_path(key_path),
                "key_sha256": sha256_file(key_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "cohort_version": protocol.cohort_version,
        "analysis_role": "development_only",
        "prepared_at": datetime.now(UTC).isoformat(),
        "direct_api_calls": False,
        "reviewer_runtime": review["reviewer_runtime"],
        "required_reviewers": int(review["required_reviewers"]),
        "reviewed_union_case_count": len(reviewed_union),
        "remaining_case_count": selected.height,
        "remaining_entry_count": int(selected["trajectory_length"].sum()),
        "cohort_sha256": cohort_sha256,
        "config_path": _manifest_path(config_path),
        "config_sha256": sha256_file(config_path),
        "response_schema_path": _manifest_path(response_schema_path),
        "response_schema_sha256": response_schema_sha256,
        "source_hashes": artifact_hashes(_source_files(paths), root=ROOT),
        "generated_hashes": artifact_hashes([selected_path, cases_path], root=ROOT),
        "code_hashes": artifact_hashes(
            [
                ROOT / "src/vif/drift_candidate_review.py",
                ROOT / "src/vif/drift_target.py",
                Path(__file__),
            ],
            root=ROOT,
        ),
        "shards": shard_records,
        "limitations": config["limitations"],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    entry_count = int(selected["trajectory_length"].sum())
    print(
        f"Prepared {selected.height} cases / {entry_count} "
        f"Journal Entry-by-Core-Value combinations in {len(shards)} shards"
    )


def validate(args: argparse.Namespace) -> None:
    config_path = _rooted(args.config)
    output = _rooted(args.output)
    config, protocol, selected, _union, _cases, paths = _load_live(config_path)
    manifest_path = output / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("cohort_version") != protocol.cohort_version:
        raise ValueError("Prepared manifest has wrong cohort version")
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise ValueError("Review config changed")
    if manifest.get("limitations") != config["limitations"]:
        raise ValueError("Prepared limitations differ from the review config")
    if manifest.get("remaining_case_count") != selected.height:
        raise ValueError("Prepared manifest has wrong remaining case count")
    for path, digest in manifest["source_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"Source changed: {path}")
    for path, digest in manifest["generated_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"Generated cohort changed: {path}")
    for path, digest in manifest["code_hashes"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"Review code changed: {path}")
    if set(manifest["source_hashes"]) != {
        _manifest_path(path) for path in _source_files(paths)
    }:
        raise ValueError("Prepared manifest source inventory changed")
    schema_path = ROOT / manifest["response_schema_path"]
    if sha256_file(schema_path) != manifest["response_schema_sha256"]:
        raise ValueError("Response schema changed")
    for shard in manifest["shards"]:
        for path_field, hash_field in (
            ("packet_path", "packet_sha256"),
            ("key_path", "key_sha256"),
        ):
            if sha256_file(ROOT / shard[path_field]) != shard[hash_field]:
                raise ValueError(f"Prepared shard changed: {shard[path_field]}")
    print(
        f"Validated {manifest['remaining_case_count']} cases across "
        f"{len(manifest['shards'])} frozen shards"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
