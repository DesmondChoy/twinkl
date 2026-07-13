#!/usr/bin/env python3
"""Prepare and reconcile the full legacy-discoverable Drift review cohort."""

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
    ADJUDICATION_RESPONSE_SCHEMA,
    COHORT_VERSION,
    REVIEW_RESPONSE_SCHEMA,
    apply_adjudication,
    artifact_hashes,
    build_adjudication_packet,
    build_blind_shard,
    build_legacy_trajectory_inventory,
    derive_review_outcomes,
    match_legacy_negative_controls,
    reconcile_reviews,
    selected_case_metadata,
    sha256_file,
    sha256_json,
    shard_review_cases,
    summarize_outcomes,
    validate_expected_counts,
    validate_shard_coverage,
)
from src.vif.drift_target import (  # noqa: E402
    build_full_trajectory_cases,
    derive_target_split,
)

DEFAULT_CONFIG = Path("config/evals/twinkl_752_4_legacy_drift_review_v1.yaml")
DEFAULT_OUTPUT = Path(
    "logs/experiments/artifacts/twinkl_752_4_legacy_drift_review_20260713"
)


def _rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _manifest_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _verify_hashes(hashes: dict[str, str], *, label: str) -> None:
    for relative, digest in hashes.items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(f"{label} changed: {relative}")


def _read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    if payload.get("schema_version") != 1:
        raise ValueError("Review config must use schema_version: 1")
    if payload.get("cohort_version") != COHORT_VERSION:
        raise ValueError(f"Review config must name {COHORT_VERSION}")
    return payload


def _load_live(config_path: Path) -> tuple[
    dict[str, Any],
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    list[dict[str, Any]],
    dict[str, Path],
]:
    config = _read_config(config_path)
    source = config["source"]
    paths = {
        name: _rooted(value)
        for name, value in source.items()
        if name.endswith("_path") or name.endswith("manifest")
    }
    paths["wrangled_dir"] = _rooted(source["wrangled_dir"])
    paths["old_target_artifact_root"] = _rooted(source["old_target_artifact_root"])
    registry = pl.read_parquet(paths["registry_path"])
    consensus = pl.read_parquet(paths["consensus_labels_path"])
    persisted = pl.read_parquet(paths["persisted_labels_path"])
    entries = load_entries(paths["wrangled_dir"])
    split = derive_target_split(registry, paths["original_holdout_manifest"])
    all_ids = tuple(sorted(str(value) for value in registry["persona_id"].to_list()))
    cases = build_full_trajectory_cases(
        entries, registry, all_ids, split="development_only"
    )
    inventory = build_legacy_trajectory_inventory(cases, consensus, persisted, split)
    pairs = match_legacy_negative_controls(inventory)
    selected = selected_case_metadata(inventory, pairs)
    validate_expected_counts(
        inventory=inventory,
        pairs=pairs,
        selected=selected,
        expected=config["expected_counts"],
    )
    return config, registry, inventory, pairs, cases, paths


def prepare(args: argparse.Namespace) -> None:
    config_path = _rooted(args.config)
    output = _rooted(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen review root: {output}")
    config, registry, inventory, pairs, cases, paths = _load_live(config_path)
    selected = selected_case_metadata(inventory, pairs)
    selected_ids = set(selected["canonical_case_id"].to_list())
    cases_by_id = {
        f"{case['persona_id']}:{case['dimension']}": case
        for case in cases
        if f"{case['persona_id']}:{case['dimension']}" in selected_ids
    }
    if set(cases_by_id) != selected_ids:
        raise ValueError("Selected cohort does not map to complete runtime cases")

    parent_dir = output / "parent_control"
    parent_dir.mkdir(parents=True)
    pairs_path = parent_dir / "selection_pairs.parquet"
    selected_path = parent_dir / "selected_cases.parquet"
    pairs.write_parquet(pairs_path)
    selected.write_parquet(selected_path)
    cases_path = parent_dir / "cohort_cases.json"
    cases_path.write_text(
        json.dumps([cases_by_id[key] for key in sorted(cases_by_id)], indent=2) + "\n",
        encoding="utf-8",
    )
    cohort_sha256 = sha256_json(
        {
            "cohort_version": COHORT_VERSION,
            "selected": selected.to_dicts(),
            "pairs": pairs.to_dicts(),
        }
    )

    review = config["review"]
    shards = shard_review_cases(
        selected,
        max_cases=int(review["max_cases_per_shard"]),
        max_entries=int(review["max_entries_per_shard"]),
    )
    schema_path = output / "response_schema.json"
    schema_path.write_text(
        json.dumps(REVIEW_RESPONSE_SCHEMA, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    schema_sha256 = sha256_file(schema_path)
    shard_records = []
    for shard_id, case_ids in shards.items():
        shard_dir = output / "reviewer_packets" / shard_id
        shard_dir.mkdir(parents=True)
        packet, key = build_blind_shard(
            shard_id=shard_id,
            case_ids=case_ids,
            cases_by_id=cases_by_id,
        )
        packet_path = shard_dir / "blind_packet.json"
        key_path = parent_dir / f"{shard_id}_reconciliation_key.json"
        packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
        key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
        shard_records.append(
            {
                "shard_id": shard_id,
                "case_count": len(case_ids),
                "entry_count": sum(
                    int(
                        selected.filter(pl.col("canonical_case_id") == case_id)[
                            "trajectory_length"
                        ][0]
                    )
                    for case_id in case_ids
                ),
                "case_ids_sha256": sha256_json(sorted(case_ids)),
                "packet_path": _manifest_path(packet_path),
                "packet_sha256": sha256_file(packet_path),
                "key_path": _manifest_path(key_path),
                "key_sha256": sha256_file(key_path),
            }
        )

    old_promotion_ids = sorted(
        inventory.filter(pl.col("historical_split") == "former_promotion")[
            "persona_id"
        ].unique().to_list()
    )
    promotion_receipt = {
        "schema_version": 1,
        "cohort_version": COHORT_VERSION,
        "authority": "twinkl-752.4 user-approved full review",
        "recorded_at": datetime.now(UTC).isoformat(),
        "former_role": "v8pb promotion population",
        "new_role": "development_only",
        "future_final_test_eligible": False,
        "successor_final_test_issue": "twinkl-pv6s",
        "persona_count": len(old_promotion_ids),
        "persona_ids": old_promotion_ids,
        "persona_ids_sha256": sha256_json(old_promotion_ids),
        "immutable_old_artifacts": artifact_hashes(
            [
                paths["old_target_manifest"],
                paths["old_target_artifact_root"]
                / "promotion/parent_control/audit_manifest.json",
                paths["old_target_artifact_root"] / "promotion/promotion_no_score.json",
            ],
            root=ROOT,
        ),
    }
    promotion_path = parent_dir / "old_promotion_reclassification.json"
    promotion_path.write_text(
        json.dumps(promotion_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    source_files = {
        name: path
        for name, path in paths.items()
        if path.is_file() and name != "old_target_artifact_root"
    }
    manifest = {
        "schema_version": 1,
        "cohort_version": COHORT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "analysis_role": "development_only",
        "candidate_discovery_scope": (
            "union of same-source adjacent -1/-1 legacy windows"
        ),
        "cohort_sha256": cohort_sha256,
        "case_count": selected.height,
        "entry_count": int(selected["trajectory_length"].sum()),
        "candidate_count": pairs.height,
        "control_count": pairs.height,
        "source_inputs": artifact_hashes(
            [
                config_path,
                *source_files.values(),
                Path(__file__),
                ROOT / "src/vif/drift_candidate_review.py",
            ],
            root=ROOT,
        ),
        "parent_outputs": artifact_hashes(
            [pairs_path, selected_path, cases_path, promotion_path], root=ROOT
        ),
        "response_schema_path": _manifest_path(schema_path),
        "response_schema_sha256": schema_sha256,
        "shards": shard_records,
        "known_limitations": config["limitations"],
    }
    manifest_path = parent_dir / "cohort_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared {selected.height} cases / "
        f"{int(selected['trajectory_length'].sum())} "
        f"entries in {len(shards)} shards at {output}"
    )


def _validate_frozen(output: Path) -> tuple[dict[str, Any], pl.DataFrame]:
    manifest_path = output / "parent_control/cohort_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("cohort_version") != COHORT_VERSION:
        raise ValueError("Frozen manifest has wrong cohort version")
    for relative, digest in manifest["source_inputs"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(f"Frozen source hash changed: {relative}")
    for relative, digest in manifest["parent_outputs"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(f"Frozen parent artifact changed: {relative}")
    schema_path = ROOT / manifest["response_schema_path"]
    if sha256_file(schema_path) != manifest["response_schema_sha256"]:
        raise ValueError("Response schema hash changed")
    selected = pl.read_parquet(output / "parent_control/selected_cases.parquet")
    observed_shards = {
        record["shard_id"]: _read_json(ROOT / record["key_path"])
        for record in manifest["shards"]
    }
    validate_shard_coverage(
        {
            shard_id: [case["canonical_case_id"] for case in key["cases"]]
            for shard_id, key in observed_shards.items()
        },
        selected["canonical_case_id"].to_list(),
    )
    return manifest, selected


def reconcile(args: argparse.Namespace) -> None:
    output = _rooted(args.output)
    manifest, selected = _validate_frozen(output)
    schema_sha256 = manifest["response_schema_sha256"]
    entry_rows = []
    response_paths = []
    reviewer_ids: dict[str, set[str]] = {"reviewer_a": set(), "reviewer_b": set()}
    for record in manifest["shards"]:
        shard_id = record["shard_id"]
        packet_path = ROOT / record["packet_path"]
        key_path = ROOT / record["key_path"]
        if sha256_file(packet_path) != record["packet_sha256"]:
            raise ValueError(f"Packet hash changed for {shard_id}")
        if sha256_file(key_path) != record["key_sha256"]:
            raise ValueError(f"Key hash changed for {shard_id}")
        packet = _read_json(packet_path)
        key = _read_json(key_path)
        reviewer_a_path = output / "reviews/reviewer_a" / f"{shard_id}.json"
        reviewer_b_path = output / "reviews/reviewer_b" / f"{shard_id}.json"
        reviewer_a = _read_json(reviewer_a_path)
        reviewer_b = _read_json(reviewer_b_path)
        reviewer_ids["reviewer_a"].add(str(reviewer_a.get("reviewer_id")))
        reviewer_ids["reviewer_b"].add(str(reviewer_b.get("reviewer_id")))
        entry_rows.extend(
            reconcile_reviews(
                packet=packet,
                key=key,
                reviewer_a=reviewer_a,
                reviewer_b=reviewer_b,
                packet_sha256=record["packet_sha256"],
                response_schema_sha256=schema_sha256,
                reviewer_a_sha256=sha256_file(reviewer_a_path),
                reviewer_b_sha256=sha256_file(reviewer_b_path),
            )
        )
        response_paths.extend([reviewer_a_path, reviewer_b_path])
    if reviewer_ids["reviewer_a"] & reviewer_ids["reviewer_b"]:
        raise ValueError("Reviewer lanes must use distinct reviewer IDs")

    results = output / "results"
    results.mkdir(exist_ok=False)
    entry_target = pl.DataFrame(entry_rows).sort(
        "canonical_case_id", "position"
    )
    if entry_target.height != int(manifest["entry_count"]):
        raise ValueError("Reconciled entry count does not match frozen cohort")
    case_outcomes, episodes = derive_review_outcomes(
        entry_target, selected, cohort_sha256=manifest["cohort_sha256"]
    )
    summary = summarize_outcomes(case_outcomes)
    summary["cohort_sha256"] = manifest["cohort_sha256"]
    summary["reviewer_ids"] = {
        lane: sorted(values) for lane, values in reviewer_ids.items()
    }
    summary["episode_count"] = episodes.height
    entry_path = results / "entry_target.parquet"
    cases_path = results / "case_outcomes.parquet"
    episodes_path = results / "drift_episodes.parquet"
    summary_path = results / "summary.json"
    entry_target.write_parquet(entry_path)
    case_outcomes.write_parquet(cases_path)
    episodes.write_parquet(episodes_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": 1,
        "cohort_version": COHORT_VERSION,
        "cohort_sha256": manifest["cohort_sha256"],
        "completed_at": datetime.now(UTC).isoformat(),
        "review_responses": artifact_hashes(response_paths, root=ROOT),
        "results": artifact_hashes(
            [entry_path, cases_path, episodes_path, summary_path], root=ROOT
        ),
    }
    audit_path = results / "audit_manifest.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Reconciled {case_outcomes.height} cases: "
        f"{summary['resolved_case_count']} resolved, "
        f"{summary['unresolved_case_count']} unresolved, {episodes.height} episodes"
    )


def build_adjudication(args: argparse.Namespace) -> None:
    output = _rooted(args.output)
    manifest, _selected = _validate_frozen(output)
    adjudication_dir = output / "adjudication"
    if adjudication_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite adjudication artifacts: {adjudication_dir}"
        )
    results_audit = _read_json(output / "results/audit_manifest.json")
    if results_audit.get("schema_version") != 1:
        raise ValueError("Paired-review audit has wrong schema version")
    if results_audit.get("cohort_version") != COHORT_VERSION:
        raise ValueError("Paired-review audit has wrong cohort version")
    if results_audit.get("cohort_sha256") != manifest["cohort_sha256"]:
        raise ValueError("Paired-review audit has wrong cohort hash")
    _verify_hashes(results_audit["review_responses"], label="Review response")
    _verify_hashes(results_audit["results"], label="Paired-review result")

    entry_target_path = output / "results/entry_target.parquet"
    outcomes_path = output / "results/case_outcomes.parquet"
    entry_target = pl.read_parquet(entry_target_path)
    outcomes = pl.read_parquet(outcomes_path)
    unresolved_ids = set(
        outcomes.filter(pl.col("case_resolution") == "unresolved")[
            "canonical_case_id"
        ].to_list()
    )
    case_material = {}
    for record in manifest["shards"]:
        packet = _read_json(ROOT / record["packet_path"])
        key = _read_json(ROOT / record["key_path"])
        reviewer_a = _read_json(
            output / "reviews/reviewer_a" / f"{record['shard_id']}.json"
        )
        reviewer_b = _read_json(
            output / "reviews/reviewer_b" / f"{record['shard_id']}.json"
        )
        packet_map = {case["review_case_id"]: case for case in packet["cases"]}
        first_map = {case["review_case_id"]: case for case in reviewer_a["cases"]}
        second_map = {case["review_case_id"]: case for case in reviewer_b["cases"]}
        for key_case in key["cases"]:
            canonical_case_id = key_case["canonical_case_id"]
            if canonical_case_id not in unresolved_ids:
                continue
            review_case_id = key_case["review_case_id"]
            packet_case = packet_map[review_case_id]
            target_rows = (
                entry_target.filter(
                    pl.col("canonical_case_id") == canonical_case_id
                )
                .sort("position")
                .to_dicts()
            )
            packet_entries = {
                entry["position"]: entry for entry in packet_case["entries"]
            }
            entries = []
            for row in target_rows:
                entries.append(
                    {
                        **row,
                        "journal_entry": packet_entries[row["position"]][
                            "journal_entry"
                        ],
                    }
                )
            case_material[canonical_case_id] = {
                "canonical_case_id": canonical_case_id,
                "declared_core_value": packet_case["declared_core_value"],
                "entries": entries,
                "review_rationales": [
                    first_map[review_case_id]["rationale"],
                    second_map[review_case_id]["rationale"],
                ],
            }
    if set(case_material) != unresolved_ids:
        raise ValueError("Could not recover blind text for every unresolved case")

    packet, key = build_adjudication_packet(list(case_material.values()))
    adjudication_dir.mkdir()
    packet_path = adjudication_dir / "reviewer_packet.json"
    schema_path = adjudication_dir / "response_schema.json"
    key_path = adjudication_dir / "parent_reconciliation_key.json"
    packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    schema_path.write_text(
        json.dumps(ADJUDICATION_RESPONSE_SCHEMA, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    adjudication_manifest = {
        "schema_version": 1,
        "cohort_version": COHORT_VERSION,
        "cohort_sha256": manifest["cohort_sha256"],
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(case_material),
        "disputed_entry_count": entry_target.filter(
            pl.col("resolution_status") == "unresolved"
        ).height,
        "packet_path": _manifest_path(packet_path),
        "packet_sha256": sha256_file(packet_path),
        "response_schema_path": _manifest_path(schema_path),
        "response_schema_sha256": sha256_file(schema_path),
        "key_path": _manifest_path(key_path),
        "key_sha256": sha256_file(key_path),
        "source_results": artifact_hashes(
            [entry_target_path, outcomes_path], root=ROOT
        ),
        "controlled_disclosure": (
            "Full displayed text and anonymous prior judgments only; no identity, "
            "cohort role, historical split, legacy label, VIF Critic output, or "
            "expected outcome."
        ),
    }
    manifest_path = adjudication_dir / "audit_manifest.json"
    manifest_path.write_text(
        json.dumps(adjudication_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared adjudication for {len(case_material)} cases / "
        f"{adjudication_manifest['disputed_entry_count']} disputed entries"
    )


def finalize_adjudication(args: argparse.Namespace) -> None:
    output = _rooted(args.output)
    cohort_manifest, selected = _validate_frozen(output)
    adjudication_dir = output / "adjudication"
    manifest_path = adjudication_dir / "audit_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Adjudication manifest has wrong schema version")
    if manifest.get("cohort_version") != COHORT_VERSION:
        raise ValueError("Adjudication manifest has wrong cohort version")
    if manifest.get("cohort_sha256") != cohort_manifest["cohort_sha256"]:
        raise ValueError("Adjudication manifest has wrong cohort hash")
    _verify_hashes(manifest["source_results"], label="Adjudication source result")
    packet_path = ROOT / manifest["packet_path"]
    schema_path = ROOT / manifest["response_schema_path"]
    key_path = ROOT / manifest["key_path"]
    checks = {
        packet_path: manifest["packet_sha256"],
        schema_path: manifest["response_schema_sha256"],
        key_path: manifest["key_sha256"],
    }
    for path, digest in checks.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Adjudication artifact hash changed: {path}")
    response_path = adjudication_dir / "adjudicator_response.json"
    response = _read_json(response_path)
    entry_target = pl.read_parquet(output / "results/entry_target.parquet")
    final_entries = apply_adjudication(
        entry_target,
        packet=_read_json(packet_path),
        key=_read_json(key_path),
        response=response,
        packet_sha256=manifest["packet_sha256"],
        response_schema_sha256=manifest["response_schema_sha256"],
        response_sha256=sha256_file(response_path),
    )
    final_outcomes, final_episodes = derive_review_outcomes(
        final_entries, selected, cohort_sha256=cohort_manifest["cohort_sha256"]
    )
    final_summary = summarize_outcomes(final_outcomes)
    final_summary["cohort_sha256"] = cohort_manifest["cohort_sha256"]
    final_summary["episode_count"] = final_episodes.height
    final_summary["adjudicator_id"] = response["adjudicator_id"]

    results = output / "results"
    entry_path = results / "entry_target_final.parquet"
    cases_path = results / "case_outcomes_final.parquet"
    episodes_path = results / "drift_episodes_final.parquet"
    summary_path = results / "summary_final.json"
    audit_path = results / "audit_manifest_final.json"
    for path in (entry_path, cases_path, episodes_path, summary_path, audit_path):
        if path.exists() and not args.replace_final:
            raise FileExistsError(f"Refusing to overwrite final artifact: {path}")
    final_entries.write_parquet(entry_path)
    final_outcomes.write_parquet(cases_path)
    final_episodes.write_parquet(episodes_path)
    summary_path.write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_audit = {
        "schema_version": 1,
        "cohort_version": COHORT_VERSION,
        "cohort_sha256": cohort_manifest["cohort_sha256"],
        "completed_at": datetime.now(UTC).isoformat(),
        "adjudication_response": artifact_hashes([response_path], root=ROOT),
        "results": artifact_hashes(
            [entry_path, cases_path, episodes_path, summary_path], root=ROOT
        ),
    }
    audit_path.write_text(
        json.dumps(final_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Finalized {final_outcomes.height} cases: "
        f"{final_summary['resolved_case_count']} resolved, "
        f"{final_summary['unresolved_case_count']} unresolved, "
        f"{final_episodes.height} episodes"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("reconcile")
    subparsers.add_parser("build-adjudication")
    finalize_parser = subparsers.add_parser("finalize-adjudication")
    finalize_parser.add_argument("--replace-final", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "reconcile":
        reconcile(args)
    elif args.command == "build-adjudication":
        build_adjudication(args)
    else:
        finalize_adjudication(args)


if __name__ == "__main__":
    main()
