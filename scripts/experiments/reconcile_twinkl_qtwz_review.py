#!/usr/bin/env python3
"""Reconcile and adjudicate the twinkl-qtwz blinded Conflict reviews."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.review_twinkl_qtwz_remaining_development import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    _load_live,
    _rooted,
)
from scripts.experiments.review_twinkl_qtwz_remaining_development import (  # noqa: E402
    validate as validate_prepared,
)
from src.vif.drift_candidate_review import (  # noqa: E402
    ReviewProtocol,
    apply_adjudication,
    artifact_hashes,
    build_adjudication_packet,
    build_adjudication_response_schema,
    reconcile_reviews,
    sha256_file,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _validated_inputs(
    config_path: Path, output: Path
) -> tuple[dict[str, Any], ReviewProtocol, pl.DataFrame, dict[str, Any]]:
    validate_prepared(argparse.Namespace(config=str(config_path), output=str(output)))
    config, protocol, selected, _union, _cases, _paths = _load_live(config_path)
    manifest = _read_json(output / "manifest.json")
    return config, protocol, selected, manifest


def _derive_outcomes(
    entry_target: pl.DataFrame,
    selected: pl.DataFrame,
    *,
    protocol: ReviewProtocol,
    cohort_sha256: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    metadata = {str(row["canonical_case_id"]): row for row in selected.to_dicts()}
    outcomes = []
    episodes = []
    for case_key, frame in (
        entry_target.sort("position")
        .partition_by("canonical_case_id", as_dict=True)
        .items()
    ):
        case_id = str(case_key[0] if isinstance(case_key, tuple) else case_key)
        rows = frame.to_dicts()
        meta = metadata[case_id]
        unresolved = any(row["final_conflict"] is None for row in rows)
        labels = [row["final_conflict"] for row in rows]
        runs: list[tuple[int, int]] = []
        if not unresolved:
            index = 0
            while index < len(labels):
                if labels[index] is not True:
                    index += 1
                    continue
                start = index
                while index < len(labels) and labels[index] is True:
                    index += 1
                if index - start >= 2:
                    runs.append((start, index - 1))
        for episode_index, (start, end) in enumerate(runs, start=1):
            onset = rows[start]
            confirmation = rows[start + 1]
            ending = rows[end]
            episodes.append(
                {
                    "cohort_version": protocol.cohort_version,
                    "cohort_sha256": cohort_sha256,
                    "episode_id": f"{case_id}:episode_{episode_index:02d}",
                    "canonical_case_id": case_id,
                    "persona_id": meta["persona_id"],
                    "dimension": meta["dimension"],
                    "historical_split": meta["historical_split"],
                    "onset_position": onset["position"],
                    "confirmation_position": confirmation["position"],
                    "end_position": ending["position"],
                    "onset_t_index": onset["t_index"],
                    "confirmation_t_index": confirmation["t_index"],
                    "end_t_index": ending["t_index"],
                    "onset_date": onset["date"],
                    "confirmation_date": confirmation["date"],
                    "end_date": ending["date"],
                    "supporting_positions": list(range(start + 1, end + 2)),
                    "length": end - start + 1,
                    "crosses_week": date.fromisoformat(
                        str(onset["date"])
                    ).isocalendar()[:2]
                    != date.fromisoformat(str(ending["date"])).isocalendar()[:2],
                    "delivery_state": (
                        "active" if end + 1 == len(rows) else "recovered"
                    ),
                }
            )
        outcomes.append(
            {
                "cohort_version": protocol.cohort_version,
                "cohort_sha256": cohort_sha256,
                **meta,
                "entry_agreement_count": sum(
                    row["resolution_method"] == "agreement" for row in rows
                ),
                "entry_count": len(rows),
                "case_resolution": "unresolved" if unresolved else "resolved",
                "has_drift": None if unresolved else bool(runs),
                "episode_count": None if unresolved else len(runs),
            }
        )
    outcome_df = pl.DataFrame(outcomes).sort("canonical_case_id")
    episode_df = (
        pl.DataFrame(episodes)
        if episodes
        else pl.DataFrame(
            schema={
                "cohort_version": pl.String,
                "cohort_sha256": pl.String,
                "episode_id": pl.String,
                "canonical_case_id": pl.String,
                "persona_id": pl.String,
                "dimension": pl.String,
                "historical_split": pl.String,
                "onset_position": pl.Int64,
                "confirmation_position": pl.Int64,
                "end_position": pl.Int64,
                "onset_t_index": pl.Int64,
                "confirmation_t_index": pl.Int64,
                "end_t_index": pl.Int64,
                "onset_date": pl.String,
                "confirmation_date": pl.String,
                "end_date": pl.String,
                "supporting_positions": pl.List(pl.Int64),
                "length": pl.Int64,
                "crosses_week": pl.Boolean,
                "delivery_state": pl.String,
            }
        )
    )
    return outcome_df, episode_df


def _summary(
    entry_target: pl.DataFrame,
    outcomes: pl.DataFrame,
    episodes: pl.DataFrame,
    *,
    protocol: ReviewProtocol,
) -> dict[str, Any]:
    resolved = outcomes.filter(pl.col("case_resolution") == "resolved")
    drift_cases = resolved.filter(pl.col("has_drift") == True)  # noqa: E712
    dimensions = {}
    for dimension, frame in outcomes.partition_by("dimension", as_dict=True).items():
        name = str(dimension[0] if isinstance(dimension, tuple) else dimension)
        resolved_frame = frame.filter(pl.col("case_resolution") == "resolved")
        dimensions[name] = {
            "cases": frame.height,
            "resolved_cases": resolved_frame.height,
            "drift_cases": resolved_frame.filter(pl.col("has_drift") == True).height,  # noqa: E712
        }
    return {
        "schema_version": 1,
        "cohort_version": protocol.cohort_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "case_count": outcomes.height,
        "entry_count": entry_target.height,
        "initial_agreement_count": entry_target.filter(
            pl.col("resolution_method") == "agreement"
        ).height,
        "initial_disagreement_count": entry_target.filter(
            pl.col("resolution_method") != "agreement"
        ).height,
        "resolved_case_count": resolved.height,
        "unresolved_case_count": outcomes.height - resolved.height,
        "drift_case_count": drift_cases.height,
        "drift_episode_count": episodes.height,
        "dimensions": dimensions,
        "cost": {
            "direct_api_calls": 0,
            "direct_api_cost_usd": 0.0,
            "reviewer_runtime": "codex-gpt-5",
            "codex_usage_note": "Codex usage is not metered as repository API spend.",
        },
    }


def _write_results(
    *,
    results_dir: Path,
    entry_target: pl.DataFrame,
    selected: pl.DataFrame,
    protocol: ReviewProtocol,
    cohort_sha256: str,
    suffix: str = "",
) -> dict[str, Path]:
    outcomes, episodes = _derive_outcomes(
        entry_target,
        selected,
        protocol=protocol,
        cohort_sha256=cohort_sha256,
    )
    paths = {
        "entry_target": results_dir / f"entry_target{suffix}.parquet",
        "case_outcomes": results_dir / f"case_outcomes{suffix}.parquet",
        "drift_episodes": results_dir / f"drift_episodes{suffix}.parquet",
        "summary": results_dir / f"summary{suffix}.json",
    }
    entry_target.write_parquet(paths["entry_target"])
    outcomes.write_parquet(paths["case_outcomes"])
    episodes.write_parquet(paths["drift_episodes"])
    paths["summary"].write_text(
        json.dumps(
            _summary(
                entry_target,
                outcomes,
                episodes,
                protocol=protocol,
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths


def reconcile(args: argparse.Namespace) -> None:
    config_path = _rooted(args.config)
    output = _rooted(args.output)
    _config, protocol, selected, manifest = _validated_inputs(config_path, output)
    results_dir = output / "results"
    overwrite = bool(getattr(args, "overwrite", False))
    if results_dir.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite review results: {results_dir}")
    results_dir.mkdir(exist_ok=overwrite)
    schema_sha256 = manifest["response_schema_sha256"]
    rows = []
    response_paths = []
    reviewer_ids: dict[str, set[str]] = {"reviewer_a": set(), "reviewer_b": set()}
    for shard in manifest["shards"]:
        shard_id = shard["shard_id"]
        packet_path = ROOT / shard["packet_path"]
        key_path = ROOT / shard["key_path"]
        reviewer_a_path = output / "reviews/reviewer_a" / f"{shard_id}.json"
        reviewer_b_path = output / "reviews/reviewer_b" / f"{shard_id}.json"
        reviewer_a = _read_json(reviewer_a_path)
        reviewer_b = _read_json(reviewer_b_path)
        reviewer_ids["reviewer_a"].add(str(reviewer_a.get("reviewer_id")))
        reviewer_ids["reviewer_b"].add(str(reviewer_b.get("reviewer_id")))
        rows.extend(
            reconcile_reviews(
                packet=_read_json(packet_path),
                key=_read_json(key_path),
                reviewer_a=reviewer_a,
                reviewer_b=reviewer_b,
                packet_sha256=shard["packet_sha256"],
                response_schema_sha256=schema_sha256,
                reviewer_a_sha256=sha256_file(reviewer_a_path),
                reviewer_b_sha256=sha256_file(reviewer_b_path),
                protocol=protocol,
            )
        )
        response_paths.extend([reviewer_a_path, reviewer_b_path])
    if reviewer_ids["reviewer_a"] & reviewer_ids["reviewer_b"]:
        raise ValueError("Reviewer lanes must use distinct reviewer IDs")
    entry_target = pl.DataFrame(rows).sort("canonical_case_id", "position")
    if entry_target.height != int(manifest["remaining_entry_count"]):
        raise ValueError("Review responses do not cover every frozen entry")
    result_paths = _write_results(
        results_dir=results_dir,
        entry_target=entry_target,
        selected=selected,
        protocol=protocol,
        cohort_sha256=manifest["cohort_sha256"],
    )
    audit = {
        "schema_version": 1,
        "cohort_version": protocol.cohort_version,
        "created_at": datetime.now(UTC).isoformat(),
        "reviewer_ids": {key: sorted(value) for key, value in reviewer_ids.items()},
        "review_responses": artifact_hashes(response_paths, root=ROOT),
        "results": artifact_hashes(list(result_paths.values()), root=ROOT),
        "reconciliation_runner_sha256": sha256_file(Path(__file__)),
    }
    (results_dir / "audit_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = _read_json(result_paths["summary"])
    print(
        f"Reconciled {summary['entry_count']} entries: "
        f"{summary['initial_agreement_count']} agreements / "
        f"{summary['initial_disagreement_count']} disagreements"
    )


def build_adjudication(args: argparse.Namespace) -> None:
    config_path = _rooted(args.config)
    output = _rooted(args.output)
    _config, protocol, _selected, manifest = _validated_inputs(config_path, output)
    results_dir = output / "results"
    entry_target = pl.read_parquet(results_dir / "entry_target.parquet")
    unresolved_ids = set(
        entry_target.filter(pl.col("resolution_status") == "unresolved")[
            "canonical_case_id"
        ]
    )
    if not unresolved_ids:
        print("No disagreements require adjudication")
        return
    adjudication_dir = output / "adjudication"
    overwrite = bool(getattr(args, "overwrite", False))
    if adjudication_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite adjudication files: {adjudication_dir}"
        )
    case_material: dict[str, dict[str, Any]] = {}
    for shard in manifest["shards"]:
        packet = _read_json(ROOT / shard["packet_path"])
        key = _read_json(ROOT / shard["key_path"])
        shard_id = shard["shard_id"]
        reviewer_a = _read_json(output / "reviews/reviewer_a" / f"{shard_id}.json")
        reviewer_b = _read_json(output / "reviews/reviewer_b" / f"{shard_id}.json")
        packet_map = {case["review_case_id"]: case for case in packet["cases"]}
        first_map = {case["review_case_id"]: case for case in reviewer_a["cases"]}
        second_map = {case["review_case_id"]: case for case in reviewer_b["cases"]}
        for key_case in key["cases"]:
            case_id = str(key_case["canonical_case_id"])
            if case_id not in unresolved_ids:
                continue
            review_case_id = key_case["review_case_id"]
            packet_case = packet_map[review_case_id]
            target_rows = entry_target.filter(
                pl.col("canonical_case_id") == case_id
            ).sort("position")
            packet_entries = {
                int(entry["position"]): entry for entry in packet_case["entries"]
            }
            case_material[case_id] = {
                "canonical_case_id": case_id,
                "declared_core_value": packet_case["declared_core_value"],
                "review_rationales": [
                    first_map[review_case_id]["rationale"],
                    second_map[review_case_id]["rationale"],
                ],
                "entries": [
                    {
                        **row,
                        "journal_entry": packet_entries[int(row["position"])][
                            "journal_entry"
                        ],
                    }
                    for row in target_rows.to_dicts()
                ],
            }
    packet, key = build_adjudication_packet(
        list(case_material.values()), protocol=protocol
    )
    adjudication_dir.mkdir(exist_ok=overwrite)
    packet_path = adjudication_dir / "reviewer_packet.json"
    key_path = adjudication_dir / "parent_reconciliation_key.json"
    schema_path = adjudication_dir / "response_schema.json"
    packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    schema_path.write_text(
        json.dumps(
            build_adjudication_response_schema(protocol), indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    adjudication_manifest = {
        "schema_version": 1,
        "cohort_version": protocol.cohort_version,
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": len(case_material),
        "disputed_entry_count": entry_target.filter(
            pl.col("resolution_status") == "unresolved"
        ).height,
        "packet_path": str(packet_path.relative_to(ROOT)),
        "packet_sha256": sha256_file(packet_path),
        "key_path": str(key_path.relative_to(ROOT)),
        "key_sha256": sha256_file(key_path),
        "response_schema_path": str(schema_path.relative_to(ROOT)),
        "response_schema_sha256": sha256_file(schema_path),
        "source_entry_target_sha256": sha256_file(results_dir / "entry_target.parquet"),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    (adjudication_dir / "audit_manifest.json").write_text(
        json.dumps(adjudication_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared adjudication for {len(case_material)} cases / "
        f"{adjudication_manifest['disputed_entry_count']} disputed entries"
    )


def finalize_adjudication(args: argparse.Namespace) -> None:
    config_path = _rooted(args.config)
    output = _rooted(args.output)
    _config, protocol, selected, manifest = _validated_inputs(config_path, output)
    results_dir = output / "results"
    adjudication_dir = output / "adjudication"
    audit = _read_json(adjudication_dir / "audit_manifest.json")
    entry_path = results_dir / "entry_target.parquet"
    if sha256_file(entry_path) != audit["source_entry_target_sha256"]:
        raise ValueError("Initial reconciled entry target changed")
    packet_path = ROOT / audit["packet_path"]
    key_path = ROOT / audit["key_path"]
    schema_path = ROOT / audit["response_schema_path"]
    response_path = adjudication_dir / "adjudicator_response.json"
    for path, digest in (
        (packet_path, audit["packet_sha256"]),
        (key_path, audit["key_sha256"]),
        (schema_path, audit["response_schema_sha256"]),
    ):
        if sha256_file(path) != digest:
            raise ValueError(f"Adjudication input changed: {path}")
    final_entries = apply_adjudication(
        pl.read_parquet(entry_path),
        packet=_read_json(packet_path),
        key=_read_json(key_path),
        response=_read_json(response_path),
        packet_sha256=audit["packet_sha256"],
        response_schema_sha256=audit["response_schema_sha256"],
        response_sha256=sha256_file(response_path),
        protocol=protocol,
    )
    final_paths = _write_results(
        results_dir=results_dir,
        entry_target=final_entries,
        selected=selected,
        protocol=protocol,
        cohort_sha256=manifest["cohort_sha256"],
        suffix="_final",
    )
    final_audit = {
        "schema_version": 1,
        "cohort_version": protocol.cohort_version,
        "created_at": datetime.now(UTC).isoformat(),
        "adjudication_response_sha256": sha256_file(response_path),
        "results": artifact_hashes(list(final_paths.values()), root=ROOT),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    (results_dir / "audit_manifest_final.json").write_text(
        json.dumps(final_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = _read_json(final_paths["summary"])
    print(
        f"Finalized {summary['resolved_case_count']}/{summary['case_count']} "
        f"cases with {summary['drift_episode_count']} Drifts"
    )


def combine_complete_development(args: argparse.Namespace) -> None:
    config_path = _rooted(args.config)
    output = _rooted(args.output)
    _config, protocol, _selected, manifest = _validated_inputs(config_path, output)
    results_dir = output / "results"
    remaining_entries_path = results_dir / "entry_target_final.parquet"
    remaining_outcomes_path = results_dir / "case_outcomes_final.parquet"
    remaining_episodes_path = results_dir / "drift_episodes_final.parquet"
    remaining = pl.read_parquet(remaining_entries_path)
    remaining_outcomes = pl.read_parquet(remaining_outcomes_path)
    remaining_episodes = pl.read_parquet(remaining_episodes_path)
    unresolved = remaining.filter(pl.col("resolution_status") != "resolved")
    if unresolved.height:
        raise ValueError("Cannot combine while remaining cases contain uncertainty")

    union_root = ROOT / (
        "logs/experiments/artifacts/twinkl_752_5_reassessment_20260714"
    )
    union_entries_path = union_root / "union_entry_target.parquet"
    union_outcomes_path = union_root / "union_case_summary.parquet"
    union_episodes_path = union_root / "union_drift_episodes.parquet"
    union_entries = pl.read_parquet(union_entries_path)
    union_outcomes = pl.read_parquet(union_outcomes_path)
    union_episodes = pl.read_parquet(union_episodes_path)

    normalized_remaining_entries = (
        remaining.join(
            remaining_outcomes.select("canonical_case_id", "historical_split"),
            on="canonical_case_id",
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.lit("twinkl_qtwz_complete_review").alias("cohort_source"),
            pl.lit(False).alias("opus_resolved"),
        )
        .select(union_entries.columns)
    )
    complete_entries = pl.concat(
        [union_entries, normalized_remaining_entries], how="vertical"
    ).sort("canonical_case_id", "position")

    normalized_remaining_outcomes = remaining_outcomes.with_columns(
        pl.lit("twinkl_qtwz_complete_review").alias("cohort_source"),
        pl.lit(False).alias("opus_resolved"),
        pl.col("entry_count").alias("resolved_entry_count"),
        pl.col("episode_count").cast(pl.Int64).alias("drift_count"),
    ).select(union_outcomes.columns)
    complete_outcomes = pl.concat(
        [union_outcomes, normalized_remaining_outcomes], how="vertical"
    ).sort("canonical_case_id")

    normalized_remaining_episodes = remaining_episodes.with_columns(
        pl.lit("twinkl_qtwz_complete_review").alias("cohort_source")
    ).select(union_episodes.columns)
    complete_episodes = pl.concat(
        [union_episodes, normalized_remaining_episodes], how="vertical"
    ).sort("canonical_case_id", "onset_position")

    if complete_outcomes.height != 292:
        raise ValueError("Complete development analysis must contain 292 cases")
    if complete_outcomes["canonical_case_id"].n_unique() != 292:
        raise ValueError("Complete development analysis contains duplicate cases")
    if complete_entries.height != (
        int(manifest["remaining_entry_count"]) + union_entries.height
    ):
        raise ValueError("Complete development entry count is inconsistent")

    paths = {
        "entry_target": results_dir / "complete_development_entry_target.parquet",
        "case_outcomes": results_dir / "complete_development_case_outcomes.parquet",
        "drift_episodes": results_dir / "complete_development_drift_episodes.parquet",
        "summary": results_dir / "complete_development_summary.json",
    }
    complete_entries.write_parquet(paths["entry_target"])
    complete_outcomes.write_parquet(paths["case_outcomes"])
    complete_episodes.write_parquet(paths["drift_episodes"])
    newly_reviewed = complete_outcomes.filter(
        pl.col("cohort_source") == "twinkl_qtwz_complete_review"
    )
    summary = {
        "schema_version": 1,
        "cohort_version": protocol.cohort_version,
        "generated_at": datetime.now(UTC).isoformat(),
        "complete_case_count": complete_outcomes.height,
        "complete_entry_count": complete_entries.height,
        "complete_resolved_entry_count": complete_entries.filter(
            pl.col("final_conflict").is_not_null()
        ).height,
        "complete_unresolved_entry_count": complete_entries.filter(
            pl.col("final_conflict").is_null()
        ).height,
        "complete_resolved_case_count": complete_outcomes.filter(
            pl.col("has_drift").is_not_null()
        ).height,
        "complete_drift_case_count": complete_outcomes.filter(
            pl.col("has_drift") == True  # noqa: E712
        ).height,
        "complete_drift_episode_count": complete_episodes.height,
        "previously_reviewed_case_count": union_outcomes.height,
        "previously_known_drift_episode_count": union_episodes.height,
        "newly_reviewed_case_count": newly_reviewed.height,
        "newly_found_drift_case_count": newly_reviewed.filter(
            pl.col("has_drift") == True  # noqa: E712
        ).height,
        "newly_found_drift_episode_count": remaining_episodes.height,
        "limitations": [
            "This is AI-reviewed synthetic development evidence, not human validation.",
            "The complete synthetic development data is not a prevalence estimate.",
            (
                "Two immutable historical LLM-Judge Conflict Labels remain "
                "unresolved inside a "
                "case with a frozen resolved Drift outcome."
            ),
            "The fresh final test remains unopened and no deployment approval follows.",
        ],
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": 1,
        "cohort_version": protocol.cohort_version,
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": artifact_hashes(
            [
                remaining_entries_path,
                remaining_outcomes_path,
                remaining_episodes_path,
                union_entries_path,
                union_outcomes_path,
                union_episodes_path,
            ],
            root=ROOT,
        ),
        "outputs": artifact_hashes(list(paths.values()), root=ROOT),
        "runner_sha256": sha256_file(Path(__file__)),
    }
    (results_dir / "complete_development_audit_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Combined {summary['complete_case_count']} cases / "
        f"{summary['complete_drift_episode_count']} Drifts"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "reconcile",
            "build-adjudication",
            "finalize-adjudication",
            "combine",
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing derived results without changing frozen reviews.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "reconcile":
        reconcile(args)
    elif args.command == "build-adjudication":
        build_adjudication(args)
    elif args.command == "finalize-adjudication":
        finalize_adjudication(args)
    else:
        combine_complete_development(args)


if __name__ == "__main__":
    main()
