#!/usr/bin/env python3
"""Freeze, run, and score the ``twinkl-752.5`` reassessment.

``freeze`` and ``prepare`` make no paid calls. ``run`` remains explicitly
opt-in so the hash-bound union and protocol can be committed before outcomes
are generated.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prompts import load_prompt
from scripts.experiments import weekly_verifier_ablation as baseline
from src.vif.drift_benchmark import match_episodes
from src.vif.drift_scoring import score_mlp_cases
from src.vif.state_encoder import concatenate_entry_text
from src.wrangling.parse_wrangled_data import parse_wrangled_file

DEFAULT_CONFIG_PATH = Path("config/evals/twinkl_752_5_reassessment_v1.yaml")
WEEKLY_WITHOUT = "weekly_without_critic"
WEEKLY_WITH_RAW = "weekly_with_raw_critic"
EARLY_WITHOUT = "early_without_critic"
SCHEDULED = "critic_scheduled_early_plus_weekly"


def _artifact_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    spec = config["artifacts"]
    output_dir = baseline._rooted(spec["output_dir"], root)
    return {
        "output_dir": output_dir,
        "union_cases": output_dir / spec["union_cases_filename"],
        "union_case_summary": output_dir / spec["union_case_summary_filename"],
        "union_entry_target": output_dir / spec["union_entry_target_filename"],
        "union_drift_episodes": output_dir / spec["union_drift_episodes_filename"],
        "union_receipt": output_dir / spec["union_receipt_filename"],
        "run_020_full_evidence": output_dir / spec["run_020_full_evidence_filename"],
        "run_020_target_evidence": output_dir
        / spec["run_020_target_evidence_filename"],
        "eligible_opportunities": output_dir / spec["eligible_opportunities_filename"],
        "trigger_schedule": output_dir / spec["trigger_schedule_filename"],
        "prompts": output_dir / spec["prompts_filename"],
        "manifest": output_dir / spec["manifest_filename"],
        "responses": output_dir / spec["responses_filename"],
        "metrics": output_dir / spec["metrics_filename"],
    }


def _source_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    return {
        key: baseline._rooted(value, root)
        for key, value in config["sources"].items()
        if key.endswith("_path")
    }


def _case_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Review document is missing cases")
    result = {str(case["review_case_id"]): case for case in cases}
    if len(result) != len(cases):
        raise ValueError("Review document contains duplicate case IDs")
    return result


def _canonical_case_id(persona_id: str, dimension: str) -> str:
    return f"{persona_id}:{baseline._normalize_value(dimension)}"


def _profile_and_entries(
    *, persona_id: str, wrangled_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile, entries, _warnings = parse_wrangled_file(
        wrangled_dir / f"persona_{persona_id}.md"
    )
    return profile, sorted(entries, key=lambda row: int(row["t_index"]))


def _runtime_text(entry: dict[str, Any]) -> str:
    return concatenate_entry_text(
        entry.get("initial_entry"),
        entry.get("nudge_text"),
        entry.get("response_text"),
    ).strip()


def _episode_rows_from_targets(
    *, case: dict[str, Any], target_rows: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    by_position = {int(row["position"]): row for row in target_rows}
    episodes: list[dict[str, Any]] = []
    run: list[dict[str, Any]] = []

    def finish() -> None:
        if len(run) < 2:
            return
        onset, confirmation, end = run[0], run[1], run[-1]
        episodes.append(
            {
                "episode_id": (
                    f"{case['canonical_case_id']}:drift_{len(episodes) + 1:02d}"
                ),
                "canonical_case_id": case["canonical_case_id"],
                "persona_id": case["persona_id"],
                "dimension": case["dimension"],
                "historical_split": case["historical_split"],
                "cohort_source": source,
                "onset_position": int(onset["position"]),
                "confirmation_position": int(confirmation["position"]),
                "end_position": int(end["position"]),
                "onset_t_index": int(onset["t_index"]),
                "confirmation_t_index": int(confirmation["t_index"]),
                "end_t_index": int(end["t_index"]),
                "onset_date": str(onset["date"]),
                "confirmation_date": str(confirmation["date"]),
                "end_date": str(end["date"]),
                "length": len(run),
                "crosses_week": baseline._week_start(str(onset["date"]))
                != baseline._week_start(str(confirmation["date"])),
                "delivery_state": "active",
            }
        )

    for position in range(1, len(case["entries"]) + 1):
        row = by_position[position]
        adjacent = not run or int(row["t_index"]) == int(run[-1]["t_index"]) + 1
        if row["final_conflict"] is True and adjacent:
            run.append(row)
            continue
        finish()
        run = [row] if row["final_conflict"] is True else []
    finish()
    return episodes


def _build_union(
    config: dict[str, Any], root: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    paths = _source_paths(config, root)
    wrangled_dir = baseline._rooted(config["sources"]["wrangled_dir"], root)
    cohort_cases_raw = baseline._read_json(paths["cohort_cases_path"])
    if isinstance(cohort_cases_raw, dict):
        cohort_cases = cohort_cases_raw.get("cases")
    else:
        cohort_cases = cohort_cases_raw
    if not isinstance(cohort_cases, list):
        raise ValueError("twinkl-752.4 cohort cases must be a list")
    cohort_by_id = {
        _canonical_case_id(str(case["persona_id"]), str(case["dimension"])): case
        for case in cohort_cases
    }
    selected = pl.read_parquet(paths["selected_cases_path"])
    selected_by_id = {str(row["canonical_case_id"]): row for row in selected.to_dicts()}
    opus_targets = pl.read_parquet(paths["opus_entry_target_path"])
    opus_target_by_case = {
        str(key[0] if isinstance(key, tuple) else key): frame.sort(
            "position"
        ).to_dicts()
        for key, frame in opus_targets.partition_by(
            "canonical_case_id", as_dict=True
        ).items()
    }
    opus_outcomes = pl.read_parquet(paths["opus_case_outcomes_path"])
    opus_outcome_by_id = {
        str(row["canonical_case_id"]): row for row in opus_outcomes.to_dicts()
    }
    opus_episodes = pl.read_parquet(paths["opus_drift_episodes_path"])
    opus_episodes_by_case = {
        str(key[0] if isinstance(key, tuple) else key): frame.sort(
            "onset_position"
        ).to_dicts()
        for key, frame in opus_episodes.partition_by(
            "canonical_case_id", as_dict=True
        ).items()
    }

    cases: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for canonical_id in sorted(selected_by_id):
        raw = cohort_by_id[canonical_id]
        metadata = selected_by_id[canonical_id]
        outcome = opus_outcome_by_id[canonical_id]
        target_rows = opus_target_by_case[canonical_id]
        persona_id = str(raw["persona_id"])
        dimension = baseline._normalize_value(str(raw["dimension"]))
        profile, wrangled_entries = _profile_and_entries(
            persona_id=persona_id, wrangled_dir=wrangled_dir
        )
        wrangled_by_index = {int(entry["t_index"]): entry for entry in wrangled_entries}
        entries = []
        for position, (raw_entry, target) in enumerate(
            zip(raw["entries"], target_rows, strict=True), start=1
        ):
            if int(target["position"]) != position:
                raise ValueError(f"Non-contiguous target positions for {canonical_id}")
            t_index = int(raw_entry["t_index"])
            wrangled = wrangled_by_index[t_index]
            text = _runtime_text(wrangled)
            if str(raw_entry["date"]) != str(wrangled["date"]) or baseline._sha256_text(
                text
            ) != str(target["runtime_text_sha256"]):
                raise ValueError(
                    f"Runtime contract mismatch for {canonical_id}:{t_index}"
                )
            entries.append(
                {
                    "position": position,
                    "t_index": t_index,
                    "date": str(raw_entry["date"]),
                    "initial_entry": wrangled.get("initial_entry"),
                    "nudge_text": wrangled.get("nudge_text"),
                    "response_text": wrangled.get("response_text"),
                    "text": text,
                }
            )
            targets.append(
                {
                    "canonical_case_id": canonical_id,
                    "persona_id": persona_id,
                    "dimension": dimension,
                    "historical_split": str(metadata["historical_split"]),
                    "cohort_source": "twinkl_752_4_opus_resolved",
                    "opus_resolved": target.get("opus_adjudicator_disposition")
                    is not None,
                    "position": position,
                    "t_index": t_index,
                    "date": str(raw_entry["date"]),
                    "runtime_text_sha256": str(target["runtime_text_sha256"]),
                    "final_conflict": bool(target["final_conflict"]),
                    "resolution_method": str(target["resolution_method"]),
                    "resolution_status": str(target["resolution_status"]),
                }
            )
        full_values = [
            baseline._normalize_value(value)
            for value in profile.get("core_values") or []
        ]
        if dimension not in full_values:
            raise ValueError(f"Reviewed value is absent from profile: {canonical_id}")
        case = {
            "canonical_case_id": canonical_id,
            "persona_id": persona_id,
            "dimension": dimension,
            "full_core_values": full_values,
            "historical_split": str(metadata["historical_split"]),
            "cohort_source": "twinkl_752_4_opus_resolved",
            "cohort_role": str(metadata["cohort_role"]),
            "analysis_role": str(metadata["analysis_role"]),
            "case_content_sha256": str(metadata["case_content_sha256"]),
            "opus_resolved": any(
                row.get("opus_adjudicator_disposition") is not None
                for row in target_rows
            ),
            "entries": entries,
        }
        cases.append(case)
        case_episodes = []
        for row in opus_episodes_by_case.get(canonical_id, []):
            case_episodes.append(
                {
                    "episode_id": str(row["episode_id"]),
                    "canonical_case_id": canonical_id,
                    "persona_id": persona_id,
                    "dimension": dimension,
                    "historical_split": str(metadata["historical_split"]),
                    "cohort_source": "twinkl_752_4_opus_resolved",
                    "onset_position": int(row["onset_position"]),
                    "confirmation_position": int(row["confirmation_position"]),
                    "end_position": int(row["end_position"]),
                    "onset_t_index": int(row["onset_t_index"]),
                    "confirmation_t_index": int(row["confirmation_t_index"]),
                    "end_t_index": int(row["end_t_index"]),
                    "onset_date": str(row["onset_date"]),
                    "confirmation_date": str(row["confirmation_date"]),
                    "end_date": str(row["end_date"]),
                    "length": int(row["length"]),
                    "crosses_week": baseline._week_start(str(row["onset_date"]))
                    != baseline._week_start(str(row["confirmation_date"])),
                    "delivery_state": (
                        "active" if bool(row["open_at_cutoff"]) else "recovered"
                    ),
                }
            )
        episodes.extend(case_episodes)
        summaries.append(
            {
                "canonical_case_id": canonical_id,
                "persona_id": persona_id,
                "dimension": dimension,
                "historical_split": str(metadata["historical_split"]),
                "cohort_source": "twinkl_752_4_opus_resolved",
                "cohort_role": str(metadata["cohort_role"]),
                "analysis_role": str(metadata["analysis_role"]),
                "opus_resolved": case["opus_resolved"],
                "entry_count": len(entries),
                "resolved_entry_count": len(entries),
                "drift_count": int(outcome["episode_count"]),
                "has_drift": bool(outcome["has_drift"]),
            }
        )

    packet = _case_map(baseline._read_json(paths["omitted_packet_path"]))
    key = _case_map(baseline._read_json(paths["omitted_reconciliation_key_path"]))
    reviewer_a = _case_map(baseline._read_json(paths["omitted_reviewer_a_path"]))
    reviewer_b = _case_map(baseline._read_json(paths["omitted_reviewer_b_path"]))
    key_by_canonical = {
        _canonical_case_id(str(row["persona_id"]), str(row["dimension"])): case_id
        for case_id, row in key.items()
    }
    for canonical_id in config["union"]["omitted_case_ids"]:
        review_case_id = key_by_canonical[str(canonical_id)]
        key_case = key[review_case_id]
        packet_case = packet[review_case_id]
        first = reviewer_a[review_case_id]
        second = reviewer_b[review_case_id]
        if (
            first["sustained_conflict"] != "yes"
            or second["sustained_conflict"] != "yes"
        ):
            raise ValueError(f"Omitted case is not a resolved Drift: {canonical_id}")
        persona_id = str(key_case["persona_id"])
        dimension = baseline._normalize_value(str(key_case["dimension"]))
        profile, wrangled_entries = _profile_and_entries(
            persona_id=persona_id, wrangled_dir=wrangled_dir
        )
        wrangled_by_index = {int(entry["t_index"]): entry for entry in wrangled_entries}
        first_labels = {
            int(row["position"]): str(row["observable_negative"])
            for row in first["entry_assessments"]
        }
        second_labels = {
            int(row["position"]): str(row["observable_negative"])
            for row in second["entry_assessments"]
        }
        entries = []
        case_targets = []
        for packet_entry, key_entry in zip(
            packet_case["entries"], key_case["entries"], strict=True
        ):
            position = int(key_entry["position"])
            t_index = int(key_entry["t_index"])
            wrangled = wrangled_by_index[t_index]
            text = _runtime_text(wrangled)
            if text != str(packet_entry["journal_entry"]).strip() or str(
                key_entry["date"]
            ) != str(wrangled["date"]):
                raise ValueError(
                    f"Omitted runtime mismatch for {canonical_id}:{t_index}"
                )
            entries.append(
                {
                    "position": position,
                    "t_index": t_index,
                    "date": str(key_entry["date"]),
                    "initial_entry": wrangled.get("initial_entry"),
                    "nudge_text": wrangled.get("nudge_text"),
                    "response_text": wrangled.get("response_text"),
                    "text": text,
                }
            )
            agreement = first_labels[position] == second_labels[position]
            final_conflict = first_labels[position] == "yes" if agreement else None
            target = {
                "canonical_case_id": str(canonical_id),
                "persona_id": persona_id,
                "dimension": dimension,
                "historical_split": "validation",
                "cohort_source": "omitted_prior_drift",
                "opus_resolved": False,
                "position": position,
                "t_index": t_index,
                "date": str(key_entry["date"]),
                "runtime_text_sha256": baseline._sha256_text(text),
                "final_conflict": final_conflict,
                "resolution_method": (
                    "paired_agreement" if agreement else "paired_disagreement"
                ),
                "resolution_status": (
                    "resolved" if agreement else "unresolved_entry_only"
                ),
            }
            targets.append(target)
            case_targets.append(target)
        full_values = [
            baseline._normalize_value(value)
            for value in profile.get("core_values") or []
        ]
        if dimension not in full_values:
            raise ValueError(f"Omitted value absent from profile: {canonical_id}")
        case = {
            "canonical_case_id": str(canonical_id),
            "persona_id": persona_id,
            "dimension": dimension,
            "full_core_values": full_values,
            "historical_split": "validation",
            "cohort_source": "omitted_prior_drift",
            "cohort_role": "known_drift_omitted_by_candidate_mining",
            "analysis_role": "development_reference",
            "case_content_sha256": baseline._sha256_text(
                baseline._canonical_json(packet_case)
            ),
            "opus_resolved": False,
            "entries": entries,
        }
        cases.append(case)
        case_episodes = _episode_rows_from_targets(
            case=case,
            target_rows=case_targets,
            source="omitted_prior_drift",
        )
        if len(case_episodes) != 1:
            raise ValueError(f"Expected one omitted Drift for {canonical_id}")
        episodes.extend(case_episodes)
        summaries.append(
            {
                "canonical_case_id": str(canonical_id),
                "persona_id": persona_id,
                "dimension": dimension,
                "historical_split": "validation",
                "cohort_source": "omitted_prior_drift",
                "cohort_role": "known_drift_omitted_by_candidate_mining",
                "analysis_role": "development_reference",
                "opus_resolved": False,
                "entry_count": len(entries),
                "resolved_entry_count": sum(
                    row["final_conflict"] is not None for row in case_targets
                ),
                "drift_count": 1,
                "has_drift": True,
            }
        )
    return cases, summaries, targets, episodes


def _union_counts(
    *,
    cases: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
) -> dict[str, int]:
    unique_entries = {
        (case["persona_id"], int(entry["t_index"]))
        for case in cases
        for entry in case["entries"]
    }
    persona_weeks = {
        (case["persona_id"], baseline._week_start(str(entry["date"])).isoformat())
        for case in cases
        for entry in case["entries"]
    }
    return {
        "trajectories": len(cases),
        "personas": len({case["persona_id"] for case in cases}),
        "unique_entries": len(unique_entries),
        "entry_value_cells": len(targets),
        "resolved_entry_value_cells": sum(
            row["final_conflict"] is not None for row in targets
        ),
        "drifts": len(episodes),
        "drift_trajectories": sum(bool(row["has_drift"]) for row in summaries),
        "persona_weeks": len(persona_weeks),
    }


def _validate_expected_counts(config: dict[str, Any], counts: dict[str, int]) -> None:
    expected = {
        key.removeprefix("expected_"): int(value)
        for key, value in config["union"].items()
        if key.startswith("expected_")
    }
    if counts != expected:
        raise ValueError(
            f"Union count mismatch: expected={expected}, observed={counts}"
        )


def _write_union(config: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = _artifact_paths(config, root)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    cases, summaries, targets, episodes = _build_union(config, root)
    counts = _union_counts(
        cases=cases, summaries=summaries, targets=targets, episodes=episodes
    )
    _validate_expected_counts(config, counts)
    paths["union_cases"].write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pl.DataFrame(summaries).sort("canonical_case_id").write_parquet(
        paths["union_case_summary"]
    )
    pl.DataFrame(targets).sort("canonical_case_id", "position").write_parquet(
        paths["union_entry_target"]
    )
    pl.DataFrame(episodes).sort("canonical_case_id", "onset_position").write_parquet(
        paths["union_drift_episodes"]
    )
    source_hashes = {
        key: baseline._sha256_file(path)
        for key, path in _source_paths(config, root).items()
    }
    omitted_ids = set(config["union"]["omitted_case_ids"])
    receipt = {
        "study_id": config["study_id"],
        "schema_version": "twinkl-752.5-union-v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "repo_head_before_freeze": baseline._git_head(root),
        "counts": counts,
        "source_hashes": source_hashes,
        "output_hashes": {
            key: baseline._sha256_file(paths[key])
            for key in (
                "union_cases",
                "union_case_summary",
                "union_entry_target",
                "union_drift_episodes",
            )
        },
        "omitted_case_payload_hashes": {
            case["canonical_case_id"]: baseline._sha256_text(
                baseline._canonical_json(case)
            )
            for case in cases
            if case["canonical_case_id"] in omitted_ids
        },
        "opus_model_proof": {
            "raw_response_model_field": baseline._read_json(
                _source_paths(config, root)["opus_raw_response_path"]
            ).get("model"),
            "recorded_runtime_models": baseline._read_json(
                _source_paths(config, root)["opus_labels_path"]
            ).get("adjudicator_runtime"),
            "note": (
                "The raw CLI JSON model field is null; the frozen command and "
                "labels receipt record the requested Opus model."
            ),
        },
        "restrictions": config["decision"]["restrictions"],
    }
    paths["union_receipt"].write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def _load_frozen_union(
    config: dict[str, Any], root: Path
) -> tuple[
    list[dict[str, Any]],
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[str, Any],
]:
    paths = _artifact_paths(config, root)
    receipt = baseline._read_json(paths["union_receipt"])
    for key, digest in receipt["source_hashes"].items():
        if baseline._sha256_file(_source_paths(config, root)[key]) != digest:
            raise ValueError(f"Frozen union source hash mismatch: {key}")
    for key, digest in receipt["output_hashes"].items():
        if baseline._sha256_file(paths[key]) != digest:
            raise ValueError(f"Frozen union output hash mismatch: {key}")
    cases_raw = json.loads(paths["union_cases"].read_text(encoding="utf-8"))
    if not isinstance(cases_raw, list):
        raise ValueError("Frozen union cases must be a list")
    summaries = pl.read_parquet(paths["union_case_summary"])
    targets = pl.read_parquet(paths["union_entry_target"])
    episodes = pl.read_parquet(paths["union_drift_episodes"])
    counts = _union_counts(
        cases=cases_raw,
        summaries=summaries.to_dicts(),
        targets=targets.to_dicts(),
        episodes=episodes.to_dicts(),
    )
    _validate_expected_counts(config, counts)
    if counts != receipt["counts"]:
        raise ValueError("Frozen union receipt counts do not match its files")
    return cases_raw, summaries, targets, episodes, receipt


def _persona_records(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    personas: dict[str, dict[str, Any]] = {}
    for case in cases:
        persona_id = str(case["persona_id"])
        record = personas.setdefault(
            persona_id,
            {
                "persona_id": persona_id,
                "reviewed_values": set(),
                "full_core_values": list(case["full_core_values"]),
                "entries": case["entries"],
            },
        )
        if record["full_core_values"] != list(
            case["full_core_values"]
        ) or baseline._canonical_json(record["entries"]) != baseline._canonical_json(
            case["entries"]
        ):
            raise ValueError(f"Conflicting duplicate persona trajectory: {persona_id}")
        record["reviewed_values"].add(str(case["dimension"]))
    for record in personas.values():
        record["reviewed_values"] = sorted(record["reviewed_values"])
    return personas


def _score_run_020(
    config: dict[str, Any],
    root: Path,
    cases: list[dict[str, Any]],
    targets: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    paths = _artifact_paths(config, root)
    personas = _persona_records(cases)
    score_cases = [
        {
            "persona_id": record["persona_id"],
            "core_values": record["full_core_values"],
            "entries": record["entries"],
        }
        for record in personas.values()
    ]
    full = score_mlp_cases(
        cases=score_cases,
        checkpoint_path=baseline._rooted(config["critic"]["checkpoint_path"], root),
        arm_id="run_020",
        output_path=paths["run_020_full_evidence"],
        mc_seed=int(config["critic"]["mc_seed"]),
        mc_samples=int(config["critic"]["mc_samples"]),
    )
    target_keys = targets.select("persona_id", "dimension", "t_index").unique()
    target_evidence = full.join(
        target_keys, on=["persona_id", "dimension", "t_index"], how="inner"
    ).sort("persona_id", "dimension", "t_index")
    if target_evidence.height != targets.height:
        raise ValueError(
            "run_020 target coverage mismatch: "
            f"{target_evidence.height}/{targets.height}"
        )
    target_evidence.write_parquet(paths["run_020_target_evidence"])
    historical = pl.read_parquet(
        baseline._rooted(config["sources"]["historical_run_020_evidence_path"], root)
    )
    overlap = target_evidence.join(
        historical,
        on=["persona_id", "dimension", "t_index", "date"],
        how="inner",
        suffix="_historical",
    )
    if overlap.is_empty():
        raise ValueError("No parity overlap with frozen run_020 evidence")
    max_probability_delta = float(
        overlap.select(
            (pl.col("p_minus1") - pl.col("p_minus1_historical"))
            .abs()
            .max()
            .alias("delta")
        )["delta"][0]
    )
    if (
        max_probability_delta > 1e-6
        or overlap.filter(
            pl.col("predicted_class") != pl.col("predicted_class_historical")
        ).height
    ):
        raise ValueError("Regenerated run_020 probabilities fail frozen parity")
    parity = {
        "overlap_cells": overlap.height,
        "max_p_minus1_absolute_delta": max_probability_delta,
        "mean_uncertainty_absolute_delta": float(
            overlap.select(
                (pl.col("uncertainty") - pl.col("uncertainty_historical"))
                .abs()
                .mean()
                .alias("delta")
            )["delta"][0]
        ),
        "uncertainty_note": (
            "P(-1) parity is exact. Uncertainty was regenerated with the newly "
            "frozen MC-dropout seed because the historical scorer was unseeded."
        ),
    }
    return target_evidence, parity


def _scheduler_inputs(
    config: dict[str, Any],
    personas: dict[str, dict[str, Any]],
    evidence: pl.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_map = {
        (str(row["persona_id"]), str(row["dimension"]), int(row["t_index"])): row
        for row in evidence.to_dicts()
    }
    probability_threshold = float(config["critic"]["probability_threshold"])
    uncertainty_threshold = float(config["critic"]["uncertainty_threshold"])
    opportunities: list[dict[str, Any]] = []
    triggers: list[dict[str, Any]] = []
    triggered_weeks: set[tuple[str, str]] = set()
    for persona_id in sorted(personas):
        persona = personas[persona_id]
        entries = sorted(persona["entries"], key=lambda row: int(row["t_index"]))
        by_index = {int(row["t_index"]): row for row in entries}
        for current in entries:
            current_index = int(current["t_index"])
            previous = by_index.get(current_index - 1)
            current_date = date.fromisoformat(str(current["date"]))
            if previous is None or current_date.weekday() == 6:
                continue
            week_start = baseline._week_start(str(current["date"])).isoformat()
            opportunity = {
                "opportunity_id": f"{persona_id}:{week_start}:{current_index}",
                "persona_id": persona_id,
                "week_start": week_start,
                "review_at_date": str(current["date"]),
                "cutoff_t_index": current_index,
                "reviewed_values": persona["reviewed_values"],
            }
            opportunities.append(opportunity)
            qualifying_values = []
            for value in persona["reviewed_values"]:
                first = evidence_map[(persona_id, value, current_index - 1)]
                second = evidence_map[(persona_id, value, current_index)]
                pair_probability = statistics.fmean(
                    [float(first["p_minus1"]), float(second["p_minus1"])]
                )
                pair_uncertainty = max(
                    float(first["uncertainty"]), float(second["uncertainty"])
                )
                if (
                    pair_probability >= probability_threshold
                    and pair_uncertainty <= uncertainty_threshold
                ):
                    qualifying_values.append(value)
            week_key = (persona_id, week_start)
            if qualifying_values and week_key not in triggered_weeks:
                triggered_weeks.add(week_key)
                triggers.append(
                    {
                        **opportunity,
                        "trigger_id": f"trigger:{persona_id}:{week_start}",
                        "qualifying_values": qualifying_values,
                    }
                )
    return opportunities, triggers


def _build_prompt_records(
    config: dict[str, Any],
    root: Path,
    personas: dict[str, dict[str, Any]],
    evidence: pl.DataFrame,
    triggers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    template = load_prompt(str(config["study"]["prompt_name"]))
    evidence_map = {
        (str(row["persona_id"]), str(row["dimension"]), int(row["t_index"])): {
            "p_minus1": float(row["p_minus1"]),
            "uncertainty": float(row["uncertainty"]),
        }
        for row in evidence.to_dicts()
    }
    trigger_by_week = {
        (str(row["persona_id"]), str(row["week_start"])): row for row in triggers
    }
    records = []
    for persona_id in sorted(personas):
        persona = personas[persona_id]
        entries = sorted(persona["entries"], key=lambda row: int(row["t_index"]))
        starts = sorted({baseline._week_start(str(row["date"])) for row in entries})
        for start in starts:
            end = start + timedelta(days=6)
            history = [
                row for row in entries if date.fromisoformat(str(row["date"])) <= end
            ]
            current = [
                row
                for row in entries
                if start <= date.fromisoformat(str(row["date"])) <= end
            ]
            shared = {
                "declared_values": "\n".join(
                    f"- {value}" for value in persona["reviewed_values"]
                ),
                "cumulative_history": baseline._format_history(history),
                "current_week_entries": baseline._format_current_entries(current),
            }
            runtime_sha = baseline._sha256_text(
                baseline._canonical_json(
                    [
                        {"t_index": int(row["t_index"]), "text": row["text"]}
                        for row in history
                    ]
                )
            )
            expected = [
                {"t_index": int(row["t_index"]), "dimension": value}
                for row in current
                for value in persona["reviewed_values"]
            ]
            critic_block = baseline._critic_block(
                persona_id=persona_id,
                values=persona["reviewed_values"],
                current_entries=current,
                evidence=evidence_map,
            )
            for arm, inserted in (
                (WEEKLY_WITHOUT, ""),
                (WEEKLY_WITH_RAW, critic_block),
            ):
                prompt = template.render(**shared, critic_block=inserted).strip() + "\n"
                records.append(
                    {
                        "review_event_id": f"{arm}:{persona_id}:{start.isoformat()}",
                        "persona_id": persona_id,
                        "week_start": start.isoformat(),
                        "week_end": end.isoformat(),
                        "review_at_date": end.isoformat(),
                        "cutoff_t_index": max(int(row["t_index"]) for row in current),
                        "arm": arm,
                        "declared_values": persona["reviewed_values"],
                        "current_t_indices": [int(row["t_index"]) for row in current],
                        "expected_coordinates": expected,
                        "runtime_text_sha256": runtime_sha,
                        "critic_block_sha256": baseline._sha256_text(inserted),
                        "prompt": prompt,
                        "prompt_sha256": baseline._sha256_text(prompt),
                    }
                )
            trigger = trigger_by_week.get((persona_id, start.isoformat()))
            if trigger is None:
                continue
            cutoff = int(trigger["cutoff_t_index"])
            early_history = [row for row in entries if int(row["t_index"]) <= cutoff]
            early_current = [
                row
                for row in early_history
                if baseline._week_start(str(row["date"])) == start
            ]
            early_shared = {
                "declared_values": shared["declared_values"],
                "cumulative_history": baseline._format_history(early_history),
                "current_week_entries": baseline._format_current_entries(early_current),
            }
            prompt = template.render(**early_shared, critic_block="").strip() + "\n"
            early_runtime_sha = baseline._sha256_text(
                baseline._canonical_json(
                    [
                        {"t_index": int(row["t_index"]), "text": row["text"]}
                        for row in early_history
                    ]
                )
            )
            records.append(
                {
                    "review_event_id": trigger["trigger_id"],
                    "persona_id": persona_id,
                    "week_start": start.isoformat(),
                    "week_end": end.isoformat(),
                    "review_at_date": trigger["review_at_date"],
                    "cutoff_t_index": cutoff,
                    "arm": EARLY_WITHOUT,
                    "declared_values": persona["reviewed_values"],
                    "current_t_indices": [int(row["t_index"]) for row in early_current],
                    "expected_coordinates": [
                        {"t_index": int(row["t_index"]), "dimension": value}
                        for row in early_current
                        for value in persona["reviewed_values"]
                    ],
                    "runtime_text_sha256": early_runtime_sha,
                    "critic_block_sha256": baseline._sha256_text(""),
                    "prompt": prompt,
                    "prompt_sha256": baseline._sha256_text(prompt),
                }
            )
    return records


def _prepare(config: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = _artifact_paths(config, root)
    cases, _summaries, targets, _episodes, union_receipt = _load_frozen_union(
        config, root
    )
    evidence, parity = _score_run_020(config, root, cases, targets)
    personas = _persona_records(cases)
    opportunities, triggers = _scheduler_inputs(config, personas, evidence)
    paths["eligible_opportunities"].write_text(
        json.dumps(opportunities, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["trigger_schedule"].write_text(
        json.dumps(triggers, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    records = _build_prompt_records(config, root, personas, evidence, triggers)
    baseline._write_jsonl(paths["prompts"], records)
    estimate = baseline.estimate_plan(records, config)
    expected_weekly = int(config["union"]["expected_persona_weeks"])
    counts = {
        arm: sum(record["arm"] == arm for record in records)
        for arm in (WEEKLY_WITHOUT, WEEKLY_WITH_RAW, EARLY_WITHOUT)
    }
    if (
        counts[WEEKLY_WITHOUT] != expected_weekly
        or counts[WEEKLY_WITH_RAW] != expected_weekly
    ):
        raise ValueError(f"Weekly prompt count mismatch: {counts}")
    if counts[EARLY_WITHOUT] != len(triggers):
        raise ValueError("Early prompt count differs from trigger count")
    source_paths = _source_paths(config, root)
    manifest = {
        "study_id": config["study_id"],
        "schema_version": "twinkl-752.5-protocol-v1",
        "prepared_at": datetime.now(UTC).isoformat(),
        "repo_head_before_preregistration_commit": baseline._git_head(root),
        "config_sha256": baseline._sha256_text(baseline._canonical_json(config)),
        "union_receipt_sha256": baseline._sha256_file(paths["union_receipt"]),
        "union_output_hashes": union_receipt["output_hashes"],
        "run_020_yaml_sha256": baseline._sha256_file(source_paths["run_020_yaml_path"]),
        "run_020_checkpoint_sha256": baseline._sha256_file(
            baseline._rooted(config["critic"]["checkpoint_path"], root)
        ),
        "trigger_threshold_receipt_sha256": baseline._sha256_file(
            source_paths["trigger_threshold_receipt_path"]
        ),
        "run_020_full_evidence_sha256": baseline._sha256_file(
            paths["run_020_full_evidence"]
        ),
        "run_020_target_evidence_sha256": baseline._sha256_file(
            paths["run_020_target_evidence"]
        ),
        "eligible_opportunities_sha256": baseline._sha256_file(
            paths["eligible_opportunities"]
        ),
        "trigger_schedule_sha256": baseline._sha256_file(paths["trigger_schedule"]),
        "prompt_template_sha256": baseline._sha256_file(
            baseline._rooted(config["study"]["prompt_path"], root)
        ),
        "runner_sha256": baseline._sha256_file(Path(__file__)),
        "scorer_sha256": baseline._sha256_file(root / "src/vif/drift_scoring.py"),
        "shared_api_runner_sha256": baseline._sha256_file(
            root / "scripts/experiments/weekly_verifier_ablation.py"
        ),
        "prompt_manifest_sha256": baseline._sha256_text(
            baseline._canonical_json(
                [
                    {
                        "review_event_id": record["review_event_id"],
                        "prompt_sha256": record["prompt_sha256"],
                    }
                    for record in records
                ]
            )
        ),
        "prompts_sha256": baseline._sha256_file(paths["prompts"]),
        "prompt_counts": counts,
        "trigger_count": len(triggers),
        "eligible_opportunity_count": len(opportunities),
        "run_020_parity": parity,
        "setups": config["study"]["setups"],
        "weekly_receipt_reuse": config["study"]["weekly_receipt_reuse"],
        "trigger_rule": config["critic"],
        "offline_trigger_placement": config["offline_trigger_placement"],
        "bootstrap": config["bootstrap"],
        "api": {
            key: config["api"][key]
            for key in (
                "provider",
                "endpoint",
                "model",
                "reasoning_effort",
                "max_output_tokens",
                "max_attempts",
                "concurrency",
                "store",
                "max_budget_usd",
            )
        },
        "estimate": estimate,
        "decision_rules": config["decision"],
        "outcomes_inspected": False,
        "offline_diagnostic_scored": False,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _load_prepared(
    config: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Path], dict[str, Any]]:
    paths = _artifact_paths(config, root)
    manifest = baseline._read_json(paths["manifest"])
    expected_hashes = {
        "config_sha256": baseline._sha256_text(baseline._canonical_json(config)),
        "runner_sha256": baseline._sha256_file(Path(__file__)),
        "scorer_sha256": baseline._sha256_file(root / "src/vif/drift_scoring.py"),
        "shared_api_runner_sha256": baseline._sha256_file(
            root / "scripts/experiments/weekly_verifier_ablation.py"
        ),
        "prompt_template_sha256": baseline._sha256_file(
            baseline._rooted(config["study"]["prompt_path"], root)
        ),
        "union_receipt_sha256": baseline._sha256_file(paths["union_receipt"]),
        "run_020_full_evidence_sha256": baseline._sha256_file(
            paths["run_020_full_evidence"]
        ),
        "run_020_target_evidence_sha256": baseline._sha256_file(
            paths["run_020_target_evidence"]
        ),
        "eligible_opportunities_sha256": baseline._sha256_file(
            paths["eligible_opportunities"]
        ),
        "trigger_schedule_sha256": baseline._sha256_file(paths["trigger_schedule"]),
        "prompts_sha256": baseline._sha256_file(paths["prompts"]),
    }
    for key, digest in expected_hashes.items():
        if manifest.get(key) != digest:
            raise ValueError(f"Prepared protocol hash mismatch: {key}")
    records = baseline._load_jsonl(paths["prompts"])
    for record in records:
        if baseline._sha256_text(record["prompt"]) != record["prompt_sha256"]:
            raise ValueError("Prepared prompt hash mismatch")
    return records, paths, manifest


def _assessment_map(
    responses: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    repeats: int,
    requested_model: str,
) -> dict[tuple[str, int, str, int, str], baseline.VerifierAssessment]:
    record_map = {
        (record["persona_id"], record["week_start"], record["arm"]): record
        for record in records
    }
    completed = baseline._completed_keys(
        responses,
        record_map,
        repeats=repeats,
        requested_model=requested_model,
    )
    expected = {
        (record["persona_id"], record["week_start"], record["arm"], repeat)
        for record in records
        for repeat in range(1, repeats + 1)
    }
    if completed != expected:
        raise ValueError(
            "Response set is incomplete: "
            f"completed={len(completed)}, expected={len(expected)}"
        )
    return baseline._flatten_predictions(responses)


def _labels_for_case(
    *,
    case: dict[str, Any],
    arm: str,
    repeat: int,
    predictions: dict[tuple[str, int, str, int, str], baseline.VerifierAssessment],
) -> list[bool | None]:
    labels = []
    for entry in case["entries"]:
        assessment = predictions.get(
            (
                arm,
                repeat,
                case["persona_id"],
                int(entry["t_index"]),
                case["dimension"],
            )
        )
        labels.append(
            None
            if assessment is None or assessment.verdict == "abstain"
            else assessment.verdict == "conflict"
        )
    return labels


def _predicted_episode_rows(
    *,
    case: dict[str, Any],
    labels: list[bool | None],
    source: str,
    alert_date_for_confirmation: Any,
) -> list[dict[str, Any]]:
    rows = []
    run: list[dict[str, Any]] = []

    def finish() -> None:
        if len(run) < 2:
            return
        onset, confirmation, end = run[0], run[1], run[-1]
        rows.append(
            {
                "episode_id": (
                    f"{source}:{case['canonical_case_id']}:"
                    f"{int(onset['t_index'])}:{int(end['t_index'])}"
                ),
                "canonical_case_id": case["canonical_case_id"],
                "persona_id": case["persona_id"],
                "dimension": case["dimension"],
                "onset_t_index": int(onset["t_index"]),
                "confirmation_t_index": int(confirmation["t_index"]),
                "end_t_index": int(end["t_index"]),
                "alert_date": str(alert_date_for_confirmation(confirmation)),
                "delivery_state": "active",
            }
        )

    for entry, label in zip(case["entries"], labels, strict=True):
        adjacent = not run or int(entry["t_index"]) == int(run[-1]["t_index"]) + 1
        if label is True and adjacent:
            run.append(entry)
            continue
        finish()
        run = [entry] if label is True else []
    finish()
    return rows


def _weekly_alert_date(entry: dict[str, Any]) -> str:
    return (baseline._week_start(str(entry["date"])) + timedelta(days=6)).isoformat()


def _merge_predicted_episodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda item: (
            item["canonical_case_id"],
            item["alert_date"],
            item["onset_t_index"],
        ),
    ):
        overlap = next(
            (
                existing
                for existing in merged
                if existing["canonical_case_id"] == row["canonical_case_id"]
                and max(existing["onset_t_index"], row["onset_t_index"])
                <= min(existing["end_t_index"], row["end_t_index"])
            ),
            None,
        )
        if overlap is None:
            merged.append(row)
            continue
        if row["alert_date"] < overlap["alert_date"]:
            overlap.update(row)
        overlap["end_t_index"] = max(overlap["end_t_index"], row["end_t_index"])
    for index, row in enumerate(merged, start=1):
        row["episode_id"] = f"merged:{row['canonical_case_id']}:{index:02d}"
    return merged


def _setup_predictions(
    *,
    cases: list[dict[str, Any]],
    records: list[dict[str, Any]],
    predictions: dict[tuple[str, int, str, int, str], baseline.VerifierAssessment],
    setup: str,
    repeat: int,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    if setup in {WEEKLY_WITHOUT, WEEKLY_WITH_RAW}:
        rows = []
        covered = {}
        for case in cases:
            labels = _labels_for_case(
                case=case,
                arm=setup,
                repeat=repeat,
                predictions=predictions,
            )
            rows.extend(
                _predicted_episode_rows(
                    case=case,
                    labels=labels,
                    source=f"{setup}:{repeat}",
                    alert_date_for_confirmation=_weekly_alert_date,
                )
            )
            covered[case["canonical_case_id"]] = baseline._trajectory_covered(
                labels, case["entries"]
            )
        return rows, covered
    if setup != SCHEDULED:
        raise ValueError(f"Unknown setup: {setup}")

    weekly_rows, _weekly_covered = _setup_predictions(
        cases=cases,
        records=records,
        predictions=predictions,
        setup=WEEKLY_WITHOUT,
        repeat=repeat,
    )
    early_records = [record for record in records if record["arm"] == EARLY_WITHOUT]
    early_by_persona = {}
    for record in early_records:
        early_by_persona.setdefault(record["persona_id"], []).append(record)
    early_rows = []
    covered = {}
    for case in cases:
        weekly_labels = _labels_for_case(
            case=case,
            arm=WEEKLY_WITHOUT,
            repeat=repeat,
            predictions=predictions,
        )
        fallback_labels = list(weekly_labels)
        for record in early_by_persona.get(case["persona_id"], []):
            cutoff = int(record["cutoff_t_index"])
            start = date.fromisoformat(record["week_start"])
            snapshot_labels = []
            for index, entry in enumerate(case["entries"]):
                entry_index = int(entry["t_index"])
                if entry_index > cutoff:
                    snapshot_labels.append(None)
                    continue
                if baseline._week_start(str(entry["date"])) == start:
                    assessment = predictions.get(
                        (
                            EARLY_WITHOUT,
                            repeat,
                            case["persona_id"],
                            entry_index,
                            case["dimension"],
                        )
                    )
                    label = (
                        None
                        if assessment is None or assessment.verdict == "abstain"
                        else assessment.verdict == "conflict"
                    )
                    snapshot_labels.append(label)
                    if fallback_labels[index] is None and label is not None:
                        fallback_labels[index] = label
                else:
                    snapshot_labels.append(weekly_labels[index])
            review_at_date = str(record["review_at_date"])
            snapshot_rows = _predicted_episode_rows(
                case=case,
                labels=snapshot_labels,
                source=f"{EARLY_WITHOUT}:{repeat}:{record['review_event_id']}",
                alert_date_for_confirmation=lambda _entry, value=review_at_date: value,
            )
            early_rows.extend(
                row for row in snapshot_rows if row["confirmation_t_index"] <= cutoff
            )
        covered[case["canonical_case_id"]] = baseline._trajectory_covered(
            fallback_labels, case["entries"]
        )
    return _merge_predicted_episodes(weekly_rows + early_rows), covered


def _episode_frame(rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
    return baseline._episode_frame(list(rows))


def _reference_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": str(row["episode_id"]),
            "canonical_case_id": str(row["canonical_case_id"]),
            "persona_id": str(row["persona_id"]),
            "dimension": str(row["dimension"]),
            "onset_t_index": int(row["onset_t_index"]),
            "confirmation_t_index": int(row["confirmation_t_index"]),
            "end_t_index": int(row["end_t_index"]),
            "confirmation_date": str(row["confirmation_date"]),
            "crosses_week": bool(row["crosses_week"]),
            "delivery_state": str(row["delivery_state"]),
        }
        for row in frame.to_dicts()
    ]


def _score_subset(
    *,
    case_ids: set[str],
    reference_rows: list[dict[str, Any]],
    predicted_rows: list[dict[str, Any]],
    covered: dict[str, bool],
    max_confirmation_lag: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    references = [row for row in reference_rows if row["canonical_case_id"] in case_ids]
    predicted = [row for row in predicted_rows if row["canonical_case_id"] in case_ids]
    matches = match_episodes(
        _episode_frame(references),
        _episode_frame(predicted),
        max_confirmation_lag=max_confirmation_lag,
    )
    match_rows = matches.to_dicts()
    predicted_by_id = {row["episode_id"]: row for row in predicted}
    reference_by_id = {row["episode_id"]: row for row in references}
    delays = []
    cross_week_hits = 0
    for match in match_rows:
        reference = reference_by_id[str(match["reference_episode_id"])]
        prediction = predicted_by_id[str(match["predicted_episode_id"])]
        delay = (
            date.fromisoformat(prediction["alert_date"])
            - date.fromisoformat(reference["confirmation_date"])
        ).days
        delays.append(delay)
        cross_week_hits += bool(reference["crosses_week"])
    hits = len(match_rows)
    false_alerts = len(predicted) - hits
    covered_count = sum(bool(covered.get(case_id)) for case_id in case_ids)
    metrics = {
        "drift_hits": hits,
        "known_drifts": len(references),
        "predicted_drift_alerts": len(predicted),
        "false_drift_alerts": false_alerts,
        "drift_recall": hits / len(references) if references else 0.0,
        "drift_precision": hits / len(predicted) if predicted else 0.0,
        "coverage_count": covered_count,
        "trajectory_count": len(case_ids),
        "coverage": covered_count / len(case_ids) if case_ids else 0.0,
        "abstention_rate": 1.0 - covered_count / len(case_ids) if case_ids else 0.0,
        "median_detection_delay_days": (
            float(statistics.median(delays)) if delays else None
        ),
        "mean_detection_delay_days": (
            float(statistics.fmean(delays)) if delays else None
        ),
        "cross_week_drift_hits": cross_week_hits,
        "cross_week_known_drifts": sum(bool(row["crosses_week"]) for row in references),
    }
    case_stats = {
        case_id: {
            "reference": 0,
            "hits": 0,
            "predicted": 0,
            "false_alerts": 0,
            "covered": bool(covered.get(case_id)),
            "delay_days": [],
        }
        for case_id in case_ids
    }
    for row in references:
        case_stats[row["canonical_case_id"]]["reference"] += 1
    for row in predicted:
        case_stats[row["canonical_case_id"]]["predicted"] += 1
    for match in match_rows:
        reference = reference_by_id[str(match["reference_episode_id"])]
        prediction = predicted_by_id[str(match["predicted_episode_id"])]
        stats = case_stats[reference["canonical_case_id"]]
        stats["hits"] += 1
        stats["delay_days"].append(
            (
                date.fromisoformat(prediction["alert_date"])
                - date.fromisoformat(reference["confirmation_date"])
            ).days
        )
    for stats in case_stats.values():
        stats["false_alerts"] = stats["predicted"] - stats["hits"]
    return metrics, case_stats


def _percentile_interval(values: list[float], interval: float) -> list[float]:
    alpha = (1.0 - interval) / 2.0
    return [
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    ]


def _comparison_bootstrap(
    *,
    case_ids: list[str],
    first: dict[int, dict[str, dict[str, Any]]],
    second: dict[int, dict[str, dict[str, Any]]],
    config: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    repeats = sorted(first)

    def aggregate(
        sample: list[str], stats: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        reference = sum(stats[case_id]["reference"] for case_id in sample)
        hits = sum(stats[case_id]["hits"] for case_id in sample)
        predicted = sum(stats[case_id]["predicted"] for case_id in sample)
        false_alerts = sum(stats[case_id]["false_alerts"] for case_id in sample)
        coverage = statistics.fmean(
            float(stats[case_id]["covered"]) for case_id in sample
        )
        delays = [delay for case_id in sample for delay in stats[case_id]["delay_days"]]
        return {
            "recall": hits / reference if reference else 0.0,
            "precision": hits / predicted if predicted else 0.0,
            "false_alerts": float(false_alerts),
            "coverage": coverage,
            "delay": float(statistics.median(delays)) if delays else math.nan,
        }

    observed = {}
    for metric in ("recall", "precision", "false_alerts", "coverage", "delay"):
        deltas = []
        for repeat in repeats:
            left = aggregate(case_ids, first[repeat])
            right = aggregate(case_ids, second[repeat])
            if not math.isnan(left[metric]) and not math.isnan(right[metric]):
                deltas.append(right[metric] - left[metric])
        observed[metric] = float(statistics.median(deltas)) if deltas else None

    rng = np.random.default_rng(int(config["bootstrap"]["random_seed"]) + seed_offset)
    distributions = {key: [] for key in observed}
    for _ in range(int(config["bootstrap"]["repeats"])):
        sample = [
            case_ids[index] for index in rng.integers(0, len(case_ids), len(case_ids))
        ]
        per_metric = {key: [] for key in distributions}
        for repeat in repeats:
            left = aggregate(sample, first[repeat])
            right = aggregate(sample, second[repeat])
            for metric in per_metric:
                if not math.isnan(left[metric]) and not math.isnan(right[metric]):
                    per_metric[metric].append(right[metric] - left[metric])
        for metric, values in per_metric.items():
            if values:
                distributions[metric].append(float(statistics.median(values)))
    interval = float(config["bootstrap"]["interval"])
    return {
        "unit": "trajectory",
        "bootstrap_repeats": int(config["bootstrap"]["repeats"]),
        "interval": interval,
        "deltas": {
            metric: {
                "observed_median": observed[metric],
                "interval": (
                    _percentile_interval(values, interval) if values else None
                ),
            }
            for metric, values in distributions.items()
        },
    }


def _placement_diagnostic(
    *,
    config: dict[str, Any],
    opportunities: list[dict[str, Any]],
    triggers: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reference_by_week: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in reference_rows:
        week = baseline._week_start(row["confirmation_date"]).isoformat()
        reference_by_week.setdefault((row["persona_id"], week), []).append(row)

    def placement_hits(row: dict[str, Any]) -> bool:
        key = (str(row["persona_id"]), str(row["week_start"]))
        cutoff = int(row["cutoff_t_index"])
        return any(
            int(reference["confirmation_t_index"]) <= cutoff
            for reference in reference_by_week.get(key, [])
        )

    observed_hits = sum(placement_hits(row) for row in triggers)
    trigger_count = len(triggers)
    observed_rate = observed_hits / trigger_count if trigger_count else 0.0
    by_week: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in opportunities:
        by_week.setdefault((row["persona_id"], row["week_start"]), []).append(row)
    week_keys = sorted(by_week)
    if trigger_count > len(week_keys):
        raise ValueError("Trigger count exceeds eligible persona-weeks")
    random_rates = []
    start_seed = int(config["offline_trigger_placement"]["random_seed_start"])
    random_repeats = int(config["offline_trigger_placement"]["random_repeats"])
    for offset in range(random_repeats):
        rng = random.Random(start_seed + offset)
        selected_weeks = rng.sample(week_keys, trigger_count)
        placements = [rng.choice(by_week[key]) for key in selected_weeks]
        random_rates.append(
            sum(placement_hits(row) for row in placements) / trigger_count
            if trigger_count
            else 0.0
        )
    return {
        "weekly_reviewer_calls": 0,
        "eligible_opportunities": len(opportunities),
        "eligible_persona_weeks": len(week_keys),
        "realized_trigger_count": trigger_count,
        "realized_trigger_rate_per_eligible_persona_week": (
            trigger_count / len(week_keys) if week_keys else 0.0
        ),
        "realized_trigger_rate_per_union_persona_week": trigger_count
        / int(config["union"]["expected_persona_weeks"]),
        "observed_trigger_hits": observed_hits,
        "observed_trigger_hit_rate": observed_rate,
        "random_repeats": random_repeats,
        "random_mean_hit_rate": float(statistics.fmean(random_rates)),
        "random_median_hit_rate": float(statistics.median(random_rates)),
        "random_95_interval": _percentile_interval(random_rates, 0.95),
        "observed_percentile": sum(rate <= observed_rate for rate in random_rates)
        / random_repeats,
        "one_sided_p_random_at_least_observed": (
            1 + sum(rate >= observed_rate for rate in random_rates)
        )
        / (random_repeats + 1),
        "interpretation_limit": (
            "This tests whether run_020 triggers target Drift-relevant review "
            "opportunities better than random placement. It does not test whether "
            "early review improves Drift detection."
        ),
    }


def _arm_response_summary(
    responses: list[dict[str, Any]], config: dict[str, Any], arm: str
) -> dict[str, Any]:
    selected = [row for row in responses if row.get("arm") == arm]
    return baseline._response_summary(selected, config)


def _score(
    *,
    config: dict[str, Any],
    root: Path,
    records: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    cases, summaries, _targets, reference_frame, _receipt = _load_frozen_union(
        config, root
    )
    predictions = _assessment_map(
        responses,
        records,
        repeats=int(config["study"]["repeats"]),
        requested_model=str(config["api"]["model"]),
    )
    reference_rows = _reference_rows(reference_frame)
    all_case_ids = sorted(case["canonical_case_id"] for case in cases)
    summary_rows = summaries.to_dicts()
    subgroup_case_ids = {
        "primary": set(all_case_ids),
        "excluding_opus_resolved": {
            row["canonical_case_id"] for row in summary_rows if not row["opus_resolved"]
        },
        "training_seen": {
            row["canonical_case_id"]
            for row in summary_rows
            if row["historical_split"] == "training"
        },
        "non_training": {
            row["canonical_case_id"]
            for row in summary_rows
            if row["historical_split"] != "training"
        },
    }
    for split in sorted({str(row["historical_split"]) for row in summary_rows}):
        subgroup_case_ids[f"historical_split:{split}"] = {
            row["canonical_case_id"]
            for row in summary_rows
            if row["historical_split"] == split
        }

    results = []
    primary_case_stats: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
        setup: {} for setup in (WEEKLY_WITHOUT, WEEKLY_WITH_RAW, SCHEDULED)
    }
    subgroup_results = {}
    for repeat in range(1, int(config["study"]["repeats"]) + 1):
        predicted_by_setup = {}
        covered_by_setup = {}
        for setup in (WEEKLY_WITHOUT, WEEKLY_WITH_RAW, SCHEDULED):
            predicted_rows, covered = _setup_predictions(
                cases=cases,
                records=records,
                predictions=predictions,
                setup=setup,
                repeat=repeat,
            )
            predicted_by_setup[setup] = predicted_rows
            covered_by_setup[setup] = covered
            metrics, case_stats = _score_subset(
                case_ids=subgroup_case_ids["primary"],
                reference_rows=reference_rows,
                predicted_rows=predicted_rows,
                covered=covered,
                max_confirmation_lag=int(
                    config["study"]["max_confirmation_lag_entries"]
                ),
            )
            primary_case_stats[setup][repeat] = case_stats
            results.append({"setup": setup, "repeat": repeat, **metrics})
        for subgroup, case_ids in subgroup_case_ids.items():
            subgroup_rows = subgroup_results.setdefault(subgroup, [])
            for setup in (WEEKLY_WITHOUT, WEEKLY_WITH_RAW, SCHEDULED):
                metrics, _case_stats = _score_subset(
                    case_ids=case_ids,
                    reference_rows=reference_rows,
                    predicted_rows=predicted_by_setup[setup],
                    covered=covered_by_setup[setup],
                    max_confirmation_lag=int(
                        config["study"]["max_confirmation_lag_entries"]
                    ),
                )
                subgroup_rows.append({"setup": setup, "repeat": repeat, **metrics})

    raw_comparison = _comparison_bootstrap(
        case_ids=all_case_ids,
        first=primary_case_stats[WEEKLY_WITHOUT],
        second=primary_case_stats[WEEKLY_WITH_RAW],
        config=config,
        seed_offset=1,
    )
    scheduling_comparison = _comparison_bootstrap(
        case_ids=all_case_ids,
        first=primary_case_stats[WEEKLY_WITHOUT],
        second=primary_case_stats[SCHEDULED],
        config=config,
        seed_offset=2,
    )
    raw_recall = raw_comparison["deltas"]["recall"]
    raw_false = raw_comparison["deltas"]["false_alerts"]["observed_median"]
    raw_coverage = raw_comparison["deltas"]["coverage"]["observed_median"]
    raw_interval = raw_recall["interval"]
    if (
        raw_recall["observed_median"] > 0
        and raw_interval is not None
        and raw_interval[0] > 0
        and raw_false <= 0
        and raw_coverage >= -0.05
    ):
        raw_verdict = "reverses"
    elif (
        raw_recall["observed_median"] < 0
        and raw_interval is not None
        and raw_interval[1] < 0
    ):
        raw_verdict = "survives"
    else:
        raw_verdict = "inconclusive"

    schedule_recall = scheduling_comparison["deltas"]["recall"]["observed_median"]
    schedule_false = scheduling_comparison["deltas"]["false_alerts"]["observed_median"]
    schedule_delay = scheduling_comparison["deltas"]["delay"]["observed_median"]
    if schedule_recall > 0 and schedule_false <= 0:
        scheduling_verdict = "added_drift_hits_without_additional_false_alerts"
    elif schedule_recall > 0:
        scheduling_verdict = "added_drift_hits_with_additional_false_alerts"
    elif schedule_delay is not None and schedule_delay < 0:
        scheduling_verdict = "earlier_detection_only"
    else:
        scheduling_verdict = "no_observed_scheduling_benefit"

    paths = _artifact_paths(config, root)
    opportunities = json.loads(
        paths["eligible_opportunities"].read_text(encoding="utf-8")
    )
    triggers = json.loads(paths["trigger_schedule"].read_text(encoding="utf-8"))
    placement = _placement_diagnostic(
        config=config,
        opportunities=opportunities,
        triggers=triggers,
        reference_rows=reference_rows,
    )
    arm_summaries = {
        arm: _arm_response_summary(responses, config, arm)
        for arm in (WEEKLY_WITHOUT, WEEKLY_WITH_RAW, EARLY_WITHOUT)
    }
    logical_calls = {
        WEEKLY_WITHOUT: arm_summaries[WEEKLY_WITHOUT]["total"],
        WEEKLY_WITH_RAW: arm_summaries[WEEKLY_WITH_RAW]["total"],
        SCHEDULED: arm_summaries[WEEKLY_WITHOUT]["total"]
        + arm_summaries[EARLY_WITHOUT]["total"],
    }
    logical_cost = {
        WEEKLY_WITHOUT: arm_summaries[WEEKLY_WITHOUT]["actual_spend_usd"],
        WEEKLY_WITH_RAW: arm_summaries[WEEKLY_WITH_RAW]["actual_spend_usd"],
        SCHEDULED: arm_summaries[WEEKLY_WITHOUT]["actual_spend_usd"]
        + arm_summaries[EARLY_WITHOUT]["actual_spend_usd"],
    }
    return {
        "study_id": config["study_id"],
        "scored_at": datetime.now(UTC).isoformat(),
        "union_counts": _union_counts(
            cases=cases,
            summaries=summary_rows,
            targets=pl.read_parquet(paths["union_entry_target"]).to_dicts(),
            episodes=reference_rows,
        ),
        "results": results,
        "subgroups": subgroup_results,
        "paired_trajectory_bootstrap": {
            "raw_input": raw_comparison,
            "scheduling": scheduling_comparison,
        },
        "architecture_questions": {
            "raw_score_value": {
                "comparison": f"{WEEKLY_WITHOUT} versus {WEEKLY_WITH_RAW}",
                "old_conditional_rejection": raw_verdict,
            },
            "scheduling_value": {
                "comparison": f"{WEEKLY_WITHOUT} versus {SCHEDULED}",
                "development_result": scheduling_verdict,
                "boundary": (
                    "This is a review-again result. Review-early was not tested, "
                    "and no architecture is adopted."
                ),
            },
        },
        "offline_trigger_placement": placement,
        "response_summary_by_call_type": arm_summaries,
        "logical_setup_calls": logical_calls,
        "logical_setup_cost_usd": logical_cost,
        "actual_api_spend_usd": sum(
            summary["actual_spend_usd"] for summary in arm_summaries.values()
        ),
        "restrictions": config["decision"]["restrictions"],
        "artifact_provenance": {
            "union_receipt_sha256": baseline._sha256_file(paths["union_receipt"]),
            "manifest_sha256": baseline._sha256_file(paths["manifest"]),
            "prompts_sha256": baseline._sha256_file(paths["prompts"]),
            "responses_sha256": baseline._sha256_file(paths["responses"]),
            "run_020_target_evidence_sha256": baseline._sha256_file(
                paths["run_020_target_evidence"]
            ),
            "trigger_schedule_sha256": baseline._sha256_file(paths["trigger_schedule"]),
        },
    }


def command_freeze(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    receipt = _write_union(config, root)
    print(json.dumps(receipt["counts"], indent=2, sort_keys=True))


def command_prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    manifest = _prepare(config, root)
    print(
        json.dumps(
            {
                "prompt_counts": manifest["prompt_counts"],
                "trigger_count": manifest["trigger_count"],
                "estimate": manifest["estimate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_estimate(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    records, _paths, manifest = _load_prepared(config, root)
    estimate = baseline.estimate_plan(records, config)
    if estimate != manifest["estimate"]:
        raise ValueError("Prepared cost estimate changed")
    print(json.dumps(estimate, indent=2, sort_keys=True))


def command_run(args: argparse.Namespace) -> None:
    if not args.execute:
        raise SystemExit("Refusing paid calls without --execute")
    from dotenv import load_dotenv

    root = Path(args.root).resolve()
    load_dotenv(root / ".env")
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    records, paths, _manifest = _load_prepared(config, root)
    result = asyncio.run(
        baseline.execute_calls(
            records=records, config=config, output_path=paths["responses"]
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def command_score(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    records, paths, _manifest = _load_prepared(config, root)
    responses = baseline._load_jsonl(paths["responses"])
    metrics = _score(
        config=config,
        root=root,
        records=records,
        responses=responses,
    )
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["architecture_questions"], indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze").set_defaults(func=command_freeze)
    subparsers.add_parser("prepare").set_defaults(func=command_prepare)
    subparsers.add_parser("estimate").set_defaults(func=command_estimate)
    run = subparsers.add_parser("run")
    run.add_argument("--execute", action="store_true")
    run.set_defaults(func=command_run)
    subparsers.add_parser("score").set_defaults(func=command_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
