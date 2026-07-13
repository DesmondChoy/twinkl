#!/usr/bin/env python3
"""Prepare, run, and score the bounded twinkl-752.3 prompt-alignment study.

The paid API path is deliberately opt-in. The frozen current-prompt receipts
from twinkl-752.1 are reused; only the aligned prompt requires new calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from prompts import load_prompt
from scripts.experiments import weekly_verifier_ablation as baseline

DEFAULT_CONFIG_PATH = Path(
    "config/evals/twinkl_752_3_weekly_drift_reviewer_prompt_alignment_v1.yaml"
)
ALIGNED_SETUP = "aligned_prompt"
CURRENT_SETUP = "current_prompt"


def _value_definitions(config: dict[str, Any], root: Path) -> dict[str, dict[str, str]]:
    path = baseline._rooted(config["study"]["value_definitions_path"], root)
    payload = baseline._read_yaml(path)
    definitions = {}
    for display_name, fields in payload["values"].items():
        definitions[baseline._normalize_value(display_name)] = {
            "definition": str(fields["definition"]).strip(),
            "core_motivation": str(fields["core_motivation"]).strip(),
        }
    return definitions


def _format_conflict_rubric(rubric: dict[str, Any]) -> str:
    entry = rubric["entry_decision"]
    drift = rubric["drift_decision"]
    lines = [
        f"Rubric version: {rubric['rubric_id']}",
        f"Conflict: {entry['conflict']}",
        f"Not Conflict: {entry['not_conflict']}",
        f"Abstain: {entry['abstain']}",
        "Exclusions:",
        *[f"- {item}" for item in entry["exclusions"]],
        "Clarifications:",
        *[f"- {item}" for item in entry["clarifications"]],
        f"Drift: {drift['definition']}",
        f"Pair yes: {drift['yes']}",
        f"Pair no: {drift['no']}",
        f"Pair abstain: {drift['abstain']}",
    ]
    return "\n".join(lines)


def _format_declared_values(
    values: list[str], definitions: dict[str, dict[str, str]]
) -> str:
    blocks = []
    for value in values:
        fields = definitions[value]
        blocks.append(
            f"[{value}]\nDefinition: {fields['definition']}\n"
            f"Core motivation: {fields['core_motivation']}"
        )
    return "\n\n".join(blocks)


def _format_entries(entries: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[ENTRY t_index={entry['t_index']}]\n{entry['text']}" for entry in entries
    )


def _format_pairs(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    if not pairs:
        return (
            "No adjacent pair ends in this week. Return an empty pair_assessments list."
        )
    return "\n\n".join(
        (
            f"[PAIR t_index={first['t_index']} -> t_index={second['t_index']}]\n"
            f"FIRST JOURNAL ENTRY\n{first['text']}\n\n"
            f"SECOND JOURNAL ENTRY\n{second['text']}"
        )
        for first, second in pairs
    )


def build_prompt_records(
    config: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one aligned prompt for every frozen persona-week."""
    personas, cases = baseline._load_population(config, root)
    definitions = _value_definitions(config, root)
    rubric_path = baseline._rooted(config["study"]["conflict_rubric_path"], root)
    rubric = baseline._read_yaml(rubric_path)
    template = load_prompt("weekly_drift_reviewer_aligned")
    records = []
    runtime_contract = []

    for persona_id in sorted(personas):
        persona = personas[persona_id]
        entries = persona["entries"]
        by_index = {int(entry["t_index"]): entry for entry in entries}
        week_starts = sorted({baseline._week_start(entry["date"]) for entry in entries})
        for start in week_starts:
            end = start + timedelta(days=6)
            history = [
                entry for entry in entries if date.fromisoformat(entry["date"]) <= end
            ]
            current = [
                entry
                for entry in entries
                if start <= date.fromisoformat(entry["date"]) <= end
            ]
            current_indices = {int(entry["t_index"]) for entry in current}
            pairs = []
            for second in current:
                first = by_index.get(int(second["t_index"]) - 1)
                if first is not None:
                    pairs.append((first, second))
            assessment_indices = current_indices | {
                int(first["t_index"]) for first, _second in pairs
            }
            assessment_entries = [
                by_index[index] for index in sorted(assessment_indices)
            ]
            expected_coordinates = [
                {"t_index": int(entry["t_index"]), "dimension": value}
                for entry in assessment_entries
                for value in persona["values"]
            ]
            current_expected_coordinates = [
                {"t_index": int(entry["t_index"]), "dimension": value}
                for entry in current
                for value in persona["values"]
            ]
            expected_pairs = [
                {
                    "first_t_index": int(first["t_index"]),
                    "second_t_index": int(second["t_index"]),
                    "dimension": value,
                    "crosses_week_boundary": baseline._week_start(first["date"])
                    != baseline._week_start(second["date"]),
                }
                for first, second in pairs
                for value in persona["values"]
            ]
            runtime_payload = [
                {"t_index": entry["t_index"], "text": entry["text"]}
                for entry in history
            ]
            runtime_sha = baseline._sha256_text(
                baseline._canonical_json(runtime_payload)
            )
            prompt = (
                template.render(
                    conflict_rubric=_format_conflict_rubric(rubric),
                    declared_value_rubrics=_format_declared_values(
                        persona["values"], definitions
                    ),
                    cumulative_history=baseline._format_history(history),
                    entries_to_assess=_format_entries(assessment_entries),
                    candidate_pairs=_format_pairs(pairs),
                ).strip()
                + "\n"
            )
            record = {
                "persona_id": persona_id,
                "week_start": start.isoformat(),
                "week_end": end.isoformat(),
                "arm": ALIGNED_SETUP,
                "response_schema": "entry_pair",
                "rubric_version": rubric["rubric_id"],
                "declared_values": persona["values"],
                "current_t_indices": sorted(current_indices),
                "expected_coordinates": expected_coordinates,
                "current_expected_coordinates": current_expected_coordinates,
                "expected_pairs": expected_pairs,
                "entry_text_by_t_index": {
                    str(entry["t_index"]): entry["text"] for entry in assessment_entries
                },
                "runtime_text_sha256": runtime_sha,
                "prompt": prompt,
                "prompt_sha256": baseline._sha256_text(prompt),
            }
            records.append(record)
            runtime_contract.append(
                {
                    "persona_id": persona_id,
                    "week_start": start.isoformat(),
                    "runtime_text_sha256": runtime_sha,
                }
            )

    expected_weeks = int(config["population"]["expected_persona_weeks"])
    if len(records) != expected_weeks:
        raise ValueError(
            f"Expected {expected_weeks} persona-weeks, built {len(records)}"
        )
    source = config["study"]["current_prompt_source"]
    source_paths = {
        name: baseline._rooted(source[name], root)
        for name in (
            "config_path",
            "prompts_path",
            "responses_path",
            "metrics_path",
        )
    }
    manifest = {
        "study_id": config["study_id"],
        "created_at": datetime.now(UTC).isoformat(),
        "repo_head": baseline._git_head(root),
        "target_split": config["population"]["split"],
        "persona_count": len(personas),
        "case_count": len(cases),
        "entry_count": sum(len(persona["entries"]) for persona in personas.values()),
        "persona_week_count": len(records),
        "prompt_count": len(records),
        "planned_successful_calls": len(records) * int(config["study"]["repeats"]),
        "setups": config["study"]["setups"],
        "repeats": int(config["study"]["repeats"]),
        "model": config["api"]["model"],
        "reasoning_effort": config["api"]["reasoning_effort"],
        "store": bool(config["api"]["store"]),
        "locked_final_test_used": False,
        "retired_benchmark_used": False,
        "config_payload_sha256": baseline._sha256_text(
            baseline._canonical_json(config)
        ),
        "packet_sha256": baseline._sha256_file(
            baseline._rooted(config["population"]["packet_path"], root)
        ),
        "reviewer_a_sha256": baseline._sha256_file(
            baseline._rooted(config["population"]["reviewer_a_path"], root)
        ),
        "reviewer_b_sha256": baseline._sha256_file(
            baseline._rooted(config["population"]["reviewer_b_path"], root)
        ),
        "reconciliation_key_sha256": baseline._sha256_file(
            baseline._rooted(config["population"]["reconciliation_key_path"], root)
        ),
        "rubric_sha256": baseline._sha256_file(rubric_path),
        "value_definitions_sha256": baseline._sha256_file(
            baseline._rooted(config["study"]["value_definitions_path"], root)
        ),
        "aligned_prompt_template_sha256": baseline._sha256_file(
            baseline._rooted(config["study"]["aligned_prompt"], root)
        ),
        "runner_sha256": baseline._sha256_file(Path(__file__)),
        "shared_validator_sha256": baseline._sha256_file(
            root / "scripts/experiments/weekly_verifier_ablation.py"
        ),
        "runtime_contract_sha256": baseline._sha256_text(
            baseline._canonical_json(runtime_contract)
        ),
        "prompt_manifest_sha256": baseline._sha256_text(
            baseline._canonical_json(
                [
                    {
                        "persona_id": record["persona_id"],
                        "week_start": record["week_start"],
                        "prompt_sha256": record["prompt_sha256"],
                    }
                    for record in records
                ]
            )
        ),
        "current_prompt_source": {
            name: {
                "path": str(path.relative_to(root)),
                "sha256": baseline._sha256_file(path),
            }
            for name, path in source_paths.items()
        },
    }
    return records, manifest


def _aligned_predictions(
    responses: list[dict[str, Any]], records: list[dict[str, Any]]
) -> tuple[
    dict[tuple[int, str, int, str], baseline.VerifierAssessment],
    dict[tuple[int, str, int, int, str], baseline.DriftPairAssessment],
]:
    current_indices = {
        (record["persona_id"], record["week_start"]): set(record["current_t_indices"])
        for record in records
    }
    entries = {}
    pairs = {}
    for response in responses:
        if response.get("status") != "ok":
            continue
        parsed = baseline.AlignedWeeklyVerifierResponse.model_validate(
            response["parsed"]
        )
        repeat = int(response["repeat"])
        persona_id = str(response["persona_id"])
        current = current_indices[(persona_id, str(response["week_start"]))]
        for assessment in parsed.assessments:
            if assessment.t_index not in current:
                continue
            key = (repeat, persona_id, assessment.t_index, assessment.dimension)
            if key in entries:
                raise ValueError(f"Duplicate current-entry prediction: {key}")
            entries[key] = assessment
        for pair in parsed.pair_assessments:
            key = (
                repeat,
                persona_id,
                pair.first_t_index,
                pair.second_t_index,
                pair.dimension,
            )
            if key in pairs:
                raise ValueError(f"Duplicate adjacent-pair prediction: {key}")
            pairs[key] = pair
    return entries, pairs


def _aligned_drift_metrics(
    *,
    cases: dict[str, dict[str, Any]],
    episode_targets: dict[str, bool],
    conflict_by_coordinate: dict[tuple[str, str, int], bool],
    pair_predictions: dict[
        tuple[int, str, int, int, str], baseline.DriftPairAssessment
    ],
    repeat: int,
) -> tuple[
    dict[str, float],
    dict[str, bool | None],
    list[dict[str, Any]],
    dict[str, Any],
]:
    from src.vif.drift_benchmark import match_episodes

    reference_rows = []
    predicted_rows = []
    reference_case = {}
    case_predictions = {}
    for review_case_id, case in cases.items():
        case_with_id = {**case, "review_case_id": review_case_id}
        target_labels = [
            conflict_by_coordinate.get(
                (case["persona_id"], case["dimension"], int(entry["t_index"]))
            )
            for entry in case["entries"]
        ]
        reference = baseline._episode_rows(
            case=case_with_id, labels=target_labels, source="reference"
        )
        reference_rows.extend(reference)
        reference_case.update({row["episode_id"]: review_case_id for row in reference})

        yes_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        decisions = []
        for first, second in zip(case["entries"], case["entries"][1:], strict=False):
            if int(second["t_index"]) != int(first["t_index"]) + 1:
                continue
            prediction = pair_predictions.get(
                (
                    repeat,
                    case["persona_id"],
                    int(first["t_index"]),
                    int(second["t_index"]),
                    case["dimension"],
                )
            )
            decision = prediction.sustained_conflict if prediction else "abstain"
            decisions.append(decision)
            if decision == "yes":
                yes_pairs.append((first, second))
        pair_rows = _collapse_yes_pairs(
            yes_pairs=yes_pairs,
            case=case,
            review_case_id=review_case_id,
            repeat=repeat,
        )
        predicted_rows.extend(pair_rows)
        if pair_rows:
            case_predictions[review_case_id] = True
        elif decisions and all(decision == "no" for decision in decisions):
            case_predictions[review_case_id] = False
        else:
            case_predictions[review_case_id] = None

    if len(reference_rows) != sum(episode_targets.values()):
        raise ValueError("Reference Drift extraction changed from the reviewed target")
    matches = match_episodes(
        baseline._episode_frame(reference_rows),
        baseline._episode_frame(predicted_rows),
        max_confirmation_lag=2,
    )
    matched_reference = {
        str(row["reference_episode_id"]): row for row in matches.to_dicts()
    }
    timing = []
    slice_counts = {
        "same_week": {"reference": 0, "detected": 0},
        "cross_week": {"reference": 0, "detected": 0},
    }
    for reference in reference_rows:
        case = cases[reference_case[reference["episode_id"]]]
        by_index = {int(entry["t_index"]): entry for entry in case["entries"]}
        first = by_index[int(reference["onset_t_index"])]
        second = by_index[int(reference["confirmation_t_index"])]
        slice_name = (
            "same_week"
            if baseline._week_start(first["date"])
            == baseline._week_start(second["date"])
            else "cross_week"
        )
        slice_counts[slice_name]["reference"] += 1
        match = matched_reference.get(reference["episode_id"])
        if match:
            slice_counts[slice_name]["detected"] += 1
        timing.append(
            {
                "review_case_id": reference_case[reference["episode_id"]],
                "slice": slice_name,
                "target_confirmation_t_index": reference["confirmation_t_index"],
                "predicted_confirmation_t_index": (
                    int(match["predicted_confirmation_t_index"]) if match else None
                ),
            }
        )
    tp = matches.height
    fp = len(predicted_rows) - tp
    covered = sum(value is not None for value in case_predictions.values())
    metrics = {
        "recall": tp / len(reference_rows) if reference_rows else 0.0,
        "precision": tp / len(predicted_rows) if predicted_rows else 0.0,
        "coverage": covered / len(case_predictions),
        "abstention_rate": 1.0 - covered / len(case_predictions),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(len(reference_rows) - tp),
        "n": float(len(case_predictions)),
    }
    return metrics, case_predictions, timing, slice_counts


def _collapse_yes_pairs(
    *,
    yes_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    case: dict[str, Any],
    review_case_id: str,
    repeat: int,
) -> list[dict[str, Any]]:
    episodes = []
    run: list[dict[str, Any]] = []

    def finish_run() -> None:
        if len(run) < 2:
            return
        episodes.append(
            {
                "episode_id": (
                    f"{ALIGNED_SETUP}:{repeat}:{review_case_id}:"
                    f"{run[0]['t_index']}:{run[-1]['t_index']}"
                ),
                "persona_id": case["persona_id"],
                "dimension": case["dimension"],
                "onset_t_index": int(run[0]["t_index"]),
                "confirmation_t_index": int(run[1]["t_index"]),
                "end_t_index": int(run[-1]["t_index"]),
                "delivery_state": "active",
            }
        )

    for first, second in yes_pairs:
        if not run:
            run = [first, second]
        elif int(first["t_index"]) == int(run[-1]["t_index"]):
            run.append(second)
        else:
            finish_run()
            run = [first, second]
    finish_run()
    return episodes


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _repeat_stability(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[bool | None]] = {}
    for row in results:
        for review_case_id, prediction in row["case_predictions"].items():
            by_case.setdefault(review_case_id, []).append(prediction)
    complete = {key: values for key, values in by_case.items() if len(values) == 3}
    stable = sum(len(set(values)) == 1 for values in complete.values())
    return {
        "case_count": len(complete),
        "stable_case_count": stable,
        "stable_case_fraction": stable / len(complete) if complete else 0.0,
        "cases_with_any_abstention": sum(
            None in values for values in complete.values()
        ),
        "cases_with_any_drift_claim": sum(
            True in values for values in complete.values()
        ),
    }


def _current_prompt_predictions(
    config: dict[str, Any], root: Path
) -> dict[tuple[int, str, int, str], baseline.VerifierAssessment]:
    source = config["study"]["current_prompt_source"]
    records = [
        row
        for row in baseline._load_jsonl(baseline._rooted(source["prompts_path"], root))
        if row["arm"] == source["source_setup"]
    ]
    responses = [
        row
        for row in baseline._load_jsonl(
            baseline._rooted(source["responses_path"], root)
        )
        if row["arm"] == source["source_setup"]
    ]
    record_map = {
        (record["persona_id"], record["week_start"], record["arm"]): record
        for record in records
    }
    effective, _recovered = baseline._recover_prompt_contract_valid_responses(
        responses, record_map
    )
    flattened = baseline._flatten_predictions(effective)
    return {
        (repeat, persona_id, t_index, dimension): assessment
        for (
            arm,
            repeat,
            persona_id,
            t_index,
            dimension,
        ), assessment in flattened.items()
        if arm == source["source_setup"]
    }


def _matched_coverage_summary(
    *,
    cells: list[baseline.TargetCell],
    current: dict[tuple[int, str, int, str], baseline.VerifierAssessment],
    aligned: dict[tuple[int, str, int, str], baseline.VerifierAssessment],
    repeats: int,
) -> list[dict[str, Any]]:
    rows = []
    for repeat in range(1, repeats + 1):
        matched = []
        for cell in cells:
            key = (repeat, cell.persona_id, cell.t_index, cell.dimension)
            current_assessment = current.get(key)
            aligned_assessment = aligned.get(key)
            if (
                current_assessment is None
                or current_assessment.verdict == "abstain"
                or aligned_assessment is None
                or aligned_assessment.verdict == "abstain"
            ):
                continue
            matched.append((cell, current_assessment, aligned_assessment))
        negatives = [item for item in matched if item[0].conflict]
        rows.append(
            {
                "repeat": repeat,
                "matched_cell_count": len(matched),
                "matched_coverage": len(matched) / len(cells),
                "negative_support": len(negatives),
                "current_prompt_recall": (
                    sum(item[1].verdict == "conflict" for item in negatives)
                    / len(negatives)
                    if negatives
                    else 0.0
                ),
                "aligned_prompt_recall": (
                    sum(item[2].verdict == "conflict" for item in negatives)
                    / len(negatives)
                    if negatives
                    else 0.0
                ),
            }
        )
    return rows


def _decision(
    current_results: list[dict[str, Any]], aligned_results: list[dict[str, Any]]
) -> dict[str, Any]:
    current_recall = _median([row["episode"]["recall"] for row in current_results])
    aligned_recall = _median([row["episode"]["recall"] for row in aligned_results])
    current_fp = _median([row["episode"]["fp"] for row in current_results])
    aligned_fp = _median([row["episode"]["fp"] for row in aligned_results])
    current_coverage = _median([row["episode"]["coverage"] for row in current_results])
    aligned_coverage = _median([row["episode"]["coverage"] for row in aligned_results])
    cross_week_repeats = sum(
        row["drift_slices"]["cross_week"]["detected"] > 0 for row in aligned_results
    )
    if (
        aligned_recall - current_recall >= 0.20 - 1e-12
        and cross_week_repeats >= 2
        and aligned_fp <= current_fp
        and aligned_coverage >= current_coverage - 0.05
    ):
        verdict = "prompt_limited"
    elif aligned_recall <= current_recall and cross_week_repeats == 0:
        verdict = "unchanged"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "current_prompt": {
            "median_drift_recall": current_recall,
            "median_false_drift_alerts": current_fp,
            "median_coverage": current_coverage,
        },
        "aligned_prompt": {
            "median_drift_recall": aligned_recall,
            "median_false_drift_alerts": aligned_fp,
            "median_coverage": aligned_coverage,
            "cross_week_reference_drift_recovered_repeats": cross_week_repeats,
        },
        "limitations": (
            "Development-only prompt evidence at reasoning effort none; reviewer/model "
            "and future-context differences remain, so this is not an intrinsic LLM "
            "ceiling or VIF architecture decision."
        ),
    }


def score_responses(
    *,
    config: dict[str, Any],
    root: Path,
    records: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    personas, cases = baseline._load_population(config, root)
    del personas
    record_map = {
        (record["persona_id"], record["week_start"], record["arm"]): record
        for record in records
    }
    baseline._completed_keys(
        responses,
        record_map,
        repeats=int(config["study"]["repeats"]),
        requested_model=str(config["api"]["model"]),
    )
    cells, episode_targets = baseline._load_targets(config, root, cases)
    unresolved = set(config["population"]["unresolved_case_ids"])
    scored_cases = {
        review_case_id: case
        for review_case_id, case in cases.items()
        if review_case_id not in unresolved
    }
    conflict_by_coordinate = {
        (cell.persona_id, cell.dimension, cell.t_index): cell.conflict for cell in cells
    }
    entry_predictions, pair_predictions = _aligned_predictions(responses, records)
    current_entry_predictions = _current_prompt_predictions(config, root)
    aligned_results = []
    for repeat in range(1, int(config["study"]["repeats"]) + 1):
        assessments = [
            entry_predictions.get(
                (repeat, cell.persona_id, cell.t_index, cell.dimension)
            )
            for cell in cells
        ]
        entry_metrics = baseline._entry_metric_bundle(
            cells, baseline._confidence_predictions(assessments, "low")
        )
        episode_metrics, case_predictions, timing, slices = _aligned_drift_metrics(
            cases=scored_cases,
            episode_targets=episode_targets,
            conflict_by_coordinate=conflict_by_coordinate,
            pair_predictions=pair_predictions,
            repeat=repeat,
        )
        aligned_results.append(
            {
                "setup": ALIGNED_SETUP,
                "repeat": repeat,
                "entry": entry_metrics,
                "episode": episode_metrics,
                "case_predictions": case_predictions,
                "detection_timing": timing,
                "drift_slices": slices,
            }
        )

    source = config["study"]["current_prompt_source"]
    current_metrics_path = baseline._rooted(source["metrics_path"], root)
    saved_current = baseline._read_json(current_metrics_path)
    current_results = [
        {**row, "setup": CURRENT_SETUP}
        for row in saved_current["results"]
        if row["arm"] == source["source_setup"]
    ]
    for row in current_results:
        row.pop("arm", None)
        row["drift_slices"] = {
            "same_week": {"reference": 3, "detected": 0},
            "cross_week": {"reference": 2, "detected": 0},
        }
        for timing in row["detection_timing"]:
            timing["slice"] = (
                "cross_week"
                if timing["review_case_id"] in {"case_022", "case_025"}
                else "same_week"
            )
            if timing["predicted_confirmation_t_index"] is not None:
                row["drift_slices"][timing["slice"]]["detected"] += 1

    reason_counts: dict[str, int] = {}
    pair_abstentions = 0
    for response in responses:
        if response.get("status") != "ok":
            continue
        parsed = baseline.AlignedWeeklyVerifierResponse.model_validate(
            response["parsed"]
        )
        for assessment in parsed.assessments:
            if assessment.verdict == "abstain":
                reason_counts[assessment.reason_code] = (
                    reason_counts.get(assessment.reason_code, 0) + 1
                )
        pair_abstentions += sum(
            pair.sustained_conflict == "abstain" for pair in parsed.pair_assessments
        )

    return {
        "study_id": config["study_id"],
        "scored_at": datetime.now(UTC).isoformat(),
        "resolved_entry_cells": len(cells),
        "resolved_trajectories": len(episode_targets),
        "positive_drifts": sum(episode_targets.values()),
        "results": current_results + aligned_results,
        "response_summary": baseline._response_summary(responses, config),
        "aligned_abstentions": {
            "entry_reason_counts": reason_counts,
            "pair_count": pair_abstentions,
        },
        "matched_coverage_entry_recall": _matched_coverage_summary(
            cells=cells,
            current=current_entry_predictions,
            aligned=entry_predictions,
            repeats=int(config["study"]["repeats"]),
        ),
        "repeat_stability": {
            CURRENT_SETUP: _repeat_stability(current_results),
            ALIGNED_SETUP: _repeat_stability(aligned_results),
        },
        "decision": _decision(current_results, aligned_results),
        "current_prompt_provenance": {
            "metrics_path": source["metrics_path"],
            "metrics_sha256": baseline._sha256_file(current_metrics_path),
            "note": "Frozen three-repeat no-Critic receipts from twinkl-752.1.",
        },
    }


def _artifact_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    artifacts = config["artifacts"]
    output_dir = baseline._rooted(artifacts["output_dir"], root)
    return {
        "output_dir": output_dir,
        "prompts": output_dir / artifacts["prompts_filename"],
        "manifest": output_dir / artifacts["manifest_filename"],
        "responses": output_dir / artifacts["responses_filename"],
        "metrics": output_dir / artifacts["metrics_filename"],
    }


def command_prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    records, manifest = build_prompt_records(config, root)
    paths = _artifact_paths(config, root)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    baseline._write_jsonl(paths["prompts"], records)
    estimate = baseline.estimate_plan(records, config)
    manifest["estimate"] = estimate
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"paths": {k: str(v) for k, v in paths.items()}, **estimate}, indent=2
        )
    )


def _load_prepared(
    config: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    paths = _artifact_paths(config, root)
    records = baseline._load_jsonl(paths["prompts"])
    if not records or not paths["manifest"].exists():
        raise FileNotFoundError("Run prepare before this command")
    rebuilt, rebuilt_manifest = build_prompt_records(config, root)
    if baseline._canonical_json(records) != baseline._canonical_json(rebuilt):
        raise ValueError("Prepared prompts no longer match the registered sources")
    manifest = baseline._read_json(paths["manifest"])
    for key in (
        "config_payload_sha256",
        "packet_sha256",
        "reviewer_a_sha256",
        "reviewer_b_sha256",
        "reconciliation_key_sha256",
        "rubric_sha256",
        "value_definitions_sha256",
        "aligned_prompt_template_sha256",
        "runner_sha256",
        "shared_validator_sha256",
        "runtime_contract_sha256",
        "prompt_manifest_sha256",
        "current_prompt_source",
    ):
        if manifest.get(key) != rebuilt_manifest.get(key):
            raise ValueError(f"Prepared manifest source mismatch: {key}")
    return records, paths


def command_estimate(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    records, _paths = _load_prepared(config, root)
    print(json.dumps(baseline.estimate_plan(records, config), indent=2))


def command_run(args: argparse.Namespace) -> None:
    if not args.execute:
        raise SystemExit("Refusing paid calls without --execute")
    from dotenv import load_dotenv

    root = Path(args.root).resolve()
    load_dotenv(root / ".env")
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    records, paths = _load_prepared(config, root)
    result = asyncio.run(
        baseline.execute_calls(
            records=records,
            config=config,
            output_path=paths["responses"],
        )
    )
    print(json.dumps(result, indent=2))


def command_score(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    records, paths = _load_prepared(config, root)
    responses = baseline._load_jsonl(paths["responses"])
    metrics = score_responses(
        config=config, root=root, records=records, responses=responses
    )
    metrics["artifact_provenance"] = {
        "manifest_sha256": baseline._sha256_file(paths["manifest"]),
        "prompts_sha256": baseline._sha256_file(paths["prompts"]),
        "responses_sha256": baseline._sha256_file(paths["responses"]),
    }
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["decision"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
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
