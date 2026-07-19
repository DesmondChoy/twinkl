#!/usr/bin/env python3
"""Compare two Weekly Drift Reviewer models on complete development data.

``prepare`` and ``estimate`` are offline. ``run`` requires ``--execute`` and
uses separate resumable response files for each model.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import statistics
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from prompts import load_prompt
from scripts.experiments import reassess_twinkl_752_5 as reassess
from scripts.experiments import weekly_verifier_ablation as baseline
from src.vif.state_encoder import concatenate_entry_text
from src.wrangling.parse_wrangled_data import parse_wrangled_file

DEFAULT_CONFIG_PATH = Path("config/evals/twinkl_52zz_model_comparison_v1.yaml")
WEEKLY_WITHOUT = reassess.WEEKLY_WITHOUT


def _artifact_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    spec = config["artifacts"]
    output_dir = baseline._rooted(spec["output_dir"], root)
    return {
        "output_dir": output_dir,
        "prompts": output_dir / spec["prompts_filename"],
        "manifest": output_dir / spec["manifest_filename"],
        "metrics": output_dir / spec["metrics_filename"],
        "responses": {
            key: output_dir / filename
            for key, filename in spec["response_filenames"].items()
        },
    }


def _source_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    return {
        key: baseline._rooted(value, root)
        for key, value in config["sources"].items()
        if key.endswith("_path")
    }


def _model_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {str(item["key"]): item for item in config["models"]}
    if len(specs) != len(config["models"]):
        raise ValueError("Model keys must be unique")
    return specs


def _model_runtime_config(
    config: dict[str, Any], model_spec: dict[str, Any]
) -> dict[str, Any]:
    runtime = copy.deepcopy(config)
    runtime["api"] = {
        **config["api"],
        "model": model_spec["model"],
        "reasoning_effort": model_spec["reasoning_effort"],
        "max_budget_usd": model_spec["max_budget_usd"],
        "pricing_usd_per_million_tokens": model_spec[
            "pricing_usd_per_million_tokens"
        ],
    }
    return runtime


def _load_complete_development(
    config: dict[str, Any], root: Path
) -> tuple[
    list[dict[str, Any]], pl.DataFrame, pl.DataFrame, pl.DataFrame
]:
    paths = _source_paths(config, root)
    targets = pl.read_parquet(paths["complete_entry_target_path"]).sort(
        "canonical_case_id", "position"
    )
    outcomes = pl.read_parquet(paths["complete_case_outcomes_path"]).sort(
        "canonical_case_id"
    )
    episodes = pl.read_parquet(paths["complete_drift_episodes_path"]).sort(
        "canonical_case_id", "onset_position"
    )
    wrangled_dir = baseline._rooted(config["sources"]["wrangled_dir"], root)
    targets_by_case = {
        str(key[0] if isinstance(key, tuple) else key): frame.sort(
            "position"
        ).to_dicts()
        for key, frame in targets.partition_by(
            "canonical_case_id", as_dict=True
        ).items()
    }
    cases: list[dict[str, Any]] = []
    for outcome in outcomes.to_dicts():
        case_id = str(outcome["canonical_case_id"])
        persona_id = str(outcome["persona_id"])
        dimension = baseline._normalize_value(str(outcome["dimension"]))
        target_rows = targets_by_case.get(case_id)
        if not target_rows:
            raise ValueError(f"Missing complete targets for {case_id}")
        profile, wrangled_entries, _warnings = parse_wrangled_file(
            wrangled_dir / f"persona_{persona_id}.md"
        )
        wrangled_by_index = {
            int(entry["t_index"]): entry for entry in wrangled_entries
        }
        full_core_values = [
            baseline._normalize_value(value)
            for value in profile.get("core_values") or []
        ]
        if dimension not in full_core_values:
            raise ValueError(f"Core Value mismatch for {case_id}")
        entries = []
        for expected_position, target in enumerate(target_rows, start=1):
            if int(target["position"]) != expected_position:
                raise ValueError(f"Non-contiguous target positions for {case_id}")
            t_index = int(target["t_index"])
            entry = wrangled_by_index.get(t_index)
            if entry is None:
                raise ValueError(f"Missing Journal Entry {case_id}:{t_index}")
            text = concatenate_entry_text(
                entry.get("initial_entry"),
                entry.get("nudge_text"),
                entry.get("response_text"),
            ).strip()
            if (
                str(entry["date"]) != str(target["date"])
                or baseline._sha256_text(text)
                != str(target["runtime_text_sha256"])
            ):
                raise ValueError(f"Runtime text mismatch for {case_id}:{t_index}")
            entries.append(
                {
                    "position": expected_position,
                    "t_index": t_index,
                    "date": str(entry["date"]),
                    "initial_entry": entry.get("initial_entry"),
                    "nudge_text": entry.get("nudge_text"),
                    "response_text": entry.get("response_text"),
                    "text": text,
                }
            )
        if len(entries) != int(outcome["entry_count"]):
            raise ValueError(f"Entry count mismatch for {case_id}")
        cases.append(
            {
                "canonical_case_id": case_id,
                "persona_id": persona_id,
                "dimension": dimension,
                "full_core_values": full_core_values,
                "historical_split": str(outcome["historical_split"]),
                "cohort_source": str(outcome["cohort_source"]),
                "entries": entries,
            }
        )
    return cases, outcomes, targets, episodes


def _build_prompt_records(
    config: dict[str, Any], root: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    cases, outcomes, targets, episodes = _load_complete_development(config, root)
    personas = reassess._persona_records(cases)
    template = load_prompt(str(config["study"]["prompt_name"]))
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
            prompt = (
                template.render(
                    declared_values="\n".join(
                        f"- {value}" for value in persona["reviewed_values"]
                    ),
                    cumulative_history=baseline._format_history(history),
                    current_week_entries=baseline._format_current_entries(current),
                ).strip()
                + "\n"
            )
            runtime_text_sha256 = baseline._sha256_text(
                baseline._canonical_json(
                    [
                        {"t_index": int(row["t_index"]), "text": row["text"]}
                        for row in history
                    ]
                )
            )
            records.append(
                {
                    "review_event_id": (
                        f"{WEEKLY_WITHOUT}:{persona_id}:{start.isoformat()}"
                    ),
                    "persona_id": persona_id,
                    "week_start": start.isoformat(),
                    "week_end": end.isoformat(),
                    "review_at_date": end.isoformat(),
                    "cutoff_t_index": max(int(row["t_index"]) for row in current),
                    "arm": WEEKLY_WITHOUT,
                    "response_schema": "entry_only",
                    "declared_values": persona["reviewed_values"],
                    "current_t_indices": [
                        int(row["t_index"]) for row in current
                    ],
                    "expected_coordinates": [
                        {"t_index": int(row["t_index"]), "dimension": value}
                        for row in current
                        for value in persona["reviewed_values"]
                    ],
                    "entry_text_by_t_index": {
                        str(row["t_index"]): row["text"] for row in history
                    },
                    "runtime_text_sha256": runtime_text_sha256,
                    "prompt": prompt,
                    "prompt_sha256": baseline._sha256_text(prompt),
                }
            )
    return records, cases, outcomes, targets, episodes


def _observed_counts(
    *,
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    outcomes: pl.DataFrame,
    targets: pl.DataFrame,
    episodes: pl.DataFrame,
) -> dict[str, int]:
    personas = reassess._persona_records(cases)
    return {
        "personas": len(personas),
        "cases": len(cases),
        "entries": sum(len(persona["entries"]) for persona in personas.values()),
        "entry_value_cells": targets.height,
        "resolved_entry_value_cells": targets.filter(
            pl.col("final_conflict").is_not_null()
        ).height,
        "drifts": episodes.height,
        "drift_trajectories": outcomes.filter(pl.col("has_drift")).height,
        "persona_weeks": len(records),
    }


def _validate_counts(config: dict[str, Any], counts: dict[str, int]) -> None:
    expected = {
        key.removeprefix("expected_"): int(value)
        for key, value in config["study"].items()
        if key.startswith("expected_")
    }
    if counts != expected:
        raise ValueError(f"Complete development count mismatch: {counts} != {expected}")


def _prepare(config: dict[str, Any], root: Path) -> dict[str, Any]:
    paths = _artifact_paths(config, root)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    records, cases, outcomes, targets, episodes = _build_prompt_records(config, root)
    counts = _observed_counts(
        records=records,
        cases=cases,
        outcomes=outcomes,
        targets=targets,
        episodes=episodes,
    )
    _validate_counts(config, counts)
    baseline._write_jsonl(paths["prompts"], records)
    source_paths = _source_paths(config, root)
    model_estimates = {
        key: baseline.estimate_plan(
            records, _model_runtime_config(config, model_spec)
        )
        for key, model_spec in _model_specs(config).items()
    }
    total_estimated_cost = sum(
        float(estimate["estimated_cost_usd"])
        for estimate in model_estimates.values()
    )
    if total_estimated_cost > float(config["api"]["total_max_budget_usd"]):
        raise ValueError(
            f"Estimated cost ${total_estimated_cost:.2f} exceeds total budget"
        )
    manifest = {
        "study_id": config["study_id"],
        "schema_version": "twinkl-52zz-model-comparison-v1",
        "prepared_at": datetime.now(UTC).isoformat(),
        "repo_head": baseline._git_head(root),
        "counts": counts,
        "setup": WEEKLY_WITHOUT,
        "repeats": int(config["study"]["repeats"]),
        "models": config["models"],
        "planned_calls": len(records)
        * int(config["study"]["repeats"])
        * len(config["models"]),
        "model_estimates": model_estimates,
        "total_estimated_cost_usd": total_estimated_cost,
        "total_max_budget_usd": float(config["api"]["total_max_budget_usd"]),
        "config_sha256": baseline._sha256_text(baseline._canonical_json(config)),
        "prompt_template_sha256": baseline._sha256_file(
            baseline._rooted(config["study"]["prompt_path"], root)
        ),
        "prompts_sha256": baseline._sha256_file(paths["prompts"]),
        "runner_sha256": baseline._sha256_file(Path(__file__)),
        "shared_api_runner_sha256": baseline._sha256_file(
            root / "scripts/experiments/weekly_verifier_ablation.py"
        ),
        "shared_scorer_sha256": baseline._sha256_file(
            root / "scripts/experiments/reassess_twinkl_752_5.py"
        ),
        "source_sha256": {
            key: baseline._sha256_file(path) for key, path in source_paths.items()
        },
        "prompt_contract": {
            "model_in_prompt": False,
            "vif_critic_input": False,
            "fresh_final_test_inspected": False,
            "development_labels_in_prompt": False,
            "store": bool(config["api"]["store"]),
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _load_prepared(
    config: dict[str, Any], root: Path
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    paths = _artifact_paths(config, root)
    if not paths["manifest"].exists() or not paths["prompts"].exists():
        raise FileNotFoundError("Run prepare before this command")
    manifest = baseline._read_json(paths["manifest"])
    records = baseline._load_jsonl(paths["prompts"])
    rebuilt, cases, outcomes, targets, episodes = _build_prompt_records(config, root)
    if baseline._canonical_json(records) != baseline._canonical_json(rebuilt):
        raise ValueError("Prepared prompts no longer match complete development data")
    expected_hashes = {
        "config_sha256": baseline._sha256_text(baseline._canonical_json(config)),
        "prompt_template_sha256": baseline._sha256_file(
            baseline._rooted(config["study"]["prompt_path"], root)
        ),
        "prompts_sha256": baseline._sha256_file(paths["prompts"]),
        "runner_sha256": baseline._sha256_file(Path(__file__)),
        "shared_api_runner_sha256": baseline._sha256_file(
            root / "scripts/experiments/weekly_verifier_ablation.py"
        ),
        "shared_scorer_sha256": baseline._sha256_file(
            root / "scripts/experiments/reassess_twinkl_752_5.py"
        ),
    }
    for key, digest in expected_hashes.items():
        if manifest.get(key) != digest:
            raise ValueError(f"Prepared protocol hash mismatch: {key}")
    for key, path in _source_paths(config, root).items():
        if manifest["source_sha256"].get(key) != baseline._sha256_file(path):
            raise ValueError(f"Complete development source changed: {key}")
    return records, cases, outcomes, targets, episodes, paths, manifest


def _latency_summary(responses: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["latency_seconds"])
        for row in responses
        if row.get("latency_seconds") is not None
    ]
    return {
        "count": len(values),
        "median_seconds": float(statistics.median(values)) if values else None,
        "p95_seconds": float(np.quantile(values, 0.95)) if values else None,
    }


def _entry_cells(targets: pl.DataFrame) -> list[baseline.TargetCell]:
    return [
        baseline.TargetCell(
            persona_id=str(row["persona_id"]),
            dimension=str(row["dimension"]),
            t_index=int(row["t_index"]),
            conflict=bool(row["final_conflict"]),
        )
        for row in targets.filter(pl.col("final_conflict").is_not_null()).to_dicts()
    ]


def _stability_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for metric in (
        "drift_recall",
        "drift_precision",
        "false_drift_alerts",
        "coverage",
        "abstention_rate",
        "median_detection_delay_days",
    ):
        values = [
            float(row[metric]) for row in results if row.get(metric) is not None
        ]
        summary[metric] = {
            "median": float(statistics.median(values)) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def _score_model(
    *,
    config: dict[str, Any],
    model_spec: dict[str, Any],
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    outcomes: pl.DataFrame,
    targets: pl.DataFrame,
    episodes: pl.DataFrame,
    responses: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, dict[str, Any]]]]:
    runtime = _model_runtime_config(config, model_spec)
    predictions = reassess._assessment_map(
        responses,
        records,
        repeats=int(config["study"]["repeats"]),
        requested_model=str(model_spec["model"]),
    )
    reference_rows = reassess._reference_rows(episodes)
    all_case_ids = sorted(case["canonical_case_id"] for case in cases)
    subgroup_case_ids = {"primary": set(all_case_ids)}
    outcome_rows = outcomes.to_dicts()
    for split in sorted({str(row["historical_split"]) for row in outcome_rows}):
        subgroup_case_ids[f"historical_split:{split}"] = {
            str(row["canonical_case_id"])
            for row in outcome_rows
            if str(row["historical_split"]) == split
        }
    subgroup_case_ids["non_training"] = {
        str(row["canonical_case_id"])
        for row in outcome_rows
        if str(row["historical_split"]) != "training"
    }
    cells = _entry_cells(targets)
    results = []
    subgroups: dict[str, list[dict[str, Any]]] = {
        key: [] for key in subgroup_case_ids
    }
    case_stats: dict[int, dict[str, dict[str, Any]]] = {}
    for repeat in range(1, int(config["study"]["repeats"]) + 1):
        predicted_rows, covered = reassess._setup_predictions(
            cases=cases,
            records=records,
            predictions=predictions,
            setup=WEEKLY_WITHOUT,
            repeat=repeat,
        )
        primary, primary_case_stats = reassess._score_subset(
            case_ids=subgroup_case_ids["primary"],
            reference_rows=reference_rows,
            predicted_rows=predicted_rows,
            covered=covered,
            max_confirmation_lag=int(
                config["study"]["max_confirmation_lag_entries"]
            ),
        )
        assessments = [
            predictions.get(
                (
                    WEEKLY_WITHOUT,
                    repeat,
                    cell.persona_id,
                    cell.t_index,
                    cell.dimension,
                )
            )
            for cell in cells
        ]
        entry_predictions = baseline._confidence_predictions(assessments, "low")
        results.append(
            {
                "repeat": repeat,
                **primary,
                "entry": baseline._entry_metric_bundle(cells, entry_predictions),
            }
        )
        case_stats[repeat] = primary_case_stats
        for subgroup, case_ids in subgroup_case_ids.items():
            metrics, _ = reassess._score_subset(
                case_ids=case_ids,
                reference_rows=reference_rows,
                predicted_rows=predicted_rows,
                covered=covered,
                max_confirmation_lag=int(
                    config["study"]["max_confirmation_lag_entries"]
                ),
            )
            subgroups[subgroup].append({"repeat": repeat, **metrics})
    return (
        {
            "model": model_spec["model"],
            "reasoning_effort": model_spec["reasoning_effort"],
            "resolved_models": sorted(
                {
                    str(row["resolved_model"])
                    for row in responses
                    if row.get("resolved_model")
                }
            ),
            "results": results,
            "stability": _stability_summary(results),
            "subgroups": subgroups,
            "response_summary": baseline._response_summary(responses, runtime),
            "latency": _latency_summary(responses),
        },
        case_stats,
    )


def _decision(comparison: dict[str, Any]) -> str:
    deltas = comparison["deltas"]
    recall = deltas["recall"]
    interval = recall["interval"]
    if (
        recall["observed_median"] is not None
        and recall["observed_median"] > 0
        and interval is not None
        and interval[0] > 0
        and deltas["false_alerts"]["observed_median"] <= 0
        and deltas["coverage"]["observed_median"] >= -0.05
    ):
        return "select_gpt_5_6_luna"
    if (
        recall["observed_median"] is not None
        and recall["observed_median"] < 0
        and interval is not None
        and interval[1] < 0
    ):
        return "keep_gpt_5_4_mini"
    return "inconclusive_keep_gpt_5_4_mini_baseline"


def command_prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    manifest = _prepare(config, root)
    print(
        json.dumps(
            {
                "counts": manifest["counts"],
                "planned_calls": manifest["planned_calls"],
                "model_estimates": manifest["model_estimates"],
                "total_estimated_cost_usd": manifest[
                    "total_estimated_cost_usd"
                ],
                "total_max_budget_usd": manifest["total_max_budget_usd"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_estimate(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    records, _cases, _outcomes, _targets, _episodes, _paths, manifest = (
        _load_prepared(config, root)
    )
    estimates = {
        key: baseline.estimate_plan(
            records, _model_runtime_config(config, model_spec)
        )
        for key, model_spec in _model_specs(config).items()
    }
    if estimates != manifest["model_estimates"]:
        raise ValueError("Prepared cost estimate changed")
    print(json.dumps(estimates, indent=2, sort_keys=True))


def command_run(args: argparse.Namespace) -> None:
    if not args.execute:
        raise SystemExit("Refusing paid calls without --execute")
    from dotenv import load_dotenv

    root = Path(args.root).resolve()
    load_dotenv(root / ".env")
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    records, _cases, _outcomes, _targets, _episodes, paths, _manifest = (
        _load_prepared(config, root)
    )
    specs = _model_specs(config)
    selected = list(specs) if args.model_key == "all" else [args.model_key]
    unknown = [key for key in selected if key not in specs]
    if unknown:
        raise ValueError(f"Unknown model key: {unknown}")
    results = {}
    for key in selected:
        runtime = _model_runtime_config(config, specs[key])
        results[key] = asyncio.run(
            baseline.execute_calls(
                records=records,
                config=runtime,
                output_path=paths["responses"][key],
            )
        )
    print(json.dumps(results, indent=2, sort_keys=True))


def command_score(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    records, cases, outcomes, targets, episodes, paths, manifest = _load_prepared(
        config, root
    )
    model_results = {}
    case_stats = {}
    for key, model_spec in _model_specs(config).items():
        responses = baseline._load_jsonl(paths["responses"][key])
        model_results[key], case_stats[key] = _score_model(
            config=config,
            model_spec=model_spec,
            records=records,
            cases=cases,
            outcomes=outcomes,
            targets=targets,
            episodes=episodes,
            responses=responses,
        )
    comparison = reassess._comparison_bootstrap(
        case_ids=sorted(case["canonical_case_id"] for case in cases),
        first=case_stats["gpt_5_4_mini"],
        second=case_stats["gpt_5_6_luna"],
        config=config,
        seed_offset=1,
    )
    metrics = {
        "study_id": config["study_id"],
        "scored_at": datetime.now(UTC).isoformat(),
        "counts": manifest["counts"],
        "comparison_direction": "gpt_5_6_luna_minus_gpt_5_4_mini",
        "models": model_results,
        "paired_trajectory_bootstrap": comparison,
        "development_selection": _decision(comparison),
        "actual_api_spend_usd": sum(
            float(result["response_summary"]["actual_spend_usd"])
            for result in model_results.values()
        ),
        "restrictions": config["decision"]["restrictions"],
        "provenance": {
            "manifest_sha256": baseline._sha256_file(paths["manifest"]),
            "prompts_sha256": baseline._sha256_file(paths["prompts"]),
            "responses_sha256": {
                key: baseline._sha256_file(path)
                for key, path in paths["responses"].items()
            },
        },
    }
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "development_selection": metrics["development_selection"],
                "paired_trajectory_bootstrap": comparison,
                "actual_api_spend_usd": metrics["actual_api_spend_usd"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare").set_defaults(func=command_prepare)
    subparsers.add_parser("estimate").set_defaults(func=command_estimate)
    run = subparsers.add_parser("run")
    run.add_argument("--execute", action="store_true")
    run.add_argument(
        "--model-key", choices=("all", "gpt_5_4_mini", "gpt_5_6_luna"), default="all"
    )
    run.set_defaults(func=command_run)
    subparsers.add_parser("score").set_defaults(func=command_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
