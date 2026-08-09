#!/usr/bin/env python3
"""Compare Luna medium, high, and xhigh reasoning with Luna low."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from scripts.experiments import compare_twinkl_52zz_luna_reasoning as low_study
from scripts.experiments import compare_twinkl_52zz_models as model_study
from scripts.experiments import reassess_twinkl_752_5 as reassess
from scripts.experiments import weekly_verifier_ablation as baseline

DEFAULT_CONFIG_PATH = Path("config/evals/twinkl_ck3w_luna_higher_reasoning_v1.yaml")


def _model_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs = {str(item["reasoning_effort"]): item for item in config["models"]}
    if len(specs) != len(config["models"]):
        raise ValueError("Reasoning efforts must be unique")
    if set(specs) != {"medium", "high", "xhigh"}:
        raise ValueError("Expected medium, high, and xhigh reasoning efforts")
    if any(item["model"] != "gpt-5.6-luna" for item in specs.values()):
        raise ValueError("All experiment setups must use gpt-5.6-luna")
    return specs


def _efforts(value: str, specs: dict[str, dict[str, Any]]) -> list[str]:
    if value == "all":
        return ["medium", "high", "xhigh"]
    if value not in specs:
        raise ValueError(f"Unknown reasoning effort: {value}")
    return [value]


def _paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    output_dir = baseline._rooted(config["artifacts"]["output_dir"], root)
    return {
        "output_dir": output_dir,
        "manifest": output_dir / config["artifacts"]["manifest_filename"],
        "metrics": output_dir / config["artifacts"]["metrics_filename"],
        "smoke": {
            effort: output_dir
            / str(config["artifacts"]["smoke_responses_pattern"]).format(effort=effort)
            for effort in _model_specs(config)
        },
        "responses": {
            effort: output_dir
            / str(config["artifacts"]["responses_pattern"]).format(effort=effort)
            for effort in _model_specs(config)
        },
        "initial_smoke": {
            effort: output_dir
            / str(config["artifacts"]["initial_smoke_responses_pattern"]).format(
                effort=effort
            )
            for effort in _model_specs(config)
        },
        "initial_responses": {
            effort: output_dir
            / str(config["artifacts"]["initial_responses_pattern"]).format(
                effort=effort
            )
            for effort in _model_specs(config)
        },
        "intermediate_xhigh": output_dir
        / config["artifacts"]["intermediate_xhigh_responses_filename"],
    }


def _runtime_config(
    config: dict[str, Any],
    base_config: dict[str, Any],
    spec: dict[str, Any],
    *,
    repeats: int | None = None,
    max_budget_usd: float,
) -> dict[str, Any]:
    runtime = copy.deepcopy(base_config)
    runtime["study"]["repeats"] = int(repeats or config["study"]["repeats"])
    runtime["api"] = {
        **config["api"],
        "model": spec["model"],
        "reasoning_effort": spec["reasoning_effort"],
        "max_output_tokens": int(spec["max_output_tokens"]),
        "max_budget_usd": float(max_budget_usd),
    }
    return runtime


def _score_config(
    config: dict[str, Any], base_config: dict[str, Any]
) -> dict[str, Any]:
    runtime = copy.deepcopy(base_config)
    runtime["study"]["repeats"] = int(config["study"]["repeats"])
    runtime["api"] = copy.deepcopy(config["api"])
    runtime["bootstrap"] = copy.deepcopy(config["bootstrap"])
    return runtime


def _score_spec(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    return {
        **spec,
        "pricing_usd_per_million_tokens": config["api"][
            "pricing_usd_per_million_tokens"
        ],
    }


async def _call_openai_detailed(
    *,
    client: Any,
    record: dict[str, Any],
    repeat: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    result = cast(
        dict[str, Any],
        await low_study._call_openai_detailed(
            client=client,
            record=record,
            repeat=repeat,
            config=config,
        ),
    )
    result["reasoning_effort"] = str(config["api"]["reasoning_effort"])
    result["max_output_tokens"] = int(config["api"]["max_output_tokens"])
    return result


async def _execute_calls(
    *,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    from openai import AsyncOpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --execute")
    existing = baseline._load_jsonl(output_path)
    records_by_key = {
        (record["persona_id"], record["week_start"], record["arm"]): record
        for record in records
    }
    completed = baseline._completed_keys(
        existing,
        records_by_key,
        repeats=int(config["study"]["repeats"]),
        requested_model=str(config["api"]["model"]),
    )
    pricing = config["api"]["pricing_usd_per_million_tokens"]
    actual_spend = sum(
        baseline._request_cost_usd(
            input_tokens=int(row.get("usage", {}).get("input_tokens", 0)),
            output_tokens=int(row.get("usage", {}).get("output_tokens", 0)),
            pricing=pricing,
        )
        for row in existing
        if row.get("status") in baseline.TERMINAL_RESPONSE_STATUSES
    )
    pending = [
        (record, repeat)
        for repeat in range(1, int(config["study"]["repeats"]) + 1)
        for record in records
        if baseline._response_key(record, repeat) not in completed
    ]
    random.Random(20260712).shuffle(pending)
    queue: asyncio.Queue[tuple[dict[str, Any], int]] = asyncio.Queue()
    for item in pending:
        queue.put_nowait(item)
    client = AsyncOpenAI()
    spend_lock = asyncio.Lock()
    max_budget = float(config["api"]["max_budget_usd"])
    reserved_spend = 0.0
    completed_now = 0
    stopped_for_budget = False

    async def worker() -> None:
        nonlocal actual_spend, completed_now, reserved_spend, stopped_for_budget
        while not stopped_for_budget:
            try:
                record, repeat = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            reservation = baseline._request_cost_usd(
                input_tokens=baseline._approx_tokens(record["prompt"]),
                output_tokens=int(config["api"]["max_output_tokens"]),
                pricing=pricing,
            )
            async with spend_lock:
                if actual_spend + reserved_spend + reservation > max_budget:
                    stopped_for_budget = True
                    return
                reserved_spend += reservation
            result = await _call_openai_detailed(
                client=client,
                record=record,
                repeat=repeat,
                config=config,
            )
            async with spend_lock:
                reserved_spend -= reservation
                baseline._append_jsonl(output_path, [result])
                if result["status"] in baseline.TERMINAL_RESPONSE_STATUSES:
                    completed_now += 1
                    actual_spend += baseline._request_cost_usd(
                        input_tokens=int(
                            result.get("usage", {}).get("input_tokens", 0)
                        ),
                        output_tokens=int(
                            result.get("usage", {}).get("output_tokens", 0)
                        ),
                        pricing=pricing,
                    )

    try:
        await asyncio.gather(
            *[worker() for _ in range(int(config["api"]["concurrency"]))]
        )
    finally:
        await client.close()
    return {
        "completed_before": len(completed),
        "completed_now": completed_now,
        "remaining": len(pending) - completed_now,
        "actual_spend_usd": actual_spend,
        "stopped_for_budget": stopped_for_budget,
    }


def _validate_receipt_effort(responses: list[dict[str, Any]], effort: str) -> None:
    mismatched = [
        row
        for row in responses
        if row.get("status") in baseline.TERMINAL_RESPONSE_STATUSES
        and row.get("reasoning_effort") != effort
    ]
    if mismatched:
        raise ValueError(f"Response receipts do not match reasoning effort {effort}")


def _validate_receipt_caps(
    responses: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    amendment = config["protocol_amendment"]
    initial_cap = int(amendment["initial_max_output_tokens"])
    allowed_caps = {initial_cap} | {
        int(spec["max_output_tokens"]) for spec in _model_specs(config).values()
    }
    for row in responses:
        if row.get("status") not in baseline.TERMINAL_RESPONSE_STATUSES:
            continue
        cap = int(row.get("max_output_tokens") or 0)
        if cap not in allowed_caps:
            raise ValueError(f"Unexpected response token cap: {cap}")
        if cap == initial_cap and row.get("response_status") == "incomplete":
            raise ValueError("An incomplete initial response was reused")


def _cap_summary(responses: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in responses:
        if row.get("status") not in baseline.TERMINAL_RESPONSE_STATUSES:
            continue
        key = str(row.get("max_output_tokens") or "unrecorded")
        summary[key] = summary.get(key, 0) + 1
    return summary


def _cost_summary(
    responses: list[dict[str, Any]], pricing: dict[str, float]
) -> dict[str, Any]:
    usage = {
        "input_tokens": sum(
            int((row.get("usage") or {}).get("input_tokens") or 0) for row in responses
        ),
        "cached_input_tokens": sum(
            int(
                ((row.get("usage") or {}).get("input_tokens_details") or {}).get(
                    "cached_tokens"
                )
                or 0
            )
            for row in responses
        ),
        "cache_write_tokens": sum(
            int(
                ((row.get("usage") or {}).get("input_tokens_details") or {}).get(
                    "cache_write_tokens"
                )
                or 0
            )
            for row in responses
        ),
        "output_tokens": sum(
            int((row.get("usage") or {}).get("output_tokens") or 0) for row in responses
        ),
    }
    return {
        "receipts": len(responses),
        "usage": usage,
        "standard_rate_token_calculation_usd": baseline._request_cost_usd(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            pricing=pricing,
        ),
        "cache_aware_token_calculation_usd": low_study._cache_aware_cost_usd(
            usage, pricing
        ),
    }


def _latency_summary(responses: list[dict[str, Any]]) -> dict[str, Any]:
    terminal = [
        row
        for row in responses
        if row.get("status") in baseline.TERMINAL_RESPONSE_STATUSES
    ]
    values = [
        float(row["latency_seconds"])
        for row in terminal
        if row.get("latency_seconds") is not None
    ]
    if len(values) != len(terminal):
        raise ValueError("A terminal response is missing latency")
    repeats = sorted({int(row["repeat"]) for row in terminal})
    repeat_medians = {
        str(repeat): float(
            statistics.median(
                float(row["latency_seconds"])
                for row in terminal
                if int(row["repeat"]) == repeat
            )
        )
        for repeat in repeats
    }
    return {
        "unit": "terminal_persona_week_api_call",
        "count": len(values),
        "median_seconds": float(statistics.median(values)),
        "repeat_median_seconds": repeat_medians,
        "diagnostic_only": True,
    }


def _protocol_overhead_costs(
    config: dict[str, Any], paths: dict[str, Any]
) -> dict[str, Any]:
    def discarded(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if row.get("response_status") == "incomplete"
            or row.get("status") not in baseline.TERMINAL_RESPONSE_STATUSES
        ]

    groups = {
        "initial_smoke": [
            row
            for effort in _model_specs(config)
            for row in baseline._load_jsonl(paths["initial_smoke"][effort])
        ],
        "amended_smoke": [
            row
            for effort in _model_specs(config)
            for row in baseline._load_jsonl(paths["smoke"][effort])
        ],
        "discarded_initial_full_run": [
            row
            for effort in _model_specs(config)
            for row in discarded(
                baseline._load_jsonl(paths["initial_responses"][effort])
            )
        ],
        "discarded_intermediate_xhigh": discarded(
            baseline._load_jsonl(paths["intermediate_xhigh"])
        ),
    }
    pricing = config["api"]["pricing_usd_per_million_tokens"]
    summaries = {
        name: _cost_summary(responses, pricing) for name, responses in groups.items()
    }
    return {
        "groups": summaries,
        "receipts": sum(item["receipts"] for item in summaries.values()),
        "standard_rate_token_calculation_usd": sum(
            item["standard_rate_token_calculation_usd"] for item in summaries.values()
        ),
        "cache_aware_token_calculation_usd": sum(
            item["cache_aware_token_calculation_usd"] for item in summaries.values()
        ),
    }


def _validate_expected_receipts(
    *,
    config: dict[str, Any],
    responses: list[dict[str, Any]],
    records: list[dict[str, Any]],
    effort: str,
    repeats: int,
) -> None:
    _validate_receipt_effort(responses, effort)
    _validate_receipt_caps(responses, config)
    record_map = {
        (record["persona_id"], record["week_start"], record["arm"]): record
        for record in records
    }
    completed = baseline._completed_keys(
        responses,
        record_map,
        repeats=repeats,
        requested_model="gpt-5.6-luna",
    )
    expected = {
        (record["persona_id"], record["week_start"], record["arm"], repeat)
        for record in records
        for repeat in range(1, repeats + 1)
    }
    if completed != expected:
        raise ValueError(
            f"{effort} response set is incomplete: {len(completed)}/{len(expected)}"
        )


def _load_no_reasoning_baseline(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    frozen = config["no_reasoning_baseline"]
    paths = {
        key: baseline._rooted(frozen[f"{key}_path"], root)
        for key in ("config", "manifest", "responses", "metrics")
    }
    expected = frozen["expected_sha256"]
    for key, path in paths.items():
        if baseline._sha256_file(path) != expected[key]:
            raise ValueError(f"Frozen Luna-none {key} changed")
    none_config = baseline._read_yaml(paths["config"])
    none_manifest = baseline._read_json(paths["manifest"])
    none_metrics = baseline._read_json(paths["metrics"])
    if none_manifest["config_sha256"] != baseline._sha256_text(
        baseline._canonical_json(none_config)
    ):
        raise ValueError("Frozen Luna-none manifest points to a different config")
    specs = {
        str(item["key"]): item
        for item in none_config["models"]
        if item["model"] == "gpt-5.6-luna"
    }
    if list(specs) != ["gpt_5_6_luna"]:
        raise ValueError("Frozen Luna-none model specification changed")
    spec = specs["gpt_5_6_luna"]
    if spec["reasoning_effort"] != "none":
        raise ValueError("Frozen Luna-none reasoning effort changed")
    provenance = none_metrics["provenance"]
    if provenance["manifest_sha256"] != expected["manifest"]:
        raise ValueError("Frozen Luna-none metrics point to a different manifest")
    if provenance["prompts_sha256"] != expected["prompts"]:
        raise ValueError("Frozen Luna-none prompt data changed")
    if provenance["responses_sha256"]["gpt_5_6_luna"] != expected["responses"]:
        raise ValueError("Frozen Luna-none metrics point to different responses")
    return none_config, spec, baseline._load_jsonl(paths["responses"]), none_metrics


def _validate_none_receipts(
    responses: list[dict[str, Any]],
    records: list[dict[str, Any]],
    repeats: int,
) -> None:
    record_map = {
        (record["persona_id"], record["week_start"], record["arm"]): record
        for record in records
    }
    completed = baseline._completed_keys(
        responses,
        record_map,
        repeats=repeats,
        requested_model="gpt-5.6-luna",
    )
    expected = {
        (record["persona_id"], record["week_start"], record["arm"], repeat)
        for record in records
        for repeat in range(1, repeats + 1)
    }
    if completed != expected:
        raise ValueError(
            f"none response set is incomplete: {len(completed)}/{len(expected)}"
        )


def _none_provenance(config: dict[str, Any], root: Path) -> dict[str, Any]:
    frozen = config["no_reasoning_baseline"]
    _load_no_reasoning_baseline(config, root)
    return {
        "published_commit": frozen["published_commit"],
        **{
            f"{key}_sha256": baseline._sha256_file(
                baseline._rooted(frozen[f"{key}_path"], root)
            )
            for key in ("config", "manifest", "responses", "metrics")
        },
        "prompts_sha256": frozen["expected_sha256"]["prompts"],
    }


def _load_frozen(config: dict[str, Any], root: Path) -> tuple[Any, ...]:
    low_config_path = baseline._rooted(config["baseline"]["config_path"], root)
    expected = config["baseline"]["expected_sha256"]
    if baseline._sha256_file(low_config_path) != expected["config"]:
        raise ValueError("Frozen Luna-low configuration changed")
    low_config = baseline._read_yaml(low_config_path)
    base_paths = low_study._baseline_paths(low_config, root)
    low_study._validate_hashes(low_config, base_paths)
    base_config = baseline._read_yaml(base_paths["config"])
    base_manifest = baseline._read_json(base_paths["manifest"])
    base_metrics = baseline._read_json(base_paths["metrics"])
    records = baseline._load_jsonl(base_paths["prompts"])
    cases, outcomes, targets, episodes = model_study._load_complete_development(
        base_config, root
    )
    scoring_source_keys = {
        "complete_case_outcomes_path",
        "complete_drift_episodes_path",
        "complete_entry_target_path",
    }
    for key, path in model_study._source_paths(base_config, root).items():
        if key not in scoring_source_keys:
            continue
        if base_manifest["source_sha256"].get(key) != baseline._sha256_file(path):
            raise ValueError(f"Complete development source changed: {key}")
    counts = model_study._observed_counts(
        records=records,
        cases=cases,
        outcomes=outcomes,
        targets=targets,
        episodes=episodes,
    )
    if counts != base_manifest["counts"]:
        raise ValueError("Complete development counts changed")
    low_paths = low_study._paths(low_config, root)
    low_manifest = baseline._read_json(low_paths["manifest"])
    if low_manifest["config_sha256"] != baseline._sha256_text(
        baseline._canonical_json(low_config)
    ):
        raise ValueError("Frozen Luna-low manifest points to a different config")
    for key in ("manifest", "responses", "metrics"):
        path = baseline._rooted(config["baseline"][f"{key}_path"], root)
        if baseline._sha256_file(path) != expected[key]:
            raise ValueError(f"Frozen Luna-low {key} changed")
    low_responses = baseline._load_jsonl(low_paths["responses"])
    low_metrics = baseline._read_json(low_paths["metrics"])
    if low_metrics["provenance"]["luna_low_responses_sha256"] != expected["responses"]:
        raise ValueError("Frozen Luna-low metrics point to different responses")
    if len(records) != int(config["study"]["expected_persona_weeks"]):
        raise ValueError("Persona-week count changed")
    calls = len(records) * int(config["study"]["repeats"])
    if calls != int(config["study"]["expected_calls_per_effort"]):
        raise ValueError("Call count per effort changed")
    if calls * len(_model_specs(config)) != int(
        config["study"]["expected_total_calls"]
    ):
        raise ValueError("Total call count changed")
    return (
        low_config,
        base_config,
        records,
        cases,
        outcomes,
        targets,
        episodes,
        low_responses,
        low_metrics,
        base_metrics,
        low_paths,
        low_manifest,
    )


def _single_effort_config(
    config: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    return {
        "study": {
            "repeats": config["study"]["repeats"],
            "expected_calls": config["study"]["expected_calls_per_effort"],
        },
        "model": spec,
        "api": {**config["api"], "max_output_tokens": spec["max_output_tokens"]},
        "smoke": config["smoke"],
    }


def _projection(
    *,
    config: dict[str, Any],
    spec: dict[str, Any],
    base_metrics: dict[str, Any],
    records: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    effort = str(spec["reasoning_effort"])
    _validate_receipt_effort(responses, effort)
    _validate_receipt_caps(responses, config)
    selected = low_study._smoke_records(records, int(config["smoke"]["prompt_count"]))
    return cast(
        dict[str, Any],
        low_study._smoke_projection(
            config=_single_effort_config(config, spec),
            base_metrics=base_metrics,
            smoke_records=selected,
            responses=responses,
        ),
    )


def _budget_summary(
    config: dict[str, Any], projections: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    projected = sum(
        float(item["projected_standard_rate_cost_usd"]) for item in projections.values()
    )
    contingency = sum(
        float(item["projected_cost_with_contingency_usd"])
        for item in projections.values()
    )
    budget = float(config["api"]["max_budget_usd"])
    effort_budgets = {
        effort: float(spec["max_budget_usd"])
        for effort, spec in _model_specs(config).items()
    }
    allocated = sum(effort_budgets.values())
    return {
        "projected_standard_rate_cost_usd": projected,
        "projected_cost_with_contingency_usd": contingency,
        "full_run_budget_usd": budget,
        "effort_budgets_usd": effort_budgets,
        "allocated_budget_usd": allocated,
        "within_budget": contingency <= budget and allocated <= budget,
    }


def _all_projections(
    *,
    config: dict[str, Any],
    base_metrics: dict[str, Any],
    records: list[dict[str, Any]],
    paths: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    specs = _model_specs(config)
    projections = {}
    for effort, spec in specs.items():
        path = paths["smoke"][effort]
        if not path.exists():
            raise FileNotFoundError(f"Run the {effort} smoke test first")
        projections[effort] = _projection(
            config=config,
            spec=spec,
            base_metrics=base_metrics,
            records=records,
            responses=baseline._load_jsonl(path),
        )
    return projections, _budget_summary(config, projections)


def _dependency_hashes(root: Path) -> dict[str, str]:
    return {
        "luna_low_runner_sha256": baseline._sha256_file(
            root / "scripts/experiments/compare_twinkl_52zz_luna_reasoning.py"
        ),
        "model_runner_sha256": baseline._sha256_file(
            root / "scripts/experiments/compare_twinkl_52zz_models.py"
        ),
        "shared_api_runner_sha256": baseline._sha256_file(
            root / "scripts/experiments/weekly_verifier_ablation.py"
        ),
        "shared_scorer_sha256": baseline._sha256_file(
            root / "scripts/experiments/reassess_twinkl_752_5.py"
        ),
    }


def _initial_attempt_summary(
    config: dict[str, Any], paths: dict[str, Any]
) -> dict[str, Any]:
    summary = {}
    for effort in _model_specs(config):
        response_path = paths["initial_responses"][effort]
        smoke_path = paths["initial_smoke"][effort]
        if not response_path.exists() or not smoke_path.exists():
            raise FileNotFoundError(f"Initial {effort} receipts are missing")
        responses = baseline._load_jsonl(response_path)
        status_counts: dict[str, int] = {}
        incomplete = 0
        for row in responses:
            status = str(row.get("status") or "missing")
            status_counts[status] = status_counts.get(status, 0) + 1
            if row.get("response_status") == "incomplete":
                incomplete += 1
        summary[effort] = {
            "max_output_tokens": int(
                config["protocol_amendment"]["initial_max_output_tokens"]
            ),
            "responses": len(responses),
            "status_counts": status_counts,
            "incomplete_responses": incomplete,
            "responses_sha256": baseline._sha256_file(response_path),
            "smoke_responses_sha256": baseline._sha256_file(smoke_path),
        }
    return summary


def _intermediate_xhigh_summary(paths: dict[str, Any]) -> dict[str, Any]:
    path = paths["intermediate_xhigh"]
    if not path.exists():
        raise FileNotFoundError("Intermediate xhigh receipts are missing")
    responses = baseline._load_jsonl(path)
    return {
        "responses": len(responses),
        "incomplete_responses": sum(
            row.get("response_status") == "incomplete" for row in responses
        ),
        "responses_sha256": baseline._sha256_file(path),
    }


def _prepare(config: dict[str, Any], root: Path) -> dict[str, Any]:
    (
        _low_config,
        base_config,
        records,
        _cases,
        _outcomes,
        _targets,
        _episodes,
        _low_responses,
        _low_metrics,
        _base_metrics,
        low_paths,
        _low_manifest,
    ) = _load_frozen(config, root)
    paths = _paths(config, root)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    specs = _model_specs(config)
    estimates = {
        effort: baseline.estimate_plan(
            records,
            _runtime_config(
                config,
                base_config,
                spec,
                max_budget_usd=float(spec["max_budget_usd"]),
            ),
        )
        for effort, spec in specs.items()
    }
    manifest = {
        "study_id": config["study_id"],
        "schema_version": "twinkl-ck3w-luna-higher-reasoning-v1",
        "prepared_at": datetime.now(UTC).isoformat(),
        "repo_head": baseline._git_head(root),
        "counts": _low_metrics["counts"],
        "setup": model_study.WEEKLY_WITHOUT,
        "repeats": int(config["study"]["repeats"]),
        "planned_calls_per_effort": len(records) * int(config["study"]["repeats"]),
        "planned_calls_total": len(records)
        * int(config["study"]["repeats"])
        * len(specs),
        "models": list(specs.values()),
        "estimates": estimates,
        "max_budget_usd": float(config["api"]["max_budget_usd"]),
        "pricing": {
            "verified_at": config["api"]["pricing_verified_at"],
            "source": config["api"]["pricing_source"],
            "usd_per_million_tokens": config["api"]["pricing_usd_per_million_tokens"],
        },
        "protocol_amendment": config["protocol_amendment"],
        "initial_attempt": _initial_attempt_summary(config, paths),
        "intermediate_xhigh_attempt": _intermediate_xhigh_summary(paths),
        "smoke_prompt_count": int(config["smoke"]["prompt_count"]),
        "smoke_prompt_sha256": [
            row["prompt_sha256"]
            for row in low_study._smoke_records(
                records, int(config["smoke"]["prompt_count"])
            )
        ],
        "config_sha256": baseline._sha256_text(baseline._canonical_json(config)),
        "runner_sha256": baseline._sha256_file(Path(__file__)),
        **_dependency_hashes(root),
        "frozen_luna_low": {
            "published_commit": config["baseline"]["published_commit"],
            "config_sha256": config["baseline"]["expected_sha256"]["config"],
            "manifest_sha256": baseline._sha256_file(low_paths["manifest"]),
            "prompts_sha256": _low_config["baseline"]["expected_sha256"]["prompts"],
            "responses_sha256": baseline._sha256_file(low_paths["responses"]),
            "metrics_sha256": baseline._sha256_file(low_paths["metrics"]),
        },
        "frozen_luna_none": _none_provenance(config, root),
        "prompt_contract": {
            "same_as_luna_low": True,
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


def _load_prepared(config: dict[str, Any], root: Path) -> tuple[Any, ...]:
    frozen = _load_frozen(config, root)
    paths = _paths(config, root)
    if not paths["manifest"].exists():
        raise FileNotFoundError("Run prepare before this command")
    manifest = baseline._read_json(paths["manifest"])
    expected = {
        "config_sha256": baseline._sha256_text(baseline._canonical_json(config)),
        "runner_sha256": baseline._sha256_file(Path(__file__)),
        **_dependency_hashes(root),
    }
    for key, digest in expected.items():
        if manifest.get(key) != digest:
            raise ValueError(f"Prepared protocol hash mismatch: {key}")
    if manifest.get("initial_attempt") != _initial_attempt_summary(config, paths):
        raise ValueError("Initial 2,000-token attempt changed")
    if manifest.get("intermediate_xhigh_attempt") != _intermediate_xhigh_summary(paths):
        raise ValueError("Intermediate xhigh attempt changed")
    if manifest.get("frozen_luna_none") != _none_provenance(config, root):
        raise ValueError("Frozen Luna-none provenance changed")
    return (*frozen, paths, manifest)


def command_prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    manifest = _prepare(config, root)
    print(
        json.dumps(
            {
                "counts": manifest["counts"],
                "planned_calls_total": manifest["planned_calls_total"],
                "estimates": manifest["estimates"],
                "max_budget_usd": manifest["max_budget_usd"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def command_smoke(args: argparse.Namespace) -> None:
    if not args.execute:
        raise SystemExit("Refusing paid calls without --execute")
    from dotenv import load_dotenv

    root = Path(args.root).resolve()
    load_dotenv(root / ".env")
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    (
        _low_config,
        base_config,
        records,
        _cases,
        _outcomes,
        _targets,
        _episodes,
        _low_responses,
        _low_metrics,
        base_metrics,
        _low_paths,
        _low_manifest,
        paths,
        _manifest,
    ) = _load_prepared(config, root)
    specs = _model_specs(config)
    selected = low_study._smoke_records(records, int(config["smoke"]["prompt_count"]))
    results = {}
    for effort in _efforts(args.effort, specs):
        runtime = _runtime_config(
            config,
            base_config,
            specs[effort],
            repeats=int(config["smoke"]["repeats"]),
            max_budget_usd=float(config["smoke"]["max_budget_usd_per_effort"]),
        )
        execution = asyncio.run(
            _execute_calls(
                records=selected,
                config=runtime,
                output_path=paths["smoke"][effort],
            )
        )
        responses = baseline._load_jsonl(paths["smoke"][effort])
        results[effort] = {
            "execution": execution,
            "projection": _projection(
                config=config,
                spec=specs[effort],
                base_metrics=base_metrics,
                records=records,
                responses=responses,
            ),
        }
    try:
        projections, budget = _all_projections(
            config=config,
            base_metrics=base_metrics,
            records=records,
            paths=paths,
        )
    except FileNotFoundError:
        projections, budget = {}, None
    print(
        json.dumps(
            {"results": results, "all_projections": projections, "budget": budget},
            indent=2,
            sort_keys=True,
        )
    )


def command_run(args: argparse.Namespace) -> None:
    if not args.execute:
        raise SystemExit("Refusing paid calls without --execute")
    from dotenv import load_dotenv

    root = Path(args.root).resolve()
    load_dotenv(root / ".env")
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    (
        _low_config,
        base_config,
        records,
        _cases,
        _outcomes,
        _targets,
        _episodes,
        _low_responses,
        _low_metrics,
        base_metrics,
        _low_paths,
        _low_manifest,
        paths,
        _manifest,
    ) = _load_prepared(config, root)
    specs = _model_specs(config)
    projections, budget = _all_projections(
        config=config,
        base_metrics=base_metrics,
        records=records,
        paths=paths,
    )
    if not budget["within_budget"]:
        raise ValueError("Smoke projections exceed the aggregate full-run budget")
    results = {}
    for effort in _efforts(args.effort, specs):
        runtime = _runtime_config(
            config,
            base_config,
            specs[effort],
            max_budget_usd=float(specs[effort]["max_budget_usd"]),
        )
        results[effort] = asyncio.run(
            _execute_calls(
                records=records,
                config=runtime,
                output_path=paths["responses"][effort],
            )
        )
        _validate_receipt_effort(
            baseline._load_jsonl(paths["responses"][effort]), effort
        )
    print(
        json.dumps(
            {"projections": projections, "budget": budget, "execution": results},
            indent=2,
            sort_keys=True,
        )
    )


def command_score(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    (
        low_config,
        base_config,
        records,
        cases,
        outcomes,
        targets,
        episodes,
        low_responses,
        _low_metrics,
        base_metrics,
        low_paths,
        _low_manifest,
        paths,
        manifest,
    ) = _load_prepared(config, root)
    specs = _model_specs(config)
    projections, budget = _all_projections(
        config=config,
        base_metrics=base_metrics,
        records=records,
        paths=paths,
    )
    _none_config, none_spec, none_responses, _none_metrics = (
        _load_no_reasoning_baseline(config, root)
    )
    _validate_none_receipts(none_responses, records, int(config["study"]["repeats"]))
    score_config = _score_config(config, base_config)
    none_result, none_case_stats = model_study._score_model(
        config=score_config,
        model_spec=_score_spec(config, {**none_spec, "key": "luna_none"}),
        records=records,
        cases=cases,
        outcomes=outcomes,
        targets=targets,
        episodes=episodes,
        responses=none_responses,
    )
    none_result["latency"] = _latency_summary(none_responses)
    low_score_config = low_study._score_config(low_config, base_config)
    low_result, low_case_stats = model_study._score_model(
        config=low_score_config,
        model_spec=low_study._model_spec(low_config),
        records=records,
        cases=cases,
        outcomes=outcomes,
        targets=targets,
        episodes=episodes,
        responses=low_responses,
    )
    low_result["latency"] = _latency_summary(low_responses)
    results = {"luna_none": none_result, "luna_low": low_result}
    comparisons = {}
    comparisons_vs_none = {}
    costs = {}
    case_ids = sorted(case["canonical_case_id"] for case in cases)
    comparisons_vs_none["luna_low_minus_luna_none"] = reassess._comparison_bootstrap(
        case_ids=case_ids,
        first=none_case_stats,
        second=low_case_stats,
        config=score_config,
        seed_offset=2,
    )
    for offset, (effort, spec) in enumerate(specs.items(), start=3):
        responses = baseline._load_jsonl(paths["responses"][effort])
        _validate_expected_receipts(
            config=config,
            responses=responses,
            records=records,
            effort=effort,
            repeats=int(config["study"]["repeats"]),
        )
        result, case_stats = model_study._score_model(
            config=score_config,
            model_spec=_score_spec(config, spec),
            records=records,
            cases=cases,
            outcomes=outcomes,
            targets=targets,
            episodes=episodes,
            responses=responses,
        )
        result["usage_details"] = low_study._usage_detail_summary(responses)
        result["max_output_token_cap_counts"] = _cap_summary(responses)
        result["latency"] = _latency_summary(responses)
        results[str(spec["key"])] = result
        comparisons[f"{spec['key']}_minus_luna_low"] = reassess._comparison_bootstrap(
            case_ids=case_ids,
            first=low_case_stats,
            second=case_stats,
            config=score_config,
            seed_offset=offset,
        )
        comparisons_vs_none[f"{spec['key']}_minus_luna_none"] = (
            reassess._comparison_bootstrap(
                case_ids=case_ids,
                first=none_case_stats,
                second=case_stats,
                config=score_config,
                seed_offset=offset + len(specs),
            )
        )
        costs[effort] = {
            "standard_rate_token_calculation_usd": float(
                result["response_summary"]["actual_spend_usd"]
            ),
            "cache_aware_token_calculation_usd": low_study._cache_aware_cost_usd(
                result["usage_details"],
                config["api"]["pricing_usd_per_million_tokens"],
            ),
        }
    active_standard_cost = sum(
        item["standard_rate_token_calculation_usd"] for item in costs.values()
    )
    active_cache_aware_cost = sum(
        item["cache_aware_token_calculation_usd"] for item in costs.values()
    )
    protocol_overhead = _protocol_overhead_costs(config, paths)
    current_pricing = config["api"]["pricing_usd_per_million_tokens"]
    reference_costs = {
        "none": _cost_summary(none_responses, current_pricing),
        "low": _cost_summary(low_responses, current_pricing),
    }
    metrics = {
        "study_id": config["study_id"],
        "scored_at": datetime.now(UTC).isoformat(),
        "counts": manifest["counts"],
        "comparison_baseline": "luna_low",
        "secondary_comparison_baseline": "luna_none",
        "models": results,
        "paired_trajectory_bootstrap": comparisons,
        "paired_trajectory_bootstrap_vs_none": comparisons_vs_none,
        "latency_definition": {
            "measure": "median end-to-end API latency per terminal persona-week call",
            "scope": "2,853 coordinates per reasoning effort across three repeats",
            "selection_use": "diagnostic_only",
            "limitation": (
                "Runs used different dates, output ceilings, and client scheduling."
            ),
        },
        "smoke_projections": projections,
        "budget": budget,
        "costs": costs,
        "reference_costs_at_current_rates": reference_costs,
        "aggregate_standard_rate_token_calculation_usd": active_standard_cost,
        "aggregate_cache_aware_token_calculation_usd": active_cache_aware_cost,
        "protocol_overhead_costs": protocol_overhead,
        "total_study_standard_rate_token_calculation_usd": (
            active_standard_cost
            + protocol_overhead["standard_rate_token_calculation_usd"]
        ),
        "total_study_cache_aware_token_calculation_usd": (
            active_cache_aware_cost
            + protocol_overhead["cache_aware_token_calculation_usd"]
        ),
        "development_selection": config["decision"]["status"],
        "metric_hierarchy": config["decision"]["metric_hierarchy"],
        "restrictions": config["decision"]["restrictions"],
        "protocol_amendment": config["protocol_amendment"],
        "provenance": {
            "manifest_sha256": baseline._sha256_file(paths["manifest"]),
            "luna_low_manifest_sha256": baseline._sha256_file(low_paths["manifest"]),
            "luna_low_responses_sha256": baseline._sha256_file(low_paths["responses"]),
            "luna_low_metrics_sha256": baseline._sha256_file(low_paths["metrics"]),
            "luna_none": _none_provenance(config, root),
            "responses_sha256": {
                effort: baseline._sha256_file(paths["responses"][effort])
                for effort in specs
            },
            "smoke_responses_sha256": {
                effort: baseline._sha256_file(paths["smoke"][effort])
                for effort in specs
            },
            "initial_responses_sha256": {
                effort: baseline._sha256_file(paths["initial_responses"][effort])
                for effort in specs
            },
            "initial_smoke_responses_sha256": {
                effort: baseline._sha256_file(paths["initial_smoke"][effort])
                for effort in specs
            },
            "intermediate_xhigh_responses_sha256": baseline._sha256_file(
                paths["intermediate_xhigh"]
            ),
        },
    }
    paths["metrics"].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "development_selection": metrics["development_selection"],
                "stability": {
                    key: value["stability"] for key, value in results.items()
                },
                "paired_trajectory_bootstrap": comparisons,
                "paired_trajectory_bootstrap_vs_none": comparisons_vs_none,
                "latency": {key: value["latency"] for key, value in results.items()},
                "costs": costs,
                "reference_costs_at_current_rates": reference_costs,
                "aggregate_standard_rate_token_calculation_usd": metrics[
                    "aggregate_standard_rate_token_calculation_usd"
                ],
                "aggregate_cache_aware_token_calculation_usd": metrics[
                    "aggregate_cache_aware_token_calculation_usd"
                ],
                "protocol_overhead_costs": protocol_overhead,
                "total_study_standard_rate_token_calculation_usd": metrics[
                    "total_study_standard_rate_token_calculation_usd"
                ],
                "total_study_cache_aware_token_calculation_usd": metrics[
                    "total_study_cache_aware_token_calculation_usd"
                ],
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
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument(
        "--effort", choices=("all", "medium", "high", "xhigh"), default="all"
    )
    smoke.add_argument("--execute", action="store_true")
    smoke.set_defaults(func=command_smoke)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--effort", choices=("all", "medium", "high", "xhigh"), default="all"
    )
    run.add_argument("--execute", action="store_true")
    run.set_defaults(func=command_run)
    subparsers.add_parser("score").set_defaults(func=command_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
