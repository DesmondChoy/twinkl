#!/usr/bin/env python3
"""Compare Luna reasoning effort ``low`` with the frozen ``none`` baseline."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.experiments import compare_twinkl_52zz_models as model_study
from scripts.experiments import reassess_twinkl_752_5 as reassess
from scripts.experiments import weekly_verifier_ablation as baseline

DEFAULT_CONFIG_PATH = Path("config/evals/twinkl_52zz_luna_low_v1.yaml")


def _paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    output_dir = baseline._rooted(config["artifacts"]["output_dir"], root)
    return {
        "output_dir": output_dir,
        "manifest": output_dir / config["artifacts"]["manifest_filename"],
        "smoke_responses": output_dir / config["artifacts"]["smoke_responses_filename"],
        "responses": output_dir / config["artifacts"]["responses_filename"],
        "metrics": output_dir / config["artifacts"]["metrics_filename"],
    }


def _baseline_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    spec = config["baseline"]
    return {
        key: baseline._rooted(spec[f"{key}_path"], root)
        for key in ("config", "manifest", "prompts", "metrics")
    } | {
        "luna_none_responses": baseline._rooted(spec["luna_none_responses_path"], root)
    }


def _model_spec(config: dict[str, Any]) -> dict[str, Any]:
    return {
        **config["model"],
        "max_budget_usd": float(config["api"]["max_budget_usd"]),
        "pricing_usd_per_million_tokens": config["api"][
            "pricing_usd_per_million_tokens"
        ],
    }


def _runtime_config(
    config: dict[str, Any],
    base_config: dict[str, Any],
    *,
    repeats: int | None = None,
    max_budget_usd: float | None = None,
) -> dict[str, Any]:
    runtime = copy.deepcopy(base_config)
    runtime["study"]["repeats"] = int(repeats or config["study"]["repeats"])
    runtime["api"] = {
        **config["api"],
        "model": config["model"]["model"],
        "reasoning_effort": config["model"]["reasoning_effort"],
        "max_budget_usd": float(
            max_budget_usd
            if max_budget_usd is not None
            else config["api"]["max_budget_usd"]
        ),
    }
    return runtime


def _score_config(
    config: dict[str, Any], base_config: dict[str, Any]
) -> dict[str, Any]:
    score_config = _runtime_config(config, base_config)
    score_config["bootstrap"] = copy.deepcopy(config["bootstrap"])
    return score_config


def _usage_payload(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    if isinstance(usage, dict):
        payload = dict(usage)
    elif hasattr(usage, "model_dump"):
        payload = usage.model_dump(mode="json")
    else:
        payload = {
            key: getattr(usage, key)
            for key in (
                "input_tokens",
                "input_tokens_details",
                "output_tokens",
                "output_tokens_details",
                "total_tokens",
            )
            if getattr(usage, key, None) is not None
        }
    input_tokens = int(payload.get("input_tokens") or 0)
    output_tokens = int(payload.get("output_tokens") or 0)
    payload["input_tokens"] = input_tokens
    payload["output_tokens"] = output_tokens
    payload["total_tokens"] = int(
        payload.get("total_tokens") or input_tokens + output_tokens
    )
    return payload


def _response_execution_details(response: Any) -> dict[str, Any]:
    incomplete = getattr(response, "incomplete_details", None)
    if hasattr(incomplete, "model_dump"):
        incomplete = incomplete.model_dump(mode="json")
    return {
        "response_status": getattr(response, "status", None),
        "incomplete_details": incomplete,
    }


def _receipt_base(
    *,
    response: Any,
    record: dict[str, Any],
    repeat: int,
    attempt: int,
    started: float,
    model: str,
) -> dict[str, Any]:
    return {
        "persona_id": record["persona_id"],
        "week_start": record["week_start"],
        "week_end": record["week_end"],
        "arm": record["arm"],
        "repeat": repeat,
        "prompt_sha256": record["prompt_sha256"],
        "runtime_text_sha256": record["runtime_text_sha256"],
        "requested_model": model,
        "resolved_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        **_response_execution_details(response),
        "attempts": attempt,
        "latency_seconds": time.monotonic() - started,
        "usage": _usage_payload(response),
    }


async def _call_openai_detailed(
    *,
    client: Any,
    record: dict[str, Any],
    repeat: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    api = config["api"]
    started = time.monotonic()
    last_error: Exception | None = None
    for attempt in range(1, int(api["max_attempts"]) + 1):
        try:
            response_model = baseline._response_model(record)
            response = await client.responses.parse(
                model=api["model"],
                input=record["prompt"],
                text_format=response_model,
                reasoning={"effort": api["reasoning_effort"]},
                max_output_tokens=int(api["max_output_tokens"]),
                store=bool(api["store"]),
                service_tier=api["service_tier"],
                timeout=float(api["timeout_seconds"]),
            )
            receipt = _receipt_base(
                response=response,
                record=record,
                repeat=repeat,
                attempt=attempt,
                started=started,
                model=str(api["model"]),
            )
            parsed = getattr(response, "output_parsed", None)
            if not isinstance(parsed, response_model):
                return {
                    "status": "refusal",
                    **receipt,
                    "refusal": baseline._response_refusal(response),
                }
            try:
                baseline.validate_parsed_response(parsed=parsed, record=record)
            except ValueError as error:
                return {
                    "status": "invalid",
                    **receipt,
                    "validation_error": str(error),
                    "parsed": parsed.model_dump(mode="json"),
                }
            return {
                "status": "ok",
                **receipt,
                "parsed": parsed.model_dump(mode="json"),
            }
        except Exception as error:  # noqa: BLE001 - persist API failures
            last_error = error
            if attempt >= int(api["max_attempts"]) or not baseline._is_transient_error(
                error
            ):
                break
            await asyncio.sleep((2 ** (attempt - 1)) + random.random())
    return {
        "status": "error",
        "persona_id": record["persona_id"],
        "week_start": record["week_start"],
        "week_end": record["week_end"],
        "arm": record["arm"],
        "repeat": repeat,
        "prompt_sha256": record["prompt_sha256"],
        "runtime_text_sha256": record["runtime_text_sha256"],
        "requested_model": api["model"],
        "attempts": attempt,
        "latency_seconds": time.monotonic() - started,
        "error_type": type(last_error).__name__ if last_error else "UnknownError",
        "error": str(last_error) if last_error else "Unknown API error",
    }


async def _execute_calls(
    *,
    records: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    original = baseline._call_openai
    baseline._call_openai = _call_openai_detailed
    try:
        return await baseline.execute_calls(
            records=records,
            config=config,
            output_path=output_path,
        )
    finally:
        baseline._call_openai = original


def _validate_hashes(config: dict[str, Any], paths: dict[str, Path]) -> None:
    expected = config["baseline"]["expected_sha256"]
    for key in ("manifest", "prompts", "metrics", "luna_none_responses"):
        observed = baseline._sha256_file(paths[key])
        if observed != expected[key]:
            raise ValueError(f"Frozen Luna-none file changed: {key}")


def _load_frozen(
    config: dict[str, Any], root: Path
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Any,
    Any,
    Any,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Path],
]:
    paths = _baseline_paths(config, root)
    _validate_hashes(config, paths)
    base_config = baseline._read_yaml(paths["config"])
    base_manifest = baseline._read_json(paths["manifest"])
    base_metrics = baseline._read_json(paths["metrics"])
    records = baseline._load_jsonl(paths["prompts"])
    rebuilt, cases, outcomes, targets, episodes = model_study._build_prompt_records(
        base_config, root
    )
    if baseline._canonical_json(records) != baseline._canonical_json(rebuilt):
        raise ValueError("Frozen prompts no longer match the complete development data")
    if base_manifest["config_sha256"] != baseline._sha256_text(
        baseline._canonical_json(base_config)
    ):
        raise ValueError("Frozen Luna-none configuration changed")
    if base_manifest["prompt_template_sha256"] != baseline._sha256_file(
        baseline._rooted(base_config["study"]["prompt_path"], root)
    ):
        raise ValueError("Weekly Drift Reviewer prompt changed")
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
    if counts["personas"] != int(config["study"]["expected_personas"]):
        raise ValueError("Persona count changed")
    if len(records) != int(config["study"]["expected_persona_weeks"]):
        raise ValueError("Persona-week count changed")
    planned_calls = len(records) * int(config["study"]["repeats"])
    if planned_calls != int(config["study"]["expected_calls"]):
        raise ValueError("Planned call count changed")
    provenance = base_metrics["provenance"]
    if (
        provenance["responses_sha256"]["gpt_5_6_luna"]
        != config["baseline"]["expected_sha256"]["luna_none_responses"]
    ):
        raise ValueError("Frozen Luna-none metrics point to different responses")
    none_responses = baseline._load_jsonl(paths["luna_none_responses"])
    return (
        base_config,
        records,
        cases,
        outcomes,
        targets,
        episodes,
        none_responses,
        base_metrics,
        paths,
    )


def _smoke_records(
    records: list[dict[str, Any]], prompt_count: int
) -> list[dict[str, Any]]:
    if not 1 < prompt_count <= len(records):
        raise ValueError("Smoke prompt count is out of range")
    ordered = sorted(
        records,
        key=lambda row: (
            len(str(row["prompt"])),
            str(row["persona_id"]),
            str(row["week_start"]),
        ),
    )
    indices = [
        round(index * (len(ordered) - 1) / (prompt_count - 1))
        for index in range(prompt_count)
    ]
    return [ordered[index] for index in indices]


def _usage_detail_summary(responses: list[dict[str, Any]]) -> dict[str, int]:
    terminal = [
        row
        for row in responses
        if row.get("status") in baseline.TERMINAL_RESPONSE_STATUSES
    ]
    return {
        "terminal_responses": len(terminal),
        "input_tokens": sum(
            int((row.get("usage") or {}).get("input_tokens") or 0) for row in terminal
        ),
        "cached_input_tokens": sum(
            int(
                ((row.get("usage") or {}).get("input_tokens_details") or {}).get(
                    "cached_tokens"
                )
                or 0
            )
            for row in terminal
        ),
        "cache_write_tokens": sum(
            int(
                ((row.get("usage") or {}).get("input_tokens_details") or {}).get(
                    "cache_write_tokens"
                )
                or 0
            )
            for row in terminal
        ),
        "output_tokens": sum(
            int((row.get("usage") or {}).get("output_tokens") or 0) for row in terminal
        ),
        "reasoning_output_tokens": sum(
            int(
                ((row.get("usage") or {}).get("output_tokens_details") or {}).get(
                    "reasoning_tokens"
                )
                or 0
            )
            for row in terminal
        ),
    }


def _cache_aware_cost_usd(usage: dict[str, int], pricing: dict[str, float]) -> float:
    cached = int(usage["cached_input_tokens"])
    cache_write = int(usage["cache_write_tokens"])
    total_input = int(usage["input_tokens"])
    uncached = total_input - cached - cache_write
    if uncached < 0:
        raise ValueError("Detailed input-token categories exceed total input tokens")
    return (
        uncached * float(pricing["input"])
        + cached * float(pricing["cached_input"])
        + cache_write * float(pricing["cache_write"])
        + int(usage["output_tokens"]) * float(pricing["output"])
    ) / 1_000_000


def _smoke_projection(
    *,
    config: dict[str, Any],
    base_metrics: dict[str, Any],
    smoke_records: list[dict[str, Any]],
    responses: list[dict[str, Any]],
) -> dict[str, Any]:
    record_map = {
        (record["persona_id"], record["week_start"], record["arm"]): record
        for record in smoke_records
    }
    completed = baseline._completed_keys(
        responses,
        record_map,
        repeats=int(config["smoke"]["repeats"]),
        requested_model=str(config["model"]["model"]),
    )
    expected = {
        (record["persona_id"], record["week_start"], record["arm"], 1)
        for record in smoke_records
    }
    if completed != expected:
        raise ValueError(
            f"Smoke response set is incomplete: {len(completed)}/{len(expected)}"
        )
    terminal = [
        row
        for row in responses
        if row.get("status") in baseline.TERMINAL_RESPONSE_STATUSES
    ]
    if any(
        row.get("response_status") == "incomplete" or row.get("incomplete_details")
        for row in terminal
    ):
        raise ValueError("Smoke test contains an incomplete API response")
    max_output = max(
        int((row.get("usage") or {}).get("output_tokens") or 0) for row in terminal
    )
    if max_output >= int(config["api"]["max_output_tokens"]) * 0.95:
        raise ValueError("Smoke output approached the configured token cap")
    details = _usage_detail_summary(terminal)
    calls = int(config["study"]["expected_calls"])
    projected_output_tokens = round(details["output_tokens"] / len(expected) * calls)
    frozen_input_tokens = int(
        base_metrics["models"]["gpt_5_6_luna"]["response_summary"]["input_tokens"]
    )
    pricing = config["api"]["pricing_usd_per_million_tokens"]
    projected_cost = baseline._request_cost_usd(
        input_tokens=frozen_input_tokens,
        output_tokens=projected_output_tokens,
        pricing=pricing,
    )
    contingency_cost = projected_cost * float(config["smoke"]["full_run_contingency"])
    return {
        **details,
        "max_output_tokens_in_one_response": max_output,
        "projected_full_input_tokens": frozen_input_tokens,
        "projected_full_output_tokens": projected_output_tokens,
        "projected_standard_rate_cost_usd": projected_cost,
        "projected_cost_with_contingency_usd": contingency_cost,
        "full_run_budget_usd": float(config["api"]["max_budget_usd"]),
        "within_budget": contingency_cost <= float(config["api"]["max_budget_usd"]),
    }


def _decision(comparison: dict[str, Any]) -> str:
    deltas = comparison["deltas"]
    recall = deltas["recall"]
    false_alerts = deltas["false_alerts"]
    coverage = deltas["coverage"]["observed_median"]
    if coverage is None or coverage < -0.05:
        return "keep_luna_none"
    recall_gain = (
        recall["observed_median"] is not None
        and recall["observed_median"] > 0
        and recall["interval"] is not None
        and recall["interval"][0] > 0
        and false_alerts["observed_median"] <= 0
    )
    false_alert_reduction = (
        false_alerts["observed_median"] is not None
        and false_alerts["observed_median"] < 0
        and false_alerts["interval"] is not None
        and false_alerts["interval"][1] < 0
        and recall["observed_median"] is not None
        and recall["observed_median"] >= 0
    )
    return (
        "select_luna_low" if recall_gain or false_alert_reduction else "keep_luna_none"
    )


def _prepare(config: dict[str, Any], root: Path) -> dict[str, Any]:
    (
        base_config,
        records,
        _cases,
        _outcomes,
        _targets,
        _episodes,
        _none_responses,
        base_metrics,
        base_paths,
    ) = _load_frozen(config, root)
    paths = _paths(config, root)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    runtime = _runtime_config(config, base_config)
    manifest = {
        "study_id": config["study_id"],
        "schema_version": "twinkl-52zz-luna-reasoning-v1",
        "prepared_at": datetime.now(UTC).isoformat(),
        "repo_head": baseline._git_head(root),
        "counts": base_metrics["counts"],
        "setup": model_study.WEEKLY_WITHOUT,
        "repeats": int(config["study"]["repeats"]),
        "planned_calls": len(records) * int(config["study"]["repeats"]),
        "model": config["model"],
        "estimate": baseline.estimate_plan(records, runtime),
        "max_budget_usd": float(config["api"]["max_budget_usd"]),
        "smoke_prompt_count": int(config["smoke"]["prompt_count"]),
        "smoke_prompt_sha256": [
            row["prompt_sha256"]
            for row in _smoke_records(records, int(config["smoke"]["prompt_count"]))
        ],
        "config_sha256": baseline._sha256_text(baseline._canonical_json(config)),
        "runner_sha256": baseline._sha256_file(Path(__file__)),
        "shared_api_runner_sha256": baseline._sha256_file(
            root / "scripts/experiments/weekly_verifier_ablation.py"
        ),
        "shared_scorer_sha256": baseline._sha256_file(
            root / "scripts/experiments/reassess_twinkl_752_5.py"
        ),
        "frozen_baseline": {
            "published_commit": config["baseline"]["published_commit"],
            "manifest_sha256": baseline._sha256_file(base_paths["manifest"]),
            "prompts_sha256": baseline._sha256_file(base_paths["prompts"]),
            "metrics_sha256": baseline._sha256_file(base_paths["metrics"]),
            "luna_none_responses_sha256": baseline._sha256_file(
                base_paths["luna_none_responses"]
            ),
            "current_complete_review_report_sha256": baseline._sha256_file(
                baseline._rooted(
                    base_config["sources"]["complete_review_report_path"], root
                )
            ),
        },
        "prompt_contract": {
            "same_as_luna_none": True,
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
        "shared_api_runner_sha256": baseline._sha256_file(
            root / "scripts/experiments/weekly_verifier_ablation.py"
        ),
        "shared_scorer_sha256": baseline._sha256_file(
            root / "scripts/experiments/reassess_twinkl_752_5.py"
        ),
    }
    for key, digest in expected.items():
        if manifest.get(key) != digest:
            raise ValueError(f"Prepared protocol hash mismatch: {key}")
    return (*frozen, paths, manifest)


def command_prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    manifest = _prepare(config, root)
    print(
        json.dumps(
            {
                "counts": manifest["counts"],
                "planned_calls": manifest["planned_calls"],
                "estimate": manifest["estimate"],
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
        base_config,
        records,
        _cases,
        _outcomes,
        _targets,
        _episodes,
        _none_responses,
        base_metrics,
        _base_paths,
        paths,
        _manifest,
    ) = _load_prepared(config, root)
    selected = _smoke_records(records, int(config["smoke"]["prompt_count"]))
    runtime = _runtime_config(
        config,
        base_config,
        repeats=int(config["smoke"]["repeats"]),
        max_budget_usd=float(config["smoke"]["max_budget_usd"]),
    )
    execution = asyncio.run(
        _execute_calls(
            records=selected,
            config=runtime,
            output_path=paths["smoke_responses"],
        )
    )
    responses = baseline._load_jsonl(paths["smoke_responses"])
    projection = _smoke_projection(
        config=config,
        base_metrics=base_metrics,
        smoke_records=selected,
        responses=responses,
    )
    print(json.dumps({"execution": execution, "projection": projection}, indent=2))


def command_run(args: argparse.Namespace) -> None:
    if not args.execute:
        raise SystemExit("Refusing paid calls without --execute")
    from dotenv import load_dotenv

    root = Path(args.root).resolve()
    load_dotenv(root / ".env")
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    (
        base_config,
        records,
        _cases,
        _outcomes,
        _targets,
        _episodes,
        _none_responses,
        base_metrics,
        _base_paths,
        paths,
        _manifest,
    ) = _load_prepared(config, root)
    smoke = _smoke_records(records, int(config["smoke"]["prompt_count"]))
    projection = _smoke_projection(
        config=config,
        base_metrics=base_metrics,
        smoke_records=smoke,
        responses=baseline._load_jsonl(paths["smoke_responses"]),
    )
    if not projection["within_budget"]:
        raise ValueError(
            "Smoke projection with contingency exceeds the full-run budget"
        )
    result = asyncio.run(
        _execute_calls(
            records=records,
            config=_runtime_config(config, base_config),
            output_path=paths["responses"],
        )
    )
    print(json.dumps({"projection": projection, "execution": result}, indent=2))


def command_score(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    config = baseline._read_yaml(baseline._rooted(args.config, root))
    (
        base_config,
        records,
        cases,
        outcomes,
        targets,
        episodes,
        none_responses,
        base_metrics,
        base_paths,
        paths,
        manifest,
    ) = _load_prepared(config, root)
    low_responses = baseline._load_jsonl(paths["responses"])
    score_config = _score_config(config, base_config)
    none_spec = model_study._model_specs(base_config)["gpt_5_6_luna"]
    none_result, none_case_stats = model_study._score_model(
        config=score_config,
        model_spec=none_spec,
        records=records,
        cases=cases,
        outcomes=outcomes,
        targets=targets,
        episodes=episodes,
        responses=none_responses,
    )
    low_result, low_case_stats = model_study._score_model(
        config=score_config,
        model_spec=_model_spec(config),
        records=records,
        cases=cases,
        outcomes=outcomes,
        targets=targets,
        episodes=episodes,
        responses=low_responses,
    )
    comparison = reassess._comparison_bootstrap(
        case_ids=sorted(case["canonical_case_id"] for case in cases),
        first=none_case_stats,
        second=low_case_stats,
        config=score_config,
        seed_offset=2,
    )
    smoke_records = _smoke_records(records, int(config["smoke"]["prompt_count"]))
    projection = _smoke_projection(
        config=config,
        base_metrics=base_metrics,
        smoke_records=smoke_records,
        responses=baseline._load_jsonl(paths["smoke_responses"]),
    )
    low_result["usage_details"] = _usage_detail_summary(low_responses)
    cache_aware_cost = _cache_aware_cost_usd(
        low_result["usage_details"], config["api"]["pricing_usd_per_million_tokens"]
    )
    metrics = {
        "study_id": config["study_id"],
        "scored_at": datetime.now(UTC).isoformat(),
        "counts": manifest["counts"],
        "comparison_direction": "luna_low_minus_luna_none",
        "models": {"luna_none": none_result, "luna_low": low_result},
        "paired_trajectory_bootstrap": comparison,
        "development_selection": _decision(comparison),
        "smoke_projection": projection,
        "standard_rate_token_calculation_usd": float(
            low_result["response_summary"]["actual_spend_usd"]
        ),
        "cache_aware_token_calculation_usd": cache_aware_cost,
        "restrictions": config["decision"]["restrictions"],
        "provenance": {
            "manifest_sha256": baseline._sha256_file(paths["manifest"]),
            "base_manifest_sha256": baseline._sha256_file(base_paths["manifest"]),
            "prompts_sha256": baseline._sha256_file(base_paths["prompts"]),
            "luna_none_responses_sha256": baseline._sha256_file(
                base_paths["luna_none_responses"]
            ),
            "luna_low_responses_sha256": baseline._sha256_file(paths["responses"]),
            "smoke_responses_sha256": baseline._sha256_file(paths["smoke_responses"]),
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
                "standard_rate_token_calculation_usd": metrics[
                    "standard_rate_token_calculation_usd"
                ],
                "cache_aware_token_calculation_usd": metrics[
                    "cache_aware_token_calculation_usd"
                ],
                "luna_low_stability": low_result["stability"],
                "luna_low_usage_details": low_result["usage_details"],
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
    smoke.add_argument("--execute", action="store_true")
    smoke.set_defaults(func=command_smoke)
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
