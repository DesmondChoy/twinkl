"""Compare GPT-5.6 and GPT-6 Luna at medium through xhigh reasoning.

The model and explicit reasoning effort are the only provider request changes.
``prepare`` and ``score`` are offline; paid ``smoke`` and ``run`` need --execute.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from prompts import get_prompt_metadata
from scripts.experiments import compare_twinkl_52zz_luna_reasoning as cost_study
from scripts.experiments import compare_twinkl_52zz_models as model_study
from scripts.experiments import reassess_twinkl_752_5 as reassess
from scripts.experiments import weekly_drift_definitions as frozen
from scripts.experiments import weekly_verifier_ablation as baseline
from src import weekly_drift_reviewer as reviewer
from src.model_guardrails import openai_response_refusal

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/evals/twinkl_gv3x_luna_higher_v1.yaml"
FROZEN_CODE = (
    "scripts/experiments/compare_twinkl_gv3x_luna_higher.py",
    "scripts/experiments/weekly_drift_definitions.py",
    "scripts/experiments/compare_twinkl_52zz_models.py",
    "scripts/experiments/reassess_twinkl_752_5.py",
    "scripts/experiments/weekly_verifier_ablation.py",
    "src/weekly_drift_reviewer.py",
    "src/drift_detector.py",
    "src/drift_rules.py",
    "src/prompt_boundary.py",
    "prompts/weekly_vif_verifier.yaml",
    "config/schwartz_values.yaml",
)


def _load_config(path: Path) -> dict[str, Any]:
    return baseline._read_yaml(path)


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    output = ROOT / config["output_dir"]
    return {
        "output": output,
        "source": ROOT / config["source_requests"],
        "reference": ROOT / config["reference_config"],
        "manifest": output / "manifest.json",
        "attempts": output / "attempts.jsonl",
        "responses": output / "responses.jsonl",
        "metrics": output / "metrics.json",
    }


def _current_rows(source: Path, arms: dict[str, Any]) -> list[dict]:
    rows = frozen.read_rows(source)
    schema = reviewer.WeeklyVerifierResponse.model_json_schema()
    version = get_prompt_metadata(reviewer.WEEKLY_DRIFT_REVIEWER_PROMPT)["version"]
    if str(version) != "4.0":
        raise ValueError("Current Weekly Drift Reviewer prompt is no longer 4.0")
    for row in rows:
        request = reviewer.WeeklyDriftReviewerRequest.model_validate(row["request"])
        payload = row["variants"]["definitions"]
        if (
            payload["instructions"] != request.instructions
            or payload["input_data"] != request.input_data
            or payload["schema"] != schema
            or str(payload["prompt_version"]) != str(version)
            or frozen.digest(
                {k: v for k, v in payload.items() if k != "request_sha256"}
            )
            != payload["request_sha256"]
        ):
            raise ValueError(
                f"Frozen v4 request no longer matches runtime: {row['case_id']}"
            )
        row["variants"] = {arm: payload for arm in arms}
    if len(rows) != 951 or len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Expected 951 unique frozen Persona weeks")
    return rows


def prepare(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    paths = _paths(config)
    if paths["manifest"].exists():
        return verify(config_path)[0]
    rows = _current_rows(paths["source"], config["models"])
    reference = _load_config(paths["reference"])
    cases, _outcomes, _targets, episodes = model_study._load_complete_development(
        reference, ROOT
    )
    if (
        len({case["persona_id"] for case in cases})
        != config["study"]["expected_personas"]
        or len(cases) != config["study"]["expected_cases"]
        or episodes.height != config["study"]["expected_drifts"]
    ):
        raise ValueError("Development reference counts changed")
    reference_paths = model_study._source_paths(reference, ROOT)
    manifest = {
        "study_id": config["study_id"],
        "issue": config["issue"],
        "prepared_at": datetime.now(UTC).isoformat(),
        "repo_head": baseline._git_head(ROOT),
        "config_sha256": frozen.file_hash(config_path),
        "source_requests_sha256": frozen.file_hash(paths["source"]),
        "reference_config_sha256": frozen.file_hash(paths["reference"]),
        "reference_source_sha256": {
            key: frozen.file_hash(path) for key, path in reference_paths.items()
        },
        "reference_cases_sha256": frozen.digest(cases),
        "code_sha256": {name: frozen.file_hash(ROOT / name) for name in FROZEN_CODE},
        "prompt_version": "4.0",
        "prompt_sha256s": [row["request"]["prompt_sha256"] for row in rows],
        "request_sha256s": [
            row["variants"][next(iter(config["models"]))]["request_sha256"]
            for row in rows
        ],
        "models": config["models"],
        "settings": config["api"],
        "repeats": config["study"]["repeats"],
        "personas": config["study"]["expected_personas"],
        "weeks": len(rows),
        "expected_terminal_requests": len(rows)
        * config["study"]["repeats"]
        * len(config["models"]),
        "comparison": config["comparison"],
    }
    paths["output"].mkdir(parents=True, exist_ok=False)
    frozen.write_json(paths["manifest"], manifest)
    return manifest


def verify(
    config_path: Path,
) -> tuple[dict[str, Any], list[dict], dict[str, Any], dict[str, Path]]:
    config = _load_config(config_path)
    paths = _paths(config)
    manifest = json.loads(paths["manifest"].read_text())
    if frozen.file_hash(config_path) != manifest["config_sha256"]:
        raise ValueError("Experiment config changed after preparation")
    if frozen.file_hash(paths["source"]) != manifest["source_requests_sha256"]:
        raise ValueError("Frozen request source changed")
    if frozen.file_hash(paths["reference"]) != manifest["reference_config_sha256"]:
        raise ValueError("Reference config changed")
    reference = _load_config(paths["reference"])
    for key, path in model_study._source_paths(reference, ROOT).items():
        if frozen.file_hash(path) != manifest["reference_source_sha256"][key]:
            raise ValueError(f"Reference source changed: {key}")
    cases, _outcomes, _targets, _episodes = model_study._load_complete_development(
        reference, ROOT
    )
    if frozen.digest(cases) != manifest["reference_cases_sha256"]:
        raise ValueError("Reference case content changed")
    for name, expected in manifest["code_sha256"].items():
        if frozen.file_hash(ROOT / name) != expected:
            raise ValueError(f"Frozen code changed: {name}")
    rows = _current_rows(paths["source"], config["models"])
    if [row["request"]["prompt_sha256"] for row in rows] != manifest["prompt_sha256s"]:
        raise ValueError("Prompt hashes changed")
    if [
        row["variants"][next(iter(config["models"]))]["request_sha256"] for row in rows
    ] != manifest["request_sha256s"]:
        raise ValueError("Request identity changed")
    return manifest, rows, config, paths


def _jobs(
    rows: list[dict], config: dict, *, smoke: bool
) -> list[tuple[dict, str, int]]:
    if smoke:
        ordered = sorted(
            rows,
            key=lambda row: len(
                row["variants"][next(iter(config["models"]))]["input_data"]
            ),
        )
        count = config["study"]["smoke_cases"]
        selected = [
            ordered[i * (len(ordered) - 1) // (count - 1)] for i in range(count)
        ]
        jobs = [(row, arm, 1) for row in selected for arm in config["models"]]
    else:
        jobs = [
            (row, arm, repeat)
            for row in rows
            for repeat in range(1, config["study"]["repeats"] + 1)
            for arm in config["models"]
        ]
    random.Random(config["api"]["schedule_seed"]).shuffle(jobs)
    return jobs


def _attempt_cost(attempt: dict, pricing: dict) -> float:
    usage = cost_study._usage_detail_summary([attempt])
    return cost_study._cache_aware_cost_usd(usage, pricing)


async def _execute_request(
    client: Any,
    row: dict,
    arm: str,
    repeat: int,
    settings: dict,
    prior: list[dict],
    append: Any,
) -> dict:
    key = frozen.request_key(row["case_id"], arm, repeat)
    payload = row["variants"][arm]
    request = reviewer.WeeklyDriftReviewerRequest.model_validate(row["request"])
    if prior and (
        prior[-1]["event"] == "started" or not prior[-1].get("retryable", False)
    ):
        return frozen.terminal_result(row, arm, repeat, prior)
    for number in range(len(prior) + 1, settings["max_attempts"] + 1):
        reservation = {
            "event": "started",
            "request_key": key,
            "request_sha256": payload["request_sha256"],
            "attempt_number": number,
            "started_at": frozen.now(),
        }
        append(reservation)
        started = time.monotonic()
        attempt = dict(reservation, event="finished", retryable=False)
        try:
            response = await client.responses.parse(
                model=settings["model"],
                instructions=payload["instructions"],
                input=payload["input_data"],
                text_format=reviewer.WeeklyVerifierResponse,
                reasoning={"effort": settings["reasoning_effort"]},
                max_output_tokens=settings["max_output_tokens"],
                store=settings["store"],
                service_tier=settings["service_tier"],
                timeout=settings["timeout_seconds"],
            )
            parsed = getattr(response, "output_parsed", None)
            refusal = openai_response_refusal(response)
            attempt.update(
                resolved_model=getattr(response, "model", None),
                response_id=getattr(response, "id", None),
                provider_request_id=getattr(response, "_request_id", None),
                provider_status=getattr(response, "status", None),
                raw_text=getattr(response, "output_text", None),
                usage=frozen.jsonable(getattr(response, "usage", None)),
            )
            if getattr(response, "status", None) != "completed":
                attempt.update(
                    status="invalid",
                    validation_error="Provider response was not completed",
                )
            elif refusal is not None or not isinstance(
                parsed, reviewer.WeeklyVerifierResponse
            ):
                attempt.update(status="refusal", refusal=refusal)
            else:
                attempt["parsed"] = parsed.model_dump(mode="json")
                try:
                    reviewer.validate_weekly_drift_reviewer_response(parsed, request)
                except ValueError as error:
                    attempt.update(status="invalid", validation_error=str(error))
                else:
                    attempt["status"] = "ok"
        except Exception as error:  # noqa: BLE001 - provider boundary
            attempt.update(
                status="error",
                error_type=type(error).__name__,
                retryable=reviewer._is_transient_error(error),
            )
        attempt.update(
            completed_at=frozen.now(), latency_seconds=time.monotonic() - started
        )
        append(attempt)
        prior.append(attempt)
        if not attempt["retryable"]:
            break
        if number < settings["max_attempts"]:
            await asyncio.sleep(2 ** (number - 1))
    return frozen.terminal_result(row, arm, repeat, prior)


async def run(config_path: Path, *, smoke: bool) -> dict[str, Any]:
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    manifest, rows, config, paths = verify(config_path)
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    events = frozen.read_rows(paths["attempts"])
    attempts = frozen.prior_attempts(events)
    completed_rows = frozen.read_rows(paths["responses"])
    completed = {row["request_key"]: row for row in completed_rows}
    if len(completed) != len(completed_rows):
        raise ValueError("Duplicate terminal result")
    spend = sum(
        _attempt_cost(
            event,
            config["models"][event["request_key"].split(":", 1)[0]][
                "pricing_usd_per_million_tokens"
            ],
        )
        for event in events
        if event["event"] == "finished" and event.get("usage")
    )
    if spend >= config["api"]["max_calculated_spend_usd"]:
        raise ValueError("Calculated spend already meets experiment cap")
    pending = [
        job
        for job in _jobs(rows, config, smoke=smoke)
        if frozen.request_key(job[0]["case_id"], job[1], job[2]) not in completed
    ]
    queue: asyncio.Queue[tuple[dict, str, int]] = asyncio.Queue()
    for job in pending:
        queue.put_nowait(job)
    statuses: Counter[str] = Counter()
    completed_here = 0
    with (
        paths["attempts"].open("a") as journal,
        paths["responses"].open("a") as results,
    ):

        def append(event: dict) -> None:
            nonlocal spend
            journal.write(json.dumps(event, ensure_ascii=False) + "\n")
            journal.flush()
            if event["event"] == "finished" and event.get("usage"):
                arm = event["request_key"].split(":", 1)[0]
                spend += _attempt_cost(
                    event, config["models"][arm]["pricing_usd_per_million_tokens"]
                )

        async with AsyncOpenAI(max_retries=0) as client:

            async def worker() -> None:
                nonlocal completed_here, spend
                while not queue.empty():
                    if spend >= config["api"]["max_calculated_spend_usd"]:
                        return
                    row, arm, repeat = queue.get_nowait()
                    key = frozen.request_key(row["case_id"], arm, repeat)
                    model = config["models"][arm]
                    settings = {
                        **config["api"],
                        "model": model["model"],
                        "reasoning_effort": model["reasoning_effort"],
                        "max_output_tokens": model["max_output_tokens"],
                    }
                    result = await _execute_request(
                        client,
                        row,
                        arm,
                        repeat,
                        settings,
                        attempts.get(key, []),
                        append,
                    )
                    results.write(json.dumps(result, ensure_ascii=False) + "\n")
                    results.flush()
                    completed_here += 1
                    statuses[f"{arm}:{result['status']}"] += 1
                    if completed_here % 100 == 0 or completed_here == len(pending):
                        print(
                            json.dumps(
                                {
                                    "completed": len(completed) + completed_here,
                                    "expected": manifest["expected_terminal_requests"],
                                    "calculated_spend_usd": round(spend, 4),
                                    "new_statuses": dict(statuses),
                                }
                            ),
                            flush=True,
                        )
                    queue.task_done()

            await asyncio.gather(
                *(worker() for _ in range(config["api"]["concurrency"]))
            )
    return {
        "processed": completed_here,
        "total_completed": len(completed) + completed_here,
        "calculated_spend_usd": spend,
        "statuses": dict(statuses),
    }


def score(config_path: Path) -> dict[str, Any]:
    manifest, rows, config, paths = verify(config_path)
    responses = frozen.read_rows(paths["responses"])
    expected = {
        frozen.request_key(row["case_id"], arm, repeat)
        for row in rows
        for arm in config["models"]
        for repeat in range(1, config["study"]["repeats"] + 1)
    }
    if (
        len(responses) != len(expected)
        or {r["request_key"] for r in responses} != expected
    ):
        raise ValueError("Scoring requires every terminal request exactly once")
    row_map = {row["case_id"]: row for row in rows}
    reference = _load_config(paths["reference"])
    cases, _outcomes, targets, episodes = model_study._load_complete_development(
        reference, ROOT
    )
    reference_rows = reassess._reference_rows(episodes)
    case_ids = sorted(case["canonical_case_id"] for case in cases)
    cells = model_study._entry_cells(targets)
    predictions: dict[str, dict] = {arm: {} for arm in config["models"]}
    response_statuses: dict[str, Counter] = defaultdict(Counter)
    for response in responses:
        arm, repeat = response["variant"], response["repeat"]
        row = row_map[response["case_id"]]
        if (
            response["request_key"] != frozen.request_key(row["case_id"], arm, repeat)
            or response["request_sha256"] != row["variants"][arm]["request_sha256"]
        ):
            raise ValueError("Response request identity mismatch")
        request = reviewer.WeeklyDriftReviewerRequest.model_validate(row["request"])
        decisions = [
            reviewer.WeeklyDriftReviewerDecision.model_validate(item)
            for item in response["decisions"]
        ]
        if {
            (d.t_index, d.core_value) for d in decisions
        } != request.expected_coordinates:
            raise ValueError("Incomplete effective decisions")
        response_statuses[arm][response["status"]] += 1
        if response["status"] != "ok":
            continue
        for decision in decisions:
            key = (
                model_study.WEEKLY_WITHOUT,
                repeat,
                decision.persona_id,
                decision.t_index,
                decision.core_value,
            )
            if key in predictions[arm]:
                raise ValueError(f"Duplicate assessment: {key}")
            predictions[arm][key] = baseline.VerifierAssessment.model_validate(
                {
                    "t_index": decision.t_index,
                    "dimension": decision.core_value,
                    "verdict": decision.verdict,
                    "confidence": decision.confidence,
                    "reason_code": decision.reason_code,
                    "evidence_quote": decision.evidence_quote,
                }
            )
    events = frozen.read_rows(paths["attempts"])
    attempts = frozen.prior_attempts(events)
    for response in responses:
        key = response["request_key"]
        if key not in attempts:
            raise ValueError(f"No attempt for {key}")
        model = config["models"][response["variant"]]["model"]
        for attempt in attempts[key]:
            if attempt["event"] == "finished" and attempt.get("resolved_model") not in (
                None,
                model,
            ):
                raise ValueError(f"Provider returned unexpected model for {key}")
    model_results = {}
    case_stats: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for arm, model in config["models"].items():
        runs = []
        case_stats[arm] = {}
        for repeat in range(1, config["study"]["repeats"] + 1):
            predicted, covered = reassess._setup_predictions(
                cases=cases,
                records=[],
                predictions=predictions[arm],
                setup=model_study.WEEKLY_WITHOUT,
                repeat=repeat,
            )
            metrics, stats = reassess._score_subset(
                case_ids=set(case_ids),
                reference_rows=reference_rows,
                predicted_rows=predicted,
                covered=covered,
                max_confirmation_lag=config["study"]["max_confirmation_lag_entries"],
            )
            assessments = [
                predictions[arm].get(
                    (
                        model_study.WEEKLY_WITHOUT,
                        repeat,
                        cell.persona_id,
                        cell.t_index,
                        cell.dimension,
                    )
                )
                for cell in cells
            ]
            entry_predictions = baseline._confidence_predictions(assessments, "low")
            runs.append(
                {
                    "repeat": repeat,
                    **metrics,
                    "entry": baseline._entry_metric_bundle(cells, entry_predictions),
                }
            )
            case_stats[arm][repeat] = stats
        arm_attempts = [
            event
            for event in events
            if event["event"] == "finished"
            and event["request_key"].startswith(arm + ":")
        ]
        usage = cost_study._usage_detail_summary(arm_attempts)
        latencies = [event["latency_seconds"] for event in arm_attempts]
        model_results[arm] = {
            "model": model["model"],
            "reasoning_effort": model["reasoning_effort"],
            "runs": runs,
            "response_statuses": dict(response_statuses[arm]),
            "usage": usage,
            "calculated_cost_usd": cost_study._cache_aware_cost_usd(
                usage, model["pricing_usd_per_million_tokens"]
            ),
            "median_attempt_latency_seconds": statistics.median(latencies),
        }
    comparisons = {
        effort: reassess._comparison_bootstrap(
            case_ids=case_ids,
            first=case_stats[f"gpt_5_6_{effort}"],
            second=case_stats[f"gpt_6_{effort}"],
            config=config,
            seed_offset=offset,
        )
        for offset, effort in enumerate(("medium", "high", "xhigh"), start=1)
    }
    metrics = {
        "study_id": config["study_id"],
        "scored_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": frozen.file_hash(paths["manifest"]),
        "responses_sha256": frozen.file_hash(paths["responses"]),
        "attempts_sha256": frozen.file_hash(paths["attempts"]),
        "reference_count": len(reference_rows),
        "models": model_results,
        "gpt_6_minus_gpt_5_6": comparisons,
        "limitations": config["comparison"]["limitations"],
    }
    frozen.write_json(paths["metrics"], metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "verify", "smoke", "run", "score")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.config)
    elif args.command == "verify":
        manifest, _rows, _config, _paths = verify(args.config)
        result = {
            "status": "verified",
            "planned_calls": manifest["expected_terminal_requests"],
        }
    elif args.command in ("smoke", "run"):
        if not args.execute:
            raise SystemExit("Refusing paid calls without --execute")
        result = asyncio.run(run(args.config, smoke=args.command == "smoke"))
    else:
        result = score(args.config)
    print(
        json.dumps(
            result
            if args.command != "score"
            else {
                "models": {
                    key: value["runs"] for key, value in result["models"].items()
                },
                "gpt_6_minus_gpt_5_6": result["gpt_6_minus_gpt_5_6"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
