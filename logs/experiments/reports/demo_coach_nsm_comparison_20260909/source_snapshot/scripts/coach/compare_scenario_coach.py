"""Freeze, generate, and install paired Coach Digests for saved demo weeks.

Dry run freezes source-bound inputs. --execute performs authorized paid calls;
--apply installs only complete validated coverage in a separate comparison file.
An interrupted request without a diagnostic always requires inspection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scripts.coach.complete_scenario_coach import (
    ROOT,
    _transport,
    _write,
    collect_cases,
)
from scripts.coach.refresh_scenario_coach import (
    INPUT_TOKEN_ALLOWANCE,
    MAXIMUM_ATTEMPTS,
    PER_REQUEST_RESERVE_USD,
    TOTAL_BUDGET_USD,
    _check_usage,
    _request_bound,
)
from src.coach.demo_comparison import (
    COMPARISONS_PATH,
    PROMPT_NAME,
    PROMPT_VERSION,
    CoachComparisonArm,
    NorthStarCoachContext,
    SavedCoachComparison,
    SavedCoachComparisonFixture,
    build_north_star_context,
    generate_demo_comparison_diagnostic,
    hash_json,
    hash_text,
    render_demo_comparison_prompt,
    validate_saved_comparison,
)
from src.coach.llm_client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_OPENAI_SERVICE_TIER,
    OPENAI_CACHE_WRITE_MULTIPLIER,
    OPENAI_LUNA_INPUT_USD_PER_MILLION,
    OPENAI_LUNA_OUTPUT_USD_PER_MILLION,
    OPENAI_LUNA_PRICING_SOURCE,
    summarize_llm_call_metrics,
)
from src.coach.schemas import (
    WEEKLY_DIGEST_COACH_RESPONSE_FORMAT,
    CoachDigestDiagnostic,
    LLMCallMetrics,
    WeeklyDigest,
)
from src.coach.weekly_digest import LLMCompleteFn
from src.demo.contracts import ContractFixtureSet
from src.demo.north_star_replay import (
    SavedExperimentRecord,
    validate_saved_record,
    verify_experiment_source,
)
from src.demo.scenarios import build_saved_north_star_request
from src.prompt_boundary import render_live_prompt_receipt

DEFAULT_OUTPUT = Path("logs/experiments/reports/demo_coach_nsm_comparison_20260909")
SOURCE_FILES = (
    "scripts/coach/compare_scenario_coach.py",
    "scripts/coach/complete_scenario_coach.py",
    "scripts/coach/refresh_scenario_coach.py",
    "src/coach/demo_comparison.py",
    "src/coach/weekly_digest.py",
    "src/coach/llm_client.py",
    "src/coach/schemas.py",
    "prompts/weekly_digest_coach.yaml",
    "prompts/demo_coach_nsm_comparison.yaml",
    "src/demo/north_star_replay_records.json",
    "src/demo/coach_digest_responses.json",
)


def collect_comparison_cases(root: Path) -> dict[str, dict[str, Any]]:
    """Use catalog-verified digests and independently validated experiment records."""
    verify_experiment_source(root)
    cases = collect_cases(root)
    fixtures: dict[str, ContractFixtureSet] = {}
    eligible = {}
    for key, case in cases.items():
        bundle = case["source_bundle_path"]
        if bundle not in fixtures:
            fixtures[bundle] = ContractFixtureSet.model_validate_json(
                (root / bundle).read_bytes()
            )
        fixture = fixtures[bundle]
        digest = WeeklyDigest.model_validate(case["digest"])
        week = next(
            w for w in fixture.scenario.weeks if w.week_start == digest.week_start
        )
        records = [
            e.details.record
            for e in fixture.trace_events
            if e.event_id in week.event_ids and e.event_type == "north_star_reviewed"
        ]
        if len(records) != 1:
            raise ValueError(f"Expected one saved North Star Moment record: {key}")
        record = validate_saved_record(
            records[0], build_saved_north_star_request(fixture, week.week_id)
        )
        if record.status != "complete" or record.selected is None:
            continue
        context = build_north_star_context(record)
        eligible[key] = {
            **case,
            "north_star_record": record.model_dump(mode="json"),
            "north_star_context": context.model_dump(mode="json", exclude_none=True),
            "base_prompts": {
                "without_north_star": render_demo_comparison_prompt(digest, None),
                "with_north_star": render_demo_comparison_prompt(digest, context),
            },
        }
    return eligible


def prepare(root: Path, output: Path) -> dict[str, Any]:
    cases = collect_comparison_cases(root)
    if not cases:
        raise ValueError("No eligible saved North Star Moment comparisons")
    maximum_reserved = len(cases) * 2 * MAXIMUM_ATTEMPTS * PER_REQUEST_RESERVE_USD
    if maximum_reserved > TOTAL_BUDGET_USD:
        raise ValueError("Maximum attempts exceed the frozen total budget")
    policy = {
        "model": DEFAULT_OPENAI_MODEL,
        "reasoning_effort": DEFAULT_OPENAI_REASONING_EFFORT,
        "service_tier": DEFAULT_OPENAI_SERVICE_TIER,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "sdk_retries": 0,
        "maximum_attempts_per_arm": MAXIMUM_ATTEMPTS,
        "per_request_reserve_usd": str(PER_REQUEST_RESERVE_USD),
        "total_budget_usd": str(TOTAL_BUDGET_USD),
        "maximum_reserved_usd": str(maximum_reserved),
        "input_token_allowance": INPUT_TOKEN_ALLOWANCE,
        "pricing_source": OPENAI_LUNA_PRICING_SOURCE,
        "input_usd_per_million": str(OPENAI_LUNA_INPUT_USD_PER_MILLION),
        "output_usd_per_million": str(OPENAI_LUNA_OUTPUT_USD_PER_MILLION),
        "cache_write_multiplier": str(OPENAI_CACHE_WRITE_MULTIPLIER),
        "prompt_name": PROMPT_NAME,
        "prompt_version": PROMPT_VERSION,
        "source_sha256": {
            name: hash_text((root / name).read_text()) for name in SOURCE_FILES
        },
    }
    plan = {
        "schema_version": "demo-coach-comparison-plan-v1",
        "policy": policy,
        "cases": cases,
    }
    path = output / "plan.json"
    if path.exists():
        if json.loads(path.read_bytes()) != plan:
            raise ValueError(
                "Frozen comparison inputs, source files, or policy changed"
            )
    else:
        _write(path, plan)
        for key, case in cases.items():
            _write(output / "inputs" / f"{key.replace('::', '_')}.json", case)
    return plan


def _state_path(output: Path, key: str, arm: str) -> Path:
    return output / "cases" / f"{key.replace('::', '_')}_{arm}.json"


def _read_diagnostic(root: Path, attempt: dict[str, Any]) -> dict[str, Any]:
    path = root / attempt["diagnostic_path"]
    if not path.exists():
        raise RuntimeError("Unresolved interrupted attempt; inspect before retry")
    saved: dict[str, Any] = json.loads(path.read_bytes())
    if hash_text(saved["prompt"]) != saved["prompt_sha256"]:
        raise ValueError("Saved diagnostic prompt changed")
    request = json.loads((root / attempt["request_path"]).read_bytes())
    if hash_json(request) != attempt["request_sha256"]:
        raise ValueError("Saved provider request changed")
    expected_settings = {
        "model": DEFAULT_OPENAI_MODEL,
        "reasoning": {"effort": DEFAULT_OPENAI_REASONING_EFFORT},
        "service_tier": DEFAULT_OPENAI_SERVICE_TIER,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "text": {"format": WEEKLY_DIGEST_COACH_RESPONSE_FORMAT},
        "store": False,
    }
    if any(request.get(key) != value for key, value in expected_settings.items()) or (
        saved["prompt"]
        != render_live_prompt_receipt(
            instructions=request["instructions"], input_data=request["input"]
        )
    ):
        raise ValueError("Saved diagnostic differs from its provider request")
    _check_usage(LLMCallMetrics.model_validate(saved["llm_call"]))
    return saved


def _validate_arm_receipts(
    root: Path,
    state: dict[str, Any],
    digest: WeeklyDigest,
) -> None:
    receipts = [_read_diagnostic(root, attempt) for attempt in state["attempts"]]
    if not receipts:
        raise ValueError("Comparison arm lacks provider receipts")
    final = receipts[-1]
    arm = CoachComparisonArm.model_validate(state["response"])
    diagnostic = CoachDigestDiagnostic.model_validate(final)
    if (
        not diagnostic.accepted
        or diagnostic.failure_stage is not None
        or diagnostic.failure_details
        or (diagnostic.persona_id, diagnostic.week_start, diagnostic.week_end)
        != (digest.persona_id, digest.week_start, digest.week_end)
        or arm.prompt != final["prompt"]
        or arm.prompt_sha256 != final["prompt_sha256"]
        or arm.repair_requirements != final["repair_requirements"]
        or arm.narrative != diagnostic.narrative
        or arm.raw_output != diagnostic.raw_output
        or arm.validation != diagnostic.validation
        or arm.call_metrics
        != [LLMCallMetrics.model_validate(row["llm_call"]) for row in receipts]
        or arm.diagnostic_paths != [row["diagnostic_path"] for row in state["attempts"]]
    ):
        raise ValueError("Staged comparison differs from accepted provider diagnostics")


async def _generate_arm(
    root: Path,
    output: Path,
    key: str,
    case: dict[str, Any],
    arm_name: str,
    complete: LLMCompleteFn,
    collected: list[LLMCallMetrics],
) -> None:
    digest = WeeklyDigest.model_validate(case["digest"])
    context = NorthStarCoachContext.model_validate(case["north_star_context"])
    arm_context = context if arm_name == "with_north_star" else None
    state_path = _state_path(output, key, arm_name)
    input_hash = hash_json(case)
    state = (
        json.loads(state_path.read_bytes())
        if state_path.exists()
        else {
            "input_sha256": input_hash,
            "attempts": [],
            "response": None,
        }
    )
    if state["input_sha256"] != input_hash:
        raise ValueError(f"Checkpoint input changed: {key}")
    if state["response"] is not None:
        _validate_arm_receipts(root, state, digest)
        return
    repair: list[str] = []
    while True:
        if state["attempts"]:
            saved = _read_diagnostic(root, state["attempts"][-1])
            diagnostic = CoachDigestDiagnostic.model_validate(saved)
            if diagnostic.accepted:
                break
            if (
                diagnostic.failure_stage
                not in {"coach_validation", "json_parse", "schema_validation"}
                or len(state["attempts"]) >= MAXIMUM_ATTEMPTS
            ):
                raise RuntimeError(f"Terminal Coach failure: {key}:{arm_name}")
            repair = diagnostic.failure_details
        number = len(state["attempts"]) + 1
        stem = f"{key.replace('::', '_')}_{arm_name}_{number}"
        diagnostic_path = output / "diagnostics" / f"{stem}.json"
        request_path = output / "requests" / f"{stem}.json"
        called = False

        async def reserved_complete(
            prompt: str,
            response_format: dict | None,
            instructions: str | None = None,
            *,
            _request_path: Path = request_path,
            _diagnostic_path: Path = diagnostic_path,
        ) -> str | None:
            nonlocal called
            if called:
                raise RuntimeError("Only one provider request is allowed per attempt")
            called = True
            request = {
                "model": DEFAULT_OPENAI_MODEL,
                "input": prompt,
                "instructions": instructions,
                "text": {"format": response_format},
                "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                "reasoning": {"effort": DEFAULT_OPENAI_REASONING_EFFORT},
                "service_tier": DEFAULT_OPENAI_SERVICE_TIER,
                "store": False,
            }
            token_bound, cost_bound = _request_bound(request)
            _write(_request_path, request)
            state["attempts"].append(
                {
                    "diagnostic_path": str(_diagnostic_path.relative_to(root)),
                    "request_path": str(_request_path.relative_to(root)),
                    "request_sha256": hash_json(request),
                    "reserved_usd": str(PER_REQUEST_RESERVE_USD),
                    "input_token_bound": token_bound,
                    "cost_bound_usd": str(cost_bound),
                }
            )
            _write(state_path, state)
            return await complete(prompt, response_format, instructions)

        before = len(collected)
        diagnostic, prompt = await generate_demo_comparison_diagnostic(
            digest, arm_context, reserved_complete, repair_requirements=repair
        )
        if len(collected) != before + 1:
            raise RuntimeError("Provider did not record exactly one attempt")
        metric = collected[-1]
        metric.call_label = f"coach_nsm_comparison:{key}:{arm_name}:attempt_{number}"
        diagnostic.llm_call = metric
        _write(
            diagnostic_path,
            {
                "prompt": prompt,
                "prompt_sha256": hash_text(prompt),
                "repair_requirements": repair,
                **diagnostic.model_dump(mode="json"),
            },
        )
    assert diagnostic.narrative is not None and diagnostic.validation is not None
    assert diagnostic.raw_output is not None
    receipts = [_read_diagnostic(root, attempt) for attempt in state["attempts"]]
    base_prompt = case["base_prompts"][arm_name]
    arm = CoachComparisonArm(
        narrative=diagnostic.narrative,
        validation=diagnostic.validation,
        base_prompt=base_prompt,
        prompt=saved["prompt"],
        repair_requirements=saved["repair_requirements"],
        raw_output=diagnostic.raw_output,
        call_metrics=[
            LLMCallMetrics.model_validate(row["llm_call"]) for row in receipts
        ],
        diagnostic_paths=[a["diagnostic_path"] for a in state["attempts"]],
        base_prompt_sha256=hash_text(base_prompt),
        prompt_sha256=saved["prompt_sha256"],
        response_sha256=hash_json(diagnostic.narrative.model_dump(mode="json")),
        raw_output_sha256=hash_text(diagnostic.raw_output),
    )
    state["response"] = arm.model_dump(mode="json")
    _write(state_path, state)
    print(f"Accepted {key}:{arm_name} ({len(receipts)} attempt(s))", flush=True)


async def generate(
    root: Path,
    output: Path,
    plan: dict[str, Any],
    *,
    llm_complete: LLMCompleteFn | None = None,
    metrics: list[LLMCallMetrics] | None = None,
) -> None:
    """Personas run concurrently; each Persona's weeks and arms remain sequential."""
    if prepare(root, output) != plan:
        raise ValueError("Supplied plan differs from frozen checkpoint")
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for key, case in plan["cases"].items():
        groups.setdefault(case["digest"]["persona_id"], []).append((key, case))

    async def persona(cases: list[tuple[str, dict[str, Any]]]) -> None:
        collected = metrics if metrics is not None else []
        complete = llm_complete or _transport(collected)
        for key, case in cases:
            for arm in ("without_north_star", "with_north_star"):
                await _generate_arm(root, output, key, case, arm, complete, collected)

    if llm_complete is not None:
        # Injection uses a single deterministic test transport and metrics list.
        for cases in groups.values():
            await persona(cases)
    else:
        results = await asyncio.gather(
            *(persona(cases) for cases in groups.values()), return_exceptions=True
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError("; ".join(str(error) for error in failures))


def apply(
    root: Path, output: Path, plan: dict[str, Any]
) -> SavedCoachComparisonFixture:
    """Install only a complete comparison; preserve all original response receipts."""
    if prepare(root, output) != plan:
        raise ValueError("Supplied plan differs from frozen checkpoint")
    comparisons = {}
    for key, case in plan["cases"].items():
        arms = {}
        for arm in ("without_north_star", "with_north_star"):
            path = _state_path(output, key, arm)
            state = json.loads(path.read_bytes()) if path.exists() else {}
            if state.get("response") is None:
                raise RuntimeError(f"Comparison is incomplete: {key}:{arm}")
            if state["input_sha256"] != hash_json(case):
                raise ValueError("Checkpoint input changed")
            _validate_arm_receipts(
                root, state, WeeklyDigest.model_validate(case["digest"])
            )
            arms[arm] = state["response"]
        digest = WeeklyDigest.model_validate(case["digest"])
        record = SavedExperimentRecord.model_validate(case["north_star_record"])
        comparison = SavedCoachComparison.model_validate(
            {
                "scenario_id": case["scenario_id"],
                "persona_id": digest.persona_id,
                "week_start": digest.week_start,
                "week_end": digest.week_end,
                "weekly_drift_input_sha256": case["input_sha256"],
                "north_star_input_hash": record.input_hash,
                "north_star_context": case["north_star_context"],
                "north_star_context_sha256": hash_json(case["north_star_context"]),
                **arms,
            }
        )
        comparisons[key] = validate_saved_comparison(
            comparison, digest, record, case["scenario_id"]
        )
    fixture = SavedCoachComparisonFixture(comparisons=comparisons)
    payload = fixture.model_dump(mode="json")
    _write(output / "coach_digest_comparisons.json", payload)
    _write(root / COMPARISONS_PATH, payload)
    return fixture


def write_report(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    states = [
        json.loads(path.read_bytes()) for path in (output / "cases").glob("*.json")
    ]
    metrics = [
        LLMCallMetrics.model_validate(json.loads(path.read_bytes())["llm_call"])
        for path in (output / "diagnostics").glob("*.json")
    ]
    summary = {
        "eligible_pairs": len(plan["cases"]),
        "expected_arms": 2 * len(plan["cases"]),
        "accepted_arms": sum(state.get("response") is not None for state in states),
        "reserved_usd": str(
            sum(
                (
                    Decimal(a["reserved_usd"])
                    for state in states
                    for a in state["attempts"]
                ),
                Decimal(0),
            )
        ),
        "maximum_reserved_usd": plan["policy"]["maximum_reserved_usd"],
        "usage": summarize_llm_call_metrics(metrics),
        "limitations": "Paired synthetic demonstration with deterministic validation; "
        "no human validation or evidence of user benefit. Initial requests differ "
        "only in north_star_context; validation retries retain separate receipts.",
    }
    _write(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT)
    plan = prepare(ROOT, output)
    print(
        f"Frozen {len(plan['cases'])} pairs; maximum reserve "
        f"US${plan['policy']['maximum_reserved_usd']}.",
        flush=True,
    )
    try:
        if args.execute:
            asyncio.run(generate(ROOT, output, plan))
        if args.apply:
            apply(ROOT, output, plan)
    finally:
        write_report(output, plan)


if __name__ == "__main__":
    main()
