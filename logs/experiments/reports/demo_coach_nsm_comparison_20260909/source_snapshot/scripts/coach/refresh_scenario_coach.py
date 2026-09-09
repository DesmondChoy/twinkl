"""Refresh every saved Persona Coach Digest with bounded, resumable generation.

Dry run freezes inputs, prompts, policy, and the original provider receipts.
Add --execute only for an authorized paid run. Accepted responses are staged;
--apply replaces the active fixture only after every case passes validation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from prompts import get_prompt_metadata
from scripts.coach.complete_scenario_coach import (
    ROOT,
    _hash,
    _transport,
    _write,
    collect_cases,
)
from src.coach.llm_client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_OPENAI_SERVICE_TIER,
    OPENAI_CACHE_WRITE_MULTIPLIER,
    OPENAI_LONG_CONTEXT_THRESHOLD,
    OPENAI_LUNA_INPUT_USD_PER_MILLION,
    OPENAI_LUNA_OUTPUT_USD_PER_MILLION,
    OPENAI_LUNA_PRICING_SOURCE,
    summarize_llm_call_metrics,
)
from src.coach.schemas import CoachDigestDiagnostic, LLMCallMetrics, WeeklyDigest
from src.coach.weekly_digest import (
    LLMCompleteFn,
    attach_coach_artifacts,
    generate_weekly_digest_coach_diagnostic,
    render_digest_prompt,
    validate_weekly_digest_narrative,
)
from src.demo.scenarios import (
    COACH_RESPONSES_PATH,
    SavedCoachResponse,
    SavedCoachResponseFixture,
    _coach_unavailable_reason,
    load_saved_coach_responses,
)

DEFAULT_OUTPUT = Path("logs/experiments/reports/coach_voice_refresh_20260909")
MAXIMUM_ATTEMPTS = 4
PER_REQUEST_RESERVE_USD = Decimal("0.018")
TOTAL_BUDGET_USD = Decimal("5.00")
# One UTF-8 byte per token is deliberately conservative; allowance covers the
# provider's message/schema framing beyond the complete serialized request.
INPUT_TOKEN_ALLOWANCE = 8192


def prepare(
    root: Path,
    output: Path,
    *,
    prior_run: Path | None = None,
    repair_requirements: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Freeze all source inputs and preserve the original response file exactly."""
    cases = collect_cases(root)
    original_bytes = (root / COACH_RESPONSES_PATH).read_bytes()
    fixture = load_saved_coach_responses(root)
    if set(fixture.responses) != set(cases):
        raise ValueError("Refresh requires an existing response for every current week")
    for key, case in cases.items():
        digest = WeeklyDigest.model_validate(case["digest"])
        if _coach_unavailable_reason(fixture.responses[key], digest) is not None:
            raise ValueError(
                f"Saved response is incompatible with current input: {key}"
            )
        case["prompt_sha256"] = hashlib.sha256(
            render_digest_prompt(digest).encode()
        ).hexdigest()
    retained: dict[str, Any] = {}
    prior: dict[str, Any] | None = None
    if prior_run is not None:
        prior_run = prior_run.resolve()
        if prior_run == output.resolve():
            raise ValueError("Repair output must differ from its prior run")
        if (
            not isinstance(repair_requirements, dict)
            or not repair_requirements
            or any(
                key not in cases
                or not isinstance(items, list)
                or not items
                or any(not isinstance(item, str) or not item.strip() for item in items)
                for key, items in repair_requirements.items()
            )
        ):
            raise ValueError(
                "Repair requirements must map current case keys to text lists"
            )
        prior_bytes = (prior_run / "plan.json").read_bytes()
        prior_plan = json.loads(prior_bytes)
        if prior_plan["cases"] != cases:
            raise ValueError(
                "Prior run inputs or base prompts differ from current cases"
            )
        prior = {
            "path": str(prior_run.relative_to(root)),
            "plan_sha256": hashlib.sha256(prior_bytes).hexdigest(),
            "state_sha256": {},
        }
        for key, case in cases.items():
            state_bytes = _state_path(prior_run, key).read_bytes()
            state = json.loads(state_bytes)
            if state.get("response") is None:
                raise ValueError(f"Prior run is incomplete: {key}")
            response = SavedCoachResponse.model_validate(state["response"])
            digest = WeeklyDigest.model_validate(case["digest"])
            if (
                _coach_unavailable_reason(response, digest) is not None
                or response.generation is None
                or _hash(response.narrative.model_dump(mode="json"))
                != response.generation.response_sha256
                or not validate_weekly_digest_narrative(
                    digest, response.narrative, validate_voice=True
                ).all_passed
            ):
                raise ValueError(f"Prior response is incompatible or invalid: {key}")
            prior["state_sha256"][key] = hashlib.sha256(state_bytes).hexdigest()
            if key not in repair_requirements:
                retained[key] = state["response"]
    elif repair_requirements:
        raise ValueError("Editorial repairs require a prior accepted run")
    maximum_reserved = (
        (len(cases) - len(retained)) * MAXIMUM_ATTEMPTS * PER_REQUEST_RESERVE_USD
    )
    if maximum_reserved > TOTAL_BUDGET_USD:
        raise ValueError("All reserved attempts would exceed the total USD budget")
    policy = {
        "model": DEFAULT_OPENAI_MODEL,
        "reasoning": DEFAULT_OPENAI_REASONING_EFFORT,
        "service_tier": DEFAULT_OPENAI_SERVICE_TIER,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "sdk_retries": 0,
        "maximum_attempts": MAXIMUM_ATTEMPTS,
        "per_request_reserve_usd": str(PER_REQUEST_RESERVE_USD),
        "total_budget_usd": str(TOTAL_BUDGET_USD),
        "maximum_reserved_usd": str(maximum_reserved),
        "input_token_allowance": INPUT_TOKEN_ALLOWANCE,
        "pricing_source": OPENAI_LUNA_PRICING_SOURCE,
        "input_usd_per_million": str(OPENAI_LUNA_INPUT_USD_PER_MILLION),
        "output_usd_per_million": str(OPENAI_LUNA_OUTPUT_USD_PER_MILLION),
        "cache_write_multiplier": str(OPENAI_CACHE_WRITE_MULTIPLIER),
        "prompt": get_prompt_metadata("weekly_digest_coach"),
        "code_sha256": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in (
                "scripts/coach/refresh_scenario_coach.py",
                "scripts/coach/complete_scenario_coach.py",
                "src/coach/weekly_digest.py",
                "src/coach/llm_client.py",
                "src/coach/schemas.py",
                "src/demo/scenarios.py",
                "prompts/weekly_digest_coach.yaml",
            )
        },
    }
    if prior is not None:
        policy["prior_run"] = prior
        policy["repair_requirements"] = repair_requirements
    plan_path = output / "plan.json"
    if plan_path.exists():
        plan: dict[str, Any] = json.loads(plan_path.read_bytes())
        if (
            plan["policy"] != policy
            or plan["cases"] != cases
            or plan.get("retained_responses", {}) != retained
        ):
            raise ValueError("Frozen generation policy, prompts, or inputs changed")
        backup = (output / "original_coach_digest_responses.json").read_bytes()
        if hashlib.sha256(backup).hexdigest() != plan["original_sha256"]:
            raise ValueError("Original provider receipt backup changed")
        if hashlib.sha256(original_bytes).hexdigest() != plan["original_sha256"]:
            replacement = output / "refreshed_coach_digest_responses.json"
            if not replacement.exists() or original_bytes != replacement.read_bytes():
                raise ValueError("Active saved responses changed outside this refresh")
        return plan
    output.mkdir(parents=True, exist_ok=True)
    backup_path = output / "original_coach_digest_responses.json"
    if backup_path.exists() and backup_path.read_bytes() != original_bytes:
        raise ValueError("Refusing to overwrite an earlier receipt backup")
    backup_path.write_bytes(original_bytes)
    plan = {
        "policy": policy,
        "cases": cases,
        "original_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "retained_responses": retained,
    }
    _write(plan_path, plan)
    for key, case in cases.items():
        _write(output / "inputs" / f"{key.replace('::', '_')}.json", case)
    return plan


def _prepare_for_plan(root: Path, output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    prior = plan["policy"].get("prior_run")
    return prepare(
        root,
        output,
        prior_run=root / prior["path"] if prior else None,
        repair_requirements=plan["policy"].get("repair_requirements"),
    )


def _request_bound(request: dict[str, Any]) -> tuple[int, Decimal]:
    input_token_bound = len(json.dumps(request, ensure_ascii=False).encode())
    input_token_bound += INPUT_TOKEN_ALLOWANCE
    if input_token_bound > OPENAI_LONG_CONTEXT_THRESHOLD:
        raise ValueError("Request exceeds the frozen short-context pricing bound")
    bound = (
        Decimal(input_token_bound)
        * OPENAI_LUNA_INPUT_USD_PER_MILLION
        * OPENAI_CACHE_WRITE_MULTIPLIER
        + Decimal(DEFAULT_MAX_OUTPUT_TOKENS) * OPENAI_LUNA_OUTPUT_USD_PER_MILLION
    ) / Decimal(1_000_000)
    if bound > PER_REQUEST_RESERVE_USD:
        raise ValueError("Request exceeds its reserved USD ceiling")
    return input_token_bound, bound


def _state_path(output: Path, key: str) -> Path:
    return output / "cases" / f"{key.replace('::', '_')}.json"


def _load_diagnostic(root: Path, attempt: dict[str, Any]) -> dict[str, Any]:
    path = root / attempt["diagnostic_path"]
    if not path.exists():
        raise RuntimeError("Unresolved interrupted attempt; inspect before retry")
    diagnostic: dict[str, Any] = json.loads(path.read_bytes())
    return diagnostic


def _check_usage(metric: LLMCallMetrics) -> None:
    if metric.calculated_cost_usd is None:
        raise RuntimeError(
            "Provider cost is unknown; inspect the receipt before continuing"
        )
    if Decimal(str(metric.calculated_cost_usd)) > PER_REQUEST_RESERVE_USD:
        raise RuntimeError("Provider usage exceeded its reserved USD ceiling")


def _reserved_transport(
    root: Path,
    output: Path,
    state_path: Path,
    state: dict[str, Any],
    stem: str,
    diagnostic_path: Path,
    complete: LLMCompleteFn,
) -> LLMCompleteFn:
    called = False

    async def reserved_complete(
        prompt: str,
        response_format: dict | None,
        instructions: str | None = None,
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
        request_path = output / "requests" / f"{stem}.json"
        _write(request_path, request)
        state["attempts"].append(
            {
                "diagnostic_path": str(diagnostic_path.relative_to(root)),
                "request_path": str(request_path.relative_to(root)),
                "request_sha256": _hash(request),
                "reserved_usd": str(PER_REQUEST_RESERVE_USD),
                "input_token_bound": token_bound,
                "cost_bound_usd": str(cost_bound),
            }
        )
        _write(state_path, state)
        return await complete(prompt, response_format, instructions)

    return reserved_complete


async def generate(
    root: Path,
    output: Path,
    plan: dict[str, Any],
    *,
    llm_complete: LLMCompleteFn | None = None,
    metrics: list[LLMCallMetrics] | None = None,
) -> None:
    """Stage accepted responses; never replace the active saved response fixture."""
    if _prepare_for_plan(root, output, plan) != plan:
        raise ValueError("The supplied plan differs from its frozen checkpoint")
    collected = metrics if metrics is not None else []
    complete = llm_complete or _transport(collected)
    for key, case in plan["cases"].items():
        digest = WeeklyDigest.model_validate(case["digest"])
        state_path = _state_path(output, key)
        state = (
            json.loads(state_path.read_bytes())
            if state_path.exists()
            else {
                "input_sha256": case["input_sha256"],
                "attempts": [],
                "response": plan.get("retained_responses", {}).get(key),
            }
        )
        if state["input_sha256"] != case["input_sha256"]:
            raise ValueError(f"Checkpoint input changed: {key}")
        if key in plan.get("retained_responses", {}) and (
            state.get("response") != plan["retained_responses"][key]
        ):
            raise ValueError(f"Retained response changed: {key}")
        if state.get("response") is not None:
            if not state_path.exists():
                _write(state_path, state)
            continue
        initial_repair = plan["policy"].get("repair_requirements", {}).get(key, [])
        repair: list[str] = initial_repair
        while True:
            if state["attempts"]:
                saved = _load_diagnostic(root, state["attempts"][-1])
                diagnostic = CoachDigestDiagnostic.model_validate(saved)
                _check_usage(LLMCallMetrics.model_validate(saved["llm_call"]))
                if diagnostic.accepted:
                    break
                if (
                    diagnostic.failure_stage
                    not in {"coach_validation", "json_parse", "schema_validation"}
                    or len(state["attempts"]) >= MAXIMUM_ATTEMPTS
                ):
                    raise RuntimeError(
                        f"Terminal Coach failure: {key}: {diagnostic.failure_stage}"
                    )
                repair = [*initial_repair, *diagnostic.failure_details]
            attempt_number = len(state["attempts"]) + 1
            stem = f"{key.replace('::', '_')}_{attempt_number}"
            diagnostic_path = output / "diagnostics" / f"{stem}.json"
            before = len(collected)
            reserved_complete = _reserved_transport(
                root, output, state_path, state, stem, diagnostic_path, complete
            )
            diagnostic, accepted_prompt = await generate_weekly_digest_coach_diagnostic(
                digest, reserved_complete, repair_requirements=repair
            )
            if len(collected) != before + 1:
                raise RuntimeError(f"Provider did not record one attempt: {key}")
            metric = collected[-1]
            metric.call_label = f"coach_voice_refresh:{key}:attempt_{attempt_number}"
            diagnostic = diagnostic.model_copy(update={"llm_call": metric})
            _write(
                diagnostic_path,
                {
                    "prompt": accepted_prompt,
                    "prompt_sha256": hashlib.sha256(
                        accepted_prompt.encode()
                    ).hexdigest(),
                    **diagnostic.model_dump(mode="json"),
                },
            )
        assert diagnostic.narrative is not None and diagnostic.validation is not None
        response_path = output / "responses" / f"{key.replace('::', '_')}.json"
        _write(
            response_path,
            attach_coach_artifacts(
                digest, diagnostic.narrative, diagnostic.validation
            ).model_dump(mode="json"),
        )
        receipts = [_load_diagnostic(root, a) for a in state["attempts"]]
        response = SavedCoachResponse.model_validate(
            {
                "scenario_id": case["scenario_id"],
                "persona_id": digest.persona_id,
                "week_start": digest.week_start,
                "week_end": digest.week_end,
                "narrative": diagnostic.narrative.model_dump(mode="json"),
                "generation": {
                    "model_contract": {
                        "provider": "openai",
                        "model": DEFAULT_OPENAI_MODEL,
                        "reasoning_effort": DEFAULT_OPENAI_REASONING_EFFORT,
                    },
                    "service_tier": DEFAULT_OPENAI_SERVICE_TIER,
                    "prompt_name": plan["policy"]["prompt"]["name"],
                    "prompt_version": plan["policy"]["prompt"]["version"],
                    "prompt_sha256": saved["prompt_sha256"],
                    "prompt": saved["prompt"],
                    "raw_output": diagnostic.raw_output,
                    "response_sha256": _hash(
                        diagnostic.narrative.model_dump(mode="json")
                    ),
                    "attempt_count": len(receipts),
                    "diagnostic_paths": [
                        a["diagnostic_path"] for a in state["attempts"]
                    ],
                    "call_metrics": [receipt["llm_call"] for receipt in receipts],
                    "weekly_drift_input_sha256": case["input_sha256"],
                    "generated_response_path": str(response_path.relative_to(root)),
                    **{
                        field: case[field]
                        for field in (
                            "source_bundle_path",
                            "source_bundle_content_sha256",
                            "weekly_digest_event_id",
                        )
                    },
                },
            }
        )
        state["response"] = response.model_dump(mode="json")
        _write(state_path, state)
        print(f"Accepted {key} ({len(receipts)} attempt(s))", flush=True)


def apply(root: Path, output: Path, plan: dict[str, Any]) -> None:
    """Replace the active fixture atomically after complete, validated coverage."""
    if _prepare_for_plan(root, output, plan) != plan:
        raise ValueError("The supplied plan differs from its frozen checkpoint")
    responses = {}
    for key, case in plan["cases"].items():
        path = _state_path(output, key)
        state = json.loads(path.read_bytes()) if path.exists() else {}
        if state.get("response") is None:
            raise RuntimeError(f"Refresh is incomplete: {key}")
        response = SavedCoachResponse.model_validate(state["response"])
        if key in plan.get("retained_responses", {}) and (
            state["response"] != plan["retained_responses"][key]
        ):
            raise ValueError(f"Retained response changed: {key}")
        digest = WeeklyDigest.model_validate(case["digest"])
        if (
            _coach_unavailable_reason(response, digest) is not None
            or not validate_weekly_digest_narrative(
                digest, response.narrative, validate_voice=True
            ).all_passed
        ):
            raise ValueError(f"Refreshed response failed validation: {key}")
        responses[key] = response
    replacement = SavedCoachResponseFixture(responses=responses).model_dump(mode="json")
    _write(output / "refreshed_coach_digest_responses.json", replacement)
    _write(root / COACH_RESPONSES_PATH, replacement)


def write_report(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    states = [
        json.loads(path.read_bytes()) for path in (output / "cases").glob("*.json")
    ]
    metrics = [
        LLMCallMetrics.model_validate(json.loads(path.read_bytes())["llm_call"])
        for path in (output / "diagnostics").glob("*.json")
    ]
    summary = {
        "accepted": sum(state.get("response") is not None for state in states),
        "retained": len(plan.get("retained_responses", {})),
        "total": len(plan["cases"]),
        "reserved_usd": str(
            sum(
                (
                    Decimal(attempt["reserved_usd"])
                    for state in states
                    for attempt in state["attempts"]
                ),
                Decimal(0),
            )
        ),
        "maximum_reserved_usd": plan["policy"]["maximum_reserved_usd"],
        "usage": summarize_llm_call_metrics(metrics),
        "limitations": (
            "Automated validation only; no human validation or new judge run."
        ),
    }
    _write(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--prior-run", type=Path)
    parser.add_argument("--repair-requirements", type=Path)
    args = parser.parse_args()
    load_dotenv()
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT)
    prior_run = (ROOT / args.prior_run).resolve() if args.prior_run else None
    requirements = (
        json.loads(args.repair_requirements.read_bytes())
        if args.repair_requirements
        else None
    )
    plan = prepare(ROOT, output, prior_run=prior_run, repair_requirements=requirements)
    print(
        f"Frozen {len(plan['cases'])} weeks; reserve at most "
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
