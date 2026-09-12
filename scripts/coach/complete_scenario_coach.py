"""Complete missing saved Coach Digests from frozen per-week scenario inputs.

Dry run: uv run python -m scripts.coach.complete_scenario_coach
Paid generation: add --execute. Existing compatible responses are never replaced.
Every attempt is checkpointed before transport; interrupted attempts without a
receipt require inspection, rather than silently repeating an unknown paid call.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

from prompts import get_prompt_metadata
from src.coach.llm_client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_OPENAI_SERVICE_TIER,
    DEFAULT_TIMEOUT_SECONDS,
    _openai_call_metrics,
    summarize_llm_call_metrics,
)
from src.coach.schemas import CoachDigestDiagnostic, LLMCallMetrics, WeeklyDigest
from src.coach.weekly_digest import (
    LLMCompleteFn,
    attach_coach_artifacts,
    generate_weekly_digest_coach_diagnostic,
    validate_weekly_digest_narrative,
)
from src.demo.contracts import ContractFixtureSet
from src.demo.scenarios import (
    CATALOG_PATH,
    COACH_RESPONSES_PATH,
    SCENARIO_DIRECTORY,
    SavedCoachResponse,
    _coach_unavailable_reason,
    _weekly_drift_input_sha256,
    load_saved_coach_responses,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(
    "logs/experiments/reports/demo_persona_replacement_20260908/current"
)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def collect_cases(root: Path) -> dict[str, dict[str, Any]]:
    """Read every week from catalog-hash-verified scenario bundles."""
    catalog = json.loads((root / CATALOG_PATH).read_bytes())
    cases: dict[str, dict[str, Any]] = {}
    for item in catalog["scenarios"]:
        path = SCENARIO_DIRECTORY / item["file"]
        raw = (root / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["content_sha256"]:
            raise ValueError(f"Scenario hash mismatch: {path}")
        fixture = ContractFixtureSet.model_validate_json(raw)
        if fixture.scenario.scenario_id != item["scenario_id"]:
            raise ValueError(f"Scenario identity mismatch: {path}")
        for week in fixture.scenario.weeks:
            events = [
                e
                for e in fixture.trace_events
                if e.event_id in week.event_ids
                and e.event_type == "weekly_digest_built"
            ]
            if len(events) != 1:
                raise ValueError(f"Expected one digest input: {week.week_id}")
            event = events[0]
            digest = event.details.digest.model_copy(
                update={"coach_narrative": None, "validation": None}
            )
            if (
                digest.persona_id != fixture.scenario.persona_id
                or digest.week_start != week.week_start
                or digest.week_end != week.week_end
            ):
                raise ValueError(f"Digest identity mismatch: {week.week_id}")
            key = f"{item['scenario_id']}::{week.week_start}"
            cases[key] = {
                "scenario_id": item["scenario_id"],
                "digest": digest.model_dump(mode="json"),
                "input_sha256": _weekly_drift_input_sha256(digest),
                "source_bundle_path": str(path),
                "source_bundle_content_sha256": item["content_sha256"],
                "weekly_digest_event_id": event.event_id,
            }
    return cases


def prepare(
    root: Path,
    output: Path,
    *,
    repair_requirements: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    cases = collect_cases(root)
    if repair_requirements is not None and (
        not isinstance(repair_requirements, dict)
        or any(
            key not in cases
            or not isinstance(requirements, list)
            or not requirements
            or any(
                not isinstance(item, str) or not item.strip() for item in requirements
            )
            for key, requirements in repair_requirements.items()
        )
    ):
        raise ValueError("Repair requirements must map current case keys to text lists")
    fixture = load_saved_coach_responses(root)
    for key, response in fixture.responses.items():
        if key not in cases:
            raise ValueError(f"Saved response is outside current scenarios: {key}")
        digest = WeeklyDigest.model_validate(cases[key]["digest"])
        if _coach_unavailable_reason(response, digest) is not None:
            raise ValueError(f"Refusing to overwrite incompatible response: {key}")
        if not validate_weekly_digest_narrative(
            digest, response.narrative, validation_policy="historical"
        ).all_passed:
            raise ValueError(f"Existing response failed validations: {key}")
    policy = {
        "model": DEFAULT_OPENAI_MODEL,
        "reasoning": DEFAULT_OPENAI_REASONING_EFFORT,
        "service_tier": DEFAULT_OPENAI_SERVICE_TIER,
        "sdk_retries": 0,
        "maximum_attempts": 2,
        "prompt": get_prompt_metadata("weekly_digest_coach"),
        "code_sha256": {
            str(p): hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in [
                Path("src/coach/weekly_digest.py"),
                Path("src/coach/llm_client.py"),
                Path("src/coach/schemas.py"),
                Path("prompts/weekly_digest_coach.yaml"),
            ]
        },
    }
    if repair_requirements:
        policy["repair_requirements"] = repair_requirements
    plan_path = output / "plan.json"
    if plan_path.exists():
        plan: dict[str, Any] = json.loads(plan_path.read_bytes())
        if plan["policy"] != policy or set(plan["cases"]) != set(cases):
            raise ValueError("Frozen generation policy or cases changed")
        for key, case in cases.items():
            if case["input_sha256"] != plan["cases"][key]["input_sha256"]:
                raise ValueError(f"Frozen weekly input changed: {key}")
        for key, saved in plan["retained_responses"].items():
            if fixture.responses[key].model_dump(mode="json") != saved:
                raise ValueError(f"Retained response changed: {key}")
        return plan
    plan = {
        "policy": policy,
        "cases": cases,
        "retained_responses": {
            key: response.model_dump(mode="json")
            for key, response in fixture.responses.items()
        },
        "missing_keys": [key for key in cases if key not in fixture.responses],
    }
    _write(plan_path, plan)
    for key in plan["missing_keys"]:
        _write(
            output / "inputs" / f"{key.replace('::', '_')}.json", cases[key]["digest"]
        )
    return plan


def _transport(metrics: list[LLMCallMetrics]) -> LLMCompleteFn:
    client = AsyncOpenAI(max_retries=0)

    async def complete(
        prompt: str, response_format: dict | None, instructions: str | None = None
    ) -> str | None:
        started = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "model": DEFAULT_OPENAI_MODEL,
                "input": prompt,
                "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                "reasoning": {"effort": DEFAULT_OPENAI_REASONING_EFFORT},
                "service_tier": DEFAULT_OPENAI_SERVICE_TIER,
                "store": False,
                "timeout": DEFAULT_TIMEOUT_SECONDS,
            }
            if instructions is not None:
                kwargs["instructions"] = instructions
            if response_format is not None:
                kwargs["text"] = {"format": response_format}
            response = await client.responses.create(**kwargs)
            metrics.append(
                _openai_call_metrics(response, time.perf_counter() - started)
            )
            return response.output_text or None
        except Exception as exc:
            metrics.append(
                LLMCallMetrics(
                    provider="openai",
                    model=DEFAULT_OPENAI_MODEL,
                    reasoning_effort=DEFAULT_OPENAI_REASONING_EFFORT,
                    service_tier=DEFAULT_OPENAI_SERVICE_TIER,
                    status="error",
                    latency_seconds=time.perf_counter() - started,
                    error_type=type(exc).__name__,
                )
            )
            return None

    return complete


def _merge(root: Path, key: str, response: SavedCoachResponse) -> None:
    fixture = load_saved_coach_responses(root)
    current = fixture.responses.get(key)
    if current is not None and current != response:
        raise ValueError(f"Refusing to overwrite saved response: {key}")
    fixture.responses[key] = response
    _write(root / COACH_RESPONSES_PATH, fixture.model_dump(mode="json"))


async def generate(
    root: Path,
    output: Path,
    plan: dict[str, Any],
    *,
    llm_complete: LLMCompleteFn | None = None,
    metrics: list[LLMCallMetrics] | None = None,
) -> None:
    collected = metrics if metrics is not None else []
    complete = llm_complete or _transport(collected)
    for key in plan["missing_keys"]:
        case = plan["cases"][key]
        digest = WeeklyDigest.model_validate(case["digest"])
        stem = key.replace("::", "_")
        state_path = output / "cases" / f"{stem}.json"
        state = (
            json.loads(state_path.read_bytes())
            if state_path.exists()
            else {"input_sha256": case["input_sha256"], "attempts": []}
        )
        if state["input_sha256"] != case["input_sha256"]:
            raise ValueError(f"Checkpoint input changed: {key}")
        if state.get("response") is not None:
            _merge(root, key, SavedCoachResponse.model_validate(state["response"]))
            continue
        if key in load_saved_coach_responses(root).responses:
            raise ValueError(f"Untracked saved response appeared: {key}")
        initial_repair = plan["policy"].get("repair_requirements", {}).get(key, [])
        repair = initial_repair
        diagnostic = None
        accepted_prompt = ""
        while True:
            if state["attempts"]:
                last = state["attempts"][-1]
                diagnostic_path = root / last["diagnostic_path"]
                if not diagnostic_path.exists():
                    raise RuntimeError(
                        f"Unresolved interrupted attempt; inspect before retry: {key}"
                    )
                saved = json.loads(diagnostic_path.read_bytes())
                diagnostic = CoachDigestDiagnostic.model_validate(saved)
                accepted_prompt = saved["prompt"]
                if diagnostic.accepted:
                    break
                if (
                    diagnostic.failure_stage != "coach_validation"
                    or len(state["attempts"]) >= 2
                ):
                    raise RuntimeError(
                        f"Terminal Coach failure: {key}: {diagnostic.failure_stage}"
                    )
                repair = [*initial_repair, *diagnostic.failure_details]
            attempt = len(state["attempts"]) + 1
            diagnostic_path = output / "diagnostics" / f"{stem}_{attempt}.json"
            relative = str(diagnostic_path.relative_to(root))
            state["attempts"].append({"diagnostic_path": relative})
            _write(state_path, state)
            before = len(collected)
            diagnostic, accepted_prompt = await generate_weekly_digest_coach_diagnostic(
                digest, complete, repair_requirements=repair
            )
            if len(collected) != before + 1:
                raise RuntimeError(f"Provider did not record one attempt: {key}")
            metric = collected[-1]
            metric.call_label = f"coach_generation:{key}:attempt_{attempt}"
            diagnostic = diagnostic.model_copy(update={"llm_call": metric})
            _write(
                diagnostic_path,
                {
                    "prompt": accepted_prompt,
                    **diagnostic.model_dump(mode="json"),
                },
            )
        assert diagnostic is not None and diagnostic.narrative is not None
        assert diagnostic.validation is not None and diagnostic.raw_output is not None
        enriched = attach_coach_artifacts(
            digest, diagnostic.narrative, diagnostic.validation
        )
        response_path = output / "responses" / f"{stem}.json"
        _write(response_path, enriched.model_dump(mode="json"))
        diagnostics = [
            json.loads((root / a["diagnostic_path"]).read_bytes())
            for a in state["attempts"]
        ]
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
                    "prompt_sha256": hashlib.sha256(
                        accepted_prompt.encode()
                    ).hexdigest(),
                    "prompt": accepted_prompt,
                    "raw_output": diagnostic.raw_output,
                    "response_sha256": _hash(
                        diagnostic.narrative.model_dump(mode="json")
                    ),
                    "attempt_count": len(diagnostics),
                    "diagnostic_paths": [
                        a["diagnostic_path"] for a in state["attempts"]
                    ],
                    "call_metrics": [d["llm_call"] for d in diagnostics],
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
        _merge(root, key, response)
        print(f"Accepted {key} ({len(diagnostics)} attempt(s))", flush=True)


def write_report(root: Path, output: Path, plan: dict[str, Any]) -> None:
    fixture = load_saved_coach_responses(root)
    metrics = []
    for path in sorted((output / "diagnostics").glob("*.json")):
        diagnostic = json.loads(path.read_bytes())
        metrics.append(LLMCallMetrics.model_validate(diagnostic["llm_call"]))
    summary = {
        "retained": len(plan["retained_responses"]),
        "generated": sum(k in fixture.responses for k in plan["missing_keys"]),
        "total": len(fixture.responses),
        "usage": summarize_llm_call_metrics(metrics),
    }
    _write(output / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--repair-requirements",
        type=Path,
        help="JSON mapping of case keys to repair instructions, frozen in the plan",
    )
    args = parser.parse_args()
    load_dotenv()
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT)
    repair_requirements = (
        json.loads(args.repair_requirements.read_bytes())
        if args.repair_requirements
        else None
    )
    plan = prepare(ROOT, output, repair_requirements=repair_requirements)
    print(
        f"Frozen {len(plan['cases'])} weeks; retain {len(plan['retained_responses'])}; "
        f"generate {len(plan['missing_keys'])} missing responses."
    )
    if args.execute:
        try:
            asyncio.run(generate(ROOT, output, plan))
        finally:
            write_report(ROOT, output, plan)


if __name__ == "__main__":
    main()
