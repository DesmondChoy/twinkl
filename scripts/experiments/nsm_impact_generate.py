"""Generate paired Coach Digests for the North Star Moment impact main run.

The main run of ``docs/north_star/coach_digest_impact_eval.md`` uses the 72
qualified-final weeks whose full-history North Star Moment the selection
experiment confirmed as a correct card. For each week this script writes two
Coach Digests from identical input, one with the North Star Moment and one
without it, using the live Coach Digest model (``gpt-5.6-luna``, effort none).

Commands (run from the repository root):

    python -m scripts.experiments.nsm_impact_generate            # free dry run
    python -m scripts.experiments.nsm_impact_generate --execute  # paid calls
    python -m scripts.experiments.nsm_impact_generate --export   # judge pairs

The dry run freezes ``plan.json``; later commands refuse a changed plan. Each
attempt is checkpointed before the request, so an interrupted run resumes
without repeating completed calls. ``--export`` writes ``pairs.json``, which
``nsm_impact_pilot prepare --pairs`` turns into blinded judge tasks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scripts.coach.compare_scenario_coach import (
    _generate_arm,
    _state_path,
    _validate_arm_receipts,
)
from scripts.coach.complete_scenario_coach import ROOT, _transport, _write
from scripts.coach.refresh_scenario_coach import (
    INPUT_TOKEN_ALLOWANCE,
    MAXIMUM_ATTEMPTS,
    PER_REQUEST_RESERVE_USD,
)
from scripts.experiments import nsm_impact_pilot as pilot
from src.coach.demo_comparison import (
    PROMPT_NAME,
    PROMPT_VERSION,
    CoachComparisonArm,
    build_north_star_context,
    hash_json,
    hash_text,
    render_demo_comparison_prompt,
)
from src.coach.llm_client import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENAI_REASONING_EFFORT,
    DEFAULT_OPENAI_SERVICE_TIER,
    OPENAI_LUNA_INPUT_USD_PER_MILLION,
    OPENAI_LUNA_OUTPUT_USD_PER_MILLION,
    OPENAI_LUNA_PRICING_SOURCE,
    summarize_llm_call_metrics,
)
from src.coach.schemas import LLMCallMetrics, WeeklyDigest
from src.coach.weekly_digest import LLMCompleteFn, build_weekly_drift_reviewer_digest
from src.drift_detector import DriftDetectorResult, detect_drift
from src.north_star.runtime import NorthStarRecord, NorthStarSelection, SourceWriting
from src.weekly_drift_reviewer import WeeklyDriftReviewerDecision
from src.wrangling.parse_wrangled_data import parse_wrangled_file

RECORD_PATH = "logs/experiments/reports/north_star_20260906/nsm_experiment.json"
VARIANT = "full_history"
EXPECTED_WEEKS = 72
ARMS = ("without_north_star", "with_north_star")
DEFAULT_OUTPUT = Path("logs/experiments/reports/nsm_impact_generation_20260926")
# Worst case: 144 arms x 4 attempts x US$0.018 reserve = US$10.368.
TOTAL_BUDGET_USD = Decimal("11.00")
SOURCE_FILES = (
    "scripts/experiments/nsm_impact_generate.py",
    "scripts/coach/compare_scenario_coach.py",
    "src/coach/demo_comparison.py",
    "src/coach/weekly_digest.py",
    "src/coach/llm_client.py",
    "prompts/demo_coach_nsm_comparison.yaml",
    "prompts/versions/weekly_digest_coach_v4_5.yaml",
    RECORD_PATH,
)


def cumulative_decisions(
    cases: list[dict[str, Any]],
) -> dict[str, list[WeeklyDriftReviewerDecision]]:
    """Replay each Persona's stored decisions; each week must reproduce its Drift."""
    by_persona: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_persona[case["persona_id"]].append(case)
    replayed = {}
    for persona_id, weeks in by_persona.items():
        accumulated: list[WeeklyDriftReviewerDecision] = []
        for case in sorted(weeks, key=lambda c: c["week_start"]):
            accumulated.extend(
                WeeklyDriftReviewerDecision.model_validate(row)
                for row in case["upstream"]["current_decisions"]
            )
            drift = detect_drift(accumulated, persona_id=persona_id)
            if drift.model_dump(mode="json") != case["drift_result"]:
                raise ValueError(f"Replayed Drift differs: {case['case_id']}")
            replayed[case["case_id"]] = list(accumulated)
    return replayed


def north_star_record(
    case: dict[str, Any], output: dict[str, Any], created_at: str
) -> NorthStarRecord:
    """Rebuild the displayed card as a runtime record for context validation."""
    selected = NorthStarSelection.model_validate(output["selected"])
    value = next(v for v in case["values"] if v["core_value"] == output["core_value"])
    source = next(s for s in value["sources"] if s["entry_id"] == selected.entry_id)
    metadata = case["source_metadata"][selected.entry_id]
    writing = SourceWriting(
        owner_id=metadata["owner_id"],
        entry_id=selected.entry_id,
        t_index=metadata["t_index"],
        date=metadata["date"],
        journal_entry=source["journal_entry"],
        nudge_response=source["nudge_response"],
        available_at=metadata["available_at"],
        response_available_at=metadata["response_available_at"],
    )
    return NorthStarRecord(
        session_id=f"nsm-impact:{case['persona_id']}",
        owner_id=case["persona_id"],
        profile_ref=case["profile_ref"],
        week_start=case["week_start"],
        week_end=case["week_end"],
        cutoff_at=case["cutoff_at"],
        input_hash=case["input_hash"],
        status="complete",
        mode=output["mode"],
        reason=output["reason"],
        core_value=output["core_value"],
        value_phrase=value["user_phrase"],
        selected=selected,
        source_ids=[selected.entry_id],
        sources=[writing],
        onset_t_index=case["onset_t_index"],
        onset_date=case["onset_date"],
        onset_available_at=case["onset_available_at"],
        created_at=created_at,
    )


def _entry(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": row["date"],
        "content": row["initial_entry"],
        "nudge_response": row["response_text"] or None,
    }


def build_cases(record: dict[str, Any], root: Path = ROOT) -> dict[str, dict]:
    """Freeze Coach Digest inputs for every correct final-set card."""
    execution = record["execution"]
    correct = {
        grade["case_id"]
        for grade in execution["grades"]
        if grade["split"] == "final" and grade["variants"][VARIANT]["correct_card"]
    }
    cases = [c for c in execution["cases"] if c["split"] == "final"]
    decisions = cumulative_decisions(cases)
    frozen = {}
    for case in cases:
        if case["case_id"] not in correct:
            continue
        persona_id, week_start = case["persona_id"], case["week_start"]
        output = execution["variants"][VARIANT][case["case_id"]]
        digest = build_weekly_drift_reviewer_digest(
            persona_id=persona_id,
            wrangled_dir=root / "logs/wrangled",
            week_start=week_start,
            week_end=case["week_end"],
            core_values=case["core_values"],
            decisions=decisions[case["case_id"]],
            drift_result=DriftDetectorResult.model_validate(case["drift_result"]),
        )
        nsm = north_star_record(case, output, record["created_at"])
        context = build_north_star_context(nsm)
        _profile, entries, warnings = parse_wrangled_file(
            root / f"logs/wrangled/persona_{persona_id}.md"
        )
        if warnings:
            raise ValueError(f"Wrangled history has warnings: {persona_id}")
        week = [e for e in entries if week_start <= e["date"] <= case["week_end"]]
        source_index = nsm.sources[0].t_index
        source_row = entries[source_index]
        if source_row["t_index"] != source_index:
            raise ValueError(f"Wrangled history order differs: {persona_id}")
        key = f"{persona_id}::{week_start}"
        frozen[key] = {
            "case_id": case["case_id"],
            "persona_id": persona_id,
            "week_start": week_start,
            "week_end": case["week_end"],
            "weekly_state": case["weekly_state"],
            "mode": context.mode,
            "selection_input_hash": case["input_hash"],
            "digest": digest.model_dump(mode="json"),
            "input_sha256": hash_json(
                digest.model_dump(
                    mode="json", exclude={"coach_narrative", "validation"}
                )
            ),
            "north_star_record": nsm.model_dump(mode="json"),
            "north_star_context": context.model_dump(mode="json", exclude_none=True),
            "base_prompts": {
                "without_north_star": render_demo_comparison_prompt(digest, None),
                "with_north_star": render_demo_comparison_prompt(digest, context),
            },
            "week_entries": [_entry(e) for e in week],
            "history_entries": []
            if week_start <= context.date <= case["week_end"]
            else [_entry(source_row)],
        }
    if len(frozen) != EXPECTED_WEEKS:
        raise ValueError(f"Expected {EXPECTED_WEEKS} weeks, found {len(frozen)}")
    return dict(sorted(frozen.items()))


def prepare(root: Path, output: Path) -> dict[str, Any]:
    record = json.loads((root / RECORD_PATH).read_bytes())
    cases = build_cases(record, root)
    arms = len(cases) * len(ARMS)
    maximum_reserved = arms * MAXIMUM_ATTEMPTS * PER_REQUEST_RESERVE_USD
    if maximum_reserved > TOTAL_BUDGET_USD:
        raise ValueError("Maximum attempts exceed the frozen total budget")
    plan = {
        "schema_version": "nsm-impact-generation-plan-v1",
        "selection": {
            "record": RECORD_PATH,
            "variant": VARIANT,
            "split": "final",
            "rule": "full-history card graded correct by AI review",
            "weeks": len(cases),
            "personas": len({c["persona_id"] for c in cases.values()}),
        },
        "policy": {
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
            "prompt_name": PROMPT_NAME,
            "prompt_version": PROMPT_VERSION,
            "evidence_policy": "complete (live runtime default)",
            "source_sha256": {
                name: hash_text((root / name).read_text()) for name in SOURCE_FILES
            },
        },
        "cases": cases,
    }
    path = output / "plan.json"
    if path.exists():
        if json.loads(path.read_bytes()) != plan:
            raise ValueError("Frozen inputs, source files, or policy changed")
    else:
        _write(path, plan)
    return plan


async def generate(
    root: Path,
    output: Path,
    plan: dict[str, Any],
    *,
    llm_complete: LLMCompleteFn | None = None,
    metrics: list[LLMCallMetrics] | None = None,
) -> None:
    """Personas run concurrently; each Persona's weeks and arms stay sequential."""
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for key, case in plan["cases"].items():
        groups[case["persona_id"]].append((key, case))

    async def persona(cases: list[tuple[str, dict[str, Any]]]) -> None:
        collected = metrics if metrics is not None else []
        complete = llm_complete or _transport(collected)
        for key, case in cases:
            for arm in ARMS:
                await _generate_arm(
                    root, output, key, case, arm, complete, collected, [], [], None
                )

    if llm_complete is not None:
        for cases in groups.values():
            await persona(cases)
        return
    results = await asyncio.gather(
        *(persona(cases) for cases in groups.values()), return_exceptions=True
    )
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        raise RuntimeError("; ".join(str(error) for error in failures))


def to_pilot_pair(
    key: str, case: dict[str, Any], arms: dict[str, CoachComparisonArm]
) -> dict[str, Any]:
    """Match the pair format that ``nsm_impact_pilot`` renders into judge tasks."""
    without, with_moment = (arms[arm].model_dump(mode="json") for arm in ARMS)
    shared = pilot.coach_input(without)
    if (
        shared["north_star_context"] is not None
        or {**pilot.coach_input(with_moment), "north_star_context": None} != shared
    ):
        raise ValueError(f"{key}: arm inputs differ beyond the North Star Moment")
    return {
        "pair_id": f"{case['persona_id']}:{case['week_start']}",
        "persona_id": case["persona_id"],
        "week_start": case["week_start"],
        "mode": case["mode"],
        "shared_input": shared,
        "week_entries": case["week_entries"],
        "history_entries": case["history_entries"],
        "responses": {
            "without": pilot.response_text(without["narrative"]),
            "with": pilot.response_text(with_moment["narrative"]),
        },
    }


def export(root: Path, output: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate every accepted arm's receipts, then write the judge pairs."""
    pairs = []
    for key, case in plan["cases"].items():
        digest = WeeklyDigest.model_validate(case["digest"])
        arms = {}
        for arm in ARMS:
            path = _state_path(output, key, arm)
            state = json.loads(path.read_bytes()) if path.exists() else {}
            if state.get("response") is None:
                raise RuntimeError(f"Arm is incomplete: {key}:{arm}")
            if state["input_sha256"] != hash_json(case):
                raise ValueError(f"Checkpoint input changed: {key}")
            _validate_arm_receipts(root, state, digest)
            arms[arm] = CoachComparisonArm.model_validate(state["response"])
        pairs.append(to_pilot_pair(key, case, arms))
    _write(output / "pairs.json", pairs)
    return pairs


def write_summary(output: Path, plan: dict[str, Any]) -> dict[str, Any]:
    states = [json.loads(p.read_bytes()) for p in (output / "cases").glob("*.json")]
    metrics = [
        LLMCallMetrics.model_validate(json.loads(p.read_bytes())["llm_call"])
        for p in (output / "diagnostics").glob("*.json")
    ]
    summary = {
        "weeks": len(plan["cases"]),
        "expected_arms": len(plan["cases"]) * len(ARMS),
        "accepted_arms": sum(s.get("response") is not None for s in states),
        "attempts": sum(len(s["attempts"]) for s in states),
        "maximum_reserved_usd": plan["policy"]["maximum_reserved_usd"],
        "usage": summarize_llm_call_metrics(metrics),
        "limitations": "Synthetic Personas; deterministic Coach Digest validation "
        "only. Initial requests differ only in north_star_context; validation "
        "retries keep separate receipts.",
    }
    _write(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="make paid calls")
    parser.add_argument("--export", action="store_true", help="write pairs.json")
    args = parser.parse_args()
    load_dotenv()
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT)
    plan = prepare(ROOT, output)
    print(
        f"Frozen {len(plan['cases'])} weeks; maximum reserve "
        f"US${plan['policy']['maximum_reserved_usd']}.",
        flush=True,
    )
    try:
        if args.execute:
            asyncio.run(generate(ROOT, output, plan))
        if args.export:
            print(f"Exported {len(export(ROOT, output, plan))} pairs.", flush=True)
    finally:
        if (output / "cases").exists():
            print(json.dumps(write_summary(output, plan)), flush=True)


if __name__ == "__main__":
    main()
