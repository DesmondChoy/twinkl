"""Freeze and run the NSM development AI-review feasibility gate.

Run --prepare before --run. No reserved history enters this script. Provider
attempts, failed responses, prompts, complete reference decisions and exact-quote
checks are retained; an identical completed request reuses its saved attempt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from scripts.experiments.north_star_phase0 import fraction, load_inputs  # noqa: E402
from src.north_star.provider import (  # noqa: E402
    POLICY_PATH,
    BudgetedProvider,
)
from src.north_star.review import (  # noqa: E402
    REVIEW_PROMPT_VERSION,
    ReviewBatch,
    ReviewValidationError,
    SourceEntry,
    build_review_prompt,
    review_json_schema,
    select_moment,
    validate_review,
)

DIRECTORY = ROOT / "logs/experiments/reports/north_star_phase0b_20260905"
RETRIEVAL = ROOT / "logs/experiments/reports/north_star_phase0_20260905/retrieval.json"
QUOTE_REVIEW = (
    "\nThis is a predefined high-risk reference check of the exact candidate "
    "quotation in candidate_quote. Read all original source context. Return "
    "supportive only if THAT exact quotation describes the writer's completed "
    "supportive action for the requested Core Value and no supplied writing "
    "conflicts with that Core Value. If supportive, evidence_quote must equal "
    "candidate_quote exactly, with the same quote_source. Never replace it with "
    "a different passage. Otherwise return not_supportive or abstain with an "
    "empty quotation and null source."
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    path = DIRECTORY / "manifest.json"
    if path.exists():
        raise ValueError("Development manifest already frozen; do not overwrite")
    result = json.loads(RETRIEVAL.read_text())
    retrieval = result["retrieval"]
    if not retrieval["gate_passed"] or retrieval["selected_k"] != 3:
        raise ValueError("Phase0A must pass and freeze k before Phase0B")
    _, _, _, histories = load_inputs(ROOT)
    cases = []
    for case in retrieval["cases"]:
        episode = case["episode"]
        persona = episode["persona_id"]
        if persona in retrieval["cohort"]["reserved_persona_ids"]:
            raise ValueError("Reserved history in development")
        entries = {entry["t_index"]: entry for entry in histories[persona]}
        sources = [
            SourceEntry(
                entry_id=row["entry_id"],
                journal_entry=entries[row["t_index"]]["initial_entry"],
                nudge_response=None,
            ).model_dump()
            for row in case["ranking"]
        ]
        cases.append(
            {
                "case_id": episode["episode_id"],
                "episode": episode,
                "core_value": episode["dimension"],
                "value": retrieval["config"]["queries"][episode["dimension"]],
                "all_eligible_sources_in_retrieval_order": sources,
                "runtime_entry_ids": [row["entry_id"] for row in case["ranking"][:3]],
                "case_categories": [
                    "no_earlier_writing"
                    if not sources
                    else "one_earlier_entry"
                    if len(sources) == 1
                    else "multiple_earlier_entries",
                    episode["dimension"],
                ],
            }
        )
    manifest = {
        "schema_version": "north-star-development-v1",
        "frozen_at": datetime.now(UTC).isoformat(),
        "cases": cases,
        "case_count": len(cases),
        "sampling": (
            "All episodes from the frozen Phase0A development group; no subsampling"
        ),
        "seed": retrieval["cohort"]["seed"],
        "source_hashes": {
            str(RETRIEVAL.relative_to(ROOT)): file_hash(RETRIEVAL),
            str(POLICY_PATH.relative_to(ROOT)): file_hash(POLICY_PATH),
            "src/north_star/review.py": file_hash(ROOT / "src/north_star/review.py"),
        },
        "prompt_version": REVIEW_PROMPT_VERSION,
        "schema": review_json_schema(),
        "reference_protocol": (
            "Gemini independently reviews every eligible original Journal Entry, "
            "without runtime decisions or labels. Legacy responses are excluded. "
            "Exact displayed quotes differing from its reference quotation receive "
            "a second Gemini check under the frozen candidate-quotation instruction. "
            "Acceptance requires both primary support and candidate-quote approval; "
            "disagreement or abstention remains incorrect. No earlier sources means "
            "no call and a separately counted structurally empty case."
        ),
        "quote_review_instruction": QUOTE_REVIEW,
        "retrieval_only_rule": (
            "Display entire original writing of the first ranked source if it passes "
            "deterministic quotation checks; semantic acceptance uses the exhaustive "
            "reference decision for that source. No source or a failed check omits it."
        ),
        "criteria": {
            "incorrect_displayed": 0,
            "correct_no_card_rate": 1.0,
            "unexpected_provider_failure_rate_max": 0.05,
            "require_reference_confirmed_no_example_with_earlier_sources": True,
            "require_saved_persona_acceptance": True,
        },
        "integration_checks": (
            "Source chronology and lifecycle adversaries use separate injected tests, "
            "not provider failure denominators."
        ),
    }
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"Frozen {len(cases)} development cases; reserved histories excluded",
        flush=True,
    )


async def review(
    provider: BudgetedProvider,
    *,
    case: dict,
    sources: list[SourceEntry],
    role: Literal["runtime", "reference", "quote_reference"],
    candidate: dict | None = None,
) -> tuple[ReviewBatch | None, list[dict]]:
    system, prompt = build_review_prompt(
        core_value=case["core_value"],
        user_phrase=case["value"]["user_phrase"],
        approved_definition=case["value"]["definition"],
        sources=sources,
    )
    if candidate:
        system += QUOTE_REVIEW
        data = json.loads(prompt)
        data["candidate_quote"] = candidate["evidence_quote"]
        data["quote_source"] = candidate["quote_source"]
        prompt = json.dumps(data, ensure_ascii=False, sort_keys=True)
    attempts = []
    for index in range(2):
        attempt = await provider.complete(
            system=system,
            prompt=prompt,
            schema=review_json_schema(),
            provider="openai" if role == "runtime" else "gemini",
            purpose=f"development-{role}",
            retry=index > 0,
        )
        batch = None
        if attempt.status == "completed" and attempt.raw_text:
            try:
                batch = validate_review(
                    attempt.raw_text, core_value=case["core_value"], sources=sources
                )
                if candidate:
                    decision = batch.results[0]
                    if decision.decision == "supportive" and (
                        decision.evidence_quote != candidate["evidence_quote"]
                        or decision.quote_source != candidate["quote_source"]
                    ):
                        raise ValueError("Reference replaced the candidate quotation")
            except (ReviewValidationError, ValueError):
                attempt = provider.invalidate(attempt, "review_contract_invalid")
                batch = None
        attempts.append(
            {"attempt": attempt.model_dump(), "system": system, "prompt": prompt}
        )
        if batch is not None:
            return batch, attempts
        if not attempt.retryable:
            break
    return None, attempts


async def run_case(
    case: dict, provider: BudgetedProvider, semaphore: asyncio.Semaphore
) -> dict:
    async with semaphore:
        sources = [
            SourceEntry(**row)
            for row in case["all_eligible_sources_in_retrieval_order"]
        ]
        result: dict[str, Any] = {
            "case_id": case["case_id"],
            "core_value": case["core_value"],
            "eligible_sources": len(sources),
            "attempts": [],
        }
        if not sources:
            result.update(
                status="no_earlier_writing",
                selected=None,
                reference_no_example=True,
                incorrect_displayed=False,
                retrieval_only_selected=False,
            )
            return result
        runtime_sources = sources[:3]
        runtime, attempts = await review(
            provider, case=case, sources=runtime_sources, role="runtime"
        )
        result["attempts"].extend(attempts)
        reference, attempts = await review(
            provider, case=case, sources=sources, role="reference"
        )
        result["attempts"].extend(attempts)
        selected = (
            select_moment(
                runtime, core_value=case["core_value"], sources=runtime_sources
            )
            if runtime
            else None
        )
        result["runtime"] = runtime.model_dump() if runtime else None
        result["reference"] = reference.model_dump() if reference else None
        result["selected"] = selected.model_dump() if selected else None
        result["status"] = "completed" if runtime and reference else "failed"
        if reference:
            valid_ids = {
                d.entry_id for d in reference.results if d.decision == "supportive"
            }
            result["reference_valid_ids"] = sorted(valid_ids)
            result["reference_no_example"] = not valid_ids
            result["task_retrieval_hit"] = bool(
                valid_ids & {e.entry_id for e in runtime_sources}
            )
            result["incorrect_displayed"] = bool(
                selected and selected.entry_id not in valid_ids
            )
            if selected and selected.entry_id in valid_ids:
                ref = next(
                    d for d in reference.results if d.entry_id == selected.entry_id
                )
                if (ref.evidence_quote, ref.quote_source) != (
                    selected.evidence_quote,
                    selected.quote_source,
                ):
                    quote_reference, attempts = await review(
                        provider,
                        case=case,
                        sources=[e for e in sources if e.entry_id == selected.entry_id],
                        role="quote_reference",
                        candidate=selected.model_dump(),
                    )
                    result["attempts"].extend(attempts)
                    result["quote_reference"] = (
                        quote_reference.model_dump() if quote_reference else None
                    )
                    result["incorrect_displayed"] = (
                        not quote_reference
                        or quote_reference.results[0].decision != "supportive"
                    )
            first = runtime_sources[0]
            retrieval_only = {
                "schema_version": "north-star-moment-review-v1",
                "core_value": case["core_value"],
                "results": [
                    {
                        "entry_id": first.entry_id,
                        "decision": "supportive",
                        "quote_source": "journal_entry",
                        "evidence_quote": first.journal_entry,
                        "reason_code": "observable_choice",
                    }
                ],
            }
            try:
                validate_review(
                    retrieval_only, core_value=case["core_value"], sources=[first]
                )
                result["retrieval_only_selected"] = True
                result["retrieval_only_correct"] = first.entry_id in valid_ids
            except ReviewValidationError:
                result["retrieval_only_selected"] = False
        print(
            f"{case['case_id']}: {result['status']}; selected={bool(selected)}; "
            f"incorrect={result.get('incorrect_displayed')}",
            flush=True,
        )
        (DIRECTORY / "cases").mkdir(exist_ok=True)
        (DIRECTORY / "cases" / f"{case['case_id'].replace(':', '_')}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return result


def summarize(results: list[dict], attempts: list[dict]) -> dict:
    shown = [r for r in results if r.get("selected")]
    no_example = [
        r for r in results if r.get("reference_no_example") and r["eligible_sources"]
    ]
    reference_valid = [r for r in results if r.get("reference_valid_ids")]
    retrieval_shown = [r for r in results if r.get("retrieval_only_selected")]
    precision = fraction(
        sum(not r.get("incorrect_displayed", True) for r in shown), len(shown)
    )
    retrieval_precision = fraction(
        sum(r.get("retrieval_only_correct", False) for r in retrieval_shown),
        len(retrieval_shown),
    )
    failures = {
        provider: fraction(
            sum(
                a["status"] != "completed"
                for a in attempts
                if a["provider"] == provider
            ),
            sum(a["provider"] == provider for a in attempts),
        )
        for provider in ("openai", "gemini")
    }
    summary = {
        "cases": len(results),
        "structurally_empty_cases": sum(not r["eligible_sources"] for r in results),
        "failed_cases": sum(r["status"] == "failed" for r in results),
        "precision": precision,
        "correct_no_card": fraction(
            sum(not r.get("selected") for r in no_example), len(no_example)
        ),
        "coverage": fraction(len(shown), len(results)),
        "task_retrieval_recall_at_3": fraction(
            sum(r["task_retrieval_hit"] for r in reference_valid), len(reference_valid)
        ),
        "retrieval_only_precision": retrieval_precision,
        "verification_lift": precision["rate"] - retrieval_precision["rate"]
        if precision["rate"] is not None and retrieval_precision["rate"] is not None
        else None,
        "unexpected_provider_failures": failures,
        "calculated_cost_usd": sum(a["calculated_cost_usd"] or 0 for a in attempts),
        "unmetered_attempts": sum(a["calculated_cost_usd"] is None for a in attempts),
        "attempts": len(attempts),
        "saved_persona_accepted": [
            r["case_id"]
            for r in shown
            if r["case_id"].split(":")[0]
            in {"8f83c818", "988d1a65", "02fb94f3", "11de77e8", "23d101f8"}
            and not r.get("incorrect_displayed", True)
        ],
    }
    summary["gate_passed"] = (
        precision["rate"] == 1
        and summary["correct_no_card"]["rate"] == 1
        and summary["failed_cases"] == 0
        and bool(summary["saved_persona_accepted"])
        and all(f["rate"] is not None and f["rate"] <= 0.05 for f in failures.values())
    )
    return summary


async def run() -> bool:
    manifest_path = DIRECTORY / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for path, digest in manifest["source_hashes"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    load_dotenv(ROOT / ".env")
    provider = BudgetedProvider()
    semaphore = asyncio.Semaphore(3)
    results = await asyncio.gather(
        *(run_case(c, provider, semaphore) for c in manifest["cases"])
    )
    attempts = provider.ledger.snapshot()["attempts"]
    summary = summarize(results, attempts)
    report = {
        "manifest_sha256": file_hash(manifest_path),
        "summary": summary,
        "cases": results,
        "runner_sha256": file_hash(Path(__file__)),
        "provider_sha256": file_hash(ROOT / "src/north_star/provider.py"),
    }
    (DIRECTORY / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return bool(summary["gate_passed"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return 0
    return 0 if asyncio.run(run()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
