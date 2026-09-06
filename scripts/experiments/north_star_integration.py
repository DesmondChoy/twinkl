"""Freeze, prepare, and evaluate source-disclosed saved North Star Moments."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from scripts.experiments.north_star_luna import (  # noqa: E402
    attempt_bound,
    parse_response,
    write_json,
)
from src.demo.scenarios import (  # noqa: E402
    NORTH_STAR_REPORT_PATH,
    SELECTIONS,
    build_saved_north_star_request,
    build_scenario_fixture,
    export_scenarios,
)
from src.north_star import assessment, input_budget  # noqa: E402
from src.north_star.provider import (  # noqa: E402
    BudgetedProvider,
    BudgetLedger,
    ProviderAttempt,
    stable_hash,
)
from src.north_star.review import ReviewValidationError, SourceEntry  # noqa: E402
from src.north_star.runtime import (  # noqa: E402
    INTEGRATION_POLICY_PATH,
    NorthStarRecord,
    NorthStarRequest,
    OpenAINorthStarRuntime,
    source_review_requests,
    validate_north_star_record,
)

DIRECTORY = ROOT / NORTH_STAR_REPORT_PATH.parent
EXECUTION_FILES = (
    "scripts/experiments/north_star_integration.py",
    "scripts/experiments/north_star_luna.py",
    "src/demo/scenarios.py",
    "src/north_star/runtime.py",
    "src/north_star/provider.py",
    "src/north_star/assessment.py",
    "src/north_star/review.py",
    "src/north_star/input_budget.py",
    "config/schwartz_values.yaml",
)


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_request(request: dict[str, Any]) -> dict[str, Any]:
    """Keep the rubric and source context, without runtime outputs or reasoning."""
    return {
        **request,
        "role": "reference",
        "purpose": "nsm-integration-reference",
    }


def candidate_bound(request: dict[str, Any], policy: dict[str, Any]) -> float:
    """Bound any one full-source exact-quotation assessment before selection."""
    payload = json.loads(request["prompt"])
    bounds = []
    for source in payload["sources"]:
        raw = {
            "system": assessment.CANDIDATE_SYSTEM_PROMPT,
            "prompt": json.dumps(
                {
                    "core_value": payload["core_value"],
                    "user_phrase": payload["user_phrase"],
                    "approved_definition": payload["approved_definition"],
                    "sources": [source],
                    "proposed_quote": source["journal_entry"],
                    "envelope_margin": "x" * 2048,
                },
                ensure_ascii=False,
            ),
            "schema": assessment.candidate_json_schema(),
            "provider": "openai",
            "role": "reference",
            "purpose": "nsm-fresh-quote",
        }
        bounds.append(attempt_bound(raw, policy))
    return max(bounds, default=0)


def prepare(directory: Path = DIRECTORY, *, root: Path = ROOT) -> dict[str, Any]:
    """Freeze all 36 weeks and the complete paid envelope without provider calls."""
    if (directory / "manifest.json").exists():
        raise ValueError("Refuse to overwrite a prepared integration run")
    policy = json.loads(INTEGRATION_POLICY_PATH.read_text())
    cohort = json.loads((root / "config/evals/north_star_cohort.json").read_text())
    if {s.persona_id for s in SELECTIONS} & set(cohort["reserved_persona_ids"]):
        raise ValueError("Saved preparation must not use reserved Personas")
    cases = []
    input_paths = {"config/evals/north_star_cohort.json"}
    runtime_requests: dict[str, dict[str, Any]] = {}
    reference_requests: dict[str, dict[str, Any]] = {}
    quote_bounds = []
    for selection in SELECTIONS:
        fixture = build_scenario_fixture(root, selection, include_north_star=False)
        input_paths.update(fixture.scenario.manifest.source_files)
        raw_source = (
            root / f"logs/synthetic_data/persona_{selection.persona_id}.md"
        ).read_text()
        for week in fixture.scenario.weeks:
            request = build_saved_north_star_request(fixture, week.week_id)
            paid_requests = source_review_requests(request)
            for paid in paid_requests:
                payload = json.loads(paid["prompt"])
                for source in payload["sources"]:
                    if source["journal_entry"] not in raw_source:
                        raise ValueError("Runtime writing lacks synthetic provenance")
                    if source["nudge_response"] is not None:
                        raise ValueError(
                            "Legacy responses lack independent availability"
                        )
                runtime_requests[stable_hash(paid)] = paid
                reference = reference_request(paid)
                reference_requests[stable_hash(reference)] = reference
            quote_bounds.append(
                max((candidate_bound(r, policy) for r in paid_requests), default=0)
            )
            cases.append(
                {
                    "scenario_id": selection.scenario_id,
                    "persona_id": selection.persona_id,
                    "week_id": week.week_id,
                    "request": request.model_dump(mode="json"),
                    "runtime_requests": paid_requests,
                }
            )
    planned = [*runtime_requests.values(), *reference_requests.values()]
    maximum_new = policy["max_attempts"] * (
        sum(attempt_bound(r, policy) for r in planned) + sum(quote_bounds)
    )
    if maximum_new + policy["prior_spend_usd"] > policy["budget_usd"]:
        raise ValueError("Complete saved-Persona protocol exceeds cumulative budget")
    manifest = {
        "schema_version": "north-star-integration-run-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "cases": cases,
        "reference_requests": list(reference_requests.values()),
        "policy": policy,
        "input_hashes": {name: sha256(root / name) for name in sorted(input_paths)},
        "execution_hashes": {name: sha256(root / name) for name in EXECUTION_FILES},
        "preparation_hashes": {
            "src/demo/contracts.py": sha256(root / "src/demo/contracts.py")
        },
        "budget_preflight": {
            "maximum_new_usd": maximum_new,
            "maximum_cumulative_usd": maximum_new + policy["prior_spend_usd"],
            "runtime_requests": len(runtime_requests),
            "reference_requests": len(reference_requests),
            "maximum_candidate_requests": sum(bound > 0 for bound in quote_bounds),
            "includes_one_retry_each": True,
        },
        "scope": (
            "Saved synthetic Persona development and offline replay preparation; "
            "independent Luna xhigh evaluation is not a runtime release gate or "
            "human validation. Encouragement and reflection are reported separately. "
            "All five saved Personas and every closed week are included. Reserved "
            "histories and original evaluation artifacts remain unchanged."
        ),
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(directory / "manifest.json", manifest)
    return manifest


def verify(directory: Path = DIRECTORY, *, root: Path = ROOT) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((directory / "manifest.json").read_text())
    digest = manifest.pop("manifest_hash")
    if stable_hash(manifest) != digest:
        raise ValueError("Integration manifest changed")
    manifest["manifest_hash"] = digest
    for name, expected in {
        **manifest["input_hashes"],
        **manifest["execution_hashes"],
    }.items():
        if sha256(root / name) != expected:
            raise ValueError(f"Frozen integration source changed: {name}")
    if json.loads(INTEGRATION_POLICY_PATH.read_text()) != manifest["policy"]:
        raise ValueError("Integration budget policy changed")
    return manifest


async def execute_assessment(
    request: dict[str, Any],
    provider: BudgetedProvider,
    directory: Path,
    *,
    allow_paid: bool,
) -> Any:
    """Resume a validated reference receipt within the shared attempt limit."""
    policy = provider.ledger.policy
    key = stable_hash({**request, "policy_hash": stable_hash(policy)})
    attempts = [
        a for a in provider.ledger.snapshot()["attempts"] if a["request_hash"] == key
    ]
    if attempts and attempts[-1]["status"] == "completed":
        try:
            return parse_response(request, attempts[-1]["raw_text"])
        except ReviewValidationError as error:
            if not allow_paid:
                return None
            provider.invalidate(
                ProviderAttempt.model_validate(attempts[-1]), str(error)
            )
    if not allow_paid:
        return None
    counts = await input_budget.measure_requests(
        [request], policy, directory / "input-counts.json"
    )
    if (
        input_budget.validate_receipt(
            request, policy, counts["counts"].get(stable_hash(request))
        )
        > policy["input_token_limit"]
    ):
        raise ValueError("Complete assessment input exceeds 16,000 tokens")
    for number in range(len(attempts), policy["max_attempts"]):
        attempt = await provider.complete(**request, retry=number > 0)
        if attempt.status == "completed":
            try:
                return parse_response(request, attempt.raw_text or "")
            except ReviewValidationError as error:
                attempt = provider.invalidate(attempt, str(error))
        if not attempt.retryable:
            break
    return None


def selected_candidate_request(
    case: dict[str, Any], selected: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the selected exact quotation without runtime assessments."""
    payload = next(
        json.loads(request["prompt"])
        for request in case["runtime_requests"]
        if json.loads(request["prompt"])["core_value"] == selected["core_value"]
    )
    source = next(
        SourceEntry.model_validate(source)
        for source in payload["sources"]
        if source["entry_id"] == selected["entry_id"]
    )
    system, prompt = assessment.build_candidate_prompt(
        core_value=payload["core_value"],
        user_phrase=payload["user_phrase"],
        approved_definition=payload["approved_definition"],
        source=source,
        quote_source=selected["quote_source"],
        evidence_quote=selected["evidence_quote"],
    )
    return {
        "system": system,
        "prompt": prompt,
        "schema": assessment.candidate_json_schema(),
        "provider": "openai",
        "role": "reference",
        "purpose": "nsm-fresh-quote",
    }


def summarize_by_mode(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep semantic omissions distinct from transport and eligibility outcomes."""
    modes = {}
    for mode in ("reflection", "non_drift"):
        rows = [row for row in results if row["evaluation_context"] == mode]
        selected_rows = [row for row in rows if row["record"].get("selected")]
        no_example = [
            row for row in rows if row["reference_state"] == "no_supportive_example"
        ]
        modes[mode] = {
            "weeks": len(rows),
            "selected": len(selected_rows),
            "accepted": sum(row["selection_grade"] == "accepted" for row in rows),
            "rejected": sum(row["selection_grade"] == "rejected" for row in rows),
            "unresolved_selections": sum(
                row["selection_grade"] == "unresolved" for row in rows
            ),
            "no_supportive_histories": len(no_example),
            "correct_omissions": sum(
                row["record"]["status"] == "complete"
                and not row["record"].get("selected")
                for row in no_example
            ),
            "unresolved_histories": sum(
                row["reference_state"] == "unresolved" for row in rows
            ),
            "selected_modes": {
                selected_mode: sum(
                    row["record"]["mode"] == selected_mode for row in rows
                )
                for selected_mode in ("reflection", "encouragement", "reminder")
            },
            "failed_weeks": sum(row["record"]["status"] == "failed" for row in rows),
            "not_eligible_weeks": sum(
                row["record"]["status"] == "not_eligible" for row in rows
            ),
        }
    return modes


async def run(
    directory: Path = DIRECTORY, *, allow_paid: bool = False, concurrency: int = 3
) -> dict[str, Any]:
    """Save application records and separate AI evaluation with shared accounting."""
    manifest = verify(directory)
    provider = BudgetedProvider(
        BudgetLedger(directory / "budget.json", INTEGRATION_POLICY_PATH)
    )
    runtime = OpenAINorthStarRuntime(
        provider=provider, counts_path=directory / "input-counts.json"
    )
    records_path = directory / "records.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else {}
    if allow_paid:
        planned = manifest["reference_requests"] + [
            request
            for case in manifest["cases"]
            for request in case["runtime_requests"]
        ]
        counts = await input_budget.measure_requests(
            planned, manifest["policy"], directory / "input-counts.json"
        )
        if any(
            input_budget.validate_receipt(
                request,
                manifest["policy"],
                counts["counts"].get(stable_hash(request)),
            )
            > manifest["policy"]["input_token_limit"]
            for request in planned
        ):
            raise ValueError("Complete integration input exceeds 16,000 tokens")
    semaphore = asyncio.Semaphore(concurrency)

    async def prepare_record(case: dict[str, Any]) -> None:
        key = f"{case['scenario_id']}::{case['week_id']}"
        request = NorthStarRequest.model_validate(case["request"])
        if key in records:
            record = NorthStarRecord.model_validate(records[key])
        elif allow_paid:
            async with semaphore:
                record = await runtime(request)
            records[key] = record.model_dump(mode="json")
            write_json(records_path, records)
            print(f"{key}: {record.status}", flush=True)
        else:
            raise ValueError(f"Offline report lacks saved application record: {key}")
        validate_north_star_record(record, request)

    await asyncio.gather(*(prepare_record(case) for case in manifest["cases"]))

    async def evaluate(request: dict[str, Any]) -> Any:
        async with semaphore:
            response = await execute_assessment(
                request, provider, directory, allow_paid=allow_paid
            )
        print(
            f"{request['purpose']}: {'valid' if response else 'unavailable'}",
            flush=True,
        )
        return response

    reference_responses = await asyncio.gather(
        *(evaluate(request) for request in manifest["reference_requests"])
    )
    references = {
        stable_hash(request): response
        for request, response in zip(
            manifest["reference_requests"], reference_responses, strict=True
        )
    }
    candidate_requests = {}
    for case in manifest["cases"]:
        key = f"{case['scenario_id']}::{case['week_id']}"
        if records[key].get("selected"):
            candidate_requests[key] = selected_candidate_request(
                case,
                {
                    **records[key]["selected"],
                    "core_value": records[key]["core_value"],
                },
            )
    if allow_paid:
        counts = await input_budget.measure_requests(
            list(candidate_requests.values()),
            manifest["policy"],
            directory / "input-counts.json",
        )
        if any(
            input_budget.validate_receipt(
                request,
                manifest["policy"],
                counts["counts"].get(stable_hash(request)),
            )
            > manifest["policy"]["input_token_limit"]
            for request in candidate_requests.values()
        ):
            raise ValueError("Complete candidate input exceeds 16,000 tokens")
    candidate_responses = await asyncio.gather(
        *(evaluate(request) for request in candidate_requests.values())
    )
    candidates = dict(zip(candidate_requests, candidate_responses, strict=True))
    results = []
    for case in manifest["cases"]:
        key = f"{case['scenario_id']}::{case['week_id']}"
        record = records[key]
        selected = (
            {**record["selected"], "core_value": record["core_value"]}
            if record.get("selected")
            else None
        )
        reference_results = []
        for request in case["runtime_requests"]:
            response = references[stable_hash(reference_request(request))]
            reference_results.append(response.model_dump() if response else None)
        candidate = None
        primary = None
        if selected:
            candidate = candidates[key]
            primary = next(
                (
                    source
                    for batch in reference_results
                    if batch and batch["core_value"] == selected["core_value"]
                    for source in batch["results"]
                    if source["entry_id"] == selected["entry_id"]
                ),
                None,
            )
        supportive = any(
            source["decision"] == "supportive"
            for batch in reference_results
            if batch
            for source in batch["results"]
        )
        unresolved = any(batch is None for batch in reference_results) or any(
            source["decision"] == "abstain"
            for batch in reference_results
            if batch
            for source in batch["results"]
        )
        source_state = (
            "not_assessed"
            if not reference_results
            else "supportive_available"
            if supportive
            else "unresolved"
            if unresolved
            else "no_supportive_example"
        )
        grade = None
        if selected:
            if primary is None or candidate is None or primary["decision"] == "abstain":
                grade = "unresolved"
            elif (primary["decision"] == "supportive") != (
                candidate.source_reason == "observable_choice"
            ):
                grade = "unresolved"
            else:
                grade = (
                    "accepted"
                    if primary["decision"] == "supportive" and candidate.accepted
                    else "rejected"
                )
        results.append(
            {
                "scenario_id": case["scenario_id"],
                "persona_id": case["persona_id"],
                "week_id": case["week_id"],
                "record": record,
                "reference_assessments": reference_results,
                "candidate_assessment": candidate.model_dump() if candidate else None,
                "selection_grade": grade,
                "reference_state": source_state,
                "evaluation_context": (
                    "reflection"
                    if case["request"]["drift_result"]["delivery_state"]
                    == "active_drift"
                    else "non_drift"
                ),
            }
        )
    ledger = provider.ledger.snapshot()
    new_cost = sum(
        attempt["calculated_cost_usd"]
        if attempt["calculated_cost_usd"] is not None
        else attempt["reserved_cost_usd"]
        for attempt in ledger["attempts"]
    )
    modes = summarize_by_mode(results)
    report = {
        "schema_version": "north-star-integration-results-v1",
        "manifest_hash": manifest["manifest_hash"],
        "cases": results,
        "metrics_by_mode": modes,
        "generation_attempts": len(ledger["attempts"]),
        "new_spent_or_reserved_usd": new_cost,
        "cumulative_spent_or_reserved_usd": (
            new_cost + manifest["policy"]["prior_spend_usd"]
        ),
        "scope": manifest["scope"],
    }
    write_json(directory / "report.json", report)
    print(json.dumps({"metrics_by_mode": modes, "cost": new_cost}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "export"))
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
        print(json.dumps(result["budget_preflight"], indent=2))
    elif args.command == "export":
        verify()
        export_scenarios(ROOT)
    else:
        if args.allow_paid != (args.command == "run"):
            parser.error("Only run --allow-paid permits provider calls")
        if args.allow_paid:
            load_dotenv(ROOT / ".env", override=False)
        asyncio.run(run(allow_paid=args.allow_paid))


if __name__ == "__main__":
    main()
