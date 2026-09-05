"""Freeze and evaluate the revised NSM reviewer on development histories only.

The original experiment, its exhaustive references, and the v2 recovery runner
are preserved. New runtime requests use review_v2; both comparison paths grade
their exact displayed quotation against the same frozen reference protocol.
"""

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

from scripts.experiments import north_star_phase0b_v2 as recovery  # noqa: E402
from scripts.experiments.north_star_phase0 import fraction  # noqa: E402
from scripts.experiments.north_star_phase0b import (  # noqa: E402
    DIRECTORY as HISTORICAL_DIRECTORY,
)
from scripts.experiments.north_star_phase0b import QUOTE_REVIEW, RETRIEVAL  # noqa: E402
from scripts.experiments.north_star_phase0b_inputs import build_manifest  # noqa: E402
from src.north_star import review as original  # noqa: E402
from src.north_star import review_v2 as revised  # noqa: E402
from src.north_star.provider import (  # noqa: E402
    POLICY_PATH,
    BudgetedProvider,
    BudgetError,
    BudgetLedger,
    stable_hash,
)
from src.north_star.review import (  # noqa: E402
    ReviewBatch,
    ReviewValidationError,
    SourceEntry,
)

PROTOCOL_PATH = Path("docs/north_star/phase0b_revision_20260905.md")
HARDENING_VALIDATION = Path(
    "logs/experiments/reports/north_star_runner_hardening_20260905/validation.json"
)
EXECUTION_SOURCES = (
    *recovery.EXECUTION_SOURCES,
    "scripts/experiments/north_star_phase0b_v3.py",
    "src/north_star/review_v2.py",
    str(PROTOCOL_PATH),
)
SAVED_PERSONAS = {"8f83c818", "988d1a65", "02fb94f3", "11de77e8", "23d101f8"}


def sources_for(case: dict) -> list[SourceEntry]:
    return [
        SourceEntry(**row) for row in case["all_eligible_sources_in_retrieval_order"]
    ]


def request_for(
    case: dict, sources: list[SourceEntry], policy: dict, *, runtime: bool
) -> dict:
    module = revised if runtime else original
    system, prompt = module.build_review_prompt(
        core_value=case["core_value"],
        user_phrase=case["value"]["user_phrase"],
        approved_definition=case["value"]["definition"],
        sources=sources,
    )
    return {
        "system": system,
        "prompt": prompt,
        "schema": module.review_json_schema(),
        "provider": "openai" if runtime else "gemini",
        "purpose": "development-v3-runtime" if runtime else "development-reference",
        "policy_hash": stable_hash(policy),
    }


def frozen_reference(case: dict, state: dict, policy: dict) -> tuple[ReviewBatch, list]:
    """Revalidate the original exhaustive request/receipt, without any provider."""
    sources = sources_for(case)
    request = request_for(case, sources, policy, runtime=False)
    attempts = [
        row for row in state["attempts"] if row["request_hash"] == stable_hash(request)
    ]
    if not attempts or attempts[-1]["status"] != "completed":
        raise ValueError(f"Missing frozen exhaustive reference: {case['case_id']}")
    batch = original.validate_review(
        attempts[-1]["raw_text"] or "", core_value=case["core_value"], sources=sources
    )
    receipts = [
        {"attempt": row, "system": request["system"], "prompt": request["prompt"]}
        for row in attempts
    ]
    return batch, receipts


def source_key(case: dict, source: SourceEntry) -> str:
    return stable_hash(
        {
            "core_value": case["core_value"],
            "value": case["value"],
            "source": source.model_dump(),
        }
    )


def contradictory_references(
    cases: list[dict], state: dict, policy: dict
) -> list[dict]:
    grouped: dict[str, dict] = {}
    for case in cases:
        sources = sources_for(case)
        if not sources:
            continue
        batch, _ = frozen_reference(case, state, policy)
        by_id = {result.entry_id: result for result in batch.results}
        for source in sources:
            row = grouped.setdefault(
                source_key(case, source),
                {
                    "source_key": source_key(case, source),
                    "entry_id": source.entry_id,
                    "core_value": case["core_value"],
                    "decisions": [],
                },
            )
            row["decisions"].append(
                {"case_id": case["case_id"], **by_id[source.entry_id].model_dump()}
            )
    return [
        row
        for _, row in sorted(grouped.items())
        if {d["decision"] == "supportive" for d in row["decisions"]} == {True, False}
    ]


def quote_request(case: dict, source: SourceEntry, quote: dict, policy: dict) -> dict:
    request = request_for(case, [source], policy, runtime=False)
    request["system"] += QUOTE_REVIEW
    payload = json.loads(request["prompt"])
    payload.update(
        candidate_quote=quote["evidence_quote"], quote_source=quote["quote_source"]
    )
    request.update(
        prompt=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        purpose="development-quote_reference",
    )
    return request


def baseline_quote(case: dict, sources: list[SourceEntry]) -> dict | None:
    if not sources:
        return None
    first = sources[0]
    candidate = {
        "entry_id": first.entry_id,
        "decision": "supportive",
        "quote_source": "journal_entry",
        "evidence_quote": first.journal_entry,
        "reason_code": "observable_choice",
    }
    try:
        original.validate_review(
            {
                "schema_version": original.REVIEW_SCHEMA_VERSION,
                "core_value": case["core_value"],
                "results": [candidate],
            },
            core_value=case["core_value"],
            sources=[first],
        )
    except ReviewValidationError:
        return None
    return candidate


def attempt_bound(request: dict, policy: dict) -> float:
    settings = policy["runtime" if request["provider"] == "openai" else "reference"]
    input_bound = len(json.dumps(request, ensure_ascii=False).encode()) + 2048
    if input_bound > 64_000:
        raise ValueError("Planned request exceeds frozen input envelope")
    bound = (
        input_bound
        * settings["input_usd_per_million"]
        * (1.25 if request["provider"] == "openai" else 1)
        + policy["max_output_tokens"] * settings["output_usd_per_million"]
    ) / 1_000_000
    if bound > policy["per_attempt_usd"]:
        raise ValueError("Planned request exceeds per-attempt ceiling")
    return float(bound)


def budget_preflight(cases: list[dict], seed: dict, policy: dict) -> dict:
    """Bound every possible request and its retry, including unmetered responses."""
    planned = []
    for case in cases:
        sources = sources_for(case)
        if not sources:
            continue
        # All primary references must already be complete before new paid work.
        frozen_reference(case, seed, policy)
        runtime = attempt_bound(
            request_for(case, sources[:3], policy, runtime=True), policy
        )
        baseline = baseline_quote(case, sources)
        baseline_bound = (
            attempt_bound(quote_request(case, sources[0], baseline, policy), policy)
            if baseline
            else 0.0
        )
        # The entire attributed source bounds every possible exact substring.
        candidate_bound = max(
            attempt_bound(
                quote_request(
                    case,
                    source,
                    {
                        "evidence_quote": text,
                        "quote_source": quote_source,
                    },
                    policy,
                ),
                policy,
            )
            for source in sources[:3]
            for quote_source, text in (
                ("journal_entry", source.journal_entry),
                ("nudge_response", source.nudge_response),
            )
            if text
        )
        planned.append(
            {
                "case_id": case["case_id"],
                "runtime_attempt_usd": runtime,
                "baseline_attempt_usd": baseline_bound,
                "candidate_attempt_usd": candidate_bound,
            }
        )
    inherited = sum(
        a["calculated_cost_usd"]
        if a["calculated_cost_usd"] is not None
        else a["reserved_cost_usd"]
        for a in seed["attempts"]
    )
    maximum_new = sum(
        (
            r["runtime_attempt_usd"]
            + r["baseline_attempt_usd"]
            + r["candidate_attempt_usd"]
        )
        * policy["max_attempts"]
        for r in planned
    )
    if inherited + maximum_new > policy["budget_usd"]:
        raise ValueError("Complete revised protocol exceeds remaining paid envelope")
    return {
        "inherited_spent_or_reserved_usd": inherited,
        "maximum_new_spent_or_reserved_usd": maximum_new,
        "maximum_cumulative_usd": inherited + maximum_new,
        "authorized_total_usd": policy["budget_usd"],
        "per_attempt_ceiling_usd": policy["per_attempt_usd"],
        "max_attempts_per_request": policy["max_attempts"],
        "case_bounds": planned,
        "method": "UTF-8 input bound plus schema margin and maximum output tokens; "
        "all runtime, baseline and candidate requests include one retry. Existing "
        "exhaustive references are required and reused. No cache discount assumed.",
    }


def revised_manifest(
    *, root: Path, retrieval_path: Path, policy_path: Path, seed: dict
) -> dict:
    manifest = build_manifest(
        root=root, retrieval_path=retrieval_path, policy_path=policy_path
    )
    historical = root / HISTORICAL_DIRECTORY.relative_to(ROOT)
    previous = json.loads((historical / "manifest.json").read_text())
    if manifest["cases"] != previous["cases"]:
        raise ValueError("Revised cases differ from frozen development sources")
    frozen = json.loads((historical / "execution_freeze.json").read_text())
    for name, digest in (frozen["source_hashes"] | previous["source_hashes"]).items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or recovery.file_hash(path) != digest
        ):
            raise ValueError(f"Historical execution source changed: {name}")
    validation = json.loads((historical / "validation.json").read_text())
    report_path = historical / "report.json"
    if (
        recovery.file_hash(report_path)
        != validation["hashes"][str(report_path.relative_to(root))]
    ):
        raise ValueError("Historical report differs from its recorded validation hash")
    hardening = json.loads((root / HARDENING_VALIDATION).read_text())
    if (
        recovery.file_hash(historical / "budget.json")
        != hardening["historical_replay"]["input_ledger_sha256"]
    ):
        raise ValueError("Historical budget differs from verified original receipts")
    policy = json.loads(policy_path.read_text())
    recovery._check_policy(policy)
    historical_seed = recovery._read_ledger(historical / "budget.json", policy)
    recovery._check_inherited(historical_seed, seed)
    manifest.update(
        schema_version="north-star-development-v3",
        prompt_version=revised.REVIEW_PROMPT_VERSION,
        schema=revised.review_json_schema(),
        reference_protocol="Reuse frozen exhaustive Gemini references without new "
        "labels. Require primary support and exact-quotation approval for both "
        "paths. Known contradictory identical source/value references are unresolved "
        "and cannot approve a displayed quotation. No runtime access to references.",
        retrieval_only_rule="Display the entire first ranked Journal Entry when "
        "code-valid; grade that exact quotation with the same primary-support and "
        "candidate-quote rules as runtime selections. Never substitute a "
        "reference quote.",
        coverage_scope="All 33 frozen development Drift episodes, including empty "
        "earlier histories. This is episode-proxy coverage; closed-week eligibility "
        "and application Core Value priority coverage await integration.",
        unresolved_reference_sources=contradictory_references(
            manifest["cases"], seed, policy
        ),
        budget_preflight=budget_preflight(manifest["cases"], seed, policy),
    )
    # Preserve every original report, receipt, image and executed source by hash.
    for path in historical.rglob("*"):
        if path.is_file():
            manifest["source_hashes"][str(path.relative_to(root))] = recovery.file_hash(
                path
            )
    manifest["source_hashes"][str(HARDENING_VALIDATION)] = recovery.file_hash(
        root / HARDENING_VALIDATION
    )
    return manifest


def prepare(
    *,
    directory: Path,
    prior_ledger: Path,
    root: Path = ROOT,
    retrieval_path: Path = RETRIEVAL,
    policy_path: Path = POLICY_PATH,
) -> dict:
    directory = recovery._check_directory(directory)
    if directory.exists():
        raise ValueError("Use a new empty run directory; never overwrite a freeze")
    policy = json.loads(policy_path.read_text())
    seed = recovery._read_ledger(prior_ledger, policy)
    if (root / recovery.PAID_LEDGER).exists():
        current = recovery._read_ledger(root / recovery.PAID_LEDGER, policy)
        recovery._check_budget_origin(root / recovery.PAID_LEDGER, current)
        recovery._check_inherited(current, seed)
    manifest = revised_manifest(
        root=root, retrieval_path=retrieval_path, policy_path=policy_path, seed=seed
    )
    manifest["preparation"] = {
        "retrieval_path": str(retrieval_path.resolve().relative_to(root.resolve())),
        "policy_path": str(policy_path.resolve().relative_to(root.resolve())),
    }
    source_hashes = {
        name: recovery.file_hash(root / name) for name in EXECUTION_SOURCES
    }
    source_hashes.update(manifest["source_hashes"])
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("manifest.json", manifest),
        ("budget_seed.json", seed),
        ("budget.json", seed),
    ):
        with (directory / name).open("x") as file:
            file.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    freeze = {
        "schema_version": "north-star-execution-v3",
        "frozen_at": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes,
        "manifest_sha256": recovery.file_hash(directory / "manifest.json"),
        "budget_seed_sha256": recovery.file_hash(directory / "budget_seed.json"),
        "paid_authorization": "User requested twinkl-fz34.1 implementation on "
        "2026-09-05; one frozen development revision under the existing USD20 total, "
        "USD0.25 per attempt, one-retry limits. Stop dependent work on gate failure.",
    }
    with (directory / "execution_freeze.json").open("x") as file:
        file.write(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_run(directory: Path, *, root: Path = ROOT) -> tuple[dict, Path]:
    directory = recovery._check_directory(directory)
    freeze = json.loads((directory / "execution_freeze.json").read_text())
    if (
        freeze.get("schema_version") != "north-star-execution-v3"
        or recovery.file_hash(directory / "manifest.json") != freeze["manifest_sha256"]
        or recovery.file_hash(directory / "budget_seed.json")
        != freeze["budget_seed_sha256"]
    ):
        raise ValueError("Execution freeze mismatch")
    manifest = json.loads((directory / "manifest.json").read_text())
    if set(freeze["source_hashes"]) != set(EXECUTION_SOURCES) | set(
        manifest["source_hashes"]
    ):
        raise ValueError("Incomplete execution freeze source coverage")
    for name, digest in freeze["source_hashes"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or recovery.file_hash(path) != digest
        ):
            raise ValueError(f"Frozen execution source changed: {name}")
        if (
            name in manifest["source_hashes"]
            and manifest["source_hashes"][name] != digest
        ):
            raise ValueError("Manifest and execution freeze disagree")
    paths = [
        (root / manifest["preparation"][key]).resolve()
        for key in ("retrieval_path", "policy_path")
    ]
    if not all(path.is_relative_to(root.resolve()) for path in paths):
        raise ValueError("Preparation path escapes source root")
    retrieval_path, policy_path = paths
    policy = json.loads(policy_path.read_text())
    seed = recovery._read_ledger(directory / "budget_seed.json", policy)
    current = revised_manifest(
        root=root, retrieval_path=retrieval_path, policy_path=policy_path, seed=seed
    )
    stored = {
        k: v for k, v in manifest.items() if k not in {"frozen_at", "preparation"}
    }
    if stored != {k: v for k, v in current.items() if k != "frozen_at"}:
        raise ValueError("Manifest differs from verified development inputs")
    recovery._check_inherited(
        seed, recovery._read_ledger(directory / "budget.json", policy)
    )
    return manifest, policy_path


async def review_runtime(
    provider: BudgetedProvider,
    *,
    case: dict,
    sources: list[SourceEntry],
    allow_paid: bool = False,
) -> tuple[ReviewBatch | None, list[dict]]:
    request = request_for(case, sources, provider.ledger.policy, runtime=True)

    def receipts() -> list[dict]:
        rows = []
        for attempt in recovery._request_attempts(provider, request):
            errors: list[str] = []
            if attempt.raw_text:
                try:
                    revised.validate_review(
                        attempt.raw_text, core_value=case["core_value"], sources=sources
                    )
                except ReviewValidationError as exc:
                    errors = list(exc.errors)
            rows.append(
                {
                    "attempt": attempt.model_dump(),
                    "system": request["system"],
                    "prompt": request["prompt"],
                    "validation_errors": errors,
                }
            )
        return rows

    while True:
        saved = recovery._request_attempts(provider, request)
        attempt = saved[-1] if saved else None
        pending = provider._inflight.get(stable_hash(request))
        if pending is not None:
            try:
                await asyncio.shield(pending)
            except BudgetError:
                return None, receipts()
            continue
        if attempt and attempt.status == "completed":
            try:
                batch = revised.validate_review(
                    attempt.raw_text or "",
                    core_value=case["core_value"],
                    sources=sources,
                )
                return batch, receipts()
            except ReviewValidationError:
                attempt = provider.invalidate(attempt, "review_contract_invalid")
        if attempt and (
            attempt.status == "pending"
            or not attempt.retryable
            or attempt.attempt_number >= provider.ledger.policy["max_attempts"]
        ):
            return None, receipts()
        if not allow_paid:
            return None, receipts()
        try:
            await provider.complete(
                system=request["system"],
                prompt=request["prompt"],
                schema=request["schema"],
                provider="openai",
                purpose=request["purpose"],
                retry=attempt is not None,
            )
        except BudgetError:
            return None, receipts()


async def grade_quote(
    *,
    case: dict,
    candidate: dict | None,
    reference: ReviewBatch,
    provider: BudgetedProvider,
    unresolved_keys: set[str],
    allow_paid: bool,
) -> tuple[dict, list[dict]]:
    if candidate is None:
        return {"accepted": False, "status": "not_selected"}, []
    source = next(s for s in sources_for(case) if s.entry_id == candidate["entry_id"])
    decision = next(d for d in reference.results if d.entry_id == source.entry_id)
    if source_key(case, source) in unresolved_keys:
        return {"accepted": False, "status": "contradictory_primary_reference"}, []
    if decision.decision != "supportive":
        return {
            "accepted": False,
            "status": "primary_" + decision.decision,
            "reason_code": decision.reason_code,
        }, []
    if (candidate["evidence_quote"], candidate["quote_source"]) == (
        decision.evidence_quote,
        decision.quote_source,
    ):
        return {"accepted": True, "status": "exact_primary_quote"}, []
    checked, receipts = await recovery.review(
        provider,
        case=case,
        sources=[source],
        role="quote_reference",
        candidate=candidate,
        allow_paid=allow_paid,
    )
    return {
        "accepted": bool(checked and checked.results[0].decision == "supportive"),
        "status": "completed" if checked else "failed",
        "reason_code": checked.results[0].reason_code if checked else None,
        "quote_reference": checked.model_dump() if checked else None,
    }, receipts


async def run_case(
    case: dict,
    provider: BudgetedProvider,
    semaphore: asyncio.Semaphore,
    *,
    seed: dict,
    unresolved_keys: set[str],
    directory: Path,
    allow_paid: bool = False,
) -> dict:
    async with semaphore:
        sources = sources_for(case)
        result: dict[str, Any] = {
            "case_id": case["case_id"],
            "core_value": case["core_value"],
            "eligible_sources": len(sources),
            "attempts": [],
            "selected": None,
            "retrieval_only_selected": None,
            "status": "no_earlier_writing",
        }
        if sources:
            reference, receipts = frozen_reference(case, seed, provider.ledger.policy)
            result["attempts"].extend(receipts)
            runtime, receipts = await review_runtime(
                provider, case=case, sources=sources[:3], allow_paid=allow_paid
            )
            result["attempts"].extend(receipts)
            selection = (
                original.select_moment(
                    runtime, core_value=case["core_value"], sources=sources[:3]
                )
                if runtime
                else None
            )
            selected = selection.model_dump() if selection else None
            baseline = baseline_quote(case, sources)
            grade, receipts = await grade_quote(
                case=case,
                candidate=selected,
                reference=reference,
                provider=provider,
                unresolved_keys=unresolved_keys,
                allow_paid=allow_paid,
            )
            result["attempts"].extend(receipts)
            baseline_grade, receipts = await grade_quote(
                case=case,
                candidate=baseline,
                reference=reference,
                provider=provider,
                unresolved_keys=unresolved_keys,
                allow_paid=allow_paid,
            )
            result["attempts"].extend(receipts)
            valid_ids = {
                d.entry_id for d in reference.results if d.decision == "supportive"
            }
            result.update(
                runtime=runtime.model_dump() if runtime else None,
                reference=reference.model_dump(),
                selected=selected,
                grade=grade,
                retrieval_only_selected=baseline,
                retrieval_only_grade=baseline_grade,
                reference_valid_ids=sorted(valid_ids),
                reference_no_example=not valid_ids,
                reference_has_abstention=any(
                    d.decision == "abstain" for d in reference.results
                ),
                reference_all_abstain=all(
                    d.decision == "abstain" for d in reference.results
                ),
                task_retrieval_hit=bool(valid_ids & {s.entry_id for s in sources[:3]}),
                status="failed"
                if runtime is None
                or "failed" in {grade["status"], baseline_grade["status"]}
                else "completed",
            )
        (directory / "cases").mkdir(exist_ok=True)
        (directory / "cases" / f"{case['case_id'].replace(':', '_')}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        print(
            f"{case['case_id']}: {result['status']}; "
            f"selected={bool(result['selected'])}; "
            f"accepted={result.get('grade', {}).get('accepted', False)}",
            flush=True,
        )
        return result


def summarize(
    results: list[dict], attempts: list[dict], new_attempts: list[dict]
) -> dict:
    shown = [r for r in results if r["selected"]]
    baseline = [r for r in results if r["retrieval_only_selected"]]
    no_example = [r for r in results if r.get("reference_no_example")]
    valid = [r for r in results if r.get("reference_valid_ids")]
    runtime_decisions = [
        d for r in results if r.get("runtime") for d in r["runtime"]["results"]
    ]
    precision = fraction(sum(r["grade"]["accepted"] for r in shown), len(shown))
    baseline_precision = fraction(
        sum(r["retrieval_only_grade"]["accepted"] for r in baseline), len(baseline)
    )
    failures = {
        provider: fraction(
            sum(
                a["status"] != "completed"
                for a in new_attempts
                if a["provider"] == provider
            ),
            sum(a["provider"] == provider for a in new_attempts),
        )
        for provider in ("openai", "gemini")
    }
    saved = [
        r["case_id"]
        for r in shown
        if r["case_id"].split(":")[0] in SAVED_PERSONAS and r["grade"]["accepted"]
    ]
    summary = {
        "cases": len(results),
        "failed_cases": sum(r["status"] == "failed" for r in results),
        "structurally_empty_cases": sum(not r["eligible_sources"] for r in results),
        "precision": precision,
        "incorrect_displayed": len(shown) - precision["numerator"],
        "correct_no_card": fraction(
            sum(not r["selected"] for r in no_example), len(no_example)
        ),
        "no_example_reference_strata": {
            "with_abstentions": sum(r["reference_has_abstention"] for r in no_example),
            "all_abstain": sum(r["reference_all_abstain"] for r in no_example),
            "resolved_rejections_only": sum(
                not r["reference_has_abstention"] for r in no_example
            ),
        },
        "coverage": fraction(len(shown), len(results)),
        "retrieval_only_quotation_precision": baseline_precision,
        "retrieval_only_coverage": fraction(len(baseline), len(results)),
        "quotation_precision_difference": precision["rate"] - baseline_precision["rate"]
        if precision["rate"] is not None and baseline_precision["rate"] is not None
        else None,
        "task_retrieval_recall_at_3": fraction(
            sum(r["task_retrieval_hit"] for r in valid), len(valid)
        ),
        "abstention": fraction(
            sum(d["decision"] == "abstain" for d in runtime_decisions),
            len(runtime_decisions),
        ),
        "wrong_value_reference_reasons": fraction(
            sum(r["grade"].get("reason_code") == "wrong_value" for r in shown),
            len(shown),
        ),
        "unresolved_reference_selections": sum(
            r["grade"]["status"] == "contradictory_primary_reference" for r in shown
        ),
        "unexpected_provider_failures": failures,
        "evaluated_unique_attempts": len(attempts),
        "new_actual_attempts": len(new_attempts),
        "new_calculated_cost_usd": sum(
            a["calculated_cost_usd"] or 0 for a in new_attempts
        ),
        "new_unmetered_attempts": sum(
            a["calculated_cost_usd"] is None for a in new_attempts
        ),
        "pending_reservations": sum(a["status"] == "pending" for a in attempts),
        "saved_persona_accepted": saved,
        "displayed_code_checks": {
            "selected_quotations": len(shown),
            "enforcement": "Selection requires full-batch exact-quotation validation; "
            "preparation and execution verification require source owner, date and "
            "stored-order agreement. These are construction-enforced checks, not "
            "separately measured adversarial failure counts.",
        },
    }
    summary["gate_passed"] = (
        precision["rate"] == 1
        and summary["correct_no_card"]["rate"] == 1
        and summary["failed_cases"] == 0
        and bool(saved)
        and all(f["denominator"] == 0 or f["rate"] <= 0.05 for f in failures.values())
    )
    return summary


async def run(*, directory: Path, root: Path = ROOT, allow_paid: bool = False) -> bool:
    manifest, policy_path = verify_run(directory, root=root)
    directory = directory.resolve()
    policy = json.loads(policy_path.read_text())
    seed = recovery._read_ledger(directory / "budget_seed.json", policy)
    local_ledger = BudgetLedger(directory / "budget.json", policy_path)
    if allow_paid:
        active = recovery._paid_ledger(root, policy_path, seed)
        load_dotenv(root / ".env")
    else:
        active = local_ledger
        if (root / recovery.PAID_LEDGER).exists():
            state = recovery._read_ledger(root / recovery.PAID_LEDGER, policy)
            recovery._check_budget_origin(root / recovery.PAID_LEDGER, state)
            recovery._check_inherited(seed, state)
            recovery._mirror_ledger(local_ledger, state)
    provider = BudgetedProvider(active)
    semaphore = asyncio.Semaphore(3)
    unresolved = {r["source_key"] for r in manifest["unresolved_reference_sources"]}
    results = await asyncio.gather(
        *(
            run_case(
                case,
                provider,
                semaphore,
                seed=seed,
                unresolved_keys=unresolved,
                directory=directory,
                allow_paid=allow_paid,
            )
            for case in manifest["cases"]
        )
    )
    ledger = provider.ledger.snapshot()
    if allow_paid:
        recovery._mirror_ledger(local_ledger, ledger)
    keys = {r["attempt"]["request_hash"] for case in results for r in case["attempts"]}
    attempts = [a for a in ledger["attempts"] if a["request_hash"] in keys]
    inherited_ids = {(a["request_hash"], a["attempt_number"]) for a in seed["attempts"]}
    new_attempts = [
        a
        for a in attempts
        if (a["request_hash"], a["attempt_number"]) not in inherited_ids
    ]
    summary = summarize(results, attempts, new_attempts)
    report = {
        "schema_version": "north-star-development-report-v3",
        "manifest_sha256": recovery.file_hash(directory / "manifest.json"),
        "execution_freeze_sha256": recovery.file_hash(
            directory / "execution_freeze.json"
        ),
        "summary": summary,
        "cases": results,
        "mode": "run" if allow_paid else "replay",
        "budget_accounting": {
            "inherited_attempts": len(seed["attempts"]),
            "total_ledger_attempts": len(ledger["attempts"]),
            "spent_or_reserved_usd": sum(
                a["calculated_cost_usd"]
                if a["calculated_cost_usd"] is not None
                else a["reserved_cost_usd"]
                for a in ledger["attempts"]
            ),
        },
        "limitations": "Synthetic development evidence with frozen independent "
        "AI references, "
        "including abstentions and cross-case contradictions. No human validation or "
        "reserved final test. Precision difference is descriptive and reported "
        "alongside "
        "coverage, not causal lift. Coverage uses known development episodes, not the "
        "unimplemented closed-week/application-priority population. Failures before "
        "successful retries are retained; reused receipts are not new provider calls.",
    }
    (directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return bool(summary["gate_passed"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--replay", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--prior-ledger", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.prior_ledger is None:
            parser.error("--prepare requires --prior-ledger")
        manifest = prepare(directory=args.directory, prior_ledger=args.prior_ledger)
        print(json.dumps(manifest["budget_preflight"], indent=2))
        return 0
    return 0 if asyncio.run(run(directory=args.directory, allow_paid=args.run)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
