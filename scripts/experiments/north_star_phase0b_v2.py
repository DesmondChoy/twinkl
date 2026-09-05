"""Prepare, replay, or resume a separately frozen NSM development experiment.

The 2026-09-05 runner and evidence remain immutable. --replay never requests
provider transport; --run permits missing work under the frozen budget and
requires a separately agreed development protocol. Neither changes the prompt.
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

from scripts.experiments.north_star_phase0b import (  # noqa: E402
    DIRECTORY as HISTORICAL_DIRECTORY,
)
from scripts.experiments.north_star_phase0b import (  # noqa: E402
    QUOTE_REVIEW,
    RETRIEVAL,
    summarize,
)
from scripts.experiments.north_star_phase0b_inputs import build_manifest  # noqa: E402
from src.north_star.provider import (  # noqa: E402
    POLICY_PATH,
    BudgetedProvider,
    BudgetError,
    BudgetLedger,
    ProviderAttempt,
    stable_hash,
)
from src.north_star.review import (  # noqa: E402
    ReviewBatch,
    ReviewValidationError,
    SourceEntry,
    build_review_prompt,
    review_json_schema,
    select_moment,
    validate_review,
)

EXECUTION_SOURCES = (
    "scripts/experiments/north_star_phase0b_v2.py",
    "scripts/experiments/north_star_phase0b_inputs.py",
    "scripts/experiments/north_star_phase0b.py",
    "scripts/experiments/north_star_phase0.py",
    "src/north_star/provider.py",
    "src/north_star/review.py",
    "src/wrangling/parse_wrangled_data.py",
    "src/models/judge.py",
)
PAID_LEDGER = Path("logs/experiments/north_star_development_budget_v2.json")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_directory(directory: Path) -> Path:
    directory = directory.resolve()
    historical = HISTORICAL_DIRECTORY.resolve()
    if directory == historical or historical in directory.parents:
        raise ValueError("Use a new directory; historical evidence is immutable")
    return directory


def _read_ledger(path: Path, policy: dict) -> dict:
    """Read without creating or rewriting the historical source ledger."""
    state: dict = json.loads(path.read_text())
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != "north-star-budget-v1"
        or state.get("policy_hash") != stable_hash(policy)
        or not isinstance(state.get("attempts"), list)
    ):
        raise ValueError("Missing or incompatible inherited budget accounting")
    grouped: dict[str, list[int]] = {}
    for row in state["attempts"]:
        attempt = ProviderAttempt.model_validate(row)
        grouped.setdefault(attempt.request_hash, []).append(attempt.attempt_number)
    if any(
        numbers != list(range(1, len(numbers) + 1))
        or len(numbers) > policy["max_attempts"]
        for numbers in grouped.values()
    ):
        raise ValueError("Invalid saved attempt sequence")
    return state


def _check_policy(policy: dict) -> None:
    # The preserved v1 transport hardcodes these settings. Do not mislabel a
    # changed policy as a new reasoning experiment before updating that adapter.
    if (
        policy.get("max_attempts") != 2
        or policy.get("sdk_retries") != 0
        or policy["runtime"].get("reasoning_effort") != "none"
        or policy["runtime"].get("service_tier") != "default"
        or policy["reference"].get("thinking_level") != "low"
    ):
        raise ValueError("Policy is unsupported by the preserved provider transport")


def _check_inherited(seed: dict, state: dict) -> None:
    saved = {(a["request_hash"], a["attempt_number"]): a for a in state["attempts"]}
    for previous in seed["attempts"]:
        attempt = saved.get((previous["request_hash"], previous["attempt_number"]))
        if (
            attempt is None
            or attempt["reserved_cost_usd"] != previous["reserved_cost_usd"]
        ):
            raise ValueError("Inherited budget reservation was removed or changed")
        if previous["status"] != "pending" and attempt != previous:
            # A completed response can have been interrupted before validation.
            invalidated = {
                **previous,
                "status": "invalid",
                "retryable": True,
                "error_type": "review_contract_invalid",
                "reused": False,
            }
            if previous["status"] != "completed" or attempt != invalidated:
                raise ValueError("Inherited terminal receipt changed")


def _paid_ledger(root: Path, policy_path: Path, seed: dict) -> BudgetLedger:
    """Every paid run in this project shares one locked cumulative ledger."""
    ledger = BudgetLedger(root / PAID_LEDGER, policy_path)
    origin_path = ledger.path.with_suffix(".seed.json")

    def initialize(state: dict) -> None:
        if "inherited_seed_hash" not in state:
            if state["attempts"]:
                raise ValueError("Existing cumulative ledger has unknown provenance")
            if origin_path.exists():
                raise ValueError("Cumulative ledger is missing; never reset its budget")
            state.update(seed)
            state["inherited_seed_hash"] = stable_hash(seed)
            # Write before the ledger: interruption may block recovery, but must
            # never silently reinitialize an allowance that could have been spent.
            with origin_path.open("x") as file:
                file.write(json.dumps({"seed_hash": stable_hash(seed)}) + "\n")
        _check_budget_origin(ledger.path, state)
        _check_inherited(seed, state)

    ledger.transact(initialize)
    return ledger


def _check_budget_origin(path: Path, state: dict) -> None:
    origin = json.loads(path.with_suffix(".seed.json").read_text())
    if origin.get("seed_hash") != state.get("inherited_seed_hash"):
        raise ValueError("Cumulative budget origin changed")


def _mirror_ledger(ledger: BudgetLedger, state: dict) -> None:
    def replace(saved: dict) -> None:
        saved.clear()
        saved.update(state)

    ledger.transact(replace)


def prepare(
    *,
    directory: Path,
    prior_ledger: Path,
    root: Path = ROOT,
    retrieval_path: Path = RETRIEVAL,
    policy_path: Path = POLICY_PATH,
) -> dict:
    """Validate first; the execution freeze is the final preparation marker."""
    directory = _check_directory(directory)
    if directory.exists():
        raise ValueError("Use a new empty run directory; do not overwrite a freeze")
    manifest = build_manifest(
        root=root, retrieval_path=retrieval_path, policy_path=policy_path
    )
    policy = json.loads(policy_path.read_text())
    _check_policy(policy)
    seed = _read_ledger(prior_ledger, policy)
    source_hashes = {name: file_hash(root / name) for name in EXECUTION_SOURCES}
    source_hashes.update(manifest["source_hashes"])
    manifest["preparation"] = {
        "retrieval_path": str(retrieval_path.resolve().relative_to(root.resolve())),
        "policy_path": str(policy_path.resolve().relative_to(root.resolve())),
    }
    seed_text = json.dumps(seed, indent=2, sort_keys=True) + "\n"
    directory.mkdir(parents=True, exist_ok=False)
    for name, contents in (
        ("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"),
        ("budget_seed.json", seed_text),
        ("budget.json", seed_text),
    ):
        with (directory / name).open("x") as file:
            file.write(contents)
    freeze = {
        "schema_version": "north-star-execution-v2",
        "frozen_at": datetime.now(UTC).isoformat(),
        "source_hashes": source_hashes,
        "manifest_sha256": file_hash(directory / "manifest.json"),
        "budget_seed_sha256": file_hash(directory / "budget_seed.json"),
        "inherited_attempts": len(seed["attempts"]),
        "paid_authorization": (
            "Preparation does not authorize a new paid experiment. Carry forward "
            "the complete prior ledger; resume only within the agreed protocol."
        ),
    }
    with (directory / "execution_freeze.json").open("x") as file:
        file.write(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return manifest


def verify_run(directory: Path, *, root: Path = ROOT) -> tuple[dict, Path]:
    """Verify inputs and execution before constructing a provider or ledger."""
    directory = _check_directory(directory)
    freeze = json.loads((directory / "execution_freeze.json").read_text())
    manifest_path = directory / "manifest.json"
    if (
        freeze.get("schema_version") != "north-star-execution-v2"
        or file_hash(manifest_path) != freeze["manifest_sha256"]
        or file_hash(directory / "budget_seed.json") != freeze["budget_seed_sha256"]
    ):
        raise ValueError("Execution freeze mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected_paths = set(EXECUTION_SOURCES) | set(manifest["source_hashes"])
    if set(freeze["source_hashes"]) != expected_paths:
        raise ValueError("Execution freeze has incomplete source coverage")
    for name, digest in freeze["source_hashes"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != digest:
            raise ValueError(f"Frozen execution source changed: {name}")
        if (
            name in manifest["source_hashes"]
            and manifest["source_hashes"][name] != digest
        ):
            raise ValueError(f"Manifest and execution freeze disagree: {name}")
    inputs = manifest["preparation"]
    retrieval_path, policy_path = (
        (root / inputs[key]).resolve() for key in ("retrieval_path", "policy_path")
    )
    for path in (retrieval_path, policy_path):
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Preparation path escapes source root")
    current = build_manifest(
        root=root, retrieval_path=retrieval_path, policy_path=policy_path
    )
    stored = {
        k: v for k, v in manifest.items() if k not in {"frozen_at", "preparation"}
    }
    if stored != {k: v for k, v in current.items() if k != "frozen_at"}:
        raise ValueError("Manifest differs from verified development inputs")
    policy = json.loads(policy_path.read_text())
    _check_policy(policy)
    seed = _read_ledger(directory / "budget_seed.json", policy)
    state = _read_ledger(directory / "budget.json", policy)
    _check_inherited(seed, state)
    return manifest, policy_path


def _request_attempts(
    provider: BudgetedProvider, request: dict
) -> list[ProviderAttempt]:
    key = stable_hash(request)
    return [
        ProviderAttempt.model_validate(row)
        for row in provider.ledger.snapshot()["attempts"]
        if row["request_hash"] == key
    ]


async def review(
    provider: BudgetedProvider,
    *,
    case: dict,
    sources: list[SourceEntry],
    role: Literal["runtime", "reference", "quote_reference"],
    candidate: dict | None = None,
    allow_paid: bool = False,
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
    request: dict[str, Any] = {
        "system": system,
        "prompt": prompt,
        "schema": review_json_schema(),
        "provider": "openai" if role == "runtime" else "gemini",
        "purpose": f"development-{role}",
        "policy_hash": stable_hash(provider.ledger.policy),
    }

    def receipts() -> list[dict]:
        return [
            {"attempt": attempt.model_dump(), "system": system, "prompt": prompt}
            for attempt in _request_attempts(provider, request)
        ]

    while True:
        saved = _request_attempts(provider, request)
        attempt = saved[-1] if saved else None
        pending = provider._inflight.get(stable_hash(request))
        if pending is not None:
            # Join this process's existing request; an orphaned ledger reservation
            # has no task to await and still fails closed below.
            try:
                await asyncio.shield(pending)
            except BudgetError:
                return None, receipts()
            continue
        if attempt and attempt.status == "completed":
            try:
                batch = validate_review(
                    attempt.raw_text or "",
                    core_value=case["core_value"],
                    sources=sources,
                )
                if candidate:
                    decision = batch.results[0]
                    if decision.decision == "supportive" and (
                        decision.evidence_quote != candidate["evidence_quote"]
                        or decision.quote_source != candidate["quote_source"]
                    ):
                        raise ValueError("Reference replaced the candidate quotation")
                return batch, receipts()
            except (ReviewValidationError, ValueError):
                attempt = provider.invalidate(attempt, "review_contract_invalid")
        if attempt and (
            attempt.status == "pending"
            or not attempt.retryable
            or attempt.attempt_number >= provider.ledger.policy["max_attempts"]
        ):
            # A pending reservation may have reached the provider. It cannot be
            # retried automatically or released on the assumption it was free.
            return None, receipts()
        if not allow_paid:
            return None, receipts()
        try:
            await provider.complete(
                system=system,
                prompt=prompt,
                schema=request["schema"],
                provider="openai" if role == "runtime" else "gemini",
                purpose=request["purpose"],
                retry=attempt is not None,
            )
        except BudgetError:
            # A concurrent reservation or exhausted budget blocks this request,
            # not reconstruction of other independently completed cases.
            return None, receipts()


async def run_case(
    case: dict,
    provider: BudgetedProvider,
    semaphore: asyncio.Semaphore,
    *,
    directory: Path,
    allow_paid: bool = False,
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
            provider,
            case=case,
            sources=runtime_sources,
            role="runtime",
            allow_paid=allow_paid,
        )
        result["attempts"].extend(attempts)
        reference, attempts = await review(
            provider,
            case=case,
            sources=sources,
            role="reference",
            allow_paid=allow_paid,
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
                        allow_paid=allow_paid,
                    )
                    result["attempts"].extend(attempts)
                    result["quote_reference"] = (
                        quote_reference.model_dump() if quote_reference else None
                    )
                    result["incorrect_displayed"] = (
                        not quote_reference
                        or quote_reference.results[0].decision != "supportive"
                    )
                    if quote_reference is None:
                        result["status"] = "failed"
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
        (directory / "cases").mkdir(exist_ok=True)
        (directory / "cases" / f"{case['case_id'].replace(':', '_')}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        return result


async def run(*, directory: Path, root: Path = ROOT, allow_paid: bool = False) -> bool:
    manifest, policy_path = verify_run(directory, root=root)
    directory = directory.resolve()
    policy = json.loads(policy_path.read_text())
    local_ledger = BudgetLedger(directory / "budget.json", policy_path)
    seed = _read_ledger(directory / "budget_seed.json", policy)
    if allow_paid:
        active_ledger = _paid_ledger(root, policy_path, seed)
        load_dotenv(root / ".env")
    else:
        active_ledger = local_ledger
        if (root / PAID_LEDGER).exists():
            # Recover receipts saved before a live process could write its report.
            state = _read_ledger(root / PAID_LEDGER, policy)
            _check_budget_origin(root / PAID_LEDGER, state)
            _check_inherited(seed, state)
            _mirror_ledger(local_ledger, state)
    provider = BudgetedProvider(active_ledger)
    semaphore = asyncio.Semaphore(3)
    results = await asyncio.gather(
        *(
            run_case(c, provider, semaphore, directory=directory, allow_paid=allow_paid)
            for c in manifest["cases"]
        )
    )
    ledger = provider.ledger.snapshot()
    if allow_paid:
        _mirror_ledger(local_ledger, ledger)
    keys = {
        receipt["attempt"]["request_hash"]
        for result in results
        for receipt in result["attempts"]
    }
    attempts = [a for a in ledger["attempts"] if a["request_hash"] in keys]
    summary = summarize(results, attempts)
    # The historical difference used source precision versus quote precision.
    # Keep reconstruction counts, but never claim that as verification lift.
    summary["retrieval_only_source_precision"] = summary.pop("retrieval_only_precision")
    summary.pop("verification_lift")
    summary["pending_reservations"] = sum(a["status"] == "pending" for a in attempts)
    report = {
        "schema_version": "north-star-development-report-v2",
        "manifest_sha256": file_hash(directory / "manifest.json"),
        "execution_freeze_sha256": file_hash(directory / "execution_freeze.json"),
        "summary": summary,
        "cases": results,
        "budget_accounting": {
            "inherited_attempts": json.loads(
                (directory / "execution_freeze.json").read_text()
            )["inherited_attempts"],
            "total_ledger_attempts": len(ledger["attempts"]),
            "spent_or_reserved_usd": sum(
                a["calculated_cost_usd"]
                if a["calculated_cost_usd"] is not None
                else a["reserved_cost_usd"]
                for a in ledger["attempts"]
            ),
        },
        "mode": "run" if allow_paid else "replay",
        "limitations": (
            "Pending reservations have unknown transport outcome and remain charged "
            "at their reservation. Retrieval-only source precision and selected "
            "quotation precision are distinct; no matched verification lift is claimed."
        ),
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
    action.add_argument("--replay", action="store_true", help="No provider calls")
    action.add_argument("--run", action="store_true", help="Resume agreed paid work")
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, default=RETRIEVAL)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--prior-ledger", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.prior_ledger is None:
            parser.error("--prepare requires --prior-ledger for cumulative accounting")
        manifest = prepare(
            directory=args.directory,
            prior_ledger=args.prior_ledger,
            retrieval_path=args.retrieval,
            policy_path=args.policy,
        )
        print(f"Frozen {manifest['case_count']} development cases; no provider calls")
        return 0
    return 0 if asyncio.run(run(directory=args.directory, allow_paid=args.run)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
