"""Fresh, full-history NSM evaluation with independently assessed exact quotes.

Prepare freezes development-only sources and requests. Run measures complete
inputs, enforces the budget, and saves resumable provider receipts. Report is a
transport-free reconstruction from those receipts; no historical labels enter.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from scripts.experiments.north_star_phase0 import (  # noqa: E402
    EPISODES,
    VALUES,
    earlier_entries,
    fraction,
    sha256,
)
from scripts.experiments.north_star_phase0b_inputs import (  # noqa: E402
    _history,
    _validate_episode,
)
from src.north_star import assessment, input_budget  # noqa: E402
from src.north_star.provider import (  # noqa: E402
    BudgetedProvider,
    BudgetLedger,
    stable_hash,
)
from src.north_star.review import ReviewValidationError, SourceEntry  # noqa: E402

COHORT = Path("config/evals/north_star_cohort.json")
POLICY = Path("config/evals/north_star_luna_20260905.json")
DEFAULT_DIRECTORY = ROOT / "logs/experiments/reports/north_star_luna_20260905"
SAVED_PERSONAS = {"8f83c818", "988d1a65", "02fb94f3", "11de77e8", "23d101f8"}
EXECUTION_SOURCES = (
    "scripts/experiments/north_star_luna.py",
    "scripts/experiments/north_star_phase0.py",
    "scripts/experiments/north_star_phase0b_inputs.py",
    "scripts/experiments/north_star_phase0b.py",
    "src/north_star/assessment.py",
    "src/north_star/review.py",
    "src/north_star/provider.py",
    "src/north_star/input_budget.py",
    "src/wrangling/parse_wrangled_data.py",
    "docs/north_star/luna_evaluation_20260905.md",
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_request(value: dict, sources: list[SourceEntry], role: str) -> dict:
    system, prompt = assessment.build_source_prompt(**value, sources=sources)
    return {
        "system": system,
        "prompt": prompt,
        "schema": assessment.source_json_schema(),
        "provider": "openai",
        "role": role,
        "purpose": "nsm-fresh-runtime" if role == "runtime" else "nsm-fresh-source",
    }


def candidate_request(case: dict, selected: dict) -> dict:
    source = next(
        SourceEntry.model_validate(s)
        for s in case["sources"]
        if s["entry_id"] == selected["entry_id"]
    )
    system, prompt = assessment.build_candidate_prompt(
        **case["value"],
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


def build_inputs(root: Path = ROOT) -> dict:
    cohort = json.loads((root / COHORT).read_text())
    if sha256(root / EPISODES) != cohort["source_episode_sha256"]:
        raise ValueError("The frozen episode corpus changed")
    development = set(cohort["development_persona_ids"])
    reserved = set(cohort["reserved_persona_ids"])
    if not development or development & reserved:
        raise ValueError("Invalid or overlapping cohort")
    # Filter identifiers before parsing any Persona writing.
    episodes = (
        pl.read_parquet(root / EPISODES)
        .filter(pl.col("persona_id").is_in(sorted(development)))
        .to_dicts()
    )
    values = {
        name.lower().replace("-", "_"): {
            "core_value": name.lower().replace("-", "_"),
            "user_phrase": definition["user_phrase"].strip(),
            "approved_definition": definition["definition"].strip(),
        }
        for name, definition in yaml.safe_load((root / VALUES).read_text())[
            "values"
        ].items()
    }
    cases: list[dict] = []
    groups: dict[str, dict] = {}
    paths = {str(COHORT), str(EPISODES), str(VALUES)}
    for persona in sorted({e["persona_id"] for e in episodes}):
        entries = _history(root, persona)
        synthetic = root / f"logs/synthetic_data/persona_{persona}.md"
        original_text = synthetic.read_text()
        paths.update(
            {f"logs/wrangled/persona_{persona}.md", str(synthetic.relative_to(root))}
        )
        for episode in sorted(
            (e for e in episodes if e["persona_id"] == persona),
            key=lambda e: e["episode_id"],
        ):
            _validate_episode(episode, entries)
            eligible = sorted(
                earlier_entries(entries, episode),
                key=lambda s: s["t_index"],
                reverse=True,
            )
            sources, group_ids = [], []
            for entry in eligible:
                text = entry["initial_entry"]
                if text not in original_text:
                    raise ValueError(
                        "Eligible writing lacks verbatim synthetic provenance"
                    )
                source = SourceEntry(
                    entry_id=f"{persona}:entry:{entry['t_index']}", journal_entry=text
                )
                value = values[episode["dimension"]]
                key = stable_hash({"value": value, "source": source.model_dump()})
                groups[key] = {
                    "group_id": key,
                    "value": value,
                    "source": source.model_dump(),
                }
                sources.append(source.model_dump())
                group_ids.append(key)
            cases.append(
                {
                    "case_id": episode["episode_id"],
                    "persona_id": persona,
                    "episode": episode,
                    "value": values[episode["dimension"]],
                    "sources": sources,
                    "group_ids": group_ids,
                    "runtime_request": source_request(
                        values[episode["dimension"]],
                        [SourceEntry.model_validate(s) for s in sources],
                        "runtime",
                    )
                    if sources
                    else None,
                }
            )
    by_value: dict[str, list[dict]] = defaultdict(list)
    for group in sorted(
        groups.values(),
        key=lambda g: (g["value"]["core_value"], g["source"]["entry_id"]),
    ):
        by_value[group["value"]["core_value"]].append(group)
    batches = []
    for group_list in by_value.values():
        for index in range(0, len(group_list), 4):
            batch = group_list[index : index + 4]
            request = source_request(
                batch[0]["value"],
                [SourceEntry.model_validate(g["source"]) for g in batch],
                "reference",
            )
            batches.append(
                {"group_ids": [g["group_id"] for g in batch], "request": request}
            )
    return {
        "cases": cases,
        "source_groups": list(groups.values()),
        "reference_batches": batches,
        "input_hashes": {name: sha256(root / name) for name in sorted(paths)},
        "reserved_persona_ids": sorted(reserved),
        "provenance": (
            "Every eligible Journal Entry verified verbatim in its "
            "synthetic source. No reserved writing parsed; no generation metadata "
            "enter prompts. Nudge responses excluded because independent "
            "availability is unknown."
        ),
    }


def attempt_bound(request: dict, policy: dict) -> float:
    settings = policy[request["role"]]
    envelope = {**request, "policy_hash": stable_hash(policy)}
    inputs = len(json.dumps(envelope, ensure_ascii=False).encode()) + 2048
    if inputs > 64_000:
        raise ValueError("Request exceeds conservative provider input envelope")
    bound = (
        inputs * settings["input_usd_per_million"] * 1.25
        + policy["max_output_tokens"] * settings["output_usd_per_million"]
    ) / 1_000_000
    if bound > policy["per_attempt_usd"]:
        raise ValueError("Request exceeds per-attempt budget")
    return float(bound)


def prepare(directory: Path, root: Path = ROOT) -> dict:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Refuse to overwrite an existing experiment")
    policy = json.loads((root / POLICY).read_text())
    data = build_inputs(root)
    planned = [b["request"] for b in data["reference_batches"]] + [
        c["runtime_request"] for c in data["cases"] if c["runtime_request"]
    ]
    bounds = [attempt_bound(r, policy) for r in planned]
    # A full duplicated source, full schema and extra envelope text bounds any
    # future exact-substring candidate request without selecting a quote now.
    for case in data["cases"]:
        if not case["sources"]:
            continue
        maximum = 0.0
        for source in case["sources"]:
            raw = {
                "system": assessment.CANDIDATE_SYSTEM_PROMPT,
                "prompt": json.dumps(
                    {
                        "value": case["value"],
                        "source": source,
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
            maximum = max(maximum, attempt_bound(raw, policy))
        bounds.append(maximum)
    maximum_new = sum(bounds) * policy["max_attempts"]
    if maximum_new + policy["prior_spend_usd"] > policy["budget_usd"]:
        raise ValueError("Entire protocol including retries exceeds total budget")
    manifest = {
        "schema_version": "north-star-luna-run-v1",
        "created_at": datetime.now(UTC).isoformat(),
        **data,
        "policy": policy,
        "execution_hashes": {name: sha256(root / name) for name in EXECUTION_SOURCES},
        "budget_preflight": {
            "maximum_new_usd": maximum_new,
            "maximum_cumulative_usd": maximum_new + policy["prior_spend_usd"],
            "max_possible_requests": len(bounds),
            "includes_one_retry_each": True,
        },
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "policy.json", policy)
    return manifest


def verify(directory: Path, root: Path = ROOT) -> dict:
    manifest: dict = json.loads((directory / "manifest.json").read_text())
    frozen_hash = manifest.pop("manifest_hash")
    if stable_hash(manifest) != frozen_hash:
        raise ValueError("Manifest integrity mismatch")
    manifest["manifest_hash"] = frozen_hash
    for name, digest in {
        **manifest["input_hashes"],
        **manifest["execution_hashes"],
    }.items():
        if sha256(root / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    if json.loads((directory / "policy.json").read_text()) != manifest["policy"]:
        raise ValueError("Frozen policy changed")
    return manifest


def parse_response(request: dict, raw: str):
    payload = json.loads(request["prompt"])
    sources = [SourceEntry.model_validate(s) for s in payload["sources"]]
    if request["purpose"] == "nsm-fresh-quote":
        quote = payload["proposed_quote"]
        return assessment.validate_candidate_review(
            raw,
            core_value=payload["core_value"],
            source=sources[0],
            quote_source=quote["quote_source"],
            evidence_quote=quote["evidence_quote"],
        )
    return assessment.validate_source_review(
        raw, core_value=payload["core_value"], sources=sources
    )


async def execute_request(
    request: dict, provider: BudgetedProvider, directory: Path, *, allow_paid: bool
):
    policy = provider.ledger.policy
    key = stable_hash({**request, "policy_hash": stable_hash(policy)})
    attempts = [
        a for a in provider.ledger.snapshot()["attempts"] if a["request_hash"] == key
    ]
    if attempts and attempts[-1]["status"] == "completed":
        return parse_response(request, attempts[-1]["raw_text"])
    if not allow_paid:
        return None
    receipt = json.loads((directory / "input_counts.json").read_text())["counts"].get(
        stable_hash(request)
    )
    if (
        input_budget.validate_receipt(request, policy, receipt)
        > policy["input_token_limit"]
    ):
        raise ValueError(
            "Complete input exceeds 16,000 tokens; no truncation permitted"
        )
    for index in range(len(attempts), policy["max_attempts"]):
        attempt = await provider.complete(**request, retry=index > 0)
        if attempt.status == "completed":
            try:
                return parse_response(request, attempt.raw_text or "")
            except ReviewValidationError as exc:
                attempt = provider.invalidate(attempt, str(exc))
        if not attempt.retryable:
            break
    return None


def runtime_selection(case: dict, batch) -> dict | None:
    if batch is None:
        return None
    by_id = {r.entry_id: r for r in batch.results}
    return next(
        (
            by_id[s["entry_id"]].model_dump()
            for s in case["sources"]
            if by_id[s["entry_id"]].decision == "supportive"
        ),
        None,
    )


def summarize(
    results: list[dict], source_reviews: dict, ledger: dict, policy: dict
) -> dict:
    selected = [c for c in results if c["selected"]]
    accepted = [c for c in selected if c["grade"] == "accepted"]
    rejected = [c for c in selected if c["grade"] == "rejected"]
    unresolved = [c for c in selected if c["grade"] == "unresolved"]
    no_example = [c for c in results if c["reference_state"] == "confirmed_no_example"]
    unknown = [c for c in results if c["reference_state"] == "unresolved"]
    failures = [a for a in ledger["attempts"] if a["status"] != "completed"]
    saved = [c["case_id"] for c in accepted if c["persona_id"] in SAVED_PERSONAS]
    spent = sum(
        a["calculated_cost_usd"]
        if a["calculated_cost_usd"] is not None
        else a["reserved_cost_usd"]
        for a in ledger["attempts"]
    )
    failure_rate = len(failures) / len(ledger["attempts"]) if ledger["attempts"] else 0
    return {
        "cases": len(results),
        "nonempty_histories": sum(c["eligible_sources"] > 0 for c in results),
        "unique_source_value_pairs": len(source_reviews),
        "source_decisions": {
            d: sum(r["decision"] == d for r in source_reviews.values())
            for d in ("supportive", "not_supportive", "abstain")
        },
        "selected": len(selected),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "unresolved_selections": len(unresolved),
        "precision": fraction(len(accepted), len(selected)),
        "coverage": fraction(len(selected), len(results)),
        "correct_omission": fraction(
            sum(c["selected"] is None and not c["failed"] for c in no_example),
            len(no_example),
        ),
        "unresolved_no_positive_histories": len(unknown),
        "structurally_empty_histories": sum(
            c["reference_state"] == "no_earlier_writing" for c in results
        ),
        "failed_cases": sum(c["failed"] for c in results),
        "attempts": len(ledger["attempts"]),
        "unexpected_failed_attempts": len(failures),
        "code_invalid_attempts": sum(
            a["status"] == "invalid" for a in ledger["attempts"]
        ),
        "new_spent_or_reserved_usd": spent,
        "prior_spend_usd": policy["prior_spend_usd"],
        "cumulative_spent_or_reserved_usd": spent + policy["prior_spend_usd"],
        "saved_persona_accepted": saved,
        "gate_passed": bool(selected)
        and len(accepted) == len(selected)
        and all(c["selected"] is None and not c["failed"] for c in no_example)
        and not unknown
        and not any(c["failed"] for c in results)
        and not any(a["status"] == "invalid" for a in ledger["attempts"])
        and failure_rate <= 0.05
        and bool(saved),
    }


async def run(
    directory: Path, *, allow_paid: bool = False, concurrency: int = 3
) -> dict:
    manifest = verify(directory)
    policy = manifest["policy"]
    provider = BudgetedProvider(
        BudgetLedger(directory / "budget.json", directory / "policy.json")
    )
    planned = [b["request"] for b in manifest["reference_batches"]] + [
        c["runtime_request"] for c in manifest["cases"] if c["runtime_request"]
    ]
    if allow_paid:
        counts = await input_budget.measure_requests(
            planned, policy, directory / "input_counts.json"
        )
        if any(
            input_budget.validate_receipt(
                r, policy, counts["counts"].get(stable_hash(r))
            )
            > policy["input_token_limit"]
            for r in planned
        ):
            raise ValueError("Planned input exceeds 16,000 tokens")
        print(f"Counted {len(planned)} inputs; starting fresh generation", flush=True)
    semaphore = asyncio.Semaphore(concurrency)

    async def execute(request):
        async with semaphore:
            result = await execute_request(
                request, provider, directory, allow_paid=allow_paid
            )
            print(
                f"{request['purpose']}: {'valid' if result else 'unavailable'}",
                flush=True,
            )
            return result

    responses = await asyncio.gather(*(execute(r) for r in planned))
    references = {}
    for batch, response in zip(manifest["reference_batches"], responses, strict=False):
        if response is None:
            continue
        by_id = {r.entry_id: r.model_dump() for r in response.results}
        sources = json.loads(batch["request"]["prompt"])["sources"]
        for group_id, source in zip(batch["group_ids"], sources, strict=True):
            references[group_id] = by_id[source["entry_id"]]
    runtime_responses = iter(responses[len(manifest["reference_batches"]) :])
    runtime = {
        c["case_id"]: next(runtime_responses) for c in manifest["cases"] if c["sources"]
    }
    selections = {
        c["case_id"]: runtime_selection(c, runtime.get(c["case_id"]))
        for c in manifest["cases"]
    }
    quote_requests = {}
    independent_quotes = []
    for case in manifest["cases"]:
        selected = selections[case["case_id"]]
        if selected is not None:
            quote_requests[case["case_id"]] = candidate_request(case, selected)
            independent_quotes.append(
                {
                    "case_id": case["case_id"],
                    "value": case["value"],
                    "source": next(
                        s
                        for s in case["sources"]
                        if s["entry_id"] == selected["entry_id"]
                    ),
                    "quote_source": selected["quote_source"],
                    "evidence_quote": selected["evidence_quote"],
                }
            )
    if allow_paid:
        await input_budget.measure_requests(
            list(quote_requests.values()), policy, directory / "input_counts.json"
        )
    candidate_results = await asyncio.gather(
        *(execute(r) for r in quote_requests.values())
    )
    candidates = dict(zip(quote_requests, candidate_results, strict=True))
    results = []
    for case in manifest["cases"]:
        selected = selections[case["case_id"]]
        candidate = candidates.get(case["case_id"])
        source_results = [references.get(k) for k in case["group_ids"]]
        missing = any(r is None for r in source_results)
        state = (
            "no_earlier_writing"
            if not source_results
            else "unresolved"
            if missing
            else "has_example"
            if any(r and r["decision"] == "supportive" for r in source_results)
            else "unresolved"
            if any(r and r["decision"] == "abstain" for r in source_results)
            else "confirmed_no_example"
        )
        primary = next(
            (
                r
                for r in source_results
                if r and selected and r["entry_id"] == selected["entry_id"]
            ),
            None,
        )
        disagreement = bool(
            primary
            and candidate
            and primary["decision"]
            != assessment.DECISION_BY_REASON[candidate.source_reason]
        )
        uncertain = bool(
            primary
            and primary["decision"] == "abstain"
            or candidate
            and assessment.DECISION_BY_REASON[candidate.source_reason] == "abstain"
        )
        failed = (
            missing
            or bool(case["sources"] and runtime.get(case["case_id"]) is None)
            or bool(selected and candidate is None)
        )
        grade = (
            "not_selected"
            if selected is None
            else "unresolved"
            if failed or disagreement or uncertain
            else "accepted"
            if candidate and candidate.accepted
            else "rejected"
        )
        results.append(
            {
                "case_id": case["case_id"],
                "persona_id": case["persona_id"],
                "eligible_sources": len(case["sources"]),
                "selected": selected,
                "runtime": runtime[case["case_id"]].model_dump()
                if runtime.get(case["case_id"])
                else None,
                "reference_state": state,
                "source_reference": source_results,
                "candidate_assessment": candidate.model_dump() if candidate else None,
                "source_judge_disagreement": disagreement,
                "source_reason_disagreement": bool(
                    primary
                    and candidate
                    and primary["reason_code"] != candidate.source_reason
                ),
                "grade": grade,
                "failed": failed,
            }
        )
    report = {
        "schema_version": "north-star-luna-report-v1",
        "manifest_hash": manifest["manifest_hash"],
        "assessment_source": (
            "GPT-5.6 Luna xhigh AI assessments of synthetic "
            "development writing; not human validation"
        ),
        "summary": summarize(results, references, provider.ledger.snapshot(), policy),
        "source_reviews": references,
        "cases": results,
    }
    write_json(directory / "report.json", report)
    # Inputs for subsequent independent review deliberately exclude all AI labels,
    # runtime explanations, primary reference choices, and candidate assessments.
    write_json(
        directory / "independent_review_inputs.json",
        {
            "manifest_hash": manifest["manifest_hash"],
            "source_groups": manifest["source_groups"],
            "selected_quotes": independent_quotes,
        },
    )
    print(json.dumps(report["summary"], indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.directory)
        print(
            json.dumps(
                {
                    "cases": len(result["cases"]),
                    "source_groups": len(result["source_groups"]),
                    "reference_batches": len(result["reference_batches"]),
                    "budget": result["budget_preflight"],
                },
                indent=2,
            )
        )
    else:
        if args.command == "run" and not args.allow_paid:
            parser.error(
                "run requires --allow-paid; use report for offline reconstruction"
            )
        if args.allow_paid and args.command != "run":
            parser.error("Only run permits paid calls")
        if args.allow_paid:
            load_dotenv(ROOT / ".env", override=False)
        asyncio.run(run(args.directory, allow_paid=args.allow_paid))


if __name__ == "__main__":
    main()
