"""Supplement the frozen NSM experiment with actual saved-frontend coverage.

Non-triggering weeks are controls, not quotation-precision cases. Reuse a prior
case only when value, onset, and complete original source inputs match. Evaluate
missing active cases separately; no frontend bundle or original result is edited.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from scripts.experiments import north_star_luna as baseline  # noqa: E402
from src.demo.scenarios import (  # noqa: E402
    CATALOG_PATH,
    SCENARIO_DIRECTORY,
    load_scenario_catalog,
    project_scenario_week,
)
from src.north_star import input_budget  # noqa: E402
from src.north_star.provider import (  # noqa: E402
    BudgetedProvider,
    BudgetLedger,
    stable_hash,
)
from src.north_star.review import SourceEntry  # noqa: E402

DIRECTORY = ROOT / "logs/experiments/reports/north_star_saved_checks_20260906"


def coverage(root: Path, original: dict, original_report: dict) -> dict:
    catalog, fixtures = load_scenario_catalog(root)
    values = {c["value"]["core_value"]: c["value"] for c in original["cases"]}
    report_by_id = {c["case_id"]: c for c in original_report["cases"]}
    weeks, missing = [], []
    for item in catalog.scenarios:
        fixture = fixtures[item.scenario_id]
        for week in fixture.scenario.weeks:
            session, _ = project_scenario_week(fixture, week.week_id)
            result = session.drift_result
            if result is None:
                raise ValueError("Saved week lacks its Drift result")
            row = {
                "scenario_id": item.scenario_id,
                "persona_id": item.persona_id,
                "persona_name": item.persona_name,
                "week_id": week.week_id,
                "week_start": week.week_start,
                "week_end": week.week_end,
                "delivery_state": result.delivery_state,
                "core_value_states": result.core_value_states,
                "expected_nsm_action": "no_request_no_card",
                "case_id": None,
                "source": "no_trigger_control",
            }
            if result.delivery_state == "active_drift":
                active = [
                    d
                    for d in result.drifts
                    if d.termination_reason is None
                    and result.core_value_states[d.core_value] == "active_drift"
                ]
                if not active:
                    raise ValueError("Active week has no supported Drift")
                priority: dict[str, int] = {
                    value: index
                    for index, value in enumerate(fixture.scenario.profile.top_values)
                }
                drift = min(
                    active,
                    key=lambda d: (
                        -result.core_value_details[d.core_value].current_run_length,
                        priority[d.core_value],
                    ),
                )
                sources = [
                    SourceEntry(entry_id=e.journal_entry_id, journal_entry=e.content)
                    for e in sorted(
                        session.journal_entries, key=lambda e: e.t_index, reverse=True
                    )
                    if e.t_index < drift.onset_t_index
                    and e.date <= drift.onset_date
                    and e.content.strip()
                ]
                value = values[drift.core_value]
                for source in sources:
                    if not any(
                        source.model_dump() in case["sources"]
                        for case in original["cases"]
                    ):
                        raise ValueError("Source lacks original synthetic provenance")
                match = next(
                    (
                        c
                        for c in original["cases"]
                        if c["persona_id"] == item.persona_id
                        and c["value"] == value
                        and c["episode"]["onset_t_index"] == drift.onset_t_index
                        and c["episode"]["onset_date"] == drift.onset_date
                        and c["sources"] == [s.model_dump() for s in sources]
                    ),
                    None,
                )
                if match is not None:
                    case_id = match["case_id"]
                    row.update(
                        source="matching_frozen_case",
                        case_id=case_id,
                        frozen_grade=report_by_id[case_id]["grade"],
                        expected_nsm_action="review_eligible_writing",
                    )
                else:
                    case_id = f"{week.week_id}:{drift.core_value}"
                    missing.append(
                        {
                            "case_id": case_id,
                            "persona_id": item.persona_id,
                            "value": value,
                            "sources": [s.model_dump() for s in sources],
                            "onset_t_index": drift.onset_t_index,
                            "onset_date": drift.onset_date,
                            "runtime_request": baseline.source_request(
                                value, sources, "runtime"
                            ),
                            "reference_request": baseline.source_request(
                                value, sources, "reference"
                            ),
                        }
                    )
                    row.update(
                        source="supplemental_active_case",
                        case_id=case_id,
                        expected_nsm_action="review_eligible_writing",
                    )
            weeks.append(row)
    paths = [
        CATALOG_PATH,
        *(SCENARIO_DIRECTORY / item.file for item in catalog.scenarios),
    ]
    return {
        "weeks": weeks,
        "missing_cases": missing,
        "frontend_hashes": {str(p): baseline.sha256(root / p) for p in paths},
    }


def prepare(directory: Path = DIRECTORY) -> dict:
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Refuse to overwrite a supplement")
    original = baseline.verify(baseline.DEFAULT_DIRECTORY)
    original_report = json.loads(
        (baseline.DEFAULT_DIRECTORY / "report.json").read_text()
    )
    data = coverage(ROOT, original, original_report)
    policy = dict(original["policy"])
    policy["prior_spend_usd"] = original_report["summary"][
        "cumulative_spent_or_reserved_usd"
    ]
    # The provider's 64k conservative input envelope bounds every candidate.
    per_attempt = (64_000 * 0.2 * 1.25 + policy["max_output_tokens"] * 1.2) / 1e6
    maximum_new = len(data["missing_cases"]) * 3 * policy["max_attempts"] * per_attempt
    if (
        per_attempt > policy["per_attempt_usd"]
        or maximum_new + policy["prior_spend_usd"] > policy["budget_usd"]
    ):
        raise ValueError("Supplement exceeds the existing budget")
    manifest = {
        "schema_version": "north-star-saved-checks-v1",
        "original_manifest_hash": original["manifest_hash"],
        "original_report_sha256": baseline.sha256(
            baseline.DEFAULT_DIRECTORY / "report.json"
        ),
        "execution_sha256": baseline.sha256(Path(__file__)),
        "policy": policy,
        "maximum_new_usd": maximum_new,
        **data,
        "scope": (
            "Original-writing offline coverage. NSM frontend rendering, actual "
            "dispatch suppression, and independently timestamped nudge-response "
            "integration remain untested."
        ),
    }
    manifest["manifest_hash"] = stable_hash(manifest)
    baseline.write_json(directory / "manifest.json", manifest)
    baseline.write_json(directory / "policy.json", policy)
    return manifest


async def run(directory: Path = DIRECTORY, *, allow_paid: bool = False) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    digest = manifest.pop("manifest_hash")
    if stable_hash(manifest) != digest:
        raise ValueError("Supplement manifest changed")
    manifest["manifest_hash"] = digest
    baseline.verify(baseline.DEFAULT_DIRECTORY)
    if (
        baseline.sha256(baseline.DEFAULT_DIRECTORY / "report.json")
        != manifest["original_report_sha256"]
    ):
        raise ValueError("Original report changed")
    if baseline.sha256(Path(__file__)) != manifest["execution_sha256"]:
        raise ValueError("Supplement execution changed")
    for name, expected in manifest["frontend_hashes"].items():
        if baseline.sha256(ROOT / name) != expected:
            raise ValueError("Frozen frontend bundle changed")
    if json.loads((directory / "policy.json").read_text()) != manifest["policy"]:
        raise ValueError("Supplement policy changed")
    provider = BudgetedProvider(
        BudgetLedger(directory / "budget.json", directory / "policy.json")
    )
    requests = [
        c[role]
        for c in manifest["missing_cases"]
        for role in ("runtime_request", "reference_request")
    ]
    if allow_paid:
        await input_budget.measure_requests(
            requests, manifest["policy"], directory / "input_counts.json"
        )
    results = []
    for case in manifest["missing_cases"]:
        runtime, reference = await asyncio.gather(
            *(
                baseline.execute_request(
                    case[role], provider, directory, allow_paid=allow_paid
                )
                for role in ("runtime_request", "reference_request")
            )
        )
        selected = baseline.runtime_selection(case, runtime)
        candidate = None
        if selected:
            request = baseline.candidate_request(case, selected)
            if allow_paid:
                await input_budget.measure_requests(
                    [request], manifest["policy"], directory / "input_counts.json"
                )
            candidate = await baseline.execute_request(
                request, provider, directory, allow_paid=allow_paid
            )
        primary = (
            next(
                (
                    r
                    for r in reference.results
                    if selected and r.entry_id == selected["entry_id"]
                ),
                None,
            )
            if reference
            else None
        )
        failed = (
            runtime is None or reference is None or bool(selected and candidate is None)
        )
        accepted = bool(
            primary
            and candidate
            and primary.decision == "supportive"
            and candidate.accepted
        )
        results.append(
            {
                "case_id": case["case_id"],
                "selected": selected,
                "runtime": runtime.model_dump() if runtime else None,
                "reference": reference.model_dump() if reference else None,
                "candidate_assessment": candidate.model_dump() if candidate else None,
                "accepted": accepted,
                "failed": failed,
            }
        )
    ledger = provider.ledger.snapshot()
    spent = sum(
        a["calculated_cost_usd"]
        if a["calculated_cost_usd"] is not None
        else a["reserved_cost_usd"]
        for a in ledger["attempts"]
    )
    result = {
        "manifest_hash": digest,
        "weeks": manifest["weeks"],
        "supplemental_cases": results,
        "generation_attempts": len(ledger["attempts"]),
        "new_cost_usd": spent,
        "cumulative_cost_usd": spent + manifest["policy"]["prior_spend_usd"],
        "original_selection_precision_unchanged": True,
        "scope": manifest["scope"],
    }
    baseline.write_json(directory / "report.json", result)
    print(
        json.dumps(
            {
                "saved_personas": len({w["persona_id"] for w in manifest["weeks"]}),
                "projected_weeks": len(manifest["weeks"]),
                "supplemental_cases": len(results),
                "failed_cases": sum(c["failed"] for c in results),
                "new_cost_usd": spent,
            },
            indent=2,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
        print(
            json.dumps(
                {
                    "weeks": len(result["weeks"]),
                    "missing_cases": len(result["missing_cases"]),
                    "maximum_new_usd": result["maximum_new_usd"],
                },
                indent=2,
            )
        )
    else:
        if args.allow_paid != (args.command == "run"):
            parser.error("Only run --allow-paid permits provider calls")
        if args.allow_paid:
            load_dotenv(ROOT / ".env", override=False)
        asyncio.run(run(allow_paid=args.allow_paid))


if __name__ == "__main__":
    main()
