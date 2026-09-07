"""Audit and execute a separately versioned NSM update from saved v4 Run 1.

The original experiment is read-only. Exact same-purpose receipts may be retained;
all effective-context changes rerun deterministic selection and grading for both
methods. This is a targeted update with retained observations, not a fresh trial.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import hashlib
import json
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.experiments import nsm_cases as cases_lib
from scripts.experiments import nsm_evaluation as evaluation
from scripts.experiments import nsm_experiment as experiment
from scripts.experiments import weekly_drift_definitions as upstream
from scripts.experiments.nsm_provider import DEFAULT_POLICY, ExperimentProvider
from src.drift_detector import detect_drift
from src.north_star import runtime
from src.north_star.input_budget import validate_receipt
from src.north_star.provider import openai_input_payload, stable_hash
from src.north_star.review import SourceEntry
from src.weekly_drift_reviewer import WeeklyDriftReviewerDecision

ROOT = experiment.ROOT
ORIGINAL = experiment.DEFAULT_RECORD
UPSTREAM = (
    ROOT / "logs/experiments/artifacts/twinkl_j3k7_core_value_definitions_20260907"
)
OUTPUT = ROOT / "logs/experiments/reports/north_star_v4_run1_20260907"
SELF = "scripts/experiments/nsm_targeted_update.py"
EXTRA_CODE = (
    "src/drift_detector.py",
    "src/drift_rules.py",
    "src/weekly_drift_reviewer.py",
)
# Full Drift evidence is provenance; selection/grade consumers use these fields.
EFFECTIVE_FIELDS = (
    "case_id",
    "persona_id",
    "split",
    "weekly_state",
    "week_start",
    "week_end",
    "cutoff",
    "cutoff_at",
    "cutoff_t_index",
    "profile",
    "profile_ref",
    "core_values",
    "values",
    "source_metadata",
    "onset_t_index",
    "onset_date",
    "onset_available_at",
    "context_reason",
    "source_availability",
    "is_sparse_history_final_week",
)


def read(path: Path) -> dict:
    result: dict = json.loads(path.read_text())
    return result


def write(path: Path, value: dict) -> None:
    experiment.persist_record(path, value)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def effective(case: dict) -> dict:
    return {key: case[key] for key in EFFECTIVE_FIELDS}


def verify_original(original: dict) -> dict:
    """Verify historical contracts without replacing their frozen source hashes."""
    e = original["execution"]
    corrections = experiment._reporting_corrections(original)
    for name, expected in e["freeze"]["code_sha256"].items():
        if experiment.file_hash(ROOT / name) != expected and name not in corrections:
            raise ValueError(f"Original NSM contract changed: {name}")
    for field, value in (
        ("cases", e["cases"]),
        ("retrieval", e["retrieval"]),
        ("policy", e["provider_policy"]),
        ("settings", original["frozen_settings"]),
        ("consistency_sample", e["consistency_case_ids"]),
    ):
        if stable_hash(value) != e["freeze"][f"{field}_sha256"]:
            raise ValueError(f"Original frozen {field} changed")
    frozen = e["freeze"]
    if (
        hashlib.sha256(frozen["methodology_text"].encode()).hexdigest()
        != frozen["methodology_sha256"]
    ):
        raise ValueError("Original methodology snapshot changed")
    changes = {}
    for name, expected in e["case_provenance"]["source_hashes"].items():
        actual = experiment.file_hash(ROOT / name)
        if actual != expected:
            # The authorized upstream migration changed this reviewer only. Verify
            # the original blob against its original revision instead of rebinding it.
            if name != "src/weekly_drift_reviewer.py":
                raise ValueError(f"Original source changed unexpectedly: {name}")
            blob = subprocess.check_output(
                ["git", "show", f"{frozen['code_revision']}:{name}"], cwd=ROOT
            )
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError("Original reviewer source cannot be verified")
            changes[name] = {
                "original_sha256": expected,
                "current_sha256": actual,
                "original_revision": frozen["code_revision"],
            }
    return changes


def rebuild_cases(
    original: dict, responses: list[dict], requests: list[dict]
) -> list[dict]:
    old_cases = original["execution"]["cases"]
    wanted = {c["case_id"] for c in old_cases}
    selected = [
        r
        for r in responses
        if r["variant"] == "definitions" and r["repeat"] == 1 and r["case_id"] in wanted
    ]
    if len(selected) != 501 or {r["case_id"] for r in selected} != wanted:
        raise ValueError("v4 Run 1 must cover the original 501 cases exactly")
    by_case = {r["case_id"]: r for r in selected}
    inputs = {r["case_id"]: r for r in requests}
    histories, accumulated = {}, defaultdict(list)
    result = []
    for old in sorted(old_cases, key=lambda c: (c["persona_id"], c["week_start"])):
        case = copy.deepcopy(old)
        pid, key = case["persona_id"], case["case_id"]
        if pid not in histories:
            stored = next(
                row["stored_core_values"]
                for row in original["execution"]["case_provenance"][
                    "profile_projections"
                ]
                if row["persona_id"] == pid
            )
            histories[pid] = cases_lib._history(ROOT, pid, stored)
        entries = histories[pid]
        writing = cases_lib._writing(pid, entries)
        receipt, request_row = by_case[key], inputs[key]
        request_data = request_row["request"]
        if set(request_data["core_values"]) != set(case["core_values"]) or len(
            request_data["core_values"]
        ) != len(case["core_values"]):
            raise ValueError("Upstream Core Values differ")
        expected_text = [
            {
                "t_index": row["t_index"],
                "date": row["date"],
                "text": cases_lib._upstream_text(row),
            }
            for row in entries
            if row["date"] <= case["week_end"]
        ]
        if request_data["history"] != expected_text:
            raise ValueError(f"Upstream writing differs: {key}")
        if (
            receipt["request_sha256"]
            != request_row["variants"]["definitions"]["request_sha256"]
        ):
            raise ValueError("v4 receipt request binding differs")
        decisions = [
            WeeklyDriftReviewerDecision.model_validate(d) for d in receipt["decisions"]
        ]
        accumulated[pid].extend(decisions)
        drift = detect_drift(accumulated[pid], persona_id=pid)
        request = runtime.build_north_star_request(
            session_id=f"nsm-benchmark:{pid}",
            owner_id=pid,
            profile_ref=case["profile_ref"],
            core_values=case["core_values"],
            week_start=case["week_start"],
            week_end=case["week_end"],
            cutoff_at=case["cutoff_at"],
            drift_result=drift,
            writing=writing,
        )
        selected_values, sources, onset, onset_at, reason = runtime._context(request)
        metadata = case["source_metadata"]
        case.update(
            weekly_state=drift.delivery_state,
            cutoff_t_index=drift.cutoff_t_index,
            drift_result=drift.model_dump(mode="json"),
            onset_t_index=onset.onset_t_index if onset else None,
            onset_date=onset.onset_date if onset else None,
            onset_available_at=onset_at,
            context_reason=reason,
            values=[
                {
                    **request.value_definitions[value].model_dump(),
                    "sources": [
                        SourceEntry(
                            entry_id=s.entry_id,
                            journal_entry=s.journal_entry,
                            nudge_response=s.nudge_response,
                        ).model_dump()
                        for s in sources
                    ],
                    "source_metadata": {
                        s.entry_id: metadata[s.entry_id] for s in sources
                    },
                    "source_count": len(sources),
                    "eligible_responses": sum(bool(s.nudge_response) for s in sources),
                    "source_order": "t_index descending",
                }
                for value in selected_values
            ],
            upstream={
                "repeat": 1,
                "prompt_version": "v4.0",
                "variant": "definitions",
                "status": receipt["status"],
                "request_sha256": receipt["request_sha256"],
                "receipt_sha256": stable_hash(receipt),
                "current_decisions": receipt["decisions"],
                "source": relative(UPSTREAM / "responses.jsonl"),
                "original_v2_upstream_sha256": stable_hash(old["upstream"]),
            },
            source_availability={
                "through_cutoff": len(request.writing),
                "eligible_entries": len(sources),
                "eligible_responses": sum(bool(s.nudge_response) for s in sources),
                "future_entries_excluded": len(writing) - len(request.writing),
                "source_window_excluded": len(request.writing) - len(sources)
                if selected_values
                else 0,
                "insufficient_evidence_control": drift.delivery_state
                == "insufficient_evidence",
                "available_source_ids": [s.entry_id for s in request.writing],
            },
        )
        case.pop("input_hash")
        case["input_hash"] = stable_hash(case)
        result.append(case)
    return result


def audit(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "impact_audit.json"
    if path.exists():
        saved = read(path)
        for name, expected in saved["upstream_hashes"].items():
            if experiment.file_hash(ROOT / name) != expected:
                raise ValueError(f"Audited input changed: {name}")
        return saved
    original = read(ORIGINAL)
    source_changes = verify_original(original)
    manifest, requests = upstream.verify(UPSTREAM)
    responses = upstream.read_rows(UPSTREAM / "responses.jsonl")
    upstream.score_rows(requests, responses, manifest["settings"]["repeats"])
    cases = rebuild_cases(original, responses, requests)
    old = {c["case_id"]: c for c in original["execution"]["cases"]}
    differences = []
    for case in cases:
        before = old[case["case_id"]]
        changed = {
            field: {"before": before[field], "after": case[field]}
            for field in EFFECTIVE_FIELDS
            if before[field] != case[field]
        }
        differences.append(
            {
                "case_id": case["case_id"],
                "persona_id": case["persona_id"],
                "split": case["split"],
                "changed": bool(changed),
                "before_effective_sha256": stable_hash(effective(before)),
                "after_effective_sha256": stable_hash(effective(case)),
                "differences": changed,
                "before_case_sha256": stable_hash(before),
                "after_case_sha256": stable_hash(case),
                "before_upstream_sha256": stable_hash(before["upstream"]),
                "after_upstream_sha256": stable_hash(case["upstream"]),
            }
        )
    affected = [row for row in differences if row["changed"]]
    hashes = {
        relative(p): experiment.file_hash(p)
        for p in (
            ORIGINAL,
            *(
                UPSTREAM / name
                for name in (
                    "manifest.json",
                    "baseline_snapshot.json",
                    "requests.jsonl",
                    "responses.jsonl",
                    "attempts.jsonl",
                    "results.json",
                )
            ),
        )
    }
    hashes.update(
        {
            p: experiment.file_hash(ROOT / p)
            for p in original["execution"]["case_provenance"]["source_hashes"]
        }
    )
    result = {
        "schema_version": "north-star-v4-run1-impact-audit-v1",
        "created_at": experiment.now(),
        "issue": "twinkl-fz34.14",
        "upstream_hashes": hashes,
        "historical_source_changes": source_changes,
        "effective_fields": list(EFFECTIVE_FIELDS),
        "provenance_only_fields": ["drift_result", "upstream", "input_hash"],
        "cases": cases,
        "case_differences": differences,
        "summary": {
            "cases": len(cases),
            "personas": len({c["persona_id"] for c in cases}),
            "changed_cases": len(affected),
            "changed_personas": len({r["persona_id"] for r in affected}),
            "changed_by_split": dict(Counter(r["split"] for r in affected)),
            "changed_fields": dict(
                Counter(k for r in affected for k in r["differences"])
            ),
            "state_transitions": dict(
                Counter(
                    f"{old[c['case_id']]['weekly_state']} -> {c['weekly_state']}"
                    for c in cases
                )
            ),
            "states_by_split": {
                s: dict(Counter(c["weekly_state"] for c in cases if c["split"] == s))
                for s in ("development", "final")
            },
        },
        "consistency_case_ids": original["execution"]["consistency_case_ids"],
        "partition": original["partition"],
        "interpretation": (
            "Effective inputs include deterministic eligibility and "
            "priority; upstream evidence-only changes do not require new "
            "NSM requests. Full ordered batches remain indivisible."
        ),
    }
    write(path, result)
    return result


def compatible_receipt(
    request: dict, receipt: dict, policy: dict, max_attempts: int
) -> bool:
    """Match the complete batch/quotation, experimental purpose and settings."""
    if receipt.get("request_hash") != stable_hash(request):
        return False
    identity = {
        "request": request,
        "policy_hash": stable_hash(policy),
        "payload_hash": stable_hash(openai_input_payload(request, policy)),
        "max_attempts": max_attempts,
    }
    if any(receipt.get(k) != value for k, value in identity.items()):
        raise ValueError(
            "Matching historical request has incompatible receipt identity"
        )
    if any(
        len(receipt.get(k, [])) > max_attempts for k in ("attempts", "count_attempts")
    ):
        raise ValueError("Historical receipt exceeds its attempt limit")
    if receipt.get("count_receipt") is not None:
        validate_receipt(request, policy, receipt["count_receipt"])
    return True


class RetainingProvider(ExperimentProvider):
    """Copy matching original receipts on demand, including terminal failures."""

    def __init__(self, record, persist, original, **provider_kwargs):
        self.original = original
        self.lineage = record.setdefault("evidence_lineage", {})
        super().__init__(record, persist, **provider_kwargs)

    async def _request(self, request, max_attempts, validate, *, count_only):
        key = stable_hash(request)
        prior = self.original["execution"]["requests"].get(key)
        if (
            key not in self.requests
            and prior is not None
            and compatible_receipt(request, prior, self.policy, max_attempts)
        ):
            self.requests[key] = copy.deepcopy(prior)
            self.lineage[key] = retained_lineage(prior)
            self.persist()
        if (
            key in self.requests
            and self.lineage.get(key, {}).get("origin") == "retained"
        ):
            receipt = self.requests[key]
            if not compatible_receipt(request, receipt, self.policy, max_attempts):
                raise ValueError("Retained receipt no longer matches")
            if stable_hash(receipt) != self.lineage[key]["original_receipt_sha256"]:
                raise ValueError("Retained receipt was modified")
            if not count_only and receipt["status"] == "completed":
                if (
                    validate(
                        copy.deepcopy(receipt["attempts"][-1]["parsed_output"]), request
                    )
                    != receipt["result"]
                ):
                    raise ValueError("Retained receipt validation changed")
            # Retained failures/unresolved requests are observations, never retries.
            return receipt
        self.lineage.setdefault(
            key, {"origin": "new", "request_sha256": key, "purpose": request["purpose"]}
        )
        return await super()._request(
            request, max_attempts, validate, count_only=count_only
        )


def retained_lineage(receipt: dict) -> dict:
    return {
        "origin": "retained",
        "request_sha256": receipt["request_hash"],
        "purpose": receipt["request"]["purpose"],
        "original_receipt_sha256": stable_hash(receipt),
        "original_record": relative(ORIGINAL),
        "justification": (
            "Identical complete ordered request, purpose "
            "(case/method/role/repeat), model/policy, payload, attempt "
            "limit and frozen semantic/validation/evaluation contracts. "
            "Terminal failures retained."
        ),
    }


def static_requests(record: dict) -> list[dict]:
    e = record["execution"]
    result = []
    for case in e["cases"]:
        for variant in experiment.VARIANTS:
            ids = experiment.candidates(record, case, variant)
            for value in case["values"]:
                if ids.get(value["core_value"]):
                    result.append(
                        experiment.source_request(
                            case, value, ids[value["core_value"]], f"runtime:{variant}"
                        )
                    )
        repeats = ["primary"] + (
            ["repeat_2", "repeat_3"]
            if case["case_id"] in e["consistency_case_ids"]
            else []
        )
        for repeat in repeats:
            for value in case["values"]:
                source_ids = [s["entry_id"] for s in value["sources"]]
                if source_ids:
                    result.append(
                        experiment.source_request(
                            case, value, source_ids, f"reference:{repeat}"
                        )
                    )
    return result


def prepare(output: Path) -> dict:
    path = output / "nsm_experiment.json"
    if path.exists():
        record = read(path)
        verify(record, output)
        return record
    audited = audit(output)
    original = read(ORIGINAL)
    verify_original(original)
    old = original["execution"]
    changed = {row["case_id"] for row in audited["case_differences"] if row["changed"]}
    affected = [case for case in audited["cases"] if case["case_id"] in changed]
    config = cases_lib.retrieval_config(ROOT)
    if config != old["retrieval"]["config"]:
        raise ValueError("Nomic configuration differs from original")
    # Original vectors were not persisted. Re-encode affected corpora locally;
    # unchanged cases retain their original rankings and timing observations.
    new_retrieval = cases_lib.prepare_retrieval(affected, ROOT)
    retrieval = copy.deepcopy(old["retrieval"])
    retrieval["cases"].update(new_retrieval["cases"])
    retrieval["per_case_preparation_seconds"] = new_retrieval[
        "per_case_preparation_seconds"
    ]
    retrieval["update_preparation"] = new_retrieval
    retrieval["preparation_allocation"] = (
        f"New encoding divided across {len(affected)} affected cases; "
        "unchanged runtime "
        "measurements retain original allocation. Mixed observations, not fresh "
        "latency."
    )
    record = {
        "schema_version": "north-star-targeted-update-v1",
        "issue": "twinkl-fz34.14",
        "created_at": experiment.now(),
        "status": "prepared",
        "scope": (
            "Targeted v4 Run 1 update with retained observations, not a "
            "wholly fresh independent experiment"
        ),
        "authorization": (
            "User authorized scoped implementation, verification and paid "
            "NSM calls for both methods; no Weekly Drift rerun, prompt "
            "changes or publishing."
        ),
        "partition": copy.deepcopy(original["partition"]),
        "frozen_settings": copy.deepcopy(original["frozen_settings"]),
        "historical_record": {
            "path": relative(ORIGINAL),
            "sha256": experiment.file_hash(ORIGINAL),
            "freeze_sha256": stable_hash(old["freeze"]),
            "reporting_corrections_sha256": stable_hash(original["reporting"]),
            "usage": old["usage"],
            "stages": old["stages"],
        },
        "impact_audit": {
            "path": relative(output / "impact_audit.json"),
            "sha256": experiment.file_hash(output / "impact_audit.json"),
            "summary": audited["summary"],
        },
        "affected_case_ids": sorted(changed),
        "evidence_lineage": {},
        "execution": {
            "cases": audited["cases"],
            "consistency_case_ids": old["consistency_case_ids"],
            "retrieval": retrieval,
            "provider_policy": copy.deepcopy(DEFAULT_POLICY),
            "requests": {},
            "stages": {},
            "repeats": {},
            "variants": {
                v: {
                    k: copy.deepcopy(r)
                    for k, r in old["variants"][v].items()
                    if k not in changed
                }
                for v in experiment.VARIANTS
            },
            **{
                field: {
                    k: copy.deepcopy(r)
                    for k, r in old[field].items()
                    if k not in changed
                }
                for field in ("references", "quote_reviews", "rechecks")
            },
        },
    }
    e = record["execution"]
    for repeat, results in old["repeats"].items():
        e["repeats"][repeat] = {
            field: {k: copy.deepcopy(r) for k, r in rows.items() if k not in changed}
            for field, rows in results.items()
        }
    unchanged = {c["case_id"] for c in e["cases"]} - changed
    for key, receipt in old["requests"].items():
        if any(f":{case_id}" in receipt["request"]["purpose"] for case_id in unchanged):
            if not compatible_receipt(
                receipt["request"],
                receipt,
                e["provider_policy"],
                receipt["max_attempts"],
            ):
                raise ValueError("Original retained request hash mismatch")
            e["requests"][key] = copy.deepcopy(receipt)
            record["evidence_lineage"][key] = retained_lineage(receipt)
    freeze = {
        "created_at": experiment.now(),
        "original_record_sha256": experiment.file_hash(ORIGINAL),
        "impact_audit_sha256": experiment.file_hash(output / "impact_audit.json"),
        "code_sha256": {
            p: experiment.file_hash(ROOT / p)
            for p in (*experiment.CODE_PATHS, *EXTRA_CODE, SELF)
        },
        "upstream_hashes": audited["upstream_hashes"],
        "cases_sha256": stable_hash(e["cases"]),
        "retrieval_sha256": stable_hash(retrieval),
        "policy_sha256": stable_hash(e["provider_policy"]),
        "settings_sha256": stable_hash(record["frozen_settings"]),
        "partition_sha256": stable_hash(record["partition"]),
        "consistency_sample_sha256": stable_hash(e["consistency_case_ids"]),
        "affected_case_ids_sha256": stable_hash(record["affected_case_ids"]),
        "source_request_hashes": {
            r["purpose"]: stable_hash(r) for r in static_requests(record)
        },
        "evaluation_contract": (
            "Current corrected original evaluation; preserves original "
            "primary/repeat/recheck distinction and paired exclusions."
        ),
        "reuse_contract": (
            "Exact same-purpose request plus payload, policy, attempt limit"
            " and verified contract hashes; no labels extracted from "
            "changed batches."
        ),
        "methodology_text": (
            ROOT / "docs/north_star/nsm_experiment_methodology.md"
        ).read_text(),
        "targeted_update_instructions": record["scope"]
        + ". v4 definitions Run 1; same 105/501 and 81/24 split; both methods; "
        "original consistency sample; keep unchanged failures; "
        "incremental accounting separate.",
    }
    write(output / "manifest.json", freeze)
    record["manifest_sha256"] = experiment.file_hash(output / "manifest.json")
    write(path, record)
    return record


def verify(record: dict, output: Path) -> None:
    manifest = read(output / "manifest.json")
    if experiment.file_hash(output / "manifest.json") != record["manifest_sha256"]:
        raise ValueError("Targeted update manifest changed")
    for group in ("code_sha256", "upstream_hashes"):
        for name, expected in manifest[group].items():
            if experiment.file_hash(ROOT / name) != expected:
                raise ValueError(f"Targeted update frozen input changed: {name}")
    if (
        experiment.file_hash(output / "impact_audit.json")
        != manifest["impact_audit_sha256"]
    ):
        raise ValueError("Targeted impact audit changed")
    e = record["execution"]
    for name, value in (
        ("cases", e["cases"]),
        ("retrieval", e["retrieval"]),
        ("policy", e["provider_policy"]),
        ("settings", record["frozen_settings"]),
        ("partition", record["partition"]),
        ("consistency_sample", e["consistency_case_ids"]),
        ("affected_case_ids", record["affected_case_ids"]),
    ):
        if stable_hash(value) != manifest[f"{name}_sha256"]:
            raise ValueError(f"Targeted update frozen {name} changed")
    if {r["purpose"]: stable_hash(r) for r in static_requests(record)} != manifest[
        "source_request_hashes"
    ]:
        raise ValueError("Targeted static requests changed")
    original = read(ORIGINAL)
    if experiment.file_hash(ORIGINAL) != manifest["original_record_sha256"]:
        raise ValueError("Original record changed")
    for key, receipt in e["requests"].items():
        if key != stable_hash(receipt["request"]):
            raise ValueError("Request hash mismatch")
        if not compatible_receipt(
            receipt["request"], receipt, e["provider_policy"], receipt["max_attempts"]
        ):
            raise ValueError("Receipt request identity changed")
        lineage = record["evidence_lineage"][key]
        if lineage["request_sha256"] != key or lineage["origin"] not in {
            "new",
            "retained",
        }:
            raise ValueError("Receipt lineage identity changed")
        if lineage["origin"] == "retained" and (
            receipt != original["execution"]["requests"][key]
            or stable_hash(receipt) != lineage["original_receipt_sha256"]
        ):
            raise ValueError("Retained evidence changed")


async def run(output: Path, concurrency: int = 8) -> dict:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    path = output / "nsm_experiment.json"
    record = read(path)
    verify(record, output)
    e = record["execution"]
    save = lambda: write(path, record)  # noqa: E731
    provider = RetainingProvider(record, save, read(ORIGINAL))
    semaphore = asyncio.Semaphore(concurrency)
    record["status"] = "running"
    save()

    async def measure(request):
        async with semaphore:
            return await provider.measure(request)

    requests = static_requests(record)
    print(
        f"{experiment.now()} measuring {len(requests)} source requests "
        "(exact receipts retained)",
        flush=True,
    )
    await asyncio.gather(*(measure(r) for r in requests))

    async def stage(name, selected, destination, work):
        saved = e["stages"].setdefault(
            name,
            {
                "started_at": experiment.now(),
                "concurrency": concurrency,
                "initial_retained_cases": len(destination),
            },
        )
        print(
            f"{experiment.now()} {name}: {len(destination)}/{len(selected)} saved",
            flush=True,
        )

        async def one(case):
            if case["case_id"] in destination:
                return
            async with semaphore:
                destination[case["case_id"]] = await work(case)
                save()
                print(
                    f"{experiment.now()} {name}: {len(destination)}/{len(selected)}",
                    flush=True,
                )

        await asyncio.gather(*(one(c) for c in selected))
        saved.update(completed_at=experiment.now(), cases=len(selected))
        save()

    cases = e["cases"]
    for variant in experiment.VARIANTS:
        await stage(
            f"runtime:{variant}",
            cases,
            e["variants"][variant],
            lambda c, v=variant: experiment.runtime_case(record, c, v, provider),
        )
    await stage(
        "reference:primary",
        cases,
        e["references"],
        lambda c: experiment.reference_case(c, "primary", provider),
    )

    def outputs(case):
        return {v: e["variants"][v][case["case_id"]] for v in experiment.VARIANTS}

    await stage(
        "quotation:primary",
        cases,
        e["quote_reviews"],
        lambda c: experiment.quotation_case(c, outputs(c), "primary", provider),
    )
    await stage(
        "recheck:primary",
        cases,
        e["rechecks"],
        lambda c: experiment.recheck_case(
            c,
            outputs(c),
            e["references"][c["case_id"]]["source_reviews"],
            e["quote_reviews"][c["case_id"]]["reviews"],
            provider,
        ),
    )
    repeated = [c for c in cases if c["case_id"] in e["consistency_case_ids"]]
    for repeat_id in ("repeat_2", "repeat_3"):
        dest = e["repeats"][repeat_id]
        await stage(
            f"reference:{repeat_id}",
            repeated,
            dest["references"],
            lambda c, r=repeat_id: experiment.reference_case(c, r, provider),
        )
        await stage(
            f"quotation:{repeat_id}",
            repeated,
            dest["quotes"],
            lambda c, r=repeat_id: experiment.quotation_case(
                c, outputs(c), r, provider
            ),
        )
    e["completed_at"] = experiment.now()
    record["status"] = "completed_ai_evaluation"
    report(record, output)
    save()
    return record


def usage(receipts: list[dict]) -> dict:
    return {
        **experiment.receipt_totals(receipts),
        "requests": len(receipts),
        "request_statuses": dict(Counter(r["status"] for r in receipts)),
        "count_attempts": sum(len(r.get("count_attempts", [])) for r in receipts),
        "sum_count_and_generation_latency_seconds": sum(
            a.get("latency_seconds", 0)
            for r in receipts
            for a in r.get("attempts", []) + r.get("count_attempts", [])
        ),
    }


def report(record: dict, output: Path) -> dict:
    verify(record, output)
    e = record["execution"]
    grades, measurements, consistency = [], {}, []
    repeat_grades: dict[str, list[dict]] = {}
    for case in e["cases"]:
        key = case["case_id"]
        outputs = {v: e["variants"][v][key] for v in experiment.VARIANTS}
        grade = evaluation.grade_case(
            case,
            outputs,
            e["references"][key]["source_reviews"],
            e["quote_reviews"][key]["reviews"],
            e["rechecks"][key]["reviews"],
        )
        grades.append(grade)
        measurements[key] = {v: outputs[v]["measurement"] for v in experiment.VARIANTS}
        if key in e["consistency_case_ids"]:
            passes = [
                {
                    "source_reviews": e["references"][key]["source_reviews"],
                    "quote_reviews": e["quote_reviews"][key]["reviews"],
                    "grade": grade,
                }
            ]
            for repeat_id, repeat in e["repeats"].items():
                g = evaluation.grade_case(
                    case,
                    outputs,
                    repeat["references"][key]["source_reviews"],
                    repeat["quotes"][key]["reviews"],
                )
                repeat_grades.setdefault(repeat_id, []).append(g)
                passes.append(
                    {
                        "source_reviews": repeat["references"][key]["source_reviews"],
                        "quote_reviews": repeat["quotes"][key]["reviews"],
                        "grade": g,
                    }
                )
            consistency.append(
                {
                    "case_id": key,
                    "persona_id": case["persona_id"],
                    "reviews": passes,
                    "case": case,
                    "displayed_variants": [
                        v for v in outputs if outputs[v].get("selected")
                    ],
                }
            )
    e["grades"] = grades
    e["metrics"] = evaluation.summarize(grades, measurements)
    e["repeated_grades"] = repeat_grades
    e["consistency"] = evaluation.consistency_report(consistency)
    receipts = list(e["requests"].values())
    accounting: dict[str, Any] = {}
    for origin in ("retained", "new"):
        selected = [
            r
            for r in receipts
            if record["evidence_lineage"][r["request_hash"]]["origin"] == origin
        ]
        accounting[origin] = {
            "all": usage(selected),
            **{
                v: usage(
                    [
                        r
                        for r in selected
                        if r["request"]["purpose"].startswith(f"runtime:{v}:")
                    ]
                )
                for v in experiment.VARIANTS
            },
            "shared_ai_evaluation": usage(
                [r for r in selected if r["request"]["role"] == "reference"]
            ),
            "by_purpose": {
                purpose: usage(
                    [
                        r
                        for r in selected
                        if r["request"]["purpose"].startswith(purpose + ":")
                    ]
                )
                for purpose in sorted(
                    {r["request"]["purpose"].split(":")[0] for r in selected}
                )
            },
        }
    original = read(ORIGINAL)
    accounting["historical_all"] = usage(
        list(original["execution"]["requests"].values())
    )
    accounting["retained_plus_new_evidence"] = usage(receipts)
    accounting["local_retrieval"] = {
        k: e["retrieval"]["update_preparation"][k]
        for k in (
            "unique_documents",
            "preparation_seconds",
            "timing_seconds",
            "per_case_preparation_seconds",
        )
    }
    accounting["interpretation"] = (
        "New is incremental paid execution only. Retained/historical costs and "
        "latencies are prior observations, not new spend. Runtime metrics combine "
        "retained and newly observed requests with local selection and allocated "
        "retrieval overhead; they are not a fresh latency benchmark. Unknown usage "
        "remains unknown."
    )
    e["accounting"] = accounting
    e["completion_validation"] = {
        "cases": len(grades),
        "both_methods": all(len(e["variants"][v]) == 501 for v in experiment.VARIANTS),
        "unchanged_cases": 501 - len(record["affected_case_ids"]),
        "affected_cases": len(record["affected_case_ids"]),
        "primary_reference_cases": len(e["references"]),
        "quotation_cases": len(e["quote_reviews"]),
        "recheck_cases": len(e["rechecks"]),
        "consistency_cases": len(consistency),
        "human_reviews": 0,
        "attempts_bounded": all(
            len(r["attempts"]) <= r["max_attempts"]
            and len(r["count_attempts"]) <= r["max_attempts"]
            for r in receipts
        ),
    }
    old_grades = {g["case_id"]: g for g in original["execution"]["grades"]}
    unchanged = [g for g in grades if g["case_id"] not in record["affected_case_ids"]]
    if any(g != old_grades[g["case_id"]] for g in unchanged):
        raise ValueError("Unchanged context grades unexpectedly changed")
    e["completion_validation"]["unchanged_grades_identical"] = len(unchanged)
    e["reported_at"] = experiment.now()
    metrics: dict = e["metrics"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("audit", "prepare", "run", "report", "verify")
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if (
        not 1 <= args.concurrency <= 32
        or args.output.resolve() == ORIGINAL.parent.resolve()
    ):
        parser.error("Use a separate output directory and concurrency 1-32")
    key = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()
    with (Path(tempfile.gettempdir()) / f"twinkl-nsm-targeted-{key}.lock").open(
        "w"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "audit":
            print(json.dumps(audit(args.output)["summary"], indent=2))
        elif args.command == "prepare":
            record = prepare(args.output)
            print(json.dumps(record["impact_audit"]["summary"], indent=2))
        elif args.command == "run":
            asyncio.run(run(args.output, args.concurrency))
        else:
            record = read(args.output / "nsm_experiment.json")
            verify(record, args.output)
            if args.command == "report":
                report(record, args.output)
                write(args.output / "nsm_experiment.json", record)
            print("TARGETED_UPDATE_VERIFIED")


if __name__ == "__main__":
    main()
