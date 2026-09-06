"""Prepare, execute and report the frozen two-variant North Star Moment study.

All requests, attempts and derived results live in the consolidated JSON record.
The run command resumes completed receipts without generating them again.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import fcntl
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments import nsm_cases, nsm_evaluation  # noqa: E402
from scripts.experiments.nsm_provider import (  # noqa: E402
    DEFAULT_POLICY,
    ExperimentProvider,
)
from src.north_star import assessment  # noqa: E402
from src.north_star.provider import stable_hash  # noqa: E402
from src.north_star.review import SourceEntry  # noqa: E402

DEFAULT_RECORD = (
    ROOT / "logs/experiments/reports/north_star_20260906/nsm_experiment.json"
)
VARIANTS = ("nomic", "full_history")
JSON_WRITER: ModuleType | None
try:
    JSON_WRITER = importlib.import_module("orjson")
except ImportError:
    JSON_WRITER = None
CODE_PATHS = (
    "scripts/experiments/nsm_experiment.py",
    "scripts/experiments/nsm_cases.py",
    "scripts/experiments/nsm_provider.py",
    "scripts/experiments/nsm_evaluation.py",
    "scripts/experiments/north_star_phase0.py",
    "src/north_star/assessment.py",
    "src/north_star/review.py",
    "src/north_star/runtime.py",
    "src/north_star/provider.py",
    "src/north_star/input_budget.py",
    "src/demo/contracts.py",
)


def now() -> str:
    return datetime.now(UTC).isoformat()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def persist_record(path: Path, record: dict) -> None:
    """Replace a complete checkpoint; never expose partially written JSON."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        payload = (
            JSON_WRITER.dumps(record)
            if JSON_WRITER
            else json.dumps(
                record, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        with temporary.open("wb") as file:
            file.write(payload + b"\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def value_arguments(value: dict) -> dict:
    return {
        key: value[key] for key in ("core_value", "user_phrase", "approved_definition")
    }


def source_request(case: dict, value: dict, ids: list[str], purpose: str) -> dict:
    selected = set(ids)
    sources = [
        SourceEntry.model_validate(source)
        for source in value["sources"]
        if source["entry_id"] in selected
    ]
    if len(sources) != len(selected):
        raise ValueError("Shortlist contains an unknown source")
    system, prompt = assessment.build_source_prompt(
        **value_arguments(value), sources=sources
    )
    return {
        "provider": "openai",
        "role": "runtime" if purpose.startswith("runtime:") else "reference",
        "purpose": f"{purpose}:{case['case_id']}:{value['core_value']}",
        "system": system,
        "prompt": prompt,
        "schema": assessment.source_json_schema(),
    }


def quote_request(case: dict, output: dict, purpose: str) -> dict:
    chosen = output["selected"]
    value = next(v for v in case["values"] if v["core_value"] == output["core_value"])
    source = next(s for s in value["sources"] if s["entry_id"] == chosen["entry_id"])
    system, prompt = assessment.build_candidate_prompt(
        **value_arguments(value),
        source=SourceEntry.model_validate(source),
        quote_source=chosen["quote_source"],
        evidence_quote=chosen["evidence_quote"],
    )
    return {
        "provider": "openai",
        "role": "reference",
        "purpose": f"{purpose}:{case['case_id']}",
        "system": system,
        "prompt": prompt,
        "schema": assessment.candidate_json_schema(),
    }


def source_validator(value: dict, ids: list[str]):
    sources = [
        SourceEntry.model_validate(source)
        for source in value["sources"]
        if source["entry_id"] in set(ids)
    ]

    def validate(raw: dict, _request: dict) -> dict:
        return assessment.validate_source_review(
            raw, core_value=value["core_value"], sources=sources
        ).model_dump()

    return validate


def quote_validator(case: dict, output: dict):
    selected = output["selected"]
    value = next(v for v in case["values"] if v["core_value"] == output["core_value"])
    source = next(s for s in value["sources"] if s["entry_id"] == selected["entry_id"])

    def validate(raw: dict, _request: dict) -> dict:
        return assessment.validate_candidate_review(
            raw,
            core_value=value["core_value"],
            source=SourceEntry.model_validate(source),
            quote_source=selected["quote_source"],
            evidence_quote=selected["evidence_quote"],
        ).model_dump()

    return validate


def candidates(record: dict, case: dict, variant: str) -> dict[str, list[str]]:
    if variant == "nomic":
        result: dict[str, list[str]] = record["execution"]["retrieval"]["cases"][
            case["case_id"]
        ]["candidate_ids_by_value"]
        return result
    return {
        v["core_value"]: [s["entry_id"] for s in v["sources"]] for v in case["values"]
    }


def prepare(path: Path) -> dict:
    record: dict = json.loads(path.read_text())
    execution = record["execution"]
    if execution.get("freeze"):
        verify(record)
        return record
    if execution.get("requests"):
        raise ValueError("Cannot prepare over existing experiment requests")
    built = nsm_cases.build_cases(record, root=ROOT)
    cases = built["cases"]
    serialization = nsm_cases.retrieval_config(ROOT)
    execution["retrieval_preparation_freeze"] = {
        "frozen_at": now(),
        "config": serialization,
        "cases_sha256": stable_hash(cases),
        "consistency_case_ids": built["consistency_case_ids"],
        "code_sha256": {p: file_hash(ROOT / p) for p in CODE_PATHS},
    }
    persist_record(path, record)
    retrieval = nsm_cases.prepare_retrieval(cases, root=ROOT)
    if retrieval["config"] != serialization:
        raise ValueError("Retrieval serialization differs from its pre-encoding freeze")
    execution.update(
        {
            "cases": cases,
            "case_validation": built["validation"],
            "case_provenance": built["provenance"],
            "retrieval": retrieval,
            "consistency_case_ids": built["consistency_case_ids"],
            "provider_policy": copy.deepcopy(DEFAULT_POLICY),
            "requests": {},
            "variants": {v: {} for v in VARIANTS},
            "references": {},
            "quote_reviews": {},
            "rechecks": {},
            "repeats": {},
            "stages": {},
        }
    )
    request_hashes: dict[str, str] = {}
    for case in cases:
        for variant in VARIANTS:
            ids = candidates(record, case, variant)
            for value in case["values"]:
                if ids.get(value["core_value"]):
                    req = source_request(
                        case, value, ids[value["core_value"]], f"runtime:{variant}"
                    )
                    request_hashes[req["purpose"]] = stable_hash(req)
        repeat_ids = ["primary"]
        if case["case_id"] in execution["consistency_case_ids"]:
            repeat_ids.extend(("repeat_2", "repeat_3"))
        for repeat_id in repeat_ids:
            for value in case["values"]:
                value_ids = [s["entry_id"] for s in value["sources"]]
                if value_ids:
                    req = source_request(
                        case, value, value_ids, f"reference:{repeat_id}"
                    )
                    request_hashes[req["purpose"]] = stable_hash(req)
    execution["freeze"] = {
        "frozen_at": now(),
        "code_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "code_sha256": {p: file_hash(ROOT / p) for p in CODE_PATHS},
        "methodology_sha256": file_hash(ROOT / record["methodology"]),
        "methodology_text": (ROOT / record["methodology"]).read_text(),
        "cases_sha256": stable_hash(cases),
        "retrieval_sha256": stable_hash(retrieval),
        "policy_sha256": stable_hash(execution["provider_policy"]),
        "settings_sha256": stable_hash(record["frozen_settings"]),
        "consistency_sample_sha256": stable_hash(execution["consistency_case_ids"]),
        "source_request_hashes": request_hashes,
        "source_system_prompt": assessment.SOURCE_SYSTEM_PROMPT,
        "candidate_system_prompt": assessment.CANDIDATE_SYSTEM_PROMPT,
        "source_schema": assessment.source_json_schema(),
        "candidate_schema": assessment.candidate_json_schema(),
        "variant_order": list(VARIANTS),
        "final_evaluation_rule": "Settings frozen before all generation; no tuning.",
        "dynamic_requests": "Exact quotations depend on runtime; frozen templates and "
        "schemas apply and each complete input is counted before generation.",
        "runtime_latency_basis": "Sum of count and generation attempt durations, "
        "local quotation selection, and Nomic ranking plus equally allocated "
        "embedding preparation. Excludes batch queue and experiment checkpoint I/O. "
        "Stage timestamps separately describe experiment wall time.",
        "json_writer": {
            "module": "orjson" if JSON_WRITER else "json",
            "version": getattr(JSON_WRITER, "__version__", sys.version),
        },
    }
    record["preparation_code_revision"] = record.get("code_revision")
    record["code_revision"] = execution["freeze"]["code_revision"]
    record["schema_version"] = "north-star-experiment-v1"
    record["scope"] = (
        "Fresh paired Nomic shortlist and full eligible-history NSM experiment, "
        "with shared Luna-xhigh AI evaluation; prior preparation audits retained."
    )
    execution["authorization"] = (
        "User requested reading the methodology and running both experiments. "
        "The methodology authorizes these API calls without monetary caps."
    )
    record["status"] = "prepared_frozen_generation_pending"
    record["remaining_preparation"] = []
    persist_record(path, record)
    return record


def _reporting_corrections(record: dict) -> dict:
    """Accept documented grading repairs only after every generation stage ends."""
    corrections = record.get("reporting", {}).get("code_corrections", {})
    if not isinstance(corrections, dict):
        raise ValueError("Malformed reporting code corrections")
    if not corrections:
        return corrections
    execution = record["execution"]
    cases = {case["case_id"] for case in execution["cases"]}
    repeated = set(execution["consistency_case_ids"])
    destinations = {
        **{
            f"runtime:{variant}": (execution["variants"].get(variant, {}), cases)
            for variant in VARIANTS
        },
        "reference:primary": (execution.get("references", {}), cases),
        "quotation:primary": (execution.get("quote_reviews", {}), cases),
        "recheck:primary": (execution.get("rechecks", {}), cases),
        **{
            f"{stage}:{repeat}": (
                execution.get("repeats", {}).get(repeat, {}).get(field, {}),
                repeated,
            )
            for repeat in ("repeat_2", "repeat_3")
            for stage, field in (("reference", "references"), ("quotation", "quotes"))
        },
    }
    if (
        record.get("status") != "completed_ai_evaluation"
        or not execution.get("completed_at")
        or any(
            set(destination) != expected
            or not execution.get("stages", {}).get(stage, {}).get("completed_at")
            for stage, (destination, expected) in destinations.items()
        )
        or any(
            attempt.get("status") == "pending"
            for receipt in execution.get("requests", {}).values()
            for field in ("count_attempts", "attempts")
            for attempt in receipt.get(field, [])
        )
    ):
        raise ValueError("Reporting code corrections require completed execution")
    allowed = {
        "scripts/experiments/nsm_experiment.py",
        "scripts/experiments/nsm_evaluation.py",
    }
    for name, correction in corrections.items():
        if name not in allowed or not isinstance(correction, dict):
            raise ValueError(f"Unsupported reporting code correction: {name}")
        original = correction.get("original_source_text")
        if (
            not isinstance(original, str)
            or hashlib.sha256(original.encode()).hexdigest()
            != execution["freeze"]["code_sha256"].get(name)
            or not isinstance(correction.get("reason"), str)
            or not correction["reason"].strip()
        ):
            raise ValueError(f"Unverified original reporting code: {name}")
        if correction.get("corrected_sha256") != file_hash(ROOT / name):
            raise ValueError(f"Corrected reporting code changed: {name}")
    return corrections


def verify(record: dict, *, allow_reporting_corrections: bool = False) -> None:
    execution = record["execution"]
    frozen = execution["freeze"]
    corrections = _reporting_corrections(record) if allow_reporting_corrections else {}
    for name, expected in frozen["code_sha256"].items():
        if file_hash(ROOT / name) != expected and name not in corrections:
            raise ValueError(f"Frozen experiment code changed: {name}")
    for name, expected in execution["case_provenance"]["source_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError(f"Frozen experiment source changed: {name}")
    if (
        hashlib.sha256(frozen["methodology_text"].encode()).hexdigest()
        != frozen["methodology_sha256"]
    ):
        raise ValueError("Frozen methodology snapshot changed")
    expected_methodology = record.get("reporting", {}).get(
        "methodology_sha256", frozen["methodology_sha256"]
    )
    if file_hash(ROOT / record["methodology"]) != expected_methodology:
        raise ValueError("Frozen experiment methodology changed")
    for key, value in (
        ("cases", execution["cases"]),
        ("retrieval", execution["retrieval"]),
        ("policy", execution["provider_policy"]),
        ("settings", record["frozen_settings"]),
        ("consistency_sample", execution["consistency_case_ids"]),
    ):
        if stable_hash(value) != frozen[f"{key}_sha256"]:
            raise ValueError(f"Frozen {key} changed")


def receipt_totals(receipts: list[dict]) -> dict:
    generation = [a for r in receipts for a in r.get("attempts", [])]
    return {
        "cost_usd": sum(a.get("calculated_cost_usd") or 0 for a in generation),
        "unknown_cost_attempts": sum(
            a.get("calculated_cost_usd") is None for a in generation
        ),
        "attempts": len(generation),
        "input_tokens": sum(a.get("input_tokens") or 0 for a in generation),
        "output_tokens": sum(a.get("output_tokens") or 0 for a in generation),
        "cached_input_tokens": sum(
            a.get("cached_input_tokens") or 0 for a in generation
        ),
        "actual_models": dict(
            Counter(a.get("actual_model") or "unknown" for a in generation)
        ),
    }


async def runtime_case(
    record: dict, case: dict, variant: str, provider: ExperimentProvider
) -> dict:
    ids = candidates(record, case, variant)
    reviews, receipts, failures = {}, [], []
    for value in case["values"]:
        core_value = value["core_value"]
        if not ids.get(core_value):
            continue
        request = source_request(case, value, ids[core_value], f"runtime:{variant}")
        receipt = await provider.assess(
            request, validator=source_validator(value, ids[core_value])
        )
        receipts.append(receipt)
        reviews[core_value] = receipt.get("result")
        if receipt["status"] != "completed":
            failures.append(
                {
                    "core_value": core_value,
                    "status": receipt["status"],
                    "request_hash": receipt["request_hash"],
                }
            )
    selection_started = time.perf_counter()
    output = nsm_evaluation.select_card(case, reviews, ids)
    selection_seconds = time.perf_counter() - selection_started
    # A missing value review can hide a preferred candidate, so runtime fails closed.
    if failures:
        output.update(
            selected=None,
            core_value=None,
            mode=None,
            status="failed",
            reason="source_assessment_failed",
            failures=failures,
        )
    overhead = 0.0
    if variant == "nomic":
        retrieval = record["execution"]["retrieval"]
        row = retrieval["cases"][case["case_id"]]
        overhead = row["latency_seconds"] + retrieval["per_case_preparation_seconds"]
    output.update(
        source_reviews=reviews,
        request_hashes=[r["request_hash"] for r in receipts],
        measurement={
            **receipt_totals(receipts),
            "latency_seconds": sum(
                attempt.get("latency_seconds", 0.0)
                for receipt in receipts
                for attempt in receipt.get("count_attempts", [])
                + receipt.get("attempts", [])
            )
            + selection_seconds
            + overhead,
            "local_selection_seconds": selection_seconds,
            "embedding_overhead_seconds": overhead,
        },
    )
    return output


async def reference_case(
    case: dict, repeat_id: str, provider: ExperimentProvider
) -> dict:
    reviews, hashes, failures = {}, [], []
    for value in case["values"]:
        ids = [source["entry_id"] for source in value["sources"]]
        if not ids:
            continue
        request = source_request(case, value, ids, f"reference:{repeat_id}")
        receipt = await provider.assess(request, validator=source_validator(value, ids))
        hashes.append(receipt["request_hash"])
        reviews[value["core_value"]] = receipt.get("result")
        if receipt["status"] != "completed":
            failures.append(
                {"core_value": value["core_value"], "status": receipt["status"]}
            )
    return {"source_reviews": reviews, "request_hashes": hashes, "failures": failures}


async def quotation_case(
    case: dict, outputs: dict, repeat_id: str, provider: ExperimentProvider
) -> dict:
    reviews, hashes = {}, {}
    for variant, output in outputs.items():
        if output.get("selected") is None:
            continue
        request = quote_request(case, output, f"quotation:{repeat_id}:{variant}")
        receipt = await provider.assess(
            request, validator=quote_validator(case, output)
        )
        reviews[variant] = receipt.get("result")
        hashes[variant] = receipt["request_hash"]
    return {"reviews": reviews, "request_hashes": hashes}


async def recheck_case(
    case: dict, outputs: dict, sources: dict, quotes: dict, provider: ExperimentProvider
) -> dict:
    needed = nsm_evaluation.contradictions(case, outputs, sources, quotes)
    reviews, hashes = {}, {}
    for conflict in needed:
        variant = conflict["variant"]
        request = quote_request(case, outputs[variant], f"recheck:primary:{variant}")
        receipt = await provider.assess(
            request, max_attempts=1, validator=quote_validator(case, outputs[variant])
        )
        reviews[variant] = receipt.get("result")
        hashes[variant] = receipt["request_hash"]
    return {"reviews": reviews, "request_hashes": hashes}


async def run(path: Path, concurrency: int = 8) -> dict:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    record: dict = json.loads(path.read_text())
    verify(record)
    execution = record["execution"]

    def save() -> None:
        persist_record(path, record)

    provider = ExperimentProvider(record, save)
    cases = execution["cases"]
    semaphore = asyncio.Semaphore(concurrency)

    static_requests = []
    for case in cases:
        for variant in VARIANTS:
            ids = candidates(record, case, variant)
            for value in case["values"]:
                if ids.get(value["core_value"]):
                    static_requests.append(
                        source_request(
                            case, value, ids[value["core_value"]], f"runtime:{variant}"
                        )
                    )
        repeat_ids = ["primary"]
        if case["case_id"] in execution["consistency_case_ids"]:
            repeat_ids.extend(("repeat_2", "repeat_3"))
        for repeat_id in repeat_ids:
            for value in case["values"]:
                value_ids = [s["entry_id"] for s in value["sources"]]
                if value_ids:
                    static_requests.append(
                        source_request(case, value, value_ids, f"reference:{repeat_id}")
                    )
    expected = execution["freeze"]["source_request_hashes"]
    if {r["purpose"]: stable_hash(r) for r in static_requests} != expected:
        raise ValueError(
            "Complete source requests differ from the pre-generation freeze"
        )

    async def measure(request: dict) -> dict:
        async with semaphore:
            receipt = await provider.measure(request)
            return receipt

    # One real request checks connectivity before scheduling the complete count audit.
    if static_requests:
        print(
            f"{now()} counting {len(static_requests)} frozen source requests",
            flush=True,
        )
        probe = await measure(static_requests[0])
        if probe.get("count_receipt") is None and len(execution["requests"]) == 1:
            raise RuntimeError(
                f"Input-count connectivity failed: {probe['request_hash']} "
                f"{probe.get('error_type', probe['status'])}"
            )
        await asyncio.gather(*(measure(request) for request in static_requests[1:]))
    measured = [execution["requests"][stable_hash(r)] for r in static_requests]
    execution["final_source_input_counts"] = {
        "completed_at": now(),
        "requests": len(static_requests),
        "maximum_input_tokens": max(
            (
                r["count_receipt"]["input_tokens"]
                for r in measured
                if r.get("count_receipt")
            ),
            default=0,
        ),
        "limit": 16000,
        "truncation": False,
        "failed_requests": [
            r["request_hash"] for r in measured if not r.get("count_receipt")
        ],
    }
    save()

    async def stage(
        name: str, selected_cases: list[dict], destination: dict, work
    ) -> None:
        saved = execution["stages"].setdefault(
            name, {"started_at": now(), "concurrency": concurrency}
        )
        print(
            f"{now()} {name}: {len(destination)}/{len(selected_cases)} saved",
            flush=True,
        )

        async def one(case: dict) -> None:
            if case["case_id"] in destination:
                return
            async with semaphore:
                destination[case["case_id"]] = await work(case)
                save()
                done = len(destination)
                if done % 10 == 0 or done == len(selected_cases):
                    print(
                        f"{now()} {name}: {done}/{len(selected_cases)} complete",
                        flush=True,
                    )

        await asyncio.gather(*(one(case) for case in selected_cases))
        saved.update(completed_at=now(), cases=len(selected_cases))
        save()

    record["status"] = "running"
    save()
    for variant in VARIANTS:
        await stage(
            f"runtime:{variant}",
            cases,
            execution["variants"][variant],
            lambda case, variant=variant: runtime_case(record, case, variant, provider),
        )
    await stage(
        "reference:primary",
        cases,
        execution["references"],
        lambda case: reference_case(case, "primary", provider),
    )

    def outputs(case: dict) -> dict:
        return {v: execution["variants"][v][case["case_id"]] for v in VARIANTS}

    await stage(
        "quotation:primary",
        cases,
        execution["quote_reviews"],
        lambda case: quotation_case(case, outputs(case), "primary", provider),
    )
    await stage(
        "recheck:primary",
        cases,
        execution["rechecks"],
        lambda case: recheck_case(
            case,
            outputs(case),
            execution["references"][case["case_id"]]["source_reviews"],
            execution["quote_reviews"][case["case_id"]]["reviews"],
            provider,
        ),
    )
    repeat_cases = [
        c for c in cases if c["case_id"] in execution["consistency_case_ids"]
    ]
    for repeat_id in ("repeat_2", "repeat_3"):
        repeat = execution["repeats"].setdefault(
            repeat_id, {"references": {}, "quotes": {}}
        )
        await stage(
            f"reference:{repeat_id}",
            repeat_cases,
            repeat["references"],
            lambda case, repeat_id=repeat_id: reference_case(case, repeat_id, provider),
        )
        await stage(
            f"quotation:{repeat_id}",
            repeat_cases,
            repeat["quotes"],
            lambda case, repeat_id=repeat_id: quotation_case(
                case, outputs(case), repeat_id, provider
            ),
        )
    report(record)
    save()
    return record


def report(record: dict) -> dict:
    verify(record, allow_reporting_corrections=True)
    execution = record["execution"]
    grades, measurements = [], {}
    for case in execution["cases"]:
        key = case["case_id"]
        outputs = {v: execution["variants"][v][key] for v in VARIANTS}
        grades.append(
            nsm_evaluation.grade_case(
                case,
                outputs,
                execution["references"][key]["source_reviews"],
                execution["quote_reviews"][key]["reviews"],
                execution["rechecks"][key]["reviews"],
            )
        )
        measurements[key] = {v: outputs[v]["measurement"] for v in VARIANTS}
    execution["grades"] = grades
    execution["metrics"] = nsm_evaluation.summarize(grades, measurements)
    repeated_grades = {}
    for repeat_id, repeat in execution["repeats"].items():
        rows = []
        for case in execution["cases"]:
            key = case["case_id"]
            if key not in repeat["references"]:
                continue
            outputs = {v: execution["variants"][v][key] for v in VARIANTS}
            rows.append(
                nsm_evaluation.grade_case(
                    case,
                    outputs,
                    repeat["references"][key]["source_reviews"],
                    repeat["quotes"][key]["reviews"],
                )
            )
        repeated_grades[repeat_id] = rows
    consistency_cases = []
    for case in execution["cases"]:
        key = case["case_id"]
        if key not in execution["consistency_case_ids"]:
            continue
        passes = [
            {
                "source_reviews": execution["references"][key]["source_reviews"],
                "quote_reviews": execution["quote_reviews"][key]["reviews"],
                "grade": next(g for g in grades if g["case_id"] == key),
            }
        ]
        for repeat_id, repeat in execution["repeats"].items():
            passes.append(
                {
                    "source_reviews": repeat["references"][key]["source_reviews"],
                    "quote_reviews": repeat["quotes"][key]["reviews"],
                    "grade": next(
                        g for g in repeated_grades[repeat_id] if g["case_id"] == key
                    ),
                }
            )
        consistency_cases.append(
            {
                "case_id": key,
                "persona_id": case["persona_id"],
                "reviews": passes,
                "case": case,
                "displayed_variants": [
                    v for v in VARIANTS if execution["variants"][v][key].get("selected")
                ],
            }
        )
    execution["consistency"] = nsm_evaluation.consistency_report(consistency_cases)
    receipts = list(execution["requests"].values())
    runtime = [r for r in receipts if r["request"]["role"] == "runtime"]
    reference = [r for r in receipts if r["request"]["role"] == "reference"]
    execution["usage"] = {
        "runtime": receipt_totals(runtime),
        "shared_ai_evaluation": receipt_totals(reference),
        "all": receipt_totals(receipts),
        "request_statuses": dict(Counter(r["status"] for r in receipts)),
    }
    execution["runtime_generation_calls"] = execution["usage"]["runtime"]["attempts"]
    execution["reference_generation_calls"] = execution["usage"][
        "shared_ai_evaluation"
    ]["attempts"]
    execution["reported_at"] = now()
    execution.setdefault("completed_at", execution["reported_at"])
    execution["completion_validation"] = {
        "case_count": len(execution["cases"]),
        "both_variants_complete": all(
            len(execution["variants"][v]) == len(execution["cases"]) for v in VARIANTS
        ),
        "primary_reference_cases": len(execution["references"]),
        "quotation_review_cases": len(execution["quote_reviews"]),
        "recheck_cases_processed": len(execution["rechecks"]),
        "evaluator_consistency_cases": execution["consistency"]["cases"],
        "human_reviews": 0,
        "all_attempts_bounded": all(
            len(r["attempts"]) <= r["max_attempts"] for r in receipts
        ),
        "source_counts_before_generation": True,
        "frozen_code_verified": True,
    }
    record["status"] = "completed_ai_evaluation"
    metrics: dict = execution["metrics"]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report", "verify"))
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32:
        parser.error("concurrency must be between 1 and 32")
    key = hashlib.sha256(str(args.record.resolve()).encode()).hexdigest()
    with (Path(tempfile.gettempdir()) / f"twinkl-nsm-experiment-{key}.lock").open(
        "w"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "prepare":
            result = prepare(args.record)
            print(json.dumps(result["execution"]["case_validation"], indent=2))
        elif args.command == "run":
            asyncio.run(run(args.record, args.concurrency))
        else:
            result = json.loads(args.record.read_text())
            verify(result, allow_reporting_corrections=True)
            if args.command == "report":
                print(json.dumps(report(result), indent=2))
                persist_record(args.record, result)
            else:
                print("FROZEN_EXPERIMENT_VERIFIED")


if __name__ == "__main__":
    main()
