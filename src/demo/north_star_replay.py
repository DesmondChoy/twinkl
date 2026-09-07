"""Project completed experiment observations into saved replay, without model calls.

Experiment receipts retain their original identity and policy. They are never
rewritten into live-runtime receipts or used as live-session cache entries.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.north_star import assessment, runtime
from src.north_star.input_budget import validate_receipt
from src.north_star.provider import openai_input_payload, stable_hash

EXPERIMENT_PATH = Path(
    "logs/experiments/reports/north_star_v4_run1_20260907/nsm_experiment.json"
)
EXPERIMENT_SHA256 = "2b86aecd7809892b6b15e660e613ddc546a816da025330283dbfb487660a31b9"
EXPORT_PATH = Path("src/demo/north_star_replay_records.json")


@lru_cache(maxsize=8)
def _source_file_sha256(path: Path, _modified_ns: int, _size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_experiment_source(root: Path) -> None:
    """Reject changed local evidence; production may omit the full study file."""
    source = (root / EXPERIMENT_PATH).resolve()
    if not source.exists():
        return
    stat = source.stat()
    if _source_file_sha256(source, stat.st_mtime_ns, stat.st_size) != EXPERIMENT_SHA256:
        raise ValueError("Saved NSM experiment source hash differs from pinned study")


class ExperimentEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    method: Literal["full_history"] = "full_history"
    case_id: str
    case: dict[str, Any] | None
    output: dict[str, Any] | None
    receipts: list[dict[str, Any]]
    policy: dict[str, Any]
    availability_basis: Literal["synthetic_immediate_parent_order"] = (
        "synthetic_immediate_parent_order"
    )


class SavedExperimentRecord(runtime.NorthStarRecord):
    experiment: ExperimentEvidence


def project_record(
    request: runtime.NorthStarRequest,
    evidence: ExperimentEvidence,
    *,
    created_at: str,
) -> SavedExperimentRecord:
    """Verify semantic compatibility and derive the exact recorded selection."""
    runtime._validate_request(request)
    values, sources, onset, onset_at, reason = runtime._context(request)
    base = runtime.pending_north_star_record(request).model_dump(mode="json")
    base.update(created_at=created_at, experiment=evidence.model_dump(mode="json"))
    if (
        evidence.source_path != EXPERIMENT_PATH.as_posix()
        or evidence.source_sha256 != EXPERIMENT_SHA256
        or evidence.case_id != f"{request.owner_id}:week:{request.week_start}"
    ):
        raise ValueError("Saved NSM experiment identity differs")
    case, output = evidence.case, evidence.output
    if case is None:
        if output is not None or evidence.receipts:
            raise ValueError("An unassessed week cannot contain experiment outputs")
        base.update(reason="not_in_completed_experiment", status="pending")
        return SavedExperimentRecord.model_validate(base)
    if output is None:
        raise ValueError("Completed experiment case lacks its runtime output")
    for field, expected in {
        "case_id": evidence.case_id,
        "persona_id": request.owner_id,
        "week_start": request.week_start,
        "week_end": request.week_end,
        "core_values": request.core_values,
        "weekly_state": request.drift_result.delivery_state,
        "drift_result": request.drift_result.model_dump(mode="json"),
    }.items():
        if case[field] != expected:
            raise ValueError(f"Saved NSM experiment context differs: {field}")
    if case["context_reason"] != reason:
        raise ValueError("Saved NSM experiment eligibility differs")
    if [value["core_value"] for value in case["values"]] != values:
        raise ValueError("Saved NSM experiment Core Value order differs")
    review_sources = runtime._review_sources(sources)
    for value in case["values"]:
        definition = request.value_definitions[value["core_value"]]
        if any(
            value[key] != expected for key, expected in definition.model_dump().items()
        ):
            raise ValueError("Saved NSM experiment Core Value definition differs")
        if value["sources"] != [source.model_dump() for source in review_sources]:
            raise ValueError("Saved NSM experiment writing differs")
        for source in sources:
            metadata = case["source_metadata"][source.entry_id]
            if any(
                metadata[key] != getattr(source, key)
                for key in ("owner_id", "entry_id", "date", "t_index")
            ):
                raise ValueError("Saved NSM source chronology differs")
    if evidence.policy["runtime"]["model"] != "gpt-5.6-luna" or (
        evidence.policy["runtime"]["reasoning_effort"] != "low"
    ):
        raise ValueError("Saved NSM runtime model differs")
    receipts = evidence.receipts
    if output["request_hashes"] != [receipt["request_hash"] for receipt in receipts]:
        raise ValueError("Saved NSM request coverage differs")
    if len(receipts) != (len(values) if reason == "eligible" else 0):
        raise ValueError("Saved NSM review coverage is incomplete")
    reviewed = []
    failed = False
    for value, receipt in zip(values if receipts else [], receipts, strict=True):
        definition = request.value_definitions[value]
        system, prompt = assessment.build_source_prompt(
            **definition.model_dump(), sources=review_sources
        )
        original_request = {
            "provider": "openai",
            "role": "runtime",
            "purpose": f"runtime:full_history:{evidence.case_id}:{value}",
            "system": system,
            "prompt": prompt,
            "schema": assessment.source_json_schema(),
        }
        if receipt["request"] != original_request or (
            receipt["request_hash"] != stable_hash(original_request)
            or receipt["policy_hash"] != stable_hash(evidence.policy)
            or receipt["payload_hash"]
            != stable_hash(openai_input_payload(original_request, evidence.policy))
        ):
            raise ValueError("Saved NSM original receipt binding differs")
        if receipt["status"] != "completed":
            failed = True
            continue
        if (
            validate_receipt(
                original_request, evidence.policy, receipt["count_receipt"]
            )
            > 16000
        ):
            raise ValueError("Saved NSM input exceeds the complete-input ceiling")
        attempt = receipt["attempts"][-1]
        if attempt["status"] != "completed" or (
            attempt["requested_model"] != "gpt-5.6-luna"
            or attempt["reasoning_effort"] != "low"
        ):
            raise ValueError("Saved NSM completed attempt differs")
        batch = assessment.validate_source_review(
            attempt["raw_text"], core_value=value, sources=review_sources
        )
        if batch.model_dump(mode="json") != receipt["result"] or (
            output["source_reviews"][value] != receipt["result"]
        ):
            raise ValueError("Saved NSM assessment differs from raw response")
        reviewed.append((value, batch))
    selected = runtime._pick(request, reviewed, sources) if not failed else None
    status = (
        "failed"
        if failed
        else "not_eligible"
        if reason != "eligible"
        else "complete"
    )
    expected_selection = None
    core_value, mode = None, None
    if selected:
        core_value, ruling, source = selected
        assert ruling.quote_source is not None
        expected_selection = runtime.NorthStarSelection(
            entry_id=source.entry_id,
            t_index=source.t_index,
            date=source.date,
            quote_source=ruling.quote_source,
            evidence_quote=ruling.evidence_quote,
        ).model_dump(mode="json")
        mode = (
            "reflection"
            if onset
            else ("encouragement" if source.date >= request.week_start else "reminder")
        )
    if any(
        output[key] != expected
        for key, expected in {
            "status": status,
            "selected": expected_selection,
            "core_value": core_value,
            "mode": mode,
        }.items()
    ):
        raise ValueError("Saved NSM selection differs from completed experiment")
    base.update(
        status=status,
        reason=output["reason"],
        mode=mode,
        core_value=core_value,
        value_phrase=request.value_definitions[core_value].user_phrase
        if core_value
        else None,
        selected=expected_selection,
        sources=[s.model_dump(mode="json") for s in sources],
        source_ids=[s.entry_id for s in sources],
        onset_available_at=onset_at,
        validation_evidence=[
            "completed_v4_run1_full_history_experiment",
            "original_provider_receipts_preserved",
            "exact_semantic_input_match",
            "synthetic_immediate_parent_order",
            "ai_assessment_not_human_validation",
        ],
        # These are historical attempts, not requests made during replay.
        attempts=sum(len(receipt["attempts"]) for receipt in receipts),
    )
    return SavedExperimentRecord.model_validate(base)


def validate_saved_record(
    record: runtime.NorthStarRecord, request: runtime.NorthStarRequest
) -> runtime.NorthStarRecord:
    if not isinstance(record, SavedExperimentRecord):
        return runtime.validate_north_star_record(record, request)
    expected = project_record(request, record.experiment, created_at=record.created_at)
    if record != expected:
        raise ValueError("Saved NSM projection differs from experiment evidence")
    return record


def export_records(root: Path) -> dict[str, Any]:
    """Export all saved weeks; absent cohort members remain explicitly pending."""
    from src.demo.scenarios import (
        SELECTIONS,
        build_saved_north_star_request,
        build_scenario_fixture,
    )

    raw = (root / EXPERIMENT_PATH).read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPERIMENT_SHA256:
        raise ValueError("Saved NSM experiment source hash differs from pinned study")
    experiment = json.loads(raw)
    if experiment["status"] != "completed_ai_evaluation":
        raise ValueError("NSM experiment is not complete")
    execution = experiment["execution"]
    cases = {case["case_id"]: case for case in execution["cases"]}
    rows = []
    for selection in SELECTIONS:
        fixture = build_scenario_fixture(root, selection, include_north_star=False)
        for week in fixture.scenario.weeks:
            request = build_saved_north_star_request(fixture, week.week_id)
            case_id = f"{selection.persona_id}:week:{week.week_start}"
            case = cases.get(case_id)
            output = execution["variants"]["full_history"].get(case_id)
            evidence = ExperimentEvidence(
                source_path=EXPERIMENT_PATH.as_posix(),
                source_sha256=hashlib.sha256(raw).hexdigest(),
                case_id=case_id,
                case=case,
                output=output,
                receipts=[
                    execution["requests"][key] for key in output["request_hashes"]
                ]
                if output
                else [],
                policy=execution["provider_policy"],
            )
            record = project_record(
                request, evidence, created_at=execution["completed_at"]
            )
            rows.append(
                {
                    "scenario_id": selection.scenario_id,
                    "week_id": week.week_id,
                    "record": record.model_dump(mode="json"),
                }
            )
    report = {"schema_version": "saved-nsm-experiment-export-v1", "cases": rows}
    (root / EXPORT_PATH).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report
