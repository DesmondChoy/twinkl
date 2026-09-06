"""Assembled experiment ordering, blinding, freeze and failure regressions."""

from __future__ import annotations

import asyncio
import copy
import json

import pytest

from scripts.experiments import nsm_experiment as experiment
from src.north_star.provider import openai_input_payload, stable_hash
from tests.evals.test_nsm_evaluation import ACTION, VALUE, case


def small_case():
    item = case()
    item["values"][0].update(
        user_phrase="Caring for people close to me",
        approved_definition="Preserving and enhancing the welfare of close people.",
    )
    return item


class FakeProvider:
    calls = []
    fail_runtime = None
    fail_count = None

    def __init__(self, record, persist):
        self.record = record
        self.persist = persist
        self.requests = record["execution"]["requests"]

    async def measure(self, request, *, max_attempts=2):
        key = stable_hash(request)
        if key not in self.requests:
            self.requests[key] = {
                "request": request,
                "request_hash": key,
                "status": "counted",
                "count_receipt": {"input_tokens": 100},
                "count_attempts": [{"latency_seconds": 0.1}],
                "attempts": [],
                "result": None,
                "max_attempts": max_attempts,
            }
            if self.fail_count and self.fail_count in request["purpose"]:
                self.requests[key].update(status="count_failed", count_receipt=None)
            self.persist()
        return self.requests[key]

    async def assess(self, request, *, validator, max_attempts=2):
        receipt = await self.measure(request, max_attempts=max_attempts)
        if receipt["status"] in ("completed", "failed", "count_failed"):
            return receipt
        self.calls.append(request["purpose"])
        payload = json.loads(request["prompt"])
        quote = payload.get("proposed_quote")
        if quote:
            raw = {
                "schema_version": "north-star-candidate-assessment-v1",
                "core_value": payload["core_value"],
                "entry_id": quote["entry_id"],
                "quote_source": quote["quote_source"],
                "source_reason": "observable_choice",
                "quote_reason": "supported_action",
                "evaluated_quote": quote["evidence_quote"],
                "action_assessment": "Called Mum.",
                "value_assessment": "Cared for Mum.",
                "conflict_assessment": "None.",
                "quote_assessment": "Describes action.",
            }
        else:
            raw = {
                "schema_version": "north-star-source-assessment-v1",
                "core_value": payload["core_value"],
                "results": [
                    {
                        "entry_id": source["entry_id"],
                        "reason_code": "observable_choice",
                        "quote_source": "journal_entry",
                        "evidence_quote": ACTION,
                        "action_assessment": "Called Mum.",
                        "value_assessment": "Cared for Mum.",
                        "conflict_assessment": "None.",
                    }
                    for source in payload["sources"]
                ],
            }
        receipt["attempts"].append(
            {"calculated_cost_usd": 0.01, "latency_seconds": 0.2}
        )
        if self.fail_runtime and self.fail_runtime in request["purpose"]:
            receipt.update(status="failed")
        else:
            receipt.update(status="completed", result=validator(raw, request))
        self.persist()
        return receipt


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    item = small_case()
    control = case(case_id="control", persona="control", state="insufficient_evidence")
    built = {
        "cases": [item, control],
        "validation": {"status": "passed"},
        "provenance": {"source_hashes": {}},
        "consistency_case_ids": ["case"],
    }
    config = {"frozen_serialization": True}
    retrieval = {
        "config": config,
        "per_case_preparation_seconds": 0.5,
        "cases": {
            "case": {
                "candidate_ids_by_value": {VALUE: ["old"]},
                "latency_seconds": 0.01,
            },
            "control": {"candidate_ids_by_value": {}, "latency_seconds": 0.01},
        },
    }
    path = tmp_path / "nsm_experiment.json"
    path.write_text(
        json.dumps(
            {
                "execution": {},
                "frozen_settings": {},
                "methodology": "docs/north_star/nsm_experiment_methodology.md",
            }
        )
    )
    monkeypatch.setattr(experiment, "CODE_PATHS", ())
    monkeypatch.setattr(
        experiment.nsm_cases, "build_cases", lambda *a, **k: copy.deepcopy(built)
    )
    monkeypatch.setattr(experiment.nsm_cases, "retrieval_config", lambda *a: config)

    def encode(*args, **kwargs):
        assert (
            json.loads(path.read_text())["execution"]["retrieval_preparation_freeze"][
                "config"
            ]
            == config
        )
        return copy.deepcopy(retrieval)

    monkeypatch.setattr(experiment.nsm_cases, "prepare_retrieval", encode)
    monkeypatch.setattr(experiment, "ExperimentProvider", FakeProvider)
    monkeypatch.setattr(FakeProvider, "calls", [])
    monkeypatch.setattr(FakeProvider, "fail_runtime", None)
    monkeypatch.setattr(FakeProvider, "fail_count", None)
    summarize = experiment.nsm_evaluation.summarize
    monkeypatch.setattr(
        experiment.nsm_evaluation,
        "summarize",
        lambda grades, measurements: summarize(grades, measurements, n_resamples=10),
    )
    experiment.prepare(path)
    return path


def test_assembled_pipeline_order_repeat_isolation_priority_and_resume(prepared):
    result = asyncio.run(experiment.run(prepared, concurrency=2))
    calls = list(FakeProvider.calls)
    assert calls[0].startswith("runtime:nomic:")
    assert calls[1].startswith("runtime:full_history:")
    assert calls[2].startswith("reference:primary:")
    assert any("reference:repeat_2:" in purpose for purpose in calls)
    assert any("quotation:repeat_3:" in purpose for purpose in calls)
    assert all("control" not in purpose for purpose in calls)
    execution = result["execution"]
    assert execution["variants"]["nomic"]["case"]["selected"]["entry_id"] == "old"
    assert (
        execution["variants"]["full_history"]["case"]["selected"]["entry_id"] == "new"
    )
    grade = next(row for row in execution["grades"] if row["case_id"] == "case")
    assert grade["variants"]["nomic"]["selection_errors"] == {
        "included": True,
        "tp": 0,
        "fp": 1,
        "fn": 1,
    }
    assert grade["variants"]["full_history"]["correct_card"]
    assert (
        execution["variants"]["nomic"]["case"]["measurement"]["latency_seconds"] >= 0.81
    )
    assert execution["consistency"]["cases"] == 1
    asyncio.run(experiment.run(prepared, concurrency=2))
    assert FakeProvider.calls == calls


@pytest.mark.parametrize("failure", ["fail_runtime", "fail_count"])
def test_runtime_failure_preserves_paired_reference_opportunity(
    prepared, monkeypatch, failure
):
    monkeypatch.setattr(FakeProvider, failure, "runtime:full_history:")
    result = asyncio.run(experiment.run(prepared, concurrency=2))
    grade = next(
        row for row in result["execution"]["grades"] if row["case_id"] == "case"
    )
    assert grade["reference"]["opportunity"] is True
    metric = grade["variants"]["full_history"]["metrics"]["opportunity_recall"]
    assert (metric["numerator"], metric["denominator"], metric["excluded"]) == (
        0,
        1,
        False,
    )


def test_frozen_cases_are_verified(prepared):
    record = json.loads(prepared.read_text())
    record["execution"]["cases"][0]["week_start"] = "2025-01-01"
    with pytest.raises(ValueError, match="Frozen cases changed"):
        experiment.verify(record)


def test_variant_metadata_is_outside_semantic_input():
    item = small_case()
    value = item["values"][0]
    requests = [
        experiment.source_request(item, value, ["old"], f"runtime:{v}")
        for v in experiment.VARIANTS
    ]
    assert stable_hash(requests[0]) != stable_hash(requests[1])
    payloads = [openai_input_payload(r, experiment.DEFAULT_POLICY) for r in requests]
    assert payloads[0] == payloads[1]
    assert set(json.loads(payloads[0]["input"])["sources"][0]) == {
        "entry_id",
        "journal_entry",
        "nudge_response",
    }
