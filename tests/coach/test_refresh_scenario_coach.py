"""Refresh preserves originals, records bounded calls, and publishes all-or-nothing."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from scripts.coach import refresh_scenario_coach as runner
from src.coach.schemas import LLMCallMetrics
from src.demo.scenarios import (
    COACH_RESPONSES_PATH,
    SavedCoachResponseFixture,
    _weekly_drift_input_sha256,
    load_saved_coach_responses,
)
from tests.coach.test_generate_approved_judge_sample import _digest


def _response() -> str:
    return json.dumps(
        {
            "weekly_mirror": (
                'You found ways to care for people around you: you "called my mom '
                'and helped a colleague debug".'
            ),
            "tension_explanation": (
                "Those two small choices made room for connection alongside "
                "the rest of your day."
            ),
            "reflective_question": (
                "Which of those moments would you like to make room for again?"
            ),
        }
    )


def _setup(tmp_path: Path, monkeypatch, *, two_cases: bool = False):
    digest = _digest("casey")
    key = "casey-scenario::2025-01-01"
    cases = {
        key: {
            "digest": digest.model_dump(mode="json"),
            "input_sha256": _weekly_drift_input_sha256(digest),
            "scenario_id": "casey-scenario",
            "source_bundle_path": "bundle.json",
            "source_bundle_content_sha256": "a" * 64,
            "weekly_digest_event_id": "casey:event:1",
        }
    }
    narrative = json.loads(_response())
    fixture = SavedCoachResponseFixture.model_validate(
        {
            "responses": {
                key: {
                    "scenario_id": "casey-scenario",
                    "persona_id": "casey",
                    "week_start": digest.week_start,
                    "week_end": digest.week_end,
                    "narrative": narrative,
                    "generation": {
                        "model_contract": {
                            "provider": "openai",
                            "model": "gpt-5.6-luna",
                            "reasoning_effort": "none",
                        },
                        "service_tier": "default",
                        "prompt_name": "weekly_digest_coach",
                        "prompt_version": "4.3",
                        "prompt_sha256": hashlib.sha256(
                            b"original provider prompt"
                        ).hexdigest(),
                        "prompt": "original provider prompt",
                        "raw_output": _response(),
                        "response_sha256": runner._hash(narrative),
                        "attempt_count": 1,
                        "diagnostic_paths": ["original.json"],
                        "call_metrics": [_metric().model_dump(mode="json")],
                        "weekly_drift_input_sha256": cases[key]["input_sha256"],
                        "generated_response_path": "original_response.json",
                        "source_bundle_path": "bundle.json",
                        "source_bundle_content_sha256": "a" * 64,
                        "weekly_digest_event_id": "casey:event:1",
                    },
                }
            }
        }
    )
    if two_cases:
        second_key = "casey2-scenario::2025-01-01"
        second_digest = _digest("casey2")
        cases[second_key] = {
            **cases[key],
            "digest": second_digest.model_dump(mode="json"),
            "scenario_id": "casey2-scenario",
            "input_sha256": _weekly_drift_input_sha256(second_digest),
        }
        response = fixture.responses[key].model_copy(deep=True)
        response.scenario_id = "casey2-scenario"
        response.persona_id = "casey2"
        assert response.generation is not None
        response.generation.weekly_drift_input_sha256 = cases[second_key][
            "input_sha256"
        ]
        fixture.responses[second_key] = response
    runner._write(tmp_path / COACH_RESPONSES_PATH, fixture.model_dump(mode="json"))
    monkeypatch.setattr(
        runner, "collect_cases", lambda root: json.loads(json.dumps(cases))
    )
    for relative in (
        "scripts/coach/refresh_scenario_coach.py",
        "scripts/coach/complete_scenario_coach.py",
        "src/coach/weekly_digest.py",
        "src/coach/llm_client.py",
        "src/coach/schemas.py",
        "src/demo/scenarios.py",
        "prompts/weekly_digest_coach.yaml",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen source")
    output = tmp_path / "run"
    return key, output, runner.prepare(tmp_path, output)


def _metric(cost: float | None = 0.001) -> LLMCallMetrics:
    return LLMCallMetrics(
        provider="openai",
        model="gpt-5.6-luna",
        reasoning_effort="none",
        service_tier="default",
        status="completed",
        latency_seconds=0.1,
        input_tokens=100,
        output_tokens=100,
        total_tokens=200,
        calculated_cost_usd=cost,
    )


def _complete(metrics, responses, *, cost=0.001):
    async def complete(prompt, response_format, instructions=None):
        metrics.append(_metric(cost))
        return responses.pop(0)

    return complete


def test_refresh_preserves_original_receipts_and_resumes_without_calls(
    tmp_path: Path,
    monkeypatch,
):
    key, output, plan = _setup(tmp_path, monkeypatch)
    original = (tmp_path / COACH_RESPONSES_PATH).read_bytes()
    metrics = []
    asyncio.run(
        runner.generate(
            tmp_path,
            output,
            plan,
            llm_complete=_complete(metrics, [_response()]),
            metrics=metrics,
        )
    )
    assert (tmp_path / COACH_RESPONSES_PATH).read_bytes() == original
    assert (output / "original_coach_digest_responses.json").read_bytes() == original
    state = json.loads(runner._state_path(output, key).read_bytes())
    request = json.loads((tmp_path / state["attempts"][0]["request_path"]).read_bytes())
    assert request["model"] == "gpt-5.6-luna"
    assert request["reasoning"] == {"effort": "none"}
    assert state["attempts"][0]["request_sha256"] == runner._hash(request)
    assert state["response"]["generation"]["prompt_version"] == "4.4"
    runner.apply(tmp_path, output, plan)
    refreshed = load_saved_coach_responses(tmp_path).responses[key]
    assert refreshed.generation.prompt != "original provider prompt"
    asyncio.run(
        runner.generate(
            tmp_path, output, plan, llm_complete=_complete(metrics, []), metrics=metrics
        )
    )
    runner.apply(tmp_path, output, plan)
    assert len(metrics) == 1
    assert runner.write_report(output, plan)["usage"]["calculated_cost_usd"] == 0.001


def test_refresh_retries_validations_and_keeps_every_receipt(
    tmp_path: Path, monkeypatch
):
    key, output, plan = _setup(tmp_path, monkeypatch)
    invalid = json.loads(_response())
    invalid["weekly_mirror"] = "This week you helped a friend."
    metrics = []
    asyncio.run(
        runner.generate(
            tmp_path,
            output,
            plan,
            llm_complete=_complete(metrics, [json.dumps(invalid), _response()]),
            metrics=metrics,
        )
    )
    state = json.loads(runner._state_path(output, key).read_bytes())
    assert len(state["attempts"]) == 2
    assert state["response"]["generation"]["attempt_count"] == 2
    request = json.loads((tmp_path / state["attempts"][1]["request_path"]).read_bytes())
    assert "conversational_voice" in request["instructions"]
    assert len(list((output / "diagnostics").glob("*.json"))) == 2


def test_incomplete_refresh_cannot_replace_active_fixture(tmp_path: Path, monkeypatch):
    _, output, plan = _setup(tmp_path, monkeypatch)
    original = (tmp_path / COACH_RESPONSES_PATH).read_bytes()
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.apply(tmp_path, output, plan)
    assert (tmp_path / COACH_RESPONSES_PATH).read_bytes() == original


def test_interrupted_call_is_reserved_and_not_repeated(tmp_path: Path, monkeypatch):
    key, output, plan = _setup(tmp_path, monkeypatch)
    calls = 0

    async def crash(prompt, response_format, instructions=None):
        nonlocal calls
        calls += 1
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(runner.generate(tmp_path, output, plan, llm_complete=crash))
    state = json.loads(runner._state_path(output, key).read_bytes())
    assert state["attempts"][0]["reserved_usd"] == str(runner.PER_REQUEST_RESERVE_USD)
    with pytest.raises(RuntimeError, match="Unresolved interrupted"):
        asyncio.run(runner.generate(tmp_path, output, plan, llm_complete=crash))
    assert calls == 1


@pytest.mark.parametrize("cost", [None, 0.1])
def test_unknown_or_excessive_usage_stops_before_more_calls(
    tmp_path: Path,
    monkeypatch,
    cost,
):
    _, output, plan = _setup(tmp_path, monkeypatch)
    metrics = []
    with pytest.raises(RuntimeError, match="Provider"):
        asyncio.run(
            runner.generate(
                tmp_path,
                output,
                plan,
                llm_complete=_complete(metrics, [_response()], cost=cost),
                metrics=metrics,
            )
        )
    assert len(metrics) == 1
    assert len(list((output / "diagnostics").glob("*.json"))) == 1


def test_policy_changes_or_concurrent_saved_edits_are_rejected(
    tmp_path: Path, monkeypatch
):
    _, output, _ = _setup(tmp_path, monkeypatch)
    source = tmp_path / "src/coach/weekly_digest.py"
    source.write_text("changed source")
    with pytest.raises(ValueError, match="Frozen generation"):
        runner.prepare(tmp_path, output)
    source.write_text("frozen source")
    path = tmp_path / COACH_RESPONSES_PATH
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Active saved responses changed"):
        runner.prepare(tmp_path, output)


def test_request_reservation_rejects_excessive_input_without_provider_call():
    with pytest.raises(ValueError, match="reserved USD ceiling"):
        runner._request_bound({"input": "a" * 70_000})


def test_terminal_failure_stays_bounded_across_resume(tmp_path: Path, monkeypatch):
    _, output, plan = _setup(tmp_path, monkeypatch)
    original = (tmp_path / COACH_RESPONSES_PATH).read_bytes()
    invalid = json.loads(_response())
    invalid["weekly_mirror"] = "This week you helped a friend."
    metrics = []
    for outputs in (
        [json.dumps(invalid)] * runner.MAXIMUM_ATTEMPTS,
        [],
    ):
        with pytest.raises(RuntimeError, match="Terminal Coach failure"):
            asyncio.run(
                runner.generate(
                    tmp_path,
                    output,
                    plan,
                    llm_complete=_complete(metrics, outputs),
                    metrics=metrics,
                )
            )
    assert len(metrics) == runner.MAXIMUM_ATTEMPTS
    assert (tmp_path / COACH_RESPONSES_PATH).read_bytes() == original
    assert runner.write_report(output, plan)["accepted"] == 0


def test_editorial_repair_retains_unaffected_responses_and_both_runs(
    tmp_path: Path,
    monkeypatch,
):
    key, prior, prior_plan = _setup(tmp_path, monkeypatch, two_cases=True)
    original = (tmp_path / COACH_RESPONSES_PATH).read_bytes()
    metrics = []
    asyncio.run(
        runner.generate(
            tmp_path,
            prior,
            prior_plan,
            llm_complete=_complete(metrics, [_response(), _response()]),
            metrics=metrics,
        )
    )
    prior_files = {
        path.relative_to(prior): path.read_bytes()
        for path in prior.rglob("*")
        if path.is_file()
    }
    repair_output = tmp_path / "repair"
    requirement = "Use complete quotations; do not invent omitted wording."
    requirements = {key: [requirement]}
    plan = runner.prepare(
        tmp_path,
        repair_output,
        prior_run=prior,
        repair_requirements=requirements,
    )
    second_key = "casey2-scenario::2025-01-01"
    prior_second = json.loads(runner._state_path(prior, second_key).read_bytes())[
        "response"
    ]
    assert plan["retained_responses"] == {second_key: prior_second}
    assert plan["cases"] == prior_plan["cases"]
    assert plan["policy"]["maximum_reserved_usd"] == "0.072"
    invalid = json.loads(_response())
    invalid["weekly_mirror"] = "This week you helped a friend."
    new_metrics = []
    asyncio.run(
        runner.generate(
            tmp_path,
            repair_output,
            plan,
            llm_complete=_complete(new_metrics, [json.dumps(invalid), _response()]),
            metrics=new_metrics,
        )
    )
    assert len(new_metrics) == 2
    assert (tmp_path / COACH_RESPONSES_PATH).read_bytes() == original
    assert prior_files == {
        path.relative_to(prior): path.read_bytes()
        for path in prior.rglob("*")
        if path.is_file()
    }
    for path in (repair_output / "requests").glob("*.json"):
        request = json.loads(path.read_bytes())
        assert requirement in request["instructions"]
        assert "A prior response needs revision" in request["instructions"]
    runner.apply(tmp_path, repair_output, plan)
    saved = load_saved_coach_responses(tmp_path)
    assert saved.responses[second_key].model_dump(mode="json") == prior_second
    assert requirement in saved.responses[key].generation.prompt
    assert runner.write_report(repair_output, plan)["retained"] == 1


def test_editorial_requirements_and_prior_receipts_are_frozen(
    tmp_path: Path, monkeypatch
):
    key, prior, prior_plan = _setup(tmp_path, monkeypatch)
    metrics = []
    asyncio.run(
        runner.generate(
            tmp_path,
            prior,
            prior_plan,
            llm_complete=_complete(metrics, [_response()]),
            metrics=metrics,
        )
    )
    output = tmp_path / "repair"
    requirements = {key: ["Keep quoted text complete."]}
    runner.prepare(tmp_path, output, prior_run=prior, repair_requirements=requirements)
    with pytest.raises(ValueError, match="Frozen generation"):
        runner.prepare(
            tmp_path,
            output,
            prior_run=prior,
            repair_requirements={key: ["Different requirement."]},
        )
    prior_state = runner._state_path(prior, key)
    prior_state.write_bytes(prior_state.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Frozen generation"):
        runner.prepare(
            tmp_path, output, prior_run=prior, repair_requirements=requirements
        )


def test_editorial_repairs_require_valid_case_keys_and_prior_run(
    tmp_path: Path, monkeypatch
):
    key, prior, _ = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="prior accepted run"):
        runner.prepare(
            tmp_path, tmp_path / "repair", repair_requirements={key: ["Fix"]}
        )
    with pytest.raises(ValueError, match="current case keys"):
        runner.prepare(
            tmp_path,
            tmp_path / "repair",
            prior_run=prior,
            repair_requirements={"unknown": ["Fix"]},
        )
    with pytest.raises(ValueError, match="differ from its prior"):
        runner.prepare(
            tmp_path, prior, prior_run=prior, repair_requirements={key: ["Fix"]}
        )
