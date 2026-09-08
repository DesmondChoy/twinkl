"""Missing-week generation preserves paid work and source-bound saved responses."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scripts.coach import complete_scenario_coach as runner
from src.coach.schemas import LLMCallMetrics
from src.demo.scenarios import (
    COACH_RESPONSES_PATH,
    SavedCoachResponseFixture,
    _weekly_drift_input_sha256,
    load_saved_coach_responses,
)
from tests.coach.test_generate_approved_judge_sample import _digest, _valid_response


def _setup(tmp_path: Path):
    digest = _digest("casey")
    key = "casey-scenario::2025-01-01"
    output = tmp_path / "run"
    runner._write(
        tmp_path / COACH_RESPONSES_PATH,
        SavedCoachResponseFixture(responses={}).model_dump(mode="json"),
    )
    plan = {
        "missing_keys": [key],
        "policy": {"prompt": {"name": "weekly_digest_coach", "version": "4.2"}},
        "cases": {
            key: {
                "digest": digest.model_dump(mode="json"),
                "input_sha256": _weekly_drift_input_sha256(digest),
                "scenario_id": "casey-scenario",
                "source_bundle_path": "bundle.json",
                "source_bundle_content_sha256": "a" * 64,
                "weekly_digest_event_id": "casey:event:1",
            }
        },
    }
    return key, output, plan


def _completion(metrics, responses):
    async def complete(prompt, response_format, instructions=None):
        metrics.append(
            LLMCallMetrics(
                provider="openai",
                model="gpt-5.6-luna",
                reasoning_effort="none",
                status="completed",
                latency_seconds=0.1,
            )
        )
        return responses.pop(0)

    return complete


def test_complete_missing_response_and_resume_without_calls(tmp_path: Path):
    key, output, plan = _setup(tmp_path)
    metrics = []
    asyncio.run(
        runner.generate(
            tmp_path,
            output,
            plan,
            llm_complete=_completion(metrics, [_valid_response()]),
            metrics=metrics,
        )
    )
    response = load_saved_coach_responses(tmp_path).responses[key]
    assert response.generation.attempt_count == 1
    saved_bytes = (tmp_path / COACH_RESPONSES_PATH).read_bytes()
    asyncio.run(
        runner.generate(
            tmp_path,
            output,
            plan,
            llm_complete=_completion(metrics, []),
            metrics=metrics,
        )
    )
    assert len(metrics) == 1
    assert (tmp_path / COACH_RESPONSES_PATH).read_bytes() == saved_bytes


def test_validation_retry_is_bounded_and_keeps_both_receipts(tmp_path: Path):
    key, output, plan = _setup(tmp_path)
    metrics = []
    invalid = json.loads(_valid_response())
    invalid["weekly_mirror"] = "You had a week."
    asyncio.run(
        runner.generate(
            tmp_path,
            output,
            plan,
            llm_complete=_completion(metrics, [json.dumps(invalid), _valid_response()]),
            metrics=metrics,
        )
    )
    response = load_saved_coach_responses(tmp_path).responses[key]
    assert response.generation.attempt_count == 2
    assert len(response.generation.diagnostic_paths) == 2
    assert len(metrics) == 2


def test_terminal_failure_does_not_gain_attempts_on_resume(tmp_path: Path):
    _, output, plan = _setup(tmp_path)
    metrics = []
    invalid = json.loads(_valid_response())
    invalid["weekly_mirror"] = "You had a week."
    for responses in ([json.dumps(invalid), json.dumps(invalid)], []):
        with pytest.raises(RuntimeError, match="Terminal Coach failure"):
            asyncio.run(
                runner.generate(
                    tmp_path,
                    output,
                    plan,
                    llm_complete=_completion(metrics, responses),
                    metrics=metrics,
                )
            )
    assert len(metrics) == 2
    assert not load_saved_coach_responses(tmp_path).responses


def test_interrupted_attempt_without_receipt_is_not_repeated(tmp_path: Path):
    key, output, plan = _setup(tmp_path)
    runner._write(
        output / "cases" / f"{key.replace('::', '_')}.json",
        {
            "input_sha256": plan["cases"][key]["input_sha256"],
            "attempts": [{"diagnostic_path": "missing.json"}],
        },
    )
    metrics = []
    with pytest.raises(RuntimeError, match="interrupted attempt"):
        asyncio.run(
            runner.generate(
                tmp_path,
                output,
                plan,
                llm_complete=_completion(metrics, []),
                metrics=metrics,
            )
        )
    assert not metrics


def test_merge_refuses_to_overwrite_a_different_saved_response(tmp_path: Path):
    key, output, plan = _setup(tmp_path)
    metrics = []
    asyncio.run(
        runner.generate(
            tmp_path,
            output,
            plan,
            llm_complete=_completion(metrics, [_valid_response()]),
            metrics=metrics,
        )
    )
    response = load_saved_coach_responses(tmp_path).responses[key]
    changed = response.model_copy(update={"week_end": "2025-01-08"})
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        runner._merge(tmp_path, key, changed)
    assert load_saved_coach_responses(tmp_path).responses[key] == response
