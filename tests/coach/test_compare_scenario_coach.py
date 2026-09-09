"""Paid comparison generation is bounded, resumable, and tied to exact receipts."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scripts.coach import compare_scenario_coach as runner
from src.coach.demo_comparison import (
    COMPARISONS_PATH,
    SavedCoachComparisonFixture,
    build_north_star_context,
    hash_json,
    hash_text,
    render_demo_comparison_prompt,
)
from tests.coach.test_demo_comparison import _pair, _record
from tests.coach.test_generate_approved_judge_sample import _digest
from tests.coach.test_refresh_scenario_coach import _complete, _response


def _setup(tmp_path: Path, monkeypatch):
    digest, record = _digest("casey"), _record()
    context = build_north_star_context(record)
    key = "casey-scenario::2025-01-01"
    cases = {
        key: {
            "scenario_id": "casey-scenario",
            "digest": digest.model_dump(mode="json"),
            "input_sha256": hash_json(
                digest.model_dump(
                    mode="json", exclude={"coach_narrative", "validation"}
                )
            ),
            "north_star_record": record.model_dump(mode="json"),
            "north_star_context": context.model_dump(mode="json", exclude_none=True),
            "base_prompts": {
                "without_north_star": render_demo_comparison_prompt(digest, None),
                "with_north_star": render_demo_comparison_prompt(digest, context),
            },
        }
    }
    monkeypatch.setattr(runner, "collect_comparison_cases", lambda root: cases)
    for name in runner.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen source")
    output = tmp_path / "run"
    return key, output, runner.prepare(tmp_path, output)


def _generate(tmp_path, output, plan, metrics, responses, *, cost=0.001):
    asyncio.run(
        runner.generate(
            tmp_path,
            output,
            plan,
            llm_complete=_complete(metrics, responses, cost=cost),
            metrics=metrics,
        )
    )


def test_complete_pairs_preserve_original_fixture_and_resume_without_calls(
    tmp_path: Path,
    monkeypatch,
):
    key, output, plan = _setup(tmp_path, monkeypatch)
    original = (tmp_path / "src/demo/coach_digest_responses.json").read_bytes()
    metrics = []
    _generate(tmp_path, output, plan, metrics, [_response(), _response()])
    fixture = runner.apply(tmp_path, output, plan)
    assert set(fixture.comparisons) == {key}
    assert (tmp_path / "src/demo/coach_digest_responses.json").read_bytes() == original
    requests = [
        json.loads(p.read_bytes()) for p in (output / "requests").glob("*.json")
    ]
    assert len(requests) == 2
    assert requests[0]["instructions"] == requests[1]["instructions"]
    payloads = [json.loads(row["input"]) for row in requests]
    assert {payload.pop("north_star_context") is None for payload in payloads} == {
        True,
        False,
    }
    assert payloads[0] == payloads[1]
    _generate(tmp_path, output, plan, metrics, [])
    assert len(metrics) == 2
    assert runner.write_report(output, plan)["accepted_arms"] == 2


def test_validation_retry_keeps_initial_and_accepted_prompts(tmp_path, monkeypatch):
    key, output, plan = _setup(tmp_path, monkeypatch)
    invalid = json.loads(_response())
    invalid["weekly_mirror"] = "This week you helped someone."
    metrics = []
    _generate(
        tmp_path, output, plan, metrics, [json.dumps(invalid), _response(), _response()]
    )
    pair = runner.apply(tmp_path, output, plan).comparisons[key]
    assert len(pair.without_north_star.call_metrics) == 2
    assert pair.without_north_star.repair_requirements
    assert pair.without_north_star.base_prompt != pair.without_north_star.prompt
    assert "weekly_mirror_verbatim" in pair.without_north_star.prompt
    assert len(pair.with_north_star.call_metrics) == 1


def test_interrupted_unknown_call_is_never_repeated(tmp_path, monkeypatch):
    key, output, plan = _setup(tmp_path, monkeypatch)
    calls = []

    async def interrupted(prompt, response_format, instructions=None):
        calls.append(prompt)
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(runner.generate(tmp_path, output, plan, llm_complete=interrupted))
    with pytest.raises(RuntimeError, match="Unresolved interrupted"):
        _generate(tmp_path, output, plan, [], [_response()])
    assert len(calls) == 1
    assert (
        len(
            json.loads(
                runner._state_path(output, key, "without_north_star").read_bytes()
            )["attempts"]
        )
        == 1
    )


@pytest.mark.parametrize("cost", [None, 0.019])
def test_unknown_or_excess_usage_stops_before_second_call(tmp_path, monkeypatch, cost):
    _, output, plan = _setup(tmp_path, monkeypatch)
    metrics = []
    with pytest.raises(RuntimeError, match="cost is unknown|exceeded"):
        _generate(
            tmp_path, output, plan, metrics, [_response(), _response()], cost=cost
        )
    assert len(metrics) == 1
    assert not (tmp_path / COMPARISONS_PATH).exists()


def test_maximum_attempts_prevent_unbounded_retry(tmp_path, monkeypatch):
    _, output, plan = _setup(tmp_path, monkeypatch)
    metrics = []
    with pytest.raises(RuntimeError, match="Terminal"):
        _generate(tmp_path, output, plan, metrics, ["bad json"] * 10)
    assert len(metrics) == runner.MAXIMUM_ATTEMPTS


def test_incomplete_pairs_cannot_install_and_sources_are_frozen(tmp_path, monkeypatch):
    _, output, plan = _setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.apply(tmp_path, output, plan)
    assert not (tmp_path / COMPARISONS_PATH).exists()
    (tmp_path / runner.SOURCE_FILES[0]).write_text("changed")
    with pytest.raises(ValueError, match="Frozen"):
        runner.prepare(tmp_path, output)


def test_oversized_request_fails_before_checkpoint_or_paid_call(tmp_path, monkeypatch):
    _, output, plan = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner,
        "_request_bound",
        lambda request: (_ for _ in ()).throw(
            ValueError("Request exceeds its reserved USD ceiling")
        ),
    )
    metrics = []
    with pytest.raises(ValueError, match="reserved"):
        _generate(tmp_path, output, plan, metrics, [_response()])
    assert metrics == []
    assert not list((output / "cases").glob("*.json"))


def test_staged_response_cannot_replace_accepted_diagnostic(tmp_path, monkeypatch):
    key, output, plan = _setup(tmp_path, monkeypatch)
    _generate(tmp_path, output, plan, [], [_response(), _response()])
    path = runner._state_path(output, key, "with_north_star")
    state = json.loads(path.read_bytes())
    raw = state["response"]["raw_output"] + "\n"
    state["response"].update(raw_output=raw, raw_output_sha256=hash_text(raw))
    runner._write(path, state)
    with pytest.raises(ValueError, match="differs from accepted"):
        runner.apply(tmp_path, output, plan)


def test_diagnostic_prompt_must_match_actual_request(tmp_path, monkeypatch):
    key, output, plan = _setup(tmp_path, monkeypatch)
    _generate(tmp_path, output, plan, [], [_response(), _response()])
    state = json.loads(runner._state_path(output, key, "with_north_star").read_bytes())
    diagnostic_path = tmp_path / state["attempts"][0]["diagnostic_path"]
    diagnostic = json.loads(diagnostic_path.read_bytes())
    diagnostic["prompt"] += "forged instruction"
    diagnostic["prompt_sha256"] = hash_text(diagnostic["prompt"])
    runner._write(diagnostic_path, diagnostic)
    with pytest.raises(ValueError, match="differs from its provider request"):
        runner.apply(tmp_path, output, plan)


def test_fixture_mapping_key_must_match_pair_identity():
    with pytest.raises(ValueError, match="fixture key"):
        SavedCoachComparisonFixture(comparisons={"wrong-scenario::2025-01-01": _pair()})


def test_repair_run_retains_accepted_arms_and_all_original_failed_receipts(
    tmp_path,
    monkeypatch,
):
    key, output, plan = _setup(tmp_path, monkeypatch)
    metrics = []
    invalid = json.loads(_response())
    invalid["weekly_mirror"] = 'You "sat with Sam for an hour" and gave him time.'
    with pytest.raises(RuntimeError, match="Terminal"):
        _generate(
            tmp_path,
            output,
            plan,
            metrics,
            [_response(), *([json.dumps(invalid)] * runner.MAXIMUM_ATTEMPTS)],
        )
    original_plan = (output / "plan.json").read_bytes()
    repair_output = tmp_path / "repair"
    repair_plan = runner.prepare(tmp_path, repair_output, prior_run=output)
    assert len(repair_plan["prior_run"]["retained_arms"]) == 1
    assert len(repair_plan["prior_run"]["attempts"]) == 5
    repair_metrics = []
    _generate(tmp_path, repair_output, repair_plan, repair_metrics, [_response()])
    fixture = runner.apply(tmp_path, repair_output, repair_plan)
    pair = fixture.comparisons[key]
    assert len(pair.without_north_star.call_metrics) == 1
    assert len(pair.with_north_star.call_metrics) == 5
    assert len(pair.with_north_star.diagnostic_paths) == 5
    assert "does NOT satisfy" in pair.with_north_star.prompt
    assert len(repair_metrics) == 1
    assert (output / "plan.json").read_bytes() == original_plan
    summary = runner.write_report(repair_output, repair_plan)
    assert summary["usage"]["n_calls"] == 6
    assert summary["usage"]["calculated_cost_usd"] == pytest.approx(0.006)
    assert summary["reserved_usd"] == "0.108"
    _generate(tmp_path, repair_output, repair_plan, repair_metrics, [])
    assert len(repair_metrics) == 1


def test_repair_run_refuses_prior_unknown_attempts(tmp_path, monkeypatch):
    key, output, plan = _setup(tmp_path, monkeypatch)
    case = plan["cases"][key]
    runner._write(
        runner._state_path(output, key, "without_north_star"),
        {
            "input_sha256": hash_json(case),
            "response": None,
            "attempts": [{"diagnostic_path": "missing.json"}],
        },
    )
    with pytest.raises(RuntimeError, match="Unresolved interrupted"):
        runner.prepare(tmp_path, tmp_path / "repair", prior_run=output)


def test_second_bounded_repair_deduplicates_entire_prior_receipt_history(
    tmp_path, monkeypatch
):
    key, output, plan = _setup(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="Terminal"):
        _generate(tmp_path, output, plan, [], [_response(), *(["bad"] * 4)])
    first = tmp_path / "repair1"
    first_plan = runner.prepare(tmp_path, first, prior_run=output)
    with pytest.raises(RuntimeError, match="Terminal"):
        _generate(tmp_path, first, first_plan, [], ["bad"] * 4)
    second = tmp_path / "repair2"
    second_plan = runner.prepare(tmp_path, second, prior_run=first)
    assert second_plan["prior_run"]["depth"] == 2
    assert len(second_plan["prior_run"]["attempts"]) == 9
    _generate(tmp_path, second, second_plan, [], [_response()])
    pair = runner.apply(tmp_path, second, second_plan).comparisons[key]
    assert len(pair.with_north_star.call_metrics) == 9
    assert len(set(pair.with_north_star.diagnostic_paths)) == 9
    assert runner.write_report(second, second_plan)["usage"]["n_calls"] == 10
    with pytest.raises(ValueError, match="limited to one"):
        runner.prepare(tmp_path, tmp_path / "repair3", prior_run=second)
    third = tmp_path / "editorial3"
    third_plan = runner.prepare(
        tmp_path,
        third,
        prior_run=second,
        editorial_repairs={
            f"{key}::with_north_star": {
                "response_sha256": pair.with_north_star.response_sha256,
                "reason": "Final review found one unsupported causal link.",
                "requirements": [
                    "Do not infer a causal link between separate dated passages."
                ],
            },
        },
    )
    assert third_plan["prior_run"]["depth"] == 3
    assert len(third_plan["prior_run"]["retained_arms"]) == 1
    revised = json.loads(_response())
    revised["tension_explanation"] = revised["tension_explanation"].replace(
        "two small", "everyday"
    )
    _generate(tmp_path, third, third_plan, [], [json.dumps(revised)])
    runner.apply(tmp_path, third, third_plan)
    with pytest.raises(ValueError, match="At most three"):
        runner.prepare(tmp_path, tmp_path / "repair4", prior_run=third)


def test_editorial_repair_binds_rejection_and_preserves_original_accepted_receipt(
    tmp_path, monkeypatch
):
    key, output, plan = _setup(tmp_path, monkeypatch)
    _generate(tmp_path, output, plan, [], [_response(), _response()])
    arm_key = f"{key}::with_north_star"
    directive = {
        arm_key: {
            "response_sha256": _pair().with_north_star.response_sha256,
            "reason": "AI editorial review identified an attribution problem.",
            "requirements": [
                "Keep the described actor and action faithful to their source."
            ],
        }
    }
    repair = tmp_path / "editorial"
    repair_plan = runner.prepare(
        tmp_path, repair, prior_run=output, editorial_repairs=directive
    )
    assert len(repair_plan["prior_run"]["retained_arms"]) == 1
    revised = json.loads(_response())
    revised["tension_explanation"] = revised["tension_explanation"].replace(
        "two small", "everyday"
    )
    _generate(tmp_path, repair, repair_plan, [], [json.dumps(revised)])
    pair = runner.apply(tmp_path, repair, repair_plan).comparisons[key]
    assert len(pair.with_north_star.call_metrics) == 2
    assert pair.with_north_star.response_sha256 != directive[arm_key]["response_sha256"]
    assert pair.with_north_star.diagnostic_paths[0].startswith("run/")
    assert directive[arm_key]["requirements"][0] in pair.with_north_star.prompt
    assert runner.write_report(repair, repair_plan)["usage"]["n_calls"] == 3
    bad = {arm_key: {**directive[arm_key], "response_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="does not match"):
        runner.prepare(
            tmp_path, tmp_path / "bad", prior_run=output, editorial_repairs=bad
        )


def test_editorial_rejected_response_cannot_be_reaccepted_unchanged(
    tmp_path, monkeypatch
):
    key, output, plan = _setup(tmp_path, monkeypatch)
    _generate(tmp_path, output, plan, [], [_response(), _response()])
    repair = tmp_path / "editorial"
    repair_plan = runner.prepare(
        tmp_path,
        repair,
        prior_run=output,
        editorial_repairs={
            f"{key}::with_north_star": {
                "response_sha256": _pair().with_north_star.response_sha256,
                "reason": "Source-attribution repair required.",
                "requirements": [
                    "Keep actors and attributed actions faithful to the source."
                ],
            },
        },
    )
    with pytest.raises(RuntimeError, match="Terminal"):
        _generate(tmp_path, repair, repair_plan, [], [_response()] * 4)
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.apply(tmp_path, repair, repair_plan)
