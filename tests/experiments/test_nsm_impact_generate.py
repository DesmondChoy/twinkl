import asyncio
import json
from collections import Counter

import pytest

from scripts.experiments import nsm_impact_generate as gen
from scripts.experiments import nsm_impact_pilot as pilot
from src.coach.schemas import LLMCallMetrics, WeeklyDigest

pytestmark = pytest.mark.skipif(
    not (gen.ROOT / gen.RECORD_PATH).exists(),
    reason="North Star Moment selection record is unavailable",
)


@pytest.fixture(scope="module")
def cases():
    record = json.loads((gen.ROOT / gen.RECORD_PATH).read_bytes())
    return gen.build_cases(record)


def _metric() -> LLMCallMetrics:
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
        calculated_cost_usd=0.001,
    )


def _valid_output(case: dict) -> str:
    """A narrative that passes Coach Digest validation for either arm."""
    excerpt = WeeklyDigest.model_validate(case["digest"]).evidence[0].excerpt
    quote = " ".join(excerpt.split()[:6]).strip(".,")
    return json.dumps(
        {
            "weekly_mirror": f'When you wrote "{quote}", you sounded relieved.',
            "tension_explanation": "That relief sat next to other things you "
            "care about, and it is not clear yet how they fit together.",
            "reflective_question": "What would you like to notice about next week?",
        }
    )


def test_build_cases_freezes_correct_final_cards(cases):
    assert len(cases) == gen.EXPECTED_WEEKS
    assert len({c["persona_id"] for c in cases.values()}) == 23
    assert Counter(c["mode"] for c in cases.values()) == {
        "encouragement": 58,
        "reminder": 7,
        "reflection": 7,
    }
    for case in cases.values():
        without, with_moment = (
            pilot.coach_input({"base_prompt": case["base_prompts"][arm]})
            for arm in gen.ARMS
        )
        # The arms differ only in the injected North Star Moment.
        assert without["north_star_context"] is None
        assert {**with_moment, "north_star_context": None} == without
        in_week = case["week_start"] <= case["north_star_context"]["date"]
        in_week &= case["north_star_context"]["date"] <= case["week_end"]
        assert bool(case["history_entries"]) is not in_week
        assert case["week_entries"]


def test_generate_retries_resumes_and_exports_judge_tasks(cases, tmp_path):
    key = next(iter(cases))
    plan = {"cases": {key: cases[key]}, "policy": {"maximum_reserved_usd": "0.072"}}
    output = tmp_path / "run"
    metrics: list[LLMCallMetrics] = []
    # The first "without" attempt is invalid JSON; the retry is accepted.
    responses = ["not json", _valid_output(cases[key]), _valid_output(cases[key])]

    async def complete(prompt, response_format, instructions=None):
        metrics.append(_metric())
        return responses.pop(0)

    asyncio.run(
        gen.generate(tmp_path, output, plan, llm_complete=complete, metrics=metrics)
    )
    assert not responses and len(metrics) == 3

    async def unexpected(prompt, response_format, instructions=None):
        raise AssertionError("A completed arm was requested again")

    asyncio.run(
        gen.generate(tmp_path, output, plan, llm_complete=unexpected, metrics=[])
    )

    pairs = gen.export(tmp_path, output, plan)
    assert [p["pair_id"] for p in pairs] == [key.replace("::", ":")]
    assert gen.write_summary(output, plan)["attempts"] == 3

    tasks = tmp_path / "tasks_out"
    manifest = pilot.prepare(tasks, pairs_path=output / "pairs.json")
    assert manifest["pairs"] == 1 and manifest["tasks"] == 2
    assert "scenario_sha256" not in manifest
    text = (tasks / "tasks" / "task-01.md").read_text()
    assert "north" not in text.lower() and "with_north_star" not in text


def test_export_refuses_incomplete_arms(cases, tmp_path):
    key = next(iter(cases))
    with pytest.raises(RuntimeError, match="incomplete"):
        gen.export(tmp_path, tmp_path / "run", {"cases": {key: cases[key]}})
