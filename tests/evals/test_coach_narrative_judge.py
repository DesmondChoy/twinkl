"""Tests for the Coach Digest response Tier-2 LLM-as-judge eval."""

from __future__ import annotations

import asyncio
import json

from src.coach.schemas import CoachNarrative, EvidenceSnippet, WeeklyDigest
from src.evals.coach_narrative_judge import (
    JudgeVerdict,
    aggregate_verdicts,
    judge_narrative,
    render_judge_prompt,
    render_markdown,
)


def _digest() -> WeeklyDigest:
    return WeeklyDigest(
        persona_id="deadbeef",
        week_start="2025-01-01",
        week_end="2025-01-07",
        response_mode="stable",
        mode_source="drift_detector",
        mode_rationale="No confirmed Drift this week.",
        n_entries=3,
        overall_mean=0.4,
        core_values=["benevolence", "self_direction"],
        drift_states={},
        top_tensions=[],
        top_strengths=["benevolence"],
        dimensions=[],
        evidence=[
            EvidenceSnippet(
                date="2025-01-03",
                t_index=1,
                direction="aligned",
                dimensions=["benevolence"],
                excerpt="called my mom and helped a colleague debug",
            )
        ],
    )


def _narrative() -> CoachNarrative:
    return CoachNarrative(
        weekly_mirror='A steady week, like when you "helped a colleague debug."',
        tension_explanation="Nothing pulled against what matters to you this week.",
        reflective_question="What let you keep showing up with intention?",
    )


def _stub_llm(payload: str | None):
    async def llm_complete(prompt: str, response_format: dict | None) -> str | None:
        return payload

    return llm_complete


def test_render_judge_prompt_includes_facts_and_narrative():
    prompt = render_judge_prompt(_digest(), _narrative())

    # Digest facts.
    assert "benevolence" in prompt
    assert "None clear this week" in prompt  # empty top_tensions fallback
    assert "helped a colleague debug" in prompt
    # Narrative fields.
    assert "A steady week" in prompt
    assert "Nothing pulled against" in prompt
    assert "showing up with intention" in prompt
    # Rubric dimensions.
    assert "tension_honesty" in prompt
    assert "non_prescriptive_tone" in prompt


def test_judge_narrative_parses_valid_verdict():
    payload = json.dumps(
        {
            "correctness": 5,
            "specificity": 4,
            "non_prescriptive_tone": 5,
            "tension_honesty": 5,
            "question_is_open_and_relevant": True,
            "justification": "Grounded in the debug quote; no invented tension.",
        }
    )

    verdict = asyncio.run(
        judge_narrative(_digest(), _narrative(), _stub_llm(payload))
    )

    assert verdict is not None
    assert verdict.correctness == 5
    assert verdict.needs_review is False


def test_judge_narrative_flags_low_score_for_review():
    payload = json.dumps(
        {
            "correctness": 2,
            "specificity": 4,
            "non_prescriptive_tone": 5,
            "tension_honesty": 5,
            "question_is_open_and_relevant": True,
            "justification": "Weekly mirror misreads the evidence.",
        }
    )

    verdict = asyncio.run(
        judge_narrative(_digest(), _narrative(), _stub_llm(payload))
    )

    assert verdict is not None
    assert verdict.needs_review is True


def test_judge_narrative_degrades_on_malformed_output():
    assert (
        asyncio.run(judge_narrative(_digest(), _narrative(), _stub_llm(None)))
        is None
    )
    assert (
        asyncio.run(
            judge_narrative(_digest(), _narrative(), _stub_llm("{not json"))
        )
        is None
    )
    # Valid JSON, invalid schema (score out of range).
    bad_schema = json.dumps(
        {
            "correctness": 9,
            "specificity": 4,
            "non_prescriptive_tone": 5,
            "tension_honesty": 5,
            "question_is_open_and_relevant": True,
            "justification": "out of range",
        }
    )
    assert (
        asyncio.run(
            judge_narrative(_digest(), _narrative(), _stub_llm(bad_schema))
        )
        is None
    )


def test_aggregate_verdicts_computes_means_and_flags():
    verdicts = [
        JudgeVerdict(
            correctness=5,
            specificity=5,
            non_prescriptive_tone=5,
            tension_honesty=5,
            question_is_open_and_relevant=True,
            justification="clean",
        ),
        JudgeVerdict(
            correctness=3,
            specificity=1,
            non_prescriptive_tone=4,
            tension_honesty=2,
            question_is_open_and_relevant=False,
            justification="weak",
        ),
        None,  # a failed/skipped verdict
    ]

    report = aggregate_verdicts(verdicts, judge_model="test-model")

    assert report.n_scored == 2
    assert report.n_failed == 1
    assert report.means["correctness"] == 4.0  # (5 + 3) / 2
    assert report.means["specificity"] == 3.0  # (5 + 1) / 2
    # Second verdict has specificity=1 and tension_honesty=2 (< 3) -> flagged.
    assert report.n_flagged == 1
    assert report.question_open_rate == 0.5


def test_render_markdown_labels_llm_as_judge():
    report = aggregate_verdicts(
        [
            JudgeVerdict(
                correctness=4,
                specificity=4,
                non_prescriptive_tone=4,
                tension_honesty=4,
                question_is_open_and_relevant=True,
                justification="ok",
            )
        ],
        judge_model="test-model",
    )

    markdown = render_markdown(report)

    assert "LLM-as-judge" in markdown
    assert "NOT human validation" in markdown
    assert "tension_honesty" in markdown
