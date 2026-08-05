"""Tests for the Coach Digest response AI evaluator."""

from __future__ import annotations

import asyncio
import json

import pytest

from prompts import get_prompt_metadata
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
        persona_name="Casey",
        week_start="2025-01-01",
        week_end="2025-01-07",
        response_mode="stable",
        mode_source="drift_detector",
        mode_rationale="No confirmed Drift this week.",
        n_entries=3,
        overall_mean=0.4,
        core_values=["benevolence", "self_direction"],
        goal_context="Make more time for family",
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

    # The evaluator receives the same factual contract as generation.
    assert "Preferred name: Casey" in prompt
    assert "Week: 2025-01-01 to 2025-01-07" in prompt
    assert "Selected Coach Digest policy: no_current_drift" in prompt
    assert "Being there for the people closest to me" in prompt
    assert 'User-confirmed current focus: "Make more time for family"' in prompt
    assert "Weekly Drift Detection did not find two consecutive" in prompt
    assert "2025-01-03 | supportive evidence" in prompt
    assert "internal Schwartz label(s): Benevolence" in prompt
    assert 'excerpt: "called my mom and helped a colleague debug"' in prompt
    assert "Primary tensions" not in prompt
    # Narrative fields.
    assert "A steady week" in prompt
    assert "Nothing pulled against" in prompt
    assert "showing up with intention" in prompt
    # Rubric dimensions.
    assert "tension_honesty" in prompt
    assert "non_prescriptive_tone" in prompt


def test_judge_prompt_declares_generation_facts_plus_narrative():
    metadata = get_prompt_metadata("coach_narrative_judge")

    assert metadata["version"] == "3.0"
    assert metadata["input_variables"] == [
        "persona_name",
        "week_window",
        "response_policy",
        "compass_context_lines",
        "drift_summary_lines",
        "state_comparison_lines",
        "evidence_lines",
        "weekly_mirror",
        "tension_explanation",
        "reflective_question",
    ]


@pytest.mark.parametrize(
    ("response_mode", "drift_states", "expected_policy", "expected_findings"),
    [
        (
            "active_drift",
            {"benevolence": "active_drift"},
            "drift_detected",
            ("Drift is active",),
        ),
        (
            "no_active_drift",
            {"benevolence": "no_active_drift"},
            "no_current_drift",
            ("No active Drift is confirmed",),
        ),
        (
            "insufficient_evidence",
            {"benevolence": "insufficient_evidence"},
            "more_reflection_needed",
            ("insufficient evidence",),
        ),
        (
            "active_drift",
            {
                "benevolence": "active_drift",
                "self_direction": "insufficient_evidence",
            },
            "drift_detected",
            ("Drift is active", "insufficient evidence"),
        ),
        (
            "stable",
            {},
            "no_current_drift",
            ("did not find two consecutive Journal Entries",),
        ),
    ],
)
def test_judge_prompt_preserves_delivery_state_facts(
    response_mode: str,
    drift_states: dict[str, str],
    expected_policy: str,
    expected_findings: tuple[str, ...],
):
    digest = _digest().model_copy(
        update={"response_mode": response_mode, "drift_states": drift_states}
    )

    prompt = render_judge_prompt(digest, _narrative())

    assert f"Selected Coach Digest policy: {expected_policy}" in prompt
    for finding in expected_findings:
        assert finding in prompt
    assert "A material policy mismatch must score 2 or lower" in " ".join(
        prompt.split()
    )


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


def test_render_markdown_labels_ai_evaluation():
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

    assert "Coach Digest Evals" in markdown
    assert "NOT human validation" in markdown
    assert "tension_honesty" in markdown
    assert report.to_dict()["eval"] == "coach_digest_evals"
    assert report.to_dict()["source"] == "ai_review"
