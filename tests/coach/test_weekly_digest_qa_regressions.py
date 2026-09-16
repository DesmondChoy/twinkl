"""Regressions from the live onboarding QA, using fictional source writing."""

import hashlib
from pathlib import Path

import pytest

from src.coach.schemas import (
    CoachNarrative,
    CoreValueWeekComparison,
    DigestValidation,
    EvidenceSnippet,
    WeeklyDigest,
)
from src.coach.weekly_digest import (
    coach_validation_policy_for_prompt_version,
    validate_weekly_digest_narrative,
)


def _digest(source: str) -> WeeklyDigest:
    return WeeklyDigest(
        persona_id="qa-designer",
        week_start="2026-09-14",
        week_end="2026-09-20",
        response_mode="no_active_drift",
        mode_source="drift_detection",
        mode_rationale="No Active Drift was confirmed.",
        n_entries=1,
        overall_mean=0.0,
        top_tensions=[],
        top_strengths=[],
        dimensions=[],
        evidence=[EvidenceSnippet(
            date="2026-09-18", t_index=0, direction="context",
            dimensions=["self_direction"], excerpt=source,
        )],
    )


def _narrative(**overrides: str) -> CoachNarrative:
    return CoachNarrative(
        **{
            "weekly_mirror": 'You wrote, "I finished the drawing".',
            "tension_explanation": "You made time to try a technique you had chosen.",
            "reflective_question": "What mattered to you about choosing that exercise?",
            **overrides,
        }
    )


def _check(validation: DigestValidation, name: str) -> bool:
    return next(check.passed for check in validation.checks if check.name == name)


def test_source_supported_improvement_is_not_a_generated_transition_claim():
    source = (
        "I finished the drawing. I liked being able to decide what to learn "
        "and to see a concrete improvement in the finished work."
    )
    narrative = _narrative(weekly_mirror=(
        'You described the illustration exercise in your own words: "I liked '
        'being able to decide what to learn and to see a concrete improvement '
        'in the finished work."'
    ))

    validation = validate_weekly_digest_narrative(_digest(source), narrative)

    assert validation.all_passed
    assert _check(validation, "state_claims")


@pytest.mark.parametrize("opening,closing", [
    ('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’"),
])
def test_source_quotation_exemption_does_not_hide_generated_claims(opening, closing):
    digest = _digest("I finished the drawing. I didn't expect this improvement.")
    quoted = f"You wrote, {opening}I didn't expect this improvement.{closing}"
    narrative = _narrative(tension_explanation=quoted)

    assert validate_weekly_digest_narrative(digest, narrative).all_passed
    with_generated_claim = narrative.model_copy(update={
        "tension_explanation": quoted + " You are making progress.",
    })
    assert not _check(
        validate_weekly_digest_narrative(digest, with_generated_claim), "state_claims",
    )


def test_prior_source_improvement_does_not_establish_current_improvement():
    digest = _digest("I finished the drawing.")
    digest.state_comparisons = [CoreValueWeekComparison(
        core_value="self_direction", previous_week_start="2026-09-07",
        previous_week_end="2026-09-13", current_week_start="2026-09-14",
        current_week_end="2026-09-20", previous_state="no_active_drift",
        current_state="no_active_drift", change="unchanged",
        previous_evidence=[EvidenceSnippet(
            date="2026-09-12", t_index=0, direction="context",
            dimensions=["self_direction"],
            excerpt="I saw an improvement in the sketch.",
        )],
    )]
    narrative = _narrative(tension_explanation=(
        'Earlier, you wrote, "I saw an improvement in the sketch." '
        "That described a different drawing."
    ))

    assert validate_weekly_digest_narrative(digest, narrative).all_passed
    unsupported = narrative.model_copy(update={
        "tension_explanation": narrative.tension_explanation + " Things are better now."
    })
    assert not _check(
        validate_weekly_digest_narrative(digest, unsupported), "state_claims",
    )


@pytest.mark.parametrize("extra_field", ["weekly_mirror", "tension_explanation"])
def test_extra_generated_question_in_either_paragraph_is_rejected(extra_field):
    narrative = _narrative()
    narrative = narrative.model_copy(update={
        extra_field: getattr(narrative, extra_field) + " What drew you to that choice?"
    })

    validation = validate_weekly_digest_narrative(
        _digest("I finished the drawing."), narrative,
    )

    assert not validation.all_passed
    assert not _check(validation, "single_generated_question")
    assert _check(validation, "reflective_question_form")


@pytest.mark.parametrize("opening,closing", [
    ('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’"),
])
def test_embedded_source_question_does_not_count_as_a_generated_question(
    opening, closing,
):
    source = "I finished the drawing. What can I try next?"
    narrative = _narrative(reflective_question=(
        f"When you wrote {opening}What can I try next?{closing}, "
        "what possibility did you have in mind?"
    ))

    validation = validate_weekly_digest_narrative(_digest(source), narrative)

    assert validation.all_passed
    assert _check(validation, "reflective_question_form")


@pytest.mark.parametrize("field", ["weekly_mirror", "tension_explanation"])
def test_exact_source_question_is_allowed_in_a_narrative_paragraph(field):
    narrative = _narrative()
    narrative = narrative.model_copy(update={
        field: getattr(narrative, field) + ' You asked yourself, "What can I try next?"'
    })

    assert validate_weekly_digest_narrative(
        _digest("I finished the drawing. What can I try next?"), narrative,
    ).all_passed


def test_a_source_question_alone_does_not_replace_the_generated_invitation():
    validation = validate_weekly_digest_narrative(
        _digest("I finished the drawing. What can I try next?"),
        _narrative(reflective_question='"What can I try next?"'),
    )

    assert not _check(validation, "reflective_question_form")


@pytest.mark.parametrize("unsupported", [
    "Your choices show improvement.",
    'You wrote, "I saw an improvement".',
    'You wrote, "I saw improvement.',
])
def test_unsupported_or_unmatched_improvement_is_still_rejected(unsupported):
    validation = validate_weekly_digest_narrative(
        _digest("I finished the drawing."),
        _narrative(tension_explanation=unsupported),
    )

    assert not validation.all_passed
    assert not _check(validation, "state_claims")


def test_matching_source_words_without_paired_quotes_are_not_exempt():
    validation = validate_weekly_digest_narrative(
        _digest("I finished the drawing. I saw improvement."),
        _narrative(tension_explanation='You wrote, "I saw improvement.'),
    )

    assert not _check(validation, "state_claims")


@pytest.mark.parametrize("question", [
    'You asked, "What shall I try tomorrow?"',
    'You asked, "What can I try next?',
])
def test_invented_or_unmatched_quoted_question_is_not_exempt(question):
    validation = validate_weekly_digest_narrative(
        _digest("I finished the drawing. What can I try next?"),
        _narrative(tension_explanation=question),
    )

    assert not validation.all_passed
    assert not _check(validation, "single_generated_question")


def test_recorded_46_response_retains_its_original_state_and_question_checks():
    policy = coach_validation_policy_for_prompt_version("4.6")
    digest = _digest("I finished the drawing. I saw an improvement in the shading.")
    quoted_improvement = _narrative(tension_explanation=(
        'You wrote, "I saw an improvement in the shading."'
    ))
    extra_question = _narrative(tension_explanation=(
        "You made time to try a technique you had chosen. What drew you to it?"
    ))
    original_improvement = validate_weekly_digest_narrative(
        digest, quoted_improvement, validation_policy=policy,
    )
    original_questions = validate_weekly_digest_narrative(
        digest, extra_question, validation_policy=policy,
    )

    assert not _check(original_improvement, "state_claims")
    assert original_questions.all_passed
    assert _check(original_questions, "reflective_question_form")
    assert "all_quotes_grounded" in {check.name for check in original_questions.checks}


@pytest.mark.parametrize("policy,expected_minimum", [
    ("historical", 25), ("4.6", 0), ("current", 0),
])
def test_versioned_length_limits_preserve_original_receipt_details(
    policy, expected_minimum,
):
    digest = _digest("I finished the drawing.")
    narrative = _narrative(
        weekly_mirror='You "finished the drawing".',
        tension_explanation="You tried a technique.",
        reflective_question="What drew you to it?",
    )
    short = validate_weekly_digest_narrative(
        digest, narrative, validation_policy=policy,
    )
    assert short.word_count == 13
    assert short.length_passed == (expected_minimum == 0)
    length = next(check for check in short.checks if check.name == "length")
    assert length.details == (
        "Combined response length is 13 words (target 25-180)."
        if expected_minimum else
        "Combined response length is 13 words (maximum 180; no minimum)."
    )
    long = validate_weekly_digest_narrative(
        digest,
        narrative.model_copy(update={"tension_explanation": "detail " * 180}),
        validation_policy=policy,
    )
    assert long.word_count == 189
    assert not long.length_passed


@pytest.mark.parametrize("version,policy", [
    (None, "historical"), ("4.4", "historical"), ("4.5", "historical"),
    ("4.6", "4.6"), ("4.7", "current"),
])
def test_recorded_prompt_version_selects_its_original_validation(version, policy):
    assert coach_validation_policy_for_prompt_version(version) == policy


def test_unknown_recorded_version_cannot_silently_use_weaker_validation():
    with pytest.raises(ValueError, match="Unsupported Coach Digest prompt version"):
        coach_validation_policy_for_prompt_version("4.8")


def test_archived_46_prompt_is_byte_identical_to_the_pre_qa_prompt():
    archived = (
        Path(__file__).resolve().parents[2]
        / "prompts/versions/weekly_digest_coach_v4_6.yaml"
    )
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == (
        "0b6c5ee47462e5de6b74159db84393de8bc52a772dd706b37f91db4db96c7e25"
    )
