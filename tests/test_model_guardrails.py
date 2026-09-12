"""Structural guardrail cases; semantic quality remains a reviewed evaluation."""

import pytest

from src.model_guardrails import extract_source_quotations, is_single_question


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "You wrote \"called home\" and 'I stole money'.",
            ["called home", "I stole money"],
        ),
        (
            "You wrote “called home” and ‘I stole money’.",
            ["called home", "I stole money"],
        ),
        ("You wrote 'I didn't call home' afterward.", ["I didn't call home"]),
        ("You wrote ‘I didn’t call home’ afterward.", ["I didn’t call home"]),
        ("I didn't ask Karen's brother about her parents' house.", []),
        ("I didn’t ask Karen’s brother about her parents’ house.", []),
        (
            "You wrote 'called home', then 'made dinner'.",
            ["called home", "made dinner"],
        ),
    ],
)
def test_source_quotations_distinguish_pairs_from_apostrophes(text, expected):
    assert extract_source_quotations(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "If you could revisit that evening, what would matter most?",
        "Looking back, what stayed with you?",
        "When you think of that conversation, how does it feel?",
        "In that moment, what did you want to say?",
        "What stayed with you?",
        'When you typed "checking the logs," what felt most important '
        "to protect in that moment?",
        "When did you buy apples, pears and bananas?",
        'If you wrote "I will try," what were you hoping for?',
        'When you wrote "apples, pears and bananas", what did you mean?',
        "Looking back, what did you want from work, home and friendships?",
    ],
)
def test_question_form_allows_fronted_context(text):
    assert is_single_question(text)


@pytest.mark.parametrize(
    "text",
    [
        "Please quit your job, what do you think?",
        "Looking back, you should quit your job?",
        "If you revisit that evening, apologize to them?",
        'When you typed "checking the logs," apologize to them?',
        "What happened? What came next?",
    ],
)
def test_question_form_still_rejects_obvious_directive_forms(text):
    assert not is_single_question(text)
