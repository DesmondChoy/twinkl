"""SVBWS onboarding: item bank, scoring, and Profile transform for the demo app.

Ports the published Schwartz Values Best-Worst Survey design and scoring math
from frontend/onboarding/src/domain.ts (Lee, Soutar & Louviere, 2008) so the
Shiny demo app runs the same instrument as the React onboarding POC, without
depending on that separate deployment. Pure functions only; no Shiny
dependency, so the scoring logic is testable independently of the UI.

Any change to BWS_SETS, BWS_OBJECTS, or the scoring math must stay in lockstep
with domain.ts — see tests/demo_tool/test_onboarding_flow.py, which
cross-validates against domain.test.ts's own fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models.judge import SCHWARTZ_VALUE_ORDER

# ── BWS object bank (domain.ts BWS_OBJECT_ORDER / BWS_OBJECTS) ──────────────
#
# 11 objects: the ten Schwartz values, with Universalism split into its
# social and nature-related facets (kept distinct until the Profile
# transformation). Descriptor triplets are the published Lee, Soutar &
# Louviere wording, preserved verbatim.

BWS_OBJECT_ORDER: list[str] = [
    "power",
    "achievement",
    "hedonism",
    "stimulation",
    "self_direction",
    "universalism_nature",
    "benevolence",
    "tradition",
    "conformity",
    "security",
    "universalism_social",
]

# object key -> (Schwartz value key, descriptor triplet)
BWS_OBJECTS: dict[str, tuple[str, str]] = {
    "power": ("power", "Social power, authority, wealth"),
    "achievement": ("achievement", "Successful, capable, ambitious"),
    "hedonism": ("hedonism", "Pleasure, enjoying life, self-indulgent"),
    "stimulation": ("stimulation", "Daring, a varied life, an exciting life"),
    "self_direction": ("self_direction", "Creativity, curious, freedom"),
    "universalism_nature": (
        "universalism",
        "Protecting the environment, a world of beauty, unity with nature",
    ),
    "benevolence": ("benevolence", "Helpful, honest, forgiving"),
    "tradition": ("tradition", "Devout, accepting portion in life, humble"),
    "conformity": ("conformity", "Politeness, honouring parents & elders, obedient"),
    "security": ("security", "Clean, national & family security, social order"),
    "universalism_social": ("universalism", "Equality, world at peace, social justice"),
}

# Lee, Soutar & Louviere's 11x6 balanced incomplete block design: every
# object appears in exactly 6 of the 11 sets, and every pair of objects
# appears together in exactly 3 sets. 0-indexed (domain.ts is 1-indexed via
# setNumber; this port drops that field and uses list position instead).
BWS_SETS: list[list[str]] = [
    [
        "achievement",
        "universalism_nature",
        "benevolence",
        "tradition",
        "security",
        "universalism_social",
    ],
    [
        "power",
        "hedonism",
        "benevolence",
        "tradition",
        "conformity",
        "universalism_social",
    ],
    ["power", "achievement", "stimulation", "tradition", "conformity", "security"],
    [
        "achievement",
        "hedonism",
        "self_direction",
        "conformity",
        "security",
        "universalism_social",
    ],
    [
        "power",
        "hedonism",
        "stimulation",
        "universalism_nature",
        "security",
        "universalism_social",
    ],
    [
        "power",
        "achievement",
        "stimulation",
        "self_direction",
        "benevolence",
        "universalism_social",
    ],
    [
        "power",
        "achievement",
        "hedonism",
        "self_direction",
        "universalism_nature",
        "tradition",
    ],
    [
        "achievement",
        "hedonism",
        "stimulation",
        "universalism_nature",
        "benevolence",
        "conformity",
    ],
    [
        "hedonism",
        "stimulation",
        "self_direction",
        "benevolence",
        "tradition",
        "security",
    ],
    [
        "stimulation",
        "self_direction",
        "universalism_nature",
        "tradition",
        "conformity",
        "universalism_social",
    ],
    [
        "power",
        "self_direction",
        "universalism_nature",
        "benevolence",
        "conformity",
        "security",
    ],
]

# (key, prompt text) — unchanged from the React app's GOALS.
# First-person phrase per Schwartz value (domain.ts VALUES[key].phrase),
# shown only on the summary screen's Core Value cards — not the raw BWS
# descriptor triplets used during the assessment itself.
VALUE_PHRASES: dict[str, str] = {
    "self_direction": "Having the freedom to choose my own path",
    "stimulation": "Seeking new experiences and challenges",
    "hedonism": "Enjoying life and having fun",
    "achievement": "Making progress toward something meaningful",
    "power": "Having influence over how things go",
    "security": "Feeling calm and secure in my life",
    "conformity": "Being someone others can count on to do the right thing",
    "tradition": "Honoring the customs and practices I was raised with",
    "benevolence": "Being there for the people closest to me",
    "universalism": "Making the world a fairer, better place",
}

GOALS: list[tuple[str, str]] = [
    ("work_life_balance", "I'm stretched too thin between work and everything else"),
    ("life_transition", "I'm going through a career or life transition"),
    ("relationships", "I want to be more present for people I care about"),
    ("health_wellbeing", "I'm neglecting my health or wellbeing"),
    ("direction", "I feel stuck or unclear about my direction"),
    ("meaningful_work", "I want to make more room for what matters to me"),
]

_ROUND_DIGITS = 8


def _round(value: float) -> float:
    return round(value, _ROUND_DIGITS)


@dataclass(frozen=True)
class BwsResponse:
    """One completed BWS set: the object tapped Most and Least of the six."""

    set_index: int  # 0-10, position into BWS_SETS
    most: str  # BWS object key
    least: str  # BWS object key


@dataclass(frozen=True)
class RawBwsScores:
    appearances: dict[str, int]
    best_counts: dict[str, int]
    worst_counts: dict[str, int]
    net_counts: dict[str, int]
    scores: dict[str, float]


@dataclass(frozen=True)
class ProfileTransform:
    scores: dict[str, float]
    weights: dict[str, float]
    top_values: list[str]
    bottom_values: list[str]


# ── Scoring (domain.ts scoreResponses / transformForProfile) ────────────────


def score_responses(
    responses: list[BwsResponse],
) -> tuple[RawBwsScores, ProfileTransform]:
    """Score BWS responses, then transform into the ten-value Profile.

    Raw object scores (net_count / appearances) stay separate from the
    Profile transform, which averages the two Universalism facets and
    shift-normalizes into non-negative weights summing to exactly 1.0.
    """
    if not responses:
        raise ValueError("At least one BWS response is required")

    appearances = dict.fromkeys(BWS_OBJECT_ORDER, 0)
    best_counts = dict.fromkeys(BWS_OBJECT_ORDER, 0)
    worst_counts = dict.fromkeys(BWS_OBJECT_ORDER, 0)
    for response in responses:
        for item in BWS_SETS[response.set_index]:
            appearances[item] += 1
        best_counts[response.most] += 1
        worst_counts[response.least] += 1

    net_counts = {
        key: best_counts[key] - worst_counts[key] for key in BWS_OBJECT_ORDER
    }
    scores = {
        key: (_round(net_counts[key] / appearances[key]) if appearances[key] else 0.0)
        for key in BWS_OBJECT_ORDER
    }
    raw = RawBwsScores(
        appearances=appearances,
        best_counts=best_counts,
        worst_counts=worst_counts,
        net_counts=net_counts,
        scores=scores,
    )
    return raw, _transform_for_profile(scores)


def _transform_for_profile(object_scores: dict[str, float]) -> ProfileTransform:
    value_scores: dict[str, float] = {}
    for value in SCHWARTZ_VALUE_ORDER:
        if value == "universalism":
            nature = object_scores["universalism_nature"]
            social = object_scores["universalism_social"]
            value_scores[value] = _round((nature + social) / 2)
        else:
            # Every other BWS object key equals its Schwartz value key 1:1.
            value_scores[value] = object_scores[value]

    highest = max(value_scores.values())
    lowest = min(value_scores.values())
    tolerance = 1e-8
    return ProfileTransform(
        scores=value_scores,
        weights=_normalized_weights(value_scores),
        top_values=[
            v
            for v in SCHWARTZ_VALUE_ORDER
            if abs(value_scores[v] - highest) <= tolerance
        ],
        bottom_values=[
            v
            for v in SCHWARTZ_VALUE_ORDER
            if abs(value_scores[v] - lowest) <= tolerance
        ],
    )


def _normalized_weights(scores: dict[str, float]) -> dict[str, float]:
    """Shift to non-negative, then normalize to sum to exactly 1.0.

    The last value absorbs the rounding remainder (1 - sum of the rest)
    instead of its own shifted/total quotient, so floating-point rounding on
    the other nine can never make the total miss 1.0.
    """
    minimum = min(scores.values())
    shifted = {v: scores[v] - minimum + 1 for v in SCHWARTZ_VALUE_ORDER}
    total = sum(shifted.values())
    weights: dict[str, float] = {}
    assigned = 0.0
    for index, value in enumerate(SCHWARTZ_VALUE_ORDER):
        if index == len(SCHWARTZ_VALUE_ORDER) - 1:
            weights[value] = _round(1 - assigned)
        else:
            weights[value] = _round(shifted[value] / total)
            assigned += weights[value]
    return weights


def top_core_values(profile: ProfileTransform, max_values: int = 2) -> list[str]:
    """The tied top-weight value(s) from a Profile, capped at ``max_values``.

    A single clear winner returns one value. A tie at the top returns up to
    ``max_values`` tied values — narrowing the Profile's top_values (which
    can be any size, including all 10 on a perfectly flat profile) to the
    app's 1-2 Core Value contract.
    """
    return profile.top_values[:max_values]
