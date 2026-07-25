"""Deterministic onboarding Profile projections for saved persona replays."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from src.demo.contracts import (
    BWS_OBJECT_ORDER,
    BWS_SETS,
    CORE_VALUE_ORDER,
    GoalCategory,
    OnboardingProfile,
)

SelectedPair = tuple[str, str]

_SELECTED_PAIRS: dict[frozenset[str], tuple[SelectedPair, ...]] = {
    frozenset({"achievement", "security"}): (
        ("achievement", "benevolence"),
        ("benevolence", "power"),
        ("security", "stimulation"),
        ("security", "hedonism"),
        ("security", "universalism_nature"),
        ("achievement", "self_direction"),
        ("achievement", "tradition"),
        ("achievement", "conformity"),
        ("security", "benevolence"),
        ("self_direction", "universalism_nature"),
        ("universalism_nature", "self_direction"),
    ),
    frozenset({"universalism"}): (
        ("universalism_social", "achievement"),
        ("universalism_social", "power"),
        ("power", "stimulation"),
        ("universalism_social", "hedonism"),
        ("universalism_social", "power"),
        ("universalism_social", "stimulation"),
        ("universalism_nature", "self_direction"),
        ("universalism_nature", "benevolence"),
        ("stimulation", "tradition"),
        ("universalism_nature", "conformity"),
        ("universalism_nature", "security"),
    ),
    frozenset({"power"}): (
        ("universalism_social", "universalism_nature"),
        ("power", "universalism_social"),
        ("power", "achievement"),
        ("achievement", "hedonism"),
        ("power", "hedonism"),
        ("power", "achievement"),
        ("power", "achievement"),
        ("achievement", "hedonism"),
        ("hedonism", "stimulation"),
        ("stimulation", "self_direction"),
        ("power", "self_direction"),
    ),
    frozenset({"self_direction", "tradition"}): (
        ("tradition", "achievement"),
        ("tradition", "power"),
        ("tradition", "power"),
        ("self_direction", "achievement"),
        ("power", "hedonism"),
        ("self_direction", "power"),
        ("self_direction", "universalism_nature"),
        ("achievement", "stimulation"),
        ("self_direction", "benevolence"),
        ("tradition", "conformity"),
        ("power", "security"),
    ),
    frozenset({"conformity", "self_direction"}): (
        ("universalism_social", "universalism_nature"),
        ("conformity", "universalism_social"),
        ("conformity", "power"),
        ("self_direction", "achievement"),
        ("power", "hedonism"),
        ("self_direction", "power"),
        ("self_direction", "power"),
        ("conformity", "stimulation"),
        ("self_direction", "benevolence"),
        ("conformity", "tradition"),
        ("power", "security"),
    ),
}


def _round(value: float, digits: int = 8) -> float:
    return round(value + 2.220446049250313e-16, digits)


def _normalized_core_values(core_values: Sequence[str]) -> list[str]:
    normalized = {
        value.strip().lower().replace("-", "_").replace(" ", "_")
        for value in core_values
    }
    invalid = normalized - set(CORE_VALUE_ORDER)
    if invalid:
        raise ValueError("Unknown Core Values: " + ", ".join(sorted(invalid)))
    return [value for value in CORE_VALUE_ORDER if value in normalized]


def build_projected_profile(
    *,
    persona_id: str,
    session_id: str,
    core_values: Sequence[str],
    goal_category: GoalCategory,
    started_at: str,
    completed_at: str,
) -> OnboardingProfile:
    """Project declared Core Values into a valid, explicitly synthetic Profile."""
    expected_top_values = _normalized_core_values(core_values)
    selected_pairs = _SELECTED_PAIRS.get(frozenset(expected_top_values))
    if selected_pairs is None:
        raise ValueError(
            "No deterministic Profile projection for: " + ", ".join(expected_top_values)
        )

    appearances = {value: 0 for value in BWS_OBJECT_ORDER}
    best_counts = {value: 0 for value in BWS_OBJECT_ORDER}
    worst_counts = {value: 0 for value in BWS_OBJECT_ORDER}
    responses: list[dict[str, Any]] = []
    for set_number, (items, pair) in enumerate(
        zip(BWS_SETS, selected_pairs, strict=True),
        start=1,
    ):
        selected_best, selected_worst = pair
        for item in items:
            appearances[item] += 1
        best_counts[selected_best] += 1
        worst_counts[selected_worst] += 1
        responses.append(
            {
                "set_number": set_number,
                "items": list(items),
                "item_order_shown": list(items),
                "selected_best": selected_best,
                "selected_worst": selected_worst,
                "response_time_ms": 0,
            }
        )

    net_counts = {
        value: best_counts[value] - worst_counts[value] for value in BWS_OBJECT_ORDER
    }
    scores = {
        value: _round(net_counts[value] / appearances[value])
        for value in BWS_OBJECT_ORDER
    }
    profile_scores = {
        value: (
            _round((scores["universalism_nature"] + scores["universalism_social"]) / 2)
            if value == "universalism"
            else scores[value]
        )
        for value in CORE_VALUE_ORDER
    }
    minimum = min(profile_scores.values())
    shifted = [profile_scores[value] - minimum + 1 for value in CORE_VALUE_ORDER]
    total = sum(shifted)
    weights: dict[str, float] = {}
    assigned = 0.0
    for index, value in enumerate(CORE_VALUE_ORDER):
        if index == len(CORE_VALUE_ORDER) - 1:
            weights[value] = _round(1 - assigned)
        else:
            weights[value] = _round(shifted[index] / total)
            assigned += weights[value]

    highest = max(profile_scores.values())
    lowest = min(profile_scores.values())
    top_values = [
        value
        for value in CORE_VALUE_ORDER
        if abs(profile_scores[value] - highest) <= 1e-8
    ]
    bottom_values = [
        value
        for value in CORE_VALUE_ORDER
        if abs(profile_scores[value] - lowest) <= 1e-8
    ]
    if top_values != expected_top_values:
        raise ValueError("Projected Profile does not preserve declared Core Values")

    return cast(
        OnboardingProfile,
        OnboardingProfile.model_validate(
            {
                "schema_version": 2,
                "onboarding_version": "2.1.0",
                "instrument": ("svbws_lee_soutar_louviere_2008_ui_adaptation_v2"),
                "scoring_method": ("best_minus_worst_divided_by_appearances_v1"),
                "user_id": persona_id,
                "session_id": session_id,
                "started_at": started_at,
                "timestamp": completed_at,
                "bws_responses": responses,
                "bws_results": {
                    "appearances": appearances,
                    "best_counts": best_counts,
                    "worst_counts": worst_counts,
                    "net_counts": net_counts,
                    "scores": scores,
                },
                "value_profile": {
                    "method": ("mean_universalism_facets_then_shift_normalize_v1"),
                    "scores": profile_scores,
                    "weights": weights,
                    "top_values": top_values,
                    "bottom_values": bottom_values,
                },
                "top_values": top_values,
                "goal_category": goal_category,
                "user_confirmed": True,
                "provenance": {
                    "source": "synthetic_persona_projection",
                    "set_order_randomized": False,
                    "card_order_randomized": False,
                },
            }
        ),
    )
