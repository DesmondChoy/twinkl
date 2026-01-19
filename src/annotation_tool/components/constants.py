"""Shared constants for annotation tool components.

This module consolidates constants used across multiple components to avoid
duplication and ensure consistency.
"""

# Human-readable labels for Schwartz values
VALUE_LABELS = {
    "self_direction": "Self-Direction",
    "stimulation": "Stimulation",
    "hedonism": "Hedonism",
    "achievement": "Achievement",
    "power": "Power",
    "security": "Security",
    "conformity": "Conformity",
    "tradition": "Tradition",
    "benevolence": "Benevolence",
    "universalism": "Universalism",
}


def format_score(score: int) -> tuple[str, str]:
    """Format a score value for display.

    Args:
        score: Integer score (-1, 0, or 1)

    Returns:
        Tuple of (display_text, css_class)
    """
    if score == 1:
        return "+1", "positive"
    if score == -1:
        return "−1", "negative"
    return "0", "neutral"


def get_match_class(human_score: int, judge_score: int) -> tuple[str, str, str]:
    """Determine match type between human and judge scores.

    Args:
        human_score: Human annotator's score
        judge_score: LLM Judge's score

    Returns:
        Tuple of (row_class, symbol, symbol_class)
    """
    if human_score == judge_score:
        return "match-exact", "✓", "exact"
    return "match-disagree", "✗", "disagree"
