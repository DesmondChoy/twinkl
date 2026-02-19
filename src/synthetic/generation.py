"""Synthetic journaling generation utilities.

This module extracts notebook-era helper logic into importable Python code so
generation pipelines and documentation can reference stable script paths.
"""

from __future__ import annotations

import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import yaml

# Banned terms include Schwartz value labels and common derivatives to reduce
# label leakage in synthetic persona bios and entries.
SCHWARTZ_BANNED_TERMS = [
    "Self-Direction",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power",
    "Security",
    "Conformity",
    "Tradition",
    "Benevolence",
    "Universalism",
    "self-directed",
    "autonomous",
    "stimulating",
    "excited",
    "hedonistic",
    "hedonist",
    "pleasure-seeking",
    "achievement-oriented",
    "ambitious",
    "powerful",
    "authoritative",
    "secure",
    "conformist",
    "conforming",
    "traditional",
    "traditionalist",
    "benevolent",
    "kind-hearted",
    "universalistic",
    "altruistic",
    "Schwartz",
    "values",
    "core values",
]


def load_yaml_config(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    config_path = Path(path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_value_context(values: Sequence[str], schwartz_config: dict) -> str:
    """Build value-elaboration context used in persona generation prompts."""
    context_parts: list[str] = []

    value_map = schwartz_config.get("values", {})
    for value_name in values:
        value_data = value_map.get(value_name)
        if not value_data:
            continue

        context_parts.append(
            f"""
### {value_name}
**Core Motivation:** {value_data["core_motivation"].strip()}

**How this manifests in behavior:**
{chr(10).join(f"- {behavior}" for behavior in value_data["behavioral_manifestations"][:5])}

**Life domain expressions:**
- Work: {value_data["life_domain_expressions"]["work"].strip()}
- Relationships: {value_data["life_domain_expressions"]["relationships"].strip()}

**Typical stressors for this person:**
{chr(10).join(f"- {stressor}" for stressor in value_data["typical_stressors"][:4])}

**Typical goals:**
{chr(10).join(f"- {goal}" for goal in value_data["typical_goals"][:3])}

**Internal conflicts they may experience:**
{value_data["internal_conflicts"].strip()}

**Narrative guidance:**
{value_data["persona_narrative_guidance"].strip()}
""".strip()
        )

    return "\n\n".join(context_parts)


def verbosity_targets(verbosity: str) -> tuple[int, int, int]:
    """Map verbosity label to (min_words, max_words, max_paragraphs)."""
    normalized = verbosity.strip().lower()
    if normalized.startswith("short"):
        return 25, 80, 1
    if normalized.startswith("medium"):
        return 90, 180, 2
    return 160, 260, 3


def build_banned_pattern(banned_terms: Sequence[str]) -> re.Pattern:
    """Build a case-insensitive regex for banned-term detection."""
    escaped = [re.escape(term) for term in banned_terms if term.strip()]
    if not escaped:
        return re.compile(r"$^")
    return re.compile(r"(?i)\b(" + "|".join(escaped) + r")\b")


SCHWARTZ_BANNED_PATTERN = build_banned_pattern(SCHWARTZ_BANNED_TERMS)


def sample_entry_gap_days(
    min_days_between_entries: int,
    max_days_between_entries: int,
    same_day_probability: float = 0.15,
    rng: random.Random | None = None,
) -> int:
    """Sample a single day gap between entries.

    Rules:
    - Same-day (0) is only sampled when min gap is 0.
    - If min/max are both 0, same-day is mandatory.
    - Otherwise sample from [max(min_gap, 1), max_gap].
    """
    if min_days_between_entries < 0:
        raise ValueError("min_days_between_entries must be >= 0")
    if max_days_between_entries < min_days_between_entries:
        raise ValueError(
            "max_days_between_entries must be >= min_days_between_entries"
        )
    if not 0 <= same_day_probability <= 1:
        raise ValueError("same_day_probability must be in [0, 1]")

    rng = rng or random
    if (
        min_days_between_entries == 0
        and same_day_probability > 0
        and rng.random() < same_day_probability
    ):
        return 0

    if min_days_between_entries == 0 and max_days_between_entries == 0:
        return 0

    return rng.randint(max(min_days_between_entries, 1), max_days_between_entries)


def generate_date_sequence(
    start_date: str,
    num_entries: int,
    min_days_between_entries: int = 0,
    max_days_between_entries: int = 7,
    same_day_probability: float = 0.15,
    rng: random.Random | None = None,
) -> list[str]:
    """Generate a date sequence from a start date using sampled gaps."""
    if num_entries <= 0:
        return []

    rng = rng or random
    current = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [current.strftime("%Y-%m-%d")]

    for _ in range(num_entries - 1):
        gap = sample_entry_gap_days(
            min_days_between_entries=min_days_between_entries,
            max_days_between_entries=max_days_between_entries,
            same_day_probability=same_day_probability,
            rng=rng,
        )
        current += timedelta(days=gap)
        dates.append(current.strftime("%Y-%m-%d"))

    return dates
