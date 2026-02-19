"""Tests for synthetic generation helpers."""

from __future__ import annotations

import random

import pytest

from src.synthetic.generation import (
    SCHWARTZ_BANNED_PATTERN,
    build_value_context,
    generate_date_sequence,
    load_yaml_config,
    sample_entry_gap_days,
    verbosity_targets,
)


def test_verbosity_targets_mapping():
    assert verbosity_targets("Short (1-3 sentences)") == (25, 80, 1)
    assert verbosity_targets("Medium (1-2 paragraphs)") == (90, 180, 2)
    assert verbosity_targets("Long (Detailed reflection)") == (160, 260, 3)


def test_banned_pattern_matches_value_terms():
    assert SCHWARTZ_BANNED_PATTERN.search("I felt very ambitious today")
    assert SCHWARTZ_BANNED_PATTERN.search("This reflects Self-Direction")
    assert not SCHWARTZ_BANNED_PATTERN.search("I took the bus to work")


def test_sample_gap_same_day_when_probability_is_one():
    rng = random.Random(42)
    gap = sample_entry_gap_days(
        min_days_between_entries=0,
        max_days_between_entries=7,
        same_day_probability=1.0,
        rng=rng,
    )
    assert gap == 0


def test_sample_gap_same_day_only_range_returns_zero():
    rng = random.Random(7)
    gap = sample_entry_gap_days(
        min_days_between_entries=0,
        max_days_between_entries=0,
        same_day_probability=0.0,
        rng=rng,
    )
    assert gap == 0


def test_sample_gap_validates_probability_bounds():
    with pytest.raises(ValueError):
        sample_entry_gap_days(
            min_days_between_entries=0,
            max_days_between_entries=7,
            same_day_probability=1.1,
        )


def test_generate_date_sequence_fixed_gap():
    dates = generate_date_sequence(
        start_date="2025-01-01",
        num_entries=4,
        min_days_between_entries=2,
        max_days_between_entries=2,
        same_day_probability=0.0,
        rng=random.Random(123),
    )
    assert dates == ["2025-01-01", "2025-01-03", "2025-01-05", "2025-01-07"]


def test_build_value_context_uses_config_data():
    schwartz_config = load_yaml_config("config/schwartz_values.yaml")
    context = build_value_context(["Achievement"], schwartz_config)

    assert "### Achievement" in context
    assert "Core Motivation" in context
