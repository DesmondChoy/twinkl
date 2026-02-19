"""Synthetic data generation helpers extracted from notebook prototypes.

This package contains reusable utilities for persona/value context building,
date sequence sampling, and banned-term safeguards used by synthetic
generation workflows.
"""

from src.synthetic.generation import (
    SCHWARTZ_BANNED_PATTERN,
    SCHWARTZ_BANNED_TERMS,
    build_banned_pattern,
    build_value_context,
    generate_date_sequence,
    load_yaml_config,
    sample_entry_gap_days,
    verbosity_targets,
)

__all__ = [
    "SCHWARTZ_BANNED_TERMS",
    "SCHWARTZ_BANNED_PATTERN",
    "load_yaml_config",
    "build_value_context",
    "verbosity_targets",
    "build_banned_pattern",
    "sample_entry_gap_days",
    "generate_date_sequence",
]
