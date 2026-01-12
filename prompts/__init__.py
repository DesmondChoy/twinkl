"""Prompt template loader for synthetic data generation.

This module provides utilities for loading Jinja2 prompt templates from YAML files.
Templates are pre-loaded at import time for convenient access.

Usage:
    from prompts import persona_generation_prompt, journal_entry_prompt

    # Render a template
    prompt_text = persona_generation_prompt.render(
        age="25-34",
        profession="Software Engineer",
        culture="Japanese",
        values=["Achievement", "Self-Direction"],
        value_context="...",
        banned_terms=["Achievement", "ambitious", ...]
    )
"""

from pathlib import Path

import yaml
from jinja2 import Template

PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> Template:
    """Load a prompt template by name.

    Args:
        name: Prompt name (without .yaml extension)

    Returns:
        Jinja2 Template ready for rendering

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        yaml.YAMLError: If the YAML is malformed
    """
    path = PROMPTS_DIR / f"{name}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return Template(data["template"])


def get_prompt_metadata(name: str) -> dict:
    """Get metadata for a prompt (name, description, version, input_variables).

    Args:
        name: Prompt name (without .yaml extension)

    Returns:
        Dict with prompt metadata (excludes template content)
    """
    path = PROMPTS_DIR / f"{name}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return {k: v for k, v in data.items() if k != "template"}


# Pre-load commonly used prompts for convenient imports
persona_generation_prompt = load_prompt("persona_generation")
journal_entry_prompt = load_prompt("journal_entry")
nudge_decision_prompt = load_prompt("nudge_decision")
nudge_generation_prompt = load_prompt("nudge_generation")
nudge_response_prompt = load_prompt("nudge_response")
judge_alignment_prompt = load_prompt("judge_alignment")

__all__ = [
    "load_prompt",
    "get_prompt_metadata",
    "persona_generation_prompt",
    "journal_entry_prompt",
    "nudge_decision_prompt",
    "nudge_generation_prompt",
    "nudge_response_prompt",
    "judge_alignment_prompt",
]
