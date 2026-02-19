#!/usr/bin/env python3
"""Quick sanity checks for synthetic generation helpers.

This script is intentionally lightweight: it exercises the extracted helper
logic without making LLM API calls.
"""

from __future__ import annotations

from src.synthetic.generation import (
    SCHWARTZ_BANNED_PATTERN,
    build_value_context,
    generate_date_sequence,
    load_yaml_config,
)


def main() -> None:
    synthetic_config = load_yaml_config("config/synthetic_data.yaml")
    schwartz_config = load_yaml_config("config/schwartz_values.yaml")

    sample_values = synthetic_config["personas"]["schwartz_values"][:2]
    context = build_value_context(sample_values, schwartz_config)
    dates = generate_date_sequence(
        start_date="2025-01-15",
        num_entries=6,
        min_days_between_entries=0,
        max_days_between_entries=7,
        same_day_probability=0.15,
    )

    print("Sample values:", sample_values)
    print("Generated dates:", dates)
    print("Context preview:")
    print(context[:500] + ("..." if len(context) > 500 else ""))
    print("Banned pattern test ('ambitious'):", bool(SCHWARTZ_BANNED_PATTERN.search("ambitious")))


if __name__ == "__main__":
    main()
