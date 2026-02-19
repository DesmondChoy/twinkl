#!/usr/bin/env python3
"""Quick sanity checks for judge labeling helpers."""

from __future__ import annotations

from src.judge.labeling import build_value_rubric_context, load_schwartz_values


def main() -> None:
    schwartz_config = load_schwartz_values("config/schwartz_values.yaml")
    rubric = build_value_rubric_context(schwartz_config)

    print("Rubric preview:")
    print(rubric[:700] + ("..." if len(rubric) > 700 else ""))


if __name__ == "__main__":
    main()
