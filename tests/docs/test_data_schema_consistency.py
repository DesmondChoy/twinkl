"""Ensure docs/pipeline/data_schema.md stays in sync with actual schemas.

These tests parse the markdown tables in data_schema.md and compare the
documented column names against the source-of-truth constants in code.
If a column is added or removed in code but not reflected in docs, these
tests will fail — preventing silent documentation drift.
"""

import json
import re
from pathlib import Path

import pytest

from src.judge.consolidate import consolidate_judge_labels
from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.registry.personas import REGISTRY_SCHEMA

DATA_SCHEMA_PATH = Path("docs/pipeline/data_schema.md")


def _extract_table_columns(markdown: str, section_heading: str) -> list[str]:
    """Extract column names from a markdown table under a given heading.

    Looks for a '### Schema' subsection under the given heading and parses
    the first column of each markdown table row (skipping the header separator).
    """
    # Find the section (## heading)
    pattern = rf"^## {re.escape(section_heading)}$"
    section_match = re.search(pattern, markdown, re.MULTILINE)
    if not section_match:
        pytest.fail(f"Section '## {section_heading}' not found in data_schema.md")

    # Get text from section start to next ## heading (or end of file)
    section_start = section_match.end()
    next_section = re.search(r"^## ", markdown[section_start:], re.MULTILINE)
    section_text = (
        markdown[section_start : section_start + next_section.start()]
        if next_section
        else markdown[section_start:]
    )

    # Find the ### Schema subsection
    schema_match = re.search(r"^### Schema$", section_text, re.MULTILINE)
    if not schema_match:
        pytest.fail(
            f"'### Schema' subsection not found under '## {section_heading}'"
        )

    schema_text = section_text[schema_match.end() :]

    # Parse markdown table rows: | `column_name` | ... |
    columns = []
    in_table = False
    for line in schema_text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if in_table:
                break  # End of table
            continue

        # Skip header separator (|---|---|...)
        if re.match(r"^\|[-\s|]+\|$", line):
            in_table = True
            continue

        # Skip header row (first row before separator)
        if not in_table:
            continue

        # Extract first cell content: | `column_name` | ... |
        cells = [c.strip() for c in line.split("|")]
        # Split produces ['', 'cell1', 'cell2', ..., ''] for | ... | ... |
        if len(cells) >= 2:
            col_name = cells[1].strip("`").strip()
            if col_name:
                columns.append(col_name)

    if not columns:
        pytest.fail(f"No table rows found under '## {section_heading} > ### Schema'")

    return columns


class TestRegistryColumnsDocumented:
    """Persona Registry docs must list every column in REGISTRY_SCHEMA."""

    def test_registry_columns_documented(self):
        markdown = DATA_SCHEMA_PATH.read_text()
        documented = set(_extract_table_columns(markdown, "Persona Registry"))
        expected = set(REGISTRY_SCHEMA.keys())

        missing_from_docs = expected - documented
        extra_in_docs = documented - expected

        assert not missing_from_docs, (
            f"Columns in REGISTRY_SCHEMA but missing from docs: {missing_from_docs}"
        )
        assert not extra_in_docs, (
            f"Columns in docs but missing from REGISTRY_SCHEMA: {extra_in_docs}"
        )


class TestJudgeLabelsColumnsDocumented:
    """Judge Labels docs must list every column produced by consolidation."""

    def test_judge_labels_columns_documented(self, tmp_path):
        # Generate a sample consolidated dataframe using the same pattern
        # as test_consolidate.py
        label_data = {
            "persona_id": "a1b2c3d4",
            "labels": [
                {
                    "t_index": 0,
                    "date": "2025-01-01",
                    "scores": {v: 0 for v in SCHWARTZ_VALUE_ORDER},
                    "rationales": {"self_direction": "Test rationale"},
                }
            ],
        }
        label_path = tmp_path / "persona_a1b2c3d4_labels.json"
        label_path.write_text(json.dumps(label_data))

        df, _ = consolidate_judge_labels(tmp_path, update_registry=False)
        actual_columns = set(df.columns)

        markdown = DATA_SCHEMA_PATH.read_text()
        documented = set(_extract_table_columns(markdown, "Judge Labels"))

        missing_from_docs = actual_columns - documented
        extra_in_docs = documented - actual_columns

        assert not missing_from_docs, (
            f"Columns in consolidated DataFrame but missing from docs: {missing_from_docs}"
        )
        assert not extra_in_docs, (
            f"Columns in docs but missing from consolidated DataFrame: {extra_in_docs}"
        )
