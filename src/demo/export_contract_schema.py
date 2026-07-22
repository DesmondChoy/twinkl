"""Export the checked-in Experience and Inspect JSON Schema."""

from __future__ import annotations

import json
from pathlib import Path

from src.demo.canonical_fixture import build_canonical_fixture
from src.demo.contracts import ContractFixtureSet

DEFAULT_OUTPUT = Path(
    "frontend/onboarding/src/contracts/experience_inspect_v1.schema.json"
)
DEFAULT_FIXTURE_OUTPUT = Path(
    "frontend/onboarding/src/contracts/experience_inspect_v1.fixture.json"
)


def export_schema(output: Path = DEFAULT_OUTPUT) -> None:
    """Write the deterministic Pydantic JSON Schema used by React tests."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = ContractFixtureSet.model_json_schema()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def export_fixture(output: Path = DEFAULT_FIXTURE_OUTPUT) -> None:
    """Write the canonical fixture after validating all contract links."""
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_canonical_fixture().model_dump(mode="json")
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    export_schema()
    export_fixture()
