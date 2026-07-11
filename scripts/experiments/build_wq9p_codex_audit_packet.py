#!/usr/bin/env python3
"""Build a metadata-blinded input packet for the twinkl-16ar Codex audit.

The packet deliberately omits consensus labels, scorer outputs, author design
notes, expected states, dates, and source-specific value formatting. A separate
reconciliation key is written alongside it for the parent reviewer to use only
after the blind assessment has been submitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
import yaml

# Allow direct execution with ``python scripts/experiments/<script>.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.vif.dataset import load_entries  # noqa: E402, I001


DEFAULT_ARTIFACT_ROOT = Path(
    "logs/experiments/artifacts/drift_trigger_benchmark_twinkl_wq9p_20260710"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ARTIFACT_ROOT / "codex_audit"
DEFAULT_REFERENCE_EPISODES = DEFAULT_ARTIFACT_ROOT / "reference_episodes_test.parquet"
DEFAULT_DESIGNED_HOLDOUT = Path("config/evals/drift_v1_designed_holdout.yaml")
DEFAULT_WRANGLED_DIR = Path("logs/wrangled")
ASSESSMENT_FILENAMES = (
    "blind_assessment.json",
    "blind_assessment_check.json",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _git_head() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    return result.stdout.strip()


def _format_dimension(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").title()


def _entry_payload(entry: dict[str, Any]) -> dict[str, str]:
    text = entry.get("initial_entry")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing journal text for {entry!r}")
    # Entry order is preserved by the list. Omit absolute dates because the
    # frozen and designed fixtures use different calendar eras, which would
    # reveal their source set to a reviewer.
    return {"journal_entry": text.strip()}


def _frozen_cases(
    reference_path: Path,
    wrangled_dir: Path,
) -> list[dict[str, Any]]:
    episodes = pl.read_parquet(reference_path)
    entries = load_entries(wrangled_dir)
    entry_lookup = {
        (row["persona_id"], int(row["t_index"])): row for row in entries.to_dicts()
    }
    cases: list[dict[str, Any]] = []

    for episode in episodes.to_dicts():
        indices = list(episode["supporting_t_indices"])
        termination_index = episode.get("termination_t_index")
        if termination_index is not None and int(termination_index) not in indices:
            indices.append(int(termination_index))

        rows = []
        for t_index in sorted(indices):
            key = (episode["persona_id"], int(t_index))
            try:
                rows.append(entry_lookup[key])
            except KeyError as exc:
                raise ValueError(f"No wrangled journal entry for {key}") from exc

        cases.append(
            {
                "hidden_source": "frozen_consensus",
                "hidden_case_category": "frozen_consensus",
                "hidden_source_case_id": episode["episode_id"],
                "hidden_persona_id": episode["persona_id"],
                "declared_core_value": _format_dimension(episode["dimension"]),
                "entries": [_entry_payload(row) for row in rows],
            }
        )

    return cases


def _designed_cases(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise ValueError(f"Invalid designed-holdout manifest: {manifest_path}")

    cases: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        core_values = case.get("core_values")
        if not isinstance(core_values, list) or len(core_values) != 1:
            case_id = case.get("case_id")
            raise ValueError(f"Expected one declared core value in {case_id}")
        expected_case_state = case.get("expected_case_state")
        if expected_case_state in {"active", "recovered"}:
            hidden_case_category = "designed_positive"
        elif expected_case_state in {"none", "uncertain"}:
            hidden_case_category = "designed_control"
        else:
            case_id = case.get("case_id")
            raise ValueError(
                "Expected designed case state active, recovered, none, or uncertain "
                f"in {case_id}; found {expected_case_state!r}"
            )
        cases.append(
            {
                "hidden_source": "designed_holdout",
                "hidden_case_category": hidden_case_category,
                "hidden_source_case_id": case["case_id"],
                "hidden_persona_id": case["persona_id"],
                "declared_core_value": _format_dimension(str(core_values[0])),
                "entries": [_entry_payload(entry) for entry in case["entries"]],
            }
        )
    return cases


def build_packet(
    reference_path: Path,
    designed_holdout_path: Path,
    wrangled_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_cases = _frozen_cases(reference_path, wrangled_dir) + _designed_cases(
        designed_holdout_path
    )
    if len(source_cases) != 25:
        raise ValueError(f"Expected 25 cases, found {len(source_cases)}")

    random.Random(20260710).shuffle(source_cases)
    packet_cases: list[dict[str, Any]] = []
    key_cases: list[dict[str, str]] = []

    for position, case in enumerate(source_cases, start=1):
        review_case_id = f"case_{position:02d}"
        packet_cases.append(
            {
                "review_case_id": review_case_id,
                "declared_core_value": case["declared_core_value"],
                "entries": case["entries"],
            }
        )
        key_cases.append(
            {
                "review_case_id": review_case_id,
                "source": case["hidden_source"],
                "case_category": case["hidden_case_category"],
                "source_case_id": case["hidden_source_case_id"],
                "persona_id": case["hidden_persona_id"],
            }
        )

    frozen_entry_count = sum(
        len(case["entries"])
        for case in source_cases
        if case["hidden_source"] == "frozen_consensus"
    )
    designed_entry_count = sum(
        len(case["entries"])
        for case in source_cases
        if case["hidden_source"] == "designed_holdout"
    )
    if frozen_entry_count != 14 or designed_entry_count != 60:
        raise ValueError(
            "Unexpected packet size: "
            f"frozen={frozen_entry_count}, designed={designed_entry_count}"
        )

    category_counts = Counter(
        case["hidden_case_category"] for case in source_cases
    )
    expected_category_counts = Counter(
        frozen_consensus=5,
        designed_positive=10,
        designed_control=10,
    )
    if category_counts != expected_category_counts:
        raise ValueError(
            "Unexpected case-category counts: "
            f"expected={dict(expected_category_counts)}, found={dict(category_counts)}"
        )

    packet = {
        "schema_version": 1,
        "purpose": "Metadata-blinded Codex audit for twinkl-16ar",
        "review_instructions": [
            "Do not inspect the reconciliation key or benchmark predictions "
            "before recording the blind assessment.",
            "Answer yes only when two immediately adjacent displayed entries each "
            "show the writer's own directly observable behavior or choice "
            "conflicting with the declared core value.",
            "A neutral, unknown, recovered, or non-conflicting entry breaks the "
            "run. Do not bridge across it, even if an earlier and later entry both "
            "look conflicting.",
            "Do not count a feeling, desire, external constraint, or a single "
            "conflicting action as a sustained conflict without a second adjacent "
            "directly conflicting action.",
            "Record whether the conclusion is visible from the displayed text, "
            "needs prior history, needs profile or biography context, is ambiguous, "
            "or appears to overreach the available evidence.",
            "Use a short evidence-based rationale. This is an AI audit, not human "
            "ground truth.",
        ],
        "cases": packet_cases,
    }
    reconciliation_key = {
        "schema_version": 1,
        "warning": "Do not use before the blind assessment is complete.",
        "cases": key_cases,
    }
    return packet, reconciliation_key


def build_audit_manifest(
    *,
    output_dir: Path,
    reference_path: Path,
    designed_holdout_path: Path,
    wrangled_dir: Path,
) -> dict[str, Any]:
    """Record reproducibility hashes and the limits of the completed audit."""
    packet_path = output_dir / "blind_packet.json"
    key_path = output_dir / "reconciliation_key.json"
    assessments = {
        filename: _sha256_file(output_dir / filename)
        for filename in ASSESSMENT_FILENAMES
        if (output_dir / filename).is_file()
    }
    frozen_persona_ids = sorted(
        pl.read_parquet(reference_path)["persona_id"].unique().to_list()
    )
    frozen_wrangled_files = []
    for persona_id in frozen_persona_ids:
        source_path = wrangled_dir / f"persona_{persona_id}.md"
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Missing frozen persona source file: {source_path}"
            )
        frozen_wrangled_files.append(
            {
                "path": _repo_relative(source_path),
                "sha256": _sha256_file(source_path),
            }
        )
    return {
        "schema_version": 1,
        "audit_protocol_version": "twinkl-16ar-final-v1",
        "generator": {
            "path": _repo_relative(Path(__file__)),
            "sha256": _sha256_file(Path(__file__)),
            "git_head_at_generation": _git_head(),
            "command": "python scripts/experiments/build_wq9p_codex_audit_packet.py",
            "shuffle_seed": 20260710,
        },
        "source_inputs": {
            "reference_episodes": {
                "path": _repo_relative(reference_path),
                "sha256": _sha256_file(reference_path),
            },
            "designed_holdout": {
                "path": _repo_relative(designed_holdout_path),
                "sha256": _sha256_file(designed_holdout_path),
            },
            "frozen_wrangled_files": {
                "loader": "src.vif.dataset.load_entries()",
                "directory": _repo_relative(wrangled_dir),
                "files": frozen_wrangled_files,
                "pin": "The displayed text is fixed by blind_packet.json's SHA-256.",
            },
        },
        "outputs": {
            "blind_packet.json": _sha256_file(packet_path),
            "reconciliation_key.json": _sha256_file(key_path),
            "assessment_sha256": assessments,
        },
        "review_protocol": {
            "reviewer_input": "blind_packet.json only, by procedural instruction",
            "reviewer_runtime": "Codex subagent",
            "model_identifier": (
                "not recorded; the orchestration API exposed no model selector"
            ),
            "review_timestamps": "not captured",
            "technical_isolation": False,
            "response_schema": (
                "unversioned legacy fields; delivery_state was advisory rather than "
                "an agreement criterion"
            ),
        },
        "known_limitations": [
            (
                "Reviewers were instructed not to inspect the reconciliation key or "
                "source artifacts, but the completed run was not technically sandboxed."
            ),
            (
                "Only sustained_conflict qualification was used for the 25/25 "
                "agreement claim; secondary delivery/context fields had disagreements."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-episodes",
        type=Path,
        default=DEFAULT_REFERENCE_EPISODES,
    )
    parser.add_argument(
        "--designed-holdout",
        type=Path,
        default=DEFAULT_DESIGNED_HOLDOUT,
    )
    parser.add_argument(
        "--wrangled-dir",
        type=Path,
        default=DEFAULT_WRANGLED_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    packet, reconciliation_key = build_packet(
        args.reference_episodes,
        args.designed_holdout,
        args.wrangled_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.output_dir / "blind_packet.json"
    key_path = args.output_dir / "reconciliation_key.json"
    manifest_path = args.output_dir / "audit_manifest.json"
    packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    key_path.write_text(
        json.dumps(reconciliation_key, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = build_audit_manifest(
        output_dir=args.output_dir,
        reference_path=args.reference_episodes,
        designed_holdout_path=args.designed_holdout,
        wrangled_dir=args.wrangled_dir,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {packet_path}")
    print(f"Wrote {key_path}")
    print(f"Wrote {manifest_path}")
    print(f"Cases: {len(packet['cases'])}; journal entries: 74")


if __name__ == "__main__":
    main()
