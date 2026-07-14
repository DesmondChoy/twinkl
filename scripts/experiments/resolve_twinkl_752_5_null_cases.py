#!/usr/bin/env python3
"""Prepare and materialize four unresolved Conflict labels with blind Opus review."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vif.drift_candidate_review import (  # noqa: E402
    COHORT_VERSION,
    artifact_hashes,
    derive_review_outcomes,
    summarize_outcomes,
)
from src.vif.drift_target import (  # noqa: E402
    CONFIDENCE_LEVELS,
    REASON_CODES,
    sha256_file,
)

SOURCE_ROOT = Path(
    "logs/experiments/artifacts/"
    "twinkl_752_4_legacy_drift_review_20260713"
)
DEFAULT_OUTPUT = Path(
    "logs/experiments/artifacts/"
    "twinkl_752_5_opus_null_resolution_20260714"
)
RUBRIC_PATH = Path("config/evals/drift_v1_conflict_rubric_v1.yaml")
VALUES_PATH = Path("config/schwartz_values.yaml")
SCHEMA_VERSION = "twinkl-752.5-opus-null-resolution-v2"
PROMPT_VERSION = "twinkl-752.5-opus-blind-fourth-review-v2"
EXPECTED_CASE_COUNT = 4
EXPECTED_DISPUTED_ENTRY_COUNT = 4

REVIEW_RULES = [
    (
        "Use only the declared Core Value and displayed Journal Entry text, "
        "including any displayed nudge and response."
    ),
    (
        "Mark yes only when the writer describes their own directly observable "
        "behavior or choice against the declared Core Value."
    ),
    (
        "Mark no for aligned or neutral behavior and when negativity rests only "
        "on feelings, wishes, biography, external constraints, or hidden context."
    ),
    (
        "Make a forced best-supported yes/no decision for each disputed position. "
        "Use low confidence when the displayed evidence is genuinely close."
    ),
    "Do not relabel positions that are not marked disputed.",
]


def _rooted(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _manifest_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _verify_hashes(hashes: dict[str, str], *, label: str) -> None:
    for relative, digest in hashes.items():
        path = ROOT / relative
        if sha256_file(path) != digest:
            raise ValueError(f"{label} changed: {relative}")


def _source_paths(source: Path) -> dict[str, Path]:
    return {
        "entry_target": source / "results/entry_target_final.parquet",
        "case_outcomes": source / "results/case_outcomes_final.parquet",
        "drift_episodes": source / "results/drift_episodes_final.parquet",
        "selected_cases": source / "parent_control/selected_cases.parquet",
        "cohort_manifest": source / "parent_control/cohort_manifest.json",
        "old_packet": source / "adjudication/reviewer_packet.json",
        "old_key": source / "adjudication/parent_reconciliation_key.json",
        "final_audit": source / "results/audit_manifest_final.json",
    }


def _verify_source(paths: dict[str, Path]) -> None:
    audit = _read_json(paths["final_audit"])
    _verify_hashes(audit["results"], label="Source result")
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing source {name}: {path}")


def _stable_order(case_id: str) -> str:
    return hashlib.sha256(f"{PROMPT_VERSION}|{case_id}".encode()).hexdigest()


def _build_packet(
    paths: dict[str, Path],
    rubric: dict[str, Any],
    values: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], pl.DataFrame, pl.DataFrame]:
    outcomes = pl.read_parquet(paths["case_outcomes"])
    entries = pl.read_parquet(paths["entry_target"])
    unresolved = outcomes.filter(pl.col("case_resolution") == "unresolved")
    unresolved_entries = entries.filter(pl.col("resolution_status") == "unresolved")
    if unresolved.height != EXPECTED_CASE_COUNT:
        raise ValueError(f"Expected four unresolved cases, found {unresolved.height}")
    if unresolved_entries.height != EXPECTED_DISPUTED_ENTRY_COUNT:
        raise ValueError(
            f"Expected four unresolved entries, found {unresolved_entries.height}"
        )

    old_packet = _read_json(paths["old_packet"])
    old_key = _read_json(paths["old_key"])
    old_packet_map = {
        case["adjudication_case_id"]: case for case in old_packet["cases"]
    }
    old_id_by_case = {
        case["canonical_case_id"]: case["adjudication_case_id"]
        for case in old_key["cases"]
    }
    unresolved_ids = sorted(
        unresolved["canonical_case_id"].to_list(), key=_stable_order
    )
    packet_cases = []
    key_cases = []
    value_definitions = values.get("values")
    if not isinstance(value_definitions, dict):
        raise ValueError("Schwartz value config has no values mapping")
    for index, canonical_case_id in enumerate(unresolved_ids, start=1):
        review_case_id = f"opus_case_{index:03d}"
        old_case = old_packet_map[old_id_by_case[canonical_case_id]]
        declared_core_value = old_case["declared_core_value"]
        value_definition = value_definitions.get(declared_core_value)
        if not isinstance(value_definition, dict):
            raise ValueError(f"Missing Core Value definition: {declared_core_value}")
        disputed_positions = (
            unresolved_entries.filter(
                pl.col("canonical_case_id") == canonical_case_id
            )["position"]
            .cast(pl.Int64)
            .to_list()
        )
        packet_cases.append(
            {
                "review_case_id": review_case_id,
                "declared_core_value": declared_core_value,
                "core_value_context": {
                    "definition": value_definition["definition"].strip(),
                    "core_motivation": value_definition["core_motivation"].strip(),
                },
                "disputed_positions": disputed_positions,
                "entries": [
                    {
                        "position": int(entry["position"]),
                        "journal_entry": entry["journal_entry"],
                        "is_disputed": int(entry["position"])
                        in disputed_positions,
                    }
                    for entry in old_case["entries"]
                ],
            }
        )
        key_cases.append(
            {
                "review_case_id": review_case_id,
                "canonical_case_id": canonical_case_id,
                "disputed_positions": disputed_positions,
            }
        )

    packet = {
        "schema_version": SCHEMA_VERSION,
        "cohort_version": COHORT_VERSION,
        "reviewer_prompt_version": PROMPT_VERSION,
        "rubric": {
            "rubric_id": rubric["rubric_id"],
            "version": rubric["version"],
            "entry_decision": rubric["entry_decision"],
            "evidence": rubric["evidence"],
            "limitations": rubric["limitations"],
        },
        "review_instructions": REVIEW_RULES,
        "cases": packet_cases,
    }
    key = {
        "schema_version": SCHEMA_VERSION,
        "cohort_version": COHORT_VERSION,
        "warning": "Parent-only identity key; never disclose before review.",
        "cases": key_cases,
    }
    return packet, key, entries, outcomes


def _structured_output_schema(packet: dict[str, Any]) -> dict[str, Any]:
    case_ids = [case["review_case_id"] for case in packet["cases"]]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["cases"],
        "properties": {
            "cases": {
                "type": "array",
                "minItems": len(case_ids),
                "maxItems": len(case_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "review_case_id",
                        "entry_adjudications",
                        "rationale",
                    ],
                    "properties": {
                        "review_case_id": {"type": "string", "enum": case_ids},
                        "entry_adjudications": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "position",
                                    "observable_conflict",
                                    "reason_code",
                                    "confidence",
                                ],
                                "properties": {
                                    "position": {"type": "integer", "minimum": 1},
                                    "observable_conflict": {
                                        "type": "string",
                                        "enum": ["yes", "no"],
                                    },
                                    "reason_code": {
                                        "type": "string",
                                        "enum": sorted(REASON_CODES),
                                    },
                                    "confidence": {
                                        "type": "string",
                                        "enum": sorted(CONFIDENCE_LEVELS),
                                    },
                                },
                            },
                        },
                        "rationale": {"type": "string", "minLength": 1},
                    },
                },
            }
        },
    }


def _response_contract(rubric: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "cohort_version": COHORT_VERSION,
        "reviewer_prompt_version": PROMPT_VERSION,
        "rubric_id": rubric["rubric_id"],
        "rubric_version": rubric["version"],
        "purpose": "Blind fourth review of four unresolved Journal Entries.",
        "rules": REVIEW_RULES,
        "labels": ["yes", "no"],
        "reason_codes": sorted(REASON_CODES),
        "confidence_levels": sorted(CONFIDENCE_LEVELS),
    }


def _render_prompt(packet: dict[str, Any]) -> str:
    return (
        "You are the independent fourth reviewer for four unresolved Twinkl "
        "development cases. Review only the material embedded below. Do not "
        "infer identities, data splits, prior reviewer answers, model outputs, "
        "or expected outcomes.\n\n"
        "A Conflict is one Journal Entry that clearly shows the writer's own "
        "behavior or choice against the declared Core Value. Apply every review "
        "instruction and the embedded versioned rubric exactly, using each case's "
        "Core Value definition and core motivation. Assess only positions marked "
        "disputed; the other Journal Entries are context and are frozen. Make a "
        "forced best-supported yes/no decision and express genuine closeness "
        "through confidence.\n\n"
        "Return the structured JSON requested by the supplied schema. Ground each "
        "rationale in the displayed text.\n\n"
        "REVIEW PACKET\n"
        f"{json.dumps(packet, indent=2, ensure_ascii=False)}\n"
    )


def prepare(args: argparse.Namespace) -> None:
    source = _rooted(args.source)
    output = _rooted(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen output: {output}")
    paths = _source_paths(source)
    _verify_source(paths)
    rubric_path = _rooted(RUBRIC_PATH)
    values_path = _rooted(VALUES_PATH)
    rubric = _read_yaml(rubric_path)
    values = _read_yaml(values_path)
    packet, key, entries, outcomes = _build_packet(paths, rubric, values)
    output.mkdir(parents=True)

    packet_path = output / "reviewer_packet.json"
    key_path = output / "parent_reconciliation_key.json"
    contract_path = output / "response_contract.json"
    schema_path = output / "structured_output_schema.json"
    prompt_path = output / "review_prompt.md"
    packet_path.write_text(
        json.dumps(packet, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    contract_path.write_text(
        json.dumps(_response_contract(rubric), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    schema_path.write_text(
        json.dumps(_structured_output_schema(packet), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    prompt_path.write_text(_render_prompt(packet), encoding="utf-8")

    source_files = [*paths.values(), rubric_path, values_path]
    frozen_files = [packet_path, key_path, contract_path, schema_path, prompt_path]
    manifest = {
        "schema_version": 1,
        "review_schema_version": SCHEMA_VERSION,
        "reviewer_prompt_version": PROMPT_VERSION,
        "cohort_version": COHORT_VERSION,
        "rubric_id": rubric["rubric_id"],
        "rubric_version": rubric["version"],
        "created_at": datetime.now(UTC).isoformat(),
        "case_count": outcomes.filter(
            pl.col("case_resolution") == "unresolved"
        ).height,
        "disputed_entry_count": entries.filter(
            pl.col("resolution_status") == "unresolved"
        ).height,
        "source_files": artifact_hashes(source_files, root=ROOT),
        "frozen_files": artifact_hashes(frozen_files, root=ROOT),
        "packet_sha256": sha256_file(packet_path),
        "structured_output_schema_sha256": sha256_file(schema_path),
        "claude_command": (
            "claude -p --model opus --effort high --tools '' "
            "--safe-mode --disable-slash-commands --no-session-persistence "
            "--output-format json --json-schema <schema>"
        ),
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared {manifest['case_count']} blind cases / "
        f"{manifest['disputed_entry_count']} disputed entries at "
        f"{_manifest_path(output)}"
    )


def _extract_structured_output(payload: dict[str, Any]) -> dict[str, Any]:
    structured = payload.get("structured_output")
    if isinstance(structured, dict):
        return structured
    result = payload.get("result")
    if isinstance(result, str):
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Claude output does not contain structured JSON")


def _validate_labels(
    structured: dict[str, Any], packet: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    packet_map = {case["review_case_id"]: case for case in packet["cases"]}
    cases = structured.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Structured output cases must be a list")
    result = {}
    for case in cases:
        case_id = case.get("review_case_id")
        if case_id not in packet_map or case_id in result:
            raise ValueError(f"Invalid or duplicate review case: {case_id}")
        rationale = case.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"Missing rationale for {case_id}")
        decisions = case.get("entry_adjudications")
        if not isinstance(decisions, list):
            raise ValueError(f"Missing entry adjudications for {case_id}")
        expected = set(packet_map[case_id]["disputed_positions"])
        observed = {decision.get("position") for decision in decisions}
        if observed != expected or len(decisions) != len(expected):
            raise ValueError(f"Incomplete disputed-position coverage for {case_id}")
        for decision in decisions:
            disposition = decision.get("observable_conflict")
            reason = decision.get("reason_code")
            confidence = decision.get("confidence")
            if disposition not in {"yes", "no"}:
                raise ValueError(f"Non-definitive label for {case_id}")
            if reason not in REASON_CODES:
                raise ValueError(f"Invalid reason code for {case_id}")
            if confidence not in CONFIDENCE_LEVELS:
                raise ValueError(f"Invalid confidence for {case_id}")
            if disposition == "yes" and reason != "direct_behavior_or_choice":
                raise ValueError(f"Yes requires direct behavior for {case_id}")
        result[case_id] = case
    if set(result) != set(packet_map):
        raise ValueError("Claude output does not cover all four cases")
    return result


def _opus_models(payload: dict[str, Any]) -> list[str]:
    usage = payload.get("modelUsage") or payload.get("model_usage") or {}
    models = sorted(str(model) for model in usage) if isinstance(usage, dict) else []
    opus_models = [model for model in models if "opus" in model.lower()]
    if not opus_models:
        raise ValueError(f"Claude output does not prove Opus use: {models}")
    return opus_models


def finalize(args: argparse.Namespace) -> None:
    source = _rooted(args.source)
    output = _rooted(args.output)
    manifest = _read_json(output / "audit_manifest.json")
    _verify_hashes(manifest["source_files"], label="Frozen source")
    _verify_hashes(manifest["frozen_files"], label="Frozen review input")

    packet = _read_json(output / "reviewer_packet.json")
    key = _read_json(output / "parent_reconciliation_key.json")
    claude_path = output / "opus_claude_output.json"
    claude_payload = _read_json(claude_path)
    if claude_payload.get("is_error") is True:
        raise ValueError("Claude reported an error")
    models = _opus_models(claude_payload)
    labels = _validate_labels(_extract_structured_output(claude_payload), packet)
    key_map = {case["review_case_id"]: case for case in key["cases"]}

    decisions = {}
    for review_case_id, case in labels.items():
        canonical_case_id = key_map[review_case_id]["canonical_case_id"]
        for decision in case["entry_adjudications"]:
            coordinate = (canonical_case_id, int(decision["position"]))
            decisions[coordinate] = decision

    source_paths = _source_paths(source)
    entries = pl.read_parquet(source_paths["entry_target"])
    response_sha256 = sha256_file(claude_path)
    rows = []
    used = set()
    for row in entries.to_dicts():
        coordinate = (str(row["canonical_case_id"]), int(row["position"]))
        decision = decisions.get(coordinate)
        updated = {
            **row,
            "opus_adjudicator_disposition": None,
            "opus_adjudicator_reason": None,
            "opus_adjudicator_confidence": None,
            "opus_response_sha256": None,
        }
        if decision is not None:
            if row["resolution_status"] != "unresolved":
                raise ValueError(f"Opus attempted to relabel agreement {coordinate}")
            used.add(coordinate)
            disposition = decision["observable_conflict"]
            updated.update(
                {
                    "opus_adjudicator_disposition": disposition,
                    "opus_adjudicator_reason": decision["reason_code"],
                    "opus_adjudicator_confidence": decision["confidence"],
                    "opus_response_sha256": response_sha256,
                    "final_conflict": disposition == "yes",
                    "resolution_method": "opus_adjudication",
                    "resolution_status": "resolved",
                }
            )
        rows.append(updated)
    if used != set(decisions):
        raise ValueError("Opus output contains unknown entry coordinates")
    resolved_entries = pl.DataFrame(rows, infer_schema_length=None).sort(
        "canonical_case_id", "position"
    )

    selected = pl.read_parquet(source_paths["selected_cases"])
    cohort_manifest = _read_json(source_paths["cohort_manifest"])
    outcomes, episodes = derive_review_outcomes(
        resolved_entries,
        selected,
        cohort_sha256=cohort_manifest["cohort_sha256"],
    )
    summary = summarize_outcomes(outcomes)
    drift_trajectories = episodes.select(
        ["persona_id", "dimension"]
    ).unique().height
    summary.update(
        {
            "cohort_sha256": cohort_manifest["cohort_sha256"],
            "reviewed_drift_count": episodes.height,
            "reviewed_drift_trajectory_count": drift_trajectories,
            "all_known_development_union": {
                "trajectory_count": outcomes.height + 2,
                "resolved_trajectory_count": summary["resolved_case_count"] + 2,
                "drift_count": episodes.height + 2,
                "drift_trajectory_count": drift_trajectories + 2,
            },
        }
    )

    normalized_labels = {
        "schema_version": SCHEMA_VERSION,
        "reviewer_prompt_version": PROMPT_VERSION,
        "adjudicator_runtime": models,
        "adjudicated_at": datetime.now(UTC).isoformat(),
        "packet_sha256": manifest["packet_sha256"],
        "structured_output_schema_sha256": manifest[
            "structured_output_schema_sha256"
        ],
        "claude_output_sha256": response_sha256,
        "cases": list(labels.values()),
    }

    results = output / "results"
    results.mkdir()
    labels_path = results / "opus_labels.json"
    entries_path = results / "entry_target_opus_resolved.parquet"
    outcomes_path = results / "case_outcomes_opus_resolved.parquet"
    episodes_path = results / "drift_episodes_opus_resolved.parquet"
    summary_path = results / "summary.json"
    labels_path.write_text(
        json.dumps(normalized_labels, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved_entries.write_parquet(entries_path)
    outcomes.write_parquet(outcomes_path)
    episodes.write_parquet(episodes_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": 1,
        "review_schema_version": SCHEMA_VERSION,
        "completed_at": datetime.now(UTC).isoformat(),
        "claude_output": artifact_hashes([claude_path], root=ROOT),
        "results": artifact_hashes(
            [labels_path, entries_path, outcomes_path, episodes_path, summary_path],
            root=ROOT,
        ),
    }
    (results / "audit_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    union = summary["all_known_development_union"]
    print(
        f"Resolved {summary['resolved_case_count']}/{summary['case_count']} "
        f"reviewed trajectories; union={union['trajectory_count']} total / "
        f"{union['resolved_trajectory_count']} resolved / "
        f"{union['drift_count']} Drifts / "
        f"{union['drift_trajectory_count']} Drift trajectories"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("finalize")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
