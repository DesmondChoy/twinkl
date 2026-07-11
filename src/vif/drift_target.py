"""Student-visible drift-target construction and paired-review helpers.

The legacy consensus-derived frozen benchmark was retired because it was not a
fair promotion yardstick for the information available to the VIF Critic. This
module deliberately does not know its old artifact path. It builds a separate
target delta from paired evidence-only reviews, then overlays that delta onto a
copy of the original consensus label table.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.models.judge import SCHWARTZ_VALUE_ORDER
from src.vif.drift_benchmark import normalize_value_name
from src.vif.state_encoder import concatenate_entry_text

TARGET_VERSION = "twinkl-v8pb-student-visible-v1"
REVIEW_SCHEMA_VERSION = "twinkl-v8pb-review-v2"
REVIEW_PROMPT_VERSION = "twinkl-v8pb-packet-only-v2"

ENTRY_DECISIONS = frozenset({"yes", "no", "uncertain"})
REASON_CODES = frozenset(
    {
        "direct_behavior_or_choice",
        "direct_aligned_or_neutral_behavior",
        "feeling_or_intent_only",
        "external_constraint",
        "needs_hidden_context",
        "ambiguous",
        "missing_text",
    }
)
CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
DELIVERY_STATES = frozenset({"active", "recovered", "none", "uncertain"})

REVIEW_RESPONSE_SCHEMA = {
    "schema_version": REVIEW_SCHEMA_VERSION,
    "purpose": (
        "Paired evidence-only review of whether displayed entries show directly "
        "observable behavior or choice against the declared core value."
    ),
    "case_fields": {
        "review_case_id": "Opaque identifier from blind_packet.json.",
        "entry_assessments": (
            "One assessment for every displayed position, in any order."
        ),
        "sustained_conflict": "yes, no, or uncertain.",
        "delivery_state": "active, recovered, none, or uncertain.",
        "rationale": "Short evidence-based explanation using only displayed text.",
    },
    "entry_assessment_fields": {
        "position": "One-based position from the reviewer packet.",
        "observable_negative": "yes, no, or uncertain.",
        "reason_code": sorted(REASON_CODES),
        "confidence": sorted(CONFIDENCE_LEVELS),
    },
    "submission_fields": {
        "target_version": TARGET_VERSION,
        "split": "The exact split named by blind_packet.json.",
        "packet_sha256": "SHA-256 of the exact blind_packet.json reviewed.",
        "reviewer_id": "Distinct reviewer identifier.",
        "reviewed_at": "Timezone-aware ISO-8601 timestamp.",
        "reviewer_prompt_version": REVIEW_PROMPT_VERSION,
        "reviewer_runtime": "Model or runtime identifier when available.",
    },
    "rules": [
        (
            "Mark yes only for the writer's directly observable behavior or choice "
            "against the declared core value."
        ),
        (
            "Do not infer a negative label from a feeling, wish, external constraint, "
            "biography, or information missing from the displayed entries."
        ),
        (
            "A sustained conflict requires two immediately adjacent displayed entries "
            "that both receive yes."
        ),
        "A no or uncertain entry breaks a possible run.",
    ],
}

DELTA_SCHEMA = {
    "target_version": pl.String,
    "split": pl.String,
    "review_case_id": pl.String,
    "persona_id": pl.String,
    "t_index": pl.Int64,
    "date": pl.String,
    "dimension": pl.String,
    "legacy_label": pl.Int64,
    "student_visible_label": pl.Int64,
    "entry_disposition": pl.String,
    "reason_code": pl.String,
    "confidence": pl.String,
    "reviewer_a_label": pl.String,
    "reviewer_b_label": pl.String,
    "reconciliation_status": pl.String,
    "sustained_conflict_a": pl.String,
    "sustained_conflict_b": pl.String,
    "qualification_agreement": pl.Boolean,
    "delivery_state_a": pl.String,
    "delivery_state_b": pl.String,
    "packet_sha256": pl.String,
    "response_schema_version": pl.String,
}


@dataclass(frozen=True)
class TargetSplit:
    """Disjoint source populations for target development and promotion."""

    training_persona_ids: tuple[str, ...]
    development_persona_ids: tuple[str, ...]
    retired_persona_ids: tuple[str, ...]
    promotion_persona_ids: tuple[str, ...]


def sha256_file(path: str | Path) -> str:
    """Return a stable digest for a local source or generated artifact."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(payload: Any) -> str:
    """Return a stable digest for a JSON-compatible structured payload."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def target_split_sha256(split: TargetSplit) -> str:
    """Fingerprint the exact disjoint populations used by the target."""
    return sha256_json(
        {
            "training_persona_ids": split.training_persona_ids,
            "development_persona_ids": split.development_persona_ids,
            "retired_persona_ids": split.retired_persona_ids,
            "promotion_persona_ids": split.promotion_persona_ids,
        }
    )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _normalised_core_values(raw_values: Any) -> tuple[str, ...]:
    if isinstance(raw_values, str):
        raw_values = [part.strip() for part in raw_values.split(",")]
    if not isinstance(raw_values, (list, tuple)):
        raise ValueError("core_values must be a list or tuple")
    values = tuple(
        dict.fromkeys(
            normalize_value_name(str(value))
            for value in raw_values
            if str(value).strip()
        )
    )
    invalid = sorted(set(values) - set(SCHWARTZ_VALUE_ORDER))
    if invalid:
        raise ValueError("Invalid core value(s): " + ", ".join(invalid))
    if not values:
        raise ValueError("A target case must have at least one declared core value")
    return values


def derive_target_split(
    registry_df: pl.DataFrame,
    original_holdout_manifest: str | Path,
) -> TargetSplit:
    """Derive v8pb populations from the original fixed 180-person manifest.

    The post-manifest registry population is fresh for ``run_020``. It is the
    only promotion population this task permits for that checkpoint.
    """
    if "persona_id" not in registry_df.columns:
        raise ValueError("Registry is missing persona_id")
    if registry_df["persona_id"].n_unique() != registry_df.height:
        raise ValueError("Registry contains duplicate persona_id rows")

    payload = _read_yaml(original_holdout_manifest)
    groups = {}
    for key in ("train_persona_ids", "val_persona_ids", "test_persona_ids"):
        raw = payload.get(key)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"Original holdout manifest has no non-empty {key}")
        groups[key] = {str(persona_id) for persona_id in raw}

    original_ids = set().union(*groups.values())
    expected_source_count = payload.get("source_persona_count")
    if expected_source_count is not None and len(original_ids) != int(
        expected_source_count
    ):
        raise ValueError(
            "Original holdout manifest persona count does not match "
            "source_persona_count"
        )
    if sum(len(group) for group in groups.values()) != len(original_ids):
        raise ValueError("Original train, validation, and test personas overlap")

    registry_ids = {
        str(persona_id) for persona_id in registry_df["persona_id"].to_list()
    }
    missing = sorted(original_ids - registry_ids)
    if missing:
        raise ValueError(
            "Original holdout manifest refers to registry personas that are missing: "
            + ", ".join(missing[:5])
        )

    promotion_ids = registry_ids - original_ids
    if not promotion_ids:
        raise ValueError("No post-manifest promotion personas are available")
    return TargetSplit(
        training_persona_ids=tuple(sorted(groups["train_persona_ids"])),
        development_persona_ids=tuple(sorted(groups["val_persona_ids"])),
        retired_persona_ids=tuple(sorted(groups["test_persona_ids"])),
        promotion_persona_ids=tuple(sorted(promotion_ids)),
    )


def validate_target_manifest(
    manifest_path: str | Path,
    registry_df: pl.DataFrame,
    entries_df: pl.DataFrame,
) -> tuple[dict[str, Any], TargetSplit]:
    """Load and fail closed on a malformed or drifting v8pb target manifest."""
    manifest = _read_yaml(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Target manifest must use schema_version: 1")
    if manifest.get("target_version") != TARGET_VERSION:
        raise ValueError(f"Target manifest must name {TARGET_VERSION}")
    if not manifest.get("promotion_locked_before_review"):
        raise ValueError("Promotion population must be locked before review")
    if not manifest.get("promotion_locked_before_threshold_selection"):
        raise ValueError(
            "Promotion population must be locked before threshold selection"
        )

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("Target manifest must contain a source mapping")
    original_holdout = source.get("original_holdout_manifest")
    if not isinstance(original_holdout, str) or not original_holdout:
        raise ValueError("Target manifest must name original_holdout_manifest")

    root = Path(manifest_path).resolve().parents[2]
    split = derive_target_split(registry_df, root / original_holdout)
    locked_promotion_ids = manifest.get("locked_promotion_persona_ids")
    if not isinstance(locked_promotion_ids, list) or not locked_promotion_ids:
        raise ValueError("Target manifest must lock exact promotion persona IDs")
    locked_promotion = tuple(
        sorted(str(persona_id) for persona_id in locked_promotion_ids)
    )
    if len(set(locked_promotion)) != len(locked_promotion):
        raise ValueError("Target manifest locks duplicate promotion persona IDs")
    if locked_promotion != split.promotion_persona_ids:
        raise ValueError(
            "Live post-manifest promotion population does not match the locked "
            "promotion persona IDs"
        )
    expected = manifest.get("expected_counts") or {}
    expected_values = {
        "development_personas": len(split.development_persona_ids),
        "retired_personas": len(split.retired_persona_ids),
        "promotion_personas": len(split.promotion_persona_ids),
        "promotion_entries": entries_df.filter(
            pl.col("persona_id").is_in(list(split.promotion_persona_ids))
        ).height,
    }
    for key, actual in expected_values.items():
        if expected.get(key) != actual:
            raise ValueError(
                f"Target manifest expected_counts.{key}={expected.get(key)!r}; "
                f"live value is {actual}"
            )

    if set(split.retired_persona_ids) & set(split.development_persona_ids):
        raise ValueError("Retired frozen personas overlap target development")
    if set(split.retired_persona_ids) & set(split.promotion_persona_ids):
        raise ValueError("Retired frozen personas overlap target promotion")
    return manifest, split


def build_full_trajectory_cases(
    entries_df: pl.DataFrame,
    registry_df: pl.DataFrame,
    persona_ids: tuple[str, ...] | list[str],
    *,
    split: str,
) -> list[dict[str, Any]]:
    """Build one full-timeline review case for every persona/core-value pair."""
    required_entry_columns = {"persona_id", "t_index", "date", "initial_entry"}
    required_registry_columns = {"persona_id", "core_values"}
    missing_entries = required_entry_columns - set(entries_df.columns)
    missing_registry = required_registry_columns - set(registry_df.columns)
    if missing_entries:
        columns = ", ".join(sorted(missing_entries))
        raise ValueError(f"Entries are missing required columns: {columns}")
    if missing_registry:
        columns = ", ".join(sorted(missing_registry))
        raise ValueError(f"Registry is missing required columns: {columns}")

    selected_ids = tuple(str(persona_id) for persona_id in persona_ids)
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError(f"{split} persona ids contain duplicates")
    selected_registry_rows = registry_df.filter(
        pl.col("persona_id").is_in(list(selected_ids))
    )
    registry_rows = {
        str(row["persona_id"]): row for row in selected_registry_rows.to_dicts()
    }
    if set(registry_rows) != set(selected_ids):
        missing = sorted(set(selected_ids) - set(registry_rows))
        raise ValueError(f"{split} personas missing from registry: {missing[:5]}")

    cases = []
    for persona_id in selected_ids:
        rows = (
            entries_df.filter(pl.col("persona_id") == persona_id)
            .sort("t_index", "date")
            .select(
                "t_index",
                "date",
                "initial_entry",
                *[
                    column
                    for column in ("nudge_text", "response_text")
                    if column in entries_df.columns
                ],
            )
            .to_dicts()
        )
        if not rows:
            raise ValueError(f"{split} persona {persona_id} has no journal entries")
        expected_indices = list(range(len(rows)))
        actual_indices = [int(row["t_index"]) for row in rows]
        if actual_indices != expected_indices:
            raise ValueError(
                f"{split} persona {persona_id} has non-contiguous t_index values: "
                f"{actual_indices}"
            )
        for row in rows:
            text = row.get("initial_entry")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"{split} persona {persona_id} has missing journal text at "
                    f"t_index={row['t_index']}"
                )

        declared_core_values = registry_rows[persona_id]["core_values"]
        for dimension in _normalised_core_values(declared_core_values):
            cases.append(
                {
                    "case_id": f"{split}:{persona_id}:{dimension}",
                    "split": split,
                    "persona_id": persona_id,
                    "dimension": dimension,
                    "entries": [
                        {
                            "t_index": int(row["t_index"]),
                            "date": str(row["date"]),
                            "initial_entry": str(row["initial_entry"]).strip(),
                            "nudge_text": row.get("nudge_text"),
                            "response_text": row.get("response_text"),
                        }
                        for row in rows
                    ],
                }
            )
    return cases


def build_blind_packet(
    cases: list[dict[str, Any]],
    *,
    split: str,
    target_version: str = TARGET_VERSION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a deterministic packet with labels, sources, and IDs removed."""
    if not cases:
        raise ValueError("Cannot build a review packet with no cases")
    case_ids = [str(case.get("case_id")) for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Review cases have duplicate case_id values")
    if any(case.get("split") != split for case in cases):
        raise ValueError("Review cases do not all belong to the requested split")

    shuffled = list(cases)
    seed_text = f"{target_version}:{split}:full-trajectory-review"
    random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)).shuffle(
        shuffled
    )

    packet_cases = []
    key_cases = []
    for index, case in enumerate(shuffled, start=1):
        review_case_id = f"case_{index:03d}"
        entries = case.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{case.get('case_id')} has no entries")
        packet_cases.append(
            {
                "review_case_id": review_case_id,
                "declared_core_value": str(case["dimension"]).replace("_", " ").title(),
                "entries": [
                    {
                        "position": position,
                        "journal_entry": concatenate_entry_text(
                            entry.get("initial_entry"),
                            entry.get("nudge_text"),
                            entry.get("response_text"),
                        ),
                    }
                    for position, entry in enumerate(entries, start=1)
                ],
            }
        )
        key_cases.append(
            {
                "review_case_id": review_case_id,
                "case_id": case["case_id"],
                "split": split,
                "persona_id": case["persona_id"],
                "dimension": case["dimension"],
                "entries": [
                    {
                        "position": position,
                        "t_index": int(entry["t_index"]),
                        "date": str(entry["date"]),
                    }
                    for position, entry in enumerate(entries, start=1)
                ],
            }
        )

    packet = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "target_version": target_version,
        "split": split,
        "review_instructions": REVIEW_RESPONSE_SCHEMA["rules"],
        "cases": packet_cases,
    }
    reconciliation_key = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "target_version": target_version,
        "warning": (
            "Parent-held reconciliation key. Do not give this file, source labels, "
            "model evidence, or source artifacts to a reviewer before submission."
        ),
        "cases": key_cases,
    }
    return packet, reconciliation_key


def write_review_bundle(
    *,
    output_dir: str | Path,
    root: str | Path,
    source_paths: dict[str, str | Path],
    cases: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    """Write reviewer-visible and parent-control material in separate directories."""
    output = Path(output_dir)
    root_path = Path(root)
    reviewer_packet_dir = output / "reviewer_packet"
    parent_control_dir = output / "parent_control"
    reviewer_packet_dir.mkdir(parents=True, exist_ok=True)
    parent_control_dir.mkdir(parents=True, exist_ok=True)
    packet, key = build_blind_packet(cases, split=split)
    packet_path = reviewer_packet_dir / "blind_packet.json"
    key_path = parent_control_dir / "reconciliation_key.json"
    schema_path = reviewer_packet_dir / "response_schema.json"
    packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    schema_path.write_text(
        json.dumps(REVIEW_RESPONSE_SCHEMA, indent=2) + "\n", encoding="utf-8"
    )

    sources = {}
    for name, source_path in source_paths.items():
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing review-bundle source {name}: {path}")
        sources[name] = {
            "path": _relative_to_root(path, root_path),
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": 1,
        "target_version": TARGET_VERSION,
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "split": split,
        "case_count": len(cases),
        "entry_count": sum(len(case["entries"]) for case in cases),
        "review_cases_sha256": sha256_json(cases),
        "source_inputs": sources,
        "outputs": {
            "reviewer_packet/blind_packet.json": sha256_file(packet_path),
            "parent_control/reconciliation_key.json": sha256_file(key_path),
            "reviewer_packet/response_schema.json": sha256_file(schema_path),
        },
        "review_protocol": {
            "reviewer_input": (
                "reviewer_packet/blind_packet.json and "
                "reviewer_packet/response_schema.json only"
            ),
            "response_schema": REVIEW_SCHEMA_VERSION,
            "reviewer_prompt_version": REVIEW_PROMPT_VERSION,
            "primary_agreement": "case-level sustained_conflict",
            "secondary_agreement": [
                "per-entry observable_negative",
                "delivery_state",
                "confidence",
            ],
            "technical_isolation": False,
            "controlled_disclosure": True,
        },
        "known_limitations": [
            (
                "Codex agents share a workspace. Packet-only disclosure is a "
                "procedural control, not proof of enforced technical isolation."
            ),
            "Paired Codex review is diagnostic AI evidence, not human ground truth.",
        ],
    }
    manifest_path_out = parent_control_dir / "audit_manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_audit_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path,
    packet_path: str | Path,
    key_path: str | Path,
    schema_path: str | Path,
    expected_cases: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    """Verify that a review bundle still has its recorded inputs and outputs."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    if payload.get("schema_version") != 1:
        raise ValueError("Audit manifest has an unsupported schema version")
    if payload.get("target_version") != TARGET_VERSION:
        raise ValueError("Audit manifest is not for the v8pb target")
    if payload.get("review_schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("Audit manifest does not match the review schema")
    if payload.get("split") != split:
        raise ValueError("Audit manifest does not match the requested split")
    if payload.get("case_count") != len(expected_cases):
        raise ValueError("Audit manifest case count does not match live source cases")
    expected_entry_count = sum(len(case["entries"]) for case in expected_cases)
    if payload.get("entry_count") != expected_entry_count:
        raise ValueError("Audit manifest entry count does not match live source cases")
    if payload.get("review_cases_sha256") != sha256_json(expected_cases):
        raise ValueError("Audit manifest review-case source fingerprint has drifted")

    bundle_dir = path.parent.parent
    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Audit manifest has no output hashes")
    expected_outputs = {
        "reviewer_packet/blind_packet.json": Path(packet_path),
        "parent_control/reconciliation_key.json": Path(key_path),
        "reviewer_packet/response_schema.json": Path(schema_path),
    }
    for relative_path, artifact_path in expected_outputs.items():
        if artifact_path.resolve() != (bundle_dir / relative_path).resolve():
            raise ValueError("Audit manifest artifact layout is inconsistent")
        if outputs.get(relative_path) != sha256_file(artifact_path):
            raise ValueError(f"Audit manifest hash mismatch for {relative_path}")

    root_path = Path(root)
    source_inputs = payload.get("source_inputs")
    if not isinstance(source_inputs, dict):
        raise ValueError("Audit manifest has no source-input hashes")
    for name, record in source_inputs.items():
        if not isinstance(record, dict):
            raise ValueError(f"Audit manifest source record is invalid for {name}")
        source_path = Path(record.get("path", ""))
        if not source_path.is_absolute():
            source_path = root_path / source_path
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Audit manifest source input is missing for {name}: {source_path}"
            )
        if record.get("sha256") != sha256_file(source_path):
            raise ValueError(f"Audit manifest source-input hash mismatch for {name}")
    return payload


def record_promotion_threshold_receipt(
    manifest_path: str | Path,
    *,
    root: str | Path,
    threshold_path: str | Path,
) -> dict[str, Any]:
    """Bind the frozen development threshold into promotion control before review."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("split") != "promotion":
        raise ValueError(
            "Promotion threshold receipt requires a promotion audit manifest"
        )
    threshold = Path(threshold_path)
    if not threshold.is_file():
        raise FileNotFoundError(f"Frozen threshold receipt is missing: {threshold}")
    receipt = {
        "threshold_path": _relative_to_root(threshold, Path(root)),
        "threshold_sha256": sha256_file(threshold),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    existing = payload.get("threshold_receipt")
    if isinstance(existing, dict):
        if {
            "threshold_path": existing.get("threshold_path"),
            "threshold_sha256": existing.get("threshold_sha256"),
        } != {
            "threshold_path": receipt["threshold_path"],
            "threshold_sha256": receipt["threshold_sha256"],
        }:
            raise ValueError(
                "Promotion audit manifest already has a different threshold receipt"
            )
        return existing
    payload["threshold_receipt"] = receipt
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def validate_promotion_threshold_receipt(
    manifest: dict[str, Any],
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Require the pre-review threshold receipt to still point at the same file."""
    receipt = manifest.get("threshold_receipt")
    if not isinstance(receipt, dict):
        raise ValueError(
            "Promotion audit manifest lacks a pre-review threshold receipt"
        )
    threshold_path = Path(receipt.get("threshold_path", ""))
    if not threshold_path.is_absolute():
        threshold_path = Path(root) / threshold_path
    if not threshold_path.is_file():
        raise FileNotFoundError("Promotion threshold receipt points to a missing file")
    if receipt.get("threshold_sha256") != sha256_file(threshold_path):
        raise ValueError("Promotion threshold receipt hash does not match its file")
    parse_aware_timestamp(receipt.get("recorded_at"), field_name="threshold receipt")
    return receipt


def finalize_audit_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path,
    reviewer_a_path: str | Path,
    reviewer_a: dict[str, Any],
    reviewer_b_path: str | Path,
    reviewer_b: dict[str, Any],
) -> dict[str, Any]:
    """Add paired-review submission provenance after both reviews are received."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    root_path = Path(root)

    def submission(
        name: str,
        response_path: str | Path,
        response: dict[str, Any],
    ) -> dict:
        runtime = response.get("reviewer_runtime")
        prompt_version = response.get("reviewer_prompt_version")
        return {
            "role": name,
            "response_path": _relative_to_root(Path(response_path), root_path),
            "sha256": sha256_file(response_path),
            "reviewer_id": response["reviewer_id"],
            "reviewed_at": response["reviewed_at"],
            "reviewer_runtime": runtime if isinstance(runtime, str) else "not recorded",
            "reviewer_prompt_version": (
                prompt_version
                if isinstance(prompt_version, str)
                else REVIEW_PROMPT_VERSION
            ),
        }

    payload["assessment_submissions"] = {
        "reviewer_a": submission("reviewer_a", reviewer_a_path, reviewer_a),
        "reviewer_b": submission("reviewer_b", reviewer_b_path, reviewer_b),
    }
    payload["assessments_recorded_at"] = datetime.now(UTC).isoformat()
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_aware_timestamp(value: Any, *, field_name: str) -> datetime:
    """Parse a timestamp and reject values without an explicit timezone."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} has an invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def _review_cases_by_id(
    response: dict[str, Any],
    *,
    expected_packet_sha256: str,
    split: str,
) -> dict[str, dict[str, Any]]:
    if response.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError(
            f"Review response must use schema_version {REVIEW_SCHEMA_VERSION}"
        )
    if response.get("target_version") != TARGET_VERSION:
        raise ValueError("Review response is not for the v8pb target")
    if response.get("split") != split:
        raise ValueError("Review response does not match the requested split")
    if response.get("packet_sha256") != expected_packet_sha256:
        raise ValueError("Review response does not match the reviewed packet")
    if response.get("reviewer_prompt_version") != REVIEW_PROMPT_VERSION:
        raise ValueError("Review response does not match the required reviewer prompt")
    if not isinstance(response.get("reviewer_id"), str) or not response["reviewer_id"]:
        raise ValueError("Review response must name reviewer_id")
    parse_aware_timestamp(response.get("reviewed_at"), field_name="reviewed_at")
    if "cases" in response and "case_responses" in response:
        raise ValueError("Review response must use only one case-response field")
    cases = response.get("cases", response.get("case_responses"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("Review response must contain non-empty cases")
    result = {}
    for case in cases:
        review_case_id = case.get("review_case_id")
        if not isinstance(review_case_id, str) or not review_case_id:
            raise ValueError("Review response case is missing review_case_id")
        if review_case_id in result:
            raise ValueError(f"Review response duplicates case {review_case_id}")
        if case.get("sustained_conflict") not in ENTRY_DECISIONS:
            raise ValueError(f"Invalid sustained_conflict for {review_case_id}")
        if case.get("delivery_state") not in DELIVERY_STATES:
            raise ValueError(f"Invalid delivery_state for {review_case_id}")
        if not isinstance(case.get("rationale"), str) or not case["rationale"].strip():
            raise ValueError(f"Review response lacks rationale for {review_case_id}")
        assessments = case.get("entry_assessments")
        if not isinstance(assessments, list) or not assessments:
            raise ValueError(
                f"Review response lacks entry_assessments for {review_case_id}"
            )
        seen_positions = set()
        for assessment in assessments:
            position = assessment.get("position")
            if not isinstance(position, int) or position < 1:
                raise ValueError(f"Invalid entry position for {review_case_id}")
            if position in seen_positions:
                raise ValueError(
                    "Review response duplicates position "
                    f"{position} for {review_case_id}"
                )
            seen_positions.add(position)
            if assessment.get("observable_negative") not in ENTRY_DECISIONS:
                raise ValueError(
                    f"Invalid observable_negative for {review_case_id}:{position}"
                )
            if assessment.get("reason_code") not in REASON_CODES:
                raise ValueError(f"Invalid reason_code for {review_case_id}:{position}")
            if assessment.get("confidence") not in CONFIDENCE_LEVELS:
                raise ValueError(f"Invalid confidence for {review_case_id}:{position}")
            if (
                assessment["observable_negative"] == "yes"
                and assessment["reason_code"] != "direct_behavior_or_choice"
            ):
                raise ValueError(
                    "An observable_negative yes must use "
                    f"direct_behavior_or_choice for {review_case_id}:{position}"
                )
        ordered_assessments = sorted(
            assessments,
            key=lambda assessment: assessment["position"],
        )
        negative_positions = {
            assessment["position"]
            for assessment in ordered_assessments
            if assessment["observable_negative"] == "yes"
        }
        has_sustained_pair = any(
            position + 1 in negative_positions for position in negative_positions
        )
        decision = case["sustained_conflict"]
        if decision == "yes" and not has_sustained_pair:
            raise ValueError(
                f"Review response says sustained_conflict yes without an adjacent "
                f"negative pair for {review_case_id}"
            )
        if decision != "yes" and has_sustained_pair:
            raise ValueError(
                f"Review response says sustained_conflict {decision} despite an "
                f"adjacent negative pair for {review_case_id}"
            )
        result[review_case_id] = case
    return result


def _packet_case_map(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if packet.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("Blind packet has an unsupported schema version")
    if packet.get("target_version") != TARGET_VERSION:
        raise ValueError("Blind packet is not for the v8pb target")
    cases = packet.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Blind packet has no cases")
    result = {}
    for case in cases:
        review_case_id = case.get("review_case_id")
        if not isinstance(review_case_id, str) or review_case_id in result:
            raise ValueError("Blind packet has invalid or duplicate review_case_id")
        entries = case.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Blind packet case {review_case_id} has no entries")
        positions = [entry.get("position") for entry in entries]
        if positions != list(range(1, len(entries) + 1)):
            raise ValueError(
                f"Blind packet case {review_case_id} has invalid positions"
            )
        result[review_case_id] = case
    return result


def _key_case_map(key: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if key.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("Reconciliation key has an unsupported schema version")
    if key.get("target_version") != TARGET_VERSION:
        raise ValueError("Reconciliation key is not for the v8pb target")
    cases = key.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Reconciliation key has no cases")
    result = {}
    for case in cases:
        review_case_id = case.get("review_case_id")
        if not isinstance(review_case_id, str) or review_case_id in result:
            raise ValueError(
                "Reconciliation key has invalid or duplicate review_case_id"
            )
        dimension = case.get("dimension")
        if dimension not in SCHWARTZ_VALUE_ORDER:
            raise ValueError(f"Reconciliation key has invalid dimension {dimension!r}")
        entries = case.get("entries")
        positions = [entry.get("position") for entry in entries or []]
        if positions != list(range(1, len(positions) + 1)):
            raise ValueError(
                f"Reconciliation key case {review_case_id} has invalid positions"
            )
        result[review_case_id] = case
    return result


def validate_review_bundle_cases(
    *,
    packet: dict[str, Any],
    reconciliation_key: dict[str, Any],
    expected_cases: list[dict[str, Any]],
    split: str,
) -> None:
    """Require a reviewed bundle to cover every locked persona/value trajectory."""
    if packet.get("split") != split:
        raise ValueError("Blind packet does not match the requested split")
    packet_cases = _packet_case_map(packet)
    key_cases = _key_case_map(reconciliation_key)
    if set(packet_cases) != set(key_cases):
        raise ValueError("Blind packet and reconciliation key have different cases")
    expected_by_case_id = {str(case["case_id"]): case for case in expected_cases}
    if len(expected_by_case_id) != len(expected_cases):
        raise ValueError("Expected review cases have duplicate case IDs")
    observed_by_case_id = {
        str(case.get("case_id")): case for case in key_cases.values()
    }
    if set(observed_by_case_id) != set(expected_by_case_id):
        raise ValueError("Review bundle does not cover every expected full trajectory")

    for review_case_id, key_case in key_cases.items():
        case_id = str(key_case["case_id"])
        expected = expected_by_case_id[case_id]
        if key_case.get("split") != split:
            raise ValueError(f"Reconciliation key has a non-{split} case")
        if str(key_case.get("persona_id")) != str(expected["persona_id"]):
            raise ValueError(
                f"Reconciliation key persona mismatch for {review_case_id}"
            )
        if str(key_case.get("dimension")) != str(expected["dimension"]):
            raise ValueError(
                f"Reconciliation key dimension mismatch for {review_case_id}"
            )
        expected_coordinates = [
            (int(entry["t_index"]), str(entry["date"])) for entry in expected["entries"]
        ]
        observed_coordinates = [
            (int(entry["t_index"]), str(entry["date"])) for entry in key_case["entries"]
        ]
        if observed_coordinates != expected_coordinates:
            raise ValueError(
                f"Reconciliation key entry coordinates mismatch for {review_case_id}"
            )
        packet_case = packet_cases[review_case_id]
        declared_value = normalize_value_name(
            str(packet_case.get("declared_core_value", ""))
        )
        if declared_value != expected["dimension"]:
            raise ValueError(f"Blind packet value mismatch for {review_case_id}")
        packet_positions = [entry.get("position") for entry in packet_case["entries"]]
        expected_positions = list(range(1, len(expected_coordinates) + 1))
        if packet_positions != expected_positions:
            raise ValueError(f"Blind packet positions mismatch for {review_case_id}")
        packet_text = [
            str(entry.get("journal_entry", "")).strip()
            for entry in packet_case["entries"]
        ]
        expected_text = [
            concatenate_entry_text(
                entry.get("initial_entry"),
                entry.get("nudge_text"),
                entry.get("response_text"),
            )
            for entry in expected["entries"]
        ]
        if packet_text != expected_text:
            raise ValueError(f"Blind packet journal text mismatch for {review_case_id}")


def _confidence_for_agreement(first: str, second: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    return first if order[first] <= order[second] else second


def build_paired_target_delta(
    *,
    packet: dict[str, Any],
    reconciliation_key: dict[str, Any],
    reviewer_a: dict[str, Any],
    reviewer_b: dict[str, Any],
    base_labels_df: pl.DataFrame,
    registry_df: pl.DataFrame,
    split: str,
    packet_sha256: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Reconcile paired reviews into a non-destructive long-form target delta.

    A disagreement on a case's main sustained-conflict qualification makes every
    entry in that case unresolved. This prevents selectively retaining easy
    entries to improve downstream metrics.
    """
    if packet.get("split") != split:
        raise ValueError("Blind packet does not match the requested split")
    packet_cases = _packet_case_map(packet)
    key_cases = _key_case_map(reconciliation_key)
    if reviewer_a.get("reviewer_id") == reviewer_b.get("reviewer_id"):
        raise ValueError("Paired reviews must use distinct reviewer_id values")
    reviewer_a_cases = _review_cases_by_id(
        reviewer_a,
        expected_packet_sha256=packet_sha256,
        split=split,
    )
    reviewer_b_cases = _review_cases_by_id(
        reviewer_b,
        expected_packet_sha256=packet_sha256,
        split=split,
    )
    expected_case_ids = set(packet_cases)
    for name, actual_case_ids in {
        "reconciliation key": set(key_cases),
        "reviewer_a": set(reviewer_a_cases),
        "reviewer_b": set(reviewer_b_cases),
    }.items():
        if actual_case_ids != expected_case_ids:
            raise ValueError(
                f"{name} case IDs do not match blind packet: "
                f"missing={sorted(expected_case_ids - actual_case_ids)[:5]}, "
                f"unexpected={sorted(actual_case_ids - expected_case_ids)[:5]}"
            )

    registry_core_values = {
        str(row["persona_id"]): _normalised_core_values(row["core_values"])
        for row in registry_df.select("persona_id", "core_values").to_dicts()
    }
    base_lookup = {
        (str(row["persona_id"]), int(row["t_index"])): row
        for row in base_labels_df.to_dicts()
    }
    rows = []
    qualification_agreement_count = 0
    entry_agreement_count = 0
    entry_count = 0
    unresolved_case_ids = []
    for review_case_id in sorted(expected_case_ids):
        packet_case = packet_cases[review_case_id]
        key_case = key_cases[review_case_id]
        first = reviewer_a_cases[review_case_id]
        second = reviewer_b_cases[review_case_id]
        packet_positions = [entry["position"] for entry in packet_case["entries"]]
        first_assessments = {
            entry["position"]: entry for entry in first["entry_assessments"]
        }
        second_assessments = {
            entry["position"]: entry for entry in second["entry_assessments"]
        }
        first_positions = set(first_assessments)
        second_positions = set(second_assessments)
        if first_positions != set(packet_positions) or second_positions != set(
            packet_positions
        ):
            raise ValueError(
                f"Review entry positions do not match blind packet for {review_case_id}"
            )
        key_entries = {entry["position"]: entry for entry in key_case["entries"]}
        if set(key_entries) != set(packet_positions):
            raise ValueError(
                "Reconciliation-key positions do not match blind packet for "
                f"{review_case_id}"
            )

        persona_id = str(key_case["persona_id"])
        dimension = str(key_case["dimension"])
        if dimension not in registry_core_values.get(persona_id, ()):
            raise ValueError(
                f"{review_case_id} targets non-core dimension "
                f"{dimension} for {persona_id}"
            )
        qualification_agreed = (
            first["sustained_conflict"] in {"yes", "no"}
            and first["sustained_conflict"] == second["sustained_conflict"]
        )
        if qualification_agreed:
            qualification_agreement_count += 1
        else:
            unresolved_case_ids.append(review_case_id)

        for position in packet_positions:
            first_entry = first_assessments[position]
            second_entry = second_assessments[position]
            key_entry = key_entries[position]
            coordinate = (persona_id, int(key_entry["t_index"]))
            base_row = base_lookup.get(coordinate)
            if base_row is None:
                raise ValueError(
                    f"{review_case_id}:{position} has no matching base label "
                    f"row {coordinate}"
                )
            if str(base_row["date"]) != str(key_entry["date"]):
                raise ValueError(
                    f"{review_case_id}:{position} date does not match base label row"
                )
            label_col = f"alignment_{dimension}"
            if label_col not in base_row:
                raise ValueError(f"Base labels are missing {label_col}")

            choices_agree = (
                first_entry["observable_negative"]
                == second_entry["observable_negative"]
            )
            entry_count += 1
            if choices_agree:
                entry_agreement_count += 1
            if qualification_agreed and choices_agree:
                choice = first_entry["observable_negative"]
                student_label = {"yes": -1, "no": 0, "uncertain": None}[choice]
                disposition = {
                    "yes": "observable_negative",
                    "no": "not_negative",
                    "uncertain": "uncertain",
                }[choice]
                reason_code = (
                    first_entry["reason_code"]
                    if first_entry["reason_code"] == second_entry["reason_code"]
                    else "ambiguous"
                )
                confidence = _confidence_for_agreement(
                    first_entry["confidence"], second_entry["confidence"]
                )
                status = "agreed"
            else:
                student_label = None
                disposition = "uncertain"
                reason_code = "ambiguous"
                confidence = "low"
                status = "unresolved"
            rows.append(
                {
                    "target_version": TARGET_VERSION,
                    "split": split,
                    "review_case_id": review_case_id,
                    "persona_id": persona_id,
                    "t_index": int(key_entry["t_index"]),
                    "date": str(key_entry["date"]),
                    "dimension": dimension,
                    "legacy_label": base_row[label_col],
                    "student_visible_label": student_label,
                    "entry_disposition": disposition,
                    "reason_code": reason_code,
                    "confidence": confidence,
                    "reviewer_a_label": first_entry["observable_negative"],
                    "reviewer_b_label": second_entry["observable_negative"],
                    "reconciliation_status": status,
                    "sustained_conflict_a": first["sustained_conflict"],
                    "sustained_conflict_b": second["sustained_conflict"],
                    "qualification_agreement": qualification_agreed,
                    "delivery_state_a": first["delivery_state"],
                    "delivery_state_b": second["delivery_state"],
                    "packet_sha256": packet_sha256,
                    "response_schema_version": REVIEW_SCHEMA_VERSION,
                }
            )
    delta = pl.DataFrame(rows, schema=DELTA_SCHEMA, strict=False)
    duplicate_count = (
        delta.group_by("persona_id", "t_index", "dimension")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError("Paired review delta contains duplicate entry coordinates")
    unresolved_entry_count = delta["student_visible_label"].null_count()
    summary = {
        "target_version": TARGET_VERSION,
        "split": split,
        "case_count": len(expected_case_ids),
        "entry_count": entry_count,
        "qualification_agreement_count": qualification_agreement_count,
        "qualification_agreement_rate": qualification_agreement_count
        / len(expected_case_ids),
        "entry_agreement_count": entry_agreement_count,
        "entry_agreement_rate": entry_agreement_count / entry_count,
        "unresolved_case_ids": unresolved_case_ids,
        "unresolved_entry_count": unresolved_entry_count,
        "promotable": split != "promotion"
        or (not unresolved_case_ids and unresolved_entry_count == 0),
        "reviewer_ids": [
            str(reviewer_a["reviewer_id"]),
            str(reviewer_b["reviewer_id"]),
        ],
        "reviewer_submission_timestamps": sorted(
            [str(reviewer_a["reviewed_at"]), str(reviewer_b["reviewed_at"])]
        ),
    }
    return delta.sort("persona_id", "dimension", "t_index"), summary


def apply_student_visible_target(
    base_labels_df: pl.DataFrame,
    target_delta_df: pl.DataFrame,
    registry_df: pl.DataFrame,
    *,
    retired_persona_ids: tuple[str, ...] | list[str],
) -> pl.DataFrame:
    """Overlay a reviewed delta onto a copy of the original wide label table.

    ``base_labels_df`` is immutable from Polars' perspective; the returned frame
    is a separate target variant. A reviewed ``uncertain`` label becomes null and
    carries ``no_majority`` confidence so it breaks, rather than bridges, a run.
    """
    required_delta = set(DELTA_SCHEMA)
    missing_delta = required_delta - set(target_delta_df.columns)
    if missing_delta:
        columns = ", ".join(sorted(missing_delta))
        raise ValueError(f"Target delta is missing required columns: {columns}")
    required_base = {"persona_id", "t_index", "date", "alignment_vector"}
    missing_base = required_base - set(base_labels_df.columns)
    if missing_base:
        columns = ", ".join(sorted(missing_base))
        raise ValueError(f"Base labels are missing required columns: {columns}")
    duplicate_count = (
        target_delta_df.group_by("persona_id", "t_index", "dimension")
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError("Target delta has duplicate (persona_id, t_index, dimension)")
    retired = {str(persona_id) for persona_id in retired_persona_ids}
    if target_delta_df.filter(pl.col("persona_id").is_in(list(retired))).height:
        raise ValueError("Target delta includes a retired frozen-test persona")

    core_values = {
        str(row["persona_id"]): _normalised_core_values(row["core_values"])
        for row in registry_df.select("persona_id", "core_values").to_dicts()
    }
    base_rows = {
        (str(row["persona_id"]), int(row["t_index"])): row
        for row in base_labels_df.to_dicts()
    }
    for row in target_delta_df.to_dicts():
        persona_id = str(row["persona_id"])
        dimension = str(row["dimension"])
        if dimension not in SCHWARTZ_VALUE_ORDER:
            raise ValueError(f"Target delta has invalid dimension {dimension!r}")
        if dimension not in core_values.get(persona_id, ()):
            raise ValueError(
                f"Target delta dimension {dimension} is not declared by {persona_id}"
            )
        if row["student_visible_label"] not in {-1, 0, None}:
            raise ValueError("Target delta has invalid student_visible_label")
        if row["reason_code"] not in REASON_CODES:
            raise ValueError("Target delta has invalid reason_code")
        coordinate = (persona_id, int(row["t_index"]))
        base_row = base_rows.get(coordinate)
        if base_row is None:
            raise ValueError(f"Target delta has no source row for {coordinate}")
        if str(base_row["date"]) != str(row["date"]):
            raise ValueError(f"Target delta date mismatch for {coordinate}")
        label_col = f"alignment_{dimension}"
        if base_row[label_col] != row["legacy_label"]:
            raise ValueError(
                f"Target delta legacy_label mismatch for {coordinate}:{dimension}"
            )

    result = base_labels_df.clone()
    for dimension in SCHWARTZ_VALUE_ORDER:
        updates = target_delta_df.filter(pl.col("dimension") == dimension)
        if updates.is_empty():
            continue
        label_col = f"alignment_{dimension}"
        confidence_col = f"confidence_{dimension}"
        agreement_col = f"consensus_agreement_{dimension}"
        for column in (label_col, confidence_col, agreement_col):
            if column not in result.columns:
                raise ValueError(f"Base labels are missing {column}")
        update_frame = updates.select(
            "persona_id",
            "t_index",
            pl.lit(True).alias("_has_target_delta"),
            pl.col("student_visible_label").cast(pl.Int64).alias("_target_label"),
            pl.col("student_visible_label").is_null().alias("_target_is_uncertain"),
        )
        result = result.join(update_frame, on=["persona_id", "t_index"], how="left")
        result = result.with_columns(
            pl.when(pl.col("_has_target_delta"))
            .then(pl.col("_target_label"))
            .otherwise(pl.col(label_col))
            .cast(pl.Int64)
            .alias(label_col),
            pl.when(pl.col("_has_target_delta"))
            .then(
                pl.when(pl.col("_target_is_uncertain"))
                .then(pl.lit("no_majority"))
                .otherwise(pl.lit("student_visible"))
            )
            .otherwise(pl.col(confidence_col))
            .alias(confidence_col),
            pl.when(pl.col("_has_target_delta"))
            .then(pl.lit(2))
            .otherwise(pl.col(agreement_col))
            .cast(pl.Int64)
            .alias(agreement_col),
        ).drop("_has_target_delta", "_target_label", "_target_is_uncertain")

    return result.with_columns(
        pl.concat_list(
            [pl.col(f"alignment_{dimension}") for dimension in SCHWARTZ_VALUE_ORDER]
        ).alias("alignment_vector")
    )
