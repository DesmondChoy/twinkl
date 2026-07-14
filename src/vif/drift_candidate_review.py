"""Deterministic cohort and review helpers for ``twinkl-752.4``.

Legacy labels only mine candidates and matched controls. The resulting review
target is sparse and binary: directly observable Conflict versus no observable
Conflict for one declared Core Value. It is not a replacement for the ternary
VIF training target.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, milp

from src.vif.drift_target import (
    CONFIDENCE_LEVELS,
    DELIVERY_STATES,
    REASON_CODES,
    TargetSplit,
    parse_aware_timestamp,
    sha256_file,
    sha256_json,
)
from src.vif.state_encoder import concatenate_entry_text

COHORT_VERSION = "twinkl-752.4-legacy-drift-review-v1"
REVIEW_SCHEMA_VERSION = "twinkl-752.4-review-v1"
REVIEW_PROMPT_VERSION = "twinkl-752.4-packet-only-v1"
ADJUDICATION_SCHEMA_VERSION = "twinkl-752.4-adjudication-v1"
ADJUDICATION_PROMPT_VERSION = "twinkl-752.4-disagreement-only-v1"
ENTRY_DECISIONS = frozenset({"yes", "no", "uncertain"})


@dataclass(frozen=True)
class ReviewProtocol:
    """Version identifiers for one blinded Conflict review cohort."""

    cohort_version: str
    review_schema_version: str
    review_prompt_version: str
    adjudication_schema_version: str
    adjudication_prompt_version: str


LEGACY_REVIEW_PROTOCOL = ReviewProtocol(
    cohort_version=COHORT_VERSION,
    review_schema_version=REVIEW_SCHEMA_VERSION,
    review_prompt_version=REVIEW_PROMPT_VERSION,
    adjudication_schema_version=ADJUDICATION_SCHEMA_VERSION,
    adjudication_prompt_version=ADJUDICATION_PROMPT_VERSION,
)

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
    "Mark uncertain when the displayed text cannot support yes or no.",
    (
        "Sustained Conflict is yes exactly when two immediately adjacent entries "
        "are both yes; no or uncertain breaks a run."
    ),
]


def build_review_response_schema(
    protocol: ReviewProtocol = LEGACY_REVIEW_PROTOCOL,
) -> dict[str, Any]:
    """Build the receipt-bound response schema for a review protocol."""
    return {
        "schema_version": protocol.review_schema_version,
        "cohort_version": protocol.cohort_version,
        "reviewer_prompt_version": protocol.review_prompt_version,
        "submission_fields": {
            "schema_version": protocol.review_schema_version,
            "cohort_version": protocol.cohort_version,
            "shard_id": "Exact shard ID from the blind packet.",
            "packet_sha256": "SHA-256 of the exact blind packet reviewed.",
            "response_schema_sha256": "SHA-256 of this response schema file.",
            "reviewer_prompt_version": protocol.review_prompt_version,
            "reviewer_id": "Stable identifier distinct from the other reviewer.",
            "reviewer_runtime": "Model or runtime identifier.",
            "reviewed_at": "Timezone-aware ISO-8601 timestamp.",
        },
        "case_fields": {
            "review_case_id": "Opaque case ID from the blind packet.",
            "entry_assessments": "One assessment for every displayed position.",
            "sustained_conflict": "yes, no, or uncertain.",
            "delivery_state": "active, recovered, none, or uncertain.",
            "rationale": "Short explanation grounded only in displayed text.",
        },
        "entry_assessment_fields": {
            "position": "One-based packet position.",
            "observable_conflict": sorted(ENTRY_DECISIONS),
            "reason_code": sorted(REASON_CODES),
            "confidence": sorted(CONFIDENCE_LEVELS),
        },
        "rules": REVIEW_RULES,
    }


def build_adjudication_response_schema(
    protocol: ReviewProtocol = LEGACY_REVIEW_PROTOCOL,
) -> dict[str, Any]:
    """Build the disagreement-only response schema for a review protocol."""
    return {
        "schema_version": protocol.adjudication_schema_version,
        "cohort_version": protocol.cohort_version,
        "adjudicator_prompt_version": protocol.adjudication_prompt_version,
        "purpose": (
            "Resolve only entry-level disagreements from two separate packet-only "
            "reviews. Agreed entries are shown for trajectory context but are frozen."
        ),
        "submission_fields": {
            "schema_version": protocol.adjudication_schema_version,
            "cohort_version": protocol.cohort_version,
            "packet_sha256": "SHA-256 of the adjudication packet.",
            "response_schema_sha256": "SHA-256 of this response schema.",
            "adjudicator_prompt_version": protocol.adjudication_prompt_version,
            "adjudicator_id": "Stable adjudicator identifier.",
            "adjudicator_runtime": "Model or runtime identifier.",
            "adjudicated_at": "Timezone-aware ISO-8601 timestamp.",
        },
        "case_fields": {
            "adjudication_case_id": "Opaque case ID from the packet.",
            "entry_adjudications": "One answer for every disputed position only.",
            "rationale": "Short explanation grounded in displayed evidence.",
        },
        "entry_fields": {
            "position": "One-based packet position.",
            "observable_conflict": sorted(ENTRY_DECISIONS),
            "reason_code": sorted(REASON_CODES),
            "confidence": sorted(CONFIDENCE_LEVELS),
        },
        "rules": [
            *REVIEW_RULES[:4],
            (
                "Decide each disputed position independently. The two prior judgments "
                "are anonymous aids, not votes; use the displayed text as authority."
            ),
            "Do not relabel agreed positions.",
        ],
    }


REVIEW_RESPONSE_SCHEMA = build_review_response_schema()
ADJUDICATION_RESPONSE_SCHEMA = build_adjudication_response_schema()


def _stable_digest(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def _maximal_negative_runs(labels: list[int | None]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(labels):
        if labels[index] != -1:
            index += 1
            continue
        start = index
        while index < len(labels) and labels[index] == -1:
            index += 1
        if index - start >= 2:
            runs.append((start, index - 1))
    return runs


def _source_rows(labels_df: pl.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    required = {"persona_id", "t_index", "date"}
    missing = required - set(labels_df.columns)
    if missing:
        raise ValueError(f"Legacy labels missing columns: {sorted(missing)}")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in labels_df.to_dicts():
        key = (str(row["persona_id"]), int(row["t_index"]))
        if key in rows:
            raise ValueError(f"Legacy labels duplicate coordinate {key}")
        rows[key] = row
    return rows


def _historical_split_map(split: TargetSplit) -> dict[str, str]:
    result: dict[str, str] = {}
    groups = {
        "training": split.training_persona_ids,
        "development": split.development_persona_ids,
        "retired": split.retired_persona_ids,
        "former_promotion": split.promotion_persona_ids,
    }
    for name, persona_ids in groups.items():
        for persona_id in persona_ids:
            if persona_id in result:
                raise ValueError(f"Historical splits overlap for {persona_id}")
            result[str(persona_id)] = name
    return result


def build_legacy_trajectory_inventory(
    cases: list[dict[str, Any]],
    consensus_labels_df: pl.DataFrame,
    persisted_labels_df: pl.DataFrame,
    split: TargetSplit,
) -> pl.DataFrame:
    """Describe candidate evidence for every full persona/Core-Value trajectory."""
    consensus = _source_rows(consensus_labels_df)
    persisted = _source_rows(persisted_labels_df)
    split_map = _historical_split_map(split)
    rows: list[dict[str, Any]] = []
    for case in cases:
        persona_id = str(case["persona_id"])
        dimension = str(case["dimension"])
        label_column = f"alignment_{dimension}"
        consensus_values: list[int | None] = []
        persisted_values: list[int | None] = []
        runtime_texts: list[str] = []
        for entry in case["entries"]:
            key = (persona_id, int(entry["t_index"]))
            consensus_row = consensus.get(key)
            persisted_row = persisted.get(key)
            if consensus_row is None or persisted_row is None:
                raise ValueError(f"Legacy labels missing trajectory coordinate {key}")
            if str(consensus_row["date"]) != str(entry["date"]) or str(
                persisted_row["date"]
            ) != str(entry["date"]):
                raise ValueError(f"Legacy label date mismatch for {key}")
            if label_column not in consensus_row or label_column not in persisted_row:
                raise ValueError(f"Legacy labels missing {label_column}")
            consensus_values.append(consensus_row[label_column])
            persisted_values.append(persisted_row[label_column])
            runtime_texts.append(
                concatenate_entry_text(
                    entry.get("initial_entry"),
                    entry.get("nudge_text"),
                    entry.get("response_text"),
                )
            )
        consensus_runs = _maximal_negative_runs(consensus_values)
        persisted_runs = _maximal_negative_runs(persisted_values)
        canonical_case_id = f"{persona_id}:{dimension}"
        rows.append(
            {
                "canonical_case_id": canonical_case_id,
                "persona_id": persona_id,
                "dimension": dimension,
                "historical_split": split_map[persona_id],
                "trajectory_length": len(case["entries"]),
                "displayed_character_count": sum(map(len, runtime_texts)),
                "consensus_candidate": bool(consensus_runs),
                "persisted_candidate": bool(persisted_runs),
                "consensus_window_count": sum(
                    end - start for start, end in consensus_runs
                ),
                "persisted_window_count": sum(
                    end - start for start, end in persisted_runs
                ),
                "consensus_episode_count": len(consensus_runs),
                "persisted_episode_count": len(persisted_runs),
                "case_content_sha256": sha256_json(
                    {
                        "persona_id": persona_id,
                        "dimension": dimension,
                        "entries": [
                            {
                                "t_index": int(entry["t_index"]),
                                "runtime_text": runtime_text,
                            }
                            for entry, runtime_text in zip(
                                case["entries"], runtime_texts, strict=True
                            )
                        ],
                    }
                ),
            }
        )
    inventory = pl.DataFrame(rows).with_columns(
        (pl.col("consensus_candidate") | pl.col("persisted_candidate")).alias(
            "legacy_candidate"
        )
    )
    if inventory["canonical_case_id"].n_unique() != inventory.height:
        raise ValueError("Trajectory inventory contains duplicate cases")
    return inventory.sort("canonical_case_id")


def select_legacy_candidate_union(inventory: pl.DataFrame) -> pl.DataFrame:
    """Select candidates found by either source without mixing pair evidence."""
    return inventory.filter(pl.col("legacy_candidate")).sort("canonical_case_id")


def _split_match_tier(candidate_split: str, control_split: str) -> int:
    if candidate_split == control_split:
        return 0
    if "retired" not in {candidate_split, control_split}:
        return 1
    return 2


def match_legacy_negative_controls(inventory: pl.DataFrame) -> pl.DataFrame:
    """Globally match one unique-person legacy-negative control per candidate."""
    candidates = select_legacy_candidate_union(inventory).to_dicts()
    candidate_personas = {str(row["persona_id"]) for row in candidates}
    controls = inventory.filter(
        (~pl.col("legacy_candidate"))
        & (~pl.col("persona_id").is_in(sorted(candidate_personas)))
    ).to_dicts()
    edges: list[tuple[dict[str, Any], dict[str, Any], int, int, str]] = []
    for candidate in candidates:
        for control in controls:
            if candidate["dimension"] != control["dimension"]:
                continue
            tier = _split_match_tier(
                str(candidate["historical_split"]),
                str(control["historical_split"]),
            )
            delta = abs(
                int(candidate["trajectory_length"]) - int(control["trajectory_length"])
            )
            digest = _stable_digest(
                candidate["canonical_case_id"], control["canonical_case_id"]
            )
            edges.append((candidate, control, tier, delta, digest))
    if not edges:
        raise ValueError("No eligible control edges")

    candidate_ids = [str(row["canonical_case_id"]) for row in candidates]
    personas = sorted({str(edge[1]["persona_id"]) for edge in edges})
    candidate_index = {case_id: index for index, case_id in enumerate(candidate_ids)}
    persona_index = {persona_id: index for index, persona_id in enumerate(personas)}
    constraint = np.zeros((len(candidate_ids) + len(personas), len(edges)))
    max_length_delta = max(edge[3] for edge in edges)
    length_weight = len(candidates) + 1
    tier_weight = length_weight * (len(candidates) * max_length_delta + 1) + len(
        candidates
    )
    costs = []
    for edge_index, (candidate, control, tier, delta, digest) in enumerate(edges):
        constraint[candidate_index[str(candidate["canonical_case_id"])], edge_index] = 1
        constraint[
            len(candidate_ids) + persona_index[str(control["persona_id"])], edge_index
        ] = 1
        tie = int(digest[:12], 16) / float(16**12)
        costs.append(tier * tier_weight + delta * length_weight + tie)

    lower = np.concatenate([np.ones(len(candidate_ids)), np.zeros(len(personas))])
    upper = np.ones(len(candidate_ids) + len(personas))
    result = milp(
        c=np.asarray(costs),
        integrality=np.ones(len(edges)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(constraint, lower, upper),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise ValueError("Cannot match every candidate to a unique-person control")

    rows = []
    selected = [index for index, value in enumerate(result.x) if value > 0.5]
    if len(selected) != len(candidates):
        raise ValueError("Control assignment did not select one edge per candidate")
    for edge_index in selected:
        candidate, control, tier, delta, digest = edges[edge_index]
        analysis_role = (
            "retired_audit_only"
            if candidate["historical_split"] == "retired"
            else "development_reference"
        )
        rows.append(
            {
                "candidate_case_id": candidate["canonical_case_id"],
                "candidate_persona_id": candidate["persona_id"],
                "candidate_historical_split": candidate["historical_split"],
                "control_case_id": control["canonical_case_id"],
                "control_persona_id": control["persona_id"],
                "control_historical_split": control["historical_split"],
                "dimension": candidate["dimension"],
                "candidate_length": candidate["trajectory_length"],
                "control_length": control["trajectory_length"],
                "split_match_tier": tier,
                "length_delta": delta,
                "stable_tie_sha256": digest,
                "analysis_role": analysis_role,
            }
        )
    pairs = pl.DataFrame(rows).sort("candidate_case_id")
    if pairs["control_persona_id"].n_unique() != pairs.height:
        raise ValueError("Control assignment reused a persona")
    return pairs


def selected_case_metadata(
    inventory: pl.DataFrame, pairs: pl.DataFrame
) -> pl.DataFrame:
    """Expand paired selection to one metadata row per reviewed trajectory."""
    inventory_rows = {
        str(row["canonical_case_id"]): row for row in inventory.to_dicts()
    }
    rows = []
    for pair_index, pair in enumerate(pairs.to_dicts(), start=1):
        pair_id = f"pair_{pair_index:03d}"
        for cohort_role, case_field, matched_field in (
            ("candidate", "candidate_case_id", "control_case_id"),
            ("control", "control_case_id", "candidate_case_id"),
        ):
            case_id = str(pair[case_field])
            source = inventory_rows[case_id]
            rows.append(
                {
                    **source,
                    "pair_id": pair_id,
                    "cohort_role": cohort_role,
                    "matched_case_id": str(pair[matched_field]),
                    "analysis_role": pair["analysis_role"],
                    "split_match_tier": pair["split_match_tier"],
                    "length_delta": pair["length_delta"],
                }
            )
    selected = pl.DataFrame(rows).sort("canonical_case_id")
    if selected["canonical_case_id"].n_unique() != selected.height:
        raise ValueError("Selected cohort contains duplicate trajectories")
    return selected


def shard_review_cases(
    selected: pl.DataFrame,
    *,
    max_cases: int,
    max_entries: int,
) -> dict[str, list[str]]:
    """Use stable first-fit decreasing to balance packet entry counts."""
    if max_cases < 1 or max_entries < 1:
        raise ValueError("Shard caps must be positive")
    ordered = sorted(
        selected.to_dicts(),
        key=lambda row: (
            -int(row["trajectory_length"]),
            str(row["canonical_case_id"]),
        ),
    )
    shards: list[list[dict[str, Any]]] = []
    for row in ordered:
        length = int(row["trajectory_length"])
        if length > max_entries:
            raise ValueError(
                f"Case exceeds shard entry cap: {row['canonical_case_id']}"
            )
        for shard in shards:
            if (
                len(shard) < max_cases
                and sum(int(item["trajectory_length"]) for item in shard) + length
                <= max_entries
            ):
                shard.append(row)
                break
        else:
            shards.append([row])
    result = {}
    for index, shard in enumerate(shards, start=1):
        result[f"shard_{index:03d}"] = sorted(
            str(row["canonical_case_id"]) for row in shard
        )
    validate_shard_coverage(result, selected["canonical_case_id"].to_list())
    return result


def validate_shard_coverage(
    shards: dict[str, list[str]], expected_case_ids: list[str]
) -> None:
    observed = [case_id for case_ids in shards.values() for case_id in case_ids]
    if len(observed) != len(set(observed)):
        raise ValueError("Review shards overlap")
    if set(observed) != set(expected_case_ids):
        raise ValueError("Review shards do not cover the frozen cohort exactly")


def build_blind_shard(
    *,
    shard_id: str,
    case_ids: list[str],
    cases_by_id: dict[str, dict[str, Any]],
    protocol: ReviewProtocol = LEGACY_REVIEW_PROTOCOL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one reviewer packet and its parent-only identity key."""
    ordered = sorted(
        case_ids,
        key=lambda case_id: _stable_digest(protocol.cohort_version, shard_id, case_id),
    )
    packet_cases = []
    key_cases = []
    for index, canonical_case_id in enumerate(ordered, start=1):
        case = cases_by_id[canonical_case_id]
        review_case_id = f"case_{index:03d}"
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
                    for position, entry in enumerate(case["entries"], start=1)
                ],
            }
        )
        key_cases.append(
            {
                "review_case_id": review_case_id,
                "canonical_case_id": canonical_case_id,
                "persona_id": case["persona_id"],
                "dimension": case["dimension"],
                "entries": [
                    {
                        "position": position,
                        "t_index": int(entry["t_index"]),
                        "date": str(entry["date"]),
                        "runtime_text_sha256": hashlib.sha256(
                            concatenate_entry_text(
                                entry.get("initial_entry"),
                                entry.get("nudge_text"),
                                entry.get("response_text"),
                            ).encode()
                        ).hexdigest(),
                    }
                    for position, entry in enumerate(case["entries"], start=1)
                ],
            }
        )
    packet = {
        "schema_version": protocol.review_schema_version,
        "cohort_version": protocol.cohort_version,
        "shard_id": shard_id,
        "review_instructions": REVIEW_RULES,
        "cases": packet_cases,
    }
    key = {
        "schema_version": protocol.review_schema_version,
        "cohort_version": protocol.cohort_version,
        "shard_id": shard_id,
        "warning": "Parent-only identity key; never disclose before review.",
        "cases": key_cases,
    }
    return packet, key


def validate_review_response(
    response: dict[str, Any],
    *,
    packet: dict[str, Any],
    packet_sha256: str,
    response_schema_sha256: str,
    protocol: ReviewProtocol = LEGACY_REVIEW_PROTOCOL,
) -> dict[str, dict[str, Any]]:
    """Validate a complete packet-bound reviewer response."""
    expected = {
        "schema_version": protocol.review_schema_version,
        "cohort_version": protocol.cohort_version,
        "shard_id": packet["shard_id"],
        "packet_sha256": packet_sha256,
        "response_schema_sha256": response_schema_sha256,
        "reviewer_prompt_version": protocol.review_prompt_version,
    }
    for field, value in expected.items():
        if response.get(field) != value:
            raise ValueError(f"Review response has invalid {field}")
    if not isinstance(response.get("reviewer_id"), str) or not response["reviewer_id"]:
        raise ValueError("Review response must name reviewer_id")
    if (
        not isinstance(response.get("reviewer_runtime"), str)
        or not response["reviewer_runtime"]
    ):
        raise ValueError("Review response must name reviewer_runtime")
    parse_aware_timestamp(response.get("reviewed_at"), field_name="reviewed_at")
    cases = response.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Review response cases must be a list")
    packet_map = {case["review_case_id"]: case for case in packet["cases"]}
    result: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = case.get("review_case_id")
        if case_id not in packet_map or case_id in result:
            raise ValueError(f"Invalid or duplicate review case {case_id}")
        assessments = case.get("entry_assessments")
        if not isinstance(assessments, list):
            raise ValueError(f"Missing assessments for {case_id}")
        expected_positions = {
            entry["position"] for entry in packet_map[case_id]["entries"]
        }
        observed_positions = {entry.get("position") for entry in assessments}
        if observed_positions != expected_positions or len(assessments) != len(
            expected_positions
        ):
            raise ValueError(f"Incomplete entry coverage for {case_id}")
        yes_positions = set()
        has_uncertain = False
        for assessment in assessments:
            position = assessment["position"]
            decision = assessment.get("observable_conflict")
            if decision not in ENTRY_DECISIONS:
                raise ValueError(f"Invalid decision for {case_id}:{position}")
            if assessment.get("reason_code") not in REASON_CODES:
                raise ValueError(f"Invalid reason for {case_id}:{position}")
            if assessment.get("confidence") not in CONFIDENCE_LEVELS:
                raise ValueError(f"Invalid confidence for {case_id}:{position}")
            if decision == "yes" and assessment["reason_code"] != (
                "direct_behavior_or_choice"
            ):
                raise ValueError(
                    f"Yes requires direct behavior for {case_id}:{position}"
                )
            if decision == "yes":
                yes_positions.add(position)
            elif decision == "uncertain":
                has_uncertain = True
        has_pair = any(position + 1 in yes_positions for position in yes_positions)
        sustained = case.get("sustained_conflict")
        valid_sustained = (
            {"yes"} if has_pair else {"no", "uncertain"} if has_uncertain else {"no"}
        )
        if sustained not in valid_sustained:
            raise ValueError(f"Inconsistent sustained_conflict for {case_id}")
        if case.get("delivery_state") not in DELIVERY_STATES:
            raise ValueError(f"Invalid delivery_state for {case_id}")
        if not isinstance(case.get("rationale"), str) or not case["rationale"].strip():
            raise ValueError(f"Missing rationale for {case_id}")
        result[case_id] = case
    if set(result) != set(packet_map):
        raise ValueError("Review response does not cover the packet exactly")
    return result


def reconcile_reviews(
    *,
    packet: dict[str, Any],
    key: dict[str, Any],
    reviewer_a: dict[str, Any],
    reviewer_b: dict[str, Any],
    packet_sha256: str,
    response_schema_sha256: str,
    reviewer_a_sha256: str,
    reviewer_b_sha256: str,
    protocol: ReviewProtocol = LEGACY_REVIEW_PROTOCOL,
) -> list[dict[str, Any]]:
    """Return one entry-level row per reviewed position, preserving disagreement."""
    if reviewer_a.get("reviewer_id") == reviewer_b.get("reviewer_id"):
        raise ValueError("Reviewers must use distinct reviewer IDs")
    first = validate_review_response(
        reviewer_a,
        packet=packet,
        packet_sha256=packet_sha256,
        response_schema_sha256=response_schema_sha256,
        protocol=protocol,
    )
    second = validate_review_response(
        reviewer_b,
        packet=packet,
        packet_sha256=packet_sha256,
        response_schema_sha256=response_schema_sha256,
        protocol=protocol,
    )
    key_map = {case["review_case_id"]: case for case in key["cases"]}
    if set(key_map) != {case["review_case_id"] for case in packet["cases"]}:
        raise ValueError("Packet and identity key have different cases")
    rows = []
    for review_case_id, key_case in sorted(key_map.items()):
        first_case = first[review_case_id]
        second_case = second[review_case_id]
        first_entries = {
            row["position"]: row for row in first_case["entry_assessments"]
        }
        second_entries = {
            row["position"]: row for row in second_case["entry_assessments"]
        }
        for key_entry in key_case["entries"]:
            position = key_entry["position"]
            first_entry = first_entries[position]
            second_entry = second_entries[position]
            first_decision = first_entry["observable_conflict"]
            second_decision = second_entry["observable_conflict"]
            agreed = first_decision == second_decision and first_decision in {
                "yes",
                "no",
            }
            rows.append(
                {
                    "cohort_version": protocol.cohort_version,
                    "shard_id": packet["shard_id"],
                    "review_case_id": review_case_id,
                    "canonical_case_id": key_case["canonical_case_id"],
                    "persona_id": key_case["persona_id"],
                    "dimension": key_case["dimension"],
                    "position": position,
                    "t_index": key_entry["t_index"],
                    "date": key_entry["date"],
                    "runtime_text_sha256": key_entry["runtime_text_sha256"],
                    "reviewer_a_disposition": first_decision,
                    "reviewer_a_reason": first_entry["reason_code"],
                    "reviewer_a_confidence": first_entry["confidence"],
                    "reviewer_a_response_sha256": reviewer_a_sha256,
                    "reviewer_b_disposition": second_decision,
                    "reviewer_b_reason": second_entry["reason_code"],
                    "reviewer_b_confidence": second_entry["confidence"],
                    "reviewer_b_response_sha256": reviewer_b_sha256,
                    "final_conflict": (first_decision == "yes" if agreed else None),
                    "resolution_method": "agreement" if agreed else "unresolved",
                    "resolution_status": "resolved" if agreed else "unresolved",
                }
            )
    return rows


def build_adjudication_packet(
    cases: list[dict[str, Any]],
    *,
    protocol: ReviewProtocol = LEGACY_REVIEW_PROTOCOL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Blind identities and lane ordering for disagreement-only adjudication."""
    if not cases:
        raise ValueError("No unresolved cases require adjudication")
    ordered = sorted(
        cases,
        key=lambda case: _stable_digest(
            protocol.cohort_version, "adjudication", case["canonical_case_id"]
        ),
    )
    packet_cases = []
    key_cases = []
    for index, case in enumerate(ordered, start=1):
        adjudication_case_id = f"adjudication_{index:03d}"
        swap_lanes = (
            int(
                _stable_digest(protocol.cohort_version, case["canonical_case_id"])[0],
                16,
            )
            % 2
            == 1
        )
        disputed_positions = []
        entries = []
        for entry in case["entries"]:
            first = {
                "disposition": entry["reviewer_a_disposition"],
                "reason_code": entry["reviewer_a_reason"],
                "confidence": entry["reviewer_a_confidence"],
            }
            second = {
                "disposition": entry["reviewer_b_disposition"],
                "reason_code": entry["reviewer_b_reason"],
                "confidence": entry["reviewer_b_confidence"],
            }
            if swap_lanes:
                first, second = second, first
            disputed = entry["resolution_status"] == "unresolved"
            if disputed:
                disputed_positions.append(int(entry["position"]))
            entries.append(
                {
                    "position": int(entry["position"]),
                    "journal_entry": entry["journal_entry"],
                    "is_disputed": disputed,
                    "review_one": first,
                    "review_two": second,
                }
            )
        if not disputed_positions:
            raise ValueError("Adjudication case has no disputed positions")
        rationales = list(case["review_rationales"])
        if swap_lanes:
            rationales.reverse()
        packet_cases.append(
            {
                "adjudication_case_id": adjudication_case_id,
                "declared_core_value": case["declared_core_value"],
                "disputed_positions": disputed_positions,
                "entries": entries,
                "anonymous_review_rationales": rationales,
            }
        )
        key_cases.append(
            {
                "adjudication_case_id": adjudication_case_id,
                "canonical_case_id": case["canonical_case_id"],
                "disputed_positions": disputed_positions,
            }
        )
    packet = {
        "schema_version": protocol.adjudication_schema_version,
        "cohort_version": protocol.cohort_version,
        "adjudication_instructions": build_adjudication_response_schema(protocol)[
            "rules"
        ],
        "cases": packet_cases,
    }
    key = {
        "schema_version": protocol.adjudication_schema_version,
        "cohort_version": protocol.cohort_version,
        "warning": "Parent-only identity key; never disclose before adjudication.",
        "cases": key_cases,
    }
    return packet, key


def validate_adjudication_response(
    response: dict[str, Any],
    *,
    packet: dict[str, Any],
    packet_sha256: str,
    response_schema_sha256: str,
    protocol: ReviewProtocol = LEGACY_REVIEW_PROTOCOL,
) -> dict[str, dict[str, Any]]:
    """Validate complete, receipt-bound decisions for disputed positions."""
    expected = {
        "schema_version": protocol.adjudication_schema_version,
        "cohort_version": protocol.cohort_version,
        "packet_sha256": packet_sha256,
        "response_schema_sha256": response_schema_sha256,
        "adjudicator_prompt_version": protocol.adjudication_prompt_version,
    }
    for field, value in expected.items():
        if response.get(field) != value:
            raise ValueError(f"Adjudication response has invalid {field}")
    for field in ("adjudicator_id", "adjudicator_runtime"):
        if not isinstance(response.get(field), str) or not response[field]:
            raise ValueError(f"Adjudication response must name {field}")
    parse_aware_timestamp(response.get("adjudicated_at"), field_name="adjudicated_at")
    packet_map = {case["adjudication_case_id"]: case for case in packet["cases"]}
    cases = response.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Adjudication response cases must be a list")
    result = {}
    for case in cases:
        case_id = case.get("adjudication_case_id")
        if case_id not in packet_map or case_id in result:
            raise ValueError(f"Invalid or duplicate adjudication case {case_id}")
        if not isinstance(case.get("rationale"), str) or not case["rationale"].strip():
            raise ValueError(f"Missing adjudication rationale for {case_id}")
        decisions = case.get("entry_adjudications")
        if not isinstance(decisions, list):
            raise ValueError(f"Missing entry adjudications for {case_id}")
        expected_positions = set(packet_map[case_id]["disputed_positions"])
        observed_positions = {decision.get("position") for decision in decisions}
        if observed_positions != expected_positions or len(decisions) != len(
            expected_positions
        ):
            raise ValueError(f"Incomplete adjudication coverage for {case_id}")
        for decision in decisions:
            position = decision["position"]
            disposition = decision.get("observable_conflict")
            if disposition not in ENTRY_DECISIONS:
                raise ValueError(f"Invalid adjudication for {case_id}:{position}")
            if decision.get("reason_code") not in REASON_CODES:
                raise ValueError(
                    f"Invalid adjudication reason for {case_id}:{position}"
                )
            if decision.get("confidence") not in CONFIDENCE_LEVELS:
                raise ValueError(
                    f"Invalid adjudication confidence for {case_id}:{position}"
                )
            if disposition == "yes" and decision["reason_code"] != (
                "direct_behavior_or_choice"
            ):
                raise ValueError(
                    f"Adjudicated yes requires direct behavior for {case_id}:{position}"
                )
        result[case_id] = case
    if set(result) != set(packet_map):
        raise ValueError("Adjudication response does not cover every disputed case")
    return result


def apply_adjudication(
    entry_target: pl.DataFrame,
    *,
    packet: dict[str, Any],
    key: dict[str, Any],
    response: dict[str, Any],
    packet_sha256: str,
    response_schema_sha256: str,
    response_sha256: str,
    protocol: ReviewProtocol = LEGACY_REVIEW_PROTOCOL,
) -> pl.DataFrame:
    """Apply adjudication only to unresolved entries; agreed entries stay frozen."""
    response_cases = validate_adjudication_response(
        response,
        packet=packet,
        packet_sha256=packet_sha256,
        response_schema_sha256=response_schema_sha256,
        protocol=protocol,
    )
    key_map = {case["adjudication_case_id"]: case for case in key["cases"]}
    if set(key_map) != set(response_cases):
        raise ValueError("Adjudication response and identity key have different cases")
    decisions = {}
    for adjudication_case_id, case in response_cases.items():
        canonical_case_id = key_map[adjudication_case_id]["canonical_case_id"]
        for decision in case["entry_adjudications"]:
            coordinate = (canonical_case_id, int(decision["position"]))
            if coordinate in decisions:
                raise ValueError(f"Duplicate adjudication coordinate {coordinate}")
            decisions[coordinate] = decision

    rows = []
    used = set()
    for row in entry_target.to_dicts():
        coordinate = (str(row["canonical_case_id"]), int(row["position"]))
        decision = decisions.get(coordinate)
        updated = {
            **row,
            "adjudicator_disposition": None,
            "adjudicator_reason": None,
            "adjudicator_confidence": None,
            "adjudicator_response_sha256": None,
        }
        if decision is not None:
            if row["resolution_status"] != "unresolved":
                raise ValueError(
                    f"Adjudication attempted to relabel agreement {coordinate}"
                )
            used.add(coordinate)
            disposition = decision["observable_conflict"]
            resolved = disposition in {"yes", "no"}
            updated.update(
                {
                    "adjudicator_disposition": disposition,
                    "adjudicator_reason": decision["reason_code"],
                    "adjudicator_confidence": decision["confidence"],
                    "adjudicator_response_sha256": response_sha256,
                    "final_conflict": (disposition == "yes" if resolved else None),
                    "resolution_method": ("adjudication" if resolved else "unresolved"),
                    "resolution_status": "resolved" if resolved else "unresolved",
                }
            )
        rows.append(updated)
    if used != set(decisions):
        raise ValueError("Adjudication contains unknown entry coordinates")
    return pl.DataFrame(rows).sort("canonical_case_id", "position")


def derive_review_outcomes(
    entry_target: pl.DataFrame, selected: pl.DataFrame, *, cohort_sha256: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Derive case outcomes and maximal consecutive Conflict episodes."""
    metadata = {str(row["canonical_case_id"]): row for row in selected.to_dicts()}
    outcomes = []
    episodes = []
    for case_key, frame in (
        entry_target.sort("position")
        .partition_by("canonical_case_id", as_dict=True)
        .items()
    ):
        case_id = str(case_key[0] if isinstance(case_key, tuple) else case_key)
        rows = frame.to_dicts()
        meta = metadata[case_id]
        unresolved = any(row["final_conflict"] is None for row in rows)
        labels = [row["final_conflict"] for row in rows]
        runs = []
        if not unresolved:
            index = 0
            while index < len(labels):
                if labels[index] is not True:
                    index += 1
                    continue
                start = index
                while index < len(labels) and labels[index] is True:
                    index += 1
                if index - start >= 2:
                    runs.append((start, index - 1))
        for episode_index, (start, end) in enumerate(runs, start=1):
            onset = rows[start]
            confirmation = rows[start + 1]
            ending = rows[end]
            next_row = rows[end + 1] if end + 1 < len(rows) else None
            episodes.append(
                {
                    "cohort_version": COHORT_VERSION,
                    "cohort_sha256": cohort_sha256,
                    "episode_id": f"{case_id}:episode_{episode_index:02d}",
                    "canonical_case_id": case_id,
                    "persona_id": meta["persona_id"],
                    "dimension": meta["dimension"],
                    "cohort_role": meta["cohort_role"],
                    "analysis_role": meta["analysis_role"],
                    "historical_split": meta["historical_split"],
                    "onset_position": onset["position"],
                    "confirmation_position": confirmation["position"],
                    "end_position": ending["position"],
                    "onset_t_index": onset["t_index"],
                    "confirmation_t_index": confirmation["t_index"],
                    "end_t_index": ending["t_index"],
                    "onset_date": onset["date"],
                    "confirmation_date": confirmation["date"],
                    "end_date": ending["date"],
                    "supporting_positions": list(range(start + 1, end + 2)),
                    "length": end - start + 1,
                    "open_at_cutoff": next_row is None,
                    "termination_position": (
                        next_row["position"] if next_row is not None else None
                    ),
                    "termination_disposition": ("no" if next_row is not None else None),
                }
            )
        outcomes.append(
            {
                "cohort_version": COHORT_VERSION,
                "cohort_sha256": cohort_sha256,
                "canonical_case_id": case_id,
                "persona_id": meta["persona_id"],
                "dimension": meta["dimension"],
                "cohort_role": meta["cohort_role"],
                "analysis_role": meta["analysis_role"],
                "historical_split": meta["historical_split"],
                "pair_id": meta["pair_id"],
                "matched_case_id": meta["matched_case_id"],
                "trajectory_length": meta["trajectory_length"],
                "consensus_candidate": meta["consensus_candidate"],
                "persisted_candidate": meta["persisted_candidate"],
                "consensus_window_count": meta["consensus_window_count"],
                "persisted_window_count": meta["persisted_window_count"],
                "case_content_sha256": meta["case_content_sha256"],
                "entry_agreement_count": sum(
                    row.get("resolution_method") == "agreement" for row in rows
                ),
                "entry_count": len(rows),
                "case_resolution": "unresolved" if unresolved else "resolved",
                "has_drift": None if unresolved else bool(runs),
                "episode_count": None if unresolved else len(runs),
            }
        )
    outcome_df = pl.DataFrame(outcomes).sort("canonical_case_id")
    episode_df = pl.DataFrame(episodes) if episodes else pl.DataFrame()
    return outcome_df, episode_df


def summarize_outcomes(outcomes: pl.DataFrame) -> dict[str, Any]:
    """Summarize candidate and control Drift confirmation by analysis stratum."""
    result: dict[str, Any] = {
        "cohort_version": COHORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "case_count": outcomes.height,
        "entry_count": int(outcomes["entry_count"].sum()),
        "resolved_case_count": outcomes.filter(
            pl.col("case_resolution") == "resolved"
        ).height,
        "unresolved_case_count": outcomes.filter(
            pl.col("case_resolution") == "unresolved"
        ).height,
        "strata": {},
    }
    for analysis_role in ("development_reference", "retired_audit_only", "all"):
        role_frame = (
            outcomes
            if analysis_role == "all"
            else outcomes.filter(pl.col("analysis_role") == analysis_role)
        )
        role_result = {}
        for cohort_role in ("candidate", "control"):
            cohort = role_frame.filter(pl.col("cohort_role") == cohort_role)
            resolved = cohort.filter(pl.col("case_resolution") == "resolved")
            drift_count = resolved.filter(pl.col("has_drift") == True).height  # noqa: E712
            role_result[cohort_role] = {
                "selected": cohort.height,
                "resolved": resolved.height,
                "unresolved": cohort.height - resolved.height,
                "drift_count": drift_count,
                "drift_confirmation_rate_among_resolved": (
                    drift_count / resolved.height if resolved.height else None
                ),
            }
        result["strata"][analysis_role] = role_result
    return result


def validate_expected_counts(
    *,
    inventory: pl.DataFrame,
    pairs: pl.DataFrame,
    selected: pl.DataFrame,
    expected: dict,
) -> None:
    """Fail if live data no longer matches the frozen pre-review contract."""
    candidates = select_legacy_candidate_union(inventory)
    checks = {
        "candidates": candidates.height,
        "controls": pairs.height,
        "candidate_personas": candidates["persona_id"].n_unique(),
        "selected_cases": selected.height,
        "selected_entries": int(selected["trajectory_length"].sum()),
        "candidate_source_both": candidates.filter(
            pl.col("consensus_candidate") & pl.col("persisted_candidate")
        ).height,
        "candidate_source_persisted_only": candidates.filter(
            (~pl.col("consensus_candidate")) & pl.col("persisted_candidate")
        ).height,
        "candidate_source_consensus_only": candidates.filter(
            pl.col("consensus_candidate") & (~pl.col("persisted_candidate"))
        ).height,
        "same_split_matches": pairs.filter(pl.col("split_match_tier") == 0).height,
        "exact_length_matches": pairs.filter(pl.col("length_delta") == 0).height,
        "split_tier_sum": int(pairs["split_match_tier"].sum()),
        "total_length_delta": int(pairs["length_delta"].sum()),
        "retired_candidate_pairs": pairs.filter(
            pl.col("analysis_role") == "retired_audit_only"
        ).height,
    }
    for name, actual in checks.items():
        if int(expected.get(name, -1)) != int(actual):
            raise ValueError(
                f"Expected {name}={expected.get(name)!r}; live value is {actual}"
            )


def artifact_hashes(paths: list[Path], *, root: Path) -> dict[str, str]:
    """Return sorted repo-relative hashes for existing artifact files."""
    result = {}
    for path in paths:
        resolved = path.resolve()
        try:
            relative = str(resolved.relative_to(root.resolve()))
        except ValueError:
            relative = str(resolved)
        result[relative] = sha256_file(resolved)
    return dict(sorted(result.items()))
