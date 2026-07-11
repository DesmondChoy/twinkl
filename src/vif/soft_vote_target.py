"""Build the twinkl-j0ck hybrid soft-vote training target."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import polars as pl

from src.judge.consensus_utils import validate_result_rows_against_ids
from src.models.judge import SCHWARTZ_VALUE_ORDER

SOFT_TARGET_CLASS_ORDER = (-1, 0, 1)
SOFT_TARGET_VALUE_ORDER = tuple(SCHWARTZ_VALUE_ORDER)
SOFT_TARGET_REGIME = "twinkl_j0ck_hybrid_soft_vote_v1"
TWINKL_754_SOURCE = "twinkl_754_profile_only_five_pass"
SECURITY_SOURCE = "security_active_critic_state_v1"
N_TWINKL_754_PASSES = 5


def _validate_unique_coordinates(frame: pl.DataFrame, *, source_name: str) -> None:
    required = {"persona_id", "t_index", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required fields: {sorted(missing)}."
        )
    if frame.select(["persona_id", "t_index"]).n_unique() != frame.height:
        raise ValueError(f"{source_name} has duplicate entry coordinates.")


def _coordinate_map(
    frame: pl.DataFrame,
    *,
    source_name: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    _validate_unique_coordinates(frame, source_name=source_name)
    return {
        (str(row["persona_id"]), int(row["t_index"])): row
        for row in frame.iter_rows(named=True)
    }


def _validate_coordinate_sets(
    expected: Mapping[tuple[str, int], Mapping[str, Any]],
    observed: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    source_name: str,
) -> None:
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValueError(
            f"{source_name} coordinates do not match the manifest. "
            f"Missing={missing[:5]}, extra={extra[:5]}."
        )
    date_mismatches = [
        key
        for key in expected
        if str(expected[key]["date"]) != str(observed[key]["date"])
    ]
    if date_mismatches:
        raise ValueError(
            f"{source_name} dates do not match the manifest: "
            f"{date_mismatches[:5]}."
        )


def resolve_activity_then_polarity(votes: Iterable[int]) -> int:
    """Reproduce the hard twinkl-754 activity-then-polarity resolver."""
    labels = [int(vote) for vote in votes]
    if len(labels) != N_TWINKL_754_PASSES:
        raise ValueError("twinkl-754 targets require exactly five votes.")
    if any(label not in SOFT_TARGET_CLASS_ORDER for label in labels):
        raise ValueError("Vote labels must be one of -1, 0, or +1.")

    if labels.count(0) >= 3:
        return 0
    positive_count = labels.count(1)
    negative_count = labels.count(-1)
    if positive_count == negative_count:
        return 0
    return 1 if positive_count > negative_count else -1


def _vote_statistics(votes: Iterable[int]) -> tuple[list[int], list[float], float]:
    labels = [int(vote) for vote in votes]
    if not labels or any(label not in SOFT_TARGET_CLASS_ORDER for label in labels):
        raise ValueError(
            "Votes must be a non-empty sequence containing only -1, 0, +1."
        )
    counts = Counter(labels)
    count_vector = [counts.get(label, 0) for label in SOFT_TARGET_CLASS_ORDER]
    probabilities = [count / len(labels) for count in count_vector]
    entropy = -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0
    )
    return count_vector, probabilities, entropy


def _load_twinkl_754_votes(
    manifest: pl.DataFrame,
    vote_passes: Mapping[int, Iterable[Mapping[str, Any]]],
) -> dict[int, dict[str, dict[str, int]]]:
    if set(vote_passes) != set(range(1, N_TWINKL_754_PASSES + 1)):
        raise ValueError("twinkl-754 requires exactly vote passes 1 through 5.")
    if "entry_id" not in manifest.columns:
        raise ValueError("twinkl-754 manifest is missing entry_id.")
    entry_ids = [str(value) for value in manifest["entry_id"].to_list()]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError("twinkl-754 manifest has duplicate entry_id values.")

    expected_entry_ids = [
        str(row["entry_id"])
        for row in manifest.sort(["persona_id", "t_index"]).iter_rows(named=True)
    ]
    expected_formula = {
        f"{row['persona_id']}__{int(row['t_index'])}"
        for row in manifest.iter_rows(named=True)
    }
    if set(entry_ids) != expected_formula:
        raise ValueError(
            "twinkl-754 entry_id values do not match persona_id and t_index."
        )

    by_pass: dict[int, dict[str, dict[str, int]]] = {}
    for pass_index in range(1, N_TWINKL_754_PASSES + 1):
        normalized, _ = validate_result_rows_against_ids(
            [dict(row) for row in vote_passes[pass_index]],
            valid_manifest_ids=set(expected_entry_ids),
            expected_entry_ids=expected_entry_ids,
        )
        by_pass[pass_index] = {
            str(row["entry_id"]): {
                value_name: int(row["scores"][value_name])
                for value_name in SOFT_TARGET_VALUE_ORDER
            }
            for row in normalized
        }
    return by_pass


def _merge_security_rationale(
    consensus_rationales_json: object,
    repaired_rationales_json: object,
) -> str | None:
    consensus = (
        json.loads(consensus_rationales_json)
        if isinstance(consensus_rationales_json, str) and consensus_rationales_json
        else {}
    )
    repaired = (
        json.loads(repaired_rationales_json)
        if isinstance(repaired_rationales_json, str) and repaired_rationales_json
        else {}
    )
    if not isinstance(consensus, dict) or not isinstance(repaired, dict):
        raise ValueError("Label rationales_json fields must contain JSON objects.")
    if "security" in repaired:
        consensus["security"] = repaired["security"]
    else:
        consensus.pop("security", None)
    return json.dumps(consensus, sort_keys=True) if consensus else None


def build_hybrid_soft_vote_target(
    *,
    twinkl_754_manifest: pl.DataFrame,
    twinkl_754_vote_passes: Mapping[int, Iterable[Mapping[str, Any]]],
    consensus_labels: pl.DataFrame,
    repaired_security_labels: pl.DataFrame,
    security_vote_target: pl.DataFrame,
) -> pl.DataFrame:
    """Combine nine five-pass targets with the active-state Security reviews.

    ``soft_alignment_vector`` is flattened value-major, with each value using
    the explicit class order ``[-1, 0, +1]``. Its length is therefore 30.
    """
    manifest_by_key = _coordinate_map(
        twinkl_754_manifest, source_name="twinkl-754 manifest"
    )
    consensus_by_key = _coordinate_map(
        consensus_labels, source_name="twinkl-754 consensus labels"
    )
    repaired_by_key = _coordinate_map(
        repaired_security_labels, source_name="repaired Security labels"
    )
    security_by_key = _coordinate_map(
        security_vote_target, source_name="Security vote target"
    )
    for name, observed in (
        ("twinkl-754 consensus labels", consensus_by_key),
        ("repaired Security labels", repaired_by_key),
        ("Security vote target", security_by_key),
    ):
        _validate_coordinate_sets(manifest_by_key, observed, source_name=name)

    required_consensus = {
        "alignment_vector",
        "rationales_json",
        *(f"alignment_{name}" for name in SOFT_TARGET_VALUE_ORDER),
    }
    missing_consensus = required_consensus - set(consensus_labels.columns)
    if missing_consensus:
        raise ValueError(
            "twinkl-754 consensus labels are missing required fields: "
            f"{sorted(missing_consensus)}."
        )
    required_security = {
        "new_label",
        "vote_minus1",
        "vote_neutral",
        "vote_plus1",
        "vote_count",
        "agreement_count",
        "decision_method",
        "target_policy",
        "state_contract_version",
    }
    missing_security = required_security - set(security_vote_target.columns)
    if missing_security:
        raise ValueError(
            "Security vote target is missing required fields: "
            f"{sorted(missing_security)}."
        )

    pass_scores = _load_twinkl_754_votes(
        twinkl_754_manifest, twinkl_754_vote_passes
    )
    output_rows: list[dict[str, Any]] = []
    for manifest_row in twinkl_754_manifest.sort(
        ["persona_id", "t_index"]
    ).iter_rows(named=True):
        key = (str(manifest_row["persona_id"]), int(manifest_row["t_index"]))
        entry_id = str(manifest_row["entry_id"])
        consensus_row = consensus_by_key[key]
        repaired_row = repaired_by_key[key]
        security_row = security_by_key[key]
        output_row = dict(consensus_row)
        for stale_security_field in (
            "confidence_security",
            "consensus_agreement_security",
            "label_changed_security",
        ):
            output_row.pop(stale_security_field, None)
        soft_alignment_vector: list[float] = []
        hard_alignment_vector: list[int] = []

        for value_name in SOFT_TARGET_VALUE_ORDER:
            if value_name == "security":
                counts = [
                    int(security_row["vote_minus1"]),
                    int(security_row["vote_neutral"]),
                    int(security_row["vote_plus1"]),
                ]
                vote_total = int(security_row["vote_count"])
                if vote_total not in {3, 4} or sum(counts) != vote_total:
                    raise ValueError(
                        f"Invalid Security vote counts for {entry_id}: {counts}."
                    )
                probabilities = [count / vote_total for count in counts]
                entropy = -sum(
                    probability * math.log2(probability)
                    for probability in probabilities
                    if probability > 0
                )
                hard_label = int(security_row["new_label"])
                winners = [
                    label
                    for label, count in zip(
                        SOFT_TARGET_CLASS_ORDER, counts, strict=True
                    )
                    if count == max(counts)
                ]
                if winners != [hard_label]:
                    raise ValueError(
                        f"Security votes do not resolve to new_label for {entry_id}."
                    )
                if int(repaired_row["alignment_security"]) != hard_label:
                    raise ValueError(
                        f"Repaired Security label does not match votes for {entry_id}."
                    )
            else:
                votes = [
                    pass_scores[index][entry_id][value_name]
                    for index in range(1, N_TWINKL_754_PASSES + 1)
                ]
                counts, probabilities, entropy = _vote_statistics(votes)
                vote_total = N_TWINKL_754_PASSES
                hard_label = resolve_activity_then_polarity(votes)
                if int(consensus_row[f"alignment_{value_name}"]) != hard_label:
                    raise ValueError(
                        "twinkl-754 votes do not match the stored consensus for "
                        f"{entry_id}/{value_name}."
                    )

            output_row[f"vote_count_{value_name}"] = counts
            output_row[f"vote_probability_{value_name}"] = probabilities
            output_row[f"vote_total_{value_name}"] = vote_total
            output_row[f"vote_entropy_bits_{value_name}"] = entropy
            output_row[f"alignment_{value_name}"] = hard_label
            soft_alignment_vector.extend(probabilities)
            hard_alignment_vector.append(hard_label)

        output_row.update(
            {
                "entry_id": entry_id,
                "alignment_vector": hard_alignment_vector,
                "soft_alignment_vector": soft_alignment_vector,
                "soft_target_class_order": list(SOFT_TARGET_CLASS_ORDER),
                "soft_target_value_order": list(SOFT_TARGET_VALUE_ORDER),
                "soft_target_regime": SOFT_TARGET_REGIME,
                "twinkl_754_source": TWINKL_754_SOURCE,
                "security_target_source": SECURITY_SOURCE,
                "security_target_policy": str(security_row["target_policy"]),
                "security_decision_method": str(
                    security_row["decision_method"]
                ),
                "security_agreement_count": int(
                    security_row["agreement_count"]
                ),
                "security_state_contract_version": str(
                    security_row["state_contract_version"]
                ),
                "rationales_json": _merge_security_rationale(
                    consensus_row.get("rationales_json"),
                    repaired_row.get("rationales_json"),
                ),
            }
        )
        if len(soft_alignment_vector) != len(SOFT_TARGET_VALUE_ORDER) * len(
            SOFT_TARGET_CLASS_ORDER
        ):
            raise AssertionError("Soft target vector has an unexpected length.")
        output_rows.append(output_row)

    return pl.DataFrame(output_rows).sort(["persona_id", "t_index"])


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
