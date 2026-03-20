"""Shared helpers for the twinkl-754 consensus re-judging workflow."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.models.judge import AlignmentScores, SCHWARTZ_VALUE_ORDER


@dataclass(frozen=True)
class RationaleCoverage:
    """Coverage stats for rationales attached to non-zero labels."""

    non_zero_score_count: int
    non_zero_rationale_count: int

    @property
    def coverage(self) -> float:
        if self.non_zero_score_count == 0:
            return 1.0
        return self.non_zero_rationale_count / self.non_zero_score_count


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_bundle_status(bundle_dir: Path) -> dict[str, Any]:
    status_path = bundle_dir / "bundle_status.json"
    if not status_path.exists():
        return {}
    return read_json(status_path)


def write_bundle_status(bundle_dir: Path, payload: dict[str, Any]) -> None:
    write_json(bundle_dir / "bundle_status.json", payload)


def normalize_rationales(raw_rationales: object) -> dict[str, str]:
    if raw_rationales is None:
        return {}
    if not isinstance(raw_rationales, dict):
        raise ValueError("`rationales` must be an object when present.")

    cleaned: dict[str, str] = {}
    valid_keys = set(SCHWARTZ_VALUE_ORDER)
    for key, value in raw_rationales.items():
        if key not in valid_keys:
            raise ValueError(f"Invalid rationale key: {key}")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid rationale value for {key!r}.")
        cleaned[key] = value.strip()
    return cleaned


def normalize_result_payload(
    payload: dict,
    *,
    require_non_zero_rationales: bool = True,
) -> tuple[dict, RationaleCoverage]:
    entry_id = payload.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id:
        raise ValueError("Each result row must include a non-empty `entry_id`.")

    scores_payload = payload.get("scores")
    if not isinstance(scores_payload, dict):
        raise ValueError(f"{entry_id} is missing a `scores` object.")

    scores = AlignmentScores.model_validate(scores_payload)
    score_map = {
        value_name: getattr(scores, value_name)
        for value_name in SCHWARTZ_VALUE_ORDER
    }
    rationales = normalize_rationales(payload.get("rationales"))

    non_zero_dimensions = [
        value_name
        for value_name in SCHWARTZ_VALUE_ORDER
        if score_map[value_name] != 0
    ]
    missing_rationales = [
        value_name
        for value_name in non_zero_dimensions
        if value_name not in rationales
    ]
    if require_non_zero_rationales and missing_rationales:
        raise ValueError(
            f"{entry_id} is missing rationales for non-zero scores: "
            + ", ".join(missing_rationales)
        )

    covered_rationales = [
        value_name
        for value_name in non_zero_dimensions
        if value_name in rationales
    ]
    coverage = RationaleCoverage(
        non_zero_score_count=len(non_zero_dimensions),
        non_zero_rationale_count=len(covered_rationales),
    )
    return {
        "entry_id": entry_id,
        "scores": score_map,
        "rationales": rationales,
    }, coverage


def load_manifest_ids(manifest_path: Path) -> set[str]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "entry_id" not in (reader.fieldnames or []):
            raise ValueError("Manifest CSV must contain an `entry_id` column.")
        return {row["entry_id"] for row in reader}


def load_expected_ids(expected_jsonl_path: Path) -> list[str]:
    expected_ids: list[str] = []
    for row in read_jsonl(expected_jsonl_path):
        entry_id = row.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError(f"{expected_jsonl_path} contains an invalid entry_id.")
        expected_ids.append(entry_id)
    return expected_ids


def validate_result_rows_against_ids(
    rows: list[dict],
    *,
    valid_manifest_ids: set[str],
    expected_entry_ids: list[str],
    require_non_zero_rationales: bool = True,
) -> tuple[list[dict], RationaleCoverage]:
    normalized_rows: list[dict] = []
    observed_ids: list[str] = []
    expected_order = {
        entry_id: index for index, entry_id in enumerate(expected_entry_ids)
    }
    non_zero_score_count = 0
    non_zero_rationale_count = 0

    for payload in rows:
        entry_id = payload.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id:
            raise ValueError("Each result row must include a non-empty `entry_id`.")
        if entry_id not in valid_manifest_ids:
            raise ValueError(f"Unknown entry_id: {entry_id}")
        normalized_row, coverage = normalize_result_payload(
            payload,
            require_non_zero_rationales=require_non_zero_rationales,
        )
        normalized_rows.append(normalized_row)
        observed_ids.append(entry_id)
        non_zero_score_count += coverage.non_zero_score_count
        non_zero_rationale_count += coverage.non_zero_rationale_count

    duplicates = [
        entry_id for entry_id, count in Counter(observed_ids).items() if count > 1
    ]
    if duplicates:
        raise ValueError(
            "Duplicate entry_ids detected: "
            + ", ".join(sorted(duplicates)[:5])
        )

    expected_set = set(expected_entry_ids)
    observed_set = set(observed_ids)
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    if missing or extra:
        raise ValueError(
            "Result rows do not match expected ids. "
            f"Missing={missing[:5]}, extra={extra[:5]}"
        )

    normalized_rows.sort(key=lambda row: expected_order[row["entry_id"]])
    return normalized_rows, RationaleCoverage(
        non_zero_score_count=non_zero_score_count,
        non_zero_rationale_count=non_zero_rationale_count,
    )


def hash_bytes(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def hash_file(path: Path) -> str:
    return hash_bytes(path.read_bytes())


def compute_score_only_hash(rows: Iterable[dict]) -> str:
    normalized_lines: list[str] = []
    for row in sorted(rows, key=lambda item: item["entry_id"]):
        normalized_lines.append(
            json.dumps(
                {
                    "entry_id": row["entry_id"],
                    "scores": {
                        value_name: int(row["scores"][value_name])
                        for value_name in SCHWARTZ_VALUE_ORDER
                    },
                },
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    payload = "\n".join(normalized_lines).encode("utf-8")
    return hash_bytes(payload)


def aggregate_rationale_coverage(rows: Iterable[dict]) -> RationaleCoverage:
    non_zero_score_count = 0
    non_zero_rationale_count = 0
    for row in rows:
        for value_name in SCHWARTZ_VALUE_ORDER:
            if int(row["scores"][value_name]) == 0:
                continue
            non_zero_score_count += 1
            if value_name in row["rationales"]:
                non_zero_rationale_count += 1
    return RationaleCoverage(
        non_zero_score_count=non_zero_score_count,
        non_zero_rationale_count=non_zero_rationale_count,
    )


def count_entry_vector_differences(
    left_rows: Iterable[dict],
    right_rows: Iterable[dict],
) -> int:
    left_lookup = {
        row["entry_id"]: tuple(
            int(row["scores"][value_name]) for value_name in SCHWARTZ_VALUE_ORDER
        )
        for row in left_rows
    }
    right_lookup = {
        row["entry_id"]: tuple(
            int(row["scores"][value_name]) for value_name in SCHWARTZ_VALUE_ORDER
        )
        for row in right_rows
    }
    if left_lookup.keys() != right_lookup.keys():
        raise ValueError("Cannot compare pass vectors with different entry_id sets.")
    return sum(
        left_lookup[entry_id] != right_lookup[entry_id]
        for entry_id in left_lookup
    )
