"""Paired Codex review helpers for the twinkl-748 Hedonism hard-set."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.synthetic.generation import SCHWARTZ_BANNED_PATTERN

TARGET_VERSION = "twinkl-748-hedonism-hard-set-v1"
SPEC_VERSION = 1
REVIEW_SCHEMA_VERSION = "twinkl-748-review-v1"
REVIEW_PROMPT_VERSION = "twinkl-748-packet-only-v1"

CONFIDENCE_LEVELS = frozenset({"high", "medium", "low"})
PAIR_DECISIONS = frozenset({"yes", "no", "uncertain"})
ISSUE_CODES = frozenset(
    {
        "ambiguous",
        "implausible",
        "lexical_leakage",
        "unintended_multi_value_change",
        "not_comparable",
    }
)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _aware_timestamp(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be a timezone-aware ISO timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return parsed


def _normalise_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def _word_count(text: str) -> int:
    return len(_normalise_text(text).split())


def _similarity(left: str, right: str) -> float:
    left_tokens = _normalise_text(left).split()
    right_tokens = _normalise_text(right).split()
    overlap = sum((Counter(left_tokens) & Counter(right_tokens)).values())
    return overlap / max(len(left_tokens), len(right_tokens))


def load_candidate_spec(path: str | Path) -> dict[str, Any]:
    """Load and fail closed on the parent-controlled candidate specification."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Candidate spec must be a mapping")
    if payload.get("schema_version") != SPEC_VERSION:
        raise ValueError(f"Candidate spec must use schema_version {SPEC_VERSION}")
    if payload.get("target_version") != TARGET_VERSION:
        raise ValueError(f"Candidate spec must name {TARGET_VERSION}")
    if payload.get("review_prompt_version") != REVIEW_PROMPT_VERSION:
        raise ValueError(f"Candidate spec must name {REVIEW_PROMPT_VERSION}")
    if payload.get("core_values") != ["Hedonism"]:
        raise ValueError("Candidate spec must use core_values: [Hedonism]")

    gates = payload.get("quality_gates")
    if not isinstance(gates, dict):
        raise ValueError("Candidate spec is missing quality_gates")
    maximum_delta = float(gates.get("maximum_word_count_delta_ratio", -1))
    minimum_similarity = float(gates.get("minimum_normalized_similarity", 2))
    if not 0 <= maximum_delta <= 1:
        raise ValueError("maximum_word_count_delta_ratio must be in [0, 1]")
    if not 0 <= minimum_similarity <= 1:
        raise ValueError("minimum_normalized_similarity must be in [0, 1]")

    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not 20 <= len(pairs) <= 30:
        raise ValueError("Candidate spec must contain 20-30 pairs")

    pair_ids: set[str] = set()
    variant_ids: set[str] = set()
    family_counts: dict[str, int] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError("Every pair must be a mapping")
        pair_id = str(pair.get("pair_id", "")).strip()
        family = str(pair.get("family", "")).strip()
        if not pair_id or pair_id in pair_ids:
            raise ValueError(f"Missing or duplicate pair_id: {pair_id!r}")
        if not family:
            raise ValueError(f"Pair {pair_id} is missing family")
        pair_ids.add(pair_id)
        family_counts[family] = family_counts.get(family, 0) + 1

        variants = pair.get("variants")
        if not isinstance(variants, list) or len(variants) != 2:
            raise ValueError(f"Pair {pair_id} must contain exactly two variants")
        labels: set[int] = set()
        texts: list[str] = []
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError(f"Pair {pair_id} has a malformed variant")
            variant_id = str(variant.get("variant_id", "")).strip()
            label = variant.get("author_label")
            text = str(variant.get("text", "")).strip()
            if not variant_id or variant_id in variant_ids:
                raise ValueError(
                    f"Missing or duplicate variant_id in {pair_id}: {variant_id!r}"
                )
            if type(label) is not int or label not in {-1, 1}:
                raise ValueError(
                    f"Pair {pair_id} author labels must be -1 and +1"
                )
            if not text:
                raise ValueError(f"Variant {variant_id} has no text")
            leaked = SCHWARTZ_BANNED_PATTERN.search(text)
            if leaked:
                raise ValueError(
                    f"Variant {variant_id} leaks banned term {leaked.group(0)!r}"
                )
            variant_ids.add(variant_id)
            labels.add(int(label))
            texts.append(text)
        if labels != {-1, 1}:
            raise ValueError(f"Pair {pair_id} must contrast -1 and +1")

        counts = [_word_count(text) for text in texts]
        delta_ratio = abs(counts[0] - counts[1]) / max(counts)
        if delta_ratio > maximum_delta:
            raise ValueError(
                f"Pair {pair_id} word-count delta {delta_ratio:.3f} exceeds "
                f"{maximum_delta:.3f}"
            )
        similarity = _similarity(*texts)
        if similarity < minimum_similarity:
            raise ValueError(
                f"Pair {pair_id} similarity {similarity:.3f} is below "
                f"{minimum_similarity:.3f}"
            )

    if len(family_counts) < 4:
        raise ValueError("Candidate spec must cover at least four scenario families")
    return payload


def _response_schema() -> dict[str, Any]:
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "target_version": TARGET_VERSION,
        "reviewer_prompt_version": REVIEW_PROMPT_VERSION,
        "response_array_field": "pair_reviews",
        "submission_fields": {
            "packet_sha256": "SHA-256 of the exact blind_packet.json reviewed.",
            "reviewer_id": "Distinct reviewer identifier.",
            "reviewed_at": "Timezone-aware ISO-8601 timestamp.",
            "reviewer_runtime": "Actual model/runtime if known; otherwise unknown.",
            "requested_runtime": "5.6 Sol / Light (request may be unenforceable).",
        },
        "pair_fields": {
            "review_pair_id": "Opaque identifier from blind_packet.json.",
            "entry_reviews": "One review for each opaque entry.",
            "comparable_except_choice": sorted(PAIR_DECISIONS),
            "realistic": sorted(PAIR_DECISIONS),
            "issue_codes": sorted(ISSUE_CODES),
            "accept_pair": sorted(PAIR_DECISIONS),
            "pair_rationale": "Short evidence-based explanation.",
        },
        "entry_fields": {
            "review_entry_id": "Opaque identifier from blind_packet.json.",
            "hedonism_label": [-1, 0, 1, "uncertain"],
            "confidence": sorted(CONFIDENCE_LEVELS),
            "rationale": "Explain the behavior or choice shown in the text.",
        },
    }


def build_review_bundle(
    *,
    spec_path: str | Path,
    output_dir: str | Path,
    root: str | Path,
    random_seed: int = 748,
) -> dict[str, Any]:
    """Write a blinded reviewer packet and parent-only reconciliation controls."""
    spec_path = Path(spec_path)
    output_dir = Path(output_dir)
    root = Path(root)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite review bundle: {output_dir}")
    spec = load_candidate_spec(spec_path)

    rng = random.Random(random_seed)
    source_pairs = list(spec["pairs"])
    rng.shuffle(source_pairs)
    packet_cases = []
    key_cases = []
    for pair_index, pair in enumerate(source_pairs, start=1):
        review_pair_id = f"pair_{pair_index:03d}"
        variants = list(pair["variants"])
        rng.shuffle(variants)
        packet_entries = []
        key_entries = []
        for entry_index, variant in enumerate(variants, start=1):
            review_entry_id = f"entry_{pair_index:03d}_{entry_index}"
            packet_entries.append(
                {
                    "review_entry_id": review_entry_id,
                    "journal_entry": variant["text"],
                }
            )
            key_entries.append(
                {
                    "review_entry_id": review_entry_id,
                    "variant_id": variant["variant_id"],
                    "author_label": variant["author_label"],
                }
            )
        packet_cases.append(
            {
                "review_pair_id": review_pair_id,
                "core_values": ["Hedonism"],
                "entries": packet_entries,
            }
        )
        key_cases.append(
            {
                "review_pair_id": review_pair_id,
                "pair_id": pair["pair_id"],
                "family": pair["family"],
                "isolated_behavior": pair["isolated_behavior"],
                "affect_band": pair["affect_band"],
                "secondary_value_risk": pair["secondary_value_risk"],
                "entries": key_entries,
            }
        )

    packet = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "target_version": TARGET_VERSION,
        "reviewer_prompt_version": REVIEW_PROMPT_VERSION,
        "purpose": (
            "Codex-reviewed diagnostic of whether close journal-entry pairs differ "
            "cleanly in behavior that supports or conflicts with Hedonism."
        ),
        "rules": [
            (
                "Score Hedonism independently: choosing present pleasure or comfort "
                "can be +1 even when another commitment is harmed."
            ),
            (
                "Use -1 only for clear self-denial, cancellation, or rejection of "
                "available pleasure, rest, or comfort."
            ),
            (
                "Use 0 when neither direction is clearly shown and uncertain when "
                "ambiguous."
            ),
            (
                "Accept a pair only if both entries are realistic and differ mainly "
                "in the behavior or choice relevant to Hedonism."
            ),
            (
                "Do not infer or search for author intent, model predictions, or "
                "source metadata."
            ),
        ],
        "cases": packet_cases,
    }
    key = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "target_version": TARGET_VERSION,
        "random_seed": random_seed,
        "cases": key_cases,
    }

    reviewer_dir = output_dir / "reviewer_packet"
    parent_dir = output_dir / "parent_control"
    reviewer_dir.mkdir(parents=True)
    parent_dir.mkdir(parents=True)
    packet_path = reviewer_dir / "blind_packet.json"
    schema_path = reviewer_dir / "response_schema.json"
    key_path = parent_dir / "reconciliation_key.json"
    packet_path.write_text(json.dumps(packet, indent=2) + "\n", encoding="utf-8")
    schema_path.write_text(
        json.dumps(_response_schema(), indent=2) + "\n", encoding="utf-8"
    )
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "target_version": TARGET_VERSION,
        "evidence_class": "Codex-reviewed diagnostic; not human validation",
        "technical_isolation": False,
        "requested_reviewer_runtime": "5.6 Sol / Light",
        "runtime_control_available": False,
        "protocol_limitations": [
            (
                "The subagent launcher does not expose model or reasoning-level "
                "selectors, so the requested runtime cannot be enforced."
            ),
            (
                "Packet-only instructions provide controlled disclosure in a shared "
                "workspace, not enforced filesystem isolation."
            ),
            (
                "Agreement between Codex reviewers is shared-model consistency, "
                "not human validity."
            ),
        ],
        "source_spec_path": _relative_to_root(spec_path, root),
        "source_spec_sha256": sha256_file(spec_path),
        "blind_packet_path": _relative_to_root(packet_path, root),
        "blind_packet_sha256": sha256_file(packet_path),
        "response_schema_path": _relative_to_root(schema_path, root),
        "response_schema_sha256": sha256_file(schema_path),
        "reconciliation_key_path": _relative_to_root(key_path, root),
        "reconciliation_key_sha256": sha256_file(key_path),
        "pair_count": len(packet_cases),
        "entry_count": 2 * len(packet_cases),
        "reviewers": [],
        "materialization_complete": False,
    }
    manifest_path = parent_dir / "audit_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _case_lookup(cases: Any, *, source: str) -> dict[str, dict[str, Any]]:
    if not isinstance(cases, list):
        raise ValueError(f"{source} cases must be a list")
    lookup: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError(f"{source} contains a malformed case")
        case_id = str(case.get("review_pair_id", ""))
        if not case_id or case_id in lookup:
            raise ValueError(f"{source} has a missing or duplicate review_pair_id")
        lookup[case_id] = case
    return lookup


def validate_review_response(
    response: dict[str, Any],
    *,
    packet: dict[str, Any],
    packet_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Validate exact response provenance and complete pair/entry coverage."""
    expected = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "target_version": TARGET_VERSION,
        "reviewer_prompt_version": REVIEW_PROMPT_VERSION,
        "packet_sha256": packet_sha256,
    }
    for field, value in expected.items():
        if response.get(field) != value:
            raise ValueError(f"Reviewer response has invalid {field}")
    reviewer_id = str(response.get("reviewer_id", "")).strip()
    if not reviewer_id:
        raise ValueError("Reviewer response is missing reviewer_id")
    _aware_timestamp(response.get("reviewed_at"), field_name="reviewed_at")
    if not str(response.get("reviewer_runtime", "")).strip():
        raise ValueError("Reviewer response is missing reviewer_runtime")

    packet_cases = _case_lookup(packet.get("cases"), source="packet")
    response_case_rows = response.get("pair_reviews")
    if response_case_rows is None:
        response_case_rows = response.get("cases")
    response_cases = _case_lookup(response_case_rows, source="reviewer response")
    if set(response_cases) != set(packet_cases):
        raise ValueError("Reviewer response does not exactly cover packet cases")

    for case_id, packet_case in packet_cases.items():
        case = response_cases[case_id]
        expected_entry_ids = {
            str(entry["review_entry_id"]) for entry in packet_case["entries"]
        }
        entry_reviews = case.get("entry_reviews")
        if not isinstance(entry_reviews, list):
            raise ValueError(f"Case {case_id} entry_reviews must be a list")
        entry_lookup: dict[str, dict[str, Any]] = {}
        for entry in entry_reviews:
            entry_id = str(entry.get("review_entry_id", ""))
            if not entry_id or entry_id in entry_lookup:
                raise ValueError(f"Case {case_id} has duplicate entry reviews")
            label = entry.get("hedonism_label")
            valid_label = label == "uncertain" or (
                type(label) is int and label in {-1, 0, 1}
            )
            if not valid_label:
                raise ValueError(f"Case {case_id} has invalid Hedonism label")
            if entry.get("confidence") not in CONFIDENCE_LEVELS:
                raise ValueError(f"Case {case_id} has invalid confidence")
            if not str(entry.get("rationale", "")).strip():
                raise ValueError(f"Case {case_id} has an empty entry rationale")
            entry_lookup[entry_id] = entry
        if set(entry_lookup) != expected_entry_ids:
            raise ValueError(f"Case {case_id} does not exactly cover packet entries")
        for field in ("comparable_except_choice", "realistic", "accept_pair"):
            if case.get(field) not in PAIR_DECISIONS:
                raise ValueError(f"Case {case_id} has invalid {field}")
        issue_codes = case.get("issue_codes")
        if not isinstance(issue_codes, list) or not set(issue_codes) <= ISSUE_CODES:
            raise ValueError(f"Case {case_id} has invalid issue_codes")
        if len(issue_codes) != len(set(issue_codes)):
            raise ValueError(f"Case {case_id} has duplicate issue_codes")
        if not str(case.get("pair_rationale", "")).strip():
            raise ValueError(f"Case {case_id} has an empty pair rationale")
        case["_entry_lookup"] = entry_lookup
    return response_cases


def materialize_reviewed_hard_set(
    *,
    bundle_dir: str | Path,
    reviewer_a_path: str | Path,
    reviewer_b_path: str | Path,
) -> dict[str, Any]:
    """Reconcile paired reviews and freeze only fully agreed, accepted pairs."""
    bundle_dir = Path(bundle_dir)
    reviewer_a_path = Path(reviewer_a_path)
    reviewer_b_path = Path(reviewer_b_path)
    if reviewer_a_path.resolve() == reviewer_b_path.resolve():
        raise ValueError("Paired reviews must use distinct response files")
    reviewer_dir = bundle_dir / "reviewer_packet"
    parent_dir = bundle_dir / "parent_control"
    packet_path = reviewer_dir / "blind_packet.json"
    schema_path = reviewer_dir / "response_schema.json"
    key_path = parent_dir / "reconciliation_key.json"
    manifest_path = parent_dir / "audit_manifest.json"
    packet = _load_json(packet_path)
    response_schema = _load_json(schema_path)
    key = _load_json(key_path)
    manifest = _load_json(manifest_path)
    for path_field, digest_field, path in (
        ("blind_packet_path", "blind_packet_sha256", packet_path),
        ("response_schema_path", "response_schema_sha256", schema_path),
        ("reconciliation_key_path", "reconciliation_key_sha256", key_path),
    ):
        if (
            not manifest.get(path_field)
            or manifest.get(digest_field) != sha256_file(path)
        ):
            raise ValueError(f"Audit manifest provenance mismatch for {path.name}")
    if manifest.get("materialization_complete"):
        raise ValueError("Review bundle is already materialized")

    packet_sha = sha256_file(packet_path)
    response_a = _load_json(reviewer_a_path)
    response_b = _load_json(reviewer_b_path)
    if (
        "response_array_field" not in response_schema
        and "pair_reviews" in response_a
        and "pair_reviews" in response_b
    ):
        manifest["protocol_limitations"].append(
            "The v1 response schema omitted the top-level review-array field; "
            "both reviewers independently used pair_reviews and the materializer "
            "accepted that shared interpretation."
        )
    if response_a.get("reviewer_id") == response_b.get("reviewer_id"):
        raise ValueError("Paired reviews must use distinct reviewer IDs")
    cases_a = validate_review_response(
        response_a, packet=packet, packet_sha256=packet_sha
    )
    cases_b = validate_review_response(
        response_b, packet=packet, packet_sha256=packet_sha
    )
    key_cases = _case_lookup(key.get("cases"), source="reconciliation key")
    packet_cases = _case_lookup(packet.get("cases"), source="packet")
    if set(key_cases) != set(packet_cases):
        raise ValueError("Reconciliation key does not exactly cover packet cases")

    frozen_rows: list[dict[str, Any]] = []
    reconciliation_rows: list[dict[str, Any]] = []
    label_agreements = 0
    total_entries = 0
    accepted_pairs = 0
    author_label_matches = 0
    accepted_entries = 0
    for review_pair_id, packet_case in packet_cases.items():
        case_a = cases_a[review_pair_id]
        case_b = cases_b[review_pair_id]
        key_case = key_cases[review_pair_id]
        key_entries = {
            str(entry["review_entry_id"]): entry for entry in key_case["entries"]
        }
        agreed_labels: dict[str, int | str] = {}
        pair_label_agreement = True
        labels_are_opposite = True
        for packet_entry in packet_case["entries"]:
            entry_id = str(packet_entry["review_entry_id"])
            label_a = case_a["_entry_lookup"][entry_id]["hedonism_label"]
            label_b = case_b["_entry_lookup"][entry_id]["hedonism_label"]
            total_entries += 1
            if label_a == label_b:
                label_agreements += 1
                agreed_labels[entry_id] = label_a
            else:
                pair_label_agreement = False
        if set(agreed_labels.values()) != {-1, 1}:
            labels_are_opposite = False
        accepted = (
            pair_label_agreement
            and labels_are_opposite
            and case_a["comparable_except_choice"] == "yes"
            and case_b["comparable_except_choice"] == "yes"
            and case_a["realistic"] == "yes"
            and case_b["realistic"] == "yes"
            and case_a["accept_pair"] == "yes"
            and case_b["accept_pair"] == "yes"
            and not case_a["issue_codes"]
            and not case_b["issue_codes"]
        )
        if accepted:
            accepted_pairs += 1
            for packet_entry in packet_case["entries"]:
                entry_id = str(packet_entry["review_entry_id"])
                target = int(agreed_labels[entry_id])
                key_entry = key_entries[entry_id]
                author_match = target == int(key_entry["author_label"])
                author_label_matches += int(author_match)
                accepted_entries += 1
                frozen_rows.append(
                    {
                        "review_pair_id": review_pair_id,
                        "review_entry_id": entry_id,
                        "source_pair_id": key_case["pair_id"],
                        "source_variant_id": key_entry["variant_id"],
                        "family": key_case["family"],
                        "core_values": ["Hedonism"],
                        "journal_entry": packet_entry["journal_entry"],
                        "hedonism_target": target,
                        "reviewer_a_confidence": case_a["_entry_lookup"][entry_id][
                            "confidence"
                        ],
                        "reviewer_b_confidence": case_b["_entry_lookup"][entry_id][
                            "confidence"
                        ],
                        "author_label_match": author_match,
                    }
                )
        reconciliation_rows.append(
            {
                "review_pair_id": review_pair_id,
                "source_pair_id": key_case["pair_id"],
                "family": key_case["family"],
                "label_agreement": pair_label_agreement,
                "opposite_non_neutral_labels": labels_are_opposite,
                "reviewer_a_accept": case_a["accept_pair"],
                "reviewer_b_accept": case_b["accept_pair"],
                "reviewer_a_issues": case_a["issue_codes"],
                "reviewer_b_issues": case_b["issue_codes"],
                "accepted": accepted,
            }
        )

    frozen = pl.DataFrame(frozen_rows) if frozen_rows else pl.DataFrame()
    reconciliation = pl.DataFrame(reconciliation_rows)
    frozen_path = parent_dir / "frozen_hedonism_hard_set.parquet"
    reconciliation_path = parent_dir / "reconciliation.parquet"
    frozen.write_parquet(frozen_path)
    reconciliation.write_parquet(reconciliation_path)
    summary = {
        "target_version": TARGET_VERSION,
        "evidence_class": "Codex-reviewed diagnostic; not human validation",
        "pair_count": len(packet_cases),
        "entry_count": total_entries,
        "entry_label_agreement_count": label_agreements,
        "entry_label_agreement_rate": label_agreements / total_entries,
        "accepted_pair_count": accepted_pairs,
        "excluded_pair_count": len(packet_cases) - accepted_pairs,
        "accepted_entry_count": accepted_entries,
        "author_label_match_count": author_label_matches,
        "author_label_match_rate": (
            author_label_matches / accepted_entries if accepted_entries else None
        ),
        "frozen_hard_set_sha256": sha256_file(frozen_path),
        "reconciliation_sha256": sha256_file(reconciliation_path),
        "reviewer_a_sha256": sha256_file(reviewer_a_path),
        "reviewer_b_sha256": sha256_file(reviewer_b_path),
    }
    summary_path = parent_dir / "review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest["reviewers"] = [
        {
            "reviewer_id": response_a["reviewer_id"],
            "reviewed_at": response_a["reviewed_at"],
            "reviewer_runtime": response_a["reviewer_runtime"],
            "response_path": str(
                reviewer_a_path.resolve().relative_to(bundle_dir.resolve())
            ),
            "response_sha256": sha256_file(reviewer_a_path),
        },
        {
            "reviewer_id": response_b["reviewer_id"],
            "reviewed_at": response_b["reviewed_at"],
            "reviewer_runtime": response_b["reviewer_runtime"],
            "response_path": str(
                reviewer_b_path.resolve().relative_to(bundle_dir.resolve())
            ),
            "response_sha256": sha256_file(reviewer_b_path),
        },
    ]
    manifest["review_summary_path"] = str(
        summary_path.resolve().relative_to(bundle_dir.resolve())
    )
    manifest["review_summary_sha256"] = sha256_file(summary_path)
    manifest["materialization_complete"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary
