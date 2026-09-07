"""Validate frozen Phase 0A inputs before preparing a new development run."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from scripts.experiments.north_star_phase0 import (
    CODE_REVISION,
    CONSENSUS,
    DOCUMENT_TEMPLATE,
    EPISODES,
    KS,
    LABELS,
    MODEL,
    REVISION,
    VALUES,
    earlier_entries,
    read_cohort,
)
from scripts.experiments.north_star_phase0b import QUOTE_REVIEW
from src.north_star.review import (
    REVIEW_PROMPT_VERSION,
    SourceEntry,
    review_json_schema,
)
from src.wrangling.parse_wrangled_data import parse_wrangled_file


def _hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"Missing or unreadable frozen source: {path}") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Missing or invalid frozen JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"Frozen JSON must be an object: {path}")
    return result


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"Frozen input escapes repository root: {path}") from exc


def _check_hash(path: Path, expected: str) -> None:
    if _hash(path) != expected:
        raise ValueError(f"Frozen source hash mismatch: {path}")


def _index(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid stored order for {label}")
    return value


def _date(value: Any, label: str) -> str:
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError(f"Invalid date for {label}")
    return value


def _validate_config(root: Path, config: dict) -> None:
    values = yaml.safe_load((root / VALUES).read_text())["values"]
    queries = {
        key.lower().replace("-", "_"): {
            "user_phrase": value["user_phrase"].strip(),
            "definition": value["definition"].strip(),
            "text": f"search_query: {value['user_phrase'].strip()}. "
            f"{value['definition'].strip()}",
        }
        for key, value in values.items()
    }
    expected = {
        "model": MODEL,
        "revision": REVISION,
        "code_revision": CODE_REVISION,
        "device": "cpu",
        "dimensions": 256,
        "normalization": "layer_norm -> truncate_256 -> L2",
        "document_template": DOCUMENT_TEMPLATE,
        "queries": queries,
        "batch_size": 8,
        "ks": list(KS),
        "tie_break": "cosine descending, t_index descending, entry_id ascending",
    }
    if config != expected:
        raise ValueError("Frozen retrieval configuration differs from Phase 0A inputs")


def _history(root: Path, persona: str) -> list[dict]:
    path = root / f"logs/wrangled/persona_{persona}.md"
    title = re.search(r"^# Persona ([a-f0-9]+):", path.read_text(), re.MULTILINE)
    if title is None or title.group(1) != persona:
        raise ValueError(f"Source owner mismatch for {persona}")
    profile, entries, warnings = parse_wrangled_file(path)
    if profile.get("persona_id") != persona:
        raise ValueError(f"Parsed source owner mismatch for {persona}")
    if warnings:
        raise ValueError(f"Unresolved source parse warnings for {persona}")
    last_index, last_date = -1, ""
    for row in entries:
        index = _index(row["t_index"], persona)
        entry_date = _date(row["date"], persona)
        if index <= last_index or entry_date < last_date:
            raise ValueError(
                f"Duplicate or inconsistent source order/date for {persona}"
            )
        if (
            not isinstance(row["initial_entry"], str)
            or not row["initial_entry"].strip()
        ):
            raise ValueError(f"Missing source text for {persona}:{index}")
        last_index, last_date = index, entry_date
    return entries


def _validate_episode(episode: dict, entries: list[dict]) -> None:
    by_index = {row["t_index"]: row for row in entries}
    positions = {row["t_index"]: i + 1 for i, row in enumerate(entries)}
    last_index, last_date = -1, ""
    points = ["onset", "confirmation", "end"]
    if (
        episode.get("cutoff_t_index") is not None
        or episode.get("cutoff_date") is not None
    ):
        points.append("cutoff")
    for point in points:
        index = _index(episode[f"{point}_t_index"], point)
        point_date = _date(episode[f"{point}_date"], point)
        if index not in by_index or by_index[index]["date"] != point_date:
            raise ValueError(f"Episode {point} differs from source order/date")
        if index < last_index or point_date < last_date:
            raise ValueError(f"Episode {point} chronology is inconsistent")
        if (
            f"{point}_position" in episode
            and episode[f"{point}_position"] != positions[index]
        ):
            raise ValueError(f"Episode {point} position differs from source order")
        last_index, last_date = index, point_date


def _case(case: dict, entries: list[dict], config: dict) -> dict:
    episode = case["episode"]
    persona = episode["persona_id"]
    _validate_episode(episode, entries)
    eligible = {
        f"{persona}:entry:{row['t_index']}": row
        for row in earlier_entries(entries, episode)
    }
    ranking = case["ranking"]
    ids = [row["entry_id"] for row in ranking]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ranked source in {episode['episode_id']}")
    if set(ids) != set(eligible):
        raise ValueError(f"Ranking differs from complete eligible sources: {persona}")
    sources = []
    for row in ranking:
        entry = eligible[row["entry_id"]]
        if _index(row["t_index"], "ranking") != entry["t_index"]:
            raise ValueError("Ranked source stored order mismatch")
        if row["date"] != entry["date"]:
            raise ValueError("Ranked source date mismatch")
        if (
            row["source_sha256"]
            != hashlib.sha256(entry["initial_entry"].encode()).hexdigest()
        ):
            raise ValueError("Ranked source text hash mismatch")
        if type(row["similarity"]) not in (int, float) or not math.isfinite(
            row["similarity"]
        ):
            raise ValueError("Invalid retrieval similarity")
        sources.append(
            SourceEntry(
                entry_id=row["entry_id"],
                journal_entry=entry["initial_entry"],
                nudge_response=None,
            ).model_dump()
        )
    if ranking != sorted(
        ranking,
        key=lambda row: (-row["similarity"], -row["t_index"], row["entry_id"]),
    ):
        raise ValueError("Ranking violates frozen similarity and tie-break order")
    return {
        "case_id": episode["episode_id"],
        "episode": episode,
        "core_value": episode["dimension"],
        "value": config["queries"][episode["dimension"]],
        "all_eligible_sources_in_retrieval_order": sources,
        "runtime_entry_ids": ids[:3],
        "case_categories": [
            "no_earlier_writing"
            if not sources
            else "one_earlier_entry"
            if len(sources) == 1
            else "multiple_earlier_entries",
            episode["dimension"],
        ],
    }


def build_manifest(*, root: Path, retrieval_path: Path, policy_path: Path) -> dict:
    """Return a verified manifest without writing files or parsing reserved writing."""
    try:
        return _build_manifest(
            root=root, retrieval_path=retrieval_path, policy_path=policy_path
        )
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("Malformed frozen Phase 0A input structure") from exc


def _build_manifest(*, root: Path, retrieval_path: Path, policy_path: Path) -> dict:
    result = _read_json(retrieval_path)
    retrieval = result["retrieval"]
    if result["schema_version"] != "north-star-phase0-v1":
        raise ValueError("Unsupported Phase 0A schema")
    if retrieval["gate_passed"] is not True or retrieval["selected_k"] != 3:
        raise ValueError("Phase 0A must pass and freeze k=3 before Phase 0B")
    cohort_path = retrieval_path.with_name("cohort.json")
    config_path = retrieval_path.with_name("retrieval_config.json")
    for name, path in (("cohort", cohort_path), ("config", config_path)):
        _check_hash(path, retrieval[f"{name}_sha256"])
        if _read_json(path) != retrieval[name]:
            raise ValueError(f"Embedded {name} differs from frozen file")
    cohort = retrieval["cohort"]
    groups = [cohort["development_persona_ids"], cohort["reserved_persona_ids"]]
    if any(len(group) != len(set(group)) for group in groups):
        raise ValueError("Duplicate Persona in frozen cohort")
    personas = (
        set(groups[0]) | set(groups[1]) | set(cohort.get("excluded_personas", {}))
    )
    if any(re.fullmatch(r"[a-f0-9]+", persona) is None for persona in personas):
        raise ValueError("Invalid canonical Persona identifier")
    required = {
        str(EPISODES),
        str(LABELS),
        str(CONSENSUS),
        str(VALUES),
        "scripts/experiments/north_star_phase0.py",
        *(f"logs/wrangled/persona_{persona}.md" for persona in personas),
    }
    source_hashes = result["sources"]
    if set(source_hashes) != required:
        raise ValueError("Frozen source inventory is incomplete or unexpected")
    # Byte hashes may include reserved files; no reserved writing is parsed.
    for relative, expected in source_hashes.items():
        path = root / relative
        if _relative(root, path) != relative:
            raise ValueError("Noncanonical frozen source path")
        _check_hash(path, expected)
    if cohort["source_episode_sha256"] != source_hashes[str(EPISODES)]:
        raise ValueError("Cohort episode hash differs from frozen source")
    episodes = pl.read_parquet(root / EPISODES).to_dicts()
    _, development = read_cohort(cohort_path, {row["persona_id"] for row in episodes})
    _validate_config(root, retrieval["config"])
    expected_episodes = {
        row["episode_id"]: row for row in episodes if row["persona_id"] in development
    }
    if len(expected_episodes) != sum(
        row["persona_id"] in development for row in episodes
    ):
        raise ValueError("Duplicate source development episode")
    cases = retrieval["cases"]
    ids = [case["episode"]["episode_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate development case")
    if set(ids) != set(expected_episodes) or retrieval["development_episodes"] != len(
        ids
    ):
        raise ValueError("Cases differ from complete development episode cohort")
    for case in cases:
        if case["episode"] != expected_episodes[case["episode"]["episode_id"]]:
            raise ValueError("Frozen case episode differs from source episode")
    histories = {persona: _history(root, persona) for persona in sorted(development)}
    frozen_cases = [
        _case(case, histories[case["episode"]["persona_id"]], retrieval["config"])
        for case in cases
    ]
    unique_sources = {
        row["entry_id"]
        for case in frozen_cases
        for row in case["all_eligible_sources_in_retrieval_order"]
    }
    if retrieval["unique_documents"] != len(unique_sources):
        raise ValueError("Frozen unique source count differs from rankings")
    hashes = dict(source_hashes)
    for path in (retrieval_path, cohort_path, config_path, policy_path):
        hashes[_relative(root, path)] = _hash(path)
    return {
        "schema_version": "north-star-development-v2",
        "frozen_at": datetime.now(UTC).isoformat(),
        "cases": frozen_cases,
        "case_count": len(frozen_cases),
        "sampling": (
            "All episodes from the frozen Phase0A development group; no subsampling"
        ),
        "seed": cohort["seed"],
        "source_hashes": hashes,
        "prompt_version": REVIEW_PROMPT_VERSION,
        "schema": review_json_schema(),
        "reference_protocol": (
            "Gemini independently reviews every eligible original Journal Entry, "
            "without runtime decisions or labels. Legacy responses are excluded. "
            "Exact displayed quotes differing from its reference quotation receive "
            "a second Gemini check under the frozen candidate-quotation instruction. "
            "Acceptance requires both primary support and candidate-quote approval; "
            "disagreement or abstention remains incorrect. No earlier sources means "
            "no call and a separately counted structurally empty case."
        ),
        "quote_review_instruction": QUOTE_REVIEW,
        "retrieval_only_rule": (
            "Display entire original writing of the first ranked source if it passes "
            "deterministic quotation checks; semantic acceptance uses the exhaustive "
            "reference decision for that source. No source or a failed check omits it."
        ),
        "criteria": {
            "incorrect_displayed": 0,
            "correct_no_card_rate": 1.0,
            "unexpected_provider_failure_rate_max": 0.05,
            "require_reference_confirmed_no_example_with_earlier_sources": True,
            "require_saved_persona_acceptance": True,
        },
        "integration_checks": (
            "Source chronology and lifecycle adversaries use separate injected tests, "
            "not provider failure denominators."
        ),
        "availability_validation": (
            "Every ranked source matches its original text hash, canonical owner, "
            "date and stored order, and the ranking contains every eligible source. "
            "Sources precede onset in stored order and do not exceed its date; "
            "onset, confirmation and end match the source history. Reserved Persona "
            "files receive byte-hash checks only and are never parsed."
        ),
    }
