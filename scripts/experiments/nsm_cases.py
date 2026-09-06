"""Build the fresh NSM paired cases and response-inclusive local retrieval.

Case metadata records synthetic provenance; only ``values[].sources`` and the
approved value definition belong in semantic requests. No provider calls or
file writes occur here. The caller persists the returned configuration and
results in the consolidated experiment record.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from scripts.experiments.north_star_phase0 import (
    CODE_REVISION,
    MODEL,
    REVISION,
    rank_entries,
)
from src.demo.contracts import CORE_VALUE_ORDER
from src.drift_detector import detect_drift
from src.north_star import runtime
from src.north_star.provider import stable_hash
from src.north_star.review import SourceEntry
from src.weekly_drift_reviewer import WeeklyDriftReviewerDecision
from src.wrangling.parse_synthetic_data import parse_persona_file
from src.wrangling.parse_wrangled_data import parse_wrangled_file

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = (
    "logs/experiments/artifacts/twinkl_52zz_model_comparison_20260714/prompts.jsonl"
)
RESPONSES = (
    "logs/experiments/artifacts/twinkl_52zz_luna_low_20260714/"
    "responses_gpt_5_6_luna_low.jsonl"
)
VALUES = "config/schwartz_values.yaml"
REGISTRY = "logs/registry/personas.parquet"
DOCUMENT_TEMPLATE = "search_document: Journal Entry:\n{journal_entry}"
RESPONSE_TEMPLATE = "\n\nPersona nudge response:\n{nudge_response}"
AVAILABILITY_CONVENTION = (
    "Synthetic replay only: a parent's date at 00:00 UTC plus 3*t_index "
    "microseconds marks its entry; the nudge and response follow at +1 and +2 "
    "microseconds. These artificial instants encode entry -> nudge -> response "
    "and same-day stored order, not observed historical timestamps. Closed-week "
    "cutoffs are the following Monday at 00:00 UTC."
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal_value(value: str) -> str:
    return value.lower().replace("-", "_")


def _monday(day: str) -> date:
    parsed = date.fromisoformat(day)
    return parsed - timedelta(days=parsed.weekday())


def _timestamp(day: str, sequence: int = 0) -> str:
    instant = datetime.combine(date.fromisoformat(day), datetime.min.time(), UTC)
    return (instant + timedelta(microseconds=sequence)).isoformat()


def _verify_sources(record: dict[str, Any], root: Path, ids: set[str]) -> dict:
    expected: dict[str, str] = {}
    for audit in ("history_readiness", "upstream_weekly_inputs"):
        for field in ("provenance", "source_hashes", "source_file_hashes"):
            for path, digest in record["audits"][audit].get(field, {}).items():
                if path in expected and expected[path] != digest:
                    raise ValueError(f"Preparation source hashes disagree: {path}")
                expected[path] = digest
    required = {
        PROMPTS,
        RESPONSES,
        VALUES,
        REGISTRY,
        "src/drift_detector.py",
        "src/weekly_drift_reviewer.py",
        *(
            f"logs/{kind}/persona_{pid}.md"
            for pid in ids
            for kind in ("synthetic_data", "wrangled")
        ),
    }
    verified = {}
    for relative in sorted(required):
        path = root / relative
        if path.resolve().relative_to(root.resolve()).as_posix() != relative:
            raise ValueError(f"Noncanonical source path: {relative}")
        if relative not in expected or file_hash(path) != expected[relative]:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
        verified[relative] = expected[relative]
    return verified


def _read_rows(path: Path, ids: set[str], *, repeat: int | None = None) -> dict:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["persona_id"] not in ids:
            continue
        if repeat is not None and row["repeat"] != repeat:
            continue
        key = (row["persona_id"], row["week_start"])
        if key in rows:
            raise ValueError(f"Duplicate frozen weekly row: {key}")
        rows[key] = row
    return rows


def _history(root: Path, pid: str, stored_values: list[str]) -> list[dict]:
    profile, entries, warnings = parse_wrangled_file(
        root / f"logs/wrangled/persona_{pid}.md"
    )
    original_profile, originals = parse_persona_file(
        root / f"logs/synthetic_data/persona_{pid}.md"
    )
    if warnings or profile["persona_id"] != pid:
        raise ValueError(f"Invalid wrangled history: {pid}")
    if profile["core_values"] != stored_values or (
        original_profile["core_values"] != stored_values
    ):
        raise ValueError(f"Core Values differ from registry: {pid}")
    fields = ("t_index", "date", "initial_entry", "nudge_text", "response_text")
    if len(entries) != len(originals) or any(
        any(left[field] != right[field] for field in fields)
        for left, right in zip(entries, originals, strict=True)
    ):
        raise ValueError(f"Original and wrangled writing differ: {pid}")
    if [entry["t_index"] for entry in entries] != list(range(len(entries))):
        raise ValueError(f"Noncontiguous source order: {pid}")
    if any(a["date"] > b["date"] for a, b in zip(entries, entries[1:], strict=False)):
        raise ValueError(f"Source dates disagree with stored order: {pid}")
    for entry in entries:
        date.fromisoformat(entry["date"])
        if not entry["initial_entry"] or (
            entry["response_text"] and not entry["nudge_text"]
        ):
            raise ValueError(f"Unavailable or incomplete source: {pid}")
    return entries


def _writing(pid: str, entries: list[dict]) -> list[runtime.SourceWriting]:
    return [
        runtime.SourceWriting(
            owner_id=pid,
            entry_id=f"{pid}:entry:{row['t_index']}",
            t_index=row["t_index"],
            date=row["date"],
            journal_entry=row["initial_entry"],
            nudge_response=row["response_text"] or None,
            available_at=_timestamp(row["date"], 3 * row["t_index"]),
            response_available_at=(
                _timestamp(row["date"], 3 * row["t_index"] + 2)
                if row["response_text"]
                else None
            ),
        )
        for row in entries
    ]


def _upstream_text(entry: dict) -> str:
    """Reproduce frozen Drift text, including its historical AI nudge context."""
    parts = [entry["initial_entry"]]
    if entry["nudge_text"]:
        parts.append(f'Nudge: "{entry["nudge_text"]}"')
    if entry["response_text"]:
        parts.append(f"Response: {entry['response_text']}")
    return "\n\n".join(parts)


def _decisions(pid: str, prompt: dict, receipt: dict, entries: list[dict]) -> list:
    if receipt["status"] not in {"ok", "invalid"}:
        raise ValueError(f"Unexpected frozen receipt status: {pid}")
    for key in ("prompt_sha256", "runtime_text_sha256", "week_end"):
        if receipt[key] != prompt[key]:
            raise ValueError(f"Prompt/receipt binding mismatch: {pid}:{key}")
    expected = {
        (index, value)
        for index in prompt["current_t_indices"]
        for value in prompt["declared_values"]
    }
    parsed = receipt.get("parsed") or {}
    assessments = parsed.get("assessments", []) if receipt["status"] == "ok" else []
    coordinates = {(row["t_index"], row["dimension"]) for row in assessments}
    if receipt["status"] == "ok" and (
        coordinates != expected or len(assessments) != len(expected)
    ):
        raise ValueError(f"Incomplete frozen v2.0 assessment: {pid}")
    by_coordinate = {(row["t_index"], row["dimension"]): row for row in assessments}
    decisions = []
    for index, value in sorted(expected):
        row = by_coordinate.get((index, value), {})
        decisions.append(
            WeeklyDriftReviewerDecision.model_validate(
                {
                    "persona_id": pid,
                    "week_start": prompt["week_start"],
                    "week_end": prompt["week_end"],
                    "t_index": index,
                    "date": entries[index]["date"],
                    "core_value": value,
                    "verdict": row.get("verdict", "abstain"),
                    "confidence": row.get("confidence"),
                    "reason_code": row.get("reason_code"),
                    "evidence_quote": row.get("evidence_quote", ""),
                    "review_status": receipt["status"],
                }
            )
        )
    return decisions


def consistency_sample(cases: list[dict]) -> list[str]:
    eligible: dict[str, list[dict]] = {}
    for case in cases:
        if case["split"] == "development" and (
            case["weekly_state"] != "insufficient_evidence"
            and any(value["sources"] for value in case["values"])
        ):
            eligible.setdefault(case["persona_id"], []).append(case)
    key = lambda text: hashlib.sha256(text.encode()).hexdigest()  # noqa: E731
    personas = sorted(eligible, key=lambda pid: key(f"20260906:repeat:{pid}"))[:20]
    return [
        min(
            eligible[pid],
            key=lambda case: key(f"20260906:repeat:{pid}:{case['week_start']}"),
        )["case_id"]
        for pid in personas
    ]


def build_cases(record: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Verify preparation sources and derive all 501 frozen repeat-1 cases."""
    started = time.perf_counter()
    partition = record["partition"]
    dev = set(partition["development_persona_ids"])
    final = set(partition["final_persona_ids"])
    ids = dev | final
    if (
        len(dev) != 81
        or len(final) != 24
        or dev & final
        or (
            len(partition["development_persona_ids"]) != 81
            or len(partition["final_persona_ids"]) != 24
        )
    ):
        raise ValueError("Frozen partition must contain disjoint 81/24 histories")
    if (
        hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
        != (partition["combined_105_ids_sha256"])
    ):
        raise ValueError("Frozen cohort ID hash mismatch")
    partition_text = (
        json.dumps(
            {
                key: partition[key]
                for key in ("development_persona_ids", "final_persona_ids")
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    if (
        hashlib.sha256(partition_text.encode()).hexdigest()
        != partition["partition_sha256"]
    ):
        raise ValueError("Frozen partition hash mismatch")
    hashes = _verify_sources(record, root, ids)
    prompts = _read_rows(root / PROMPTS, ids)
    receipts = _read_rows(root / RESPONSES, ids, repeat=1)
    audited = {
        (row["persona_id"], row["week_start"]): row
        for row in record["audits"]["upstream_weekly_inputs"]["weekly_states"]
        if row["run"] == 1
    }
    if (
        len(prompts) != 501
        or set(prompts) != set(receipts)
        or set(prompts) != set(audited)
    ):
        raise ValueError("Frozen upstream records must cover every audited week once")
    registry = {
        row["persona_id"]: row for row in pl.read_parquet(root / REGISTRY).to_dicts()
    }
    cases: list[dict] = []
    history_counts: Counter = Counter()
    empty_weeks = []
    decisions_count: Counter = Counter()
    profile_orders = []
    for pid in sorted(ids):
        stored_values = registry[pid]["core_values"]
        declared = {_normal_value(value) for value in stored_values}
        core_values = [value for value in CORE_VALUE_ORDER if value in declared]
        if len(core_values) != len(stored_values):
            raise ValueError(f"Unknown or duplicate declared Core Value: {pid}")
        entries = _history(root, pid, stored_values)
        profile = {
            "schema_version": "north-star-declared-value-benchmark-profile-v1",
            "persona_id": pid,
            "core_values": core_values,
            "basis": "Declared Core Values; no questionnaire responses constructed",
            "confirmation": (
                "Benchmark projection under the confirmed Profile ordering contract"
            ),
        }
        profile_orders.append(
            {
                "persona_id": pid,
                "stored_core_values": stored_values,
                "canonical_profile_order": core_values,
            }
        )
        writing = _writing(pid, entries)
        all_metadata = {
            source.entry_id: {
                **source.model_dump(exclude={"journal_entry", "nudge_response"}),
                "entry_sequence": 3 * source.t_index,
                "nudge_sequence": 3 * source.t_index + 1 if row["nudge_text"] else None,
                "response_sequence": 3 * source.t_index + 2
                if row["response_text"]
                else None,
                "nudge_text": row["nudge_text"],
                "nudge_present": bool(row["nudge_text"]),
                "response_present": bool(row["response_text"]),
                "journal_entry_sha256": hashlib.sha256(
                    source.journal_entry.encode()
                ).hexdigest(),
                "nudge_response_sha256": hashlib.sha256(
                    source.nudge_response.encode()
                ).hexdigest()
                if source.nudge_response
                else None,
                "availability_basis": "synthetic_immediate_parent_order",
            }
            for source, row in zip(writing, entries, strict=True)
        }
        history_counts.update(
            {
                "personas": 1,
                "entries": len(entries),
                "responses": sum(bool(row["response_text"]) for row in entries),
                "nudges": sum(bool(row["nudge_text"]) for row in entries),
                "entry_core_value_coordinates": len(entries) * len(core_values),
                "personas_with_same_day_entries": len({row["date"] for row in entries})
                < len(entries),
            }
        )
        weeks = sorted({_monday(row["date"]).isoformat() for row in entries})
        if {(pid, week) for week in weeks} != {key for key in prompts if key[0] == pid}:
            raise ValueError(f"Observed weeks differ from frozen prompts: {pid}")
        cursor = date.fromisoformat(weeks[0])
        while cursor <= date.fromisoformat(weeks[-1]):
            if cursor.isoformat() not in weeks:
                empty_weeks.append(
                    {
                        "persona_id": pid,
                        "week_start": cursor.isoformat(),
                        "split": "development" if pid in dev else "final",
                        "status": "unreviewed_empty_week_outside_comparison",
                    }
                )
            cursor += timedelta(days=7)
        accumulated: list[WeeklyDriftReviewerDecision] = []
        for week_start in weeks:
            prompt, receipt = prompts[(pid, week_start)], receipts[(pid, week_start)]
            week_end = (date.fromisoformat(week_start) + timedelta(days=6)).isoformat()
            cutoff = _timestamp(
                (date.fromisoformat(week_start) + timedelta(days=7)).isoformat()
            )
            current = [
                row["t_index"]
                for row in entries
                if week_start <= row["date"] <= week_end
            ]
            historical = {
                str(row["t_index"]): _upstream_text(row)
                for row in entries
                if row["date"] <= week_end
            }
            if (
                prompt["week_end"] != week_end
                or prompt["review_at_date"] != week_end
                or (
                    prompt["current_t_indices"] != current
                    or prompt["entry_text_by_t_index"] != historical
                    or set(prompt["declared_values"]) != declared
                    or prompt["cutoff_t_index"] != max(current)
                )
            ):
                raise ValueError(
                    "Frozen prompt chronology or membership mismatch: "
                    f"{pid}:{week_start}"
                )
            latest = _decisions(pid, prompt, receipt, entries)
            accumulated.extend(latest)
            decisions_count.update(row.verdict for row in latest)
            drift = detect_drift(accumulated, persona_id=pid)
            audit = audited[(pid, week_start)]
            if drift.delivery_state != audit["state"] or (
                drift.core_value_states != audit["core_value_states"]
                or drift.cutoff_t_index != audit["cutoff_t_index"]
                or len(drift.drifts) != audit["drift_count"]
            ):
                raise ValueError(
                    f"Recomputed repeat-1 Drift differs from audit: {pid}:{week_start}"
                )
            request = runtime.build_north_star_request(
                session_id=f"nsm-benchmark:{pid}",
                owner_id=pid,
                profile_ref=runtime.profile_reference(profile),
                core_values=core_values,
                week_start=week_start,
                week_end=week_end,
                cutoff_at=cutoff,
                drift_result=drift,
                writing=writing,
            )
            # Share the application's selection and eligibility authority.
            selected_values, sources, onset, onset_at, reason = runtime._context(
                request
            )
            metadata = {
                source.entry_id: all_metadata[source.entry_id]
                for source in request.writing
            }
            values = [
                {
                    **request.value_definitions[value].model_dump(),
                    "sources": [
                        SourceEntry(
                            entry_id=source.entry_id,
                            journal_entry=source.journal_entry,
                            nudge_response=source.nudge_response,
                        ).model_dump()
                        for source in sources
                    ],
                    "source_metadata": {
                        source.entry_id: metadata[source.entry_id] for source in sources
                    },
                    "source_count": len(sources),
                    "eligible_responses": sum(
                        bool(source.nudge_response) for source in sources
                    ),
                    "source_order": "t_index descending",
                }
                for value in selected_values
            ]
            case = {
                "case_id": f"{pid}:week:{week_start}",
                "persona_id": pid,
                "split": "development" if pid in dev else "final",
                "weekly_state": drift.delivery_state,
                "week_start": week_start,
                "week_end": week_end,
                "cutoff": cutoff,
                "cutoff_at": cutoff,
                "cutoff_t_index": drift.cutoff_t_index,
                "profile": profile,
                "profile_ref": request.profile_ref,
                "core_values": core_values,
                "values": values,
                "source_metadata": metadata,
                "drift_result": drift.model_dump(mode="json"),
                "onset_t_index": onset.onset_t_index if onset else None,
                "onset_date": onset.onset_date if onset else None,
                "onset_available_at": onset_at,
                "context_reason": reason,
                "upstream": {
                    "repeat": 1,
                    "prompt_version": "v2.0",
                    "status": receipt["status"],
                    "prompt_sha256": prompt["prompt_sha256"],
                    "receipt_sha256": stable_hash(receipt),
                    "review_at_date": prompt["review_at_date"],
                    "validation_error": receipt.get("validation_error"),
                    "current_decisions": [
                        row.model_dump(mode="json") for row in latest
                    ],
                    "historical_v2_status_retained": pid == "621be543"
                    and week_start == "2025-09-29",
                },
                "source_availability": {
                    "through_cutoff": len(request.writing),
                    "eligible_entries": len(sources),
                    "eligible_responses": sum(
                        bool(source.nudge_response) for source in sources
                    ),
                    "future_entries_excluded": len(writing) - len(request.writing),
                    "source_window_excluded": len(request.writing) - len(sources)
                    if selected_values
                    else 0,
                    "insufficient_evidence_control": drift.delivery_state
                    == "insufficient_evidence",
                    "available_source_ids": [
                        source.entry_id for source in request.writing
                    ],
                },
                "is_sparse_history_final_week": week_start == weeks[-1]
                and entries[-1]["date"] != week_end,
            }
            case["input_hash"] = stable_hash(case)
            cases.append(case)
    expected_counts = {
        "personas": 105,
        "entries": 881,
        "responses": 400,
        "nudges": 542,
        "entry_core_value_coordinates": 1255,
        "personas_with_same_day_entries": 64,
    }
    if dict(history_counts) != expected_counts or len(empty_weeks) != 3:
        raise ValueError("Source inventory differs from the methodology")
    split_counts = {}
    for split in ("development", "final"):
        selected = [case for case in cases if case["split"] == split]
        counts = dict(Counter(case["weekly_state"] for case in selected))
        expected = record["audits"]["upstream_weekly_inputs"]["partition"][
            "repeat1_counts"
        ][split]
        if counts != expected["states"] or len(selected) != expected["observed_weeks"]:
            raise ValueError(f"Split weekly state counts differ: {split}")
        split_counts[split] = {
            "cases": len(selected),
            "states": counts,
            "personas": len({case["persona_id"] for case in selected}),
        }
    sample = consistency_sample(cases)
    if len(sample) != 20:
        raise ValueError("Evaluator consistency sample requires twenty Personas")
    return {
        "schema_version": "north-star-experiment-cases-v1",
        "cases": cases,
        "consistency_case_ids": sample,
        "validation": {
            "status": "passed",
            "cases": len(cases),
            "history_counts": dict(history_counts),
            "split_counts": split_counts,
            "source_hashes_verified": len(hashes),
            "weekly_decision_counts": dict(decisions_count),
            "insufficient_evidence_controls": sum(
                case["weekly_state"] == "insufficient_evidence" for case in cases
            ),
            "empty_source_cases": sum(
                case["context_reason"] == "no_eligible_writing" for case in cases
            ),
            "invalid_repeat1_reviews": [
                case["case_id"] for case in cases if case["upstream"]["status"] != "ok"
            ],
            "sparse_history_final_weeks": sum(
                case["is_sparse_history_final_week"] for case in cases
            ),
            "parse_warnings": [],
            "missing_histories": [],
        },
        "provenance": {
            "source_hashes": hashes,
            "partition_sha256": partition["partition_sha256"],
            "source_metadata": "Per-case metadata is excluded from all semantic inputs",
            "profile_order": list(CORE_VALUE_ORDER),
            "profile_projections": profile_orders,
            "synthetic_availability_convention": AVAILABILITY_CONVENTION,
            "unreviewed_empty_weeks": empty_weeks,
            "upstream_policy": (
                "Frozen repeat 1 and original v2.0 status; "
                "invalid reviews become Abstain"
            ),
            "cases_sha256": stable_hash(cases),
            "preparation_seconds": time.perf_counter() - started,
        },
    }


def document_text(source: dict[str, Any]) -> str:
    """Serialize one complete candidate; omit only an absent response block."""
    validated = SourceEntry.model_validate(source)
    result = DOCUMENT_TEMPLATE.format(journal_entry=validated.journal_entry)
    if validated.nudge_response:
        result += RESPONSE_TEMPLATE.format(nudge_response=validated.nudge_response)
    return result


def retrieval_config(root: Path = ROOT) -> dict[str, Any]:
    values = yaml.safe_load((root / VALUES).read_text())["values"]
    queries = {
        _normal_value(name): {
            "user_phrase": value["user_phrase"].strip(),
            "definition": value["definition"].strip(),
            "text": f"search_query: {value['user_phrase'].strip()}. "
            f"{value['definition'].strip()}",
        }
        for name, value in values.items()
    }
    config = {
        "model": MODEL,
        "revision": REVISION,
        "code_revision": CODE_REVISION,
        "device": "cpu",
        "local_files_only": True,
        "dimensions": 256,
        "normalization": "layer_norm -> truncate_256 -> L2",
        "batch_size": 8,
        "document_template": DOCUMENT_TEMPLATE,
        "response_template": RESPONSE_TEMPLATE,
        "response_block": "Append exactly when an eligible nonempty response exists",
        "queries": queries,
        "k": 3,
        "tie_break": "cosine descending, t_index descending, entry_id ascending",
        "runtime_order_after_retrieval": "t_index descending",
        "truncation": False,
        "model_max_seq_length": 8192,
        "approved_values_sha256": file_hash(root / VALUES),
    }
    return {**config, "config_sha256": stable_hash(config)}


def rank_case(
    case: dict,
    document_vectors: dict,
    query_vectors: dict,
) -> dict[str, Any]:
    """Rank eligible distinct entry IDs; preserve recency for runtime review."""
    started = time.perf_counter()
    values = {}
    for value in case["values"]:
        entries = [
            {
                "entry_id": source["entry_id"],
                "t_index": value["source_metadata"][source["entry_id"]]["t_index"],
            }
            for source in value["sources"]
        ]
        if len({row["entry_id"] for row in entries}) != len(entries):
            raise ValueError("Duplicate retrieval candidate entry ID")
        similarity = np.array(
            [
                document_vectors[row["entry_id"]] @ query_vectors[value["core_value"]]
                for row in entries
            ]
        )
        if not np.isfinite(similarity).all():
            raise ValueError("Nonfinite Nomic similarity")
        ranking = [
            {**entries[index], "rank": rank + 1, "similarity": float(similarity[index])}
            for rank, index in enumerate(rank_entries(entries, similarity))
        ]
        top_ids = {row["entry_id"] for row in ranking[:3]}
        values[value["core_value"]] = {
            "ranking": ranking,
            "top_entry_ids": [row["entry_id"] for row in ranking[:3]],
            "runtime_entry_ids": [
                source["entry_id"]
                for source in value["sources"]
                if source["entry_id"] in top_ids
            ],
        }
    return {
        "case_id": case["case_id"],
        "values": values,
        "candidate_ids_by_value": {
            value: row["runtime_entry_ids"] for value, row in values.items()
        },
        "latency_seconds": time.perf_counter() - started,
    }


def prepare_retrieval(cases: list[dict], root: Path = ROOT) -> dict[str, Any]:
    """Encode unique eligible candidates once with the pinned cached Nomic model."""
    started = time.perf_counter()
    config = retrieval_config(root)
    documents: dict[str, str] = {}
    for case in cases:
        for value in case["values"]:
            for source in value["sources"]:
                key, text = source["entry_id"], document_text(source)
                if key in documents and documents[key] != text:
                    raise ValueError(
                        f"Eligible source content differs across weeks: {key}"
                    )
                documents[key] = text
    keys = sorted(documents)
    query_keys = sorted(
        {value["core_value"] for case in cases for value in case["values"]}
    )
    all_texts = [config["queries"][key]["text"] for key in query_keys] + [
        documents[key] for key in keys
    ]
    import torch
    from sentence_transformers import SentenceTransformer

    imported = time.perf_counter()
    model = SentenceTransformer(
        MODEL,
        revision=REVISION,
        trust_remote_code=True,
        device="cpu",
        local_files_only=True,
        model_kwargs={"code_revision": CODE_REVISION},
        config_kwargs={"code_revision": CODE_REVISION},
    )
    loaded = time.perf_counter()
    lengths = [
        len(ids) for ids in model.tokenizer(all_texts, truncation=False)["input_ids"]
    ]
    if model.max_seq_length != config["model_max_seq_length"]:
        raise ValueError("Pinned Nomic maximum sequence length differs")
    if max(lengths, default=0) > model.max_seq_length:
        raise ValueError("Nomic would truncate a complete candidate")
    vectors = model.encode(
        all_texts,
        batch_size=8,
        convert_to_tensor=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    vectors = torch.nn.functional.layer_norm(vectors, vectors.shape[1:])[:, :256]
    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1).cpu().numpy()
    encoded = time.perf_counter()
    queries = dict(zip(query_keys, vectors[: len(query_keys)], strict=True))
    document_vectors = dict(zip(keys, vectors[len(query_keys) :], strict=True))
    rankings = {
        case["case_id"]: rank_case(case, document_vectors, queries) for case in cases
    }
    return {
        "schema_version": "north-star-response-inclusive-nomic-v1",
        "config": config,
        "cases": rankings,
        "unique_documents": len(keys),
        "source_document_sha256": {
            key: hashlib.sha256(text.encode()).hexdigest()
            for key, text in documents.items()
        },
        "embedding_vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "query_vectors_sha256": hashlib.sha256(
            vectors[: len(query_keys)].tobytes()
        ).hexdigest(),
        "embedding_dtype": str(vectors.dtype),
        "maximum_input_tokens": max(lengths),
        "preparation_seconds": encoded - started,
        "timing_seconds": {
            "imports": imported - started,
            "model_load": loaded - imported,
            "encode": encoded - loaded,
            "rank": time.perf_counter() - encoded,
        },
        "preparation_allocation": (
            "Shared encoding preparation divided equally across all Nomic weekly "
            "cases for paired total-latency comparisons"
        ),
        "per_case_preparation_seconds": (encoded - started) / len(cases),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "sentence-transformers", "transformers", "numpy")
        },
        "torch_threads": torch.get_num_threads(),
    }
