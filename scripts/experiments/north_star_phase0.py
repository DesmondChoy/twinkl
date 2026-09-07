"""Reproduce NSM's label baseline and run its local, pre-paid retrieval gate.

The baseline uses every known development Drift. Retrieval requires an explicit
cohort manifest, recorded before any rankings are inspected. Labels are joined
only after text encoding; legacy responses have no availability evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.wrangling.parse_wrangled_data import parse_wrangled_file  # noqa: E402

EPISODES = Path(
    "logs/experiments/artifacts/twinkl_qtwz_complete_development_review_20260714/"
    "results/complete_development_drift_episodes.parquet"
)
LABELS = Path("logs/judge_labels/judge_labels.parquet")
CONSENSUS = Path("logs/judge_labels/consensus_labels.parquet")
VALUES = Path("config/schwartz_values.yaml")
MODEL = "nomic-ai/nomic-embed-text-v1.5"
REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
KS = (1, 3, 5)
DOCUMENT_TEMPLATE = "search_document: Journal Entry:\n{initial_entry}"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fraction(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def earlier_entries(
    entries: list[dict[str, Any]], episode: dict[str, Any]
) -> list[dict[str, Any]]:
    """Require both stored order and date before the first Conflict."""
    return [
        row
        for row in entries
        if row["t_index"] < episode["onset_t_index"]
        and row["date"] <= episode["onset_date"]
        and str(row.get("initial_entry") or "").strip()
    ]


def load_inputs(root: Path) -> tuple[list[dict[str, Any]], dict, dict, dict]:
    episodes = pl.read_parquet(root / EPISODES).to_dicts()
    persisted = {
        (row["persona_id"], row["t_index"]): row
        for row in pl.read_parquet(root / LABELS).to_dicts()
    }
    consensus = {
        (row["persona_id"], row["t_index"]): row
        for row in pl.read_parquet(root / CONSENSUS).to_dicts()
    }
    histories = {}
    for persona in sorted({row["persona_id"] for row in episodes}):
        _, entries, warnings = parse_wrangled_file(
            root / f"logs/wrangled/persona_{persona}.md"
        )
        if warnings:
            raise ValueError(f"Unresolved parse warnings for {persona}: {warnings}")
        histories[persona] = entries
    return episodes, persisted, consensus, histories


def baseline(episodes: list[dict], persisted: dict, consensus: dict) -> dict:
    dimensions = [
        key
        for key in next(iter(persisted.values()))
        if key.startswith("alignment_") and key != "alignment_vector"
    ]
    coordinates: set[tuple[str, int, str]] = set()
    counts = {"any_earlier": 0, "persisted_positive": 0, "consensus_positive": 0}
    active = {"episodes": 0, "persisted_positive": 0, "consensus_positive": 0}
    for episode in episodes:
        persona, dim = episode["persona_id"], episode["dimension"]
        earlier = [
            key
            for key, row in persisted.items()
            if key[0] == persona
            and key[1] < episode["onset_t_index"]
            and row["date"] <= episode["onset_date"]
        ]
        coordinates.update((persona, key[1], dim) for key in earlier)
        counts["any_earlier"] += bool(earlier)
        for name, labels in (("persisted", persisted), ("consensus", consensus)):
            positive = any(labels[key][f"alignment_{dim}"] == 1 for key in earlier)
            counts[f"{name}_positive"] += positive
            if episode["delivery_state"] == "active":
                active[f"{name}_positive"] += positive
        active["episodes"] += episode["delivery_state"] == "active"
    return {
        "episodes": len(episodes),
        "availability": counts,
        "active_at_final_cutoff": active,
        "onset_zero": sum(e["onset_t_index"] == 0 for e in episodes),
        "onset_one": sum(e["onset_t_index"] == 1 for e in episodes),
        "label_coordinates": len(persisted) * len(dimensions),
        "all_label_disagreements": sum(
            row[dim] != consensus[key][dim]
            for key, row in persisted.items()
            for dim in dimensions
        ),
        "unique_earlier_coordinates": len(coordinates),
        "earlier_disagreements": sum(
            persisted[(p, t)][f"alignment_{d}"] != consensus[(p, t)][f"alignment_{d}"]
            for p, t, d in coordinates
        ),
        "earlier_positives": {
            name: sum(labels[(p, t)][f"alignment_{d}"] == 1 for p, t, d in coordinates)
            for name, labels in (("persisted", persisted), ("consensus", consensus))
        },
        "consensus_positive_agreement": {
            str(n): sum(
                consensus[(p, t)][f"alignment_{d}"] == 1
                and consensus[(p, t)][f"consensus_agreement_{d}"] == n
                for p, t, d in coordinates
            )
            for n in (3, 4, 5)
        },
    }


def rank_entries(entries: list[dict], similarities: np.ndarray) -> list[int]:
    """Freeze similarity, recency, then stable source identifier as tie-breaks."""
    return sorted(
        range(len(entries)),
        key=lambda i: (
            -float(similarities[i]),
            -entries[i]["t_index"],
            entries[i]["entry_id"],
        ),
    )


def source_availability(episodes: list[dict], histories: dict) -> dict:
    sources = {
        f"{episode['persona_id']}:entry:{entry['t_index']}": entry
        for episode in episodes
        for entry in earlier_entries(histories[episode["persona_id"]], episode)
    }
    return {
        "unique_eligible_original_entries": len(sources),
        "eligible_responses": 0,
        "excluded_responses": {
            key: "response_availability_unknown"
            for key, entry in sources.items()
            if entry.get("response_text")
        },
        "availability_basis": (
            "Stored t_index and date for original writing; wrangled responses have "
            "no independent event order or timestamp and are excluded."
        ),
    }


def retrieval_metrics(cases: list[dict]) -> dict:
    groups = (
        "persisted_positive",
        "consensus_5",
        "consensus_4",
        "consensus_3",
        "persisted_positive_disagreement",
    )
    metrics = {}
    for group in groups:
        eligible = [c for c in cases if any(d[group] for d in c["ranking"])]
        metrics[group] = {
            str(k): fraction(
                sum(any(d[group] for d in c["ranking"][:k]) for c in eligible),
                len(eligible),
            )
            for k in KS
        }
    return metrics


def read_cohort(path: Path, personas: set[str]) -> tuple[dict, set[str]]:
    manifest = json.loads(path.read_text())
    scope = manifest.get("evaluation_scope")
    if scope not in {"small_separate", "large_separate", "development_only"}:
        raise ValueError("Decision 11 must be recorded before retrieval")
    if not manifest.get("decision_source") or not manifest.get("frozen_at"):
        raise ValueError("Record the user decision source and pre-run freeze time")
    frozen_at = datetime.fromisoformat(manifest["frozen_at"])
    if frozen_at.tzinfo is None or frozen_at > datetime.now(UTC):
        raise ValueError(
            "Freeze time must be a past timestamp with an explicit timezone"
        )
    development = set(manifest["development_persona_ids"])
    reserved = set(manifest["reserved_persona_ids"])
    excluded = manifest.get("excluded_personas", {})
    if not development or development & reserved or development - personas:
        raise ValueError("Invalid or overlapping development/final Persona groups")
    if reserved - personas:
        raise ValueError("Reserved Personas must exist in the source corpus")
    if set(excluded) & (development | reserved) or any(
        not x for x in excluded.values()
    ):
        raise ValueError("Excluded Personas require reasons and disjoint membership")
    if development | reserved | set(excluded) != personas:
        raise ValueError("Assign or explicitly exclude every source Persona")
    if scope != "development_only" and not reserved:
        raise ValueError("A separate benchmark requires reserved Persona histories")
    return manifest, development


def retrieve(root: Path, cohort_path: Path, inputs: tuple) -> dict:
    episodes, persisted, consensus, histories = inputs
    cohort, development = read_cohort(cohort_path, set(histories))
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
    config = {
        "model": MODEL,
        "revision": REVISION,
        "code_revision": CODE_REVISION,
        "device": "cpu",
        "dimensions": 256,
        "normalization": "layer_norm -> truncate_256 -> L2",
        "document_template": DOCUMENT_TEMPLATE,
        "queries": queries,
        "batch_size": 8,
        "ks": KS,
        "tie_break": "cosine descending, t_index descending, entry_id ascending",
    }
    # Persist the exact configuration before encoding or inspecting rankings.
    freeze_path = cohort_path.with_name("retrieval_config.json")
    encoded_config = json.dumps(config, indent=2, sort_keys=True) + "\n"
    if freeze_path.exists() and freeze_path.read_text() != encoded_config:
        raise ValueError("Existing frozen retrieval configuration differs")
    freeze_path.write_text(encoded_config)
    selected = [e for e in episodes if e["persona_id"] in development]
    documents = {}
    exclusions = {}
    for episode in selected:
        persona = episode["persona_id"]
        for entry in earlier_entries(histories[persona], episode):
            key = f"{persona}:entry:{entry['t_index']}"
            documents[key] = {**entry, "entry_id": key, "persona_id": persona}
            if entry.get("response_text"):
                exclusions[key] = "response_availability_unknown"
    keys = sorted(documents)
    texts = [
        DOCUMENT_TEMPLATE.format(initial_entry=documents[k]["initial_entry"])
        for k in keys
    ]
    query_keys = sorted({e["dimension"] for e in selected})
    start = time.perf_counter()
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
    all_texts = [queries[k]["text"] for k in query_keys] + texts
    max_tokens = max(len(ids) for ids in model.tokenizer(all_texts)["input_ids"])
    if max_tokens > model.max_seq_length:
        raise ValueError("Encoder would truncate source text; revise feasibility plan")
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
    query_vectors = dict(zip(query_keys, vectors[: len(query_keys)], strict=True))
    document_vectors = dict(zip(keys, vectors[len(query_keys) :], strict=True))
    cases = []
    for episode in selected:
        persona, dim = episode["persona_id"], episode["dimension"]
        entries = [
            documents[f"{persona}:entry:{row['t_index']}"]
            for row in earlier_entries(histories[persona], episode)
        ]
        similarities = np.array(
            [document_vectors[e["entry_id"]] @ query_vectors[dim] for e in entries]
        )
        ranking = []
        for i in rank_entries(entries, similarities):
            entry = entries[i]
            p = persisted[(persona, entry["t_index"])][f"alignment_{dim}"]
            c = consensus[(persona, entry["t_index"])]
            ranking.append(
                {
                    "entry_id": entry["entry_id"],
                    "t_index": entry["t_index"],
                    "date": entry["date"],
                    "similarity": float(similarities[i]),
                    "source_sha256": hashlib.sha256(
                        entry["initial_entry"].encode()
                    ).hexdigest(),
                    "persisted_positive": p == 1,
                    **{
                        f"consensus_{n}": c[f"alignment_{dim}"] == 1
                        and c[f"consensus_agreement_{dim}"] == n
                        for n in (3, 4, 5)
                    },
                    "persisted_positive_disagreement": p == 1
                    and p != c[f"alignment_{dim}"],
                }
            )
        cases.append({"episode": episode, "ranking": ranking})
    metrics = retrieval_metrics(cases)
    passing = [
        k for k in KS if (metrics["persisted_positive"][str(k)]["rate"] or 0) >= 0.9
    ]
    return {
        "cohort": cohort,
        "cohort_sha256": sha256(cohort_path),
        "config": config,
        "config_sha256": sha256(freeze_path),
        "development_episodes": len(selected),
        "unique_documents": len(keys),
        "excluded_responses": exclusions,
        "maximum_input_tokens": max_tokens,
        "metrics": metrics,
        "selected_k": min(passing) if passing else None,
        "gate_passed": bool(passing),
        "cases": cases,
        "timing_seconds": {
            "imports": imported - start,
            "model_load": loaded - imported,
            "encode": encoded - loaded,
            "total": encoded - start,
        },
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2 if sys.platform == "darwin" else 1024),
        "torch_threads": torch.get_num_threads(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "sentence-transformers", "transformers", "numpy")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, help="Run retrieval after decision 11")
    args = parser.parse_args()
    inputs = load_inputs(ROOT)
    episodes, persisted, consensus, histories = inputs
    source_paths = [
        EPISODES,
        LABELS,
        CONSENSUS,
        VALUES,
        Path(__file__).relative_to(ROOT),
    ]
    source_paths.extend(Path(f"logs/wrangled/persona_{p}.md") for p in histories)
    result: dict[str, Any] = {
        "schema_version": "north-star-phase0-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "sources": {str(path): sha256(ROOT / path) for path in source_paths},
        "baseline": baseline(episodes, persisted, consensus),
        "source_availability": source_availability(episodes, histories),
        "paid_calls": 0,
        "limitation": "LLM-Judge VIF Labels are a proxy, not NSM reference decisions.",
    }
    if args.cohort:
        result["retrieval"] = retrieve(ROOT, args.cohort, inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k in {"baseline", "paid_calls"}},
            indent=2,
        )
    )
    if args.cohort:
        retrieval = result["retrieval"]
        print(
            json.dumps(
                {
                    k: retrieval[k]
                    for k in (
                        "selected_k",
                        "gate_passed",
                        "metrics",
                        "timing_seconds",
                        "peak_rss_mib",
                    )
                },
                indent=2,
            )
        )
        return 0 if retrieval["gate_passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
