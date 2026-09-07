"""Reproduce development-only NSM diagnostics without provider calls.

The historical files are read-only. An explicit, separate output directory is
required. No source corpus or reserved Persona writing is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.north_star.review import (  # noqa: E402
    ReviewValidationError,
    SourceEntry,
    select_moment,
    validate_review,
)

PHASE0A = Path("logs/experiments/reports/north_star_phase0_20260905")
HISTORICAL = Path("logs/experiments/reports/north_star_phase0b_20260905")
SCRIPT = Path("scripts/experiments/north_star_phase0b_diagnostics.py")
JsonObject = dict[str, Any]


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_object(value: Any) -> str:
    return digest_bytes(json.dumps(value, sort_keys=True).encode())


def read_object(path: Path) -> JsonObject:
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result


def index_cases(cases: list[JsonObject], key: str = "case_id") -> JsonObject:
    result = {case[key]: case for case in cases}
    if len(result) != len(cases):
        raise ValueError("Duplicate case identifiers")
    return result


def load_verified_inputs(root: Path = ROOT) -> tuple[JsonObject, ...]:
    """Check the freeze chain before parsing the development writing."""
    freeze_path = HISTORICAL / "execution_freeze.json"
    validation_path = HISTORICAL / "validation.json"
    freeze = read_object(root / freeze_path)
    provenance = {}
    for relative, expected in freeze["source_hashes"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Frozen input path escapes repository")
        actual = digest_bytes(path.read_bytes())
        if actual != expected:
            raise ValueError(f"Execution-freeze hash mismatch: {relative}")
        provenance[relative] = actual

    manifest_path = HISTORICAL / "manifest.json"
    report_path = HISTORICAL / "report.json"
    retrieval_path = PHASE0A / "retrieval.json"
    manifest = read_object(root / manifest_path)
    for relative, expected in manifest["source_hashes"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Manifest input path escapes repository")
        actual = digest_bytes(path.read_bytes())
        if actual != expected:
            raise ValueError(f"Manifest input hash mismatch: {relative}")
        provenance[relative] = actual

    validation = read_object(root / validation_path)
    report_hash = digest_bytes((root / report_path).read_bytes())
    if report_hash != validation["hashes"][str(report_path)]:
        raise ValueError("Historical report hash mismatch")
    report = read_object(root / report_path)
    if report["manifest_sha256"] != provenance[str(manifest_path)]:
        raise ValueError("Report does not identify the frozen manifest")
    for field, relative in [
        ("provider_sha256", "src/north_star/provider.py"),
        ("runner_sha256", "scripts/experiments/north_star_phase0b.py"),
    ]:
        if report[field] != provenance[relative]:
            raise ValueError(f"Report execution hash mismatch: {field}")
    for relative in [freeze_path, validation_path, report_path, SCRIPT]:
        provenance[str(relative)] = digest_bytes((root / relative).read_bytes())
    return manifest, report, read_object(root / retrieval_path), provenance


def validate_cases(
    manifest: JsonObject, report: JsonObject, phase0a: JsonObject
) -> None:
    """Require matching development episodes, ranks, source hashes and decisions."""
    retrieval = phase0a["retrieval"]
    if retrieval["selected_k"] != 3:
        raise ValueError("Historical runtime retrieval must be frozen at k=3")
    manifests = index_cases(manifest["cases"])
    reports = index_cases(report["cases"])
    ranked = index_cases(
        [
            dict(case, case_id=case["episode"]["episode_id"])
            for case in retrieval["cases"]
        ]
    )
    if set(manifests) != set(reports) or set(manifests) != set(ranked):
        raise ValueError("Manifest/report/development ranking cases differ")
    if manifest["case_count"] != len(manifests):
        raise ValueError("Manifest case count differs")
    development = set(retrieval["cohort"]["development_persona_ids"])
    reserved = set(retrieval["cohort"]["reserved_persona_ids"])
    for case_id, case in manifests.items():
        episode = case["episode"]
        persona = episode["persona_id"]
        if persona in reserved or persona not in development:
            raise ValueError("Non-development Persona in diagnostics")
        if episode != ranked[case_id]["episode"]:
            raise ValueError(f"Episode mismatch: {case_id}")
        if case_id != episode["episode_id"]:
            raise ValueError(f"Case identifier does not match episode: {case_id}")
        if case["core_value"] != episode["dimension"]:
            raise ValueError(f"Core Value mismatch: {case_id}")
        if case["value"] != retrieval["config"]["queries"][case["core_value"]]:
            raise ValueError(f"Requested value definition differs: {case_id}")
        sources = case["all_eligible_sources_in_retrieval_order"]
        ranking = ranked[case_id]["ranking"]
        ids = [source["entry_id"] for source in sources]
        if len(set(ids)) != len(ids) or ids != [r["entry_id"] for r in ranking]:
            raise ValueError(f"Source membership/order mismatch: {case_id}")
        if case["runtime_entry_ids"] != ids[:3]:
            raise ValueError(f"Runtime retrieval changed: {case_id}")
        for source, rank in zip(sources, ranking, strict=True):
            if source["entry_id"] != f"{persona}:entry:{rank['t_index']}":
                raise ValueError(f"Wrong-persona source: {case_id}")
            if source.get("nudge_response") is not None:
                raise ValueError("Historical protocol excludes nudge responses")
            if digest_bytes(source["journal_entry"].encode()) != rank["source_sha256"]:
                raise ValueError(f"Historical source text changed: {case_id}")
        result = reports[case_id]
        if result["core_value"] != case["core_value"]:
            raise ValueError(f"Report Core Value mismatch: {case_id}")
        if result["eligible_sources"] != len(sources):
            raise ValueError(f"Report source count mismatch: {case_id}")
        typed_sources = [SourceEntry.model_validate(source) for source in sources]
        if not sources:
            if result.get("selected") or result.get("attempts"):
                raise ValueError(f"Structurally empty case has output: {case_id}")
            continue
        if result.get("reference") is not None:
            reference = validate_review(
                result["reference"],
                core_value=case["core_value"],
                sources=typed_sources,
            )
            valid_ids = [
                r.entry_id for r in reference.results if r.decision == "supportive"
            ]
            if (
                len(result["reference_valid_ids"]) != len(valid_ids)
                or set(result["reference_valid_ids"]) != set(valid_ids)
            ):
                raise ValueError(f"Saved reference IDs differ: {case_id}")
            if result["reference_no_example"] != (not valid_ids):
                raise ValueError(f"Saved reference absence flag differs: {case_id}")
        if result.get("runtime") is not None:
            selected = select_moment(
                result["runtime"],
                core_value=case["core_value"],
                sources=typed_sources[:3],
            )
            if result.get("selected") != (selected.model_dump() if selected else None):
                raise ValueError(f"Saved selection differs: {case_id}")
        elif result.get("selected"):
            raise ValueError(f"Selection lacks a valid runtime batch: {case_id}")


def fraction(numerator: int, denominator: int) -> JsonObject:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def derive_diagnostics(manifest: JsonObject, report: JsonObject) -> JsonObject:
    """Compute statistics only; no semantic relabeling or model calls occur."""
    cases = index_cases(manifest["cases"])
    reports = index_cases(report["cases"])
    nonempty = [
        c for c in cases.values() if c["all_eligible_sources_in_retrieval_order"]
    ]
    references = {
        c["case_id"]: c["reference"]["results"]
        for c in reports.values()
        if c.get("reference") is not None
    }
    positive_ids = {
        cid: {r["entry_id"] for r in rows if r["decision"] == "supportive"}
        for cid, rows in references.items()
    }
    positive_histories = sum(bool(ids) for ids in positive_ids.values())
    workloads = []
    misses = []
    for k in [1, 3, 5, 7]:
        supplied = [
            (c["case_id"], source)
            for c in nonempty
            for source in c["all_eligible_sources_in_retrieval_order"][:k]
        ]
        hits = sum(
            any(
                s["entry_id"] in positive_ids.get(c["case_id"], set())
                for s in c["all_eligible_sources_in_retrieval_order"][:k]
            )
            for c in nonempty
        )
        workloads.append(
            {
                "k": k,
                "source_decisions": len(supplied),
                "source_characters": sum(
                    len(s["journal_entry"]) + len(s.get("nudge_response") or "")
                    for _, s in supplied
                ),
                "reference_supportive_sources_in_pool": sum(
                    s["entry_id"] in positive_ids.get(cid, set()) for cid, s in supplied
                ),
                "task_reference_retrieval_recall": fraction(hits, positive_histories),
            }
        )
    for cid, ids in positive_ids.items():
        if ids and not ids.intersection(cases[cid]["runtime_entry_ids"]):
            misses.append(
                {
                    "case_id": cid,
                    "reference_supportive_source_ranks": [
                        {"entry_id": s["entry_id"], "rank": i + 1}
                        for i, s in enumerate(
                            cases[cid]["all_eligible_sources_in_retrieval_order"]
                        )
                        if s["entry_id"] in ids
                    ],
                }
            )
    disagreements = []
    no_support = []
    groups: dict[str, list[JsonObject]] = defaultdict(list)
    identities = {}
    invalid_attempts = []
    for cid, case in cases.items():
        source_map = {
            s["entry_id"]: s for s in case["all_eligible_sources_in_retrieval_order"]
        }
        result = reports[cid]
        rows = references.get(cid)
        if source_map and rows is not None and not positive_ids[cid]:
            counts = Counter(r["decision"] for r in rows)
            stratum = (
                "all_abstain"
                if counts["abstain"] == len(rows)
                else "includes_abstain"
                if counts["abstain"]
                else "all_not_supportive"
            )
            no_support.append(
                {
                    "case_id": cid,
                    "stratum": stratum,
                    "decision_counts": dict(sorted(counts.items())),
                    "omitted": not bool(result.get("selected")),
                }
            )
        selected = result.get("selected")
        if selected and result.get("incorrect_displayed"):
            ref = next(
                (r for r in rows or [] if r["entry_id"] == selected["entry_id"]), None
            )
            disagreements.append(
                {
                    "case_id": cid,
                    "core_value": case["core_value"],
                    "selected": selected,
                    "primary_reference": ref,
                    "basis": "primary_reference_nonacceptance"
                    if ref and ref["decision"] != "supportive"
                    else "candidate_reference_nonacceptance_or_unresolved",
                }
            )
        for row in rows or []:
            source = source_map[row["entry_id"]]
            identity = {
                "source": source,
                "core_value": case["core_value"],
                "user_phrase": case["value"]["user_phrase"],
                "approved_definition": case["value"]["definition"],
            }
            identity_hash = digest_object(identity)
            identities[identity_hash] = identity
            groups[identity_hash].append(
                {
                    "case_id": cid,
                    "entry_id": row["entry_id"],
                    "core_value": case["core_value"],
                    "source_sha256": digest_object(source),
                    "requested_value_sha256": digest_object(case["value"]),
                    "decision": row["decision"],
                    "reason_code": row["reason_code"],
                }
            )
        for receipt in result.get("attempts", []):
            attempt = receipt["attempt"]
            if attempt["provider"] != "openai" or attempt["status"] != "invalid":
                continue
            try:
                validate_review(
                    attempt["raw_text"],
                    core_value=case["core_value"],
                    sources=[
                        SourceEntry.model_validate(s)
                        for s in list(source_map.values())[:3]
                    ],
                )
            except ReviewValidationError as exc:
                invalid_attempts.append(
                    {
                        "case_id": cid,
                        "attempt_number": attempt["attempt_number"],
                        "saved_error_type": attempt["error_type"],
                        "replayed_validation_error": str(exc),
                        "raw_results": json.loads(attempt["raw_text"])["results"],
                    }
                )
    conflicts = [
        {
            "source_and_requested_value_sha256": key,
            "identity": identities[key],
            "observations": observations,
            "changes_supportive_status": len(
                {r["decision"] == "supportive" for r in observations}
            )
            > 1,
        }
        for key, observations in sorted(groups.items())
        if len({(r["decision"], r["reason_code"]) for r in observations}) > 1
    ]
    return {
        "counts": {
            "development_episodes": len(cases),
            "nonempty_histories": len(nonempty),
            "structurally_empty_histories": len(cases) - len(nonempty),
            "histories_with_reference": len(references),
            "nonempty_histories_without_reference": sum(
                c["case_id"] not in references for c in nonempty
            ),
            "reference_positive_histories": positive_histories,
            "selected_quotations": sum(
                bool(r.get("selected")) for r in reports.values()
            ),
        },
        "selection_disagreements": disagreements,
        "disagreement_primary_reason_counts": dict(
            sorted(
                Counter(
                    d["primary_reference"]["reason_code"]
                    if d["primary_reference"]
                    else "unresolved"
                    for d in disagreements
                ).items()
            )
        ),
        "retrieval_workloads": workloads,
        "retrieval_misses_at_3": misses,
        "no_reference_supportive_histories": no_support,
        "no_reference_supportive_strata": {
            stratum: {
                "histories": len(members),
                "correct_omission": fraction(
                    sum(bool(m["omitted"]) for m in members), len(members)
                ),
            }
            for stratum in ["all_not_supportive", "includes_abstain", "all_abstain"]
            if (members := [r for r in no_support if r["stratum"] == stratum])
        },
        "repeated_identical_source_reference_conflicts": conflicts,
        "invalid_runtime_attempts": invalid_attempts,
    }


INTERPRETATIONS = {
    "152df7a4:universalism:episode_01": (
        "Explicit paperwork assistance; the Universalism relationship is unclear "
        "because the writing emphasizes operational efficiency."
    ),
    "2541429a:tradition:episode_01": (
        "The quotation reflects on a daughter's learning; a completed supportive "
        "action by the writer is inferred rather than clearly described."
    ),
    "5fa8b540:universalism:episode_01": (
        "The quotation describes an outcome. The full source explicitly mentions "
        "a water-pollution lesson the writer delivered, so source-level abstention "
        "remains debatable even though the selected quotation is weak."
    ),
    "66ced716:universalism:episode_01": (
        "Praising a painting and helping place it on a rack establishes action; "
        "the broader welfare relationship depends on missing workshop context."
    ),
    "7ff1d0fb:security:episode_01": (
        "A claim that rent is covered supplies no action that secured it."
    ),
    "bf44e50f:hedonism:episode_01": (
        "The full source explicitly interrupts the enjoyment in the selected "
        "passage; a whole-source Conflict check is needed."
    ),
    "dbe2c53d:conformity:episode_01": (
        "Deleting confrontational replies and closing a family chat demonstrates "
        "restraint. Regret about restraint does not necessarily establish Conflict "
        "against Conformity; the frozen reference remains disputed."
    ),
}

LIMITATIONS = [
    "All semantic judgments are AI-derived development references, not human "
    "validation; the frozen gate and labels are unchanged.",
    "No accepted reference example is not proof of absence. Report all-rejected, "
    "mixed-abstention, all-abstention, structurally empty and unresolved cases "
    "separately.",
    "Repeated episodes may share identical writing. Episode counts are not "
    "independent participants; repeated-source inconsistencies expose "
    "reference limits.",
    "Source-character workloads exclude prompts and provider tokenization; they "
    "are not token counts, cost estimates, or observed results at larger k.",
    "Historical retrieval-only precision grades a source identifier whereas "
    "reviewed precision grades a quotation. Their subtraction is not matched "
    "quotation-level verification lift or a causal estimate.",
    "No reserved writing, paid review, embeddings, application integration or "
    "browser behavior is evaluated by these diagnostics.",
]


def render_markdown(payload: JsonObject) -> str:
    derived = payload["mechanically_derived"]
    lines = [
        "# North Star Moment: offline development diagnostics",
        "",
        "These diagnostics preserve the failed Phase 0B gate and all frozen AI "
        "reference decisions. They make no provider calls.",
        "",
        "## Reproduction and provenance",
        "",
        "```sh",
        "source .venv/bin/activate",
        payload["command"],
        "```",
        "",
        "Execution-freeze and manifest input hashes, the historical report hash "
        "in validation.json, development episode membership, source text hashes, "
        "retrieval order, review contracts, and saved selections are checked "
        "before analysis. diagnostics.json records every input hash.",
        "",
        "## Mechanically derived evidence",
        "",
        f"The {derived['counts']['development_episodes']} development episodes "
        f"include {derived['counts']['nonempty_histories']} nonempty histories "
        f"and {derived['counts']['structurally_empty_histories']} structurally "
        "empty histories. Seven selections disagree with the primary reference: "
        "three wrong_value, two ambiguous, and two same_value_conflict.",
        "",
        "| k | Reference-positive histories reached | Source decisions | "
        "Source characters |",
        "|---|---:|---:|---:|",
    ]
    for row in derived["retrieval_workloads"]:
        recall = row["task_reference_retrieval_recall"]
        lines.append(
            f"| {row['k']} | {recall['numerator']}/{recall['denominator']} "
            f"| {row['source_decisions']} | {row['source_characters']} |"
        )
    lines += [
        "",
        "At k=3, Noor's second Tradition episode misses entry 7 "
        "at rank 7. Lukas's second Self-Direction episode misses entries "
        "4 and 5 at ranks 5 and 7. Top-5 would add 30 source decisions "
        "(40.5%) while reaching only one additional reference-positive "
        "history. These are reference/ranking calculations, not rerun "
        "selection outcomes.",
        "",
        "| No-reference-supportive stratum | Histories | Correct omission |",
        "|---|---:|---:|",
    ]
    for label, row in derived["no_reference_supportive_strata"].items():
        rate = row["correct_omission"]
        lines.append(
            f"| {label} | {row['histories']} | "
            f"{rate['numerator']}/{rate['denominator']} |"
        )
    conflicts = derived["repeated_identical_source_reference_conflicts"]
    lines += [
        "",
        f"{len(conflicts)} identical source/requested-value groups "
        "change reference decision or reason across episodes. One changes "
        "supportive status: `87e92805:entry:4` is supportive in Security "
        "episode 02 and same_value_conflict in episode 03. Its text and "
        "requested phrase/definition are identical; it describes authorizing "
        "necessary motorcycle brake repairs despite financial anxiety.",
        "",
        "Both invalid OpenAI attempts belong to "
        "`dbe2c53d:universalism:episode_01`. Replaying their saved JSON "
        "reproduces malformed_decision:results.0:value_error; both combine "
        "abstain with other_actor. diagnostics.json retains the contradictory "
        "fields. Invalid historical attempts remain invalid.",
        "",
        "## Separate analyst interpretations",
        "",
        "These explanations are a new AI analysis of development evidence; "
        "they do not replace the frozen reference labels.",
        "",
        "| Case | Interpretation |",
        "|---|---|",
    ]
    lines.extend(
        f"| `{cid}` | {explanation} |"
        for cid, explanation in payload["analyst_interpretations"].items()
    )
    lines += [
        "",
        "The first revision should retain Nomic top-3 and the model "
        "settings, require concise action/value/context assessment, and "
        "derive decision from one reason enum. An already-made commitment "
        "can be a completed choice even if the event is future; mere "
        "intentions remain insufficient. Compare any later retrieval "
        "revision at matched k and candidate workload. Grade reviewed "
        "and retrieval-only quotations with the same exact-candidate "
        "protocol, reporting precision and coverage separately.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {line}" for line in payload["limitations"])
    return "\n".join(lines) + "\n"


def write_diagnostics(output_dir: Path, root: Path = ROOT) -> JsonObject:
    output_dir = output_dir.resolve()
    targets = [output_dir / name for name in ["diagnostics.json", "diagnostics.md"]]
    protected = [(root / directory).resolve() for directory in [PHASE0A, HISTORICAL]]
    for target in targets:
        resolved = target.resolve()
        if any(resolved.is_relative_to(directory) for directory in protected):
            raise ValueError("Diagnostics must not write into historical evidence")
        if resolved.parent != output_dir:
            raise ValueError("Output file must not redirect outside explicit target")
    manifest, report, retrieval, hashes = load_verified_inputs(root)
    validate_cases(manifest, report, retrieval)
    try:
        output_arg = str(output_dir.relative_to(root.resolve()))
    except ValueError:
        output_arg = str(output_dir)
    payload = {
        "schema_version": "north-star-phase0b-offline-diagnostics-v1",
        "command": shlex.join(
            ["uv", "run", "python", str(SCRIPT), "--output-dir", output_arg]
        ),
        "source_hashes": dict(sorted(hashes.items())),
        "mechanically_derived": derive_diagnostics(manifest, report),
        "analyst_interpretations": INTERPRETATIONS,
        "limitations": LIMITATIONS,
        "paid_calls": 0,
        "reserved_writing_loaded": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    targets[0].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    targets[1].write_text(render_markdown(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    result = write_diagnostics(directory)
    print(json.dumps(result["mechanically_derived"]["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
