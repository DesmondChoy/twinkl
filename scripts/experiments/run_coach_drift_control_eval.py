"""Select Drift and control targets and optionally generate Coach Digest responses.

The default command is a dry run. It writes a deterministic target catalog and
makes no provider calls. ``--execute`` runs Weekly Drift Detection and Coach
Digest generation for targets that are not already in the manifest.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.coach.llm_client import (  # noqa: E402
    build_llm_complete,
    resolve_coach_model,
)
from src.coach.schemas import LLMCallMetrics, WeeklyDigest  # noqa: E402
from src.coach.weekly_drift_runtime import run_weekly_drift_coach_cycle  # noqa: E402
from src.wrangling.parse_wrangled_data import parse_wrangled_file  # noqa: E402

DEFAULT_RESULTS_DIR = Path(
    "logs/experiments/artifacts/"
    "twinkl_qtwz_complete_development_review_20260714/results"
)
DEFAULT_EPISODES = DEFAULT_RESULTS_DIR / "complete_development_drift_episodes.parquet"
DEFAULT_CASE_OUTCOMES = (
    DEFAULT_RESULTS_DIR / "complete_development_case_outcomes.parquet"
)
DEFAULT_REPORT_DIR = Path(
    "logs/experiments/reports/coach_digest_drift_control"
)
DEFAULT_PARQUET = Path(
    "logs/exports/weekly_digests/coach_digest_drift_control.parquet"
)
DEFAULT_OUTPUT_DIR = Path("logs/exports/weekly_drift_coach/drift_control")
DEFAULT_SEED = 20260823

TargetGroup = Literal["drift", "control"]


@dataclass(frozen=True)
class EvalTarget:
    """One Persona cutoff for the Drift/control Coach Digest study."""

    target_id: str
    persona_id: str
    end_date: str
    group: TargetGroup
    historical_split: str
    entry_count: int
    reviewed_week_count: int
    dimension: str | None = None
    delivery_state: str | None = None
    episode_id: str | None = None
    matched_to: str | None = None
    match_quality: str = "exact"


def _entry_count_bucket(entry_count: int) -> str:
    if entry_count <= 6:
        return "at_most_6"
    if entry_count <= 9:
        return "7_to_9"
    return "10_to_12"


def _week_end(raw: str) -> date:
    value = date.fromisoformat(raw)
    return value + timedelta(days=6 - value.weekday())


def _persona_week_ends(wrangled_dir: Path, persona_id: str) -> list[str]:
    path = wrangled_dir / f"persona_{persona_id}.md"
    _profile, entries, _warnings = parse_wrangled_file(path)
    week_ends = {_week_end(str(entry["date"])) for entry in entries}
    return [value.isoformat() for value in sorted(week_ends)]


def _reviewed_week_count(week_ends: list[str], end_date: str) -> int:
    cutoff = _week_end(end_date).isoformat()
    return sum(1 for week_end in week_ends if week_end <= cutoff)


def load_drift_targets(
    episodes_path: Path,
    case_outcomes_path: Path,
    wrangled_dir: Path,
) -> list[EvalTarget]:
    """Build one target for each known development-set Drift."""
    episodes = pl.read_parquet(episodes_path)
    cases = pl.read_parquet(case_outcomes_path)
    case_index = {
        (row["persona_id"], row["dimension"]): row for row in cases.to_dicts()
    }
    week_ends_by_persona: dict[str, list[str]] = {}
    targets: list[EvalTarget] = []
    for row in episodes.to_dicts():
        persona_id = str(row["persona_id"])
        dimension = str(row["dimension"])
        case = case_index[(persona_id, dimension)]
        if persona_id not in week_ends_by_persona:
            week_ends_by_persona[persona_id] = _persona_week_ends(
                wrangled_dir,
                persona_id,
            )
        week_ends = week_ends_by_persona[persona_id]
        end_date = str(row["confirmation_date"])
        episode_id = str(row["episode_id"])
        targets.append(
            EvalTarget(
                target_id=f"drift:{episode_id}",
                persona_id=persona_id,
                end_date=end_date,
                group="drift",
                historical_split=str(case["historical_split"]),
                entry_count=int(case["entry_count"]),
                reviewed_week_count=_reviewed_week_count(week_ends, end_date),
                dimension=dimension,
                delivery_state=str(row["delivery_state"]),
                episode_id=episode_id,
            )
        )
    return sorted(targets, key=lambda target: target.target_id)


def _candidate_ladder(
    *,
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]],
    by_split: dict[str, list[dict[str, Any]]],
    pool: list[dict[str, Any]],
    split: str,
    bucket: str,
    used_personas: set[str],
) -> tuple[list[dict[str, Any]], str]:
    def available(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [row for row in rows if row["persona_id"] not in used_personas]

    exact = available(by_stratum.get((split, bucket), []))
    if exact:
        return exact, "exact"
    same_bucket = available(
        [
            row
            for (_split, row_bucket), rows in by_stratum.items()
            if row_bucket == bucket
            for row in rows
        ]
    )
    if same_bucket:
        return same_bucket, "split_relaxed"
    same_split = available(by_split.get(split, []))
    if same_split:
        return same_split, "entry_count_relaxed"
    return available(pool), "unmatched"


def sample_control_targets(
    case_outcomes_path: Path,
    episodes_path: Path,
    drift_targets: list[EvalTarget],
    wrangled_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
) -> list[EvalTarget]:
    """Match one no-Drift Persona cutoff to each Drift target."""
    episodes = pl.read_parquet(episodes_path)
    drift_personas = set(episodes["persona_id"].to_list())
    cases = pl.read_parquet(case_outcomes_path)
    pool = [
        row
        for row in cases.to_dicts()
        if row["persona_id"] not in drift_personas and row["has_drift"] is False
    ]

    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        bucket = _entry_count_bucket(int(row["entry_count"]))
        by_stratum[(str(row["historical_split"]), bucket)].append(row)
        by_split[str(row["historical_split"])].append(row)

    rng = random.Random(seed)
    for rows in by_stratum.values():
        rng.shuffle(rows)

    used_personas: set[str] = set()
    week_ends_by_persona: dict[str, list[str]] = {}
    controls: list[EvalTarget] = []
    for drift_target in drift_targets:
        candidates, match_quality = _candidate_ladder(
            by_stratum=by_stratum,
            by_split=by_split,
            pool=pool,
            split=drift_target.historical_split,
            bucket=_entry_count_bucket(drift_target.entry_count),
            used_personas=used_personas,
        )
        if not candidates:
            continue
        same_dimension = [
            row for row in candidates if row["dimension"] == drift_target.dimension
        ]
        chosen = rng.choice(same_dimension or candidates)
        persona_id = str(chosen["persona_id"])
        if persona_id not in week_ends_by_persona:
            week_ends_by_persona[persona_id] = _persona_week_ends(
                wrangled_dir,
                persona_id,
            )
        week_ends = week_ends_by_persona[persona_id]
        if not week_ends:
            continue
        used_personas.add(persona_id)
        end_date = min(
            week_ends,
            key=lambda value: (
                abs(
                    _reviewed_week_count(week_ends, value)
                    - drift_target.reviewed_week_count
                ),
                value,
            ),
        )
        controls.append(
            EvalTarget(
                target_id=f"control:{drift_target.episode_id}:{persona_id}",
                persona_id=persona_id,
                end_date=end_date,
                group="control",
                historical_split=str(chosen["historical_split"]),
                entry_count=int(chosen["entry_count"]),
                reviewed_week_count=_reviewed_week_count(week_ends, end_date),
                dimension=str(chosen["dimension"]),
                matched_to=drift_target.episode_id,
                match_quality=match_quality,
            )
        )
    return controls


def build_targets(
    *,
    episodes_path: Path,
    case_outcomes_path: Path,
    wrangled_dir: Path,
    group: str,
    seed: int,
    limit: int | None,
) -> list[EvalTarget]:
    drift_targets = load_drift_targets(
        episodes_path,
        case_outcomes_path,
        wrangled_dir,
    )
    if limit is not None:
        drift_targets = drift_targets[:limit]
    controls = (
        sample_control_targets(
            case_outcomes_path,
            episodes_path,
            drift_targets,
            wrangled_dir,
            seed=seed,
        )
        if group in {"control", "both"}
        else []
    )
    if group == "control":
        drift_targets = []
    return drift_targets + controls


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_sha256(path: Path) -> str:
    """Hash each wrangled Markdown file name and content in order."""
    digest = hashlib.sha256()
    for file_path in sorted(path.glob("persona_*.md")):
        digest.update(file_path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _target_catalog(
    targets: list[EvalTarget],
    *,
    seed: int,
    episodes_path: Path,
    case_outcomes_path: Path,
    wrangled_dir: Path,
) -> dict[str, Any]:
    target_ids = [target.target_id for target in targets]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Target selection produced duplicate target IDs.")
    return {
        "schema_version": "coach-digest-drift-control-targets-v1",
        "seed": seed,
        "sources": {
            "episodes": {"path": str(episodes_path), "sha256": _sha256(episodes_path)},
            "case_outcomes": {
                "path": str(case_outcomes_path),
                "sha256": _sha256(case_outcomes_path),
            },
            "wrangled": {
                "path": str(wrangled_dir),
                "sha256": _directory_sha256(wrangled_dir),
            },
        },
        "targets": [asdict(target) for target in targets],
    }


def merge_target_catalog(
    path: Path,
    catalog: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    """Keep earlier targets in a catalog that the same inputs can reproduce."""
    if not resume or not path.exists():
        return catalog
    existing = json.loads(path.read_text())
    if (
        existing.get("schema_version") != catalog["schema_version"]
        or existing.get("seed") != catalog["seed"]
        or existing.get("sources") != catalog["sources"]
    ):
        raise ValueError(
            "Existing target catalog uses a different schema, inputs, or seed."
        )
    merged: dict[str, dict[str, Any]] = {}
    for item in existing.get("targets", []):
        target_id = str(item["target_id"])
        if target_id in merged:
            raise ValueError(f"Duplicate target catalog ID: {target_id}")
        merged[target_id] = item
    merged.update({item["target_id"]: item for item in catalog["targets"]})
    return {**catalog, "targets": [merged[key] for key in sorted(merged)]}


def _manifest_index(path: Path, *, resume: bool) -> dict[str, dict[str, Any]]:
    if not resume or not path.exists():
        return {}
    items = json.loads(path.read_text())
    manifest: dict[str, dict[str, Any]] = {}
    for item in items:
        target_id = item.get("target", {}).get("target_id")
        if not target_id:
            raise ValueError("Manifest record has no target ID.")
        target_id = str(target_id)
        if target_id in manifest:
            raise ValueError(f"Duplicate manifest target ID: {target_id}")
        manifest[target_id] = item
    return manifest


def _write_manifest(path: Path, manifest: dict[str, dict[str, Any]]) -> None:
    """Write completed target records after each paid generation step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps([manifest[key] for key in sorted(manifest)], indent=2) + "\n"
    )
    temporary_path.replace(path)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _target_output_dir(output_dir: Path, target_id: str) -> Path:
    suffix = hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:16]
    return output_dir / f"target_{suffix}"


def _manifest_entry(
    digest: WeeklyDigest,
    target: EvalTarget,
    generator_model: str,
    call_metrics: list[LLMCallMetrics],
    output_paths: dict[str, str],
) -> dict[str, Any] | None:
    if digest.coach_narrative is None or digest.validation is None:
        return None
    digest_payload = digest.model_dump(
        mode="json",
        exclude={"coach_narrative", "validation"},
    )
    narrative_payload = digest.coach_narrative.model_dump(mode="json")
    return {
        "digest": digest_payload,
        "narrative": narrative_payload,
        "validation": {
            **digest.validation.model_dump(mode="json"),
            "all_passed": digest.validation.all_passed,
        },
        "generator_model": generator_model,
        "target": asdict(target),
        "provenance": {
            "weekly_drift_input_sha256": _canonical_sha256(digest_payload),
            "coach_response_sha256": _canonical_sha256(narrative_payload),
            "call_metrics": [metric.model_dump(mode="json") for metric in call_metrics],
            "output_paths": output_paths,
        },
    }


async def generate_missing_targets(
    targets: list[EvalTarget],
    existing: dict[str, dict[str, Any]],
    *,
    parquet_path: Path,
    output_dir: Path,
    wrangled_dir: Path,
    manifest_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    pending = [target for target in targets if target.target_id not in existing]
    if not pending:
        return existing

    provider, model = resolve_coach_model()
    generator_model = f"{provider}:{model}"
    previous_models = {
        str(entry["generator_model"])
        for entry in existing.values()
        if entry.get("generator_model")
    }
    if previous_models and previous_models != {generator_model}:
        raise ValueError("Resume uses a different Coach Digest generator model.")

    call_metrics: list[LLMCallMetrics] = []
    coach_llm_complete = build_llm_complete(call_metrics=call_metrics)
    if coach_llm_complete is None:
        raise SystemExit("No Coach Digest provider is available. Check its API key.")

    for index, target in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {target.target_id}", flush=True)
        metrics_start = len(call_metrics)
        digest, output_paths = await run_weekly_drift_coach_cycle(
            persona_id=target.persona_id,
            wrangled_dir=wrangled_dir,
            output_dir=_target_output_dir(output_dir, target.target_id),
            parquet_path=parquet_path,
            end_date=target.end_date,
            coach_llm_complete=coach_llm_complete,
            attach_failed_validation=True,
        )
        target_metrics = call_metrics[metrics_start:]
        for attempt, metric in enumerate(target_metrics, start=1):
            metric.call_label = f"coach_generation:{target.target_id}:attempt_{attempt}"
        entry = _manifest_entry(
            digest,
            target,
            generator_model,
            target_metrics,
            output_paths,
        )
        if entry is None:
            print(f"[warn] {target.target_id}: no response to evaluate", flush=True)
            continue
        existing[target.target_id] = entry
        if manifest_path is not None:
            _write_manifest(manifest_path, existing)
    return existing


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select and optionally run the Coach Digest Drift/control study."
    )
    parser.add_argument("--episodes-parquet", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--case-outcomes-parquet",
        type=Path,
        default=DEFAULT_CASE_OUTCOMES,
    )
    parser.add_argument("--wrangled-dir", type=Path, default=Path("logs/wrangled"))
    parser.add_argument("--parquet-path", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_REPORT_DIR / "judge_sample_manifest.json",
    )
    parser.add_argument(
        "--targets-out",
        type=Path,
        default=DEFAULT_REPORT_DIR / "targets.json",
    )
    parser.add_argument(
        "--group",
        choices=["drift", "control", "both"],
        default="both",
    )
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Authorize paid Weekly Drift Reviewer and Coach Digest calls.",
    )
    return parser


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)
    targets = build_targets(
        episodes_path=args.episodes_parquet,
        case_outcomes_path=args.case_outcomes_parquet,
        wrangled_dir=args.wrangled_dir,
        group=args.group,
        seed=args.seed,
        limit=args.limit,
    )
    generated_outputs_exist = (
        args.parquet_path.exists()
        or args.manifest_out.exists()
        or (
            args.output_dir.exists()
            and any(args.output_dir.iterdir())
        )
    )
    if args.execute and generated_outputs_exist and not args.resume:
        raise ValueError(
            "Generated outputs exist. Use --resume or select new output paths."
        )
    if (
        args.execute
        and args.resume
        and generated_outputs_exist
        and not args.manifest_out.exists()
    ):
        raise ValueError("Resume needs the existing manifest for generated outputs.")
    catalog = _target_catalog(
        targets,
        seed=args.seed,
        episodes_path=args.episodes_parquet,
        case_outcomes_path=args.case_outcomes_parquet,
        wrangled_dir=args.wrangled_dir,
    )
    catalog = merge_target_catalog(args.targets_out, catalog, resume=args.resume)
    args.targets_out.parent.mkdir(parents=True, exist_ok=True)
    args.targets_out.write_text(json.dumps(catalog, indent=2) + "\n")

    counts = {
        group: sum(target.group == group for target in targets)
        for group in ("drift", "control")
    }
    print(f"Drift targets: {counts['drift']}")
    print(f"Control targets: {counts['control']}")
    print(f"Targets: {args.targets_out}")
    if not args.execute:
        print("[dry run] No provider calls made.")
        return 0

    manifest = _manifest_index(args.manifest_out, resume=args.resume)
    manifest = asyncio.run(
        generate_missing_targets(
            targets,
            manifest,
            parquet_path=args.parquet_path,
            output_dir=args.output_dir,
            wrangled_dir=args.wrangled_dir,
            manifest_path=args.manifest_out,
        )
    )
    _write_manifest(args.manifest_out, manifest)
    print(f"Manifest responses: {len(manifest)}")
    print(f"Manifest: {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
