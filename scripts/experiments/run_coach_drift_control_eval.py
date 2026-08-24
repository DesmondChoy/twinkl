"""Generate Coach Digest responses for Drift episodes and matched control weeks.

The earlier Coach Digest evaluation sampled five Personas that had no Drift, so
it measured the Weekly Drift Coach only on stable weeks. This script builds a
sample that covers Drift:

  drift arm    one target for each Drift episode, with ``end_date`` set to the
               episode confirmation date. The runtime truncates history at that
               date and reports on the last week, so the confirmation week
               becomes the reported week.

  control arm  one matched target for each Drift episode, taken from the
               Personas that have no Drift episode at all. Matching keeps
               history length comparable, because evidence volume and the
               groundedness check both scale with history length.

Target selection is a pure function of the two Parquet files, the wrangled
directory, and the seed. Run without ``--execute`` to print and save the target
list before any paid call.

Run:
    .venv/bin/python scripts/experiments/run_coach_drift_control_eval.py \
        --arm both --execute
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

# Allow running as a bare script by putting the repo root on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

from src.coach.llm_client import build_llm_complete, resolve_coach_model  # noqa: E402
from src.coach.schemas import WeeklyDigest  # noqa: E402
from src.coach.weekly_drift_runtime import (  # noqa: E402
    _week_bounds,
    run_weekly_drift_coach_cycle,
)
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
    "logs/experiments/reports/coach_narrative_drift_control_20260823"
)
DEFAULT_PARQUET = Path(
    "logs/exports/weekly_digests/coach_drift_control_eval_20260823.parquet"
)
DEFAULT_OUTPUT_DIR = Path("logs/exports/weekly_drift_coach/drift_control_20260823")
DEFAULT_SEED = 20260823

Arm = Literal["drift", "control"]


def _entry_count_bucket(entry_count: int) -> str:
    """Bucket history length. Finer bins do not fill at this sample size."""
    if entry_count <= 6:
        return "<=6"
    if entry_count <= 9:
        return "7-9"
    return "10-12"


@dataclass(frozen=True)
class EvalTarget:
    """One Persona week to generate a Coach Digest response for."""

    target_id: str
    persona_id: str
    end_date: str
    arm: str
    historical_split: str
    entry_count: int
    n_truncated_weeks: int
    dimension: str | None = None
    delivery_state: str | None = None
    episode_id: str | None = None
    matched_to: str | None = None
    match_quality: str = "exact"


def _persona_week_ends(wrangled_dir: Path, persona_id: str) -> list[str]:
    """Return each ISO week end in a Persona's history, in order."""
    path = wrangled_dir / f"persona_{persona_id}.md"
    _profile, entries, _warnings = parse_wrangled_file(path)
    week_ends = {_week_bounds(str(entry["date"]))[1] for entry in entries}
    return [week_end.isoformat() for week_end in sorted(week_ends)]


def _truncated_week_count(week_ends: list[str], end_date: str) -> int:
    """Weeks the runtime reviews when history is truncated at ``end_date``.

    This drives the reviewer call count and the evidence available to the
    Coach Digest response, so the control arm matches on it.
    """
    cutoff = _week_bounds(end_date)[1].isoformat()
    return sum(1 for week_end in week_ends if week_end <= cutoff)


def load_drift_targets(
    episodes_path: Path,
    case_outcomes_path: Path,
    wrangled_dir: Path,
) -> list[EvalTarget]:
    """Build one target per Drift episode, reported on the confirmation week."""
    episodes = pl.read_parquet(episodes_path)
    cases = pl.read_parquet(case_outcomes_path)
    case_index = {
        (row["persona_id"], row["dimension"]): row for row in cases.to_dicts()
    }

    week_ends_cache: dict[str, list[str]] = {}
    targets: list[EvalTarget] = []
    for row in episodes.to_dicts():
        persona_id = row["persona_id"]
        case = case_index[(persona_id, row["dimension"])]
        if persona_id not in week_ends_cache:
            week_ends_cache[persona_id] = _persona_week_ends(wrangled_dir, persona_id)
        end_date = row["confirmation_date"]
        targets.append(
            EvalTarget(
                target_id=f"{persona_id}:{end_date}:drift",
                persona_id=persona_id,
                end_date=end_date,
                arm="drift",
                historical_split=case["historical_split"],
                entry_count=int(case["entry_count"]),
                n_truncated_weeks=_truncated_week_count(
                    week_ends_cache[persona_id], end_date
                ),
                dimension=row["dimension"],
                delivery_state=row["delivery_state"],
                episode_id=row["episode_id"],
            )
        )
    return targets


def sample_control_targets(
    case_outcomes_path: Path,
    episodes_path: Path,
    drift_targets: list[EvalTarget],
    wrangled_dir: Path,
    seed: int = DEFAULT_SEED,
) -> list[EvalTarget]:
    """Draw one non-Drift control week for each Drift episode.

    The control pool holds only Personas with no Drift episode in any Core
    Value. A Persona that drifts elsewhere is not a clean control, because its
    Coach Digest response may refer to the drifting Core Value.

    Controls are matched on ``historical_split`` and history length bucket. The
    pool skews shorter than the Drift arm, so unmatched sampling would confound
    history length with Drift.
    """
    episodes = pl.read_parquet(episodes_path)
    drift_personas = set(episodes["persona_id"].to_list())
    cases = pl.read_parquet(case_outcomes_path)
    pool = [
        row
        for row in cases.to_dicts()
        if row["persona_id"] not in drift_personas and not row["has_drift"]
    ]

    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        bucket = _entry_count_bucket(int(row["entry_count"]))
        by_stratum[(row["historical_split"], bucket)].append(row)
        by_split[row["historical_split"]].append(row)

    rng = random.Random(seed)
    for rows in by_stratum.values():
        rng.shuffle(rows)
    used_personas: set[str] = set()
    week_ends_cache: dict[str, list[str]] = {}
    controls: list[EvalTarget] = []

    for drift_target in sorted(drift_targets, key=lambda t: t.target_id):
        bucket = _entry_count_bucket(drift_target.entry_count)
        candidates, quality = _candidate_ladder(
            by_stratum=by_stratum,
            by_split=by_split,
            pool=pool,
            split=drift_target.historical_split,
            bucket=bucket,
            used_personas=used_personas,
        )
        if not candidates:
            continue

        # Prefer a matching Core Value, as a tie break only. Hard matching would
        # over constrain: universalism alone carries 12 Drift episodes.
        preferred = [
            row for row in candidates if row["dimension"] == drift_target.dimension
        ]
        chosen = rng.choice(preferred) if preferred else rng.choice(candidates)
        persona_id = chosen["persona_id"]
        used_personas.add(persona_id)

        if persona_id not in week_ends_cache:
            week_ends_cache[persona_id] = _persona_week_ends(wrangled_dir, persona_id)
        week_ends = week_ends_cache[persona_id]
        if not week_ends:
            continue

        # Match the reviewed week count, which drives evidence volume.
        end_date = min(
            week_ends,
            key=lambda week_end: (
                abs(
                    _truncated_week_count(week_ends, week_end)
                    - drift_target.n_truncated_weeks
                ),
                week_end,
            ),
        )
        controls.append(
            EvalTarget(
                target_id=f"{persona_id}:{end_date}:control",
                persona_id=persona_id,
                end_date=end_date,
                arm="control",
                historical_split=chosen["historical_split"],
                entry_count=int(chosen["entry_count"]),
                n_truncated_weeks=_truncated_week_count(week_ends, end_date),
                dimension=chosen["dimension"],
                matched_to=drift_target.episode_id,
                match_quality=quality,
            )
        )
    return controls


def _candidate_ladder(
    *,
    by_stratum: dict[tuple[str, str], list[dict[str, Any]]],
    by_split: dict[str, list[dict[str, Any]]],
    pool: list[dict[str, Any]],
    split: str,
    bucket: str,
    used_personas: set[str],
) -> tuple[list[dict[str, Any]], str]:
    """Find control candidates, relaxing the match only when a level is empty."""

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
        return same_split, "bucket_relaxed"

    return available(pool), "random"


def build_targets(
    *,
    episodes_path: Path,
    case_outcomes_path: Path,
    wrangled_dir: Path,
    arm: str,
    seed: int,
    limit: int | None,
) -> list[EvalTarget]:
    """Resolve the full target list for the requested arm or arms."""
    drift_targets = load_drift_targets(episodes_path, case_outcomes_path, wrangled_dir)
    control_targets = (
        sample_control_targets(
            case_outcomes_path,
            episodes_path,
            drift_targets,
            wrangled_dir,
            seed=seed,
        )
        if arm in {"control", "both"}
        else []
    )
    if arm == "control":
        drift_targets = []

    if limit is not None:
        drift_targets = drift_targets[:limit]
        control_targets = control_targets[:limit]
    return drift_targets + control_targets


def _digest_to_manifest_entry(
    digest: WeeklyDigest,
    target: EvalTarget,
    generator_model: str,
) -> dict[str, Any] | None:
    """Build a judge manifest entry that also carries its evaluation arm."""
    if digest.coach_narrative is None:
        return None
    return {
        "digest": {
            "persona_id": digest.persona_id,
            "persona_name": digest.persona_name,
            "week_start": digest.week_start,
            "week_end": digest.week_end,
            "response_mode": digest.response_mode,
            "mode_source": digest.mode_source,
            "mode_rationale": digest.mode_rationale,
            "signal_source": digest.signal_source,
            "n_entries": digest.n_entries,
            "overall_mean": digest.overall_mean,
            "overall_uncertainty": digest.overall_uncertainty,
            "core_values": digest.core_values,
            "goal_context": digest.goal_context,
            "drift_states": digest.drift_states,
            "drift_reasons": digest.drift_reasons,
            "top_tensions": digest.top_tensions,
            "top_strengths": digest.top_strengths,
            # state_comparisons feeds the state_claims validation check. Without
            # it a manifest driven re-validation fails that check by mistake.
            "state_comparisons": [row.model_dump() for row in digest.state_comparisons],
            "dimensions": [row.model_dump() for row in digest.dimensions],
            "evidence": [row.model_dump() for row in digest.evidence],
        },
        "narrative": digest.coach_narrative.model_dump(),
        "validation": (
            digest.validation.model_dump() if digest.validation is not None else None
        ),
        "generator_model": generator_model,
        "target": asdict(target),
    }


def _completed_keys(parquet_path: Path) -> set[tuple[str, str]]:
    """Persona weeks already generated, so --resume does not pay twice."""
    if not parquet_path.exists():
        return set()
    frame = pl.read_parquet(parquet_path)
    return {
        (row["persona_id"], row["week_end"])
        for row in frame.select(["persona_id", "week_end"]).to_dicts()
    }


async def _generate(
    targets: list[EvalTarget],
    *,
    parquet_path: Path,
    output_dir: Path,
    wrangled_dir: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    coach_llm_complete = build_llm_complete()
    if coach_llm_complete is None:
        raise SystemExit(
            "No Weekly Coach provider available (missing API key). "
            "Set OPENAI_API_KEY or GEMINI_API_KEY."
        )
    generator_model = resolve_coach_model()
    done = _completed_keys(parquet_path) if resume else set()

    manifest: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        week_end = _week_bounds(target.end_date)[1].isoformat()
        if (target.persona_id, week_end) in done:
            print(f"[skip] {target.target_id} already generated", flush=True)
            continue
        print(f"[{index}/{len(targets)}] {target.target_id} ...", flush=True)
        digest, _artifacts = await run_weekly_drift_coach_cycle(
            persona_id=target.persona_id,
            wrangled_dir=wrangled_dir,
            output_dir=output_dir,
            parquet_path=parquet_path,
            end_date=target.end_date,
            coach_llm_complete=coach_llm_complete,
            # Record failed validations; the failure rate is the measurement.
            attach_failed_validation=True,
        )
        entry = _digest_to_manifest_entry(digest, target, generator_model)
        if entry is None:
            print(
                f"[warn] {target.target_id}: no Coach Narrative produced; "
                "excluded from the manifest.",
                flush=True,
            )
            continue
        manifest.append(entry)
        passed = digest.validation.all_passed if digest.validation else None
        print(
            f"[done] {target.target_id} mode={digest.response_mode} "
            f"drift_states={digest.drift_states} validation_passed={passed}",
            flush=True,
        )
    return manifest


def _summarize(targets: list[EvalTarget]) -> str:
    """Describe the resolved sample, for review before any paid call."""
    drift = [t for t in targets if t.arm == "drift"]
    control = [t for t in targets if t.arm == "control"]
    lines = [
        f"Drift targets:   {len(drift)}",
        f"Control targets: {len(control)}",
        f"Reviewer calls:  {sum(t.n_truncated_weeks for t in targets)} "
        "(one per reviewed week)",
        f"Coach calls:     {len(targets)}",
    ]
    keys = [(t.persona_id, t.end_date) for t in targets]
    if len(set(keys)) != len(keys):
        lines.append("WARNING: duplicate Persona weeks in the sample")
    quality: dict[str, int] = defaultdict(int)
    for target in control:
        quality[target.match_quality] += 1
    if quality:
        lines.append("Match quality:   " + ", ".join(
            f"{name}={count}" for name, count in sorted(quality.items())
        ))
    if drift and control:
        drift_weeks = sum(t.n_truncated_weeks for t in drift) / len(drift)
        control_weeks = sum(t.n_truncated_weeks for t in control) / len(control)
        lines.append(
            f"Mean reviewed weeks: drift={drift_weeks:.2f} "
            f"control={control_weeks:.2f}"
        )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Coach Digest responses for Drift episodes and matched "
            "control weeks."
        )
    )
    parser.add_argument("--episodes-parquet", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--case-outcomes-parquet", type=Path, default=DEFAULT_CASE_OUTCOMES
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
        "--targets-out", type=Path, default=DEFAULT_REPORT_DIR / "targets.json"
    )
    parser.add_argument(
        "--arm", choices=["drift", "control", "both"], default="both"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Take only the first N targets of each arm, for a pilot run.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip Persona weeks already present in the target Parquet.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Authorize the paid Weekly Drift Reviewer and Coach calls.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)

    targets = build_targets(
        episodes_path=args.episodes_parquet,
        case_outcomes_path=args.case_outcomes_parquet,
        wrangled_dir=args.wrangled_dir,
        arm=args.arm,
        seed=args.seed,
        limit=args.limit,
    )

    args.targets_out.parent.mkdir(parents=True, exist_ok=True)
    args.targets_out.write_text(
        json.dumps(
            {"seed": args.seed, "arm": args.arm,
             "targets": [asdict(t) for t in targets]},
            indent=2,
        )
        + "\n"
    )
    print(_summarize(targets))
    print(f"targets: {args.targets_out}")

    if not args.execute:
        print(
            "\n[dry run] No calls made. Review the target list, then re-run "
            "with --execute."
        )
        return 0

    manifest = asyncio.run(
        _generate(
            targets,
            parquet_path=args.parquet_path,
            output_dir=args.output_dir,
            wrangled_dir=args.wrangled_dir,
            resume=args.resume,
        )
    )
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {len(manifest)} manifest entries: {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
