"""Prepare and score the North Star Moment Coach Digest impact pilot.

The pilot judges the 22 saved demo comparison pairs with the
``nsm-impact-judge`` Claude Code subagent (see
``docs/north_star/coach_digest_impact_eval.md``). ``prepare`` writes one blinded
task file per pair and order plus a sealed answer key; the judge writes one
verdict file per task; ``score`` joins verdicts to the key.

Judgments are AI review of synthetic Personas, not human validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "frontend/onboarding/public/scenarios"
SCENARIOS = (
    "active-nisha",
    "persistent-lukas",
    "stable-noor",
    "two-values-meera",
    "uncertain-wei-jun",
)
AGENT_PATH = ROOT / ".claude/agents/nsm-impact-judge.md"
PROMPT_VERSION = "nsm-impact-judge-1.0"
INPUT_MARKER = "\n\nUNTRUSTED INPUT DATA\n"
NARRATIVE_FIELDS = ("weekly_mirror", "tension_explanation", "reflective_question")
DEFAULT_SEED = 20260926
BOOTSTRAP_RESAMPLES = 10_000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def agent_settings(path: Path = AGENT_PATH) -> dict[str, str]:
    """Read the pinned model and effort from the judge's frontmatter."""
    frontmatter = path.read_text().split("---")[1]
    fields = dict(
        line.split(":", 1) for line in frontmatter.strip().splitlines() if ":" in line
    )
    return {key: fields[key].strip() for key in ("name", "model", "effort")}


def coach_input(arm: dict[str, Any]) -> dict[str, Any]:
    prompt = arm["base_prompt"]
    return json.loads(prompt[prompt.index(INPUT_MARKER) + len(INPUT_MARKER) :])


def response_text(narrative: dict[str, str]) -> str:
    return "\n\n".join(narrative[field] for field in NARRATIVE_FIELDS)


def entry_block(entry: dict[str, Any]) -> str:
    text = f"### Journal Entry, {entry['date']}\n\n{entry['content']}"
    if entry.get("nudge_response"):
        text += f"\n\n*Their reply to a follow-up question:* {entry['nudge_response']}"
    return text


def load_pairs(scenario_dir: Path = SCENARIO_DIR) -> list[dict[str, Any]]:
    """Extract every saved comparison pair with the facts the judge needs."""
    pairs = []
    for scenario_id in SCENARIOS:
        data = json.loads((scenario_dir / f"{scenario_id}.json").read_text())
        entries = {
            e["journal_entry_id"]: e for e in data["scenario"]["journal_entries"]
        }
        weeks = {w["week_start"]: w for w in data["scenario"]["weeks"]}
        for event in data["trace_events"]:
            pair = event["details"].get("comparison")
            if event["event_type"] != "weekly_coach_generated" or not isinstance(
                pair, dict
            ):
                continue
            without, with_moment = pair["without_north_star"], pair["with_north_star"]
            shared = coach_input(without)
            # The arms must differ only in the injected North Star Moment.
            if (
                shared["north_star_context"] is not None
                or {
                    **coach_input(with_moment),
                    "north_star_context": None,
                }
                != shared
            ):
                raise ValueError(f"{scenario_id} {pair['week_start']}: inputs differ")
            context = pair["north_star_context"]
            week_entries = [
                entries[i] for i in weeks[pair["week_start"]]["journal_entry_ids"]
            ]
            history = (
                []
                if context["source_id"] in {e["journal_entry_id"] for e in week_entries}
                else [entries[context["source_id"]]]
            )
            pairs.append(
                {
                    "pair_id": f"{scenario_id}:{pair['week_start']}",
                    "scenario_id": scenario_id,
                    "persona_id": pair["persona_id"],
                    "week_start": pair["week_start"],
                    "mode": context["mode"],
                    "shared_input": shared,
                    "week_entries": week_entries,
                    "history_entries": history,
                    "responses": {
                        "without": response_text(without["narrative"]),
                        "with": response_text(with_moment["narrative"]),
                    },
                }
            )
    return pairs


LABEL = r"- Internal Schwartz label: (?P<label>.+?) \| "
COMPASS_LINE = re.compile(LABEL + r'user-facing compass phrase: "(?P<phrase>.+)"$')
SUMMARY_LINE = re.compile(
    LABEL + r"(?P<state>No active Drift is confirmed|Drift is active)[^|]*"
    r"\| current run length: (?P<run>\d+) \| last decision: (?P<decision>\w+)"
)
CHANGE_LINE = re.compile(
    LABEL + r"previous state: (?P<previous>\w+) \| current state: \w+ "
    r"\| deterministic change: (?P<change>\w+)"
)
EVIDENCE_LINE = re.compile(
    r"- (?P<date>\S+) \| (?P<role>Drift evidence|weekly context) "
    r"\| internal Schwartz label\(s\): (?P<labels>[^|]+) \| excerpt: (?P<excerpt>.+)$"
)
DECISIONS = {
    "conflict": "Conflict",
    "not_conflict": "Not Conflict",
    "abstain": "no decision",
}
CHANGES = {
    "unchanged": "same as the previous week",
    "active_drift_started": "Active Drift started this week",
    "active_drift_ended": "Active Drift ended this week",
}


def _match(pattern: re.Pattern[str], line: str) -> re.Match[str]:
    match = pattern.match(line)
    if match is None:
        raise ValueError(f"Unrecognized Coach Digest input line: {line}")
    return match


def plain_facts(facts: dict[str, Any]) -> list[str]:
    """Restate the Coach Digest's input facts without internal field names."""
    phrases = {
        m["label"]: m["phrase"]
        for m in (_match(COMPASS_LINE, line) for line in facts["compass_context_lines"])
    }
    changes = {
        m["label"]: m["change"]
        for line in facts["state_comparison_lines"]
        if line.startswith("- Internal")
        for m in [_match(CHANGE_LINE, line)]
    }
    results = []
    for line in facts["drift_summary_lines"]:
        m = _match(SUMMARY_LINE, line)
        state = "Active Drift" if m["state"] == "Drift is active" else "No Active Drift"
        change = changes.get(m["label"])
        compared = CHANGES[change] if change else "no earlier week to compare"
        if m["decision"] not in DECISIONS:
            raise ValueError(f"Unrecognized review decision: {m['decision']}")
        decision = DECISIONS[m["decision"]]
        text = (
            f'- "{phrases[m["label"]]}": {state} at the end of this week '
            f"({compared}). Latest review decision: {decision}."
        )
        if int(m["run"]) > 0:
            text += f" Conflicts in a row so far: {m['run']}."
        results.append(text)
    evidence = []
    for line in facts["evidence_lines"]:
        m = _match(EVIDENCE_LINE, line)
        if m["role"] == "Drift evidence":
            values = ", ".join(
                f'"{phrases[v.strip()]}"' for v in m["labels"].split(",")
            )
            role = f"Conflict evidence for {values}"
        else:
            role = "context"
        evidence.append(f"- {m['date']}, {role}: {m['excerpt']}")
    return [
        "## The person's Core Values\n\n"
        + "\n".join(f'- "{phrase}"' for phrase in phrases.values()),
        "## Weekly Drift Detection result\n\n" + "\n".join(results),
        "## Cited Journal Entries\n\n" + "\n".join(evidence),
    ]


def task_text(task_id: str, pair: dict[str, Any], first: str) -> str:
    """Render a blinded task: no name, arm names, hashes, or North Star wording."""
    second = "with" if first == "without" else "without"
    facts = pair["shared_input"]
    sections = [
        f"# Task {task_id}",
        f"Week reviewed: {facts['week_window']}",
        *plain_facts(facts),
        "## This week's Journal Entries\n\n"
        + "\n\n".join(entry_block(e) for e in pair["week_entries"]),
    ]
    if pair["history_entries"]:
        sections.append(
            "## Earlier writing from the person's history\n\n"
            + "\n\n".join(entry_block(e) for e in pair["history_entries"])
        )
    sections += [
        f"## Response 1\n\n{pair['responses'][first]}",
        f"## Response 2\n\n{pair['responses'][second]}",
    ]
    return "\n\n".join(sections) + "\n"


def prepare(out: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    pairs = load_pairs()
    rng = random.Random(seed)
    tasks = []
    for pair in pairs:
        first = rng.choice(("with", "without"))
        for order, lead in enumerate((first, "without" if first == "with" else "with")):
            tasks.append((pair, order, lead))
    rng.shuffle(tasks)
    key = {}
    for index, (pair, order, lead) in enumerate(tasks, start=1):
        task_id = f"task-{index:02d}"
        (out / "tasks").mkdir(parents=True, exist_ok=True)
        (out / "tasks" / f"{task_id}.md").write_text(task_text(task_id, pair, lead))
        key[task_id] = {
            "pair_id": pair["pair_id"],
            "persona_id": pair["persona_id"],
            "mode": pair["mode"],
            "order": order,
            "label_of": {"1": lead, "2": "without" if lead == "with" else "with"},
        }
    write_json(out / "sealed" / "answer_key.json", key)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "prompt_version": PROMPT_VERSION,
        "judge": agent_settings(),
        "judge_agent_sha256": sha256_file(AGENT_PATH),
        "pairs": len(pairs),
        "tasks": len(key),
        "scenario_sha256": {
            s: sha256_file(SCENARIO_DIR / f"{s}.json") for s in SCENARIOS
        },
        "label_source": "AI review (Claude Code subagent), not human validation",
    }
    write_json(out / "manifest.json", manifest)
    (out / "RUN.md").write_text(run_instructions(out, sorted(key)))
    return manifest


def run_instructions(out: Path, task_ids: list[str]) -> str:
    rel = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    score_command = (
        f"uv run python -m scripts.experiments.nsm_impact_pilot score --out {rel}"
    )
    return "\n".join(
        [
            "# Running the pilot judge",
            "",
            "Spawn one `nsm-impact-judge` subagent per task, each in a fresh context.",
            "Pass no `model` override; the agent definition pins model and effort.",
            "Give each subagent only these two paths:",
            "",
            f"- task: `{rel}/tasks/<task_id>.md`",
            f"- verdict: `{rel}/verdicts/<task_id>.json`",
            "",
            "Never give a subagent the `sealed/` directory. Then run:",
            "",
            "```sh",
            score_command,
            "```",
            "",
            f"Tasks ({len(task_ids)}): {', '.join(task_ids)}",
            "",
        ]
    )


def load_verdict(path: Path, task_id: str) -> dict[str, Any]:
    verdict = json.loads(path.read_text())
    if verdict.get("task_id") != task_id or verdict.get("preferred") not in {
        "1",
        "2",
        "tie",
    }:
        raise ValueError(f"{path.name}: invalid task_id or preferred")
    flags = verdict.get("flags")
    if not isinstance(flags, dict) or not all(
        isinstance(flags.get(k), bool) for k in ("1", "2")
    ):
        raise ValueError(f"{path.name}: flags must hold booleans for 1 and 2")
    return verdict


def score_pairs(
    key: dict[str, dict[str, Any]],
    verdicts: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map each verdict to arms, then combine the two orders of each pair."""
    judgments = []
    for task_id, entry in key.items():
        verdict = verdicts[task_id]
        label_of = entry["label_of"]
        judgments.append(
            {
                "task_id": task_id,
                "pair_id": entry["pair_id"],
                "persona_id": entry["persona_id"],
                "mode": entry["mode"],
                "position": verdict["preferred"],
                "preferred": label_of.get(verdict["preferred"], "tie"),
                "flag_with": verdict["flags"]["1" if label_of["1"] == "with" else "2"],
                "flag_without": verdict["flags"][
                    "1" if label_of["1"] == "without" else "2"
                ],
            }
        )
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for judgment in judgments:
        by_pair.setdefault(judgment["pair_id"], []).append(judgment)
    pairs = []
    for pair_id, (first, second) in sorted(by_pair.items()):
        consistent = first["preferred"] == second["preferred"]
        outcome = (
            {"with": "win", "without": "loss", "tie": "tie"}[first["preferred"]]
            if consistent
            else "tie"
        )
        pairs.append(
            {
                "pair_id": pair_id,
                "persona_id": first["persona_id"],
                "mode": first["mode"],
                "verdicts": [first["preferred"], second["preferred"]],
                "consistent": consistent,
                "outcome": outcome,
            }
        )
    return judgments, pairs


def win_difference(pairs: list[dict[str, Any]]) -> float | None:
    if not pairs:
        return None
    wins = sum(p["outcome"] == "win" for p in pairs)
    losses = sum(p["outcome"] == "loss" for p in pairs)
    return 100 * (wins - losses) / len(pairs)


def persona_bootstrap(
    pairs: list[dict[str, Any]],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[float] | None:
    """95% percentile interval from resampling whole Personas."""
    clusters: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        clusters.setdefault(pair["persona_id"], []).append(pair)
    groups = list(clusters.values())
    if len(groups) < 2:
        return None
    rng = random.Random(seed)
    stats = sorted(
        win_difference(
            [p for group in rng.choices(groups, k=len(groups)) for p in group]
        )
        for _ in range(resamples)
    )
    return [
        round(stats[int(0.025 * resamples)], 1),
        round(stats[int(0.975 * resamples) - 1], 1),
    ]


def summarize(
    judgments: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    decisive = [j for j in judgments if j["position"] != "tie"]
    count = {o: sum(p["outcome"] == o for p in pairs) for o in ("win", "loss", "tie")}
    difference = win_difference(pairs)
    return {
        "pairs": len(pairs),
        "judgments": len(judgments),
        "order_consistency": {
            "consistent_pairs": sum(p["consistent"] for p in pairs),
            "pairs": len(pairs),
        },
        "position_choice": {
            "response_1": sum(j["position"] == "1" for j in decisive),
            "response_2": sum(j["position"] == "2" for j in decisive),
            "decisive_judgments": len(decisive),
        },
        "outcomes": count,
        "win_difference": None if difference is None else round(difference, 1),
        "win_difference_95": persona_bootstrap(pairs),
        "honesty_flags": {
            "with": sum(j["flag_with"] for j in judgments),
            "without": sum(j["flag_without"] for j in judgments),
            "judgments": len(judgments),
        },
        "by_mode": {
            mode: {
                "pairs": len(subset),
                "outcomes": {
                    o: sum(p["outcome"] == o for p in subset)
                    for o in ("win", "loss", "tie")
                },
            }
            for mode in sorted({p["mode"] for p in pairs})
            for subset in [[p for p in pairs if p["mode"] == mode]]
        },
    }


def report(summary: dict[str, Any], manifest: dict[str, Any]) -> str:
    oc = summary["order_consistency"]
    pc = summary["position_choice"]
    fl = summary["honesty_flags"]
    judge = manifest["judge"]
    wins, losses, ties = (summary["outcomes"][o] for o in ("win", "loss", "tie"))
    rows = [
        f"| {mode} | {v['pairs']} | "
        + " | ".join(str(v["outcomes"][o]) for o in ("win", "loss", "tie"))
        + " |"
        for mode, v in summary["by_mode"].items()
    ]
    lines = [
        "# North Star Moment impact pilot",
        "",
        f"Judge: `{judge['model']}` at effort `{judge['effort']}`, run as the "
        f"`{judge['name']}` Claude Code subagent "
        f"(prompt `{manifest['prompt_version']}`).",
        f"Pairs: the {summary['pairs']} saved demo comparison pairs, "
        "each judged in both orders.",
        "These are AI review results on synthetic Personas, not human validation.",
        "",
        "## Judge reliability",
        "",
        f"- Order consistency: {oc['consistent_pairs']}/{oc['pairs']} pairs gave "
        "the same verdict in both orders.",
        f"- Position choice among decisive judgments: Response 1 {pc['response_1']}, "
        f"Response 2 {pc['response_2']} (of {pc['decisive_judgments']}).",
        "",
        "## Result",
        "",
        f"- Wins {wins}, losses {losses}, ties {ties} "
        "(inconsistent orders count as ties).",
        f"- Win difference: {summary['win_difference']} points; 95% "
        f"Persona-resampled range: {summary['win_difference_95']}.",
        "  With only five Personas, this range is a rough indication.",
        "",
        "| North Star Moment mode | Pairs | Wins | Losses | Ties |",
        "| --- | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Honesty flags",
        "",
        f"Flagged judgments: with the North Star Moment {fl['with']}/"
        f"{fl['judgments']}, without {fl['without']}/{fl['judgments']}.",
        "",
    ]
    return "\n".join(lines)


def score(out: Path) -> dict[str, Any]:
    key = json.loads((out / "sealed" / "answer_key.json").read_text())
    missing = [t for t in key if not (out / "verdicts" / f"{t}.json").exists()]
    if missing:
        raise SystemExit(f"Missing verdicts: {', '.join(missing)}")
    verdicts = {t: load_verdict(out / "verdicts" / f"{t}.json", t) for t in key}
    judgments, pairs = score_pairs(key, verdicts)
    summary = summarize(judgments, pairs)
    manifest = json.loads((out / "manifest.json").read_text())
    write_json(out / "pairs.json", pairs)
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text(report(summary, manifest))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    if args.command == "prepare":
        if (out / "sealed").exists():
            sys.exit(f"{out} already prepared; use a new --out directory")
        print(json.dumps(prepare(out, args.seed), indent=2))
    else:
        print(json.dumps(score(out), indent=2))


if __name__ == "__main__":
    main()
