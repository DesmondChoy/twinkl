"""Trajectory-level EDA on judge labels to ground the drift-definition debate.

Treats the draft drift taxonomy (docs/evolution/drift_detection.md) as a
hypothesis menu, not a spec: measures per-dimension base rates of each
candidate pattern (dip / sustained conflict / fade / rise / spike) across a parameter
grid, at entry-level and weekly granularity, gated by declared core values.
Consensus labels are the default reference because the drift benchmark should
target the more stable five-pass resolver output, while still reporting
persisted single-pass label impact for comparison.

Outputs figures + CSV tables to docs/drift/ and a stats summary to stdout.

Run: .venv/bin/python scripts/drift/trajectory_eda.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "drift"
FIG = OUT / "figures"
TAB = OUT / "tables"

LABEL_SOURCES = {
    "judge": ROOT / "logs/judge_labels/judge_labels.parquet",
    "consensus": ROOT / "logs/judge_labels/consensus_labels.parquet",
}

DIMS = [
    "self_direction", "stimulation", "hedonism", "achievement", "power",
    "security", "conformity", "tradition", "benevolence", "universalism",
]
CORE_NAME_MAP = {
    "Self-Direction": "self_direction", "Stimulation": "stimulation",
    "Hedonism": "hedonism", "Achievement": "achievement", "Power": "power",
    "Security": "security", "Conformity": "conformity",
    "Tradition": "tradition", "Benevolence": "benevolence",
    "Universalism": "universalism",
}
DIM_LABELS = {v: k for k, v in CORE_NAME_MAP.items()}

# dataviz reference palette (light mode)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
RED = "#e34948"
DIV_NEUTRAL = "#f0efec"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlecolor": INK, "font.size": 10,
})


# ---------------------------------------------------------------- pattern fns

def max_run(seq: list[int], val: int) -> int:
    best = cur = 0
    for s in seq:
        cur = cur + 1 if s == val else 0
        best = max(best, cur)
    return best


def dip_events(seq: list[int]) -> list[tuple[int, int]]:
    """(index, severity) for transitions into -1; severity = size of drop."""
    return [
        (i, seq[i - 1] - seq[i])
        for i in range(1, len(seq))
        if seq[i] == -1 and seq[i - 1] >= 0
    ]


def has_fade(seq: list[int], c_min: int) -> bool:
    """+1 followed immediately by >= c_min consecutive 0s."""
    for i, s in enumerate(seq):
        if s == 1 and i + c_min < len(seq):
            if all(x == 0 for x in seq[i + 1 : i + 1 + c_min]):
                return True
    return False


def dip_recovers(seq: list[int], idx: int, within: int = 2) -> bool:
    """Does the score return to >= 0 within `within` steps after a dip?"""
    return any(s >= 0 for s in seq[idx + 1 : idx + 1 + within])


def weekly_low_mean(means: list[float], tau: float, c_min: int) -> bool:
    return max_run([1 if m < tau else 0 for m in means], 1) >= c_min


# ------------------------------------------------------------------ load data

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run drift trajectory EDA over persisted or consensus judge labels."
    )
    parser.add_argument(
        "--labels",
        choices=sorted(LABEL_SOURCES),
        default="consensus",
        help=(
            "Label table to use for generated figures/tables. The default is "
            "consensus; the stdout summary also includes judge-vs-consensus "
            "impact for the single-definition options."
        ),
    )
    parser.add_argument(
        "--week-mode",
        choices=("runtime", "persona_anchor"),
        default="runtime",
        help=(
            "Weekly binning mode. runtime matches src/vif/runtime.py "
            "dt.truncate('1w'); persona_anchor preserves the first-entry-relative "
            "bins from the initial EDA."
        ),
    )
    return parser.parse_args()


def load(label_source: str) -> tuple[pl.DataFrame, dict[str, list[str]], Path]:
    label_path = LABEL_SOURCES[label_source]
    lab = (
        pl.read_parquet(label_path)
        .with_columns(pl.col("date").str.to_date())
        .sort("persona_id", "t_index")
    )
    reg = pl.read_parquet(ROOT / "logs/registry/personas.parquet")
    core = {
        r["persona_id"]: [CORE_NAME_MAP[v] for v in r["core_values"]]
        for r in reg.iter_rows(named=True)
    }
    return lab, core, label_path


def build_sequences(lab: pl.DataFrame, *, week_mode: str):
    """Per persona: entry-level sequences per dim + weekly means per dim."""
    personas = {}
    for pid, grp in lab.group_by("persona_id", maintain_order=True):
        pid = pid[0]
        dates = grp["date"].to_list()
        first = min(dates)
        if week_mode == "persona_anchor":
            week_idx = [(d - first).days // 7 for d in dates]
        elif week_mode == "runtime":
            week_idx = grp.select(
                pl.col("date").dt.truncate("1w").alias("_week")
            )["_week"].to_list()
        else:
            raise ValueError(f"Unsupported week_mode: {week_mode}")
        entry = {d: grp[f"alignment_{d}"].to_list() for d in DIMS}
        weekly = {}
        n_by_week = defaultdict(int)
        for w in week_idx:
            n_by_week[w] += 1
        weeks = sorted(n_by_week)
        for d in DIMS:
            sums = defaultdict(float)
            negs = defaultdict(int)
            for w, s in zip(week_idx, entry[d]):
                sums[w] += s
                negs[w] += s == -1
            weekly[d] = {
                "weeks": weeks,
                "mean": [sums[w] / n_by_week[w] for w in weeks],
                "neg_density": [negs[w] / n_by_week[w] for w in weeks],
                "n": [n_by_week[w] for w in weeks],
            }
        personas[pid] = {
            "entry": entry,
            "weekly": weekly,
            "n_entries": len(dates),
            "n_weeks": len(weeks),
            "span_days": (max(dates) - first).days,
        }
    return personas


# ------------------------------------------------------------------- analyses

def pattern_grid(personas, core):
    """Prevalence of each pattern per dim, all-dims vs core-gated."""
    rows = []
    n_personas = len(personas)
    core_count = {d: sum(d in cvs for cvs in core.values()) for d in DIMS}

    def add(pattern, param, flags):
        # flags: dict[(pid, dim) -> bool]
        for d in DIMS:
            all_hits = sum(flags[(p, d)] for p in personas)
            core_hits = sum(flags[(p, d)] for p in personas if d in core[p])
            rows.append({
                "pattern": pattern, "param": param, "dim": d,
                "all_hits": all_hits, "all_n": n_personas,
                "core_hits": core_hits, "core_n": core_count[d],
            })

    def flags_for(fn):
        return {(p, d): fn(personas[p]["entry"][d]) for p in personas for d in DIMS}

    add("dip_hard (+1→-1)", "entry", flags_for(
        lambda s: any(sev == 2 for _, sev in dip_events(s))))
    add("dip_any (0/+1→-1)", "entry", flags_for(
        lambda s: len(dip_events(s)) > 0))
    for c in (2, 3, 4):
        add("sustained (-1 run)", f"C={c}", flags_for(lambda s, c=c: max_run(s, -1) >= c))
    for c in (2, 3):
        add("fade (+1→0s)", f"C={c}", flags_for(lambda s, c=c: has_fade(s, c)))
    for c in (2, 3, 4):
        add("rise (+1 run)", f"C={c}", flags_for(lambda s, c=c: max_run(s, 1) >= c))
    add("all-neutral", "entry", flags_for(lambda s: all(x == 0 for x in s)))
    add("never +1 (n>=3)", "entry", flags_for(
        lambda s: len(s) >= 3 and max(s) < 1))

    # weekly patterns
    for delta in (0.5, 1.0, 1.5, 2.0):
        fl = {}
        for p in personas:
            for d in DIMS:
                m = personas[p]["weekly"][d]["mean"]
                fl[(p, d)] = any(m[i - 1] - m[i] >= delta for i in range(1, len(m)))
        add("weekly mean drop", f"δ={delta}", fl)
    for tau, c in [(-0.4, 2), (-0.4, 3), (-0.25, 2), (-0.25, 3)]:
        fl = {}
        for p in personas:
            for d in DIMS:
                fl[(p, d)] = weekly_low_mean(personas[p]["weekly"][d]["mean"], tau, c)
        add("low-mean weeks (mean<τ)", f"τ={tau},C={c}", fl)

    return pl.DataFrame(rows)


def spike_stats(personas, core):
    """How often do dips self-recover within 2 steps (spike = noise)?"""
    out = {"all": [0, 0], "core": [0, 0]}  # [recovered, total]
    for p, data in personas.items():
        for d in DIMS:
            seq = data["entry"][d]
            for idx, _sev in dip_events(seq):
                if idx + 1 >= len(seq):
                    continue  # dip at trajectory end: recovery unobservable
                rec = dip_recovers(seq, idx)
                out["all"][1] += 1
                out["all"][0] += rec
                if d in core[p]:
                    out["core"][1] += 1
                    out["core"][0] += rec
    return out


def run_length_stats(personas):
    """Distribution of maximal -1 runs across persona-dim trajectories."""
    runs = defaultdict(int)
    for data in personas.values():
        for d in DIMS:
            r = max_run(data["entry"][d], -1)
            runs[r] += 1
    return dict(sorted(runs.items()))


def conflict_heavy_week_candidates(personas, core):
    """Label-derived candidate conflict-heavy weeks for the wq9p benchmark."""
    rows = []
    for p, data in personas.items():
        for d in DIMS:
            wk = data["weekly"][d]
            for i, w in enumerate(wk["weeks"]):
                rows.append({
                    "persona_id": p, "dim": d, "week": w,
                    "n_entries": wk["n"][i], "mean": wk["mean"][i],
                    "neg_density": wk["neg_density"][i],
                    "is_core": d in core[p],
                })
    return pl.DataFrame(rows)


def single_definition_impact(personas, core):
    """Persona impact of the three candidate single drift definitions.

    R1: a core dim scores -1 on >=2 consecutive entries.
    R2: a week contains >=2 entries scoring -1 on a core dim.
    R3: observed departure on a core dim: >=0 -> -1 -> -1.
    """
    r1, r2, r3 = set(), set(), set()
    for p, data in personas.items():
        for d in core[p]:
            seq = data["entry"][d]
            if max_run(seq, -1) >= 2:
                r1.add(p)
            wk = data["weekly"][d]
            if any(dens * n >= 2 for dens, n in zip(wk["neg_density"], wk["n"])):
                r2.add(p)
            if any(seq[i - 2] >= 0 and seq[i - 1] == -1 and seq[i] == -1
                   for i in range(2, len(seq))):
                r3.add(p)
    n = len(personas)
    return {
        "R1 sustained conflict (-1 run >= 2)": (len(r1), n),
        "R2 conflict week (>=2 -1 entries in a week)": (len(r2), n),
        "R3 unrecovered departure (>=0 -> -1 -> -1)": (len(r3), n),
        "union": (len(r1 | r2 | r3), n),
    }


def impact_comparison(*, week_mode: str) -> pl.DataFrame:
    """Judge-vs-consensus impact for the de-scoped single-definition options."""
    rows = []
    for label_source in ("judge", "consensus"):
        lab, core, label_path = load(label_source)
        personas = build_sequences(lab, week_mode=week_mode)
        impacts = single_definition_impact(personas, core)
        for definition, (personas_flagged, n_personas) in impacts.items():
            rows.append(
                {
                    "label_source": label_source,
                    "label_path": str(label_path.relative_to(ROOT)),
                    "week_mode": week_mode,
                    "definition": definition,
                    "personas_flagged": personas_flagged,
                    "n_personas": n_personas,
                    "pct_personas": round(100 * personas_flagged / n_personas, 1),
                }
            )
    return pl.DataFrame(rows)


def persona_coverage(personas, core):
    """Share of personas flagged on >=1 core dim, per pattern/param."""
    rows = []
    defs = {
        ("dip_hard (+1→-1)", "entry"): lambda s: any(sev == 2 for _, sev in dip_events(s)),
        ("dip_any (0/+1→-1)", "entry"): lambda s: len(dip_events(s)) > 0,
        ("sustained (-1 run)", "C=2"): lambda s: max_run(s, -1) >= 2,
        ("sustained (-1 run)", "C=3"): lambda s: max_run(s, -1) >= 3,
        ("fade (+1→0s)", "C=2"): lambda s: has_fade(s, 2),
    }
    for (pattern, param), fn in defs.items():
        hit = sum(
            any(fn(personas[p]["entry"][d]) for d in core[p]) for p in personas
        )
        rows.append({"pattern": pattern, "param": param,
                     "personas_flagged": hit, "n_personas": len(personas)})
    return pl.DataFrame(rows)


# -------------------------------------------------------------------- figures

def fig_structure(personas):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    n_entries = [d["n_entries"] for d in personas.values()]
    spans = [d["span_days"] for d in personas.values()]
    n_weeks = [d["n_weeks"] for d in personas.values()]
    for ax, (vals, title, bins) in zip(axes, [
        (n_entries, "Entries per persona", np.arange(1.5, 13.5)),
        (spans, "Span (days)", np.arange(0, 80, 7)),
        (n_weeks, "Active weeks per persona", np.arange(0.5, 12.5)),
    ]):
        ax.hist(vals, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_axisbelow(True)
        med = float(np.median(vals))
        ax.axvline(med, color=INK2, linewidth=1, linestyle="--")
        ax.text(med, ax.get_ylim()[1] * 0.97, f" median {med:.0f}",
                color=INK2, fontsize=8, va="top",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    fig.suptitle("Trajectory structure — 204 personas, 1,651 judge-labeled entries",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "fig1_structure.png", dpi=160)
    plt.close(fig)


def fig_trajectories(personas, core):
    """Score strips for the 18 core-dim trajectories with most activity."""
    scored = []
    for p, data in personas.items():
        for d in core[p]:
            seq = data["entry"][d]
            activity = sum(1 for s in seq if s != 0)
            n_neg = sum(1 for s in seq if s == -1)
            scored.append((n_neg, activity, p, d, seq))
    scored.sort(reverse=True)
    top = scored[:18]

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    cmap = {-1: RED, 0: DIV_NEUTRAL, 1: BLUE}
    for row, (_, _, p, d, seq) in enumerate(top):
        for i, s in enumerate(seq):
            ax.add_patch(plt.Rectangle(
                (i, row), 0.92, 0.82, facecolor=cmap[s],
                edgecolor=SURFACE, linewidth=0.5))
        ax.text(-0.4, row + 0.41, f"{p[:8]} · {DIM_LABELS[d]}",
                ha="right", va="center", fontsize=8, color=INK2)
    ax.set_xlim(-6, 13)
    ax.set_ylim(-0.5, len(top) + 0.3)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks(range(0, 13, 2))
    ax.set_xlabel("entry index (t)")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Most-negative core-value trajectories (entry-level judge scores)",
                 fontsize=11, loc="left", color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=BASE, linewidth=0.5)
               for c in (RED, DIV_NEUTRAL, BLUE)]
    ax.legend(handles, ["−1 conflict", "0 neutral", "+1 affirm"],
              loc="lower right", bbox_to_anchor=(1.0, 1.005), frameon=False,
              fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_trajectories.png", dpi=160)
    plt.close(fig)


def fig_prevalence(grid: pl.DataFrame):
    """Per-dimension core-gated prevalence, one panel per pattern."""
    panels = [
        ("dip_any (0/+1→-1)", "entry", "Dip into −1 (any)"),
        ("sustained (-1 run)", "C=2", "Sustained: ≥2 consecutive −1"),
        ("fade (+1→0s)", "C=2", "Fade: +1 then ≥2 zeros"),
        ("never +1 (n>=3)", "entry", "Never +1 (onboarding gap)"),
    ]
    # one fixed dim order across panels, sorted by dip prevalence
    first = grid.filter(
        (pl.col("pattern") == panels[0][0]) & (pl.col("param") == panels[0][1])
    ).with_columns(pct=pl.col("core_hits") / pl.col("core_n") * 100).sort("pct")
    dim_order = first["dim"].to_list()

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), sharey=True)
    ypos = np.arange(len(dim_order))
    for ax, (pattern, param, title) in zip(axes, panels):
        sub = grid.filter(
            (pl.col("pattern") == pattern) & (pl.col("param") == param)
        )
        by_dim = {r["dim"]: r for r in sub.iter_rows(named=True)}
        pcts = [100 * by_dim[d]["core_hits"] / by_dim[d]["core_n"] for d in dim_order]
        ax.barh(ypos, pcts, color=BLUE, height=0.62)
        for y, d, pct in zip(ypos, dim_order, pcts):
            ax.text(pct + 1.5, y, f'{by_dim[d]["core_hits"]}/{by_dim[d]["core_n"]}',
                    va="center", fontsize=7.5, color=MUTED)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlim(0, 100)
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [DIM_LABELS[d] for d in dim_order])
    fig.suptitle("Pattern prevalence on declared core values — % of core-value trajectories",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIG / "fig3_prevalence_core.png", dpi=160)
    plt.close(fig)


def fig_cliffs(grid: pl.DataFrame):
    """Prevalence sensitivity to parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    ax = axes[0]
    for pattern, color, label in [
        ("sustained (-1 run)", RED, "sustained: −1 run ≥ C"),
        ("rise (+1 run)", BLUE, "rise: +1 run ≥ C"),
    ]:
        sub = (grid.filter(pl.col("pattern") == pattern)
               .group_by("param").agg(
                   pl.col("core_hits").sum(), pl.col("core_n").sum())
               .sort("param"))
        cs = [int(p.split("=")[1]) for p in sub["param"]]
        pct = (sub["core_hits"] / sub["core_n"] * 100).to_list()
        ax.plot(cs, pct, color=color, marker="o", markersize=5, linewidth=2)
        ax.annotate(label, (cs[-1], pct[-1]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8.5, color=color)
    ax.set_xticks([2, 3, 4])
    ax.set_xlabel("C (consecutive entries)")
    ax.set_ylabel("% core trajectories")
    ax.set_xlim(1.8, 4.9)
    ax.set_title("Entry-level run patterns vs C", fontsize=10, loc="left")

    ax = axes[1]
    sub = (grid.filter(pl.col("pattern") == "weekly mean drop")
           .group_by("param").agg(pl.col("core_hits").sum(), pl.col("core_n").sum())
           .sort("param"))
    deltas = [float(p.split("=")[1]) for p in sub["param"]]
    order = np.argsort(deltas)
    deltas = [deltas[i] for i in order]
    pct = (sub["core_hits"] / sub["core_n"] * 100).to_list()
    pct = [pct[i] for i in order]
    ax.plot(deltas, pct, color=AQUA, marker="o", markersize=5, linewidth=2)
    ax.set_ylim(0, max(pct) * 1.18)
    ax.set_xlabel("δ (weekly-mean drop)")
    ax.set_title("Weekly-mean drop vs δ", fontsize=10, loc="left")
    for x, y in zip(deltas, pct):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 7), fontsize=8, color=INK2, ha="center")

    fig.suptitle("Threshold sensitivity (core values, all dims pooled)",
                 fontsize=11, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIG / "fig4_threshold_sensitivity.png", dpi=160)
    plt.close(fig)


def fig_conflict_heavy_weeks(cand: pl.DataFrame):
    """Candidate conflict-heavy-week counts under -1 density definitions."""
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    thresholds = [0.25, 0.5, 0.75, 1.0]
    for is_core, color, label in [(True, RED, "core dims"), (False, MUTED, "non-core dims")]:
        counts = [
            cand.filter(
                (pl.col("is_core") == is_core)
                & (pl.col("n_entries") >= 2)
                & (pl.col("neg_density") >= t)
            ).height
            for t in thresholds
        ]
        ax.plot(thresholds, counts, color=color, marker="o", markersize=5, linewidth=2)
        dy = 7 if is_core is False else -14
        for x, y in zip(thresholds, counts):
            ax.annotate(str(y), (x, y), textcoords="offset points",
                        xytext=(0, dy), fontsize=8.5, color=color, ha="center")
        ax.annotate(label, (thresholds[-1], counts[-1]),
                    textcoords="offset points", xytext=(8, -3 if is_core else 4),
                    fontsize=8.5, color=color)
    ax.set_xticks(thresholds)
    ax.set_xlabel("−1 density threshold (share of week's entries labeled −1)")
    ax.set_ylabel("candidate conflict-heavy weeks")
    ax.set_xlim(0.2, 1.13)
    ax.set_title("Label-derived conflict-heavy weeks (weeks with ≥2 entries)",
                 fontsize=11, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_conflict_heavy_weeks.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------- main

def main():
    args = parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    lab, core, label_path = load(args.labels)
    personas = build_sequences(lab, week_mode=args.week_mode)

    grid = pattern_grid(personas, core)
    grid.write_csv(TAB / "pattern_prevalence_grid.csv")

    coverage = persona_coverage(personas, core)
    coverage.write_csv(TAB / "persona_coverage.csv")

    cand = conflict_heavy_week_candidates(personas, core)
    cand.filter(
        (pl.col("neg_density") >= 0.5) & (pl.col("n_entries") >= 2)
    ).sort("neg_density", descending=True).write_csv(
        TAB / "conflict_heavy_week_candidates.csv")

    impact = impact_comparison(week_mode=args.week_mode)
    impact.write_csv(TAB / "single_definition_impact_comparison.csv")

    spikes = spike_stats(personas, core)
    runs = run_length_stats(personas)

    fig_structure(personas)
    fig_trajectories(personas, core)
    fig_prevalence(grid)
    fig_cliffs(grid)
    fig_conflict_heavy_weeks(cand)

    # ---- stdout summary (consumed for the report) ----
    n_weeks = [d["n_weeks"] for d in personas.values()]
    summary = {
        "label_source": args.labels,
        "label_path": str(label_path.relative_to(ROOT)),
        "week_mode": args.week_mode,
        "n_personas": len(personas),
        "n_entries": int(lab.height),
        "core_declarations": sum(len(v) for v in core.values()),
        "weeks_per_persona": {
            "median": float(np.median(n_weeks)),
            "p25": float(np.percentile(n_weeks, 25)),
            "p75": float(np.percentile(n_weeks, 75)),
        },
        "spike_recovery": {
            k: {"recovered": v[0], "total": v[1],
                "share": round(v[0] / v[1], 3) if v[1] else None}
            for k, v in spikes.items()
        },
        "neg_run_length_dist": runs,
        "persona_coverage": coverage.to_dicts(),
        "single_definition_impact": {
            k: {"personas": v[0], "pct": round(100 * v[0] / v[1], 1)}
            for k, v in single_definition_impact(personas, core).items()
        },
        "single_definition_impact_comparison": impact.to_dicts(),
    }
    print(json.dumps(summary, indent=2))

    # key grid slices for the report
    for pattern, param in [
        ("dip_any (0/+1→-1)", "entry"), ("dip_hard (+1→-1)", "entry"),
        ("sustained (-1 run)", "C=2"), ("sustained (-1 run)", "C=3"),
        ("fade (+1→0s)", "C=2"), ("rise (+1 run)", "C=3"),
        ("all-neutral", "entry"), ("never +1 (n>=3)", "entry"),
        ("weekly mean drop", "δ=1.0"), ("low-mean weeks (mean<τ)", "τ=-0.4,C=2"),
    ]:
        sub = grid.filter((pl.col("pattern") == pattern) & (pl.col("param") == param))
        tot_all = sub["all_hits"].sum(), sub["all_n"].sum()
        tot_core = sub["core_hits"].sum(), sub["core_n"].sum()
        print(f"{pattern:28s} {param:12s} all: {tot_all[0]:4d}/{tot_all[1]} "
              f"({100*tot_all[0]/tot_all[1]:.1f}%)  core: {tot_core[0]:3d}/{tot_core[1]} "
              f"({100*tot_core[0]/tot_core[1]:.1f}%)")


if __name__ == "__main__":
    main()
