"""Generate drift detection notebooks (01-05) for notebooks/annotations/."""

import json
import pathlib

OUT = pathlib.Path("notebooks/annotations")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def notebook(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Shared boilerplate
# ---------------------------------------------------------------------------

SHARED_IMPORTS = """\
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=0.9)
"""

SHARED_DATA = """\
# --- Load judge labels (204 personas, 1651 entries, integer {-1, 0, 1} scores) ---
# Using judge labels rather than human annotations: more data (all 204 personas vs 24),
# and clean integer scores (no averaging artefacts).

VALUE_COLS = [
    "alignment_self_direction", "alignment_stimulation", "alignment_hedonism",
    "alignment_achievement", "alignment_power", "alignment_security",
    "alignment_conformity", "alignment_tradition", "alignment_benevolence",
    "alignment_universalism",
]
SHORT_NAMES = [c.replace("alignment_", "") for c in VALUE_COLS]
DIM_LABELS  = ["SD", "ST", "HE", "AC", "PO", "SE", "CO", "TR", "BE", "UN"]

judge_df = pl.read_parquet("../../logs/judge_labels/judge_labels.parquet")
mean_df  = (
    judge_df
    .select(["persona_id", "t_index"] + VALUE_COLS)
    .with_columns([pl.col(c).cast(pl.Float64) for c in VALUE_COLS])
    .sort(["persona_id", "t_index"])
)

registry   = pl.read_parquet("../../logs/registry/personas.parquet").select(
                ["persona_id", "name", "core_values"])
id_to_name = dict(zip(registry["persona_id"].to_list(), registry["name"].to_list()))
id_to_core = dict(zip(registry["persona_id"].to_list(), registry["core_values"].to_list()))

persona_ids = sorted(mean_df["persona_id"].unique().to_list())
print(f"Loaded {len(judge_df)} judge-labelled entries across {len(persona_ids)} personas")

# --- Sample: verify input scores look correct ---
# Pick first persona with ≥5 entries so the sample is representative
sample_pid = next(
    pid for pid in persona_ids
    if mean_df.filter(pl.col("persona_id") == pid).height >= 5
)
sample_data = mean_df.filter(pl.col("persona_id") == sample_pid).head(5)
print(f"\\nSample scores (first 5 entries) for '{id_to_name.get(sample_pid, sample_pid)}':")
print(f"  Core values: {id_to_core.get(sample_pid, [])}")
print(sample_data.select(["t_index"] + VALUE_COLS))
print("  Scores are integers in {{-1, 0, +1}} across all 10 Schwartz dimensions")


def get_persona_matrix(pid: str) -> tuple[list[int], np.ndarray]:
    pdata = mean_df.filter(pl.col("persona_id") == pid).sort("t_index")
    return pdata["t_index"].to_list(), np.array([pdata[c].to_list() for c in VALUE_COLS]).T


def get_profile_weights(pid: str) -> np.ndarray:
    core = id_to_core.get(pid, [])
    name_to_idx = {s.lower().replace("-", "_"): i for i, s in enumerate(SHORT_NAMES)}
    w = np.zeros(10)
    for v in core:
        key = v.lower().replace("-", "_").replace(" ", "_")
        if key in name_to_idx:
            w[name_to_idx[key]] = 1.0
    if w.sum() > 0:
        w /= w.sum()
    return w


def core_dim_indices(weights: np.ndarray, w_min: float = 0.15) -> list[int]:
    return [j for j, wj in enumerate(weights) if wj >= w_min]
"""

# ---------------------------------------------------------------------------
# 01 — Rule-based
# ---------------------------------------------------------------------------

NB01_CELLS = [
    md("""\
# Drift Detection — 01: Rule-Based (6 Sub-Approaches)

Evaluates six rule-based drift detectors on human-annotated persona data:

| # | Sub-approach | Philosophy |
|---|---|---|
| 0 | **Baseline (dual-trigger)** | Crash: single-step drop > δ. No-recovery: score < τ for ≥ C_min steps |
| 1 | **EMA** | Exponentially weighted running worry score |
| 2 | **CUSUM** | Cumulative sum — marbles in a jar |
| 3 | **Cosine Similarity** | Holistic direction check against value profile |
| 4 | **Control Charts** | Score outside mean ± nσ of a baseline period |
| 5 | **KL Divergence** | Distribution shift between baseline and recent window |

All approaches are rule-based: fixed formulas with tunable parameters, no learning.
The signal taxonomy (crash / fade / spike / no-recovery / onboarding-gap / rise) is specific
to rule-based detection — ML approaches (BOCPD, GP, HMM) subsume these patterns without naming them.

**Doc reference:** `docs/evolution/drift_detection.md § 3.1`
"""),

    code(SHARED_IMPORTS + "\n" + SHARED_DATA),

    md("## Approach 0: Baseline (dual-trigger)\n\nThe simplest possible implementation from `docs/vif/04_uncertainty_logic.md`:\n- **Crash:** single-step drop in profile-weighted alignment > δ\n- **No-recovery:** alignment stays below τ_low for ≥ C_min consecutive steps\n\nServes as the benchmark. If this meets targets (≥80% hit rate, <20% FPR) the more complex sub-approaches are unnecessary."),

    code("""\
def detect_baseline(
    scores: np.ndarray,
    weights: np.ndarray,
    delta: float = 0.5,
    tau_low: float = -0.4,
    c_min: int = 3,
    w_min: float = 0.15,
) -> dict:
    \"\"\"Dual-trigger baseline: crash + no-recovery per core dimension.\"\"\"
    T, K = scores.shape
    alerts = []          # (t, j, kind)  kind = 'crash' | 'no_recovery'
    state = np.zeros(K)  # consecutive low-score counter per dimension

    scalar = scores @ weights  # profile-weighted scalar alignment

    for t in range(1, T):
        drop = scalar[t - 1] - scalar[t]
        if drop > delta:
            alerts.append((t, -1, "crash"))  # j=-1 means holistic

        for j in range(K):
            if weights[j] < w_min:
                state[j] = 0
                continue
            if scores[t, j] < tau_low:
                state[j] += 1
                if state[j] >= c_min:
                    alerts.append((t, j, "no_recovery"))
            else:
                state[j] = 0

    return {"alerts": alerts, "scalar": scalar}
"""),

    md("## Approach 1: EMA (Exponential Moving Average)\n\nA running 'worry score' per dimension. Each week, mix in the new misalignment but let old worry fade:\n`worry = α × new + (1-α) × old`. Alert when worry > threshold.\n\n- Detects **gradual fades** well (accumulates slowly).\n- Weak on isolated crashes (a single bad week only partially fills the score)."),

    code("""\
def detect_ema(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.3,
    threshold: float = 0.10,
    w_min: float = 0.15,
) -> dict:
    T, K = scores.shape
    ema = np.zeros(K)
    ema_history = np.zeros((T, K))
    alerts = []

    for t in range(T):
        for j in range(K):
            if weights[j] < w_min:
                continue
            misalign = max(0.0, -scores[t, j])
            signal = weights[j] * misalign
            ema[j] = alpha * signal + (1 - alpha) * ema[j]
            if ema[j] > threshold:
                alerts.append((t, j))
        ema_history[t] = ema.copy()

    return {"alerts": alerts, "ema_history": ema_history}
"""),

    md("## Approach 2: CUSUM (Cumulative Sum)\n\nEvery bad week adds a marble to a jar. Every okay week removes one (but never below zero). If the jar fills past a threshold, something systemic is happening.\n\n- Detects **sustained small shifts** that individually look harmless.\n- Resets only on explicit drain — memory is longer than EMA."),

    code("""\
def detect_cusum(
    scores: np.ndarray,
    weights: np.ndarray,
    k: float = 0.3,
    h: float = 1.5,
    w_min: float = 0.15,
) -> dict:
    T, K = scores.shape
    jar = np.zeros(K)
    jar_history = np.zeros((T, K))
    alerts = []

    for t in range(T):
        for j in range(K):
            if weights[j] < w_min:
                continue
            misalign = weights[j] * max(0.0, -scores[t, j])
            jar[j] = max(0.0, jar[j] + misalign - k)
            if jar[j] > h:
                alerts.append((t, j))
        jar_history[t] = jar.copy()

    return {"alerts": alerts, "jar_history": jar_history}
"""),

    md("## Approach 3: Cosine Similarity\n\nForget magnitudes — just ask: is the behaviour vector pointing in the same direction as the value profile?\nA negative cosine similarity means behaviour is anti-correlated with declared values.\n\n- Holistic (not per-dimension).\n- Immediate — detects direction reversal in a single step.\n- Weak on duration (can't track whether the reversal persists)."),

    code("""\
def detect_cosine(
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.0,
) -> dict:
    T, K = scores.shape
    cosines = []
    alerts = []

    for t in range(T):
        a = scores[t]
        norm = np.linalg.norm(weights) * np.linalg.norm(a)
        cos = float(np.dot(weights, a) / norm) if norm > 1e-8 else 0.0
        cosines.append(cos)
        if cos < threshold:
            alerts.append(t)

    return {"alerts": alerts, "cosines": cosines}
"""),

    md("## Approach 4: Control Charts\n\nLearn what 'normal' looks like from a baseline period, then flag any score outside the expected range.\nUses mean ± nσ control limits.\n\n- Per-dimension.\n- Strong on **crashes** (immediate LCL breach after stable baseline).\n- Requires a meaningful baseline period (first few steps)."),

    code("""\
def detect_control_chart(
    scores: np.ndarray,
    weights: np.ndarray,
    baseline_end: int = 3,
    n_sigma: float = 2.0,
    w_min: float = 0.15,
) -> dict:
    T, K = scores.shape
    alerts = []
    lcls = np.full(K, np.nan)

    if baseline_end >= T:
        return {"alerts": [], "lcls": lcls}

    baseline = scores[:baseline_end]
    mu   = baseline.mean(axis=0)
    sigma = baseline.std(axis=0)

    for j in range(K):
        if weights[j] >= w_min:
            lcls[j] = mu[j] - n_sigma * sigma[j]

    for t in range(baseline_end, T):
        for j in range(K):
            if weights[j] < w_min or np.isnan(lcls[j]):
                continue
            if scores[t, j] < lcls[j]:
                alerts.append((t, j))

    return {"alerts": alerts, "lcls": lcls, "mu": mu, "sigma": sigma}
"""),

    md("## Approach 5: KL Divergence\n\nCompare the *mix* of scores in a sliding window against the baseline mix. Even if the mean stays similar,\na shift in the distribution (e.g., consistently -1 instead of a mix of +1 and 0) triggers an alert.\n\n- Captures both **level** and **shape** changes.\n- Needs enough data in both baseline and comparison windows.\n- Slower to detect (needs window to fill)."),

    code("""\
def _scores_to_dist(vals: np.ndarray, smoothing: float = 0.05) -> np.ndarray:
    bins = [-1.5, -0.5, 0.5, 1.5]
    counts, _ = np.histogram(vals, bins=bins)
    dist = counts.astype(float) + smoothing
    return dist / dist.sum()


def detect_kl_divergence(
    scores: np.ndarray,
    weights: np.ndarray,
    baseline_end: int = 3,
    window: int = 3,
    kl_threshold: float = 0.15,
    w_min: float = 0.15,
) -> dict:
    from scipy.special import rel_entr
    T, K = scores.shape
    alerts = []
    kl_history = np.zeros((T, K))

    for j in range(K):
        if weights[j] < w_min:
            continue
        baseline_dist = _scores_to_dist(scores[:baseline_end, j])
        for t in range(baseline_end + window, T + 1):
            recent_dist = _scores_to_dist(scores[t - window:t, j])
            kl = float(np.sum(rel_entr(recent_dist, baseline_dist)))
            kl_history[t - 1, j] = kl
            if kl > kl_threshold:
                alerts.append((t - 1, j))

    return {"alerts": alerts, "kl_history": kl_history}
"""),

    md("## Run All 6 Sub-Approaches\n\nFilter to personas with ≥5 steps. Profile weights are derived equally from declared core values."),

    code("""\
MIN_STEPS = 5
APPROACH_KEYS  = ["baseline", "ema", "cusum", "cosine", "control_chart", "kl"]
APPROACH_NAMES = ["Baseline", "EMA", "CUSUM", "Cosine", "Control Chart", "KL Div"]

results = {}

for pid in persona_ids:
    t_idx, matrix = get_persona_matrix(pid)
    T = len(t_idx)
    if T < MIN_STEPS:
        continue

    w = get_profile_weights(pid)
    baseline_end = max(2, min(4, T // 3))

    results[pid] = {
        "name":        id_to_name.get(pid, pid[:8]),
        "core":        id_to_core.get(pid, []),
        "T":           T,
        "t_idx":       t_idx,
        "matrix":      matrix,
        "weights":     w,
        "baseline_end": baseline_end,
        "baseline":    detect_baseline(matrix, w),
        "ema":         detect_ema(matrix, w),
        "cusum":       detect_cusum(matrix, w),
        "cosine":      detect_cosine(matrix, w),
        "control_chart": detect_control_chart(matrix, w, baseline_end=baseline_end),
        "kl":          detect_kl_divergence(matrix, w, baseline_end=baseline_end),
    }

print(f"Ran 6 sub-approaches on {len(results)} personas (≥{MIN_STEPS} steps)\\n")
hdr = f"{'Persona':<25s}  {'Core':<28s}  T  " + "  ".join(f"{n:>5s}" for n in APPROACH_NAMES)
print(hdr)
print("-" * len(hdr))
for pid, r in results.items():
    counts = [len(r[k]["alerts"]) for k in APPROACH_KEYS]
    core_str = ", ".join(r["core"])
    print(f"{r['name']:<25s}  {core_str:<28s}  {r['T']:2d}  " +
          "  ".join(f"{c:>5d}" for c in counts))
"""),

    md("## Consensus Ground Truth\n\nRather than manually labelling crisis points, use **cross-approach agreement** as a proxy for ground truth.\nIf ≥4 of 6 sub-approaches independently flag the same `(persona, t_index)`, treat it as a high-confidence crisis.\n\n```\nconsensus_score = number of approaches flagging (pid, t)   # 0–6\nstrong  → score ≥ 4   (majority agreement)\nweak    → score ∈ {2, 3}\nnone    → score ≤ 1\n```\n\nThis produces labels without manual work and covers all personas with ≥5 steps."),

    code("""\
def get_alert_steps(r: dict, key: str) -> set[int]:
    \"\"\"Return the set of t-steps (0-indexed within persona) where any alert fired.\"\"\"
    alerts = r[key]["alerts"]
    if not alerts:
        return set()
    if isinstance(alerts[0], tuple):
        return {a[0] for a in alerts}
    return set(alerts)   # cosine: bare ints


consensus = {}   # pid -> {t: score}

for pid, r in results.items():
    T = r["T"]
    step_votes = {t: 0 for t in range(T)}
    for key in APPROACH_KEYS:
        for t in get_alert_steps(r, key):
            if t in step_votes:
                step_votes[t] += 1
    consensus[pid] = step_votes

# Classify each step
def classify(score: int) -> str:
    if score >= 4: return "strong"
    if score >= 2: return "weak"
    return "none"

# Summary
strong_total = sum(1 for pid, sv in consensus.items()
                   for score in sv.values() if score >= 4)
weak_total   = sum(1 for pid, sv in consensus.items()
                   for score in sv.values() if 2 <= score < 4)

print(f"Consensus summary across {len(consensus)} personas:")
print(f"  Strong crisis steps (≥4 agree): {strong_total}")
print(f"  Weak crisis steps   (2-3 agree): {weak_total}")
print()
for pid, sv in consensus.items():
    strong_steps = [t for t, s in sv.items() if s >= 4]
    if strong_steps:
        name = results[pid]["name"]
        print(f"  {name:<25s} strong at t={strong_steps}")
"""),

    md("## Metrics: Score Each Sub-Approach Against Consensus\n\nFor each sub-approach, compute hit rate, precision, F1, FPR, and first-alert latency against the consensus strong-crisis labels.\n\n| Metric | Target |\n|---|---|\n| Hit Rate | ≥ 80% |\n| Precision | > 60% |\n| F1 | > 0.5 |\n| FPR | < 20% |\n| First-alert latency | ≤ 2 steps |"),

    code("""\
from dataclasses import dataclass

@dataclass
class Metrics:
    hit_rate:  float
    precision: float
    f1:        float
    fpr:       float
    latency:   float   # mean steps after first strong-crisis step


def score_approach(key: str, threshold: int = 4) -> Metrics:
    tp = fp = fn = tn = 0
    latencies = []

    for pid, r in results.items():
        sv = consensus[pid]
        crisis_steps = {t for t, s in sv.items() if s >= threshold}
        non_crisis   = {t for t, s in sv.items() if s < threshold}

        alerted = get_alert_steps(r, key)

        tp += len(alerted & crisis_steps)
        fp += len(alerted & non_crisis)
        fn += len(crisis_steps - alerted)
        tn += len(non_crisis - alerted)

        # Latency: steps between first crisis step and first alert (if any)
        if crisis_steps and alerted:
            first_crisis = min(crisis_steps)
            first_alert  = min((t for t in alerted if t >= first_crisis), default=None)
            if first_alert is not None:
                latencies.append(first_alert - first_crisis)

    hit_rate  = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    f1        = 2 * hit_rate * precision / max(hit_rate + precision, 1e-9)
    fpr       = fp / max(fp + tn, 1)
    latency   = float(np.mean(latencies)) if latencies else float("nan")
    return Metrics(hit_rate, precision, f1, fpr, latency)


print(f"{'Approach':<15s}  {'Hit%':>6s}  {'Prec%':>6s}  {'F1':>6s}  {'FPR%':>6s}  {'Lat':>5s}")
print("-" * 58)
metrics = {}
for key, name in zip(APPROACH_KEYS, APPROACH_NAMES):
    m = score_approach(key)
    metrics[key] = m
    hit_ok  = "✓" if m.hit_rate  >= 0.80 else " "
    prec_ok = "✓" if m.precision >= 0.60 else " "
    fpr_ok  = "✓" if m.fpr       <= 0.20 else " "
    print(f"{name:<15s}  {m.hit_rate*100:>5.1f}{hit_ok}  "
          f"{m.precision*100:>5.1f}{prec_ok}  "
          f"{m.f1:>6.3f}  "
          f"{m.fpr*100:>5.1f}{fpr_ok}  "
          f"{m.latency:>5.1f}")
"""),

    md("## Per-Persona Alert Dashboard\n\nFor each persona: alignment scores on core dimensions across time, with alert markers per sub-approach."),

    code("""\
palette = sns.color_palette("tab10", n_colors=10)
APPROACH_COLORS = ["#2c3e50", "#3498db", "#e74c3c", "#9b59b6", "#2ecc71", "#f39c12"]


def plot_persona(pid: str):
    r = results[pid]
    matrix  = r["matrix"]
    weights = r["weights"]
    T       = r["T"]
    core_j  = core_dim_indices(weights)
    if not core_j:
        return

    fig, axes = plt.subplots(len(core_j), 1, figsize=(12, 2.5 * len(core_j)), sharex=True)
    if len(core_j) == 1:
        axes = [axes]

    for ax, j in zip(axes, core_j):
        ax.plot(range(T), matrix[:, j], "o-", color=palette[j], lw=1.8,
                label=DIM_LABELS[j], zorder=3)
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_ylim(-1.4, 1.4)
        ax.set_ylabel(DIM_LABELS[j], fontsize=9)

        # Mark consensus crisis steps
        sv = consensus[pid]
        for t, score in sv.items():
            if score >= 4:
                ax.axvspan(t - 0.4, t + 0.4, color="red", alpha=0.15, zorder=1)
            elif score >= 2:
                ax.axvspan(t - 0.4, t + 0.4, color="orange", alpha=0.10, zorder=1)

        # Mark alerts per approach (offset markers)
        for idx, (key, name, col) in enumerate(zip(APPROACH_KEYS, APPROACH_NAMES, APPROACH_COLORS)):
            alerted = get_alert_steps(r, key)
            alerted_j = alerted  # cosine is holistic; others may be per-dim
            if isinstance(r[key]["alerts"], list) and r[key]["alerts"] and isinstance(r[key]["alerts"][0], tuple):
                alerted_j = {a[0] for a in r[key]["alerts"] if a[1] in (j, -1)}
            for t in alerted_j:
                ax.plot(t, -1.1 - idx * 0.05, "|", color=col,
                        ms=8, mew=1.5, label=name if t == min(alerted_j) else "")

    axes[0].set_title(f"{r['name']} — core: {', '.join(r['core'])}", fontweight="bold")
    axes[-1].set_xlabel("t_index")

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, marker="|", ms=8, mew=1.5, ls="none", label=n)
               for c, n in zip(APPROACH_COLORS, APPROACH_NAMES)]
    from matplotlib.patches import Patch
    handles += [Patch(color="red", alpha=0.3, label="strong consensus"),
                Patch(color="orange", alpha=0.2, label="weak consensus")]
    axes[0].legend(handles=handles, fontsize=7, ncol=4, loc="upper right")

    plt.tight_layout()
    plt.show()


for pid in list(results.keys())[:6]:   # show first 6
    plot_persona(pid)
"""),

    md("## Agreement Heatmap\n\nFor each persona, what fraction of monitored steps did each sub-approach alert on?"),

    code("""\
def alert_rate(r: dict, key: str) -> float:
    T = r["T"]
    alerted = get_alert_steps(r, key)
    return len(alerted) / max(T, 1)


persona_labels = [f"{r['name']} ({','.join(r['core'])})" for r in results.values()]
rate_matrix = np.array([
    [alert_rate(r, k) for k in APPROACH_KEYS]
    for r in results.values()
])

fig, ax = plt.subplots(figsize=(10, max(5, len(results) * 0.45)))
sns.heatmap(rate_matrix, ax=ax,
            xticklabels=APPROACH_NAMES, yticklabels=persona_labels,
            cmap="YlOrRd", vmin=0, vmax=1, annot=True, fmt=".0%",
            linewidths=0.5, linecolor="white")
ax.set_title("Alert Rate by Sub-Approach (fraction of steps alerting)", fontweight="bold")
plt.tight_layout()
plt.show()
"""),

    md("## Summary\n\n| Sub-approach | Detects Rut? | Detects Crash? | Latency | Memory | Per-dim? |\n|---|:---:|:---:|:---:|:---:|:---:|\n| **Baseline** | ✓ (no-recovery counter) | ✓ (single-step drop) | Immediate | None | Crash: no; Rut: yes |\n| **EMA** | ✓ (strong) | Weak (fades) | 2-3 steps | Exponential decay | Yes |\n| **CUSUM** | ✓ (strong) | Weak (needs accumulation) | 2-3 steps | Indefinite (until reset) | Yes |\n| **Cosine** | Weak (no duration) | ✓ (instant) | Immediate | None | No (holistic) |\n| **Control Chart** | Medium (run rule) | ✓ (LCL breach) | Immediate after baseline | Fixed baseline | Yes |\n| **KL Div** | ✓ (distribution shift) | Weak (needs window) | Slow | Sliding window | Yes |\n\nSee `docs/evolution/drift_detection.md § 3.1` for full parameter grid and selection rationale."),
]


# ---------------------------------------------------------------------------
# 02 — BOCPD
# ---------------------------------------------------------------------------

NB02_CELLS = [
    md("""\
# Drift Detection — 02: Bayesian Online Changepoint Detection (BOCPD)

Instead of hand-coded signal patterns, BOCPD models the **posterior probability that a changepoint
occurred at each time step**. It maintains a "run length" distribution — how long the current
regime has been active — and updates it with each new observation.

The signal taxonomy (crash / fade / no-recovery) is **not an input**. These patterns emerge
from the posterior shape — a crash produces a sudden spike in P(changepoint); a fade produces
a gradual rise over multiple steps.

**Doc reference:** `docs/evolution/drift_detection.md § 3.2`

### Implementation

Uses a **Dirichlet-Categorical conjugate prior** for {-1, 0, +1} alignment scores.
- Prior: Dirichlet(α₀) — uninformative (α₀ = [1, 1, 1])
- Likelihood: Categorical(θ) — probability of each score value in the current regime
- Hazard rate: constant H (probability of changepoint at each step)
- At each step: update run-length distribution using Bayes' theorem

### Data note

With ~10-15 steps per persona per dimension, posterior estimates will be wide.
BOCPD is most powerful at ~50+ personas with confirmed changepoints. This notebook
validates the implementation and shows the qualitative behaviour on current data.
"""),

    code(SHARED_IMPORTS + "\n" + SHARED_DATA),

    md("## BOCPD Implementation\n\nDirichlet-Categorical conjugate update for ordinal {-1, 0, +1} scores.\nThe run-length posterior `R[t]` is a distribution over how many steps the current regime has lasted."),

    code("""\
def bocpd_dimension(
    scores_1d: np.ndarray,
    hazard: float = 0.1,
    alpha0: np.ndarray | None = None,
) -> dict:
    \"\"\"
    Bayesian Online Changepoint Detection with Dirichlet-Categorical likelihood.

    Correct formulation (Fearnhead & Liu, 2007):
      P(r_t=0, x_{1:t}) = H * p(x_t | alpha0) * sum_r P(r_{t-1}=r, x_{1:t-1})

    The new-run predictive uses alpha0 (fresh regime prior), NOT the old run's
    posterior — that was the bug in the earlier version.

    Parameters
    ----------
    scores_1d : (T,) array of alignment scores — integers {-1, 0, +1}
    hazard    : prior probability of changepoint at each step (H)
    alpha0    : Dirichlet prior pseudo-counts for bins [-1, 0, +1]. Default [1,1,1].

    Returns
    -------
    dict with key:
        p_change : (T,) — P(changepoint at t | data_{1:t})
    \"\"\"
    if alpha0 is None:
        alpha0 = np.array([1.0, 1.0, 1.0])

    T = len(scores_1d)

    def score_to_bin(s: float) -> int:
        if s < -0.5: return 0
        if s <  0.5: return 1
        return 2

    # Each particle: (alpha posterior for this run)
    # log_weights[i] = log P(run i, x_{1:t-1}) — normalised after each step
    alphas      = [alpha0.copy()]
    log_weights = np.array([0.0])   # starts normalised (single run, weight 1)

    p_change = np.zeros(T)

    log_p_new = np.log(alpha0[0] / alpha0.sum())  # placeholder, recomputed each step

    for t in range(T):
        obs_bin = score_to_bin(scores_1d[t])

        # Predictive log-prob under each existing run
        log_preds = np.array([np.log(a[obs_bin] / a.sum()) for a in alphas])

        # --- Changepoint term ---
        # P(r_t=0, x_{1:t}) = H * p(x_t | alpha0) * sum_r exp(log_weights[r])
        # Since weights are normalised, sum = 1 → log_sum = 0
        log_new = np.log(hazard) + np.log(alpha0[obs_bin] / alpha0.sum())

        # --- Continue terms ---
        log_continue = log_weights + log_preds + np.log(1 - hazard)

        # --- Combine and normalise ---
        all_lw = np.append(log_continue, log_new)
        max_lw = all_lw.max()
        w_unnorm = np.exp(all_lw - max_lw)
        w_norm   = w_unnorm / w_unnorm.sum()

        p_change[t] = w_norm[-1]   # weight of the new run = P(changepoint at t)

        # --- Update alphas and weights ---
        new_alphas = []
        for i, a in enumerate(alphas):
            upd = a.copy(); upd[obs_bin] += 1
            new_alphas.append(upd)
        # New run has already observed x_t
        new_run_alpha = alpha0.copy(); new_run_alpha[obs_bin] += 1
        new_alphas.append(new_run_alpha)

        alphas      = new_alphas
        log_weights = np.log(np.maximum(w_norm, 1e-300))

    return {"p_change": p_change}
"""),

    md("""\
## Why BOCPD Has No Training Phase

BOCPD is a **fully online** algorithm: it updates its beliefs with each new observation and
requires no separate training step. This is a fundamental design difference from supervised
approaches (like the Critic) or batch methods (like the Autoencoder).

What BOCPD does instead of training:

| Rule-based equivalent | BOCPD equivalent |
|---|---|
| Grid search over δ, τ_low, C_min | Grid search over **H** (hazard rate) and **α₀** (Dirichlet prior) |
| Consensus hit-rate tuning | Consensus hit-rate tuning — same metric, different parameters |
| Threshold picked once, applied online | H picked once, applied online |

**H (hazard rate):** prior probability of a changepoint at each step.
- Low H (e.g. 0.05) → conservative: needs strong evidence before flagging.
- High H (e.g. 0.40) → sensitive: fires frequently, higher false-positive rate.

**α₀ (Dirichlet prior):** initial belief about score distribution within a regime.
- Uninformative (α₀ = [1, 1, 1]) → equal prior over {-1, 0, +1}.
- Informative (e.g. α₀ = [1, 2, 4]) → prior belief that aligned scores are more likely.

The cell below selects H by grid search against consensus labels derived from
the 6 rule-based sub-approaches (same method as notebook 01).
"""),

    code("""\
# --- Inline consensus from rule-based approaches (mirrors notebook 01) ---
# We need a ground-truth proxy to tune H. Recompute it here using the same 6 sub-approaches.

def _ema_alerts(matrix, w, alpha=0.3, threshold=0.10, w_min=0.15):
    T, K = matrix.shape
    ema = np.zeros(K)
    alerts = set()
    for t in range(T):
        for j in range(K):
            if w[j] < w_min: continue
            ema[j] = alpha * w[j] * max(0., -matrix[t, j]) + (1 - alpha) * ema[j]
            if ema[j] > threshold: alerts.add(t)
    return alerts

def _cusum_alerts(matrix, w, k=0.3, h=1.5, w_min=0.15):
    T, K = matrix.shape
    jar = np.zeros(K)
    alerts = set()
    for t in range(T):
        for j in range(K):
            if w[j] < w_min: continue
            jar[j] = max(0., jar[j] + w[j] * max(0., -matrix[t, j]) - k)
            if jar[j] > h: alerts.add(t)
    return alerts

def _cosine_alerts(matrix, w, threshold=0.0):
    alerts = set()
    for t in range(len(matrix)):
        a = matrix[t]; norm = np.linalg.norm(w) * np.linalg.norm(a)
        if norm > 1e-8 and np.dot(w, a) / norm < threshold: alerts.add(t)
    return alerts

def _cc_alerts(matrix, w, baseline_end=3, n_sigma=2.0, w_min=0.15):
    T, K = matrix.shape
    if baseline_end >= T: return set()
    mu = matrix[:baseline_end].mean(0); sig = matrix[:baseline_end].std(0)
    alerts = set()
    for t in range(baseline_end, T):
        for j in range(K):
            if w[j] >= w_min and matrix[t, j] < mu[j] - n_sigma * sig[j]:
                alerts.add(t)
    return alerts

def _kl_alerts(matrix, w, baseline_end=3, window=3, kl_thresh=0.15, w_min=0.15):
    from scipy.special import rel_entr
    T, K = matrix.shape
    bins = [-1.5, -0.5, 0.5, 1.5]
    def dist(vals):
        c, _ = np.histogram(vals, bins=bins); d = c.astype(float) + 0.05
        return d / d.sum()
    alerts = set()
    for j in range(K):
        if w[j] < w_min: continue
        bd = dist(matrix[:baseline_end, j])
        for t in range(baseline_end + window, T + 1):
            if np.sum(rel_entr(dist(matrix[t-window:t, j]), bd)) > kl_thresh:
                alerts.add(t - 1)
    return alerts

def _baseline_alerts(matrix, w, delta=0.5, tau=-0.4, c_min=3, w_min=0.15):
    T, K = matrix.shape
    scalar = matrix @ w; state = np.zeros(K); alerts = set()
    for t in range(1, T):
        if scalar[t-1] - scalar[t] > delta: alerts.add(t)
        for j in range(K):
            if w[j] < w_min: state[j] = 0; continue
            if matrix[t, j] < tau: state[j] += 1;
            else: state[j] = 0
            if state[j] >= c_min: alerts.add(t)
    return alerts

def compute_consensus(pid):
    t_idx, matrix = get_persona_matrix(pid)
    T = len(t_idx)
    if T < 5: return {}
    w = get_profile_weights(pid)
    be = max(2, min(4, T // 3))
    votes = {t: 0 for t in range(T)}
    for fn, kw in [
        (_baseline_alerts, {}), (_ema_alerts, {}), (_cusum_alerts, {}),
        (_cosine_alerts, {}), (_cc_alerts, {"baseline_end": be}),
        (_kl_alerts, {"baseline_end": be}),
    ]:
        for t in fn(matrix, w, **kw):
            if t in votes: votes[t] += 1
    return votes


# --- Grid search over H and cp_thresh ---
print("Grid searching H × cp_thresh against consensus labels...\\n")

hazard_grid  = [0.05, 0.10, 0.15, 0.25, 0.40]
thresh_grid  = [0.30, 0.40, 0.50, 0.60]
w_min_gs     = 0.15

# Cache BOCPD outputs per (pid, j, H) to avoid recomputation
from functools import lru_cache

bocpd_cache = {}
for H in hazard_grid:
    for pid in persona_ids:
        t_idx, matrix = get_persona_matrix(pid)
        if len(t_idx) < 5: continue
        w = get_profile_weights(pid)
        for j in core_dim_indices(w, w_min_gs):
            bocpd_cache[(pid, j, H)] = bocpd_dimension(matrix[:, j], hazard=H)["p_change"]

grid_results = []
for H in hazard_grid:
    for cp_thresh in thresh_grid:
        tp = fp = fn = tn = 0
        for pid in persona_ids:
            t_idx, matrix = get_persona_matrix(pid)
            T = len(t_idx)
            if T < 5: continue
            w  = get_profile_weights(pid)
            sv = compute_consensus(pid)
            crisis     = {t for t, s in sv.items() if s >= 4}
            non_crisis = {t for t, s in sv.items() if s < 4}

            alerted = set()
            for j in core_dim_indices(w, w_min_gs):
                p_change = bocpd_cache.get((pid, j, H), [])
                for t, pc in enumerate(p_change):
                    if pc > cp_thresh: alerted.add(t)

            tp += len(alerted & crisis); fp += len(alerted & non_crisis)
            fn += len(crisis - alerted); tn += len(non_crisis - alerted)

        hit  = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        f1   = 2 * hit * prec / max(hit + prec, 1e-9)
        fpr  = fp / max(fp + tn, 1)
        grid_results.append({"H": H, "cp_thresh": cp_thresh,
                              "hit": hit, "prec": prec, "f1": f1, "fpr": fpr})

print(f"{'H':>5s}  {'Thresh':>6s}  {'Hit%':>5s}  {'Prec%':>5s}  {'F1':>6s}  {'FPR%':>5s}")
print("-" * 50)
best = max(grid_results, key=lambda r: r["f1"])
for r in grid_results:
    marker = " ←" if r == best else ""
    print(f"{r['H']:>5.2f}  {r['cp_thresh']:>6.2f}  {r['hit']*100:>4.1f}%  "
          f"{r['prec']*100:>4.1f}%  {r['f1']:>6.3f}  {r['fpr']*100:>4.1f}%{marker}")

HAZARD    = best["H"]
CP_THRESH = best["cp_thresh"]
print(f"\\nSelected H={HAZARD}, cp_thresh={CP_THRESH} (best F1={best['f1']:.3f})")
"""),

    md("## Run BOCPD on All Personas\n\nUsing the selected hazard rate H from grid search above.\nFor each persona × core dimension, compute P(changepoint at t) across all time steps."),

    code("""\
# HAZARD and CP_THRESH are set by grid search above
W_MIN = 0.15

bocpd_results = {}

for pid in persona_ids:
    t_idx, matrix = get_persona_matrix(pid)
    T = len(t_idx)
    if T < 5:
        continue

    w = get_profile_weights(pid)
    core_j = core_dim_indices(w, W_MIN)

    dim_results = {}
    alerts = []

    for j in core_j:
        out = bocpd_dimension(matrix[:, j], hazard=HAZARD)
        dim_results[j] = out
        for t, pc in enumerate(out["p_change"]):
            if pc > CP_THRESH:
                alerts.append((t, j))

    bocpd_results[pid] = {
        "name":       id_to_name.get(pid, pid[:8]),
        "core":       id_to_core.get(pid, []),
        "T":          T,
        "t_idx":      t_idx,
        "matrix":     matrix,
        "weights":    w,
        "dim_results": dim_results,
        "alerts":     alerts,
    }

print(f"BOCPD ran on {len(bocpd_results)} personas\\n")
print(f"{'Persona':<25s}  {'Core':<28s}  T  Alerts")
print("-" * 70)
for pid, r in bocpd_results.items():
    print(f"{r['name']:<25s}  {', '.join(r['core']):<28s}  {r['T']:2d}  {len(r['alerts'])}")
"""),

    md("## Visualise P(changepoint) Over Time\n\nFor each persona, show alignment scores on core dimensions alongside P(changepoint).\nRed shading = P(changepoint) > threshold."),

    code("""\
palette = sns.color_palette("tab10", n_colors=10)


def plot_bocpd_persona(pid: str):
    r = bocpd_results[pid]
    matrix = r["matrix"]
    T = r["T"]
    core_j = [j for j in r["dim_results"]]
    if not core_j:
        return

    fig, axes = plt.subplots(len(core_j), 2, figsize=(14, 2.5 * len(core_j)),
                             sharex=True, gridspec_kw={"width_ratios": [2, 1]})
    if len(core_j) == 1:
        axes = [axes]

    for ax_row, j in zip(axes, core_j):
        ax_score, ax_cp = ax_row
        p_change = r["dim_results"][j]["p_change"]

        # Score plot
        ax_score.plot(range(T), matrix[:, j], "o-", color=palette[j], lw=1.8)
        ax_score.axhline(0, color="grey", lw=0.5, ls="--")
        ax_score.set_ylim(-1.4, 1.4)
        ax_score.set_ylabel(DIM_LABELS[j], fontsize=9)

        # Shade high P(changepoint) regions
        for t, pc in enumerate(p_change):
            if pc > CP_THRESH:
                ax_score.axvspan(t - 0.4, t + 0.4, color="red", alpha=pc * 0.4, zorder=1)

        # P(changepoint) plot
        ax_cp.bar(range(T), p_change, color="salmon", alpha=0.8)
        ax_cp.axhline(CP_THRESH, color="red", lw=1, ls="--")
        ax_cp.set_ylim(0, 1)
        ax_cp.set_ylabel("P(change)", fontsize=8)
        ax_cp.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    axes[0][0].set_title(f"{r['name']} — {', '.join(r['core'])}", fontweight="bold")
    axes[-1][0].set_xlabel("t_index")
    axes[-1][1].set_xlabel("t_index")
    plt.tight_layout()
    plt.show()


for pid in list(bocpd_results.keys())[:4]:
    plot_bocpd_persona(pid)
"""),

    md("## Sensitivity Analysis: Hazard Rate\n\nThe hazard rate H controls how often the model expects changepoints a priori.\nLow H = conservative (needs strong evidence). High H = sensitive (fires more often)."),

    code("""\
hazard_values = [0.05, 0.10, 0.15, 0.25, 0.40]

# Pick first persona with ≥6 steps and ≥1 core dimension
example_pid = next(
    pid for pid, r in bocpd_results.items()
    if r["T"] >= 6 and r["dim_results"]
)
example_r = bocpd_results[example_pid]
example_j = list(example_r["dim_results"].keys())[0]
scores_1d  = example_r["matrix"][:, example_j]
T          = example_r["T"]

fig, axes = plt.subplots(len(hazard_values), 1,
                         figsize=(12, 2.2 * len(hazard_values)), sharex=True)

for ax, H in zip(axes, hazard_values):
    out = bocpd_dimension(scores_1d, hazard=H)
    ax.bar(range(T), out["p_change"], color="salmon", alpha=0.8, label=f"H={H}")
    ax.axhline(CP_THRESH, color="red", lw=1, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"H={H}", fontsize=9)
    ax2 = ax.twinx()
    ax2.plot(range(T), scores_1d, "o-", color="steelblue", ms=4, lw=1.2)
    ax2.set_ylim(-1.4, 1.4)
    ax2.set_ylabel("score", fontsize=8)

axes[0].set_title(
    f"{example_r['name']} — {DIM_LABELS[example_j]}: P(changepoint) vs hazard rate",
    fontweight="bold")
axes[-1].set_xlabel("t_index")
plt.tight_layout()
plt.show()
"""),

    md("## Limitations\n\n- **Small data:** ~10-15 steps per persona per dimension. Posterior is prior-dominated early on.\n- **Discrete scores:** {-1, 0, +1} means regime estimates are noisy on short windows.\n- **No profile weights `w_u` natively:** BOCPD operates per-dimension independently. Profile weighting requires a wrapper (e.g., suppress alerts on low-weight dimensions).\n- **Interpretability:** `P(changepoint) = 0.87` needs translation to user-facing language for the Coach.\n\nAt ~50+ personas with confirmed changepoints, BOCPD becomes the recommended upgrade from rule-based (see `docs/evolution/drift_detection.md § 3.2`)."),
]


# ---------------------------------------------------------------------------
# 03 — GP regression
# ---------------------------------------------------------------------------

NB03_CELLS = [
    md("""\
# Drift Detection — 03: Gaussian Process Regression

Models each dimension's alignment trajectory as a **Gaussian Process** — a distribution over
smooth functions. Drift = observation falling outside the GP's predictive interval.

The signal taxonomy is **not an input**. Anomalies are detected purely by deviation from the
learned trajectory, without naming the pattern (crash, fade, etc.).

**Doc reference:** `docs/evolution/drift_detection.md § 3.3`

### Key properties

- **Uncertainty-aware natively:** The predictive interval widens during gaps and narrows where data is dense.
- **Handles irregular spacing:** GPs work with arbitrary input spacing — sporadic journaling doesn't break the model.
- **Con:** With ~10-15 data points per dimension, the posterior may be wide (uncertainty bands too generous).
- **Con:** Requires choosing a kernel. Matérn-3/2 is used here (allows non-smooth functions).
"""),

    code(SHARED_IMPORTS + "\nfrom sklearn.gaussian_process import GaussianProcessRegressor\nfrom sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel\n\n" + SHARED_DATA),

    md("## GP Implementation\n\nFor each persona × core dimension:\n1. Fit a GP to (t_index, score) pairs.\n2. Predict mean and std at each observed step.\n3. Flag steps where the observed score falls below `predicted_mean - n_sigma * std`."),

    code("""\
def gp_dimension(
    t_indices: list[int],
    scores_1d: np.ndarray,
    n_sigma: float = 1.5,
    min_train: int = 3,
) -> dict:
    \"\"\"
    Fit a GP to alignment scores and flag anomalies.

    Parameters
    ----------
    t_indices  : step indices (used as X)
    scores_1d  : (T,) alignment scores
    n_sigma    : alert when score < mean - n_sigma * std
    min_train  : minimum points before making predictions

    Returns
    -------
    dict with keys:
        mean      : (T,) predicted mean
        std       : (T,) predicted std
        lower     : (T,) lower confidence bound (mean - n_sigma * std)
        alerts    : list of t-indices where score < lower bound
    \"\"\"
    T = len(scores_1d)
    X = np.array(t_indices).reshape(-1, 1).astype(float)
    y = scores_1d.astype(float)

    kernel = ConstantKernel(1.0) * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3,
                                   normalize_y=True, random_state=42)

    mean   = np.full(T, np.nan)
    std    = np.full(T, np.nan)
    lower  = np.full(T, np.nan)
    alerts = []

    for t in range(min_train, T):
        # Train on all points up to (but not including) t, then predict at t
        X_train, y_train = X[:t], y[:t]
        X_pred  = X[t:t+1]

        try:
            gpr.fit(X_train, y_train)
            mu, sigma = gpr.predict(X_pred, return_std=True)
            mean[t]  = mu[0]
            std[t]   = sigma[0]
            lower[t] = mu[0] - n_sigma * sigma[0]
            if y[t] < lower[t]:
                alerts.append(t)
        except Exception:
            pass

    return {"mean": mean, "std": std, "lower": lower, "alerts": alerts}
"""),

    md("## Run GP on All Personas"),

    code("""\
N_SIGMA = 1.5
W_MIN   = 0.15

gp_results = {}

for pid in persona_ids:
    t_idx, matrix = get_persona_matrix(pid)
    T = len(t_idx)
    if T < 5:
        continue

    w = get_profile_weights(pid)
    core_j = core_dim_indices(w, W_MIN)

    dim_results = {}
    alerts = []

    for j in core_j:
        out = gp_dimension(t_idx, matrix[:, j], n_sigma=N_SIGMA)
        dim_results[j] = out
        for t in out["alerts"]:
            alerts.append((t, j))

    gp_results[pid] = {
        "name":        id_to_name.get(pid, pid[:8]),
        "core":        id_to_core.get(pid, []),
        "T":           T,
        "t_idx":       t_idx,
        "matrix":      matrix,
        "weights":     w,
        "dim_results": dim_results,
        "alerts":      alerts,
    }

print(f"GP ran on {len(gp_results)} personas\\n")
print(f"{'Persona':<25s}  {'Core':<28s}  T  Alerts")
print("-" * 70)
for pid, r in gp_results.items():
    print(f"{r['name']:<25s}  {', '.join(r['core']):<28s}  {r['T']:2d}  {len(r['alerts'])}")
"""),

    md("## Visualise GP Fit\n\nFor each persona: alignment scores with GP predictive mean, confidence band, and flagged anomalies."),

    code("""\
palette = sns.color_palette("tab10", n_colors=10)


def plot_gp_persona(pid: str):
    r = gp_results[pid]
    matrix = r["matrix"]
    T = r["T"]
    core_j = list(r["dim_results"].keys())
    if not core_j:
        return

    fig, axes = plt.subplots(len(core_j), 1, figsize=(12, 2.8 * len(core_j)), sharex=True)
    if len(core_j) == 1:
        axes = [axes]

    xs = list(range(T))

    for ax, j in zip(axes, core_j):
        out = r["dim_results"][j]
        scores = matrix[:, j]
        color = palette[j]

        ax.plot(xs, scores, "o-", color=color, lw=1.8, zorder=4, label="Observed")

        valid = ~np.isnan(out["mean"])
        if valid.any():
            ax.plot(np.where(valid)[0], out["mean"][valid], "--", color="grey",
                    lw=1.2, label="GP mean")
            ax.fill_between(np.where(valid)[0],
                            out["lower"][valid],
                            out["mean"][valid] + (out["mean"][valid] - out["lower"][valid]),
                            color="grey", alpha=0.2, label=f"±{N_SIGMA}σ")

        for t in out["alerts"]:
            ax.scatter([t], [scores[t]], color="red", s=80, zorder=5, label="Alert" if t == out["alerts"][0] else "")
            ax.axvline(t, color="red", lw=0.8, alpha=0.4)

        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.set_ylim(-1.6, 1.6)
        ax.set_ylabel(DIM_LABELS[j], fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

    axes[0].set_title(f"{r['name']} — {', '.join(r['core'])}", fontweight="bold")
    axes[-1].set_xlabel("t_index")
    plt.tight_layout()
    plt.show()


for pid in list(gp_results.keys())[:4]:
    plot_gp_persona(pid)
"""),

    md("## Kernel Sensitivity\n\nCompare Matérn-1/2, Matérn-3/2, and RBF kernels on an example persona."),

    code("""\
from sklearn.gaussian_process.kernels import RBF

example_pid = next(pid for pid, r in gp_results.items() if r["T"] >= 6 and r["dim_results"])
example_r   = gp_results[example_pid]
example_j   = list(example_r["dim_results"].keys())[0]
t_idx_ex    = example_r["t_idx"]
scores_ex   = example_r["matrix"][:, example_j]

kernels = {
    "Matérn-1/2": ConstantKernel(1.0) * Matern(length_scale=2.0, nu=0.5) + WhiteKernel(0.1),
    "Matérn-3/2": ConstantKernel(1.0) * Matern(length_scale=2.0, nu=1.5) + WhiteKernel(0.1),
    "RBF":        ConstantKernel(1.0) * RBF(length_scale=2.0)             + WhiteKernel(0.1),
}

fig, axes = plt.subplots(len(kernels), 1, figsize=(12, 2.8 * len(kernels)), sharex=True)

for ax, (kname, kern) in zip(axes, kernels.items()):
    out = gp_dimension(t_idx_ex, scores_ex, n_sigma=N_SIGMA)
    T = len(scores_ex)
    xs = list(range(T))
    valid = ~np.isnan(out["mean"])

    ax.plot(xs, scores_ex, "o-", color="steelblue", lw=1.8)
    if valid.any():
        ax.plot(np.where(valid)[0], out["mean"][valid], "--", color="grey", lw=1.2)
        ax.fill_between(np.where(valid)[0],
                        out["lower"][valid],
                        out["mean"][valid] + (out["mean"][valid] - out["lower"][valid]),
                        color="grey", alpha=0.2)
    for t in out["alerts"]:
        ax.axvline(t, color="red", lw=0.8, alpha=0.6)
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.set_ylim(-1.6, 1.6)
    ax.set_ylabel(kname, fontsize=9)

axes[0].set_title(f"{example_r['name']} — {DIM_LABELS[example_j]}: kernel comparison",
                  fontweight="bold")
axes[-1].set_xlabel("t_index")
plt.tight_layout()
plt.show()
"""),

    md("## Limitations\n\n- **Wide posteriors:** With ≤15 data points per dimension, confidence bands are often too generous — few anomalies detected.\n- **Kernel choice matters:** Matérn-3/2 is a reasonable default for non-smooth value trajectories, but the optimal kernel is not obvious.\n- **Discrete score space:** GP assumes continuous observations. {-1, 0, +1} scores violate this assumption; results are approximate.\n- **No profile weights `w_u` natively:** GP operates per-dimension. Must suppress alerts on low-weight dimensions externally.\n\nGP regression is a lateral upgrade — most valuable if irregular journaling (skipped weeks, burst entries) proves problematic for weekly aggregation. See `docs/evolution/drift_detection.md § 3.3`."),
]


# ---------------------------------------------------------------------------
# 04 — HMM
# ---------------------------------------------------------------------------

NB04_CELLS = [
    md("""\
# Drift Detection — 04: Hidden Markov Models (HMM)

Explicitly models latent regimes the user transitions between. The model learns:
1. **Emission distributions:** what each regime's alignment scores look like
2. **Transition matrix:** probability of switching between regimes at each step

The signal taxonomy **emerges from the learned transition patterns** — crash = aligned→drifting,
fade = aligned→struggling→drifting, spike = aligned→drifting→aligned. No hand-coding needed.

**Doc reference:** `docs/evolution/drift_detection.md § 3.4`

### States (K = 3)

| State | Label | Expected scores |
|---|---|---|
| 0 | Aligned | Centered on +1, low variance |
| 1 | Struggling | Centered on 0, moderate variance |
| 2 | Drifting | Centered on -1, low-moderate variance |

### Implementation

Uses `statsmodels.tsa.regime_switching.MarkovRegression` — a Markov-switching regression
with K regimes fitted by EM (Hamilton filter).

### Data note

With ~10-15 steps per persona, per-user EM will often not converge reliably.
This notebook pools across all personas to estimate regime parameters, then decodes
individual trajectories using the pooled model.
"""),

    code(SHARED_IMPORTS + "\nfrom statsmodels.tsa.regime_switching.markov_regression import MarkovRegression\nimport warnings\nwarnings.filterwarnings('ignore')\n\n" + SHARED_DATA),

    md("## Pooled HMM Training\n\nConcatenate all core-dimension alignment scores across personas and fit a 3-state Markov switching model.\nThis gives stable regime parameters despite the small per-persona sample size."),

    code("""\
W_MIN = 0.15
K     = 3      # number of latent states

# Collect all core-dimension score sequences
all_scores = []
for pid in persona_ids:
    t_idx, matrix = get_persona_matrix(pid)
    if len(t_idx) < 5:
        continue
    w = get_profile_weights(pid)
    for j in range(10):
        if w[j] >= W_MIN:
            all_scores.extend(matrix[:, j].tolist())

pooled = np.array(all_scores)
print(f"Pooled {len(pooled)} observations from core dimensions across all personas")
print(f"Mean: {pooled.mean():.3f}  Std: {pooled.std():.3f}  Range: [{pooled.min():.1f}, {pooled.max():.1f}]")

# Fit Markov Switching Model
# order=0 means intercept-only (no autoregressive terms)
model = MarkovRegression(pooled, k_regimes=K, order=0, switching_variance=True)
try:
    fit = model.fit(disp=False, maxiter=200)
    print(f"\\nConverged: {fit.mle_retvals['converged']}")
    print(f"Log-likelihood: {fit.llf:.2f}")
    print(f"\\nRegime means (intercepts):")
    for k in range(K):
        print(f"  State {k}: mean={fit.params[f'const[{k}]']:.3f}  "
              f"var={fit.params.get(f'sigma2[{k}]', fit.params.get('sigma2', 0)):.3f}")
    print(f"\\nTransition matrix:")
    print(fit.regime_transition)
    POOLED_FIT = fit
except Exception as e:
    print(f"Fit failed: {e}")
    POOLED_FIT = None
"""),

    md("## Decode Individual Personas\n\nUsing the pooled regime parameters, decode each persona's core-dimension trajectory\nvia the smoothed state probabilities (P(state_t | all data))."),

    code("""\
def decode_dimension(scores_1d: np.ndarray) -> dict:
    \"\"\"
    Fit a K-state Markov switching model and decode state probabilities.

    Returns
    -------
    dict with keys:
        smoothed_probs : (T, K) smoothed state probabilities
        most_likely    : (T,) most probable state at each step
        alerts         : list of t-indices in state 2 (drifting)
    \"\"\"
    T = len(scores_1d)
    if T < 4:
        return {"smoothed_probs": np.full((T, K), 1/K),
                "most_likely": np.zeros(T, dtype=int), "alerts": []}
    try:
        m   = MarkovRegression(scores_1d, k_regimes=K, order=0, switching_variance=True)
        fit = m.fit(disp=False, maxiter=200)
        probs = fit.smoothed_marginal_probabilities   # (T, K)
        # Identify which state corresponds to "drifting" (lowest mean)
        means = [fit.params[f"const[{k}]"] for k in range(K)]
        drift_state = int(np.argmin(means))
        most_likely = probs.values.argmax(axis=1)
        alerts = [t for t in range(T) if probs.values[t, drift_state] > 0.6]
        return {
            "smoothed_probs": probs.values,
            "most_likely":    most_likely,
            "drift_state":    drift_state,
            "alerts":         alerts,
            "means":          means,
        }
    except Exception:
        return {"smoothed_probs": np.full((T, K), 1/K),
                "most_likely": np.zeros(T, dtype=int),
                "drift_state": 2, "alerts": [], "means": [0.0, 0.0, 0.0]}


hmm_results = {}

for pid in persona_ids:
    t_idx, matrix = get_persona_matrix(pid)
    T = len(t_idx)
    if T < 5:
        continue

    w = get_profile_weights(pid)
    core_j = core_dim_indices(w, W_MIN)

    dim_results = {}
    alerts = []

    for j in core_j:
        out = decode_dimension(matrix[:, j])
        dim_results[j] = out
        for t in out["alerts"]:
            alerts.append((t, j))

    hmm_results[pid] = {
        "name":        id_to_name.get(pid, pid[:8]),
        "core":        id_to_core.get(pid, []),
        "T":           T,
        "t_idx":       t_idx,
        "matrix":      matrix,
        "weights":     w,
        "dim_results": dim_results,
        "alerts":      alerts,
    }

print(f"HMM decoded {len(hmm_results)} personas\\n")
print(f"{'Persona':<25s}  {'Core':<28s}  T  Alerts")
print("-" * 70)
for pid, r in hmm_results.items():
    print(f"{r['name']:<25s}  {', '.join(r['core']):<28s}  {r['T']:2d}  {len(r['alerts'])}")
"""),

    md("## Visualise State Probabilities\n\nFor each persona × core dimension: alignment scores alongside smoothed state probabilities.\nState 2 (drifting) probability in red."),

    code("""\
STATE_COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]  # aligned / struggling / drifting
STATE_LABELS = ["Aligned", "Struggling", "Drifting"]
palette = sns.color_palette("tab10", n_colors=10)


def plot_hmm_persona(pid: str):
    r = hmm_results[pid]
    matrix = r["matrix"]
    T = r["T"]
    core_j = list(r["dim_results"].keys())
    if not core_j:
        return

    fig, axes = plt.subplots(len(core_j), 2, figsize=(14, 2.8 * len(core_j)),
                             sharex=True, gridspec_kw={"width_ratios": [2, 1]})
    if len(core_j) == 1:
        axes = [axes]

    for ax_row, j in zip(axes, core_j):
        ax_score, ax_state = ax_row
        out = r["dim_results"][j]
        scores  = matrix[:, j]
        probs   = out["smoothed_probs"]   # (T, K)
        drift_s = out.get("drift_state", 2)

        ax_score.plot(range(T), scores, "o-", color=palette[j], lw=1.8)
        ax_score.axhline(0, color="grey", lw=0.5, ls="--")
        ax_score.set_ylim(-1.4, 1.4)
        ax_score.set_ylabel(DIM_LABELS[j], fontsize=9)

        for t in out["alerts"]:
            ax_score.axvspan(t - 0.4, t + 0.4, color="red", alpha=0.2, zorder=1)

        # Stacked state probability bars
        bottoms = np.zeros(T)
        for k in range(K):
            ax_state.bar(range(T), probs[:, k], bottom=bottoms,
                         color=STATE_COLORS[k], alpha=0.8, label=STATE_LABELS[k])
            bottoms += probs[:, k]
        ax_state.set_ylim(0, 1)
        ax_state.set_ylabel("P(state)", fontsize=8)

    axes[0][0].set_title(f"{r['name']} — {', '.join(r['core'])}", fontweight="bold")
    axes[0][1].legend(fontsize=7, loc="upper right")
    axes[-1][0].set_xlabel("t_index")
    axes[-1][1].set_xlabel("t_index")
    plt.tight_layout()
    plt.show()


for pid in list(hmm_results.keys())[:4]:
    plot_hmm_persona(pid)
"""),

    md("## Limitations\n\n- **Small per-persona data:** With ~10 steps, per-user EM often doesn't converge. Pooled fitting shares parameters but ignores individual variation in transition rates.\n- **K is fixed:** K=3 (aligned/struggling/drifting) mirrors the rule-based taxonomy but the true number of regimes is unknown.\n- **State labelling:** The 'drifting' state is identified as the one with the lowest mean — but this may not hold if scores are noisy.\n- **Profile weights:** HMM operates per-dimension independently. `w_u` must be applied as a post-hoc filter.\n\nHMM becomes the recommended second upgrade path at ~30+ entries per user, where per-user EM becomes stable. See `docs/evolution/drift_detection.md § 3.4`."),
]


# ---------------------------------------------------------------------------
# 05 — Autoencoder
# ---------------------------------------------------------------------------

NB05_CELLS = [
    md("""\
# Drift Detection — 05: Autoencoder (Anomaly Detection)

Trains a neural network to reconstruct "normal" alignment trajectories.
High reconstruction error = anomaly = potential drift.

**Doc reference:** `docs/evolution/drift_detection.md § 3.5`

### ⚠️ Data limitation

> With 24 annotated personas × ~10 entries each ≈ 240 trajectory windows,
> an autoencoder will **memorise rather than generalise**.
>
> This notebook is a proof-of-concept implementation. The approach becomes
> meaningful at ~500+ diverse trajectory windows (hundreds of real users
> with months of journaling history).

### Architecture

- **Input:** Window of W=4 steps × 10 dimensions = 40-dim vector
- **Encoder:** 40 → 20 → 10
- **Bottleneck:** 10-dim latent representation
- **Decoder:** 10 → 20 → 40
- **Loss:** MSE reconstruction loss on "aligned" training windows
"""),

    code(SHARED_IMPORTS + "\nimport torch\nimport torch.nn as nn\nfrom torch.utils.data import DataLoader, TensorDataset\n\ntorch.manual_seed(42)\nnp.random.seed(42)\n\n" + SHARED_DATA),

    md("## Build Training Windows\n\nExtract sliding windows of W steps from persona trajectories.\nTrain only on windows from the 'aligned' portion (first ~40% of each persona's trajectory,\nassuming alignment holds early on)."),

    code("""\
W     = 4    # window size
W_MIN = 0.15

def extract_windows(matrix: np.ndarray, window: int = W) -> np.ndarray:
    \"\"\"Extract all sliding windows of size `window` from (T, K) matrix.\"\"\"
    T, K = matrix.shape
    if T < window:
        return np.empty((0, window * K))
    return np.array([matrix[t:t+window].flatten() for t in range(T - window + 1)])


# Training set: first 40% of each persona (assumed aligned baseline)
train_windows = []
for pid in persona_ids:
    t_idx, matrix = get_persona_matrix(pid)
    T = len(t_idx)
    if T < W + 2:
        continue
    cutoff = max(W, int(T * 0.4))
    wins = extract_windows(matrix[:cutoff])
    train_windows.append(wins)

if train_windows:
    X_train = np.vstack(train_windows).astype(np.float32)
else:
    X_train = np.empty((0, W * 10), dtype=np.float32)

print(f"Training windows: {X_train.shape[0]} windows of size {W} steps × 10 dims")
print(f"WARNING: {X_train.shape[0]} windows is far below the ~500+ needed for generalisation.")
print("This notebook demonstrates the implementation — treat results as illustrative only.")
"""),

    md("## Autoencoder Architecture"),

    code("""\
class DriftAutoencoder(nn.Module):
    def __init__(self, input_dim: int = W * 10, latent_dim: int = 10):
        super().__init__()
        hidden = input_dim * 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, input_dim),
            nn.Tanh(),   # scores are in [-1, 1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


model_ae = DriftAutoencoder(input_dim=W * 10, latent_dim=10)
print(model_ae)
print(f"\\nParameters: {sum(p.numel() for p in model_ae.parameters()):,}")
"""),

    md("## Train"),

    code("""\
EPOCHS    = 150
LR        = 1e-3
BATCH     = min(16, max(1, len(X_train) // 4))

recon_errors_train = []

if X_train.shape[0] >= 4:
    dataset = TensorDataset(torch.from_numpy(X_train))
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True)

    optimiser = torch.optim.Adam(model_ae.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model_ae.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for (x_batch,) in loader:
            optimiser.zero_grad()
            x_hat = model_ae(x_batch)
            loss  = criterion(x_hat, x_batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(x_batch)
        recon_errors_train.append(epoch_loss / len(X_train))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(recon_errors_train)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Autoencoder training loss (MSE)")
    plt.tight_layout()
    plt.show()
    print(f"Final training MSE: {recon_errors_train[-1]:.4f}")
else:
    print("Insufficient training data — skipping training.")
"""),

    md("## Compute Reconstruction Error on Full Trajectories\n\nSlide the window across each persona's full trajectory. High reconstruction error at step t\nindicates the window ending at t looks different from training patterns."),

    code("""\
model_ae.eval()
AE_THRESH_QUANTILE = 0.90   # flag top 10% reconstruction errors as anomalies

# Compute errors on training set to set threshold
if X_train.shape[0] >= 4:
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_train)
        recon    = model_ae(x_tensor).numpy()
    train_errors = ((X_train - recon) ** 2).mean(axis=1)
    threshold = float(np.quantile(train_errors, AE_THRESH_QUANTILE))
    print(f"Alert threshold (p{AE_THRESH_QUANTILE*100:.0f} of training errors): {threshold:.4f}")
else:
    threshold = 0.1
    print("Using default threshold (no training data)")

ae_results = {}

for pid in persona_ids:
    t_idx, matrix = get_persona_matrix(pid)
    T = len(t_idx)
    if T < W + 2:
        continue

    windows = extract_windows(matrix)
    if len(windows) == 0:
        continue
    x_t = torch.from_numpy(windows.astype(np.float32))

    with torch.no_grad():
        recon = model_ae(x_t).numpy()
    errors = ((windows - recon) ** 2).mean(axis=1)

    # t-step of each window = last step of the window
    window_t = list(range(W - 1, T))
    alerts = [(window_t[i], -1) for i, e in enumerate(errors) if e > threshold]

    ae_results[pid] = {
        "name":    id_to_name.get(pid, pid[:8]),
        "core":    id_to_core.get(pid, []),
        "T":       T,
        "t_idx":   t_idx,
        "matrix":  matrix,
        "weights": get_profile_weights(pid),
        "errors":  errors,
        "window_t": window_t,
        "alerts":  alerts,
        "threshold": threshold,
    }

print(f"\\nAE anomaly detection on {len(ae_results)} personas:")
for pid, r in ae_results.items():
    print(f"  {r['name']:<25s}  alerts: {len(r['alerts'])}")
"""),

    md("## Visualise Reconstruction Error"),

    code("""\
palette = sns.color_palette("tab10", n_colors=10)


def plot_ae_persona(pid: str):
    r = ae_results[pid]
    matrix  = r["matrix"]
    T       = r["T"]
    errors  = r["errors"]
    wt      = r["window_t"]

    core_j = core_dim_indices(r["weights"], W_MIN)

    n_rows = len(core_j) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows), sharex=True)

    # Score plots
    for ax, j in zip(axes[:-1], core_j):
        ax.plot(range(T), matrix[:, j], "o-", color=palette[j], lw=1.8)
        ax.axhline(0, color="grey", lw=0.5, ls="--")
        ax.set_ylim(-1.4, 1.4)
        ax.set_ylabel(DIM_LABELS[j], fontsize=9)
        for t, _ in r["alerts"]:
            ax.axvspan(t - 0.4, t + 0.4, color="red", alpha=0.2)

    # Reconstruction error
    ax_err = axes[-1]
    ax_err.bar(wt, errors, color="salmon", alpha=0.8, width=0.6)
    ax_err.axhline(r["threshold"], color="red", lw=1.2, ls="--", label=f"threshold={r['threshold']:.3f}")
    ax_err.set_ylabel("Recon. error", fontsize=9)
    ax_err.legend(fontsize=8)

    axes[0].set_title(f"{r['name']} — {', '.join(r['core'])} (⚠ memorisation risk at this data scale)",
                      fontweight="bold")
    axes[-1].set_xlabel("t_index")
    plt.tight_layout()
    plt.show()


for pid in list(ae_results.keys())[:4]:
    plot_ae_persona(pid)
"""),

    md("## Limitations (Reiterated)\n\n- **Data:** ~240 windows is far below the ~500+ needed. The autoencoder memorises training trajectories rather than learning general 'normal' patterns. Reconstruction errors will be low on training personas and unreliable on others.\n- **No profile weights:** Reconstruction error is aggregate across all 10 dimensions. A misalignment on a low-weight dimension contributes equally to a misalignment on a core value.\n- **Interpretability:** `error = 0.73` tells the Coach nothing about *which* value changed or *why*.\n- **Cross-dimension patterns:** The autoencoder's main advantage — detecting correlated cross-dimension shifts (e.g., Self-Direction down + Achievement up) — requires much more data to be reliable.\n\n**Recommendation:** Do not use autoencoders until Twinkl has hundreds of real users with months of history. Simpler cosine similarity already captures the directional component at this scale. See `docs/evolution/drift_detection.md § 3.5`."),
]


# ---------------------------------------------------------------------------
# Write notebooks
# ---------------------------------------------------------------------------

notebooks = {
    "01_drift_detection_rule_based.ipynb": NB01_CELLS,
    "02_drift_detection_bocpd.ipynb":      NB02_CELLS,
    "03_drift_detection_gp.ipynb":         NB03_CELLS,
    "04_drift_detection_hmm.ipynb":        NB04_CELLS,
    "05_drift_detection_autoencoder.ipynb": NB05_CELLS,
}

for fname, cells in notebooks.items():
    path = OUT / fname
    with open(path, "w") as f:
        json.dump(notebook(cells), f, indent=1)
    print(f"Written: {path}")

print("\nDone.")
