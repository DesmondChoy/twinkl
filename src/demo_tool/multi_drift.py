"""Multi-detector drift comparison for the demo tool.

Runs all 6 rule-based detectors from the notebook evaluation on a chosen
data source (judge labels or critic predictions) and returns a bundle
suitable for UI rendering and comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.special import rel_entr

from src.models.judge import SCHWARTZ_VALUE_ORDER

JUDGE_LABELS_PATH = Path("logs/judge_labels/judge_labels.parquet")
REGISTRY_PATH = Path("logs/registry/personas.parquet")

VALUE_COLS = [f"alignment_{dim}" for dim in SCHWARTZ_VALUE_ORDER]
DIM_LABELS = ["SD", "ST", "HE", "AC", "PO", "SE", "CO", "TR", "BE", "UN"]

DETECTOR_KEYS = ["baseline", "ema", "cusum", "cosine", "control_chart", "kl"]
DETECTOR_NAMES = ["Baseline", "EMA", "CUSUM", "Cosine", "Control Chart", "KL Div"]


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DetectorResult:
    name: str
    key: str
    alert_steps: set[int]  # t_index values where an alert fired (any dimension)
    alert_tuples: list[tuple]  # raw (t, j, ...) tuples for per-dimension detail
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiDriftBundle:
    persona_id: str
    source: str  # "judge" or "critic"
    n_entries: int
    t_indices: list[int]
    dates: list[str]
    core_values: list[str]
    scores_matrix: np.ndarray  # (T, 10) float
    weights: np.ndarray        # (10,) float
    detectors: list[DetectorResult]
    consensus: dict[int, int]  # t_index -> vote count across detectors


# ── Profile weights ────────────────────────────────────────────────────────────

def _get_profile_weights(core_values: list[str]) -> np.ndarray:
    name_to_idx = {s.lower().replace("-", "_").replace(" ", "_"): i for i, s in enumerate(SCHWARTZ_VALUE_ORDER)}
    w = np.zeros(10)
    for v in core_values:
        key = v.lower().replace("-", "_").replace(" ", "_")
        if key in name_to_idx:
            w[name_to_idx[key]] = 1.0
    if w.sum() > 0:
        w /= w.sum()
    return w


# ── Detectors (ported verbatim from 01_drift_detection_rule_based.ipynb) ───────

def detect_baseline(
    scores: np.ndarray,
    weights: np.ndarray,
    tau_low: float = -0.4,
    c_min: int = 3,
    w_min: float = 0.15,
) -> dict:
    """Per-dimension dual-trigger: crash + rut."""
    T, K = scores.shape
    alerts = []

    rut_count = np.zeros(K)
    rut_active = np.zeros(K, dtype=bool)

    for t in range(1, T):
        for j in range(K):
            if weights[j] < w_min:
                rut_count[j] = 0
                rut_active[j] = False
                continue

            prev, curr = scores[t - 1, j], scores[t, j]

            if prev >= 0 and curr < 0:
                alerts.append((t, j, "crash"))

            if curr < tau_low:
                rut_count[j] += 1
                if rut_count[j] >= c_min and not rut_active[j]:
                    alerts.append((t, j, "rut"))
                    rut_active[j] = True
            else:
                rut_count[j] = 0
                rut_active[j] = False

    scalar = scores @ weights
    return {"alerts": alerts, "scalar": scalar}


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


def detect_cosine(
    scores: np.ndarray,
    weights: np.ndarray,
    threshold: float = 0.0,
) -> dict:
    """Holistic profile alignment via dot product with declared values."""
    cosines = []
    alerts = []

    for t in range(scores.shape[0]):
        a = scores[t]
        norm = np.linalg.norm(weights) * np.linalg.norm(a)
        cos = float(np.dot(weights, a) / norm) if norm > 1e-8 else 0.0
        cosines.append(cos)
        if cos < threshold:
            alerts.append(t)

    return {"alerts": alerts, "cosines": cosines}


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
    mu = baseline.mean(axis=0)
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


# ── Orchestrator ───────────────────────────────────────────────────────────────

def _run_detectors(scores: np.ndarray, weights: np.ndarray, T: int) -> list[DetectorResult]:
    baseline_end = max(2, min(4, T // 3))

    raw_results = [
        detect_baseline(scores, weights),
        detect_ema(scores, weights),
        detect_cusum(scores, weights),
        detect_cosine(scores, weights),
        detect_control_chart(scores, weights, baseline_end=baseline_end),
        detect_kl_divergence(scores, weights, baseline_end=baseline_end),
    ]

    detector_results = []
    for key, name, raw in zip(DETECTOR_KEYS, DETECTOR_NAMES, raw_results):
        alert_tuples = raw["alerts"]
        # Normalise: cosine returns bare ints, others return (t, j, ...) tuples
        if key == "cosine":
            alert_steps = set(alert_tuples)
        else:
            alert_steps = {t for t, *_ in alert_tuples}
        meta = {k: v for k, v in raw.items() if k != "alerts"}
        detector_results.append(DetectorResult(
            name=name,
            key=key,
            alert_steps=alert_steps,
            alert_tuples=alert_tuples,
            meta=meta,
        ))

    return detector_results


def _build_consensus(detectors: list[DetectorResult], t_indices: list[int]) -> dict[int, int]:
    return {t: sum(1 for d in detectors if t in d.alert_steps) for t in t_indices}


def run_multi_drift_from_judge(
    persona_id: str,
    judge_labels_path: Path = JUDGE_LABELS_PATH,
    registry_path: Path = REGISTRY_PATH,
    min_entries: int = 3,
) -> MultiDriftBundle | None:
    """Load judge labels for a persona and run all 6 detectors."""
    labels = pl.read_parquet(judge_labels_path)
    persona_labels = labels.filter(pl.col("persona_id") == persona_id).sort("t_index")
    if len(persona_labels) < min_entries:
        return None

    scores = persona_labels.select(VALUE_COLS).cast(pl.Float64).to_numpy()
    t_indices = persona_labels["t_index"].to_list()
    dates = [str(d) for d in persona_labels["date"].to_list()]

    registry = pl.read_parquet(registry_path)
    persona_row = registry.filter(pl.col("persona_id") == persona_id)
    core_values: list[str] = []
    if len(persona_row) > 0:
        raw = persona_row["core_values"][0]
        core_values = list(raw) if raw is not None else []

    weights = _get_profile_weights(core_values)
    detectors = _run_detectors(scores, weights, len(t_indices))
    consensus = _build_consensus(detectors, t_indices)

    return MultiDriftBundle(
        persona_id=persona_id,
        source="judge",
        n_entries=len(t_indices),
        t_indices=t_indices,
        dates=dates,
        core_values=core_values,
        scores_matrix=scores,
        weights=weights,
        detectors=detectors,
        consensus=consensus,
    )


def run_multi_drift_from_critic(
    persona_id: str,
    timeline_df: pl.DataFrame,
    registry_path: Path = REGISTRY_PATH,
    min_entries: int = 3,
) -> MultiDriftBundle | None:
    """Use critic predictions (timeline_df) as input for all 6 detectors."""
    persona_df = timeline_df.filter(pl.col("persona_id") == persona_id).sort("t_index") \
        if "persona_id" in timeline_df.columns \
        else timeline_df.sort("t_index")

    available_cols = [c for c in VALUE_COLS if c in persona_df.columns]
    if len(persona_df) < min_entries or not available_cols:
        return None

    scores = persona_df.select(available_cols).cast(pl.Float64).to_numpy()
    t_indices = persona_df["t_index"].to_list()
    dates = [str(d) for d in persona_df["date"].to_list()]

    registry = pl.read_parquet(registry_path)
    persona_row = registry.filter(pl.col("persona_id") == persona_id)
    core_values: list[str] = []
    if len(persona_row) > 0:
        raw = persona_row["core_values"][0]
        core_values = list(raw) if raw is not None else []

    weights = _get_profile_weights(core_values)
    detectors = _run_detectors(scores, weights, len(t_indices))
    consensus = _build_consensus(detectors, t_indices)

    return MultiDriftBundle(
        persona_id=persona_id,
        source="critic",
        n_entries=len(t_indices),
        t_indices=t_indices,
        dates=dates,
        core_values=core_values,
        scores_matrix=scores,
        weights=weights,
        detectors=detectors,
        consensus=consensus,
    )
