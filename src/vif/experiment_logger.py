"""Experiment logging for VIF Critic ablation studies.

Writes structured YAML files (one per model) and a compact Markdown index
optimized for LLM agent consumption. Designed to be called from notebook
cells after training completes.

Output structure:
    logs/experiments/
    ├── index.md              # Markdown table for quick LLM context reads
    └── runs/                 # One YAML per model per run
        ├── run_001_MSE.yaml
        ├── run_001_CORAL.yaml
        └── ...

Usage (in notebook):
    from src.vif.experiment_logger import log_experiment_run

    logged_paths = log_experiment_run(
        config=CONFIG,
        trained_models=trained_models,
        all_results=all_results,
        all_calibration=all_calibration,
        all_hedging=all_hedging,
        all_recall_data=all_recall_data,
        n_train=n_train, n_val=n_val, n_test=n_test,
        pct_truncated=pct_truncated,
        state_dim=state_encoder.state_dim,
        observations="Your notes here",
    )
"""

import hashlib
import json
import math
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from src.models.judge import SCHWARTZ_VALUE_ORDER

# ─── Constants ────────────────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path("logs/experiments")
RUNS_DIR = EXPERIMENTS_DIR / "runs"
INDEX_PATH = EXPERIMENTS_DIR / "index.md"

INDEX_HEADER = (
    "# VIF Experiment Index\n\n"
    "| run | model | encoder | ws | hd | do | loss | params | ratio | "
    "MAE | Acc | QWK | Spear | Cal | MinR | file |\n"
    "|-----|-------|---------|---:|---:|---:|------|-------:|------:|"
    "----:|----:|----:|------:|----:|-----:|------|\n"
)


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _to_python(obj):
    """Recursively convert numpy/torch types to Python natives for YAML."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        if math.isnan(val):
            return val  # PyYAML renders as .nan
        return val
    if isinstance(obj, np.ndarray):
        return _to_python(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _round_val(v, dp=4):
    """Round a float to dp decimal places, pass through non-floats."""
    if isinstance(v, float) and not math.isnan(v):
        return round(v, dp)
    return v


def _next_run_id() -> str:
    """Scan runs/ for run_\\d{3}_*.yaml and return next sequential ID."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    existing = []
    for f in RUNS_DIR.glob("run_*_*.yaml"):
        match = re.match(r"run_(\d{3})_", f.name)
        if match:
            existing.append(int(match.group(1)))
    next_num = max(existing) + 1 if existing else 1
    return f"run_{next_num:03d}"


def _config_fingerprint(config_section: dict) -> str:
    """Deterministic hash of the config section for dedup. Returns 12-char hex."""
    canonical = json.dumps(config_section, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _find_existing_run(model_name: str, config_hash: str) -> Path | None:
    """Find an existing YAML run file with matching model and config hash."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(RUNS_DIR.glob(f"run_*_{model_name}.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            if data and data.get("metadata", {}).get("config_hash") == config_hash:
                return f
        except Exception:
            continue
    return None


def _get_git_commit() -> str:
    """Return short commit hash with (dirty) suffix. Never crashes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if dirty:
            commit += "(dirty)"
        return commit
    except Exception:
        return "unknown"


def _strip_dirty(commit: str) -> str:
    """Remove (dirty) suffix from commit hash for use in git log ranges."""
    return commit.replace("(dirty)", "")


def _encoder_family(model_name: str) -> str:
    """Map encoder model name to a family key for run matching.

    Examples:
        all-MiniLM-L6-v2 → "MiniLM"
        nomic-ai/nomic-embed-text-v1.5 → "nomic"
        sentence-transformers/all-mpnet-base-v2 → "mpnet"
    """
    lower = model_name.lower()
    if "minilm" in lower:
        return "MiniLM"
    if "nomic" in lower:
        return "nomic"
    if "mpnet" in lower:
        return "mpnet"
    return model_name.split("/")[-1]


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten nested dict to dot-notation keys for config diffing.

    Example: {"a": {"b": 1}} → {"a.b": 1}
    """
    items: dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def _compute_config_delta(current: dict, prev: dict) -> dict:
    """Deep-diff two config dicts → {added, removed, changed}.

    Both dicts are flattened to dot-notation before comparison so nested
    changes (e.g. state_encoder.ema_alpha) appear as single keys.
    """
    curr_flat = _flatten_dict(current)
    prev_flat = _flatten_dict(prev)

    added = {k: v for k, v in curr_flat.items() if k not in prev_flat}
    removed = {k: v for k, v in prev_flat.items() if k not in curr_flat}
    changed = {
        k: {"from": prev_flat[k], "to": curr_flat[k]}
        for k in curr_flat
        if k in prev_flat and curr_flat[k] != prev_flat[k]
    }

    return {"added": added, "removed": removed, "changed": changed}


def _canonicalize_run_config(config: dict) -> dict:
    """Normalize config to run-level fields for provenance diffing.

    Provenance should explain run-to-run changes, not differences between
    model heads within the same run. Exclude head-specific keys here.
    """
    canonical = deepcopy(config)
    training = canonical.get("training")
    if isinstance(training, dict):
        training.pop("loss_fn", None)
        training.pop("weighted_mse_scale", None)
    return canonical


def _get_git_log_between(prev_commit: str, current_commit: str) -> list[str]:
    """Return ``git log --oneline prev..current`` as a list of strings.

    Handles (dirty) suffixes, same-commit case, and git errors gracefully.
    """
    prev_clean = _strip_dirty(prev_commit)
    curr_clean = _strip_dirty(current_commit)

    if prev_clean == curr_clean:
        return []

    try:
        output = subprocess.check_output(
            ["git", "log", "--oneline", f"{prev_clean}..{curr_clean}"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if not output:
            return []
        return output.splitlines()
    except Exception:
        return []


def _find_previous_run(run_id: str, family: str) -> dict | None:
    """Find the most recent predecessor run with the same encoder family.

    Scans all YAML run files with a lower run number and matching encoder
    family. Returns the parsed data dict of the first matching model found
    in the most recent predecessor run, or None if no predecessor exists.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    match = re.match(r"run_(\d{3})", run_id)
    if not match:
        return None
    current_num = int(match.group(1))

    candidates: list[tuple[int, dict]] = []
    for f in sorted(RUNS_DIR.glob("run_*_*.yaml")):
        m = re.match(r"run_(\d{3})_", f.name)
        if not m:
            continue
        num = int(m.group(1))
        if num >= current_num:
            continue
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            if not data or "config" not in data:
                continue
            encoder_name = data["config"]["encoder"]["model_name"]
            if _encoder_family(encoder_name) == family:
                candidates.append((num, data))
        except Exception:
            continue

    if not candidates:
        return None

    # Most recent predecessor (highest run number)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _build_provenance(run_id: str, config: dict, git_commit: str) -> dict:
    """Build the provenance section for a run.

    Finds the most recent predecessor with the same encoder family, then
    computes the git log and config delta between them.
    """
    encoder_name = config.get("encoder", {}).get("model_name", "unknown")
    family = _encoder_family(encoder_name)

    prev_data = _find_previous_run(run_id, family)

    if prev_data is None:
        return {
            "prev_run_id": None,
            "prev_git_commit": None,
            "git_log": [],
            "config_delta": {"added": {}, "removed": {}, "changed": {}},
            "rationale": "",
        }

    prev_run_id = prev_data["metadata"]["run_id"]
    prev_git_commit = prev_data["metadata"]["git_commit"]

    git_log = _get_git_log_between(prev_git_commit, git_commit)
    config_delta = _compute_config_delta(
        _canonicalize_run_config(config),
        _canonicalize_run_config(prev_data["config"]),
    )

    return {
        "prev_run_id": prev_run_id,
        "prev_git_commit": prev_git_commit,
        "git_log": git_log,
        "config_delta": _to_python(config_delta),
        "rationale": "",
    }


def _encoder_shorthand(config: dict) -> str:
    """Map encoder config to compact label for index table.

    Examples:
        nomic-ai/nomic-embed-text-v1.5 + truncate_dim=256 → "nomic-256d"
        all-MiniLM-L6-v2 → "MiniLM-384d"
    """
    model = config.get("encoder_model", "unknown")
    truncate_dim = config.get("truncate_dim")

    if "nomic" in model.lower():
        dim = truncate_dim if truncate_dim else "768"
        return f"nomic-{dim}d"
    elif "MiniLM" in model:
        return "MiniLM-384d"
    elif "mpnet" in model.lower():
        return "mpnet-768d"
    else:
        return model.split("/")[-1][:15]


def _loss_shorthand(model_name: str, config: dict) -> str:
    """Map model name + config to lowercase loss identifier.

    For the MSE model, checks config["loss_fn"] to distinguish plain MSE from
    weighted_mse (with scale suffix), so different loss settings produce
    different fingerprints.
    """
    mapping = {
        "CORAL": "coral",
        "CORN": "corn",
        "EMD": "emd",
        "SoftOrdinal": "soft_ordinal",
    }
    if model_name == "MSE":
        loss_fn = config.get("loss_fn", "mse")
        if loss_fn == "weighted_mse":
            scale = config.get("weighted_mse_scale", 5.0)
            return f"weighted_mse_s{scale}"
        return "mse"
    return mapping.get(model_name, model_name.lower())


def _build_experiment_dict(
    run_id: str,
    model_name: str,
    config: dict,
    trained_result: dict,
    eval_result: dict,
    calibration: dict,
    hedging: np.ndarray,
    recall_data: dict,
    n_train: int,
    n_val: int,
    n_test: int,
    pct_truncated: float,
    state_dim: int,
    observations: str,
    provenance: dict | None = None,
) -> dict:
    """Assemble the full experiment dict matching the YAML schema."""
    model = trained_result["model"]
    n_parameters = sum(p.numel() for p in model.parameters())
    history = trained_result["history"]
    best_epoch = trained_result["best_epoch"]

    # Training dynamics
    total_epochs = len(history["train_loss"])
    train_loss_at_best = history["train_loss"][best_epoch] if total_epochs > 0 else None
    val_loss_at_best = history["val_loss"][best_epoch] if total_epochs > 0 else None
    gap_at_best = (
        (val_loss_at_best - train_loss_at_best)
        if train_loss_at_best is not None
        else None
    )
    # Notebooks use "lr"; train.py uses "learning_rate"
    lr_key = "lr" if "lr" in history else "learning_rate"
    final_lr = history[lr_key][-1] if total_epochs > 0 else None

    # Per-dimension metrics
    per_dimension = {}
    for dim in SCHWARTZ_VALUE_ORDER:
        dim_metrics = {
            "mae": _round_val(eval_result["mae_per_dim"][dim]),
            "accuracy": _round_val(eval_result["accuracy_per_dim"][dim]),
            "qwk": _round_val(eval_result["qwk_per_dim"][dim]),
            "spearman": _round_val(eval_result["spearman_per_dim"][dim]),
            "calibration": _round_val(calibration["per_dim"][dim]),
        }
        # Add per-dim hedging
        dim_idx = SCHWARTZ_VALUE_ORDER.index(dim)
        dim_metrics["hedging"] = _round_val(float(hedging[dim_idx]))
        per_dimension[dim] = dim_metrics

    experiment = {
        "metadata": {
            "experiment_id": f"{run_id}_{model_name}",
            "run_id": run_id,
            "model_name": model_name,
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _get_git_commit(),
        },
        "config": {
            "encoder": {
                "model_name": config.get("encoder_model", "unknown"),
                "truncate_dim": config.get("truncate_dim"),
                "text_prefix": config.get("text_prefix"),
                "trust_remote_code": config.get("trust_remote_code"),
            },
            "state_encoder": {
                "window_size": config.get("window_size", 1),
            },
            "data": {
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "train_ratio": config.get("train_ratio", 0.70),
                "val_ratio": config.get("val_ratio", 0.15),
                "split_seed": config.get("seed", 42),
                "pct_truncated": _round_val(pct_truncated),
                "state_dim": state_dim,
            },
            "model": {
                "hidden_dim": config.get("hidden_dim", 256),
                "dropout": config.get("dropout", 0.2),
            },
            "training": {
                "loss_fn": _loss_shorthand(model_name, config),
                "learning_rate": config.get("learning_rate", 0.001),
                "weight_decay": config.get("weight_decay", 0.01),
                "batch_size": config.get("batch_size", 16),
                "epochs": config.get("epochs", 100),
                "early_stopping_patience": config.get("early_stopping_patience", 20),
                "scheduler_factor": config.get("scheduler_factor", 0.5),
                "scheduler_patience": config.get("scheduler_patience", 10),
                "seed": config.get("seed", 42),
                "weighted_mse_scale": (
                    config.get("weighted_mse_scale")
                    if model_name == "MSE" and config.get("loss_fn") == "weighted_mse"
                    else None
                ),
            },
        },
        "capacity": {
            "n_parameters": n_parameters,
            "param_sample_ratio": _round_val(n_parameters / n_train if n_train > 0 else 0),
        },
        "training_dynamics": {
            "best_epoch": best_epoch + 1,  # 1-indexed for readability
            "total_epochs": total_epochs,
            "train_loss_at_best": _round_val(train_loss_at_best),
            "val_loss_at_best": _round_val(val_loss_at_best),
            "gap_at_best": _round_val(gap_at_best),
            "final_lr": _round_val(final_lr, dp=6),
        },
        "evaluation": {
            "mae_mean": _round_val(eval_result["mae_mean"]),
            "accuracy_mean": _round_val(eval_result["accuracy_mean"]),
            "qwk_mean": _round_val(eval_result["qwk_mean"]),
            "spearman_mean": _round_val(eval_result["spearman_mean"]),
            "calibration_global": _round_val(calibration["global"]),
            "calibration_positive_dims": calibration["positive_count"],
            "mean_uncertainty": _round_val(calibration["mean_uncertainty"]),
            "minority_recall_mean": _round_val(recall_data["mean_minority"]),
            "recall_minus1": _round_val(recall_data["recall_minus1"]),
            "recall_plus1": _round_val(recall_data["recall_plus1"]),
            "hedging_mean": _round_val(float(hedging.mean())),
        },
        "per_dimension": per_dimension,
        "observations": observations if observations else "",
    }

    # Remove None values from encoder and training config
    experiment["config"]["encoder"] = {
        k: v for k, v in experiment["config"]["encoder"].items() if v is not None
    }
    experiment["config"]["training"] = {
        k: v for k, v in experiment["config"]["training"].items() if v is not None
    }

    # Compute config fingerprint and store in metadata
    experiment["metadata"]["config_hash"] = _config_fingerprint(experiment["config"])

    # Insert provenance between metadata and config (preserves YAML key order)
    if provenance is not None:
        ordered: dict = {}
        for key, value in experiment.items():
            ordered[key] = value
            if key == "metadata":
                ordered["provenance"] = provenance
        experiment = ordered

    return _to_python(experiment)


def _write_yaml(data: dict, path: Path) -> None:
    """Write experiment dict to YAML with header comment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        f"# VIF Experiment: {data['metadata']['experiment_id']}\n"
        f"# Generated: {data['metadata']['timestamp']}\n"
        f"# Git: {data['metadata']['git_commit']}\n"
    )
    content = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    path.write_text(header + content, encoding="utf-8")


def _fmt_metric(v, dp=3) -> str:
    """Format metric for index: 0.2323 → '.232', NaN → 'N/A'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    rounded = round(v, dp)
    return f"{rounded:.{dp}f}"


def _format_index_row(data: dict) -> str:
    """Format a single experiment dict as a pipe-delimited index row."""
    meta = data["metadata"]
    cfg = data["config"]
    ev = data["evaluation"]

    run_num = meta["run_id"].replace("run_", "")
    model = meta["model_name"]
    encoder = _encoder_shorthand({
        "encoder_model": cfg["encoder"]["model_name"],
        "truncate_dim": cfg["encoder"].get("truncate_dim"),
    })
    ws = cfg["state_encoder"]["window_size"]
    hd = cfg["model"]["hidden_dim"]
    do = cfg["model"]["dropout"]
    loss = cfg["training"]["loss_fn"]
    params = data["capacity"]["n_parameters"]
    ratio = _fmt_metric(data["capacity"]["param_sample_ratio"], dp=1)
    mae = _fmt_metric(ev["mae_mean"])
    acc = _fmt_metric(ev["accuracy_mean"])
    qwk = _fmt_metric(ev["qwk_mean"])
    spear = _fmt_metric(ev["spearman_mean"])
    cal = _fmt_metric(ev["calibration_global"])
    minr = _fmt_metric(ev["minority_recall_mean"])
    file_path = f"runs/{meta['experiment_id']}.yaml"

    return (
        f"| {run_num} | {model} | {encoder} | {ws} | {hd} | {do} | "
        f"{loss} | {params} | {ratio} | {mae} | {acc} | {qwk} | "
        f"{spear} | {cal} | {minr} | {file_path} |\n"
    )


def _rebuild_index() -> None:
    """Rebuild index.md from all YAML files in runs/.

    Globs all run files, reads each, sorts by (run_id, model_name), and writes
    the complete index. Guarantees correctness after overwrites or deletions.
    """
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    for f in RUNS_DIR.glob("run_*_*.yaml"):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            if data and "metadata" in data:
                entries.append(data)
        except Exception:
            continue

    entries.sort(
        key=lambda d: (d["metadata"]["run_id"], d["metadata"]["model_name"])
    )

    content = INDEX_HEADER
    for data in entries:
        content += _format_index_row(data)

    INDEX_PATH.write_text(content, encoding="utf-8")


# ─── Public API ───────────────────────────────────────────────────────────────


def log_experiment_run(
    config: dict,
    trained_models: dict,
    all_results: dict,
    all_calibration: dict,
    all_hedging: dict,
    all_recall_data: dict,
    n_train: int,
    n_val: int,
    n_test: int,
    pct_truncated: float,
    state_dim: int,
    observations: str = "",
) -> list[dict]:
    """Log all models from one notebook run with config-based deduplication.

    If the config section hasn't changed since the last run for a given model,
    the existing YAML file is updated in place (preserving its run_id) instead
    of creating a new one. A new run_id is only allocated when at least one
    model has a config change.

    Args:
        config: Notebook CONFIG dict (flat keys like encoder_model, hidden_dim).
        trained_models: Dict of {model_name: {model, history, best_epoch, ...}}.
        all_results: Dict of {model_name: eval_result_dict}.
        all_calibration: Dict of {model_name: {per_dim, global, positive_count, ...}}.
        all_hedging: Dict of {model_name: ndarray of per-dim hedging values}.
        all_recall_data: Dict of {model_name: {recall_minus1, recall_plus1, ...}}.
        n_train: Number of training samples.
        n_val: Number of validation samples.
        n_test: Number of test samples.
        pct_truncated: Percentage of entries truncated by tokenizer.
        state_dim: Dimensionality of the state vector.
        observations: Free-text notes about this experiment run.

    Returns:
        List of dicts with keys: path, model, status ("created"|"updated"), run_id.
    """
    new_run_id = None  # Lazily allocated if any model needs a new file
    provenance_cache: dict[str, dict] = {}  # run_id → provenance (computed once)
    results = []

    for model_name in trained_models:
        # Skip models missing from any required metrics dict
        if model_name not in all_results:
            continue
        if model_name not in all_calibration:
            continue
        if model_name not in all_hedging:
            continue
        if model_name not in all_recall_data:
            continue

        # Build with placeholder run_id to compute config_hash (no provenance yet)
        experiment = _build_experiment_dict(
            run_id="__pending__",
            model_name=model_name,
            config=config,
            trained_result=trained_models[model_name],
            eval_result=all_results[model_name],
            calibration=all_calibration[model_name],
            hedging=all_hedging[model_name],
            recall_data=all_recall_data[model_name],
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            pct_truncated=pct_truncated,
            state_dim=state_dim,
            observations=observations,
        )

        config_hash = experiment["metadata"]["config_hash"]
        existing_path = _find_existing_run(model_name, config_hash)

        if existing_path is not None:
            # Config unchanged — update in place, preserving original run_id
            existing_data = yaml.safe_load(
                existing_path.read_text(encoding="utf-8")
            )
            original_run_id = existing_data["metadata"]["run_id"]
            actual_run_id = original_run_id
            experiment["metadata"]["run_id"] = original_run_id
            experiment["metadata"]["experiment_id"] = (
                f"{original_run_id}_{model_name}"
            )
        else:
            # New config — allocate run_id once, shared by all new models
            if new_run_id is None:
                new_run_id = _next_run_id()
            actual_run_id = new_run_id
            experiment["metadata"]["run_id"] = new_run_id
            experiment["metadata"]["experiment_id"] = (
                f"{new_run_id}_{model_name}"
            )

        # Compute provenance once per run_id, then insert into experiment dict
        if actual_run_id not in provenance_cache:
            provenance_cache[actual_run_id] = _build_provenance(
                actual_run_id,
                experiment["config"],
                experiment["metadata"]["git_commit"],
            )
        provenance = provenance_cache[actual_run_id]

        # Insert provenance between metadata and config
        ordered: dict = {}
        for key, value in experiment.items():
            ordered[key] = value
            if key == "metadata":
                ordered["provenance"] = provenance
        experiment = ordered

        if existing_path is not None:
            _write_yaml(experiment, existing_path)
            results.append({
                "path": existing_path,
                "model": model_name,
                "status": "updated",
                "run_id": actual_run_id,
            })
        else:
            yaml_path = RUNS_DIR / f"{new_run_id}_{model_name}.yaml"
            _write_yaml(experiment, yaml_path)
            results.append({
                "path": yaml_path,
                "model": model_name,
                "status": "created",
                "run_id": new_run_id,
            })

    _rebuild_index()
    return results


def backfill_provenance() -> list[str]:
    """Add provenance section to existing YAML run files that lack it.

    Scans all run files, computes provenance for each, and writes back.
    Files that already have a ``provenance`` key are skipped.

    Returns:
        List of file paths that were updated.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    updated: list[str] = []

    for f in sorted(RUNS_DIR.glob("run_*_*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            if not data or "provenance" in data:
                continue

            run_id = data["metadata"]["run_id"]
            git_commit = data["metadata"]["git_commit"]
            config = data["config"]

            provenance = _build_provenance(run_id, config, git_commit)

            # Insert provenance after metadata, preserving key order
            ordered: dict = {}
            for key, value in data.items():
                ordered[key] = value
                if key == "metadata":
                    ordered["provenance"] = provenance

            _write_yaml(ordered, f)
            updated.append(str(f))
        except Exception as e:
            print(f"Warning: could not backfill {f.name}: {e}")
            continue

    return updated
