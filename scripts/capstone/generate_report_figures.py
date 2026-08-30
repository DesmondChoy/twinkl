"""Generate the capstone paper figures from committed evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "capstone_report" / "images"

INK = "#14233B"
MUTED = "#637083"
TEAL = "#248D83"
BLUE = "#4F6FE8"
GOLD = "#D8A62A"
CORAL = "#C05A40"
PAPER = "#FBFAF6"
GRID = "#D8DDE5"
TEAL_LIGHT = "#E6F3F1"
BLUE_LIGHT = "#E9EDFC"
GOLD_LIGHT = "#FAF3DD"
CORAL_LIGHT = "#F8EAE6"


def configure_matplotlib() -> None:
    """Set one print-safe style for all charts."""
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a figure as a high-resolution PNG."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)


def add_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 10,
    text_color: str = INK,
    linewidth: float = 1.4,
) -> None:
    """Add one consistently styled rounded box to a diagram."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.11",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        linespacing=1.25,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    rad: float = 0,
    linestyle: str = "solid",
    linewidth: float = 1.6,
) -> None:
    """Add one arrow with optional curvature to a diagram."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            shrinkA=2,
            shrinkB=2,
        )
    )


def draw_adopted_architecture() -> None:
    """Show the implemented core path and separate offline research path."""
    fig, ax = plt.subplots(figsize=(12.2, 6.2))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.add_patch(
        FancyBboxPatch(
            (0.12, 2.42),
            11.96,
            3.58,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            facecolor="#F3F5F8",
            edgecolor=GRID,
            linewidth=1.2,
        )
    )
    ax.text(
        0.42,
        5.73,
        "IMPLEMENTED CORE ASSESSMENT PATH",
        color=MUTED,
        fontsize=10,
        weight="bold",
    )

    core_nodes = [
        (0.45, 4.38, 1.35, "Profile"),
        (2.15, 4.38, 2.05, "Journal Entries\n+ displayed-nudge response"),
        (4.58, 4.38, 2.12, "Weekly Drift Reviewer\ncumulative history · Luna-low"),
        (7.08, 4.38, 1.72, "Drift Detector\ndeterministic rule"),
        (9.18, 4.38, 2.42, "Weekly Drift Detection output\nstate + cited evidence"),
    ]
    for x, y, width, label in core_nodes:
        add_box(
            ax,
            x=x,
            y=y,
            width=width,
            height=0.82,
            text=label,
            facecolor="white",
            edgecolor=TEAL,
            fontsize=9.6,
        )
    for left, right in zip(core_nodes, core_nodes[1:], strict=False):
        add_arrow(
            ax,
            (left[0] + left[2], left[1] + 0.41),
            (right[0], right[1] + 0.41),
            color=TEAL,
        )
    add_arrow(
        ax,
        (1.42, 5.2),
        (5.64, 5.2),
        color=TEAL,
        rad=-0.22,
        linewidth=1.4,
    )
    ax.text(
        3.55,
        5.48,
        "Core Value input",
        ha="center",
        fontsize=8.8,
        color=TEAL,
    )

    add_box(
        ax,
        x=0.46,
        y=2.92,
        width=1.9,
        height=0.74,
        text="React Experience\ninput + display",
        facecolor=TEAL_LIGHT,
        edgecolor=TEAL,
        fontsize=9.5,
    )
    add_box(
        ax,
        x=4.32,
        y=2.92,
        width=2.65,
        height=0.74,
        text="Inspect\nshared session + trace events",
        facecolor=GOLD_LIGHT,
        edgecolor=GOLD,
        fontsize=9.5,
    )
    add_box(
        ax,
        x=9.45,
        y=2.92,
        width=1.9,
        height=0.74,
        text="Coach Digest\ncited reflection",
        facecolor=TEAL_LIGHT,
        edgecolor=TEAL,
        fontsize=9.5,
    )
    add_arrow(ax, (1.42, 3.66), (1.12, 4.38), color=TEAL)
    add_arrow(ax, (1.67, 3.66), (2.7, 4.38), color=TEAL)
    add_arrow(ax, (10.4, 4.38), (10.4, 3.66), color=TEAL)
    add_arrow(
        ax,
        (3.18, 4.38),
        (5.2, 3.66),
        color=GOLD,
        linestyle="dashed",
    )
    add_arrow(
        ax,
        (10.4, 4.38),
        (6.2, 3.66),
        color=GOLD,
        linestyle="dashed",
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.12, 0.16),
            11.96,
            1.88,
            boxstyle="round,pad=0.02,rounding_size=0.16",
            facecolor=BLUE_LIGHT,
            edgecolor=BLUE,
            linewidth=1.2,
        )
    )
    ax.text(
        0.42,
        1.76,
        "SEPARATE OFFLINE RESEARCH PATH · NO USER-FACING DRIFT AUTHORITY",
        color=BLUE,
        fontsize=10,
        weight="bold",
    )
    offline_nodes = [
        (0.5, 0.56, 2.0, "Synthetic Journal Entries"),
        (3.03, 0.56, 2.12, "LLM-Judge VIF Labels"),
        (5.68, 0.56, 2.0, "VIF Critic (Offline)"),
        (8.22, 0.56, 3.12, "VIF Critic Predictions\n+ experiment reports"),
    ]
    for x, y, width, label in offline_nodes:
        add_box(
            ax,
            x=x,
            y=y,
            width=width,
            height=0.72,
            text=label,
            facecolor="white",
            edgecolor=BLUE,
            fontsize=9.6,
        )
    for left, right in zip(offline_nodes, offline_nodes[1:], strict=False):
        add_arrow(
            ax,
            (left[0] + left[2], left[1] + 0.36),
            (right[0], right[1] + 0.36),
            color=BLUE,
        )

    fig.text(
        0.5,
        0.01,
        (
            "Solid arrows show processing; dashed arrows show Inspect access "
            "to the shared trace."
        ),
        ha="center",
        color=MUTED,
        fontsize=9.7,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.07)
    save_figure(fig, "adopted-architecture.png")


def draw_drift_detector_transitions() -> None:
    """Render the exact state and substate path in the Drift Detector."""
    fig, ax = plt.subplots(figsize=(12.2, 6.2))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    nodes = {
        "n0": (0.45, 4.05, 2.05, "No Active Drift\nrun length 0"),
        "n1": (3.25, 4.05, 2.05, "No Active Drift\nrun length 1 · first Conflict"),
        "active": (8.95, 4.05, 2.45, "Active Drift\nrun length ≥ 2"),
        "i0": (1.85, 1.55, 2.45, "Insufficient Evidence\nrun length 0 · unresolved"),
        "i1": (5.45, 1.55, 2.45, "Insufficient Evidence\nrun length 1 · unresolved"),
    }
    styles = {
        "n0": (TEAL_LIGHT, TEAL),
        "n1": (TEAL_LIGHT, TEAL),
        "active": (GOLD_LIGHT, GOLD),
        "i0": (CORAL_LIGHT, CORAL),
        "i1": (CORAL_LIGHT, CORAL),
    }
    for key, (x, y, width, label) in nodes.items():
        facecolor, edgecolor = styles[key]
        add_box(
            ax,
            x=x,
            y=y,
            width=width,
            height=0.9,
            text=label,
            facecolor=facecolor,
            edgecolor=edgecolor,
            fontsize=10.2,
            linewidth=1.6,
        )

    add_arrow(ax, (2.5, 4.5), (3.25, 4.5), color=TEAL)
    ax.text(2.87, 4.72, "valid Conflict", ha="center", fontsize=9.1)
    add_arrow(ax, (5.3, 4.5), (8.95, 4.5), color=GOLD)
    ax.text(
        7.12,
        4.72,
        "adjacent valid Conflict",
        ha="center",
        fontsize=9.1,
    )
    add_arrow(ax, (4.3, 2.0), (5.45, 2.0), color=CORAL)
    ax.text(4.87, 2.22, "valid Conflict", ha="center", fontsize=9.1)
    add_arrow(ax, (7.9, 2.0), (9.55, 4.05), color=GOLD, rad=-0.13)
    ax.text(
        8.68,
        2.9,
        "adjacent valid Conflict",
        ha="center",
        fontsize=9.1,
        rotation=47,
    )

    add_arrow(ax, (0.78, 4.95), (2.18, 4.95), color=TEAL, rad=-0.55)
    ax.text(
        1.48,
        5.72,
        "valid Not Conflict or\nstandalone valid Abstain",
        ha="center",
        fontsize=8.9,
    )
    add_arrow(ax, (9.35, 4.95), (11.0, 4.95), color=GOLD, rad=-0.55)
    ax.text(
        10.18,
        5.72,
        "adjacent valid Conflict\nextends the run",
        ha="center",
        fontsize=8.9,
    )

    add_arrow(ax, (1.5, 4.05), (2.55, 2.45), color=CORAL, rad=0.08)
    ax.text(1.55, 3.05, "failed review", fontsize=8.8, rotation=-47)
    add_arrow(ax, (4.25, 4.05), (3.55, 2.45), color=CORAL, rad=-0.05)
    ax.text(
        4.25,
        3.03,
        "valid Abstain\nor failed review",
        ha="center",
        fontsize=8.8,
    )
    add_arrow(ax, (9.15, 4.05), (4.3, 2.18), color=CORAL, rad=-0.1)
    ax.text(
        6.85,
        3.05,
        "valid Abstain or failed review",
        ha="center",
        fontsize=8.8,
        rotation=18,
    )
    add_arrow(ax, (1.85, 1.72), (1.85, 2.25), color=CORAL, rad=-1.2)
    ax.text(
        0.3,
        1.75,
        "valid Abstain\nor failed review",
        ha="left",
        fontsize=8.7,
    )
    add_arrow(ax, (5.45, 1.78), (4.3, 1.78), color=CORAL)
    ax.text(
        4.88,
        1.25,
        "valid Abstain\nor failed review",
        ha="center",
        fontsize=8.7,
    )

    ax.text(
        0.5,
        0.66,
        (
            "GLOBAL RESET · A valid Not Conflict returns any node to No Active "
            "Drift, run length 0."
        ),
        fontsize=9.3,
        color=INK,
        weight="bold",
    )
    ax.text(
        0.5,
        0.34,
        (
            "GAP RULE · A Journal Entry gap breaks adjacency. After recent Conflict "
            "or unresolved evidence, a next Conflict enters the unresolved "
            "run-length-1 node."
        ),
        fontsize=9.1,
        color=MUTED,
    )
    ax.text(
        0.5,
        0.07,
        (
            "HISTORY · Historical Drift Records remain stored after the current "
            "state changes."
        ),
        fontsize=9.1,
        color=MUTED,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.04)
    save_figure(fig, "drift-detector-transitions.png")


def parse_float(value: str) -> float:
    """Parse a decimal value from one Markdown table cell."""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace("`", ""))
    if match is None:
        raise ValueError(f"No number in {value!r}")
    return float(match.group())


def markdown_rows(path: Path, header_start: str) -> list[list[str]]:
    """Read the first Markdown table below a selected header row."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(
        index
        for index, line in enumerate(lines)
        if line.strip().startswith(header_start)
    )
    rows: list[list[str]] = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one committed YAML evidence file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return data


def draw_synthetic_data_lifts() -> None:
    """Show corpus growth and the bounded effects of two targeted data lifts."""
    first_snapshot = load_yaml(
        ROOT / "config" / "experiments" / "vif" / "twinkl_681_5_baseline_snapshot.yaml"
    )
    second_snapshot = load_yaml(
        ROOT / "config" / "experiments" / "vif" / "twinkl_691_2_baseline_snapshot.yaml"
    )
    second_config = load_yaml(
        ROOT / "config" / "experiments" / "vif" / "twinkl_691_2.yaml"
    )
    reference_run = load_yaml(
        ROOT / "logs" / "experiments" / "runs" / "run_019_BalancedSoftmax.yaml"
    )
    first_report = (
        ROOT
        / "logs"
        / "experiments"
        / "reports"
        / "experiment_review_2026-03-08_twinkl_681_5.md"
    )
    second_report = (
        ROOT
        / "logs"
        / "experiments"
        / "reports"
        / "experiment_review_2026-03-09_twinkl_691_3.md"
    )

    baseline_entries = sum(reference_run["data"].values())
    first_text = first_report.read_text(encoding="utf-8")
    first_added_match = re.search(r"(\d+) new entries", first_text)
    if first_added_match is None:
        raise ValueError("Could not read the first targeted-batch entry count")
    first_added_entries = int(first_added_match.group(1))
    final_entries = pl.read_parquet(
        ROOT / "logs" / "judge_labels" / "judge_labels.parquet"
    ).height

    personas = [
        int(first_snapshot["registry_persona_count"]),
        int(second_snapshot["registry_persona_count"]),
        int(second_snapshot["registry_persona_count"])
        + int(second_config["generation"]["num_personas"]),
    ]
    entries = [baseline_entries, baseline_entries + first_added_entries, final_entries]
    stages = [
        "Corrected-split\nbaseline",
        "Power/Security\nlift",
        "Hedonism/Security\nlift",
    ]

    first_rows = markdown_rows(first_report, "| Dimension | Baseline QWK")
    first_by_dimension = {row[0].replace("`", ""): row for row in first_rows}
    second_rows = markdown_rows(second_report, "| Family | `hedonism qwk`")
    second_by_family = {row[0]: row for row in second_rows}

    effect_labels = [
        "Power/Security batch\nPower Conflict recall",
        "Power/Security batch\nSecurity Conflict recall",
        "Hedonism/Security batch\nHedonism QWK",
        "Hedonism/Security batch\nSecurity QWK",
    ]
    before = [
        parse_float(first_by_dimension["power"][4]),
        parse_float(first_by_dimension["security"][4]),
        parse_float(second_by_family["Current default BalancedSoftmax"][1]),
        parse_float(second_by_family["Current default BalancedSoftmax"][4]),
    ]
    after = [
        parse_float(first_by_dimension["power"][5]),
        parse_float(first_by_dimension["security"][5]),
        parse_float(second_by_family["New BalancedSoftmax"][1]),
        parse_float(second_by_family["New BalancedSoftmax"][4]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8), gridspec_kw={"wspace": 0.34})
    x = list(range(3))
    axes[0].plot(x, entries, color=TEAL, linewidth=2.5, zorder=2)
    axes[0].scatter(x, entries, s=100, color=TEAL, zorder=3)
    annotation_layout = [(18, "left"), (0, "center"), (-10, "right")]
    for index, (entry_count, persona_count) in enumerate(
        zip(entries, personas, strict=True)
    ):
        horizontal_offset, alignment = annotation_layout[index]
        axes[0].annotate(
            f"{entry_count:,} entries\n{persona_count} personas",
            (index, entry_count),
            xytext=(horizontal_offset, 14),
            textcoords="offset points",
            ha=alignment,
            weight="bold",
        )
    axes[0].set_xticks(x, stages)
    axes[0].set_ylim(1390, 1725)
    axes[0].set_ylabel("Persisted Journal Entries")
    axes[0].set_title("(a) Train-only corpus growth", fontsize=13, weight="bold")
    axes[0].yaxis.grid(True, color=GRID, linewidth=0.8)
    axes[0].set_axisbelow(True)

    y = list(range(len(effect_labels)))
    for index, (start, end) in enumerate(zip(before, after, strict=True)):
        end_color = TEAL if end >= start else CORAL
        axes[1].plot([start, end], [index, index], color=GRID, linewidth=2.2, zorder=1)
        axes[1].scatter(start, index, s=66, color=MUTED, marker="o", zorder=2)
        axes[1].scatter(end, index, s=78, color=end_color, marker="D", zorder=3)
        if start == end:
            axes[1].annotate(
                f"{end:.3f} unchanged",
                (end, index),
                xytext=(8, 8),
                textcoords="offset points",
                ha="left",
                fontsize=8.5,
                color=end_color,
                weight="bold",
            )
            continue
        axes[1].annotate(
            f"{start:.3f}",
            (start, index),
            xytext=(-7, -14),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            color=MUTED,
        )
        axes[1].annotate(
            f"{end:.3f}",
            (end, index),
            xytext=(7, 8),
            textcoords="offset points",
            ha="left",
            fontsize=8.5,
            color=end_color,
            weight="bold",
        )
    axes[1].set_yticks(y, effect_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 0.66)
    axes[1].set_xlabel("Held-out metric value")
    axes[1].set_title(
        "(b) Selected target-dimension effects", fontsize=13, weight="bold"
    )
    axes[1].xaxis.grid(True, color=GRID, linewidth=0.8)
    axes[1].set_axisbelow(True)

    for ax in axes:
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.text(
        0.56,
        0.012,
        (
            "Circles show the pre-lift reference; diamonds show the post-lift "
            "result. Compare each row only within its named metric."
        ),
        color=MUTED,
        fontsize=9.7,
    )
    fig.subplots_adjust(bottom=0.20, top=0.90)
    save_figure(fig, "synthetic-data-lifts.png")


def vif_conflict_recall() -> tuple[list[str], list[float], list[int]]:
    """Compute per-value median Conflict recall across the three VIF seeds."""
    run_paths = [
        ROOT / "logs" / "experiments" / "runs" / f"run_{run}_BalancedSoftmax.yaml"
        for run in ("019", "020", "021")
    ]
    per_run: list[dict[str, float]] = []
    support_by_dimension: dict[str, int] = {}
    for run_path in run_paths:
        run = load_yaml(run_path)
        outputs = pl.read_parquet(ROOT / run["artifacts"]["test_outputs"])
        run_recalls: dict[str, float] = {}
        for dimension in outputs.get_column("dimension").unique().sort().to_list():
            rows = outputs.filter(pl.col("dimension") == dimension)
            support = rows.filter(pl.col("target") == -1).height
            true_positives = rows.filter(
                (pl.col("target") == -1) & (pl.col("predicted_class") == -1)
            ).height
            run_recalls[dimension] = true_positives / support
            support_by_dimension[dimension] = support
        per_run.append(run_recalls)

    dimensions = [
        "self_direction",
        "stimulation",
        "hedonism",
        "achievement",
        "power",
        "security",
        "conformity",
        "tradition",
        "benevolence",
        "universalism",
    ]
    median_recalls = [
        median(run[dimension] for run in per_run) for dimension in dimensions
    ]
    supports = [support_by_dimension[dimension] for dimension in dimensions]
    return dimensions, median_recalls, supports


def weekly_reviewer_conflict_recall() -> tuple[list[str], list[float], list[int]]:
    """Read per-value Luna-low entry Conflict recall from saved metrics."""
    source = (
        ROOT
        / "logs"
        / "experiments"
        / "artifacts"
        / "twinkl_52zz_luna_low_20260714"
        / "metrics.json"
    )
    metrics = json.loads(source.read_text(encoding="utf-8"))
    results = metrics["models"]["luna_low"]["results"]
    dimensions = [
        "self_direction",
        "stimulation",
        "hedonism",
        "achievement",
        "power",
        "security",
        "conformity",
        "tradition",
        "benevolence",
        "universalism",
    ]
    recalls = [
        median(
            result["entry"]["per_dimension"][dimension]["recall"]
            for result in results
        )
        for dimension in dimensions
    ]
    supports = [
        int(results[0]["entry"]["per_dimension"][dimension]["negative_support"])
        for dimension in dimensions
    ]
    return dimensions, recalls, supports


def draw_per_value_conflict_recall() -> None:
    """Contrast per-value difficulty without implying a direct model comparison."""
    vif_dimensions, vif_recall, vif_support = vif_conflict_recall()
    weekly_dimensions, weekly_recall, weekly_support = weekly_reviewer_conflict_recall()
    if vif_dimensions != weekly_dimensions:
        raise ValueError("Per-value evidence uses inconsistent value order")

    labels = [dimension.replace("_", " ").title() for dimension in vif_dimensions]
    labels[0] = "Self-Direction"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.2, 7.0),
        sharey=True,
        gridspec_kw={"wspace": 0.26},
    )
    y = list(range(len(labels)))
    panels = (
        (
            axes[0],
            vif_recall,
            vif_support,
            "(a) VIF Critic (Offline)",
            "LLM-Judge VIF Labels · frozen 221-entry test set",
            TEAL,
        ),
        (
            axes[1],
            weekly_recall,
            weekly_support,
            "(b) Weekly Drift Reviewer · Luna-low",
            "LLM-Judge Conflict Labels · complete development data",
            BLUE,
        ),
    )
    for ax, values, supports, title, subtitle, color in panels:
        ax.hlines(y, 0, values, color=GRID, linewidth=2.0, zorder=1)
        ax.scatter(values, y, s=82, color=color, zorder=3)
        for index, (value, support) in enumerate(zip(values, supports, strict=True)):
            ax.annotate(
                f"{value:.3f} · n={support}",
                (value, index),
                xytext=(7, -1),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=color,
                weight="bold",
            )
        ax.set_xlim(0, 0.92)
        ax.set_xlabel("Median entry-level Conflict recall")
        ax.set_title(f"{title}\n{subtitle}", fontsize=11.5, weight="bold")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.text(
        0.5,
        0.018,
        (
            "The panels use different data, label rubrics, and model inputs. "
            "Compare value patterns within each panel, not scores across panels."
        ),
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    fig.subplots_adjust(bottom=0.13, top=0.88)
    save_figure(fig, "per-value-conflict-recall.png")


def draw_label_agreement() -> None:
    """Compare human-human and LLM-Judge-human agreement by dimension."""
    source = ROOT / "docs" / "evals" / "judge_validation_summary.md"
    rows = markdown_rows(source, "| Value Dimension")[:10]
    dimensions = [row[0] for row in rows]
    human = [parse_float(row[1]) for row in rows]
    judge = [parse_float(row[2]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    y = list(range(len(dimensions)))
    human_y = [index - 0.14 for index in y]
    judge_y = [index + 0.14 for index in y]
    ax.scatter(
        human,
        human_y,
        s=70,
        color=MUTED,
        label="Human-human Fleiss' κ",
        zorder=2,
    )
    ax.scatter(
        judge,
        judge_y,
        s=76,
        color=TEAL,
        label="Mean LLM-Judge-human Cohen's κ",
        zorder=3,
    )
    for value, position in zip(human, human_y, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (value, position),
            xytext=(-7, -1),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=8.3,
            color=MUTED,
        )
    for value, position in zip(judge, judge_y, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (value, position),
            xytext=(7, -1),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.3,
            color=TEAL,
        )
    ax.set_yticks(y, dimensions)
    ax.invert_yaxis()
    ax.set_xlim(0.25, 0.9)
    ax.set_xlabel("Chance-corrected agreement (κ)")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        frameon=False,
        ncol=2,
        fontsize=9.5,
        borderaxespad=0,
    )
    fig.subplots_adjust(top=0.91)
    save_figure(fig, "label-agreement.png")


def draw_vif_handoff() -> None:
    """Show the VIF Critic input ablation without mixing unrelated metrics."""
    source = (
        ROOT
        / "logs"
        / "experiments"
        / "reports"
        / "experiment_review_2026-07-14_twinkl_752_5_reassessment.md"
    )
    rows = markdown_rows(source, "| Setup")[:3]
    labels = [
        "Without VIF Critic\ninput",
        "With VIF Critic\nPredictions",
        "Early + weekly\nVIF triggers",
    ]
    recall = [parse_float(row[2]) for row in rows]
    false_alerts = [parse_float(row[5]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), gridspec_kw={"wspace": 0.28})
    x = list(range(3))
    colors = [TEAL, CORAL, GOLD]
    axes[0].bar(x, recall, color=colors, width=0.62)
    axes[0].set_ylim(0, 0.34)
    axes[0].set_ylabel("Median Drift recall")
    axes[0].set_title("(a) Drift recall", fontsize=13, weight="bold")
    axes[1].bar(x, false_alerts, color=colors, width=0.62)
    axes[1].set_ylim(0, 4)
    axes[1].set_ylabel("Median false Drift alerts")
    axes[1].set_title("(b) False Drift alerts", fontsize=13, weight="bold")

    for ax, values, decimals in ((axes[0], recall, 3), (axes[1], false_alerts, 0)):
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=9.2)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        for index, value in enumerate(values):
            label = f"{value:.{decimals}f}" if decimals else f"{value:.0f}"
            ax.text(
                index,
                value + (0.012 if decimals else 0.12),
                label,
                ha="center",
                weight="bold",
            )

    fig.text(
        0.06,
        0.01,
        (
            "gpt-5.4-mini-2026-03-17 at reasoning none; 33 known Drifts across "
            "106 cases; medians across three repeats."
        ),
        color=MUTED,
        fontsize=10.5,
    )
    fig.subplots_adjust(top=0.91, bottom=0.22)
    save_figure(fig, "vif-handoff-ablation.png")


def draw_weekly_drift_tradeoff() -> None:
    """Show the selected Weekly Drift Reviewer operating point."""
    source = (
        ROOT
        / "logs"
        / "experiments"
        / "reports"
        / "experiment_review_2026-08-09_twinkl_ck3w_luna_higher_reasoning.md"
    )
    rows = markdown_rows(source, "| Reasoning effort")[:5]
    effort = [row[0].replace("`", "") for row in rows]
    recall = [parse_float(row[2]) for row in rows]
    false_alerts = [parse_float(row[3]) for row in rows]
    coverage = [parse_float(row[5]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = [MUTED, TEAL, BLUE, GOLD, CORAL]
    for index, label in enumerate(effort):
        ax.scatter(
            false_alerts[index],
            recall[index],
            s=230,
            color=colors[index],
            edgecolor="white",
            linewidth=1.3,
            zorder=3,
        )
        offset = {
            "none": (-76, -31),
            "low": (13, -38),
            "medium": (13, 10),
            "high": (13, -18),
            "xhigh": (13, 8),
        }[label]
        suffix = " · selected" if label == "low" else ""
        ax.annotate(
            f"{label}{suffix}\ncoverage {coverage[index]:.3f}",
            (false_alerts[index], recall[index]),
            xytext=offset,
            textcoords="offset points",
            fontsize=10.5,
            weight="bold" if label == "low" else "normal",
            arrowprops=(
                {"arrowstyle": "-", "color": TEAL, "linewidth": 0.8}
                if label == "low"
                else None
            ),
        )

    ax.set_xlim(2, 14.5)
    ax.set_ylim(0.43, 0.70)
    ax.set_xlabel("Median false Drift alerts (lower is better)")
    ax.set_ylabel("Median Drift recall (higher is better)")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    fig.subplots_adjust(top=0.95)
    save_figure(fig, "weekly-drift-tradeoff.png")


def main() -> None:
    """Generate all static figures used by the capstone paper."""
    configure_matplotlib()
    draw_adopted_architecture()
    draw_drift_detector_transitions()
    draw_synthetic_data_lifts()
    draw_label_agreement()
    draw_per_value_conflict_recall()
    draw_vif_handoff()
    draw_weekly_drift_tradeoff()


if __name__ == "__main__":
    main()
