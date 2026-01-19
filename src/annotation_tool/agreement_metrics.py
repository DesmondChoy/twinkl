"""Agreement metrics for comparing human annotations vs judge labels.

This module provides inter-rater agreement calculations using Cohen's κ (two raters)
and Fleiss' κ (multiple raters), along with export functionality.

Agreement interpretation follows Landis & Koch (1977):
    κ < 0.00: Poor
    0.00-0.20: Slight
    0.21-0.40: Fair
    0.41-0.60: Moderate
    0.61-0.80: Substantial
    0.81-1.00: Almost Perfect

Usage:
    from src.annotation_tool.agreement_metrics import (
        calculate_cohen_kappa,
        calculate_fleiss_kappa,
        interpret_kappa,
        load_all_annotator_dfs,
    )

    # Load annotations
    annotator_dfs = load_all_annotator_dfs()

    # Cohen's kappa: one annotator vs judge
    kappa_scores = calculate_cohen_kappa(annotator_dfs["alice"], judge_df)

    # Fleiss' kappa: multiple annotators
    fleiss_scores = calculate_fleiss_kappa(list(annotator_dfs.values()))
"""

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from sklearn.metrics import cohen_kappa_score

from src.annotation_tool.annotation_store import ANNOTATIONS_DIR
from src.models.judge import SCHWARTZ_VALUE_ORDER


def interpret_kappa(kappa: float) -> str:
    """Interpret kappa value using Landis & Koch (1977) scale.

    Args:
        kappa: Kappa coefficient value

    Returns:
        String interpretation of the agreement level
    """
    if kappa < 0.00:
        return "Poor"
    elif kappa < 0.21:
        return "Slight"
    elif kappa < 0.41:
        return "Fair"
    elif kappa < 0.61:
        return "Moderate"
    elif kappa < 0.81:
        return "Substantial"
    else:
        return "Almost Perfect"


def load_all_annotator_dfs() -> dict[str, pl.DataFrame]:
    """Load all annotator parquet files.

    Returns:
        Dict mapping annotator_id to their annotation DataFrame
    """
    annotator_dfs = {}

    if not ANNOTATIONS_DIR.exists():
        return annotator_dfs

    for parquet_file in ANNOTATIONS_DIR.glob("*.parquet"):
        # Skip lock files
        if parquet_file.suffix == ".lock":
            continue

        annotator_id = parquet_file.stem
        try:
            df = pl.read_parquet(parquet_file)
            if len(df) > 0:
                annotator_dfs[annotator_id] = df
        except Exception:
            # Skip files that can't be read
            continue

    return annotator_dfs


def load_judge_labels() -> pl.DataFrame:
    """Load judge labels from parquet file.

    Returns:
        DataFrame with judge labels, or empty DataFrame if not found
    """
    labels_path = Path("logs/judge_labels/judge_labels.parquet")

    if not labels_path.exists():
        return pl.DataFrame()

    return pl.read_parquet(labels_path)


def calculate_cohen_kappa(
    human_df: pl.DataFrame,
    judge_df: pl.DataFrame | None = None,
) -> dict[str, float]:
    """Calculate Cohen's kappa between human annotations and judge labels.

    Performs inner join on (persona_id, t_index) to compare only paired entries.

    Args:
        human_df: Human annotator DataFrame with alignment_* columns
        judge_df: Judge labels DataFrame. If None, loads from default path.

    Returns:
        Dict mapping value names to kappa scores, plus "aggregate" for overall
    """
    if judge_df is None:
        judge_df = load_judge_labels()

    if len(human_df) == 0 or len(judge_df) == 0:
        return {value: float("nan") for value in SCHWARTZ_VALUE_ORDER + ["aggregate"]}

    # Inner join on (persona_id, t_index)
    merged = human_df.join(
        judge_df.select(
            ["persona_id", "t_index"]
            + [f"alignment_{v}" for v in SCHWARTZ_VALUE_ORDER]
        ),
        on=["persona_id", "t_index"],
        how="inner",
        suffix="_judge",
    )

    if len(merged) == 0:
        return {value: float("nan") for value in SCHWARTZ_VALUE_ORDER + ["aggregate"]}

    kappa_scores = {}
    all_human_scores = []
    all_judge_scores = []

    for value in SCHWARTZ_VALUE_ORDER:
        human_col = f"alignment_{value}"
        judge_col = f"alignment_{value}_judge"

        human_scores = merged[human_col].to_list()
        judge_scores = merged[judge_col].to_list()

        all_human_scores.extend(human_scores)
        all_judge_scores.extend(judge_scores)

        # Calculate kappa with explicit labels to handle missing classes
        try:
            kappa = cohen_kappa_score(
                human_scores,
                judge_scores,
                labels=[-1, 0, 1],
            )
            kappa_scores[value] = round(kappa, 3)
        except Exception:
            kappa_scores[value] = float("nan")

    # Aggregate kappa across all values
    try:
        aggregate_kappa = cohen_kappa_score(
            all_human_scores,
            all_judge_scores,
            labels=[-1, 0, 1],
        )
        kappa_scores["aggregate"] = round(aggregate_kappa, 3)
    except Exception:
        kappa_scores["aggregate"] = float("nan")

    return kappa_scores


def calculate_fleiss_kappa(
    annotator_dfs: list[pl.DataFrame],
) -> dict[str, float]:
    """Calculate Fleiss' kappa for multi-annotator agreement.

    Only considers entries that ALL annotators have labeled (intersection).

    Args:
        annotator_dfs: List of annotation DataFrames from different annotators

    Returns:
        Dict mapping value names to Fleiss kappa scores, plus "aggregate"
        Also includes "n_shared" for the number of shared entries
    """
    from statsmodels.stats.inter_rater import fleiss_kappa

    if len(annotator_dfs) < 2:
        result = {value: float("nan") for value in SCHWARTZ_VALUE_ORDER}
        result["aggregate"] = float("nan")
        result["n_shared"] = 0
        return result

    # Find shared entries (intersection of all annotators)
    shared_keys = None
    for df in annotator_dfs:
        keys = set(
            (row["persona_id"], row["t_index"]) for row in df.to_dicts()
        )
        if shared_keys is None:
            shared_keys = keys
        else:
            shared_keys &= keys

    if not shared_keys:
        result = {value: float("nan") for value in SCHWARTZ_VALUE_ORDER}
        result["aggregate"] = float("nan")
        result["n_shared"] = 0
        return result

    n_shared = len(shared_keys)

    # Build rating matrices for each value
    kappa_scores = {"n_shared": n_shared}

    for value in SCHWARTZ_VALUE_ORDER:
        # Build matrix: rows = items, columns = category counts (3 categories: -1, 0, 1)
        # For Fleiss' kappa, we need counts of how many raters assigned each category
        rating_matrix = []

        for persona_id, t_index in sorted(shared_keys):
            counts = {-1: 0, 0: 0, 1: 0}
            for df in annotator_dfs:
                row = df.filter(
                    (pl.col("persona_id") == persona_id)
                    & (pl.col("t_index") == t_index)
                ).to_dicts()
                if row:
                    score = row[0][f"alignment_{value}"]
                    counts[score] += 1
            rating_matrix.append([counts[-1], counts[0], counts[1]])

        try:
            kappa = fleiss_kappa(rating_matrix)
            kappa_scores[value] = round(kappa, 3)
        except Exception:
            kappa_scores[value] = float("nan")

    # Aggregate: compute Fleiss kappa treating all values as separate items
    aggregate_matrix = []
    for value in SCHWARTZ_VALUE_ORDER:
        for persona_id, t_index in sorted(shared_keys):
            counts = {-1: 0, 0: 0, 1: 0}
            for df in annotator_dfs:
                row = df.filter(
                    (pl.col("persona_id") == persona_id)
                    & (pl.col("t_index") == t_index)
                ).to_dicts()
                if row:
                    score = row[0][f"alignment_{value}"]
                    counts[score] += 1
            aggregate_matrix.append([counts[-1], counts[0], counts[1]])

    try:
        aggregate_kappa = fleiss_kappa(aggregate_matrix)
        kappa_scores["aggregate"] = round(aggregate_kappa, 3)
    except Exception:
        kappa_scores["aggregate"] = float("nan")

    return kappa_scores


def get_per_value_agreement(
    human_df: pl.DataFrame,
    judge_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Get detailed per-value agreement statistics.

    Args:
        human_df: Human annotator DataFrame
        judge_df: Judge labels DataFrame. If None, loads from default path.

    Returns:
        DataFrame with columns: value, kappa, interpretation, match_count,
        total_count, match_rate
    """
    if judge_df is None:
        judge_df = load_judge_labels()

    if len(human_df) == 0 or len(judge_df) == 0:
        return pl.DataFrame()

    # Inner join
    merged = human_df.join(
        judge_df.select(
            ["persona_id", "t_index"]
            + [f"alignment_{v}" for v in SCHWARTZ_VALUE_ORDER]
        ),
        on=["persona_id", "t_index"],
        how="inner",
        suffix="_judge",
    )

    if len(merged) == 0:
        return pl.DataFrame()

    total_count = len(merged)
    rows = []

    for value in SCHWARTZ_VALUE_ORDER:
        human_col = f"alignment_{value}"
        judge_col = f"alignment_{value}_judge"

        human_scores = merged[human_col].to_list()
        judge_scores = merged[judge_col].to_list()

        # Calculate exact match count
        match_count = sum(h == j for h, j in zip(human_scores, judge_scores))

        # Calculate kappa
        try:
            kappa = cohen_kappa_score(
                human_scores,
                judge_scores,
                labels=[-1, 0, 1],
            )
            kappa = round(kappa, 3)
        except Exception:
            kappa = float("nan")

        rows.append({
            "value": value,
            "kappa": kappa,
            "interpretation": interpret_kappa(kappa) if not (kappa != kappa) else "N/A",
            "match_count": match_count,
            "total_count": total_count,
            "match_rate": round(match_count / total_count * 100, 1),
        })

    return pl.DataFrame(rows)


# =============================================================================
# Export Functions
# =============================================================================

EXPORTS_DIR = Path("logs/exports")


def _ensure_exports_dir() -> None:
    """Create exports directory if needed."""
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def export_annotations_csv(
    annotator_id: str,
    output_path: Path | None = None,
) -> Path:
    """Export a single annotator's annotations to CSV.

    Args:
        annotator_id: Annotator identifier
        output_path: Custom output path. If None, uses default location.

    Returns:
        Path to the created CSV file
    """
    from src.annotation_tool.annotation_store import load_annotations

    _ensure_exports_dir()

    df = load_annotations(annotator_id)

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = EXPORTS_DIR / f"{annotator_id}_annotations_{timestamp}.csv"

    df.write_csv(output_path)
    return output_path


def export_annotations_parquet(
    annotator_id: str,
    output_path: Path | None = None,
) -> Path:
    """Export a single annotator's annotations to Parquet.

    Args:
        annotator_id: Annotator identifier
        output_path: Custom output path. If None, uses default location.

    Returns:
        Path to the created Parquet file
    """
    from src.annotation_tool.annotation_store import load_annotations

    _ensure_exports_dir()

    df = load_annotations(annotator_id)

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = EXPORTS_DIR / f"{annotator_id}_annotations_{timestamp}.parquet"

    df.write_parquet(output_path)
    return output_path


def export_combined_annotations(
    output_path: Path | None = None,
) -> Path:
    """Export all annotators' annotations to a single Parquet file.

    Args:
        output_path: Custom output path. If None, uses default location.

    Returns:
        Path to the created Parquet file
    """
    _ensure_exports_dir()

    annotator_dfs = load_all_annotator_dfs()

    if not annotator_dfs:
        raise ValueError("No annotations found to export")

    combined = pl.concat(list(annotator_dfs.values()))

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = EXPORTS_DIR / f"all_annotations_{timestamp}.parquet"

    combined.write_parquet(output_path)
    return output_path


def generate_agreement_report(
    output_path: Path | None = None,
) -> Path:
    """Generate a Markdown report of agreement metrics.

    Args:
        output_path: Custom output path. If None, uses default location.

    Returns:
        Path to the created Markdown file
    """
    _ensure_exports_dir()

    annotator_dfs = load_all_annotator_dfs()
    judge_df = load_judge_labels()

    lines = [
        "# Agreement Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
    ]

    # Annotator summary
    lines.append("## Annotator Summary")
    lines.append("")
    lines.append("| Annotator | Entries Labeled |")
    lines.append("|-----------|----------------|")
    for annotator_id, df in sorted(annotator_dfs.items()):
        lines.append(f"| {annotator_id} | {len(df)} |")
    lines.append("")

    # Cohen's kappa table
    if len(judge_df) > 0 and annotator_dfs:
        lines.append("## Cohen's κ: Annotator vs Judge")
        lines.append("")

        # Header row
        header = "| Value |"
        separator = "|-------|"
        for annotator_id in sorted(annotator_dfs.keys()):
            header += f" {annotator_id} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        # Data rows
        for value in SCHWARTZ_VALUE_ORDER:
            row = f"| {value.replace('_', ' ').title()} |"
            for annotator_id in sorted(annotator_dfs.keys()):
                kappa_scores = calculate_cohen_kappa(annotator_dfs[annotator_id], judge_df)
                kappa = kappa_scores.get(value, float("nan"))
                if kappa != kappa:  # NaN check
                    row += " N/A |"
                else:
                    row += f" {kappa:.2f} |"
            lines.append(row)

        # Aggregate row
        row = "| **Aggregate** |"
        for annotator_id in sorted(annotator_dfs.keys()):
            kappa_scores = calculate_cohen_kappa(annotator_dfs[annotator_id], judge_df)
            kappa = kappa_scores.get("aggregate", float("nan"))
            if kappa != kappa:
                row += " **N/A** |"
            else:
                row += f" **{kappa:.2f}** |"
        lines.append(row)
        lines.append("")

    # Fleiss' kappa
    if len(annotator_dfs) >= 2:
        lines.append("## Fleiss' κ: Inter-Annotator Agreement")
        lines.append("")
        fleiss_scores = calculate_fleiss_kappa(list(annotator_dfs.values()))
        n_shared = fleiss_scores.get("n_shared", 0)

        lines.append(f"**Shared Entries:** {n_shared}")
        lines.append("")

        if n_shared > 0:
            lines.append("| Value | κ | Interpretation |")
            lines.append("|-------|---|----------------|")

            for value in SCHWARTZ_VALUE_ORDER:
                kappa = fleiss_scores.get(value, float("nan"))
                if kappa != kappa:
                    lines.append(f"| {value.replace('_', ' ').title()} | N/A | N/A |")
                else:
                    interp = interpret_kappa(kappa)
                    lines.append(f"| {value.replace('_', ' ').title()} | {kappa:.2f} | {interp} |")

            agg_kappa = fleiss_scores.get("aggregate", float("nan"))
            if agg_kappa != agg_kappa:
                lines.append("| **Aggregate** | **N/A** | **N/A** |")
            else:
                interp = interpret_kappa(agg_kappa)
                lines.append(f"| **Aggregate** | **{agg_kappa:.2f}** | **{interp}** |")
            lines.append("")
        else:
            lines.append("*No shared entries found between annotators.*")
            lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    # Find lowest agreement values
    if annotator_dfs and len(judge_df) > 0:
        low_agreement_values = []
        for annotator_id, df in annotator_dfs.items():
            kappa_scores = calculate_cohen_kappa(df, judge_df)
            for value in SCHWARTZ_VALUE_ORDER:
                kappa = kappa_scores.get(value, float("nan"))
                if not (kappa != kappa) and kappa < 0.6:
                    low_agreement_values.append((value, kappa, annotator_id))

        if low_agreement_values:
            low_agreement_values.sort(key=lambda x: x[1])
            seen_values = set()
            for value, kappa, annotator in low_agreement_values[:3]:
                if value not in seen_values:
                    lines.append(f"- Review **{value.replace('_', ' ').title()}** rubric (lowest κ: {kappa:.2f})")
                    seen_values.add(value)
        else:
            lines.append("- All values show substantial agreement (κ ≥ 0.60)")
    else:
        lines.append("- Insufficient data for recommendations")

    lines.append("")

    # Write report
    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = EXPORTS_DIR / f"agreement_report_{timestamp}.md"

    output_path.write_text("\n".join(lines))
    return output_path
