"""Analysis view component for agreement metrics and export.

This module provides a collapsible accordion panel showing:
- Annotator progress summary
- Cohen's kappa (annotator vs judge)
- Fleiss' kappa (inter-annotator, when multiple annotators exist)
- Export buttons (CSV, Parquet, Markdown report)

Metrics are computed on-demand when the accordion is expanded to avoid
unnecessary recalculation on every save.

Usage:
    from src.annotation_tool.components import analysis_view

    # In UI definition (after the main layout)
    analysis_view.analysis_view_ui("analysis")

    # In server function
    analysis_view.analysis_view_server(
        "analysis",
        state=state,
        annotator_name=annotator_name,
    )
"""

import math

from shiny import module, reactive, render, ui

from src.annotation_tool.agreement_metrics import (
    calculate_cohen_kappa,
    calculate_fleiss_kappa,
    export_annotations_csv,
    export_annotations_parquet,
    export_combined_annotations,
    generate_agreement_report,
    get_per_value_agreement,
    interpret_kappa,
    load_all_annotator_dfs,
    load_judge_labels,
)
from src.annotation_tool.state import AppState
from src.models.judge import SCHWARTZ_VALUE_ORDER


@module.ui
def analysis_view_ui():
    """Generate the analysis view UI component.

    Returns:
        UI div containing the collapsible accordion with metrics and export buttons
    """
    return ui.div(
        ui.accordion(
            ui.accordion_panel(
                "Analysis & Metrics",
                ui.div(
                    # Annotator progress section
                    ui.output_ui("annotator_progress"),
                    # Cohen's kappa section
                    ui.output_ui("cohen_kappa_section"),
                    # Fleiss' kappa section
                    ui.output_ui("fleiss_kappa_section"),
                    # Export section
                    ui.div(
                        ui.div("Export", class_="analysis-section-header"),
                        ui.div(
                            ui.input_action_button(
                                "export_csv",
                                "CSV",
                                class_="export-btn",
                            ),
                            ui.input_action_button(
                                "export_parquet",
                                "Parquet",
                                class_="export-btn",
                            ),
                            ui.input_action_button(
                                "export_report",
                                "Markdown Report",
                                class_="export-btn export-btn-primary",
                            ),
                            class_="export-btn-group",
                        ),
                        class_="analysis-section export-section",
                    ),
                    class_="analysis-content",
                ),
                value="analysis_panel",
                icon=ui.tags.span("📊", class_="accordion-icon"),
            ),
            id="analysis_accordion",
            open=False,
            class_="analysis-accordion",
        ),
        class_="analysis-view-container",
    )


@module.server
def analysis_view_server(
    input,
    output,
    session,
    state: AppState,
    annotator_name: reactive.calc,
):
    """Server logic for the analysis view component.

    Args:
        input, output, session: Shiny module parameters
        state: Centralized app state
        annotator_name: Reactive calc returning the current annotator's name
    """

    # Cache for metrics (invalidated when annotated_count changes)
    @reactive.calc
    def cached_annotator_dfs():
        """Load all annotator DataFrames (cached per session)."""
        # Depend on annotated count to invalidate cache after saves
        _ = state.annotated_count()
        return load_all_annotator_dfs()

    @reactive.calc
    def cached_judge_df():
        """Load judge labels (cached per session)."""
        return load_judge_labels()

    @render.ui
    def annotator_progress():
        """Render annotator progress summary."""
        annotator_dfs = cached_annotator_dfs()

        if not annotator_dfs:
            return ui.div(
                ui.div("Annotator Progress", class_="analysis-section-header"),
                ui.p("No annotations yet.", class_="analysis-empty"),
                class_="analysis-section",
            )

        # Build progress chips
        progress_items = []
        for annotator_id, df in sorted(annotator_dfs.items()):
            count = len(df)
            progress_items.append(
                ui.span(
                    f"{annotator_id} ({count})",
                    class_="annotator-chip",
                )
            )

        return ui.div(
            ui.div("Annotator Progress", class_="analysis-section-header"),
            ui.div(*progress_items, class_="annotator-progress-chips"),
            class_="analysis-section",
        )

    @render.ui
    def cohen_kappa_section():
        """Render Cohen's kappa section."""
        name = annotator_name()
        if not name:
            return ui.div()

        annotator_dfs = cached_annotator_dfs()
        judge_df = cached_judge_df()

        if not annotator_dfs or name not in annotator_dfs:
            return ui.div(
                ui.div("Cohen's κ: You vs Judge", class_="analysis-section-header"),
                ui.p("No annotations to compare.", class_="analysis-empty"),
                class_="analysis-section",
            )

        if len(judge_df) == 0:
            return ui.div(
                ui.div("Cohen's κ: You vs Judge", class_="analysis-section-header"),
                ui.p("Judge labels not available.", class_="analysis-empty"),
                class_="analysis-section",
            )

        human_df = annotator_dfs[name]
        per_value_df = get_per_value_agreement(human_df, judge_df)

        if len(per_value_df) == 0:
            return ui.div(
                ui.div("Cohen's κ: You vs Judge", class_="analysis-section-header"),
                ui.p("No overlapping entries with judge labels.", class_="analysis-empty"),
                class_="analysis-section",
            )

        # Build table rows
        table_rows = []
        for row in per_value_df.to_dicts():
            value_name = row["value"].replace("_", " ").title()
            kappa = row["kappa"]
            interp = row["interpretation"]
            match_count = row["match_count"]
            total_count = row["total_count"]
            match_rate = row["match_rate"]

            # Determine color class
            if math.isnan(kappa):
                kappa_class = "kappa-na"
                kappa_display = "N/A"
            elif kappa >= 0.61:
                kappa_class = "kappa-good"
                kappa_display = f"{kappa:.2f}"
            elif kappa >= 0.41:
                kappa_class = "kappa-moderate"
                kappa_display = f"{kappa:.2f}"
            else:
                kappa_class = "kappa-poor"
                kappa_display = f"{kappa:.2f}"

            # Add warning for low agreement
            warning = " ⚠️" if kappa < 0.61 and not math.isnan(kappa) else ""

            table_rows.append(
                ui.tags.tr(
                    ui.tags.td(value_name, class_="value-name-cell"),
                    ui.tags.td(
                        ui.span(kappa_display, class_=f"kappa-badge {kappa_class}"),
                        class_="kappa-cell",
                    ),
                    ui.tags.td(f"{interp}{warning}", class_="interp-cell"),
                    ui.tags.td(
                        f"{match_rate:.0f}% ({match_count}/{total_count})",
                        class_="match-cell",
                    ),
                )
            )

        # Calculate aggregate
        kappa_scores = calculate_cohen_kappa(human_df, judge_df)
        agg_kappa = kappa_scores.get("aggregate", float("nan"))
        agg_interp = interpret_kappa(agg_kappa) if not math.isnan(agg_kappa) else "N/A"

        if math.isnan(agg_kappa):
            agg_class = "kappa-na"
            agg_display = "N/A"
        elif agg_kappa >= 0.61:
            agg_class = "kappa-good"
            agg_display = f"{agg_kappa:.2f}"
        elif agg_kappa >= 0.41:
            agg_class = "kappa-moderate"
            agg_display = f"{agg_kappa:.2f}"
        else:
            agg_class = "kappa-poor"
            agg_display = f"{agg_kappa:.2f}"

        return ui.div(
            ui.div("Cohen's κ: You vs Judge", class_="analysis-section-header"),
            ui.div(
                ui.span("Aggregate: ", class_="aggregate-label"),
                ui.span(agg_display, class_=f"kappa-badge kappa-badge-large {agg_class}"),
                ui.span(f" ({agg_interp})", class_="aggregate-interp"),
                class_="aggregate-summary",
            ),
            ui.tags.table(
                ui.tags.thead(
                    ui.tags.tr(
                        ui.tags.th("Value"),
                        ui.tags.th("κ"),
                        ui.tags.th("Interpretation"),
                        ui.tags.th("Match Rate"),
                    )
                ),
                ui.tags.tbody(*table_rows),
                class_="kappa-table",
            ),
            class_="analysis-section",
        )

    @render.ui
    def fleiss_kappa_section():
        """Render Fleiss' kappa section for multi-annotator agreement."""
        annotator_dfs = cached_annotator_dfs()

        if len(annotator_dfs) < 2:
            return ui.div(
                ui.div("Fleiss' κ: Inter-Annotator", class_="analysis-section-header"),
                ui.p(
                    "Requires 2+ annotators for inter-rater agreement.",
                    class_="analysis-empty",
                ),
                class_="analysis-section",
            )

        fleiss_scores = calculate_fleiss_kappa(list(annotator_dfs.values()))
        n_shared = fleiss_scores.get("n_shared", 0)

        if n_shared == 0:
            return ui.div(
                ui.div("Fleiss' κ: Inter-Annotator", class_="analysis-section-header"),
                ui.p(
                    "No shared entries between annotators.",
                    class_="analysis-empty",
                ),
                class_="analysis-section",
            )

        agg_kappa = fleiss_scores.get("aggregate", float("nan"))
        agg_interp = interpret_kappa(agg_kappa) if not math.isnan(agg_kappa) else "N/A"

        if math.isnan(agg_kappa):
            agg_class = "kappa-na"
            agg_display = "N/A"
        elif agg_kappa >= 0.61:
            agg_class = "kappa-good"
            agg_display = f"{agg_kappa:.2f}"
        elif agg_kappa >= 0.41:
            agg_class = "kappa-moderate"
            agg_display = f"{agg_kappa:.2f}"
        else:
            agg_class = "kappa-poor"
            agg_display = f"{agg_kappa:.2f}"

        return ui.div(
            ui.div("Fleiss' κ: Inter-Annotator", class_="analysis-section-header"),
            ui.p(
                f"{n_shared} shared entries across {len(annotator_dfs)} annotators",
                class_="fleiss-shared-info",
            ),
            ui.div(
                ui.span("Aggregate: ", class_="aggregate-label"),
                ui.span(agg_display, class_=f"kappa-badge kappa-badge-large {agg_class}"),
                ui.span(f" ({agg_interp})", class_="aggregate-interp"),
                class_="aggregate-summary",
            ),
            class_="analysis-section",
        )

    # Export handlers
    @reactive.effect
    @reactive.event(input.export_csv)
    def _on_export_csv():
        name = annotator_name()
        if not name:
            ui.notification_show(
                "Please enter your name to export annotations.",
                type="warning",
                duration=3,
            )
            return

        try:
            path = export_annotations_csv(name)
            ui.notification_show(
                f"CSV exported to: {path}",
                type="message",
                duration=5,
            )
        except Exception as e:
            ui.notification_show(
                f"Export failed: {e}",
                type="error",
                duration=5,
            )

    @reactive.effect
    @reactive.event(input.export_parquet)
    def _on_export_parquet():
        name = annotator_name()
        if not name:
            ui.notification_show(
                "Please enter your name to export annotations.",
                type="warning",
                duration=3,
            )
            return

        try:
            path = export_annotations_parquet(name)
            ui.notification_show(
                f"Parquet exported to: {path}",
                type="message",
                duration=5,
            )
        except Exception as e:
            ui.notification_show(
                f"Export failed: {e}",
                type="error",
                duration=5,
            )

    @reactive.effect
    @reactive.event(input.export_report)
    def _on_export_report():
        try:
            path = generate_agreement_report()
            ui.notification_show(
                f"Report generated: {path}",
                type="message",
                duration=5,
            )
        except Exception as e:
            ui.notification_show(
                f"Report generation failed: {e}",
                type="error",
                duration=5,
            )
