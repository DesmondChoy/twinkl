"""Shiny demo app for reviewing the live Twinkl pipeline."""

from __future__ import annotations

import asyncio
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
from shiny import App, reactive, render, ui

# Add project root to path for imports when running via shiny run
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.demo_tool.data_loader import (
    build_persona_choices,
    get_persona,
    load_demo_personas,
    summarize_warnings,
)
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget

from src.demo_tool.runtime_bridge import (
    CheckpointOption,
    build_checkpoint_choices,
    discover_checkpoints,
    get_checkpoint_option,
    load_cached_run,
    load_multi_drift_bundle,
    run_demo_pipeline,
)
from src.demo_tool.state import DemoAppState, create_demo_state
from src.models.judge import SCHWARTZ_VALUE_ORDER


@dataclass
class CatalogLoadResult:
    """Loaded demo catalog, warnings, and error state."""

    personas: list[dict[str, Any]] | None = None
    checkpoints: list[CheckpointOption] = field(default_factory=list)
    warning: str | None = None
    error: Exception | None = None
    error_traceback: str | None = None

    @property
    def is_success(self) -> bool:
        return self.error is None and self.personas is not None

    @property
    def total_personas(self) -> int:
        return len(self.personas) if self.personas else 0


def _load_catalog_data() -> CatalogLoadResult:
    """Load personas plus available checkpoints for the demo app."""
    result = CatalogLoadResult()
    warnings: list[str] = []

    try:
        persona_result = load_demo_personas()
        result.personas = persona_result.personas
        persona_warning = summarize_warnings(persona_result.warnings)
        if persona_warning:
            warnings.append(persona_warning)
    except Exception as exc:
        result.error = exc
        result.error_traceback = traceback.format_exc()
        return result

    checkpoints = discover_checkpoints()
    result.checkpoints = checkpoints
    if not checkpoints:
        warnings.append(
            "No local checkpoints were found under logs/experiments/artifacts, "
            "models/vif, or logs/experiments."
        )

    if warnings:
        result.warning = " ".join(warnings)
    return result


_catalog_result: CatalogLoadResult | None = None


def _get_catalog_result() -> CatalogLoadResult:
    """Get or initialize the cached catalog load result."""
    global _catalog_result
    if _catalog_result is None:
        _catalog_result = _load_catalog_data()
    return _catalog_result


def _reload_catalog_result() -> CatalogLoadResult:
    """Force a reload of personas and checkpoints."""
    global _catalog_result
    _catalog_result = _load_catalog_data()
    return _catalog_result


def _format_dimension(dimension: str) -> str:
    return dimension.replace("_", " ").title()


def _format_signed(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def _top_dimensions(row: dict[str, Any], *, limit: int = 2) -> tuple[list[str], list[str]]:
    """Extract strongest positive and negative dimensions from one signal row."""
    scored = [
        (dimension, float(row.get(f"alignment_{dimension}", 0.0)))
        for dimension in SCHWARTZ_VALUE_ORDER
    ]
    positives = [
        _format_dimension(dimension)
        for dimension, _score in sorted(scored, key=lambda item: item[1], reverse=True)
        if _score > 0
    ][:limit]
    negatives = [
        _format_dimension(dimension)
        for dimension, _score in sorted(scored, key=lambda item: item[1])
        if _score < 0
    ][:limit]
    return positives, negatives


def _build_badges(values: list[str], badge_class: str, empty_label: str) -> ui.Tag:
    """Render a row of small badges."""
    labels = values or [empty_label]
    return ui.div(
        *[ui.span(label, class_=f"badge {badge_class}") for label in labels],
        class_="badge-row",
    )


def _build_table(records: list[dict[str, Any]], columns: list[tuple[str, str]]) -> ui.Tag:
    """Render a compact HTML table from simple row records."""
    if not records:
        return ui.div("No rows to display.", class_="empty-state")

    header_cells = [ui.tags.th(label) for _key, label in columns]
    body_rows = []
    for record in records:
        cells = []
        for key, _label in columns:
            value = record.get(key)
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            cells.append(ui.tags.td("" if value is None else value))
        body_rows.append(ui.tags.tr(*cells))

    return ui.div(
        ui.tags.table(
            ui.tags.thead(ui.tags.tr(*header_cells)),
            ui.tags.tbody(*body_rows),
            class_="data-table",
        ),
        class_="table-shell",
    )


def _build_error_view(result: CatalogLoadResult) -> ui.Tag:
    """Render a startup error view."""
    error_type = type(result.error).__name__ if result.error else "UnknownError"
    error_message = str(result.error) if result.error else "Unknown error"
    return ui.div(
        ui.div(
            ui.h2("Unable to load demo data", class_="error-title"),
            ui.p(
                "The demo app needs wrangled persona files before it can render the review UI.",
                class_="error-copy",
            ),
            ui.div(
                ui.div("What went wrong", class_="error-section-label"),
                ui.p(f"{error_type}: {error_message}", class_="error-message"),
                class_="error-section",
            ),
            ui.div(
                ui.div("Expected input", class_="error-section-label"),
                ui.code("logs/wrangled/persona_*.md"),
                class_="error-section",
            ),
            ui.tags.details(
                ui.tags.summary("Technical details"),
                ui.tags.pre(result.error_traceback or "No traceback available"),
                class_="error-details",
            ),
            ui.input_action_button("refresh_catalog_btn", "Retry loading", class_="btn-secondary"),
            class_="error-card",
        ),
        class_="error-shell",
    )


def _build_main_app_ui() -> ui.Tag:
    """Render the main demo app shell."""
    return ui.TagList(
        ui.div(
            ui.div(
                ui.div(
                    ui.h1("Twinkl Demo Review", class_="demo-title"),
                    ui.p(
                        "Browse a persona, inspect the journal history, then run the live critic to drift and weekly digest flow.",
                        class_="demo-subtitle",
                    ),
                    class_="hero-copy",
                ),
                ui.div(
                    ui.output_ui("run_status_banner"),
                    class_="hero-status",
                ),
                class_="hero",
            ),
            ui.output_ui("catalog_warning_banner"),
            ui.div(
                ui.div(
                    ui.h3("Controls", class_="panel-title"),
                    ui.input_select("persona_id", "Persona", choices={}),
                    ui.output_ui("persona_summary_compact"),
                    ui.input_select("checkpoint_path", "Critic checkpoint", choices={}),
                    ui.input_select(
                        "drift_source",
                        "Detector input source",
                        choices={"judge": "Judge labels", "critic": "Critic predictions"},
                        selected="judge",
                    ),
                    ui.output_ui("checkpoint_summary"),
                    ui.div(
                        ui.input_action_button(
                            "refresh_catalog_btn",
                            "Refresh data",
                            class_="btn-secondary",
                        ),
                        ui.input_task_button(
                            "run_pipeline_btn",
                            "Run critic flow",
                            class_="btn-primary",
                        ),
                        class_="control-actions",
                    ),
                    class_="panel left-panel",
                ),
                ui.div(
                    ui.output_ui("persona_detail_card"),
                    ui.output_ui("journal_timeline"),
                    class_="panel center-panel",
                ),
                ui.div(
                    ui.output_ui("results_panel"),
                    class_="panel right-panel",
                ),
                class_="demo-grid",
            ),
            class_="app-shell",
        )
    )


app_ui = ui.page_fluid(
    ui.tags.link(rel="stylesheet", href="styles.css"),
    ui.output_ui("app_content"),
)


def server(input, output, session):
    state: DemoAppState = create_demo_state()
    catalog_reload_trigger = reactive.value(0)

    @reactive.calc
    def catalog_result() -> CatalogLoadResult:
        _ = catalog_reload_trigger()
        return _get_catalog_result()

    @render.ui
    def app_content():
        result = catalog_result()
        if not result.is_success:
            return _build_error_view(result)
        return _build_main_app_ui()

    @reactive.calc
    def personas() -> list[dict[str, Any]]:
        result = catalog_result()
        return result.personas if result.personas else []

    @reactive.calc
    def checkpoints() -> list[CheckpointOption]:
        return catalog_result().checkpoints

    @reactive.calc
    def current_persona() -> dict[str, Any] | None:
        return get_persona(personas(), input.persona_id())

    @reactive.calc
    def current_checkpoint() -> CheckpointOption | None:
        return get_checkpoint_option(checkpoints(), input.checkpoint_path())

    @reactive.calc
    def multi_drift_bundle():
        persona_id = input.persona_id()
        source = input.drift_source()
        if not persona_id:
            return None
        timeline_df = None
        if source == "critic":
            bundle = state.run_bundle()
            if bundle is None:
                return None
            timeline_df = bundle["artifacts"]["timeline_df"]
        return load_multi_drift_bundle(persona_id, source=source, timeline_df=timeline_df)

    @reactive.effect
    def _sync_persona_select():
        result = catalog_result()
        if not result.is_success:
            return
        choices = build_persona_choices(personas())
        selected = input.persona_id() if input.persona_id() in choices else None
        if selected is None and choices:
            selected = next(iter(choices))
        ui.update_select(
            "persona_id",
            choices=choices,
            selected=selected,
            session=session,
        )

    @reactive.effect
    def _sync_checkpoint_select():
        result = catalog_result()
        if not result.is_success:
            return
        choices = build_checkpoint_choices(checkpoints())
        selected = input.checkpoint_path() if input.checkpoint_path() in choices else None
        if selected is None and choices:
            selected = next(iter(choices))
        ui.update_select(
            "checkpoint_path",
            choices=choices,
            selected=selected,
            session=session,
        )

    @reactive.effect
    @reactive.event(input.refresh_catalog_btn)
    def _refresh_catalog():
        _reload_catalog_result()
        catalog_reload_trigger.set(catalog_reload_trigger() + 1)
        ui.notification_show("Reloaded personas and checkpoint catalog.", type="message", duration=3)

    @reactive.effect
    @reactive.event(input.persona_id, input.checkpoint_path)
    def _sync_selection_and_cache():
        persona_id = input.persona_id()
        checkpoint_path = input.checkpoint_path()
        state.set_selection(persona_id, checkpoint_path)
        state.bio_expanded.set(True)

        if not persona_id or not checkpoint_path:
            state.clear_result("idle")
            return

        cached = load_cached_run(persona_id, checkpoint_path)
        if cached is None:
            state.clear_result("idle")
            return
        state.set_result(cached, status="cached")

    @ui.bind_task_button(button_id="run_pipeline_btn")
    @reactive.extended_task
    async def run_pipeline_task(persona_id: str, checkpoint_path: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            run_demo_pipeline,
            persona_id=persona_id,
            checkpoint_path=checkpoint_path,
        )

    @reactive.effect
    @reactive.event(input.run_pipeline_btn)
    def _start_pipeline_run():
        persona_id = input.persona_id()
        checkpoint_path = input.checkpoint_path()
        if not persona_id:
            ui.notification_show("Select a persona first.", type="warning", duration=4)
            return
        if not checkpoint_path:
            ui.notification_show("Select a checkpoint first.", type="warning", duration=4)
            return

        state.run_status.set("running")
        state.run_error.set(None)
        run_pipeline_task.invoke(persona_id, checkpoint_path)

    @reactive.effect
    def _observe_pipeline_task():
        status = run_pipeline_task.status()
        if status == "running":
            state.run_status.set("running")
            return
        if status == "success":
            bundle = run_pipeline_task.result()
            selection_key = DemoAppState.selection_key(
                bundle["persona_id"],
                bundle["checkpoint_path"],
            )
            if selection_key == state.active_selection_key():
                state.set_result(bundle, status="success")
                ui.notification_show(
                    "Pipeline run complete. Results refreshed from live artifacts.",
                    type="message",
                    duration=4,
                )
            return
        if status == "error":
            try:
                run_pipeline_task.result()
            except Exception as exc:  # pragma: no cover - exercised via UI
                state.set_error(f"{type(exc).__name__}: {exc}")
                ui.notification_show(
                    f"Pipeline run failed: {type(exc).__name__}",
                    type="error",
                    duration=6,
                )

    @render_widget
    def detector_chart():
        return _build_detector_chart(multi_drift_bundle())

    @render.ui
    def detector_table():
        return _render_detector_table(multi_drift_bundle())

    @render.ui
    def catalog_warning_banner():
        warning = catalog_result().warning
        if not warning:
            return None
        return ui.div(
            ui.strong("Catalog note"),
            ui.span(warning),
            class_="banner banner-warning",
        )

    @render.ui
    def persona_summary_compact():
        persona = current_persona()
        if persona is None:
            return ui.div("Select a persona to inspect its details.", class_="empty-state")

        core_values = persona.get("persona_core_values") or []
        return ui.div(
            ui.div(
                ui.span(str(persona["n_entries"]), class_="stat-number"),
                ui.span("entries", class_="stat-label"),
                class_="mini-stat",
            ),
            ui.div(
                ui.span(persona["first_entry_date"], class_="stat-number"),
                ui.span("first entry", class_="stat-label"),
                class_="mini-stat",
            ),
            ui.div(
                ui.span(persona["last_entry_date"], class_="stat-number"),
                ui.span("latest entry", class_="stat-label"),
                class_="mini-stat",
            ),
            _build_badges(core_values, "badge-core", "No core values"),
            class_="summary-stack",
        )

    @render.ui
    def checkpoint_summary():
        checkpoint = current_checkpoint()
        if checkpoint is None:
            return ui.div(
                "Select a checkpoint to enable live runtime inference.",
                class_="empty-state",
            )

        metrics = checkpoint.metrics_summary or {}
        metric_bits = []
        for label, key in (
            ("QWK", "qwk_mean"),
            ("Recall -1", "recall_minus1"),
            ("Calibration", "calibration_global"),
        ):
            if key in metrics:
                metric_bits.append(
                    ui.div(
                        ui.span(label, class_="metric-label"),
                        ui.span(f"{metrics[key]:.3f}", class_="metric-value"),
                        class_="metric-chip",
                    )
                )

        return ui.div(
            ui.div(checkpoint.label, class_="checkpoint-label"),
            ui.div(
                f"Source: {checkpoint.source}",
                class_="checkpoint-source",
            ),
            ui.div(*metric_bits, class_="metric-chip-row") if metric_bits else None,
            class_="checkpoint-summary",
        )

    @render.ui
    def run_status_banner():
        status = state.run_status()
        if status == "running":
            return ui.div(
                ui.strong("Running now"),
                ui.span("The critic flow is executing in the background."),
                class_="banner banner-info",
            )
        if status == "success":
            return ui.div(
                ui.strong("Live results"),
                ui.span("This view reflects a freshly completed pipeline run."),
                class_="banner banner-success",
            )
        if status == "cached":
            return ui.div(
                ui.strong("Cached results"),
                ui.span("Loaded the latest persisted artifacts for this persona and checkpoint."),
                class_="banner banner-neutral",
            )
        if status == "error":
            return ui.div(
                ui.strong("Run failed"),
                ui.span(state.run_error() or "Unknown error"),
                class_="banner banner-danger",
            )
        return ui.div(
            ui.strong("Ready"),
            ui.span("Pick a persona and checkpoint, then run the critic flow."),
            class_="banner banner-neutral",
        )

    @render.ui
    def persona_detail_card():
        persona = current_persona()
        if persona is None:
            return ui.div("No persona selected.", class_="empty-state")

        bio = persona.get("persona_bio")
        is_expanded = state.bio_expanded()
        return ui.div(
            ui.div(
                ui.div(
                    ui.h2(persona["persona_name"], class_="persona-name"),
                    ui.div(
                        f"{persona.get('persona_age') or 'Age n/a'} • "
                        f"{persona.get('persona_profession') or 'Profession n/a'}",
                        class_="persona-subtitle",
                    ),
                    ui.div(persona.get("persona_culture") or "Culture n/a", class_="persona-subtitle"),
                ),
                ui.input_action_button(
                    "bio_toggle",
                    "Hide bio" if is_expanded else "Show bio",
                    class_="btn-secondary",
                ),
                class_="persona-header",
            ),
            _build_badges(
                persona.get("persona_core_values") or [],
                "badge-core",
                "No core values",
            ),
            ui.div(bio, class_="persona-bio") if bio and is_expanded else None,
            class_="persona-card",
        )

    @reactive.effect
    @reactive.event(input.bio_toggle)
    def _toggle_bio():
        state.bio_expanded.set(not state.bio_expanded())

    @render.ui
    def journal_timeline():
        persona = current_persona()
        if persona is None:
            return ui.div("No journal history available.", class_="empty-state")

        timeline_cards = []
        for entry in persona["entries"]:
            timeline_cards.append(
                ui.div(
                    ui.div(
                        ui.span(f"Entry {entry['t_index'] + 1}", class_="entry-index"),
                        ui.span(entry["date"], class_="entry-date"),
                        class_="entry-meta",
                    ),
                    ui.p(entry["initial_entry"], class_="entry-text"),
                    ui.div(
                        ui.div(
                            ui.strong("Nudge"),
                            ui.p(entry["nudge_text"], class_="thread-text"),
                            class_="thread-block thread-nudge",
                        )
                        if entry.get("has_nudge") and entry.get("nudge_text")
                        else None,
                        ui.div(
                            ui.strong("Response"),
                            ui.p(entry["response_text"], class_="thread-text"),
                            class_="thread-block thread-response",
                        )
                        if entry.get("has_response") and entry.get("response_text")
                        else None,
                        class_="thread-stack",
                    ),
                    class_="timeline-card",
                )
            )

        return ui.div(
            ui.div("Journal timeline", class_="section-title"),
            ui.div(*timeline_cards, class_="timeline-stack"),
        )

    @render.ui
    def results_panel():
        bundle = state.run_bundle()
        if bundle is None:
            status_msg = (
                "The live pipeline is running. Tabs will populate when the artifacts are ready."
                if state.run_status() == "running"
                else "No artifacts loaded yet. Use the run button or choose a selection with cached outputs."
            )
            return ui.TagList(
                ui.h3("Results", class_="panel-title"),
                ui.div(status_msg, class_="empty-state"),
                ui.navset_tab(
                    ui.nav_panel(
                        "Detector comparison",
                        output_widget("detector_chart"),
                        ui.output_ui("detector_table"),
                    ),
                ),
            )

        artifacts = bundle["artifacts"]
        digest = artifacts["digest_payload"]
        drift = artifacts["drift_payload"]
        timeline_df = artifacts["timeline_df"].sort(["date", "t_index"])
        weekly_df = artifacts["weekly_df"].sort(["week_start"])

        # Build critic-based detector alert map for per-entry annotations
        critic_alert_map: dict[int, list[str]] = {}
        try:
            critic_bundle = load_multi_drift_bundle(
                bundle["persona_id"], source="critic", timeline_df=timeline_df
            )
            if critic_bundle is not None:
                for detector in critic_bundle.detectors:
                    for t in detector.alert_steps:
                        critic_alert_map.setdefault(t, []).append(detector.name)
        except Exception:
            pass

        return ui.TagList(
            ui.h3("Pipeline results", class_="panel-title"),
            ui.div(
                ui.div(
                    ui.span("Mode", class_="metric-label"),
                    ui.span(str(digest.get("response_mode", "n/a")), class_="metric-value"),
                    class_="metric-card",
                ),
                ui.div(
                    ui.span("Overall mean", class_="metric-label"),
                    ui.span(_format_signed(digest.get("overall_mean")), class_="metric-value"),
                    class_="metric-card",
                ),
                ui.div(
                    ui.span("Uncertainty", class_="metric-label"),
                    ui.span(_format_signed(digest.get("overall_uncertainty")), class_="metric-value"),
                    class_="metric-card",
                ),
                class_="metric-card-row",
            ),
            ui.navset_tab(
                ui.nav_panel(
                    "Per-entry critic",
                    _render_timeline_results(timeline_df, critic_alert_map),
                ),
                ui.nav_panel(
                    "Weekly signals",
                    _render_weekly_results(weekly_df),
                ),
                ui.nav_panel(
                    "Drift",
                    _render_drift_results(drift),
                ),
                ui.nav_panel(
                    "Weekly digest",
                    _render_digest_results(artifacts["digest_markdown"], digest),
                ),
                ui.nav_panel(
                    "Detector comparison",
                    output_widget("detector_chart"),
                    ui.output_ui("detector_table"),
                ),
            ),
        )


def _render_timeline_results(
    timeline_df: pl.DataFrame,
    alert_map: dict[int, list[str]] | None = None,
) -> ui.Tag:
    """Render per-entry VIF outputs with optional detector alert annotations."""
    alert_map = alert_map or {}
    cards = []
    for row in timeline_df.to_dicts():
        strengths, tensions = _top_dimensions(row)
        t = int(row["t_index"])
        fired_detectors = alert_map.get(t, [])

        alert_row = None
        if fired_detectors:
            alert_row = ui.div(
                ui.span("Drift alert", class_="alert-label"),
                *[ui.span(name, class_="badge badge-alert") for name in fired_detectors],
                class_="alert-row",
            )

        cards.append(
            ui.div(
                ui.div(
                    ui.span(f"Entry {t + 1}", class_="entry-index"),
                    ui.span(str(row["date"]), class_="entry-date"),
                    class_="entry-meta",
                ),
                alert_row,
                ui.p(str(row.get("initial_entry") or ""), class_="entry-text compact"),
                ui.div(
                    ui.div(
                        ui.span("Overall mean", class_="metric-label"),
                        ui.span(_format_signed(float(row["overall_mean"])), class_="metric-value"),
                        class_="metric-chip",
                    ),
                    ui.div(
                        ui.span("Uncertainty", class_="metric-label"),
                        ui.span(
                            f"{float(row['overall_uncertainty']):.3f}",
                            class_="metric-value",
                        ),
                        class_="metric-chip",
                    ),
                    class_="metric-chip-row",
                ),
                ui.div(
                    ui.div(
                        ui.span("Strongest alignment", class_="subsection-label"),
                        _build_badges(strengths, "badge-positive", "None clear"),
                    ),
                    ui.div(
                        ui.span("Main tensions", class_="subsection-label"),
                        _build_badges(tensions, "badge-negative", "None clear"),
                    ),
                    class_="subsection-stack",
                ),
                class_=f"timeline-card {'timeline-card-alert' if fired_detectors else ''}",
            )
        )

    return ui.div(*cards, class_="timeline-stack")


def _render_weekly_results(weekly_df: pl.DataFrame) -> ui.Tag:
    """Render weekly aggregate rows."""
    records = []
    for row in weekly_df.to_dicts():
        strengths, tensions = _top_dimensions(row)
        records.append(
            {
                "window": f"{row['week_start']} to {row['week_end']}",
                "overall_mean": f"{float(row['overall_mean']):+.3f}",
                "overall_uncertainty": f"{float(row['overall_uncertainty']):.3f}",
                "top_strengths": _build_badges(strengths, "badge-positive", "None"),
                "top_tensions": _build_badges(tensions, "badge-negative", "None"),
            }
        )

    return _build_table(
        records,
        [
            ("window", "Week window"),
            ("overall_mean", "Overall mean"),
            ("overall_uncertainty", "Uncertainty"),
            ("top_strengths", "Top strengths"),
            ("top_tensions", "Top tensions"),
        ],
    )


def _render_drift_results(drift: dict[str, Any]) -> ui.Tag:
    """Render structured drift output."""
    dimension_cards = []
    for signal in drift.get("dimension_signals", []):
        dimension_cards.append(
            ui.div(
                ui.div(
                    ui.span(_format_dimension(signal["dimension"]), class_="dimension-name"),
                    ui.span(signal.get("classification", "unknown"), class_="dimension-status"),
                    class_="dimension-header",
                ),
                ui.div(
                    ui.span(f"Mean {signal.get('mean_alignment', 0.0):+.3f}"),
                    ui.span(f"Uncertainty {signal.get('mean_uncertainty', 0.0):.3f}"),
                    ui.span(f"Trigger {signal.get('trigger') or 'none'}"),
                    class_="dimension-metrics",
                ),
                class_="dimension-card",
            )
        )

    return ui.div(
        ui.div(
            ui.div(
                ui.span("Trigger type", class_="metric-label"),
                ui.span(str(drift.get("trigger_type") or "stable"), class_="metric-value"),
                class_="metric-card",
            ),
            ui.div(
                ui.span("Window", class_="metric-label"),
                ui.span(
                    f"{drift.get('week_start') or 'n/a'} to {drift.get('week_end') or 'n/a'}",
                    class_="metric-value",
                ),
                class_="metric-card",
            ),
            class_="metric-card-row",
        ),
        ui.div(
            ui.strong("Rationale"),
            ui.p(str(drift.get("rationale") or "No rationale captured."), class_="drift-rationale"),
            class_="drift-summary",
        ),
        ui.div(
            ui.span("Triggered dimensions", class_="subsection-label"),
            _build_badges(
                [_format_dimension(dim) for dim in drift.get("triggered_dimensions", [])],
                "badge-negative",
                "None",
            ),
        ),
        ui.div(*dimension_cards, class_="dimension-grid")
        if dimension_cards
        else ui.div("No per-dimension drift signals were recorded.", class_="empty-state"),
    )


def _render_digest_results(markdown_text: str, digest: dict[str, Any]) -> ui.Tag:
    """Render the deterministic weekly digest plus summary badges."""
    return ui.div(
        ui.div(
            ui.div(
                ui.span("Top tensions", class_="subsection-label"),
                _build_badges(
                    [_format_dimension(dim) for dim in digest.get("top_tensions", [])],
                    "badge-negative",
                    "None clear",
                ),
            ),
            ui.div(
                ui.span("Top strengths", class_="subsection-label"),
                _build_badges(
                    [_format_dimension(dim) for dim in digest.get("top_strengths", [])],
                    "badge-positive",
                    "None clear",
                ),
            ),
            class_="subsection-stack",
        ),
        ui.div(ui.markdown(markdown_text), class_="digest-markdown"),
    )


def _build_detector_chart(bundle: Any | None) -> go.Figure:
    """Build a Plotly figure showing per-dimension alignment trajectories with detector alert markers."""
    from src.demo_tool.multi_drift import DETECTOR_KEYS, DETECTOR_NAMES, DIM_LABELS

    fig = go.Figure()

    if bundle is None:
        fig.add_annotation(
            text="Select a persona (and run the critic if using Critic source)",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color="#6b625c"),
        )
        fig.update_layout(_detector_chart_layout("Detector comparison — no data"))
        return fig

    T = bundle.n_entries
    x_labels = [f"E{t + 1}" for t in range(T)]
    colors = [
        "#8a4b24", "#285943", "#294f74", "#7a3d7a", "#8a6a24", "#2a6a6a",
        "#c0392b", "#27ae60", "#2980b9", "#8e44ad", "#d4a017", "#16a085",
    ]

    # One trace per dimension
    for j, short in enumerate(DIM_LABELS):
        y_vals = bundle.scores_matrix[:, j].tolist()
        is_core = bundle.weights[j] > 0
        opacity = 1.0 if is_core else 0.35
        fig.add_trace(go.Scatter(
            x=x_labels, y=y_vals,
            mode="lines+markers",
            name=short,
            line=dict(color=colors[j % len(colors)], width=2 if is_core else 1),
            marker=dict(size=5),
            opacity=opacity,
        ))

    # Alert markers per detector (symbols along bottom)
    marker_symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up"]
    marker_y_offset = -1.35
    for di, detector in enumerate(bundle.detectors):
        for t in sorted(detector.alert_steps):
            if t < T:
                fig.add_trace(go.Scatter(
                    x=[x_labels[t]],
                    y=[marker_y_offset - di * 0.12],
                    mode="markers",
                    marker=dict(symbol=marker_symbols[di], size=9, color=colors[di % len(colors)]),
                    name=detector.name,
                    showlegend=False,
                    hovertext=f"{detector.name} alert at {x_labels[t]}",
                    hoverinfo="text",
                ))

    # Consensus bar as subtle background shading
    for t, votes in bundle.consensus.items():
        if votes >= 3 and t < T:
            fig.add_vrect(
                x0=t - 0.4, x1=t + 0.4,
                fillcolor="rgba(138,45,42,0.08)", line_width=0,
            )

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="#d8cdc0", line_width=1)

    source_label = "Judge labels" if bundle.source == "judge" else "Critic predictions"
    fig.update_layout(_detector_chart_layout(
        f"Alignment trajectories — {source_label} ({bundle.n_entries} entries)"
    ))
    return fig


def _detector_chart_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=13, color="#1f1a17")),
        height=420,
        margin=dict(l=40, r=20, t=40, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,253,250,0.6)",
        font=dict(family="Iowan Old Style, Palatino Linotype, serif", size=11, color="#1f1a17"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(gridcolor="#ece7e0", showgrid=True),
        yaxis=dict(gridcolor="#ece7e0", showgrid=True, range=[-1.7, 1.1]),
    )


def _render_detector_table(bundle: Any | None) -> ui.Tag:
    """Render the step × detector alert grid table."""
    from src.demo_tool.multi_drift import DETECTOR_KEYS, DETECTOR_NAMES

    if bundle is None:
        return ui.div(
            "Select a persona to see detector comparison. Judge labels work without running the critic.",
            class_="empty-state",
        )

    if bundle.n_entries < 3:
        return ui.div(
            f"This persona has only {bundle.n_entries} entries — some detectors need more history.",
            class_="empty-state",
        )

    # Summary chips row
    chips = []
    for detector in bundle.detectors:
        fired = len(detector.alert_steps)
        chips.append(
            ui.div(
                ui.div(detector.name, class_="detector-name"),
                ui.div(
                    f"{fired} step{'s' if fired != 1 else ''} fired",
                    class_="detector-fired" if fired > 0 else "detector-silent",
                ),
                class_="detector-card",
            )
        )

    # Step × detector table
    columns = [("step", "Step"), ("date", "Date")]
    for key, name in zip(DETECTOR_KEYS, DETECTOR_NAMES):
        columns.append((key, name))
    columns.append(("consensus", "Votes"))

    records = []
    for i, (t, date) in enumerate(zip(bundle.t_indices, bundle.dates)):
        row: dict[str, Any] = {"step": f"E{t + 1}", "date": date}
        for detector in bundle.detectors:
            row[detector.key] = "●" if t in detector.alert_steps else "–"
        votes = bundle.consensus.get(t, 0)
        row["consensus"] = ui.span(
            str(votes),
            class_=f"consensus-badge consensus-{min(votes, 4)}",
        )
        records.append(row)

    source_label = "Judge labels" if bundle.source == "judge" else "Critic predictions"
    return ui.div(
        ui.div(f"Source: {source_label} · {bundle.n_entries} entries · core values: {', '.join(bundle.core_values) or 'none'}", class_="detector-meta"),
        ui.div(*chips, class_="detector-summary-row"),
        _build_table(records, columns),
    )


_app_dir = Path(__file__).parent

app = App(
    app_ui,
    server,
    static_assets=_app_dir / "static",
)


def _free_port(port: int = 8001) -> None:
    """Kill any process using the specified port before starting the app directly."""
    import subprocess

    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
        )
        pids = [pid for pid in result.stdout.strip().split("\n") if pid]
        for pid in pids:
            subprocess.run(["kill", pid], capture_output=True)
    except Exception:
        pass


if __name__ == "__main__":
    from shiny import run_app

    _free_port(8001)
    run_app(app, port=8001)
