"""UI components for the Schwartz Value Annotation Tool.

This package provides modular Shiny components for the annotation interface:

- header: App header with annotator input, progress bar, navigation
- sidebar: Left sidebar with persona info and entry navigation
- entry_display: Center column with entry content and status
- scoring_grid: Right column with value scoring and save button
- modals: Modal dialogs for warnings and comparisons
- comparison_view: Inline comparison view for post-save display
- analysis_view: Collapsible accordion with agreement metrics and export
- constants: Shared constants and helper functions

Usage:
    from src.annotation_tool.components import (
        header,
        sidebar,
        entry_display,
        scoring_grid,
        modals,
        comparison_view,
        analysis_view,
    )
"""

from src.annotation_tool.components import (
    analysis_view,
    comparison_view,
    constants,
    entry_display,
    header,
    modals,
    scoring_grid,
    sidebar,
)

__all__ = [
    "analysis_view",
    "comparison_view",
    "constants",
    "entry_display",
    "header",
    "modals",
    "scoring_grid",
    "sidebar",
]
