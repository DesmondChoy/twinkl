# Future Work Index

This folder collects exploratory directions that sit outside the current
committed capstone scope. These documents describe possible extensions and
distribution ideas, not active product guarantees.

## Current Documents

- [`practice_module_ideas.md`](practice_module_ideas.md)
  - twelve candidate projects for the NUS-ISS Architecting AI Systems practice
    module, each doubling as a Twinkl enhancement; written in plain language
    for a non-specialist audience
- [`habit_recommendations.md`](habit_recommendations.md)
  - concept note for a goal-aligned recommendation layer
- [`OpenClaw_Twinkl_Integration_Research.pdf`](OpenClaw_Twinkl_Integration_Research.pdf)
  - integration research for packaging Twinkl as an OpenClaw-compatible skill
- [`twinkl-openclaw-workflow.html`](twinkl-openclaw-workflow.html)
  - visual workflow for the same OpenClaw distribution/integration direction
- [`viz/dashboard_directions.html`](viz/dashboard_directions.html)
  - four synchronized developer-facing VIF dashboard directions (Phosphor TUI,
    Specimen Swiss, Mission Control, and Field Notes) over seeded fake data;
    interaction and layout study only, with no backend wiring

Serve the dashboard study locally with the `mockups` launch configuration or:

```sh
python3 -m http.server 8734 --directory docs/future_work/viz
```

Then open `http://127.0.0.1:8734/dashboard_directions.html`.

## Scope Note

These documents are useful for architecture planning, capstone discussion, and
go-to-market exploration. The authoritative current-state product scope remains
[`docs/prd.md`](../prd.md).
