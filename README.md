# Twinkl

**Overview** | [Onboarding and Experience](docs/readme/onboarding_and_experience.md) | [Review Apps](docs/readme/review_apps.md) | [Research and Data](docs/readme/research_and_data.md) | [Status and Setup](docs/readme/status_and_setup.md)

---

Twinkl is an "inner compass" that helps users compare daily behavior with
confirmed Core Values. Unlike traditional journaling apps that summarize moods and
topics, Twinkl starts with a confirmed Profile, reviews Journal Entries over
time, and shows evidence when behavior conflicts with Core Values. The current
POC keeps the Profile fixed. Profile evolution is future work.

## Experience

- **Try the Demo** compares five saved synthetic Personas: Nisha, Noor, Lukas,
  Wei Jun, and Meera. A selectable table shows their weekly Drift states, with
  a short explanation and one action to start or continue the selected replay.
  Each week shows its Journal Entries, nudges, and responses;
  **Review Weekly Drift Detection** opens the saved weekly result.
  The 27-week replay uses recorded model responses and requires no provider key.
- **Try Onboarding** begins the values assessment and manual Experience.
  A confirmed Profile holds at most two Core Values. Manual Journal Entries
  support contextual nudges, replies, skips, and closed-week review.
- **Weekly Drift Detection** uses a Luna-low Weekly Drift Reviewer and the
  deterministic rule that two consecutive Conflicts for one Core Value form
  Drift. The **Coach Digest** explains the result with cited writing and one
  reflective question.
- **North Star Moment** shows at most one exact quotation of a supportive
  action beneath an eligible weekly reflection. Saved replay includes its
  selections and no-card outcomes. Live review requires a valid integration
  budget and otherwise leaves the card unavailable.
- **Inspect** exposes the same evidence, model provenance, decisions, and
  trace events as Experience. The VIF Critic (Offline) is a separate research
  component.

> **Project status:** Twinkl is under active development as an academic capstone. Status markers in each section indicate capability maturity:
> ✅ Complete · 🧪 Experimental · 🚧 In Progress · ⚠️ Partial · 📋 Specified · ❌ Not Started
>
> See [Known Gaps](docs/readme/status_and_setup.md#known-gaps) for the summary and [Implementation Status](docs/prd.md#implementation-status) for the full breakdown.

## Architecture

See the [end-to-end architecture](docs/architecture/e2e_architecture.md) for the
Experience, Weekly Drift Detection, North Star Moment, and offline research
paths. [Status and Setup](docs/readme/status_and_setup.md) covers local launch,
checks, scenario export, and experiment commands.

## Documentation Guide

- [`docs/prd.md`](docs/prd.md) — authoritative product intent and implementation status
- [`docs/canonical_nouns.md`](docs/canonical_nouns.md) — required Twinkl product terms and evidence-language rules
- [`docs/architecture/e2e_architecture.md`](docs/architecture/e2e_architecture.md) — current Experience, Weekly Drift Detection, and offline research paths
- [`docs/demo/experience_inspect_app.md`](docs/demo/experience_inspect_app.md) and [`frontend/onboarding/README.md`](frontend/onboarding/README.md) — Experience and Inspect behavior, local launch, checks, and assessment deployment
- [`docs/vif/01_concepts_and_roadmap.md`](docs/vif/01_concepts_and_roadmap.md), [`docs/vif/02_system_architecture.md`](docs/vif/02_system_architecture.md), [`docs/vif/03_model_training.md`](docs/vif/03_model_training.md), [`docs/vif/04_uncertainty_logic.md`](docs/vif/04_uncertainty_logic.md) — VIF design, runtime, training, and uncertainty logic
- [`docs/pipeline/pipeline_specs.md`](docs/pipeline/pipeline_specs.md), [`docs/pipeline/data_schema.md`](docs/pipeline/data_schema.md), [`docs/pipeline/consensus_rejudging_instructions.md`](docs/pipeline/consensus_rejudging_instructions.md) — data generation, label datasets, and consensus diagnostics
- [`docs/drift/trajectory_eda.md`](docs/drift/trajectory_eda.md) — historical Drift-definition analysis comparing five-pass LLM-Judge consensus with persisted labels
- [`docs/evals/drift_v1_student_visible_target.md`](docs/evals/drift_v1_student_visible_target.md) — historical five-Drift development result and withheld former final-test score
- [`docs/weekly/weekly_drift_detection.md`](docs/weekly/weekly_drift_detection.md) — Weekly Drift Detection and Coach Digest contracts and runtime CLI
- [`docs/north_star/north_star_moment.md`](docs/north_star/north_star_moment.md) — North Star Moment eligibility, exact-quotation checks, Experience, and Inspect
- [`docs/north_star/nsm_experiment_methodology.md`](docs/north_star/nsm_experiment_methodology.md) — full-history and retrieval comparison, cohort selection, reproducibility, and AI evaluation limits
- [`docs/evals/overview.md`](docs/evals/overview.md) and [`docs/evals/coach_narrative_test_and_eval_guide.md`](docs/evals/coach_narrative_test_and_eval_guide.md) — evaluation status, offline checks, paid Coach Digest Evals, and the Drift/control study
- [`docs/demo/weekly_drift_review_app.md`](docs/demo/weekly_drift_review_app.md) — read-only Drift inspection of the frozen Weekly Drift Reviewer comparison Runs
- [`docs/demo/review_app.md`](docs/demo/review_app.md) — deprecated Runtime Demo Review App for the VIF Critic (Offline) compatibility path
- [`docs/capstone_report/capstone_project_report.md`](docs/capstone_report/capstone_project_report.md) — maintained Phase 2 Technical Paper source; [report instructions](docs/capstone_report/README.md) cover PDF generation and visual verification
- [`docs/future_work/README.md`](docs/future_work/README.md) — exploratory directions, including OpenClaw integration research
