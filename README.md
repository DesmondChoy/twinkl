# Twinkl

**[Open Onboarding, Experience and Inspect →](https://onboarding-production-1dd2.up.railway.app/)**

Try the values assessment or select **Try the Demo → Nisha** to explore saved Journal Entries, Weekly Drift Detection, Coach Digest, and their Inspect traces.

**[Open the Drift Inspection App →](https://twinkl-drift-review-production.up.railway.app/)**

Browse the synthetic corpus, reference labels, reviewer decisions, and detected or missed Drift. Select a count to inspect its source records.

Both apps are deployed on Railway. Saved demo replay and Drift inspection require no API key or paid model calls.

Twinkl is an NUS-ISS capstone proof of concept for comparing daily behavior with a confirmed Profile and Core Values. The **[capstone report (PDF)](docs/capstone_report/capstone_report.pdf)** presents the research; this README points to its data and code. Evaluation uses synthetic data, project-team human annotation for label agreement, and AI review of downstream outputs. Benefit to real users remains untested.

## Inspect the report's data and evidence

| Report topic | Inspect the data | Read the evidence |
|---|---|---|
| **RQ1 · Synthetic data and label reliability** (§3.2, §4.1) | [Cleaned Journal Entries](logs/wrangled/), [Persona registry](logs/registry/personas.parquet), [LLM-Judge VIF Labels](logs/judge_labels/judge_labels.parquet), [human annotations](logs/annotations/) | [Schemas and queries](docs/pipeline/data_schema.md), [label-agreement study](docs/evals/judge_validation_summary.md) |
| **RQ2 · VIF Critic (Offline)** (§3.3, §4.2) | [run_060 predictions and checkpoint](logs/experiments/artifacts/ordinal_v4_s2025_m22_20260711_162502/BalancedSoftmax/), [configuration and metrics](logs/experiments/runs/run_060_BalancedSoftmax.yaml) | [Experiment index](logs/experiments/index.md), [VIF Critic evaluation](docs/evals/value_modeling_eval.md) |
| **RQ3 · Weekly Drift Detection** (§3.4, §4.3) | [Drift Inspection App](https://twinkl-drift-review-production.up.railway.app/), [saved model-comparison responses and metrics](logs/experiments/artifacts/twinkl_q6pt_luna_model_20260923/) | [Drift evaluation](docs/evals/drift_detection_eval.md), [matched model comparison](logs/experiments/reports/experiment_review_2026-09-23_twinkl_q6pt_luna_model.md) |
| **RQ4 · Coach Digest and North Star Moment** (§3.5, §4.4) | [Saved Persona scenarios](frontend/onboarding/public/scenarios/), [Coach Digest responses](src/demo/coach_digest_responses.json), [North Star Moment records](src/demo/north_star_replay_records.json) | [Coach Digest evaluation scope and limits](docs/evals/explanation_quality_eval.md), [North Star Moment study](logs/experiments/reports/north_star_v4_run1_20260907/report.md) |

The corpus contains **204 Personas and 1,651 Journal Entries**. Weekly Drift Detection uses the Weekly Drift Reviewer and a deterministic rule: two consecutive Conflicts for the same Core Value form Drift. The VIF Critic (Offline) remains a research component: `run_019`–`run_021` are the historical reference; `run_060` is the nominated checkpoint with repaired Security labels. The [research guide](docs/readme/research_and_data.md) explains these distinctions; report Appendix A preserves historical evidence snapshots.

## Run locally

From the repository root, with **Python 3.12+, uv, and Node.js 22 (22.12+)**:

```sh
uv sync --locked --group dev
source .venv/bin/activate
uv run uvicorn src.demo.api:app --port 8000
```

In a second terminal, start the frontend and open the URL printed by Vite:

```sh
npm --prefix frontend/onboarding ci
npm --prefix frontend/onboarding run dev
```

Live Journal Entry processing and Weekly Drift Detection require `OPENAI_API_KEY` in a local `.env`; add `--env-file .env` to the API command. Saved replay works without it. See the [setup guide](docs/readme/status_and_setup.md) for provider settings.

Launch the read-only Drift Inspection App:

```sh
uv run shiny run --host 127.0.0.1 --port 8001 --no-dev-mode src/drift_review_app/app.py
```

Inspect saved VIF Critic test predictions without loading a model:

```sh
uv run python - <<'PY'
import polars as pl

path = "logs/experiments/artifacts/ordinal_v4_s2025_m22_20260711_162502/BalancedSoftmax/selected_test_outputs.parquet"
outputs = pl.read_parquet(path)
print(outputs.select("persona_id", "t_index", "dimension", "target", "predicted_class").head(10))
PY
```

Verify frozen model-comparison inputs and run application checks:

```sh
uv run python -m scripts.experiments.replay_luna_comparison q6pt verify
uv run python -m scripts.experiments.replay_luna_comparison gv3x verify
uv run pytest
npm --prefix frontend/onboarding test
npm --prefix frontend/onboarding run build
```

These commands make no paid model calls. Frozen-study verification and the full Python suite require a Git clone with recorded history; a source ZIP or shallow clone is insufficient. Further commands: [VIF training](docs/vif/03_model_training.md) and [experiments and checks](docs/readme/status_and_setup.md).

## Repository map

| Path | Contents |
|---|---|
| [`frontend/onboarding/`](frontend/onboarding/) | React Onboarding, Experience, saved replay, and Inspect |
| [`src/`](src/) | Python API (`demo/`), Drift inspector (`drift_review_app/`), Weekly Drift Detection and Coach Digest (`coach/`), VIF Critic (`vif/`), data workflows, and evaluations |
| [`scripts/`](scripts/), [`config/`](config/), [`prompts/`](prompts/) | Experiment runners, data tools, study configurations, and prompt templates |
| [`logs/`](logs/) | Synthetic data, labels, annotations, saved predictions, and experiment reports |
| [`tests/`](tests/) | Python verification; frontend tests also live under `frontend/onboarding/` |
| [`docs/`](docs/) | [Report](docs/capstone_report/), [architecture](docs/architecture/e2e_architecture.md), [product scope](docs/prd.md), and guides; `archive/` and `future_work/` identify historical and proposed work |
