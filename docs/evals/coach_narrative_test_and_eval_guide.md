# Weekly Coach Narrative & Weekly Digest — Test and Eval Guide

How to run every automated test and evaluation that covers the Weekly Digest and
the Weekly Coach narrative (`weekly_mirror`, `tension_explanation`,
`reflective_question`). This is the operational companion to
[`explanation_quality_eval.md`](./explanation_quality_eval.md), which defines the
tiered evaluation design; this file is the runbook.

All commands assume the virtual environment is active and you are at the repo
root:

```sh
source .venv/bin/activate        # Bash/Zsh
# source .venv/bin/activate.fish # Fish
```

---

## What covers what

| Layer | Kind | LLM calls? | Where |
| --- | --- | --- | --- |
| Weekly Digest builders + rendering | Unit tests | No (mocked) | `tests/coach/test_weekly_digest.py`, `tests/coach/test_runtime.py`, `tests/coach/test_weekly_drift_runtime.py` |
| Tier-1 validation checks (groundedness, non_circularity, value_leakage, length) | Unit tests | No | `tests/coach/test_weekly_digest.py` |
| Tier-1 batch report over a real Weekly Digest set | Eval | No | `src/evals/coach_narrative_tier1.py` |
| Tier-2 LLM-as-judge (correctness, specificity, non-prescriptive tone, tension honesty) | Eval | **Yes (paid)** | `src/evals/coach_narrative_judge.py` |

The Tier-1 checks are mechanical code checks, not human validation. The Tier-2
scores are **LLM-as-judge, not human validation**. Tier-3 human calibration
(Cohen's κ) is future work.

---

## 1. Unit tests (no API calls)

These run offline; every LLM call is mocked with a fixture.

Run all Weekly Coach and Weekly Digest unit tests:

```sh
uv run pytest tests/coach
```

Run just the Weekly Digest builder + Tier-1 validation tests:

```sh
uv run pytest tests/coach/test_weekly_digest.py
```

Run only the Tier-1 validation tests (groundedness, non_circularity,
value_leakage, length — both pass and fail paths):

```sh
uv run pytest tests/coach/test_weekly_digest.py -k validation
```

Run the eval-module unit tests (Tier-1 batch aggregator and Tier-2 judge
plumbing, both mocked):

```sh
uv run pytest tests/evals/test_coach_narrative_tier1.py \
              tests/evals/test_coach_narrative_judge.py
```

Lint and type-check touched code when you change it:

```sh
uv run ruff check src/evals src/coach tests/coach tests/evals
uv run mypy src/evals            # when type behavior changed
```

---

## 2. Tier-1 batch eval (no API calls)

Runs `validate_weekly_digest_narrative()` over every narrative in a persisted
Weekly Digest parquet and reports per-check pass rates against the targets in
[`explanation_quality_eval.md`](./explanation_quality_eval.md).

```sh
uv run python -m src.evals.coach_narrative_tier1 \
  --parquet logs/exports/weekly_digests/weekly_digests.parquet \
  --out logs/experiments/reports/coach_narrative_tier1_20260727
```

- `--parquet` defaults to `logs/exports/weekly_digests/weekly_digests.parquet`.
- `--out` writes `metrics.json` and `report.md`; omit it to only print the
  summary.
- Rows with no narrative are skipped; unparseable narratives are reported under
  `skipped_persona_weeks`.

Pass-rate targets: groundedness > 70%, non_circularity > 95%, length > 90%.
`value_leakage` has no published target and is reported for information.

---

## 3. Tier-2 LLM-as-judge eval (paid API calls)

Scores narratives on correctness, specificity, non-prescriptive tone, and
tension honesty, and flags whether the reflective question is open-ended. Judges
a fixed sample described by a manifest. **Makes paid calls and is gated behind
`--execute`.**

### 3a. Build the sample (approved path)

The judge should score narratives as the approved path actually produces them,
not leftover demo-tool outputs. Regenerate the fixed persona roster through
`run_weekly_drift_coach_cycle` and rebuild the manifest:

```sh
# Dry run — prints the plan, makes no calls:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py
# OR
uv run python scripts/coach/generate_approved_judge_sample.py

# Real run — paid Weekly Drift Reviewer calls (one per week of history per
# persona) plus one Weekly Coach narrative call per persona; overwrites their
# rows in the Weekly Digest parquet, then rebuilds the judge manifest:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py --execute

# Smoke-test one persona first:
.venv/bin/python scripts/coach/generate_approved_judge_sample.py \
  --personas 7cc5cf92 --execute
```

The Weekly Coach provider comes from `TWINKL_COACH_PROVIDER` (`openai` or
`gemini`; default `openai`). Set the matching API key in `.env`
(`OPENAI_API_KEY` and/or `GEMINI_API_KEY`). The Weekly Drift Reviewer uses
OpenAI.

### 3b. Run the judge

```sh
# Dry run — prints the plan, makes no calls:
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/coach_narrative_tier1_20260727/judge_sample_manifest.json

# Real run — paid judge calls; writes metrics.json and report.md:
uv run python -m src.evals.coach_narrative_judge \
  --manifest logs/experiments/reports/coach_narrative_tier1_20260727/judge_sample_manifest.json \
  --out logs/experiments/reports/coach_narrative_tier1_20260727 \
  --execute
```

Targets: mean > 3.5/5 per dimension. Any narrative scoring below 3 on any
dimension is flagged for human review. Report and doc lines label these scores
as LLM-as-judge, not human validation.

---

## Provenance and honesty notes

- Record the parquet source, sample manifest, judge provider/model, and row/
  sample counts with every committed report.
- Do not treat LLM-as-judge scores as human validation. State the source
  wherever it affects the conclusion.
- The Weekly Digest parquet under `logs/exports/weekly_digests/` is a local,
  gitignored artifact; regenerate it via the approved path rather than assuming
  its rows reflect current behavior.

---

## References

- [`explanation_quality_eval.md`](./explanation_quality_eval.md) — tiered
  evaluation design and status
- [`overview.md`](./overview.md) — where explanation quality sits in the VIF
  evaluation flow
- `src/coach/weekly_digest.py` — Weekly Digest builders,
  `validate_weekly_digest_narrative()`, and Weekly Coach generation
- `src/coach/weekly_drift_runtime.py` — the approved Weekly Drift Reviewer and
  Drift Detector runtime
