# Saved demo v4 Run 1 integration

The saved demo consumes Weekly Drift v4 Run 1 and the full-history runtime
observations from the [completed targeted NSM study](../north_star_v4_run1_20260907/report.md).
The five Personas cover 27 reviewed weeks and 52 Journal Entries. Their NSM
outcomes are 23 selected quotations, three ineligible outcomes, and one
completed review with no supportive source. These are AI assessments of
synthetic writing; replay does not substitute evaluation grades for runtime
observations or establish human validity.

| Persona | Scenario | Key week | Demonstration |
| --- | --- | --- | --- |
| Noor Haddad | `stable-noor` | 2025-05-19 | No Active Drift |
| Nisha Agarwal | `active-nisha` | 2025-03-03 | Active Drift |
| Lim Sook Yin | `ended-sook-yin` | 2025-02-10 | Historical Drift Record ends |
| Wei Jun Chen | `uncertain-wei-jun` | 2025-06-30 | Insufficient Evidence |
| Henrik Larsson | `two-values-henrik` | 2025-02-17 | Different states across two Core Values |

Five fresh Coach Digest responses are now attached to the curated key weeks.
All five passed Coach Digest Validations using prompt v4.2 and the current
`gpt-5.6-luna` Coach Digest setting, reasoning `none`. Seven calls were made:
Noor and Wei Jun each required one validation-guided retry. The recorded
usage implies US$0.00434595 at the configured published rates; this is not a
billing export. See the [generation report](report.md),
[execution record](execution.json), and [exact displayed sample](judge_sample_manifest.json).

The [prepared sources](prepared_sources.json) and `prepared_inputs/` preserve
inputs from before generation. `generated_responses/` contains the accepted
response JSON; exact accepted prompts remain in the generation receipts in
`judge_sample_manifest.json` and `src/demo/coach_digest_responses.json`.
`generation_diagnostics/` retains all seven attempts,
including the two rejected drafts. Generation completed before an outdated
4.1-only saved-response contract stopped export. The contract now accepts the
current 4.2 version, and export resumed from saved responses without more calls.

Duplicate Markdown renderings, standalone accepted prompts, the matching
Parquet export, and its temporary lock were removed before publication. Their
contents remain in the retained JSON inputs, responses, and receipts. The
generation command below recreates these optional exports during a separately
authorized rerun.

The assistant initially described the calls as Luna low. The actual Coach
Digest contract and receipts use `none`; this distinction was corrected to the
user. Weekly Drift and NSM continue to use their separate low-reasoning settings.

To regenerate the completed experiment export without provider calls:

```sh
source .venv/bin/activate
uv run python -m scripts.export_demo_experiments
```

For a separately authorized rerun, the existing generator can produce five source-bound
Coach Digest responses, with at most one validation-guided retry per Persona.
The completed run disabled SDK retries with a process-local wrapper setting
`AsyncOpenAI(max_retries=0)`; the wrapper is recorded in `execution.json`.
It used the saved Weekly Drift Detection outputs and made no Weekly Drift
Reviewer or NSM calls:

```sh
source .venv/bin/activate
TWINKL_COACH_PROVIDER=openai TWINKL_COACH_MODEL=gpt-5.6-luna \
uv run python scripts/coach/generate_approved_judge_sample.py \
  --personas 02fb94f3 5fa8b540 ed67c9cc 8f83c818 2d928d8a \
  --reuse-scenario-key-weeks \
  --manifest-out logs/experiments/reports/demo_v4_run1_20260907/judge_sample_manifest.json \
  --parquet-path logs/experiments/reports/demo_v4_run1_20260907/weekly_digests.parquet \
  --execute
```

The generator applies Coach Digest Validations. It does not perform Coach
Digest Evals or human review. August 2026 generation/evaluation reports remain
historical evidence and must not be presented as evaluation of these new
responses.
