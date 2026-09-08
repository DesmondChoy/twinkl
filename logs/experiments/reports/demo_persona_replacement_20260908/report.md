# Demo Persona replacement and weekly comparison chooser

Date: 2026-09-08. Tracking issue: `twinkl-rklc.41`.

The saved demo now presents Nisha Agarwal, Noor Haddad, Lukas Vetter, Wei Jun
Chen, and Meera Krishnamurthy in one weekly comparison table. Lukas replaces
Lim Sook Yin; Meera replaces Henrik Larsson. The replacement covers only the
demo roster, runtime response registries, public replay bundles, and deployment
source allowlist. Research datasets and historical experiment reports remain
unchanged. No fallback demo bundles were retained.

## Final coverage

| Persona | Journal Entries | Weeks | Demonstration |
| --- | ---: | ---: | --- |
| Nisha Agarwal | 8 | 5 | Active Drift appears in week 4 and ends in week 5 |
| Noor Haddad | 12 | 6 | No Active Drift across six weeks |
| Lukas Vetter | 10 | 5 | One Universalism Drift stays active in weeks 1–4 and ends in week 5 |
| Wei Jun Chen | 11 | 6 | Insufficient Evidence in weeks 5–6 |
| Meera Krishnamurthy | 12 | 5 | Self-Direction has Active Drift in week 1 while Tradition has No Active Drift |

The roster contains 53 Journal Entries, 27 weeks, and three distinct Core Values.
Every week has a validated Coach Digest. The table reads each Core Value's
weekly state from the same saved Drift Detector event as the replay; Meera's
two Core Values appear in separate rows. Selection updates a detail panel and
one start/resume action. The mobile layout places details beneath the selected
Persona. Removed synthetic sessions return to the chooser only after a
successful catalog read; transient failures preserve saved progress.

## Source and generation provenance

Weekly Drift Detection uses Run 1 of
`logs/experiments/artifacts/twinkl_j3k7_core_value_definitions_20260907/`.
The replacement source identities are `a24b8d8f` and `961a4e3f`. Their original
Journal Entries and nudge responses are preserved from `logs/synthetic_data/`
and `logs/wrangled/`. Some saved nudges do not follow today's live spacing
rule. Replay traces therefore mark `policy_applied: false` and describe that
rule as a reference check. The live rule and its default remain unchanged.

The initial [frozen plan](plan.json) retained 17 compatible Coach responses and
scheduled ten missing weeks. Generation used `gpt-5.6-luna`, reasoning `none`,
service tier `default`, Coach prompt version `4.2`, and zero SDK retries.
Each case allowed at most two attempts within a batch. No seed was supplied;
the saved raw responses and exact rendered prompts are the reproducible
records, rather than a claim that a later model call will return identical text.

Seven cases passed on their first attempt. Meera's week beginning November 24
failed twice because its quoted punctuation differed from the supplied excerpt
and its response included the internal value label Tradition. Both failures
remain in [diagnostics](diagnostics/), including raw outputs and provider
receipts. The first batch stopped at its existing two-attempt limit.

A separate [completion plan](completion/plan.json) retained the resulting 24
responses and scheduled the remaining three. Its frozen
[repair requirements](repair_requirements.json) gave the failed case explicit
guidance about exact quotation punctuation and internal labels. All three
passed on their first attempt in that batch. The successful November 24
response was therefore its third attempt overall. No validator was relaxed and
no model output was manually rewritten. The optional repair instructions are
included in the actual trusted prompt and its stored hash.

Across both batches, 12 calls produced ten accepted responses and two preserved
rejections. Usage was 21,160 input tokens and 2,138 output tokens, with calculated
cost **US$0.00780415** and summed provider latency 52.94 seconds. These are new
Coach generation costs only. The [final summary](final_summary.json) includes
aggregate usage, final runtime file hashes, and scenario content hashes; the
two batch summaries retain their original stopping points.

North Star Moment records were reused from the exact-compatible `full_history`
cases in
`logs/experiments/reports/north_star_v4_run1_20260907/nsm_experiment.json`
(SHA-256 `2b86aecd7809892b6b15e660e613ddc546a816da025330283dbfb487660a31b9`).
The exporter verifies semantic inputs, ownership, chronology, policy, and exact
quotation source. All 17 retained records remain identical; ten outgoing records
were replaced by the ten Lukas/Meera records. Their original 13 model-attempt
receipts remain attached. No new North Star Moment calls were necessary.

The complete roster has 22 selected quotations, three ineligible outcomes, and
two completed reviews with no supportive source. The replacements contribute
eight quotation cards. Lukas week 1 has no supportive source; Meera week 1 has
no eligible earlier writing. These completed outcomes produce no empty card.
All 27 records have final status `complete` or `not_eligible`; none is pending
or rejected. This remains AI-reviewed synthetic development evidence, not
human validation.

## Commands and verification

Commands ran from the repository root with the virtual environment activated.
`UV_CACHE_DIR=/private/tmp/twinkl-uv-cache` was used for local cache access.

```sh
source .venv/bin/activate
uv run --no-sync python -m scripts.coach.complete_scenario_coach \
  --output logs/experiments/reports/demo_persona_replacement_20260908 --execute
uv run --no-sync python -m scripts.coach.complete_scenario_coach \
  --output logs/experiments/reports/demo_persona_replacement_20260908/completion \
  --repair-requirements logs/experiments/reports/demo_persona_replacement_20260908/repair_requirements.json \
  --execute
uv run --no-sync python -m src.demo.scenarios
uv run --no-sync python -m scripts.coach.complete_scenario_coach
uv run --no-sync pytest tests/demo tests/coach tests/nudge -q -o addopts=''
uv run --no-sync mypy --follow-imports=silent \
  src/demo/contracts.py src/demo/scenarios.py scripts/coach/complete_scenario_coach.py
```

The final backend run passed **252 tests**, with four existing deprecation
warnings. Ruff passed for every changed Python file. MyPy 2.3.0 passed the three
changed source files with imported-file diagnostics suppressed. The unrestricted
import traversal reports 65 errors across 19 unchanged dependency files; the
repository-wide type check is therefore not claimed as passing.

From `frontend/onboarding`, `npm test -- --run` passed **267 tests** and
`npm run build` passed TypeScript and the production build. The existing bundle
size warning remains. Deployment tests verify the new raw/wrangled source
allowlist and load the packaged replay sources without the full North Star
Moment study. A Docker image build and a deployed service were not exercised.

After the requested typography and layout refinement, all 52 relevant
PersonaReplay and PersonaReplayLoading tests and the TypeScript/production
build passed again. The default Coach completion command also passed without
provider calls; [its frozen plan](current/plan.json) retains all 27 responses
and has no missing cases.

Browser verification covered the full-width desktop comparison with selected
Persona details below and compact layouts at 1440, 1280, 1024, 820, 391, and
320 CSS pixels, with no horizontal overflow; all ten replacement weeks;
Meera's independent first-week Core Value decisions; valid Coach Digests;
North Star Moment cards and source links; Lukas's Inspect provenance; and
returning to the chooser with a single week-5 resume action. The temporary
viewport override was reset. Automated regressions additionally cover keyboard
selection, retired-session migration, transient catalog failure, saved-nudge
provenance, and catalog/source tampering. `git diff --check` passed.
