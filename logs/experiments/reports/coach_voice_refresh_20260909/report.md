# Saved Coach Digest voice refresh — 9 September 2026

The 27 saved Coach Digest responses for Nisha Agarwal, Noor Haddad, Lukas
Vetter, Wei Jun Chen, and Meera Krishnamurthy were regenerated from their
existing Weekly Drift Detection inputs using OpenAI `gpt-5.6-luna`, reasoning
effort `none`, and prompt `4.4`. The user authorized this generation and
replacement of validated saved responses. The original response fixture is
preserved byte-for-byte in
[`original_coach_digest_responses.json`](original_coach_digest_responses.json).
Weekly Drift Reviewer Decisions, Drift Detector results, Journal Entries, and
North Star Moment selections are outside the refresh scope.

The prompt asks for a concrete opening addressed to the person, ordinary
language instead of clinical findings, and one short reflective question.
New generation and live Experience validation reject recap openings and a
narrow set of clinical finding phrases outside quotations. Historical
responses retain their original validation rules. These checks supplement
groundedness, score-language, value-label, current-state-claim, and length
checks; they cannot establish conversational quality or semantic correctness.

## Generation and editorial review

The initial run accepted all 27 cases after 29 provider calls. Two first
attempts failed groundedness and were regenerated with the recorded failure
instructions. The [summary](summary.json) records 54,362 input tokens, 4,910
output tokens, and a calculated cost of US$0.01289233 using the repository's
published-rate calculation, including reported cache usage. Inputs, exact
requests, raw responses, validation results, usage, and response identifiers
are retained under `inputs/`, `requests/`, `diagnostics/`, and `responses/`.
The [plan](plan.json) records input, prompt, source-bundle, and code hashes.
No sampling seed was supplied; new model calls are nondeterministic.

An AI editorial review read all 27 responses against their supplied inputs and
the full synthetic Journal Entries. It identified six responses for targeted
repair, while finding no direct contradiction with the selected North Star
Moment quotations. The repair keeps the same model, base prompt, and weekly
inputs, adds explicit editorial constraints, and retains the other 21 response
objects. Its requests and results are recorded separately in
[`coach_voice_repair_20260909`](../coach_voice_repair_20260909/).

| Persona and week start | Required correction |
| --- | --- |
| Nisha, 17 February | Replace an unfinished quoted fragment with a meaningful supplied phrase. |
| Nisha, 24 February | Remove the unsupported link between refusing furniture and fairness; use a complete supplied quotation. |
| Nisha, 3 March | Open with a concrete supplied moment instead of an unfinished quotation. |
| Noor, 28 April | Replace the unfinished quotation and describe Noor directly. |
| Wei Jun, 23 June | Preserve “cut corners on data validation”; avoid declaring that no decision was stated or asking for a decision already described in the full Journal Entry. |
| Wei Jun, 30 June | Distinguish Xiao Yu's question from the writer's answer and avoid implying that he had never raised the concern. |

The six-case repair required eight calls. Two responses still needed editorial
corrections: Nisha's 24 February response retained an unfinished quotation,
and Wei Jun's 30 June response continued to describe missing context. A
[two-case pass](../coach_voice_final_20260909/summary.json) corrected Nisha while
retaining the other 25 response objects. A
[one-case pass](../coach_voice_last_20260909/summary.json) then corrected Wei Jun
while retaining the other 26. Positive instructions to acknowledge the supplied
pressures and express openness through the final question resolved the remaining
voice problem. Automated wording checks had accepted those earlier attempts;
the AI editorial review remained necessary.

Across all four runs, 40 provider calls used 74,171 input tokens and 6,743 output
tokens, with a total calculated cost of **US$0.02004253**. The final
[response fixture](../coach_voice_last_20260909/refreshed_coach_digest_responses.json)
was applied to `src/demo/coach_digest_responses.json`, then exported into all
five saved Persona bundles and their hash catalog. All 27 current responses use
prompt `4.4` and pass Coach Digest Validations. Prior plans, accepted responses,
rejected attempts, and their original receipts remain available.

Some initial errors arose because the Coach Digest receives selected excerpts,
whereas North Star Moment assesses complete eligible writing. The repairs
avoid unsupported claims without silently adding the omitted text to the Coach
inputs. This does not resolve the broader source-context and semantic-review
follow-up tracked in `twinkl-rklc.39` or change the frozen North Star Moment
experiment. This review is AI editorial work, not new Coach Digest Evals,
human validation, or evidence of user benefit.

## Commands and verification

The initial generation used:

```sh
source .venv/bin/activate
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run python -m scripts.coach.refresh_scenario_coach --execute
```

The final accepted fixture was applied without further provider calls:

```sh
UV_CACHE_DIR=/tmp/twinkl-uv-cache uv run python -m scripts.coach.refresh_scenario_coach \
  --output logs/experiments/reports/coach_voice_last_20260909 \
  --prior-run logs/experiments/reports/coach_voice_final_20260909 \
  --repair-requirements logs/experiments/reports/coach_voice_last_20260909/editorial_requirements.json \
  --apply
```

The runner stages accepted responses and requires complete validated coverage
before `--apply` can replace the active fixture. A fresh generation should use
a new output directory; frozen plans reject changes to their inputs or policy.
The export changes source-bundle hashes, so the pre-export frozen plan is a
receipt of this run, not a command to rerun against the newly exported catalog.
Saved Persona bundles and their SHA-256 catalog are exported without provider
calls using:

```sh
source .venv/bin/activate
uv run python -m src.demo.scenarios
```

The associated UI changes add more space above the selected Persona panel,
two finite reduced-motion-aware sparkle cues, 50-word Journal Entry previews,
and a centered reading dialog. The Coach Digest uses natural North Star Moment
introductions and labels its final user-directed question **Something to
reflect on**. Inspect now exposes the supplied writing, exact prompts,
runtime AI source assessments, and deterministic selection and validation.
These interface changes do not regenerate North Star Moment selections or
rewrite quoted Journal Entry text.

Comparison with the pre-change Git revision confirmed that all 53 Journal
Entries, 27 Weekly Drift Reviewer results, and 27 Drift Detector results are
unchanged. All 27 North Star Moment records preserve their inputs, prompts,
provider receipts, assessments, validations, and selections. Only their replay
start and completion timestamps move with the refreshed Coach Digest duration.
The original response fixture backup is byte-identical to the pre-change file.

Verification included all 335 frontend tests and the production TypeScript/Vite
build. The affected Python run covered Coach Digest, North Star Moment, and
Experience/scenario tests: **515 passed**. Its historical preservation assertion
was updated to verify the original response backup before the final passing run.
Scoped Ruff checks pass. The changed runner passes isolated MyPy; broader MyPy
still reports six pre-existing errors in `weekly_digest.py`, reproduced against
unchanged HEAD. The deprecated runtime test file retains 13 pre-existing Ruff
line-length violations. The build retains its large-chunk warning. No full
repository Python suite was run.

Browser checks covered desktop and a 391-CSS-pixel phone viewport: chooser
spacing and finite animation, longer previews, centered reading and keyboard
dismissal, refreshed reflection text, the exact North Star Moment quotation
before the labelled question, and the four-stage Inspect explanation. The
refreshed card fits the phone width without horizontal overflow. Verification
was performed locally; no deployment was performed as part of this refresh.
