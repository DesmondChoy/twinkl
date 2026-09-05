# North Star Moment: Phase 0A retrieval feasibility

**Date:** 5 September 2026. **Issue:** `twinkl-fz34`.
**Branch:** `codex/north-star-moment`. **Status:** Baseline and encoder preflight
complete; retrieval gate passed at k=3. Phase 0A made no paid calls.

This report records independent work completed while the two required user
decisions were initially pending. The dated retrieval results below supersede
that preflight state. The [subsequent Phase 0B gate](../north_star_phase0b_20260905/README.md)
failed, preventing dependent application work. This report does not establish
selection accuracy, implemented cards, or successful browser integration.
The [specification](../../../../docs/north_star/north_star_moment.md) remains
the implementation authority. Existing Experience and Coach Digest behavior
has not changed.

## Reproduced baseline

[baseline.json](baseline.json) records the exact source hashes, script hash,
creation time, counts, and availability exclusions. The source is the existing
AI-reviewed synthetic development corpus; these are LLM-Judge VIF Labels,
not North Star Moment reference decisions.

| Measure | Reproduced count |
|---|---:|
| Known development Drifts | 42 |
| Any Journal Entry before the first Conflict | 34/42 (81.0%) |
| Earlier same-Core-Value persisted positive label | 26/42 (61.9%) |
| Earlier same-Core-Value consensus positive label | 26/42 (61.9%) |
| Active Drift at the final cutoff | 10 |
| Final-cutoff Active Drift with an earlier positive in each label file | 9/10 (90.0%) |
| Drift onset at stored order 0 | 8 |
| Drift onset at stored order 1 | 5 |
| Persisted/consensus disagreements across all labels | 1,368/16,510 (8.29%) |
| Disagreements among unique earlier same-Core-Value coordinates | 17/145 (11.72%) |
| Earlier positive coordinates, persisted / consensus | 68 / 64 |
| Consensus positives with five / four / three agreeing passes | 49 / 3 / 12 |

Every published baseline count matches. The earlier sources comprise 142 unique
Journal Entries. Of these, 64 contain a nudge response without its own timestamp
or event order. All 64 responses are excluded from prospective retrieval and
review; the original writing remains eligible when its date and stored order
both satisfy the onset boundary. The AI-written nudge is never an NSM source.
The baseline's labels can reflect writing excluded by this stricter source
boundary, which further limits their role as a retrieval proxy.

## Encoder preflight and hosting recommendation

[encoder_probe.json](encoder_probe.json) records a separate, reproducible local
process using two disclosed generic texts, without inspecting or assigning
evaluation histories. The fixed encoder and custom model code load from cache
with network access disabled.

| Setting or measurement | Value |
|---|---|
| Encoder | `nomic-ai/nomic-embed-text-v1.5` |
| Model revision | `e9b6763023c676ca8431644204f50c2b100d9aab` |
| Custom code revision | `7710840340a098cfb869c4f65e87cf2b1b70caca` |
| Device / threads | CPU / 10 |
| Representation | Layer normalization, 256-dimensional truncation, L2 normalization |
| Platform | macOS 26.6.2, arm64; Python 3.12.11 |
| Library imports | 1.872 s |
| Cached model load | 1.270 s |
| First / warm two-text encode | 34.0 / 18.1 ms |
| Complete timed probe | 3.214 s |
| Peak process RSS | 1,533.8 MiB |

These measurements exclude download and container build time and do not
represent full-history latency or Railway/Linux capacity. The local cache
contains approximately 548 MB of model blobs. Installed package versions are
recorded in the JSON result. An earlier exploratory process had different
import timings and peak RSS; the table reports the checked-in probe's result.

The recommended capstone architecture is one lazily loaded, revision-pinned
CPU encoder in the existing backend, with encoding outside the asyncio event
loop and bounded concurrency. It avoids a second deployment and network hop.
A separate inference service isolates model startup and allows independent
scaling, but retains the encoder's memory requirement while adding another
process, authentication, health checks, network failures, and operational work.
The current POC does not yet demonstrate a need for that separation.

This local hosting choice was adopted in the frozen policy after the retrieval
gate passed; it remains unimplemented because Phase 0B failed. The current
`requirements-experience.txt` omits Torch,
Sentence Transformers, and einops. Live integration would require pinned
runtime dependencies and cached model/code artifacts, one model copy per
process, representative history measurements, and verification of the actual
container configuration. No infrastructure purchase or deployment is authorized
by this recommendation.

The expanded scope is estimated at 11–18 working days, conditional on passing
feasibility and resolving provider decisions. This planning estimate extends
the former 7–11-day saved-replay estimate by approximately 4–7 days for live
availability, concurrency, retry/invalidation, packaging, fresh onboarding QC,
and reporting; it is not a measured throughput claim.

## Adopted decisions and paid envelope

The user approved Decision 11's small separate benchmark on 5 September 2026
before assignment or ranking. Seed 20260905 and SHA-256 identifier ordering
reserved eight complete non-demo Persona histories (nine Drift episodes);
27 development Personas supply 33 episodes. All five saved Personas remain
outside the reserved group. The [cohort](cohort.json) records approval, source
hash, identities, and selection method. The reserved histories remain unused
for NSM development or final evaluation.

The approved small-benchmark budget is **US$20 total**, **US$0.25 per attempt**,
and **at most one retry** per request. The initial estimate was US$5–12,
covering development, all five saved Persona preparations, runtime and
independent reference review, adjudication, onboarding/browser calls, final
evaluation, and retries. Phase 0B actually used US$0.2062 before the failed gate
stopped dependent work; the unused ceiling does not justify further calls.

The planning envelope is roughly 150 OpenAI attempts across NSM and existing
live application calls, 60 Gemini reference/adjudication attempts, and a retry
reserve within the same total ceiling. At an illustrative 6,000 input and
2,000 output tokens per OpenAI attempt, and 10,000 input and 6,000 output tokens
per Gemini attempt, those calls cost approximately US$4.68 before retries and
variation. Actual case count, sampling seed, provider configuration, token
caps, and attempt reservations must be frozen before paid work. The larger
benchmark option requires a revised estimate within an agreed ceiling.

Frozen experimental review model: `gpt-5.6-luna`, reasoning `none`, standard
service tier. Separate reference provider: `gemini-3.5-flash`, thinking `low`.
Published standard prices checked on 5 September 2026 are US$0.20 input and
US$1.20 output per million Luna tokens, and US$1.50 input and US$9.00 output
(including thinking) per million Gemini tokens. Sources:
[OpenAI](https://developers.openai.com/api/docs/models/gpt-5.6-luna) and
[Google](https://ai.google.dev/gemini-api/docs/pricing).

The implemented offline provider adapter disables implicit SDK retries,
reserves a conservative upper bound before each attempt, records model/usage/
cost, and stops at the agreed ceiling. Both providers were exercised in Phase
0B. The [policy](../../../../config/evals/north_star_moment_v1.json) records
the exact limits and prices; the existing Coach adapter remains unchanged.

## Frozen retrieval result

[retrieval_config.json](retrieval_config.json) was written before encoding;
[retrieval.json](retrieval.json) contains all development rankings and source
hashes. The primary proxy is the persisted positive LLM-Judge VIF Label. Of 33
development episodes, 22 have at least one eligible positive-label source.

| Retrieved sources | Positive-label history hit | Recall |
|---|---:|---:|
| 1 | 13/22 | 59.1% |
| 3 | 21/22 | 95.5% |
| 5 | 22/22 | 100.0% |

The smallest passing setting is **k=3**, frozen for Phase 0B. Encoding the
119 unique eligible original Journal Entries took 2.174 seconds; the complete
retrieval measurement took 5.482 seconds and reached 1,584.1 MiB peak RSS.
These are local CPU measurements, not live-service latency or container limits.
Consensus agreement and persisted-label disagreement diagnostics are retained
in the JSON. Labels were joined only after meaning-based ranking.

## Reproduction and validation

From the repository root, reproduce the unpaid work:

```sh
source .venv/bin/activate
export UV_CACHE_DIR=/tmp/twinkl-uv-cache
uv run python scripts/experiments/north_star_phase0.py \
  --output logs/experiments/reports/north_star_phase0_20260905/baseline.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  uv run python scripts/experiments/north_star_encoder_probe.py \
  --output logs/experiments/reports/north_star_phase0_20260905/encoder_probe.json
uv run pytest tests/evals/test_north_star_phase0.py
uv run ruff check scripts/experiments/north_star_phase0.py \
  scripts/experiments/north_star_encoder_probe.py tests/evals/test_north_star_phase0.py
uv run --with 'mypy==2.3.0' mypy --follow-imports=silent \
  scripts/experiments/north_star_phase0.py scripts/experiments/north_star_encoder_probe.py
```

The offline probe requires the two recorded Hugging Face revisions and their
dependencies to exist in the local cache. It does not download replacements.

The retrieval command additionally takes `--cohort path/to/cohort.json`.
The manifest must record `evaluation_scope`, `decision_source`, `frozen_at`,
`development_persona_ids`, `reserved_persona_ids`, and optional
`excluded_personas` mapping identifiers to reasons. It writes
`retrieval_config.json` beside that manifest before encoding. Queries use only
`user_phrase` and `definition` from `config/schwartz_values.yaml`; no generation
examples, biography, labels, or current Conflict text enter the encoder. The
recorded document template is also the rendering template. Labels are joined
to ranked identifiers after encoding. Similarity ties use recency, then stable
identifier order. The smallest k among 1, 3, and 5 reaching 90% persisted-label
proxy recall is selected; no passing k produces exit code 2 and stops dependent
work. Empty denominators remain undefined. Consensus-positive agreement groups
and persisted-positive disagreements are reported as separate diagnostics.

Validation: 14 targeted tests, Ruff, and isolated MyPy pass. The ordinary MyPy
invocation also reports five pre-existing errors in imported
`parse_wrangled_data.py`, `parse_synthetic_data.py`, and `registry/personas.py`;
these are outside the changed experiment scripts and remain unresolved.
Fresh review found and corrected missing reserved-identity/time validation
and a difference between the saved and rendered document template.

## Remaining implementation and evidence

The 90% retrieval gate passed; the subsequent development AI review failed.
The pure review contract and budgeted provider adapter are implemented and
tested for the offline experiment. Offline Persona bundles, live lifecycle,
React cards, NSM Inspect, and source navigation are not implemented. Browser QC
at narrow and wide widths, screenshots, fresh onboarding, and final evaluation
remain blocked. The Technical Paper and generated PDF now report the earned
feasibility results, with an offline evidence walkthrough separate from UI
validation.

The live service currently holds one global lock across provider awaits.
NSM must publish successful weekly results first, snapshot and reserve work
briefly under the lock, encode/review outside it, and validate the input hash
before publishing. New response availability evidence, request coalescing,
successful no-card reuse, bounded independent retry, resume, source invalidation,
and deletion must be tested together. None of these guarantees follows from
the baseline or the encoder smoke test.
