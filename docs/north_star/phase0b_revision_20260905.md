# North Star Moment Phase 0B development revision

**Issue:** `twinkl-fz34.1`. **Protocol:** `north-star-development-v3`.
**Decision date:** 5 September 2026.

The user requested implementation of `twinkl-fz34.1` after the runner-hardening
dependency closed. This resumes one separately frozen development revision
under the existing US$20 cumulative, US$0.25 per-attempt, and one-retry limits.
The [North Star Moment specification](north_star_moment.md) remains the product
authority. A failed development gate leaves dependent implementation blocked.

## Evidence and choice

The original [Phase 0B run](../../logs/experiments/reports/north_star_phase0b_20260905/README.md)
had seven selected quotations rejected or unresolved by independent AI review,
two task-reference retrieval misses, and one case whose two attempts paired
`abstain` with `other_actor`. The original files and executed code remain
unchanged. The [offline diagnostics](../../logs/experiments/reports/north_star_phase0b_revision_20260905/diagnostics.md)
separate mechanically derived counts from interpretations of the disagreements.

| Option | Benefit | Limitation and decision |
|---|---|---|
| Revise review instructions and remove redundant decision fields | Directly addresses action specificity, requested-value relevance, whole-entry Conflict, and the reproduced response contradiction | Selected for this revision. Semantic correctness still needs independent review. |
| Change encoder/query at the same top-3 candidate budget | Could improve the two task-reference misses | Does not explain the seven false selections. Defer until review quality is measured separately. |
| Increase k, add BM25/hybrid retrieval, or use bounded agentic search | Could find valid examples ranked below three | Adds candidate work or architecture. Top-5 fixes only one of the two known misses. No such change is adopted here. |

Keep the pinned Nomic encoder, query, ranking, k=3, provider models, reasoning
settings, and selection order fixed. The revised runtime schema requires brief
factual assessments of the writer's action, its relationship to the approved
Core Value definition, and Conflict against that same Core Value across the
complete supplied writing. The model returns one reason code; code derives
`supportive`, `not_supportive`, or `abstain`. Exact quotation, source membership,
complete-batch, and internal-value-label checks remain in the original validator.
An already-made choice can qualify; a desirable state or merely imagined future
action cannot establish that choice. Discomfort or another value's Conflict
does not by itself establish Conflict against the requested Core Value.

This is an experimental review revision. It does not change the Profile,
Weekly Drift Detection, Coach Digest, application service, or React contracts.

## Frozen inputs and reference rules

Use all 33 original development Drift episodes from 27 Personas, including the
five without eligible earlier writing. Preserve all eight reserved Persona
histories; input verification may hash their bytes but must not parse their
writing. Use only the independently available original Journal Entries in the
frozen retrieval manifest. Legacy nudge responses remain excluded. No biography,
generation instructions, labels, or independent reference decisions enter the
runtime request.

Reuse the original complete Gemini exhaustive reference requests and raw
receipts after validating the exact request hash, source membership, approved
definition, and response contract. Require all such references before new paid
work. Do not regenerate or revise references after seeing new runtime answers.

Some frozen AI judgments are unresolved. Five of nine histories with no
reference-accepted source contain abstentions, and two contain only abstentions.
Report those strata while preserving the original nine-case omission denominator;
do not present them as proof that no supportive action occurred. Identical
Security writing at `87e92805:entry:4` received supportive and conflicting
decisions in different episodes. Before execution, mark identical source/value
groups with supportive-versus-nonsupportive reference disagreement as unresolved.
A quotation from such a group cannot count as accepted in either comparison
path. Keep the original per-case decisions visible rather than relabeling them.

For each runtime selection, require both a supportive primary reference and
approval of the exact proposed quotation in its full source context. A quotation
identical to the primary reference quotation already has that approval. Otherwise
use the existing predefined candidate-quotation Gemini check; it must not replace
the proposed text with a different passage. Rejection, abstention, contradiction,
missing response, or invalid response cannot approve the selection. Reference
results never alter the runtime selection or cause fallback to another source.

## Matched comparison and measurements

The retrieval-only display rule is fixed before execution: quote the entire
original Journal Entry at rank one when it passes deterministic quotation
checks. Omit it when those checks fail. Grade that exact quotation with the same
primary-support, contradiction, and candidate-quotation rules used for runtime
selections. Never substitute the primary reference's shorter accepted passage.
This corrects the original comparison of source-level and quotation-level
precision. The two methods can display different quotations and cover different
numbers of cases; their precision difference is descriptive, not a causal effect.

Report counts and denominators for:

- Exact-quotation precision for both methods, plus their difference when both
  denominators are nonzero. Zero displayed quotations means undefined precision.
- Correct omission on all nine nonempty histories with no primary
  reference-accepted source, with resolved-rejection and abstention strata.
- Coverage over all 33 development episodes, including the five empty histories.
  This is development episode coverage. It does not measure the future closed-week
  population under application Core Value priority; that belongs to integration.
- Frozen task-reference retrieval recall at k=3, separately from label-proxy
  recall. Report unresolved source-reference groups separately.
- Runtime abstentions among valid reviewed-source decisions, selected quotations
  rejected with `wrong_value`, and unresolved-reference selections. A wrong-value
  reason alone does not prove support for another specific Core Value.
- Failed cases and unexpected provider-attempt failures, including invalid
  attempts before a successful retry. Distinguish each provider and reused
  historical receipts from new actual attempts. Pending reservations retain
  unknown outcomes and their reserved cost.
- Source identity, chronology, and quotation checks; all input and displayed
  quotation checks must pass. Synthetic injected failures are separate evidence.
- Usage, calculated cost or retained reservation, model identities, and latency
  for every attempt, and cumulative spending across the original and revised runs.

The original gates remain: zero incorrect selections, 100% correct omission with
a nonempty denominator, zero quotation/chronology/identity failures, at most 5%
unexpected provider failures, and at least one accepted saved-Persona example.
Every case and required reference check must finish successfully. A provider with
no new attempts has no new failure-rate estimate; do not fabricate a denominator.
All-zero display coverage cannot pass. A failed gate remains failed and blocks
dependent implementation; the reserved final benchmark is not used to tune it.

## Execution and accounting

The [v3 runner](../../scripts/experiments/north_star_phase0b_v3.py) uses the
existing v2 source verification, request recovery, and locked cumulative ledger.
Preparation binds the complete development manifest, original evidence tree,
executed code, this protocol, budget seed, and policy by hash. It refuses changed
source text, incomplete references, missing inherited attempts, unsupported
provider settings, or a worst-case protocol cost above the remaining envelope.

The cost preflight uses the provider's conservative UTF-8 input bound, schema
margin, and maximum output allowance. It includes one runtime request, one
baseline quotation check, and one selected-quotation check per nonempty case,
each with at most one retry. Full source text bounds any selected substring;
no cache discount or successful reuse is needed to fit the envelope. Actual
unneeded or identical completed requests are omitted or reused. The resulting
per-case and cumulative bounds are frozen in `manifest.json` before `--run`.
SDK retries remain disabled. All paid v2/v3 runs share
`logs/experiments/north_star_development_budget_v2.json` and its origin record.

```sh
source .venv/bin/activate
export UV_CACHE_DIR=/private/tmp/twinkl-nsm-v3-uv
uv run python scripts/experiments/north_star_phase0b_v3.py --prepare \
  --directory logs/experiments/reports/north_star_phase0b_revision_20260905/run \
  --prior-ledger logs/experiments/reports/north_star_phase0b_20260905/budget.json
uv run python scripts/experiments/north_star_phase0b_v3.py --run \
  --directory logs/experiments/reports/north_star_phase0b_revision_20260905/run
```

Before preparation of any later separately approved run, supply the latest
cumulative ledger instead of the historical seed. `--replay` reconstructs the
frozen revision without provider transport. Missing or exhausted work stays a
failed case; it cannot silently restart its retry allowance. Exit code 2 means
the gate failed, including when work remains missing. Preserve that outcome and
its receipts. Report the result in the separate revision directory and the
maintained experiment log; do not overwrite the original failed study.
