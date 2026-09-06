# North Star Moment: fresh Luna evaluation

The user authorized deleting the earlier North Star Moment experiment outputs
and repairing the evaluation on 5 September 2026. This protocol replaces those
outputs. It retains the previously approved development/reserved assignment,
source corpus, 16,000-token input limit and paid envelope. Previously incurred
spending is carried forward as a single amount, US$0.34732602; previous model
responses, labels, scores and runtime choices are not reused.

## Frozen inputs and models

The [cohort](../../config/evals/north_star_cohort.json) supplies 27 development
Personas and eight reserved Personas. The runner filters the shared episode
table to development identifiers before parsing writing. It includes every
nonempty original Journal Entry with stored order before the Drift onset and
date no later than onset. Nudge responses lack independent availability
evidence and remain excluded. Each included text is checked verbatim against
its synthetic source. Only the requested value identifier, user-facing phrase,
approved definition, source identifiers and eligible writing enter prompts.

The runtime model is `gpt-5.6-luna` with `low` reasoning. It reviews all eligible
writing, newest first. The application selects the first supportive result in
that order. The reference evaluator is `gpt-5.6-luna` with `xhigh` reasoning.
It reviews each unique source/value pair once, in deterministic batches of up
to four sources for the same value. Repeated occurrences across episodes share
that reference assessment. No runtime choices or explanations enter this stage.

Both roles use the exact same source rubric and response schema. The rubric
separately assesses actual writer action, support for the approved definition,
and any actual writer behavior against that same value. It explicitly addresses
joint actions, completed decisions, repeated actions, intentions, outcomes,
negative emotions and external obstacles. Code derives the decision from the
reason and checks membership, identifiers and exact quotation fidelity.

A separate `xhigh` call assesses each exact runtime-selected quote with its full
source, requested value and approved definition. It first independently assesses
the source, then the quote itself. It receives no other judgments or runtime
reasoning and cannot substitute a better quote. A supportive source alone does
not validate an outcome-only or otherwise unsupported quotation.

The model settings, current price assumptions and limits are recorded in the
[policy](../../config/evals/north_star_luna_20260905.json). Every complete input,
including instructions and response schema, is measured with the OpenAI token
counting endpoint before generation. Inputs above 16,000 tokens stop the run;
no truncation or reduced history is permitted. Generation has a 32,768-token
output ceiling including reasoning. The provider ledger allows at most two
attempts per request, disables SDK retries, reserves incomplete/unmetered costs,
and enforces US$0.25 per attempt and US$20 cumulatively including prior spending.
Preparation also checks a conservative bound for the entire protocol and retries.

## Scoring and independent review

A selected quote is accepted only when the primary source judgment is supportive,
the separate source assessment agrees, and the exact-quote assessment accepts it.
Source-level disagreement or abstention is unresolved, not a confidently
incorrect runtime selection. Agreement on source rejection yields a rejected selection. All such
outcomes remain in the selected-quote denominator, with separate counts.

The correct-omission denominator includes only nonempty histories whose complete
reference assessments all reject support without abstention. Histories with no
positive reference and any abstention or missing review remain unresolved.
Structurally empty histories are reported separately. A failed runtime request
does not count as a correct omission. Coverage has no minimum threshold.

The adopted strict gate requires every selection accepted, correct omission in
every confirmed no-example history, zero invalid response-contract attempts,
no unresolved no-positive histories or failed cases, at most 5% unexpected failed
attempts, and at least one accepted saved-Persona case. Quotation, identity and
chronology are enforced by construction; these checks do not prove semantics.
An unresolved judgment prevents declaring the strict gate met. No result of this
development experiment alone authorizes frontend integration or final evaluation.

After execution, independent reviewers assess the same complete sources and
exact selected quotes using the frozen rubric. Their input file excludes all
runtime explanations and evaluator decisions. Review assignments and judgments
are saved before comparison. Disagreements are reported with source evidence;
independent opinions do not silently replace the evaluator's recorded judgments
or official scores. Both passes are AI assessment of synthetic development
writing, not human validation or evidence of user benefit. The source histories
have already informed development; the eight reserved histories remain unused.

## Reproduction

Activate the project virtual environment, then prepare once:

```sh
source .venv/bin/activate
uv run scripts/experiments/north_star_luna.py prepare
uv run scripts/experiments/north_star_luna.py run --allow-paid
uv run scripts/experiments/north_star_luna.py report
```

Preparation writes source, code, prompt, policy and manifest hashes. Execution
fails if these change. Completed validated provider receipts are reused when
resuming. `report` reconstructs the result without network transport. Request
hashes, actual model identifiers, reasoning settings, token usage, costs and
latency remain in the fresh run directory. The latest result is
`logs/experiments/reports/north_star_luna_20260905/report.json`.
