# North Star Moment saved-Persona integration

The completed run covers all 36 closed weeks across Meera, Wei Jun, Marc,
Noor, and Lukas. It uses the shared application source filter, full-history
Luna-low assessment, and deterministic single-card selection. Independent
Luna-xhigh source and exact-quotation assessments are separate development
evaluation; their labels do not select or suppress application cards.

All 36 runtime records are complete and have been exported to the five saved
scenario bundles. Thirty weeks contain selected quotations: 23 encouragement,
four earlier reminders, and three reflection cards. Four assessed weeks omit
a card; two weeks have Insufficient Evidence and are not eligible. Independent
xhigh evaluation accepts 26 of the 30 selected quotations and leaves four
unresolved because the source and exact-quotation assessments disagree. No
selection receives a definitive rejection under the frozen grading rule.

| Evaluation context | Weeks | Selected | Accepted | Unresolved | Correct omissions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Active Drift / reflection | 4 | 3 | 3 | 0 | 1 of 1 |
| Non-drift | 32 | 27 | 23 | 4 | 2 of 2 |

The non-drift group includes two ineligible weeks. Its selected cards comprise
23 encouragement cards (19 accepted, four unresolved) and four reminders (all
accepted). Lukas week 2 omits a card although the independent source assessment
finds a supportive historical example. These results describe overlapping
synthetic histories from five Personas and AI evaluation, not independent
human observations or human validation.

## Disagreements and omissions

Meera weeks 1 and 2 expose an Achievement attribution boundary: whether helping
a son achieve exam results establishes the writer's own Achievement, and
whether mentioning parent-teacher conferences establishes an observable action
by the writer. Lukas week 6 exposes disagreement about whether committing to
put his best foot forward supports Self-Direction. Lukas week 7 exposes the
Conformity boundary between reported social discomfort and demonstrated
restraint. In each case the independent source and selected-quotation judgments
disagree, so the frozen evaluation retains an unresolved result. The application
cards remain exactly as selected by Luna-low; reference labels never filter
them. [diagnostics.json](diagnostics.json) records the corresponding cases and
judgments. No prompt or selection retuning followed these results.

Wei Jun week 1, Marc week 1, and Noor week 3 correctly omit a card against the
reference assessment. Lukas week 2 is the additional missed supportive history;
Noor week 1 and Lukas week 9 are ineligible because of Insufficient Evidence.
Legacy nudge responses have no verified availability timestamp and are excluded,
so this run does not establish live response-quotation semantic quality.

The initial network-enabled run was blocked by automatic approval review before
any provider work. The user subsequently explicitly approved the saved-Persona
runs after disclosure of the synthetic-data transfer and paid bound. The same
verified protocol then proceeded. No controlled onboarding calls are included.

## Frozen scope

- Manifest SHA256 (canonical content):
  `4d637f17f4c04f98b6587648d67dccf91ff623207a7b706f9752801b07917199`.
- 52 complete runtime requests, 52 independent source-assessment requests, and
  26 unique selected-quotation requests for 30 selected cards; repeated exact
  requests reuse validated cached results. The frozen ceiling allowed at most
  34 selected-quotation requests.
- Conservative additional ceiling: US$11.78699060 including one retry for
  every possible request. Cumulative ceiling: US$12.41677475, including
  US$0.62978415 already incurred before this integration run.
- Existing US$20 cumulative, US$0.25 per-attempt, and 16,000-token complete-input
  limits remain enforced. No truncation or silent subset selection is allowed.
- Source writing is verified verbatim against the five generated Persona
  files. Original Journal Entry availability comes from each saved submission
  event. Legacy nudge responses lack independently recorded availability and
  are excluded. The reserved eight Persona histories are not inspected or
  evaluated; their 16 original source-file hashes remain unchanged.
- Code, source, prompt, model, policy, source-order, and request hashes are in
  [manifest.json](manifest.json). Contract code is recorded as preparation
  provenance; frozen generation consumes the saved runtime request schema.

## Commands

Activate the repository virtual environment before running these commands.

```sh
uv run scripts/experiments/north_star_integration.py prepare
uv run scripts/experiments/north_star_integration.py run --allow-paid
uv run scripts/experiments/north_star_integration.py report
uv run scripts/experiments/north_star_integration.py export
```

`prepare` refuses to replace an existing manifest. `run --allow-paid` is the
only command that can call providers. `report` reconstructs evaluations using
saved validated records and receipts. `export` attaches one source-disclosed
record to each saved week and regenerates bundle/catalog hashes without
provider calls. Existing Coach Digest text, generation provenance, and event
identifiers are retained.

## Cost and receipt audit

There were 134 generation attempts: 130 completed, three invalid responses, and
one timed-out attempt. All requests finished with validated results within the
two-attempt limit. Metered new cost is US$0.77218975; the unmetered timeout retains
US$0.04328610 of reserved cost. New spend or reservation is US$0.81547585 and
cumulative spend or reservation is US$1.44526000, including the prior
US$0.62978415. The reservation is retained because the timeout's actual charge
is unknown.

All 130 unique requests have matching complete-input count receipts, ranging
from 1,390 to 3,378 tokens, below the 16,000-token cap. Observed generation usage
is one input token below the count-only receipt for all 54 Luna-low attempts;
all 79 metered Luna-xhigh attempts match exactly. Both receipt types are retained
without alteration. Counting completed sequentially before concurrent workers,
and the audit confirms no missing receipt. See [audit.json](audit.json).

## Verification

Twenty-one targeted saved-scenario/integration tests passed after export, including all-week
source availability, exclusion of legacy responses, offline attachment and
replay, stale-Profile rejection, complete budget freezing, malformed cached
assessment handling, interrupted cached-response recovery with metered cost
preservation and the one-retry limit, and oversized-input rejection before generation. Scoped
Ruff and MyPy checks passed for the changed source files. Every original
non-NSM trace event, Coach Digest event/text, and Journal Entry is unchanged;
the 26 previous experiment artifacts and 16 reserved source hashes remain
unchanged. All five final scenario bundles and 36 weeks passed loader validation.

Offline reconstruction with provider and token-count methods forbidden produced
byte-identical report, budget, input-count, and runtime-record files. A live
ledger seeded in a temporary directory inherited all 134 attempt costs and the
prior spend while redacting raw responses; the finalized experiment files
remained unchanged. No further paid calls were needed for these checks.

The [implementation validation](implementation-validation.md) records the broader
checks and known baseline failures. The [browser verification](browser-qc/README.md)
records Experience and Inspect walkthroughs, including responsive layouts and
removal during an in-flight request. Reserved-history evaluation and controlled
paid onboarding runs remain outside this saved-Persona integration run.
