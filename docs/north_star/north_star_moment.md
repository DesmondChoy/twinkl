# North Star Moment

[Workflow infographic](assets/north-star-moment-workflow.png): decision paths
after Coach Digest, source checks, and the supportive-action definition.

**Status:** Development evaluation and the first POC integration are complete
under `twinkl-fz34.1`–`.6`. Both frontend paths have source-checked cards and
Inspect, with saved-Persona evaluation and browser QC reported separately. On
5 September 2026, the user authorized removal of previous NSM experiment
results and a fresh run under the [Luna evaluation protocol](luna_evaluation_20260905.md).
The approved experiment supplies all eligible earlier Journal Entries within
a 16,000-token input limit, with `gpt-5.6-luna` at `low` reasoning for selection
and `xhigh` reasoning for evaluation. Source judgments are deduplicated, and
selected quotations receive a separate assessment against full source context
under the same explicit semantic rubric. The run and independent AI review of the evaluator are complete. Incorporation
of results into maintained reports remains paused at the user's request.
The [raw experiment result](../../logs/experiments/reports/north_star_luna_20260905/report.json)
and [saved-Persona supplement](../../logs/experiments/reports/north_star_saved_checks_20260906/README.md)
retain the evidence separately.

The [frozen Persona assignment](../../config/evals/north_star_cohort.json),
source eligibility rules, and criteria used to score the completed experiment
remain unchanged. The original strict gate was not met. The user nevertheless
accepted the development evidence for continued work on the time-boxed POC,
with shortcomings retained, as recorded in `twinkl-fz34.1`. This decision
supersedes the development stop rule below for this run; it does not relabel
the gate as passed or set a new numerical threshold for final evaluation.
The current experiment uses no embedding ranker. The first application
integration implements cards and Inspect in both frontend paths. Its saved
Persona evaluation and browser checks are recorded separately in the
[integration report](../../logs/experiments/reports/north_star_integration_20260906/README.md).
The reserved final evaluation remains outstanding. The detailed protocol is
the execution authority for the original run; integrated application
requirements are specified below.

**Scope decision, 6 September 2026:** The user adopted full-history Luna `low`
as the first integration baseline and extended North Star Moment to weeks
without Active Drift. Both onboarding and all five saved Personas use the same
rules. For Active Drift, an earlier supportive action provides perspective on
the current tension. For No Active Drift, a verified supportive action from the
reviewed week receives specific encouragement; an older example receives
historical reminder wording. No Active Drift alone never establishes alignment.
Insufficient Evidence produces no card. The frozen development experiment and
its 24/25 result remain evidence for the original Drift-triggered task only;
the encouragement path is assessed separately in the integration report,
which retains four unresolved selections and one missed supportive example.

**First version:** Both frontend paths: the demo with all five saved Personas
and onboarding from scratch with the user's own writing. Each reviewed week
can show at most one optional card.

**Documentation:** Original brief: `twinkl-b8w3`; this revision:
`twinkl-thgx`; scope expansion to both frontend paths: `twinkl-peu9`.

## 1. What this adds

Twinkl compares Journal Entries with the user's confirmed Core Values.
Weekly Drift Detection identifies repeated Conflict, and the Coach Digest
explains the finding and asks one reflective question. North Star Moment adds
an example of behaviour supporting a confirmed Core Value, quoted from the
user's own writing.

After a closed-week Weekly Drift Detection result, North Star Moment reviews
eligible Journal Entries and eligible user nudge responses for a supportive
action. Active Drift uses writing from before onset; No Active Drift permits
writing through the reviewed week. The current development experiment supplies all eligible
original Journal Entries within the input budget; legacy nudge responses lack
independent availability evidence and are excluded. A separate North Star
Moment AI review checks whether the writing describes a supportive action.
Code checks confirm that the quotation is exact, belongs to the same user or
Persona, and satisfies the source window for the selected treatment. If no example passes, no card
appears.

The demo prepares results offline for all five saved Personas: Meera, Wei Jun,
Marc, Noor, and Lukas. The onboarding path generates results during the user's
session after an eligible closed-week review. Both paths use the same
selection and validation rules. Experience displays the quotation beneath the
Coach Digest. Inspect shows why it was selected. This adds a personal example
to the existing reflection without generating advice or a second question.

A Persona or user need not receive a card. Weeks without an eligible trigger
or a suitable earlier action correctly show none. The onboarding path uses
only that user's confirmed Profile, Journal Entries, and nudge responses;
saved Persona writing must never become evidence for a new user.

## 2. Which writing qualifies

North Star Moment runs after Weekly Drift Detection completes for a closed
week. Active Drift requires a known start and supporting Journal Entries.
No Active Drift permits a separate supportive-action search; Insufficient
Evidence omits the card without a provider call. The confirmed Profile and Core Values must be the same ones used for
that Weekly Drift Detection result.

If several Core Values have Active Drift, select the one with the longest
current Conflict run. Break a tie using the confirmed Profile order. If that
Core Value has no suitable earlier quotation, show no card; do not move to
another Core Value.

For No Active Drift, review confirmed Core Values in Profile order. Prefer an
accepted action from the reviewed week, choosing the first Core Value with
such an example and its newest eligible source. If none has a current-week
example, use the first Core Value with an accepted older example, newest
first, and frame it as a historical reminder. Every selected action receives
its own NSM assessment; a Not Conflict decision is never a support label.

An earlier example must:

- belong to the same user or Persona;
- be available at the selected replay point or live review cutoff; for Active
  Drift it must also precede the first Conflict; for No Active Drift its date
  must be no later than the reviewed week end;
- describe the user's action or choice supporting that Core Value;
- contain no Conflict against that same Core Value in the writing reviewed;
- pass AI review and every required code check.

An earlier Journal Entry from the same day can qualify when its stored order
precedes the first Conflict and its date agrees with that order. A Journal
Entry written after Drift began stays ineligible for that Drift, even when
replay or the live session later reaches it.

Both the original Journal Entry and the user's nudge response can supply
evidence. The quotation must come from one identified source. The AI-written
nudge cannot supply evidence or be quoted as the user's words. A response
recorded later must not be treated as available when the original Journal
Entry was written.

Mentioning a Core Value, expressing an intention, or describing someone else's
action is insufficient. If the writing includes both support and Conflict
against the requested Core Value, reject it for version one. If its meaning
depends on missing context, the AI review should abstain.

Generation instructions, biography-only claims, LLM-Judge VIF Labels,
LLM-Judge Conflict Labels, and VIF Critic Predictions are excluded from
source selection and North Star Moment review.
Not Conflict also does not establish supportive behaviour.

## 3. What appears in Experience and Inspect

Twinkl's application-written text in Experience must not tell the user they
are "in a drift", "drifting", "drifting away", or "back on track", or use
related drifting language to describe them. This applies to status badges,
Coach Digest copy, North Star Moment copy, notices, and accessibility labels.
Describe concrete actions and experiences in plain language instead. Drift,
Active Drift, and the other detection terms remain internal terminology for
the implementation, research documentation, and developer-facing Inspect.
This wording rule does not change detection decisions or permit rewriting
the user's quoted words.

Experience shows at most one card per reviewed week, in either path. The
Coach-facing introduction is deterministic and based on the accepted record:
reflection for Active Drift, specific acknowledgment for a current-week
action, or historical reminder for older writing. It does not add a second
question or infer overall alignment, improvement, or recovery. The card contains:

- **A moment in your own words** for current-week encouragement, or
  **A past moment in your own words** for reflection and historical reminders;
- the user-facing Core Value phrase and Journal Entry date;
- one exact quotation, identified as coming from the Journal Entry or the
  user's nudge response;
- an expandable quotation with no fixed word limit;
- an action to open the complete Journal Entry without losing the current week;
- for reflection and historical reminders, the notice: **This earlier writing
  is a reference point for your Core Value.**

Collapsing a long quotation changes its presentation only. Expanding it must
reveal the complete accepted quotation without paraphrasing or joining
separate passages. The card must remain usable on a narrow screen and with a
keyboard or screen reader.

Inspect links the reviewed state and selected treatment to eligible writing, source order,
AI review, code checks, and selected quotation. A **saved review
record** contains these inputs and decisions, their versions, and model details.
It lets teammates inspect why a card appeared or why it was omitted.

Pending, missing, invalid, or failed records produce no card and leave Weekly
Drift Detection and any valid Coach Digest available. Inspect distinguishes
not evaluated, not eligible, pending, completed with or without a quotation,
and failed results, with reasons. Older results without a record remain
**not evaluated**. Onboarding sessions support live generation and bounded
retry; they are not excluded because they are manual sessions. Changes to
source data invalidate affected results in both paths.

North Star Moment leaves the Profile, Core Values, Weekly Drift Reviewer
Decisions, Drift Detector result, Historical Drift Records, and existing Coach Digest
response unchanged. The NSM introduction supplies the additional context-sensitive
Coach framing beside that response. Advice, action plans, habits, external quotations, Profile
evolution, model training, production multi-user storage, and background
scheduling remain outside this version.

## 4. Two examples

### An earlier supportive action

In the saved Wei Jun Persona, Active Drift concerns **Making the world a
fairer, better place**. The Coach Digest asks:

> When you notice yourself saying “okay” or nodding despite knowing what matters,
> what feels at stake in speaking or acting differently?

The card shows this earlier Journal Entry:

> **A past moment in your own words**
>
> Making the world a fairer, better place
>
> 22 June 2025 · From your Journal Entry
>
> “Helped two new guys file their claims.”
>
> This earlier writing is a reference point for your Core Value.

The quotation is present at `t_index=7`, before Drift starts at `t_index=8`.

### A related phrase that should be rejected

An earlier Journal Entry says:

> “The promotion process here is never fair.”

The phrase relates to fairness but does not describe the user's supportive
action. North Star Moment review
rejects it, Experience shows no card, and Inspect records the reason. The
existing Coach Digest remains available.

## 5. Academic purpose

The research question is:

> Can AI review of eligible earlier writing select a Journal Entry
> or user nudge response that describes behaviour supporting the same Core
> Value involved in Active Drift, and quote it faithfully?

Self-affirmation theory motivates the idea of placing a difficult observation
within a broader account of the person. Steele's foundational account and
Cohen and Sherman's review provide the theoretical background. North Star
Moment uses earlier writing to place the current Conflict alongside a past
supportive action.

| Capstone contribution | Evidence to produce |
|---|---|
| Intelligent Reasoning Systems | Source-level support assessments and exact-quotation selection, measured separately. |
| Intelligent Sensing Systems | Chronological Journal Entries and user responses restricted to what was available at each replay point or live review cutoff. |
| Architecting AI Systems | Shared contracts, saved review records, Experience, Inspect, and failure tests. |
| Technical Paper and implementation demonstration | Method, results, limitations, and a walkthrough linking the displayed quotation to its source. |

Evaluation measures selection and quotation accuracy using AI-reviewed
synthetic histories. Browser QC exercises both frontend paths, including
controlled writing entered through a fresh onboarding session. This is
integration evidence, not a human user study. Runtime checks cover the
capstone app's launch and inference configuration; production deployment
approval and user-benefit studies remain outside scope. The linked Capstone
Requirements describe the assessment criteria.

## 6. Evaluation decisions

The approved development experiment reviews all eligible earlier writing
within a 16,000-token input limit. Freeze the assembled inputs and prompt,
model and reasoning settings, source ordering, evaluation criteria, and
budget before paid execution. The evaluator uses the same semantic rubric
as the selector and separately assesses complete sources and exact selected
quotations. A further independent AI assessment checks the evaluator after
the run; it does not replace the frozen results or constitute human validation.

For the task-specific benchmark, the adopted criteria are:

- no incorrect displayed North Star Moments;
- correct omission in every history confirmed to contain no valid example;
- zero quotation, chronology, and wrong-user failures;
- no more than 5% unexpected provider failures;
- at least one accepted saved-Persona demonstration.

Integration acceptance also requires checking every saved Persona and a fresh
onboarding session through closed-week review, including an accepted card and
correct omissions using controlled test writing. These checks do not require
every Persona to receive a card and do not replace the task-specific benchmark.

Report coverage, the proportion of eligible reviewed weeks receiving a card,
separately for Active Drift and No Active Drift and by selected treatment,
without imposing a minimum percentage. Report counts alongside rates.
Report deliberately injected failures separately from unexpected provider
failures.

**Decision 11 adopted on 5 September 2026: reserve a small, separate benchmark.**
Development histories are examples used to adjust prompts or source selection.
Final evaluation histories are examples examined after those choices are
finished. Adjusting a prompt after seeing its test answers can make the
reported result look better than performance on unseen writing.

The considered options were:

1. **Recommended:** Reserve a small, separate benchmark covering the required
   cases. Keep its histories out of North Star Moment prompt development.
2. Reserve a larger separate benchmark for more detailed results, with more
   review work and cost.
3. Use development histories only and report feasibility evidence. Defer final
   evaluation.

The user approved option 1 before history assignment. Eight entire non-demo
Persona histories were reserved using seed 20260905 and SHA-256 identifier
ordering; the other 27 Personas supply 33 development Drift episodes. The
[frozen cohort](../../config/evals/north_star_cohort.json)
records identifiers, source hash, selection method, and approval. Saved Persona
demos and onboarding QC are separate from final evaluation.

The approved paid envelope is US$20 total, US$0.25 per attempt, and at most one
retry per request, including offline/live review, reference decisions, browser
validation calls, and evaluation. SDK retries are disabled. The current
[evaluation protocol](luna_evaluation_20260905.md) records the model and reasoning
settings, input limit, evaluation procedure, and execution requirements. Paid
requests must use frozen token, timeout, rate, retry, and budget settings.

## 7. Work plan and completion

The authorized development experiment and independent evaluator review are
complete. Expanded application integration and validation are in progress. Account separately for
live generation, retries, invalidation, and browser QC on both application paths.

| Phase | Work and output |
|---|---|
| 0: development evaluation | Freeze all eligible development writing and the 16,000-token budget; run Luna at `low` for selection and `xhigh` for source and exact-quotation evaluation; review the evaluator independently. |
| 1: source review and validation | Under the accepted POC continuation decision, implement shared filtering, bounded review, saved records, and code checks for offline preparation and live sessions. Include retry, reuse, and invalidation. |
| 2: Experience and Inspect | Add the card, source links, all five saved Persona results, onboarding integration, migration, and replay/accessibility tests. Launch the frontend and backend for browser QC on both paths. |
| 3: final evaluation and reporting | Evaluate the frozen implementation on the reserved histories. Report errors and limitations, document browser QC separately, and update the Technical Paper and walkthrough for both paths. |

The user has authorized the fresh experiment and its model settings. Enforce
the approved per-attempt and total limits before every paid request, including
evaluation and retries. An over-budget history must be recorded explicitly;
do not silently truncate its writing or select a smaller subset.

The original development gate remains failed under the explicit POC
continuation decision above. A failed NSM request omits the card and retains
the valid Coach Digest. The final benchmark criteria must be settled before
opening reserved histories; no new numerical threshold is inferred here.

Completion requires a focused implementation issue, an adopted PRD scope,
passing contract and regression checks, reproducible reports, all five saved
Persona replays, and a fresh onboarding session demonstrating the live path.
Across the two paths, demonstrate acceptance, omission, provider failure and
retry, invalidation, and exclusion of future writing. Final evaluation
requirements depend on decision 11.
Prepare reports with their exact source and configuration records.

## Technical appendix

### A. Source selection and AI review

#### Inputs and eligibility

The request records the schema version, user or Persona identifier, confirmed
Profile reference, reviewed week, replay or live review cutoff, selected Core
Value and user-facing phrase, Active Drift start, supporting Journal Entry
identifiers, eligible source text, prompt version and hash, model settings,
and creation time. Each source distinguishes `journal_entry` from `nudge_response`.

Filter before any provider call. Require a matching identity,
non-empty user-written text, and independent original-source availability
through the cutoff. For Active Drift, require
`t_index < active_drift_start_t_index` and `date <= active_drift_start_date`.
Both ordering checks must pass. For No Active Drift, include original writing
through the reviewed week end and prefer current-week examples. Exclude
removed Journal Entries, current Drift evidence, and anything unavailable at
the replay point or live review cutoff. Dates and stored order must agree.

A nudge response needs its own evidence of availability at the replay point
or live review cutoff, and before the first Conflict when Active Drift applies.
The live application records a separate server-timestamped response event;
legacy responses without such evidence remain excluded. Use recorded event
order or timestamps; do not copy the parent Journal Entry's date as proof. If
availability cannot be established, exclude that response while retaining an
otherwise eligible original Journal Entry. Preserve source boundaries when composing text for
selection and review. AI-written nudges and hidden generation or labelling
information are excluded.

#### Bounded full-history input

Supply the user-facing Core Value phrase and its approved definition from
`config/schwartz_values.yaml`. Do not import Persona-generation examples,
instructions, biography, labels, or current Conflict text into the semantic
assessment. The caller selects the affected Core Value before this review.

The current experiment sends all eligible original Journal Entries in the
frozen source order without embedding ranking. The 16,000-token input ceiling
covers instructions, the Core Value definition, source identifiers and text,
and response-schema/request formatting. Output has a separate allowance.
Count assembled requests before execution, preserve the measurements, and
stop or record an explicit over-budget outcome if the complete input does not
fit. Do not silently truncate writing. The current protocol defines ordering
and selection; freeze both before reading new model results.

#### North Star Moment review

Use a prompt and typed schema separate from the Weekly Drift Reviewer.
The selector uses `gpt-5.6-luna` at reasoning effort `low`; the evaluator uses
`gpt-5.6-luna` at reasoning effort `xhigh`. Record requested and actual model
identifiers and reasoning settings. Both roles apply the same explicit rubric
in the [current protocol](luna_evaluation_20260905.md).

For every requested source and Core Value, assess whether the writer reports
an action, whether that action supports the approved Core Value definition,
and whether the complete eligible context contains the writer's behavior
against that same Core Value. Negative emotion, external hardship, uncertainty
about outcomes, and another person's behavior do not alone establish the
writer's Conflict. Joint actions can include the writer's own participation;
intentions and outcomes alone do not establish a completed supportive action.

Accept only an exact, continuous quotation that itself conveys the writer's
supportive action when read with its supplied context. Do not require the
quotation to state the Core Value label. Reject unsupported interpretations;
abstain when the available writing cannot resolve a necessary fact. Preserve
concise evidence-based explanations that can be audited, rather than asking
for hidden model reasoning. A model must not change source identifiers, the
requested Core Value, or the application-selected priority.

The evaluator judges each distinct source/Core Value pair once, without
seeing the selector's decision. A separate exact-quotation assessment then
checks every selected quotation against its full eligible source. Source
acceptance alone cannot approve a different quotation. Preserve both judgments
and their provenance. A further independent AI assessment reviews the evaluator
and records agreements and disagreements separately from the frozen scores.

Require complete, schema-valid decisions for all requested sources. Reject
missing, duplicate, extra, or malformed decisions, refusals, timeouts, and
provider errors. Validate exact source matching, chronology, and identity in
code before selecting a quotation under the frozen rule. A missing valid
selection produces no card.

### B. Code checks, saved records, and integration

#### Required checks

Validate the following before rendering:

| Check | Required behaviour |
|---|---|
| Identity and membership | Every returned Journal Entry was requested for the same user or Persona. |
| Core Value | The response identifies the requested Core Value. |
| Chronology | Original text and responses satisfy their independent availability rules and mode-specific source window; Active Drift additionally requires pre-onset writing. |
| Exact quotation | The quotation is a continuous exact substring of the identified user-written source. Never combine sources or repair a quotation by paraphrasing. |
| Complete response | All requested decisions are present once, with permitted fields and decision/reason/source combinations. |
| User-facing terminology | Application-written Experience text, including badges, notices, and accessibility labels, must not expose internal Drift states or describe the user as drifting, drifting away, or back on track. Use concrete descriptions of actions and experiences. Preserve exact user quotations. |
| User-facing claims | Application-written text must not infer recovery, improvement, typical behaviour, success, or an ended Active Drift from the quotation. Review the quotation in context; these checks must not rewrite the user's words. |
| Internal value labels | No raw internal Schwartz label appears in card fields. The current POC rejects the whole response batch when any supportive quotation contains one, permits one retry within the attempt budget, and shows no card if validation still fails. Never rewrite the quotation. |
| Display | Expansion preserves the full accepted quotation, its source, and the route back to the current week. There is no fixed quotation word limit. |
| Failure | A missing, stale, invalid, refused, or failed saved result produces no card. |

The internal-label check is a conservative POC limitation: it matches ordinary
words such as “security,” “power,” and “tradition” even when the writer uses
them naturally. A match invalidates the batch, including other potentially
suitable quotations; the retry can therefore consume both attempts without
producing a card. This restriction preserves the existing fail-closed policy.
Candidate-level exclusion or distinguishing ordinary language from internal
labels would require a separate policy change and validation.

These are North Star Moment checks. Existing Coach Digest Validations do not
automatically cover the card. Code can check identity, ordering, and text
matching; semantic suitability still depends on AI review. Inspect records
both results.

#### Saved review record

A versioned record, called a receipt in existing code, should preserve:

- session or Persona, week, cutoff, Profile reference, Core Value, and Drift start;
- eligible and supplied Journal Entry identifiers in order;
- source text references, availability evidence, content hashes, selected
  Journal Entry, quotation source, and exact quotation;
- every AI decision and code-check result, including why no card appeared;
- schema and prompt versions, prompt hash, creation time, and input hash;
- source ordering and complete-input token measurements;
- requested and actual provider/model settings, usage, latency, calculated cost,
  and status.

Experience must not present model assessments as confidence scores or call
the selected quotation the user's best or strongest example.

Inspect links the trigger, filtering, source order, prompt and model, response,
checks, and selection. Do not expose hidden provider reasoning, secrets, or
generation metadata. Store source-disclosed failure records for both offline
fixture generation and live execution. Freeze retry limits before paid work;
repeating an identical completed request reuses its record instead of
duplicating calls. Changed inputs require a new record.

#### Experience and Inspect integration

Use one selection and validation implementation for both paths. The demo
precomputes records for every reviewed week in all five saved Personas,
including explicit Insufficient Evidence and no-card outcomes. Normal replay reads
these records without provider calls and shows only records available at the
selected cutoff. Optional demo live rerun is not a prerequisite for NSM; if
used, its changed review output invalidates dependent saved NSM results and
any regeneration follows the live rules below.

In the onboarding path, the frontend requests `review_north_star` separately
after the existing closed-week review action returns. The backend snapshots
that week's recorded Weekly Drift Detection result and source window, awaits
NSM outside the session lock, then verifies the input hash before publication. Saving writing in an open
week does not trigger NSM. Keep the result and any valid Coach Digest usable
while NSM runs or fails. No eligible writing for the selected treatment means no card and no
unnecessary review call. Use the current session's confirmed Profile and
source availability; do not borrow dates from a parent entry or treat
backdated writing as available at an earlier cutoff.

The live path must be executable in the documented capstone frontend/backend
launch configuration. Full-history review does not require an embedding
service. Keep provider credentials server-side and verify token, cost,
concurrency, timeout, and latency limits in the intended configuration before
claiming live readiness. Define explicit behavior for histories exceeding the
input limit before application integration. Update affected launch and hosting
documentation when that integration is implemented.

Add optional records compatibly to existing session and scenario contracts.
Older weeks remain usable with a **not evaluated** status and no card. Bind
results to the session, governing Profile, source content and availability,
review output, and cutoff. Removing a Journal Entry, changing its response,
changing the governing Profile, or recomputing affected Weekly Drift Detection
output invalidates dependent results before display. Discard an in-flight
result if its inputs changed or its session was deleted. Regenerate affected
live results only after the replacement closed-week result is ready and
eligible; an ineligible result clears the card. Rebuild affected demo records
offline rather than showing a stale card.

Reuse completed requests for identical inputs, including valid no-card
outcomes, and coalesce duplicate in-flight requests. Allow bounded retry of
retryable failures within the agreed budget without rerunning successful
Weekly Drift Detection or Coach Digest work. Record attempts and reasons in
Inspect. Store live NSM records within the existing POC session lifecycle,
including supported resume and Delete session behavior. Do not promise durable
multi-user storage or add background scheduling.

Likely code locations are `src/demo/contracts.py`,
`src/demo/experience_service.py`, `src/demo/scenarios.py`, a focused North
Star Moment module, and the Experience/Inspect components under
`frontend/onboarding/src/`. Reuse provider and validation patterns in
`src/coach/`; inspect current callers and contracts before editing.
Pydantic remains the shared schema source. Update generated schemas, React
validation, saved Persona hashes, and manifests together.

### C. Evaluation protocol

#### Development and final evaluation

Keep development and final work separately named in scripts and reports.
Development cases may inform prompt changes. Evaluate the reserved histories
only after the prompt, ordering, selection rule, code, and criteria are frozen.
The existing frozen assignment keeps related cutoffs and Core Values from the
same Persona together. Preserve that assignment when resetting experiment
outputs; do not repartition after seeing development results.

Decision 11 is resolved as the small separate benchmark. Reserved Persona
histories remain excluded from prompt development and independent development
review. Final evaluation stays outstanding until the frozen implementation is
evaluated.

#### Benchmark cases and reference decisions

Each case includes the Persona, confirmed Core Values, week, Active Drift
start and selected Core Value, every eligible earlier Journal Entry and user
response, case categories, AI reference decisions, and a manifest with hashes.
Freeze case count, sampling seed, selection method, source files, and exclusions
before requesting reference decisions. Preserve the complete eligible source
manifest and any sampling decisions. Review only the task-specific benchmark,
not all 1,651 Journal Entries.

Include the following cases:

| Category | Expected outcome |
|---|---|
| Clear support for the affected Core Value | Accept an exact quotation from the original Journal Entry or eligible user response. |
| A related phrase, intention, or emotion without action | Reject or abstain. |
| Someone else's action or support only for another value | Reject for the requested Core Value. |
| Conflict, including mixed support and Conflict for that Core Value | Reject. |
| Ambiguous or context-dependent writing | Abstain. |
| Several valid earlier examples | Apply the frozen source and selection order. |
| No valid earlier example | Show no card. |
| Same-day writing with earlier stored order | Permit only when dates, order, and source availability agree. |
| Writing or a response after Drift start or the replay/live review cutoff | Exclude before source selection, provider input, and current-point Inspect records. |
| Multiple Active Drifts with no example for the priority Core Value | Show no card without trying another Core Value. |
| No Active Drift | Review eligible writing for an actual supportive action; distinguish current-week encouragement from an older reminder. |
| Insufficient Evidence | Do not request North Star Moment. |
| Refusal, invalid/incomplete JSON, timeout, stale or missing record | Show no card and retain existing Weekly Drift Detection and valid Coach Digest. |
| Fresh onboarding with no eligible earlier writing | Show no card without borrowing Persona data or making an unnecessary review call. |
| Source edit, removal, replacement review, or session deletion during a live request | Invalidate dependent records and discard obsolete in-flight output. |
| Repeated live request or retry | Reuse completed results, coalesce in-flight work, and apply agreed retry and cost limits. |

Sample selected examples, reviewed-but-rejected examples, histories returning
no card, and deliberately selected difficult cases. Include short or ambiguous
writing and harder Core Values. Any use of implementation results for sampling
must follow the agreed development/final separation and be recorded.

To establish that a history has **no valid example**, reference review must
examine every eligible earlier Journal Entry and user response. An incomplete
review cannot establish this. A No Active Drift Persona now tests supportive
action selection or appropriate omission, separately from the original
Drift-triggered experiment.

Reference review uses the exact North Star Moment definition, source
boundaries, and same-Core-Value Conflict exclusion. The selector and evaluator
share the rubric, with separate requests and recorded model settings. The
source evaluator does not see selection decisions. Deduplicate repeated
source/Core Value inputs, request auditable explanations, and evaluate the
exact selected quotations separately. The subsequent independent AI review
records disagreement without silently overwriting the frozen evaluator output.
An unresolved reference cannot count as an accepted displayed example or prove
that a history has no valid example. Preserve prompts, model configurations,
review sources, timestamps, usage, costs, hashes, and adjudications.

Exercise all five saved Personas; choose highlighted demonstration weeks
after inspecting eligible writing and review results. Also exercise a fresh
onboarding session with controlled test writing. Keep walkthrough and browser
QC cases separate from a reserved final benchmark. Record whether failure
examples came from injected tests or provider errors.

#### Source corpus and prior labels

The source histories and known development Drift records remain shared input
data. The [frozen assignment](../../config/evals/north_star_cohort.json) defines
which Personas may be used in this experiment. Existing LLM-Judge VIF Labels
and consensus labels are not NSM reference decisions and must not be supplied
to the selector or evaluator. The fresh full-history protocol does not use
label-proxy retrieval scores.

#### Metrics and acceptance criteria

| Metric | Definition and adopted criterion |
|---|---|
| North Star Moment precision | Displayed quotations accepted by the task-specific reference, divided by all displayed quotations. Require zero incorrect selections in the reported benchmark. Reference rejection or abstention counts as incorrect. Zero displayed quotations gives undefined precision, not 100%. |
| Correct no-card rate | Reference-confirmed histories with no valid example that receive no card, divided by all reference-confirmed histories with no valid example. Require 100%. Include such histories; an empty denominator is not a pass. |
| Coverage | Cases receiving a card divided by eligible reviewed weeks under the adopted priority rule, reported separately by state and treatment. Report counts and exclusions, including histories without earlier writing. No minimum percentage; require at least one accepted saved-Persona demonstration. |
| Wrong-Core-Value rate | Displayed examples supporting another value but not the affected Core Value, divided by displayed examples. These also count as incorrect selections. |
| Abstention rate | AI review abstentions divided by reviewed Journal Entries. |
| Quotation, chronology, and wrong-user failures | Report separate counts; require zero for displayed results. |
| Unexpected provider failure rate | Unexpected refused, invalid, timed-out, or error calls divided by actual non-injected calls. Include failed attempts before successful retries; require at most 5%. Report runtime review and reference-review calls separately. |
| Cost and latency | Calculated cost, usage, and processing time per offline and live attempt and for the full benchmark, including reference decisions and retries. Report live user wait time separately from offline preparation time. |

Declare denominators and exclusions before the respective runs. Report errors
by Core Value and case category, counts alongside percentages, and uncertainty
appropriate to the sample size. Preserve failed responses and diagnostics
where permitted. Deliberately injected failure tests are reported separately.

If evaluating the card beside the Coach Digest, report additional criteria
for the relationship to the reviewed state, specificity, tone, treatment of tension,
prohibited current-state claims, and whether the single reflective question
remains appropriate. Name these separately from existing Coach Digest Evals
unless that contract is explicitly extended.

### D. Verification and reporting checklist

- Verify trigger rules, priority selection, same-user and same-Core-Value checks,
  same-day ordering, response availability, removal, and future-data exclusion
  before provider calls.
- Test frozen source order, batch completeness, supportive/rejected/abstaining
  decisions, mixed Conflict rejection, quotation source attribution, exact text,
  and malformed, refused, missing, stale, and failed results.
- Verify saved-request reuse, controlled offline retries, session resume, older
  sessions, source-change invalidation, and affected-week recomputation.
- Verify live request reuse, in-flight deduplication, bounded retries, source
  edits during generation, Delete session, and budget enforcement. Confirm
  identical eligible inputs use the same selection rules in both paths.
- Check narrow and wide layouts, expandable long quotations, keyboard focus,
  screen-reader labels, quotation semantics, reduced motion, Journal Entry
  navigation, Inspect links, and preserved week selection.
- Check that application-written Experience copy and accessibility labels
  contain no internal Drift states or related drifting language. Keep the
  checks against unsupported recovery or improvement claims.
- Launch the React frontend and Python backend with the documented inference
  configuration. Perform browser QC at narrow and wide widths across all five
  demo Personas and a fresh onboarding session through a closed-week review.
  Exercise accepted cards, omissions, failures/retries, source changes, future
  writing exclusion, quotation expansion, source navigation, and Inspect.
  Save screenshots and distinguish mocked failure checks from live-provider
  evidence. Repair QC findings and recheck affected flows.
- Update and check Python/React contracts, generated schemas, fixtures, saved
  Persona hashes, manifests, and no-future-data replay.
- Run relevant Python and React tests, Ruff, and MyPy when typed interfaces
  change; run existing Weekly Drift Detection and Coach Digest regression
  tests. Confirm no mutation of the Profile or existing Drift decisions.
- Save commands, inputs, seeds, configuration, model revisions, prompt hashes,
  results, errors, exclusions, calculated costs, latency, and limitations with
  the reproduction script and report.
- Update the PRD when scope is adopted and implementation status when earned.
  After evaluation, update architecture/evaluation documentation, the Technical
  Paper's method and results, and a walkthrough with accepted and omitted cards
  in both paths. Update affected sources and generated outputs under
  `docs/capstone_report/`, and keep implementation, AI evaluation, and browser
  QC evidence distinct.
- Report the benchmark data, review sources, and whether histories were
  separate from development.
- Read the revised logic and affected callers, inspect the final diff and
  working-tree state, and record validation and remaining risks in the
  implementation issue.

## References

- [Product Requirements Document](../prd.md)
- [Canonical Nouns and Communication Rules](../canonical_nouns.md)
- [Capstone Requirements](../capstone_report/capstone_requirements.pdf)
- [Technical Paper source](../capstone_report/capstone_project_report.md)
- [Experience and Inspect design](../demo/experience_inspect_app.md)
- [Coach Digest explanation quality](../evals/explanation_quality_eval.md)
- [VIF Critic (Offline) concepts and roadmap](../vif/01_concepts_and_roadmap.md)
- [Value evolution concept note](../evolution/01_value_evolution.md)
- [Habit recommendation future work](../future_work/habit_recommendations.md)
- Steele, C. M. (1988). [The psychology of self-affirmation: Sustaining the integrity of the self](https://doi.org/10.1016/S0065-2601(08)60229-4). *Advances in Experimental Social Psychology, 21*, 261–302.
- Cohen, G. L., & Sherman, D. K. (2014). [The psychology of change: Self-affirmation and social psychological intervention](https://doi.org/10.1146/annurev-psych-010213-115137). *Annual Review of Psychology, 65*, 333–371.
