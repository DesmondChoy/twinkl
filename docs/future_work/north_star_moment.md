# North Star Moment

**Status:** Design specification.

**First version:** Both frontend paths: the demo with all five saved Personas
and onboarding from scratch with the user's own writing. Each reviewed week
can show at most one optional card.

**Documentation:** Original brief: `twinkl-b8w3`; this revision:
`twinkl-thgx`; scope expansion to both frontend paths: `twinkl-peu9`.

## 1. What this adds

Twinkl compares Journal Entries with the user's confirmed Core Values.
Weekly Drift Detection identifies repeated Conflict, and the Coach Digest
explains the finding and asks one reflective question. North Star Moment adds
an earlier example of behaviour supporting the affected Core Value, quoted
from the user's own writing.

When Weekly Drift Detection reports Active Drift, Twinkl searches earlier
Journal Entries and the user's nudge responses. It uses **semantic retrieval**,
which searches by meaning, to find possible examples. A separate North Star
Moment AI review checks whether the writing describes a supportive action.
Code checks confirm that the quotation is exact, belongs to the same user or
Persona, and comes from before the Drift began. If no example passes, no card
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

North Star Moment runs only after Weekly Drift Detection completes for a
closed week and reports Active Drift with a known start and supporting Journal
Entries. The confirmed Profile and Core Values must be the same ones used for
that Weekly Drift Detection result.

If several Core Values have Active Drift, select the one with the longest
current Conflict run. Break a tie using the confirmed Profile order. If that
Core Value has no suitable earlier quotation, show no card; do not move to
another Core Value.

An earlier example must:

- belong to the same user or Persona;
- come before the first Conflict in the selected Active Drift and be available
  at the selected replay point or live review cutoff;
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
retrieval and North Star Moment review.
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

Experience shows at most one card per reviewed week, in either path, containing:

- **A past moment in your own words**;
- the user-facing Core Value phrase and Journal Entry date;
- one exact quotation, identified as coming from the Journal Entry or the
  user's nudge response;
- an expandable quotation with no fixed word limit;
- an action to open the complete Journal Entry without losing the current week;
- the notice: **This earlier writing is a reference point for your Core Value.**

Collapsing a long quotation changes its presentation only. Expanding it must
reveal the complete accepted quotation without paraphrasing or joining
separate passages. The card must remain usable on a narrow screen and with a
keyboard or screen reader.

Inspect links the selected Active Drift to the eligible writing, retrieval
results, AI review, code checks, and selected quotation. A **saved review
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
Decisions, Drift Detector result, Historical Drift Records, and Coach Digest
response unchanged. Advice, action plans, habits, external quotations, Profile
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

Semantic retrieval might find this phrase because it relates to fairness.
It does not describe the user's supportive action. North Star Moment review
rejects it, Experience shows no card, and Inspect records the reason. The
existing Coach Digest remains available.

## 5. Academic purpose

The research question is:

> Can semantic retrieval followed by AI review select an earlier Journal Entry
> or user nudge response that describes behaviour supporting the same Core
> Value involved in Active Drift, and quote it faithfully?

Self-affirmation theory motivates the idea of placing a difficult observation
within a broader account of the person. Steele's foundational account and
Cohen and Sherman's review provide the theoretical background. North Star
Moment uses earlier writing to place the current Conflict alongside a past
supportive action.

| Capstone contribution | Evidence to produce |
|---|---|
| Intelligent Reasoning Systems and Pattern Recognition Systems | Retrieval results and AI review decisions, measured separately. |
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

First test retrieval locally, using existing LLM-Judge VIF Labels as a rough
reference. Later evaluation uses North Star Moment-specific reference
decisions. Compare retrieval of the top 1, 3, and 5 Journal Entries,
then choose the smallest number reaching **at least 90% proxy retrieval
recall**: finding an earlier positively labelled Journal Entry in at least
90% of histories that contain one.

For the later task-specific benchmark, the adopted criteria are:

- no incorrect displayed North Star Moments;
- correct omission in every history confirmed to contain no valid example;
- zero quotation, chronology, and wrong-user failures;
- no more than 5% unexpected provider failures;
- at least one accepted saved-Persona demonstration.

Integration acceptance also requires checking every saved Persona and a fresh
onboarding session through closed-week review, including an accepted card and
correct omissions using controlled test writing. These checks do not require
every Persona to receive a card and do not replace the task-specific benchmark.

Report coverage, the proportion of eligible Active Drift cases receiving a
card, without imposing a minimum percentage. Report counts alongside rates.
Report deliberately injected failures separately from unexpected provider
failures.

**Decision 11 remains open: which histories should be reserved for evaluation?**
Development histories are examples used to adjust the prompt or retrieval.
Final evaluation histories are examples examined after those choices are
finished. Adjusting a prompt after seeing its test answers can make the
reported result look better than performance on unseen writing.

The options are:

1. **Recommended:** Reserve a small, separate benchmark covering the required
   cases. Keep its histories out of North Star Moment prompt development.
2. Reserve a larger separate benchmark for more detailed results, with more
   review work and cost.
3. Use development histories only and report feasibility evidence. Defer final
   evaluation.

This decision must be recorded before assigning histories or requesting
task-specific reference decisions.

## 7. Work plan and completion

The earlier **7–11 working day** estimate covered saved Persona replay only.
Re-estimate the expanded scope after inspecting the live session contracts and
choosing how to host the fixed retrieval encoder. Account for live generation,
retry, invalidation, and browser QC on both paths.

| Phase | Work and output |
|---|---|
| 0A: local retrieval check | Reproduce the existing baseline; freeze the query and encoder settings; compare retrieval at 1, 3, and 5. Stop if no setting reaches 90%. No paid calls. Inspect live hosting options and revise the effort estimate. |
| 0B: development AI review | After the evaluation choice and paid budget are agreed, build development cases and reference decisions. Test the new prompt against the adopted criteria. |
| 1: retrieval and review | Implement shared filtering, retrieval, AI review, records, and code checks, with offline preparation and live session execution. Include bounded retry, request reuse, and invalidation. |
| 2: Experience and Inspect | Add the card, source links, all five saved Persona results, onboarding integration, migration, and replay/accessibility tests. Launch the frontend and backend for browser QC on both paths. |
| 3: evaluation and reporting | If a separate benchmark is adopted, evaluate the frozen implementation once. Report errors and limitations, document browser QC separately, and update the Technical Paper and walkthrough for both paths. |

Start with Phase 0A. Before paid work, present a cost estimate and obtain agreed
per-attempt and total limits covering offline and live review, reference
decisions, retries, browser validation calls, and evaluation. Continue
independent work while those decisions are pending; do not begin dependent
paid work without agreement.

If a phase fails its criteria, stop and keep the existing Coach Digest
behaviour.

Completion requires a focused implementation issue, an adopted PRD scope,
passing contract and regression checks, reproducible reports, all five saved
Persona replays, and a fresh onboarding session demonstrating the live path.
Across the two paths, demonstrate acceptance, omission, provider failure and
retry, invalidation, and exclusion of future writing. Final evaluation
requirements depend on decision 11.
Prepare reports with their exact source and configuration records.

## Technical appendix

### A. Retrieval and AI review

#### Inputs and eligibility

The request records the schema version, user or Persona identifier, confirmed
Profile reference, reviewed week, replay or live review cutoff, selected Core
Value and user-facing phrase, Active Drift start, supporting Journal Entry
identifiers, eligible source text, prompt version and hash, model settings,
and creation time. Each source distinguishes `journal_entry` from `nudge_response`.

Filter before any embedding or provider call. Require a matching identity,
non-empty user-written text, `t_index < active_drift_start_t_index`, and
`date <= active_drift_start_date`. Both ordering checks must pass. Exclude
removed Journal Entries, current Drift evidence, and anything unavailable at
the replay point or live review cutoff. Dates and stored order must agree.

A nudge response needs its own evidence of availability before the first
Conflict and at the replay point or live review cutoff. Use recorded event
order or timestamps; do not copy the parent Journal Entry's date as proof. If
availability cannot be established, exclude that response while retaining an
otherwise eligible original Journal Entry. Preserve source boundaries when composing text for
retrieval and review. AI-written nudges and hidden generation or labelling
information are excluded.

#### Semantic retrieval

Use the user-facing Core Value phrase plus its approved definition. Freeze
the exact definition and its source before Phase 0A. Use only the value
definition; do not import Persona-generation examples, instructions, biography,
labels, or current Conflict text into the query.

The fixed encoder is `nomic-ai/nomic-embed-text-v1.5`, using:

- a recorded model revision and 256-dimensional Matryoshka representation;
- `search_query: ` for the query and `search_document: ` for eligible writing;
- the encoder's normalization sequence: layer normalization, truncation to
  256 dimensions, then L2 normalization;
- cosine similarity for ranking.

The existing VIF Critic (Offline) encoder uses `classification: ` and must
not be reused unchanged. Three-dimensional PCA or t-SNE coordinates from the
Embedding Explorer are unsuitable for retrieval.

Rank one document per eligible Journal Entry, including its eligible user
response with its source identified. Compare top-k values of 1, 3, and 5;
select the smallest meeting the 90% proxy threshold. If none passes, stop.
Freeze k before paid review and final benchmark work. Preserve retrieval
order, use recency for equal similarities, and use a stable identifier for
any remaining tie.

#### North Star Moment review

Use a new prompt and schema, separate from the Weekly Drift Reviewer.
Start with the current Coach Digest generation settings:
`gpt-5.6-luna`, reasoning effort `none`.
Record requested and actual model identifiers. Use a different AI provider
for benchmark reference decisions; select and freeze its exact configuration
before paid work. Revise the evaluation plan if that provider is unavailable.

Send the retrieved Journal Entries in one batch. Require exactly one decision
for every requested Journal Entry and the requested Core Value. For example:

~~~json
{
  "schema_version": "north-star-moment-review-v1",
  "core_value": "universalism",
  "results": [
    {
      "entry_id": "8f83c818:entry:7",
      "decision": "supportive",
      "quote_source": "journal_entry",
      "evidence_quote": "Helped two new guys file their claims.",
      "reason_code": "observable_choice"
    }
  ]
}
~~~

Permitted decisions are `supportive`, `not_supportive`, and `abstain`. An accepted
decision requires one non-empty exact quotation from the identified source:
`journal_entry` or `nudge_response`. Rejected and abstaining decisions
require an empty quotation and null source. Permit reason codes for observable
choices, wrong values, intentions, hypotheticals, another person's action,
same-Core-Value Conflict, ambiguity, and insufficient text.

Review all eligible user-written text attached to each Journal Entry.
Reject an example containing Conflict against the requested Core Value even
when another passage supports it. A model must not change identifiers, the
requested Core Value, or the application-selected priority.

Reject the whole batch for missing, duplicate, extra, or malformed decisions,
refusal, timeout, or provider error. A complete valid batch may contain a
mixture of supportive, not-supportive, and abstaining decisions. Apply code
checks to the batch before selecting the first accepted Journal Entry in the
frozen retrieval order.

### B. Code checks, saved records, and integration

#### Required checks

Validate the following before rendering:

| Check | Required behaviour |
|---|---|
| Identity and membership | Every returned Journal Entry was requested for the same user or Persona. |
| Core Value | The response identifies the requested Core Value. |
| Chronology | The original text and any included response satisfy their availability rules and precede Drift start. |
| Exact quotation | The quotation is a continuous exact substring of the identified user-written source. Never combine sources or repair a quotation by paraphrasing. |
| Complete response | All requested decisions are present once, with permitted fields and decision/reason/source combinations. |
| User-facing terminology | Application-written Experience text, including badges, notices, and accessibility labels, must not expose internal Drift states or describe the user as drifting, drifting away, or back on track. Use concrete descriptions of actions and experiences. Preserve exact user quotations. |
| User-facing claims | Application-written text must not infer recovery, improvement, typical behaviour, success, or an ended Active Drift from the quotation. Review the quotation in context; these checks must not rewrite the user's words. |
| Internal value labels | No raw internal Schwartz label appears in card fields. Omit an unsuitable quotation rather than rewriting it. |
| Display | Expansion preserves the full accepted quotation, its source, and the route back to the current week. There is no fixed quotation word limit. |
| Failure | A missing, stale, invalid, refused, or failed saved result produces no card. |

These are North Star Moment checks. Existing Coach Digest Validations do not
automatically cover the card. Code can check identity, ordering, and text
matching; semantic suitability still depends on AI review. Inspect records
both results.

#### Saved review record

A versioned record, called a receipt in existing code, should preserve:

- session or Persona, week, cutoff, Profile reference, Core Value, and Drift start;
- eligible and retrieved Journal Entry identifiers in order;
- source text references, availability evidence, content hashes, selected
  Journal Entry, quotation source, and exact quotation;
- every AI decision and code-check result, including why no card appeared;
- schema and prompt versions, prompt hash, creation time, and input hash;
- encoder name, revision, prefixes, dimensions, and normalization;
- requested and actual provider/model settings, usage, latency, calculated cost,
  and status.

Similarity values may be saved as retrieval diagnostics. Experience must not
display them as relevance scores or confidence. Do not call the selected
quotation the user's best or strongest example.

Inspect links the trigger, filtering, retrieval, prompt and model, response,
checks, and selection. Do not expose hidden provider reasoning, secrets, or
generation metadata. Store source-disclosed failure records for both offline
fixture generation and live execution. Freeze retry limits before paid work;
repeating an identical completed request reuses its record instead of
duplicating calls. Changed inputs require a new record.

#### Experience and Inspect integration

Use one selection and validation implementation for both paths. The demo
precomputes records for every reviewed week in all five saved Personas,
including explicit non-triggering and no-card outcomes. Normal replay reads
these records without provider calls and shows only records available at the
selected cutoff. Optional demo live rerun is not a prerequisite for NSM; if
used, its changed review output invalidates dependent saved NSM results and
any regeneration follows the live rules below.

In the onboarding path, the existing closed-week review action triggers NSM
after an eligible Weekly Drift Detection result. Saving writing in an open
week does not trigger NSM. Keep the result and any valid Coach Digest usable
while NSM runs or fails. No earlier eligible writing means no card and no
unnecessary review call. Use the current session's confirmed Profile and
source availability; do not borrow dates from a parent entry or treat
backdated writing as available at an earlier cutoff.

The live path must be executable in the documented capstone frontend/backend
launch configuration. `requirements-experience.txt` currently omits PyTorch
and Sentence Transformers, so the former offline-only packaging is insufficient.
Compare hosting the fixed encoder in the backend with a separate inference
service, including memory, startup time, latency, cost, and operational work.
Record the chosen approach and any required infrastructure decision before
implementing live retrieval. Prefer the smallest viable option; do not
silently substitute an embedding model or build a speculative service layer.
Keep inference and provider credentials server-side. Update launch and
hosting documentation and verify the chosen configuration.

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
Phase 0B uses development cases to adjust the prompt. Under either separate
benchmark option, Phase 3 uses different histories after the prompt, query, k,
selection rule, code, and criteria are frozen. Exclude histories used for
Phase 0A retrieval tuning as well as Phase 0B prompt development. If reserving
final histories from the 42-Drift corpus, separate them before selecting k and
report the development denominator separately from the full baseline. Group
related replay cutoffs and Core Values from the same Persona to avoid reuse
of the same writing across both groups.

Decision 11 must be resolved before assigning these histories. If the
development-only option is adopted, report the development results and list
final evaluation as outstanding work.

#### Benchmark cases and reference decisions

Each case includes the Persona, confirmed Core Values, week, Active Drift
start and selected Core Value, every eligible earlier Journal Entry and user
response, case categories, AI reference decisions, and a manifest with hashes.
Freeze case count, sampling seed, selection method, source files, and exclusions
before requesting reference decisions. Keep retrieval output used for
sampling. Review only the task-specific benchmark, not all 1,651 Journal
Entries.

Include the following cases:

| Category | Expected outcome |
|---|---|
| Clear support for the affected Core Value | Accept an exact quotation from the original Journal Entry or eligible user response. |
| A related phrase, intention, or emotion without action | Reject or abstain. |
| Someone else's action or support only for another value | Reject for the requested Core Value. |
| Conflict, including mixed support and Conflict for that Core Value | Reject. |
| Ambiguous or context-dependent writing | Abstain. |
| Several valid earlier examples | Apply the frozen retrieval and selection order. |
| No valid earlier example | Show no card. |
| Same-day writing with earlier stored order | Permit only when dates, order, and source availability agree. |
| Writing or a response after Drift start or the replay/live review cutoff | Exclude before retrieval, provider input, and current-point Inspect records. |
| Multiple Active Drifts with no example for the priority Core Value | Show no card without trying another Core Value. |
| No Active Drift or Insufficient Evidence | Do not request North Star Moment. |
| Refusal, invalid/incomplete JSON, timeout, stale or missing record | Show no card and retain existing Weekly Drift Detection and valid Coach Digest. |
| Fresh onboarding with no eligible earlier writing | Show no card without borrowing Persona data or making an unnecessary review call. |
| Source edit, removal, replacement review, or session deletion during a live request | Invalidate dependent records and discard obsolete in-flight output. |
| Repeated live request or retry | Reuse completed results, coalesce in-flight work, and apply agreed retry and cost limits. |

Sample displayed examples, retrieved-but-rejected examples, histories returning
no card, and deliberately selected difficult cases. Include short or ambiguous
writing and harder Core Values. Any use of implementation results for sampling
must follow the agreed development/final separation and be recorded.

To establish that a history has **no valid example**, reference review must
examine every eligible earlier Journal Entry and user response. Reviewing only
the top-k results cannot establish this. A No Active Drift Persona tests
non-triggering, not omission after an unsuccessful search.

Reference review uses the exact North Star Moment definition, source
boundaries, and mixed-Conflict exclusion. Request a decision and an exact
quotation for accepted writing. Use a second AI review only for disagreements
and predefined high-risk cases. Record how disagreements are handled; an
unresolved reference cannot count as an accepted displayed example. Preserve
prompts, model configurations, review sources, timestamps, usage, costs, hashes,
and adjudications.

Exercise all five saved Personas; choose highlighted demonstration weeks
after inspecting eligible writing and review results. Also exercise a fresh
onboarding session with controlled test writing. Keep walkthrough and browser
QC cases separate from a reserved final benchmark. Record whether failure
examples came from injected tests or provider errors.

#### Existing label baseline

The existing label baseline comes from
`complete_development_drift_episodes.parquet`,
`logs/judge_labels/judge_labels.parquet`, and
`logs/judge_labels/consensus_labels.parquet`. Reproduce it in Phase 0A.
These figures describe LLM-Judge VIF Labels. Phase 0A also records which user
responses meet the source-availability rules.

| Existing development result | Count |
|---|---:|
| Known development Drifts | 42 |
| Any Journal Entry before Drift start | 34 of 42 (81.0%) |
| Earlier same-Core-Value `+1` persisted LLM-Judge VIF Label | 26 of 42 (61.9%) |
| Earlier same-Core-Value `+1` consensus LLM-Judge VIF Label | 26 of 42 (61.9%) |
| Active Drift at the final history cutoff | 10 of 42 |
| Final-cutoff Active Drift with an earlier `+1` in each label file | 9 of 10 (90.0%) |

Eight Drifts start at `t_index=0` and have no earlier Journal Entry. Five
others start at `t_index=1` and each has one earlier Journal Entry; keep
these groups separate.

Use persisted labels as the primary Phase 0A proxy. Five-pass consensus
measures agreement and helps group diagnostic results. Across 16,510 labels
for 1,651 Journal Entries and ten values, the files differ at 1,368 coordinates
(8.29%). Among 145 unique earlier same-Core-Value coordinates in the 42 Drift
histories, 17 differ (11.72%). The persisted file has 68 positive coordinates;
consensus has 64. Those 64 comprise 49 unanimous decisions, three four-of-five
decisions, and 12 three-of-five decisions.

Wei Jun illustrates the task mismatch. Before Drift starts at `t_index=8`,
persisted labels mark Universalism `+1` at 1, 6, and 7. Consensus adds 0
and 2, both by three-of-five majorities. At 0, the persisted LLM-Judge VIF Label is
`-1`, consensus is `+1`, and the LLM-Judge Conflict Label is Conflict:
he notices harm but stays silent.

Report proxy retrieval recall separately for five-of-five, four-of-five,
three-of-five, and persisted-versus-consensus disagreement groups. The 26-of-42
availability result describes the persisted-label proxy. Task-specific review
can reject positive-label examples or accept writing that lacked a positive
label.

#### Metrics and acceptance criteria

| Metric | Definition and adopted criterion |
|---|---|
| Proxy retrieval recall at k | Histories where top-k includes at least one earlier persisted `+1`, divided by histories containing such a label. Require at least 90%; compare k = 1, 3, 5 and report agreement groups separately. |
| North Star Moment precision | Displayed quotations accepted by the task-specific reference, divided by all displayed quotations. Require zero incorrect selections in the reported benchmark. Reference rejection or abstention counts as incorrect. Zero displayed quotations gives undefined precision, not 100%. |
| Correct no-card rate | Reference-confirmed histories with no valid example that receive no card, divided by all reference-confirmed histories with no valid example. Require 100%. Include such histories; an empty denominator is not a pass. |
| Coverage | Cases receiving a card divided by eligible Active Drift cases under the adopted priority rule. Report counts and exclusions, including histories without earlier writing. No minimum percentage; require at least one accepted saved-Persona demonstration. |
| Task-specific retrieval recall at k | Histories with at least one reference-valid example in top-k, divided by histories containing any reference-valid example. Report separately from proxy recall. |
| Verification lift | Difference between precision after AI review and retrieval-only precision on the same cases. Freeze the retrieval-only display rule before the run. |
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
for the relationship to Active Drift, specificity, tone, treatment of tension,
prohibited current-state claims, and whether the single reflective question
remains appropriate. Name these separately from existing Coach Digest Evals
unless that contract is explicitly extended.

### D. Verification and reporting checklist

- Verify trigger rules, priority selection, same-user and same-Core-Value checks,
  same-day ordering, response availability, removal, and future-data exclusion
  before embeddings and provider calls.
- Test frozen retrieval order, batch completeness, supportive/rejected/abstaining
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
- [Habit recommendation future work](habit_recommendations.md)
- Steele, C. M. (1988). [The psychology of self-affirmation: Sustaining the integrity of the self](https://doi.org/10.1016/S0065-2601(08)60229-4). *Advances in Experimental Social Psychology, 21*, 261–302.
- Cohen, G. L., & Sherman, D. K. (2014). [The psychology of change: Self-affirmation and social psychological intervention](https://doi.org/10.1146/annurev-psych-010213-115137). *Annual Review of Psychology, 65*, 333–371.
