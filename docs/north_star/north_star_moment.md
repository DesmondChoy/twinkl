# North Star Moment

[Workflow infographic](assets/north-star-moment-workflow.png): selection paths,
source checks, and the supportive-action definition. The integrated reading
order is specified below.

**Status:** Application code supports North Star Moment in onboarding and all
five saved Persona replays, for Active Drift and No Active Drift. The saved
replay uses the completed full-history outcomes from the targeted Weekly Drift
v4 Run 1 update, preserving no-card outcomes and original receipt provenance.
Live review uses a separately authorized US$1 allowance shared across sessions
and restarts. Both Experience paths place a selected quotation within a valid
Coach Digest after its reflective question. The saved demo can switch between
complete responses generated with and without the selected source context;
onboarding retains its original narrative and question.

The [8 September integration check](../../logs/experiments/reports/integrated_coach_validation_20260908/report.md)
records five unresolved AI editorial concerns across the previous roster's
combined reflections, including a Coach/quotation contradiction caused by differing
source context. These observations do not change the frozen experiment results
or constitute human validation.

The [experiment methodology](nsm_experiment_methodology.md) records the reset
and the original 501-week AI comparison using Weekly Drift v2 repeat 1.
Full history achieved higher Card precision and Opportunity recall than Nomic
top-three retrieval in both partitions; human review remains deferred. It replaces the previous experimental
protocols, cohort decisions, and acceptance gates. This specification describes
the existing product behavior and reusable implementation contracts; it does
not establish results or authorize a new experiment.

A separate [targeted v4 Run 1 update](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md)
is complete for both methods. It reassessed 38 affected weeks across 22
Personas and retained observations for 463 unchanged cases, including failures
and unresolved judgments. The original 105 Personas, 501 weeks, 81/24 partition,
and 20-case evaluator consistency sample remain fixed. Full eligible history
still achieved higher Card precision and Opportunity recall in both partitions.
These are updated results with retained observations from the original v2
repeat-1 study, not a wholly fresh independent experiment. Both records remain
available; the update does not establish human validity or user benefit.

**Current scope:** Both frontend paths: the demo with all five saved Personas
and onboarding from scratch with the user's own writing. Each reviewed week
can include at most one optional passage within a valid Coach Digest. Active
Drift uses pre-onset supportive writing. No Active Drift prefers current-week
encouragement, then an older
historical reminder. No Active Drift alone never establishes alignment.
Insufficient Evidence produces no card.

**Saved demo comparison:** The [five-Persona comparison](demo_coach_comparison.md)
specifies the toggle, exact Coach Digest prompts, injected field, and paired
response provenance. The default is **Without North Star Moment**. Switching to
**With North Star Moment** replaces both narrative paragraphs and the reflective
question for the same Persona and week, and displays the existing quotation
panel. Inspect shows both exact prompts beside their associated responses.
This comparison applies only to saved demo Personas; it preserves onboarding
behavior and the frozen North Star Moment selection experiment.

## 1. Role in the weekly reflection

Twinkl compares Journal Entries with the user's confirmed Core Values.
Weekly Drift Detection identifies repeated Conflict, and the Coach Digest
explains the finding and asks one reflective question. North Star Moment supplies
an example of behaviour supporting a confirmed Core Value, quoted from the
user's own writing.

After a closed-week Weekly Drift Detection result, North Star Moment reviews
eligible Journal Entries and eligible user nudge responses for a supportive
action. Active Drift uses writing from before onset; No Active Drift permits
writing through the reviewed week. A separate North Star Moment AI review
checks whether the writing describes a supportive action.
Code checks confirm that the quotation is exact, belongs to the same user or
Persona, and satisfies the source window for the selected treatment. If no example passes, no card
appears.

The demo uses offline records for Nisha, Noor, Lukas, Wei Jun, and Meera,
covering 27 reviewed weeks from the completed targeted experiment. The
onboarding path supports generation during the user's session after an
eligible closed-week review when its runtime budget is available. Both paths
use the same selection and validation rules. Experience places the quotation
after the displayed Coach Digest narrative and reflective question,
within one reading area. Inspect shows why it was selected. In the saved demo,
the with-context response receives the exact quotation and complete selected
source in its prompt; its narrative and question can therefore refer to that
action. Onboarding keeps its existing response and inserts the quotation without
a model rewrite. Neither path adds advice or a second question.

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

Experience adds at most one North Star Moment passage to a valid Coach Digest
per reviewed week. The exact selected quotation is preserved after the complete
displayed response, including its reflective question. Saved replay shows this passage only
with its validated with-context response; the without-context response hides it.
Onboarding preserves its original generated narrative and question. The introduction
is deterministic and based on the accepted record:
reflection for Active Drift, specific acknowledgment for a current-week
action, or historical reminder for older writing. It does not add a second
question or infer overall alignment, improvement, or recovery. The passage contains:

- **A moment in your own words** for current-week encouragement, or
  **A past moment in your own words** for reflection and historical reminders;
- the user-facing Core Value phrase and Journal Entry date;
- one exact quotation, identified as coming from the Journal Entry or the
  user's nudge response;
- an expandable quotation with no fixed word limit;
- an action to open the complete Journal Entry in a centered dialog without
  losing the current week.

The introductory wording points directly to the quotation. Current-week
encouragement says, **Here’s a moment worth holding onto, in your own words:**
Reflection says, **You’ve acted on what matters to you before. You wrote:**
Historical reminders say, **Your earlier words offer a reminder of how you’ve
made room for what matters to you:** The heading and source date identify
earlier writing without a second explanatory notice.

Collapsing a long quotation changes its presentation only. Expanding it must
reveal the complete accepted quotation without paraphrasing or joining
separate passages. The card must remain usable on a narrow screen and with a
keyboard or screen reader.

Onboarding uses these plain headings without the feature name. Persona replay
adds discreet **North Star Moment** attribution and **Inspect this moment**,
linked to the exact selected backend event. Inspect labels the introduction as
deterministic application wording and the quotation as exact source text.

Inspect links the reviewed state and selected treatment to eligible writing, source order,
AI review, code checks, and selected quotation. A **saved review
record** contains these inputs and decisions, their versions, and model details.
It lets teammates inspect why a card appeared or why it was omitted.
Inspect presents the complete supplied writing, exact system and user prompts
and response schema, each source's factual action/value/Conflict assessments,
and the application's source-order selection and validation. The selected
source's assessment opens by default. Original provider attempts remain visible;
missing receipts are explicitly unavailable. These runtime AI assessments do
not stand in for the separate benchmark evaluation or human validation.

For an available saved comparison, the Coach Digest Inspect event contains
**Without North Star Moment** and **With North Star Moment** panels. Each panel
shows its generated narrative, question, exact accepted prompt, raw response,
and model details. The initial prompt remains available when repair feedback
changed the accepted request. Inspect also shows the changed `north_star_context`
field and the shared weekly input. Switching the Experience toggle does not hide
either Inspect panel and never makes a provider call.

Pending, missing, invalid, or failed records omit the passage and leave Weekly
Drift Detection and any valid Coach Digest available. An absent or invalid Coach
Digest also suppresses augmentation; Experience does not show an empty moment
card. Inspect distinguishes
not evaluated, not eligible, pending, completed with or without a quotation,
and failed results, with reasons. Older results without a record remain
**not evaluated**. Onboarding sessions support live generation and bounded
retry; they are not excluded because they are manual sessions. Changes to
source data invalidate affected results in both paths.

North Star Moment leaves the Profile, Core Values, Weekly Drift Reviewer
Decisions, Drift Detector result, and Historical Drift Records unchanged. The
saved comparison varies only the Coach Digest's selected source context; the
two complete responses retain separate provider receipts. Onboarding keeps its
existing response and uses the deterministic introduction beside the quotation.
Advice, action plans, habits, external quotations, Profile
evolution, model training, production multi-user storage, and background
scheduling remain outside this version.

## 4. Two examples

### An earlier supportive action

In Nisha's saved week starting 3 March 2025, Active Drift concerns
**Making the world a fairer, better place**. The completed full-history runtime
selects this earlier supportive action:

> **A past moment in your own words**
>
> Making the world a fairer, better place
>
> 7 March 2025 · From your Journal Entry
>
> You’ve acted on what matters to you before. You wrote:
>
> “Stayed back to help the peon uncle fix a broken desk in my classroom.”

The quotation is present at `t_index=4`, before Drift starts at `t_index=5`
on 8 March. It is an exact runtime selection from the targeted experiment,
not a new generation or a human-validated example.

### A related phrase that should be rejected

An earlier Journal Entry says:

> “The promotion process here is never fair.”

The phrase relates to fairness but does not describe the user's supportive
action. The review rule requires rejection, no card in Experience, and a
reason in Inspect. The existing Coach Digest remains available.

## 5. Academic purpose

The research question and evaluation design are documented in the
[fresh experiment methodology](nsm_experiment_methodology.md).

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

AI review and browser checks serve different purposes: semantic assessment
and application verification. Neither constitutes a human user study.
Production deployment approval and user-benefit studies remain outside scope.
The linked Capstone Requirements describe the assessment criteria.

## 6. Evaluation status

Results from before the September 2026 reset remain retired. The
[experiment methodology](nsm_experiment_methodology.md) documents the original
Weekly Drift v2 repeat-1 comparison and the subsequent targeted v4 Run 1
update. Both compare Nomic top-three retrieval plus Luna-low review with full
eligible history plus Luna-low review, using shared Luna-xhigh AI evaluation
across 105 Personas: 81 in development and 24 in qualified final evaluation.
The [targeted report](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md)
records new and retained evidence, paired metric exclusions, Persona-bootstrap
intervals, failures, and incremental costs separately from historical costs
and latency. The original record, frozen methodology, and correction history
remain preserved. This is AI assessment of synthetic histories; it does not
establish human validity.

## 7. Work status

Housekeeping is tracked in `twinkl-fz34.13`, the original experiment in
`twinkl-fz34.7`, and the targeted update in `twinkl-fz34.14`. The saved replay
integration is tracked in `twinkl-fz34.15`. Its five Personas demonstrate the
current v4 Run 1 Drift patterns using the completed full-history outcomes.
The final capstone walkthrough and human review remain separate work.

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

The current runtime requires each nudge response to have availability evidence
at the replay point or live review cutoff, and before the first Conflict when
Active Drift applies. The live application records a separate
server-timestamped response event; the runtime excludes responses without an
availability value while retaining an otherwise eligible original Journal
Entry. These describe existing code, not the fresh experiment's synthetic-data
policy. Preserve source boundaries when composing text for selection and
review. AI-written nudges and hidden generation or labelling information are
excluded.

#### Bounded full-history input

Supply the user-facing Core Value phrase and its approved definition from
`config/schwartz_values.yaml`. Do not import Persona-generation examples,
instructions, biography, labels, or current Conflict text into the semantic
assessment. The caller selects the affected Core Value before this review.

The current application sends all eligible writing in newest-first source
order without embedding ranking. Its configured 16,000-token input ceiling
covers instructions, the Core Value definition, source identifiers and text,
and response-schema/request formatting. Output has a separate allowance.
The runtime counts assembled requests and rejects over-budget input instead
of silently truncating writing. These are retained implementation settings,
not a choice of variants or limits for the fresh experiment.

#### North Star Moment review

The current runtime uses a prompt and typed schema separate from the Weekly
Drift Reviewer, with `gpt-5.6-luna` at reasoning effort `low`. Requested and
actual model identifiers and reasoning settings are recorded. The reusable
semantic assessment is implemented in
[`src/north_star/assessment.py`](../../src/north_star/assessment.py).

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

Both paths use one selection and validation implementation. The demo supports
precomputed records for every reviewed week in all five saved Personas,
including explicit Insufficient Evidence and no-card outcomes. Normal replay
reads available records without provider calls and restricts them to the
selected cutoff. Current records come from the targeted v4 Run 1 experiment;
export preserves the original provider requests and raw responses rather than
presenting them as fresh application calls.
Optional demo live rerun is not a prerequisite for NSM; if used, its changed
review output invalidates dependent saved NSM results and any regeneration
follows the live rules below.

In the onboarding path, the frontend requests `review_north_star` separately
after the existing closed-week review action returns. The backend snapshots
that week's recorded Weekly Drift Detection result and source window, awaits
NSM outside the session lock, then verifies the input hash before publication.
Saving writing in an open week does not trigger NSM. Keep the result and any
valid Coach Digest usable while NSM runs or fails. No eligible writing for the
selected treatment means no card and no unnecessary review call. Use the current session's confirmed Profile and
source availability; do not borrow dates from a parent entry or treat
backdated writing as available at an earlier cutoff.

Full-history review does not require an embedding service. Provider credentials
remain server-side. Under the user's 8 September 2026 authorization, live
requests use a fresh, conservative US$1 allowance in
[`north_star_live_v1.json`](../../config/evals/north_star_live_v1.json), with zero
prior spend and the existing Luna-low request and retry limits. The fixed ledger
persists accounting across sessions and restarts; session deletion does not
reset it. This allowance does not reconstruct the removed integration budget.
Saved records keep their original policy and provenance. Revalidation accepts
only the repository-pinned integration and live policy hashes, with one policy
across every value review. Corrupt, mismatched, or exhausted budgets fail closed.

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

Implementation locations include `src/demo/contracts.py`,
`src/demo/experience_service.py`, `src/demo/scenarios.py`, `src/north_star/`,
and the Experience/Inspect components under `frontend/onboarding/src/`. Reuse
provider and validation patterns in `src/coach/`; inspect current callers and
contracts before editing.
Pydantic remains the shared schema source. Update generated schemas, React
validation, saved Persona hashes, and manifests together.

### C. Evaluation methodology

See the [fresh experiment methodology](nsm_experiment_methodology.md). The
previous dataset assignment, evaluator procedure, metric definitions, and
acceptance thresholds are no longer experimental authority. They are not
carried forward by this specification.

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
