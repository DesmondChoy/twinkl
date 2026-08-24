# Experience and Inspect React App

## Status

This document specifies the capstone assessment experience. The shared React
Experience and Inspect shell, resumable client session, view selector, and
focused Inspect navigation are implemented. Manual Journal Entry processing,
displayed nudges with reply and skip actions, safe retry, and linked nudge
events in Inspect are also implemented. Closed-week review populates Weekly
Drift Reviewer Decisions, the Drift Detector result, cited Weekly Drift
Detection output, a Coach Digest run, and linked Inspect events only after the
Monday-through-Sunday week closes. A valid Coach Digest response appears when
available. A missing or invalid response does not remove the Weekly Drift
Detection result.

The five deterministic persona replays now load into the shared React session
with manual next-step replay, previous-week navigation, optional automatic
replay and pause, restart, Jump to key moment, reduced-motion behavior,
no-future-data projection, and browser-side scenario hash verification. The
release quality gate is implemented. The five Persona key-week Coach Digest
responses now match the [current evaluation
manifest](../../logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json).
Current capstone work is
Coach Digest feedback capture, longitudinal Core Value history, and the final
professor walkthrough. The
optional live rerun does not block the final walkthrough. The versioned
React-Python boundary, JSON Schema, and
canonical fixtures are implemented in
[`src/demo/contracts.py`](../../src/demo/contracts.py) and
[`frontend/onboarding/src/contracts/`](../../frontend/onboarding/src/contracts/).
The existing React onboarding implementation and the
[Onboarding Specification](../onboarding/onboarding_spec.md) remain
authoritative for the Schwartz Values Best-Worst Survey (SVBWS), Profile, and
Core Value contracts.

## Public Assessment Deployment

- **URL:** [Twinkl Experience and
  Inspect](https://onboarding-production-1dd2.up.railway.app/)
- **Hosting:** Railway serves the React app and the same-origin Python
  boundary.
- **Access:** The assessment URL allows anonymous browser access.
- **Provider boundary:** Provider credentials remain on the server.
- **Provider cost:** Live Journal Entry work can make paid provider calls.
- **Deletion behavior:** After Profile confirmation, Delete session removes the
  matching in-memory Python session and request receipts before React clears
  browser storage. Before Profile confirmation, Start over clears browser-only
  progress.
- **Scope:** The deployment is for capstone assessment only. It is not
  deployment approval.
- **Excluded production controls:** Production authentication, multi-tenant
  persistence, and service-level commitments.

## 1. Purpose

The app presents the product experience and the AI architecture from the same
session. A persistent two-option control switches between:

- **Experience** — the user-facing journey through onboarding, Journal Entries,
  displayed nudges and responses, Drift, and the Coach Digest response.
- **Inspect** — the professor-facing explanation of the exact browser
  calculation and backend work that produced the currently selected result.

The two views are not separate flows. They read the same Profile,
Journal Entries, Weekly Drift Reviewer Decisions, Drift state, Weekly Drift
Detection output, and run trace. Switching views must preserve the current
session, selected week, selected Journal Entry, and selected backend event.

This design lets a professor assess both user value and Architecting AI Systems
work without waiting for a real week of journaling.

The React app is mobile-first. Design and verify the complete Experience and
Inspect walkthrough for narrow-screen phones first, then progressively enhance
the same views for wider screens. Desktop convenience must not determine the
information hierarchy, interaction order, or acceptance of the mobile flow.

## 2. Product and Evidence Boundaries

- The React implementation in `frontend/onboarding/` remains the single
  onboarding implementation. Do not port it to Shiny or duplicate its SVBWS
  scoring in Python.
- The user-facing Drift path is fixed: Journal Entries and Core Values go to
  the `gpt-5.6-luna` reasoning-effort-`low` Weekly Drift Reviewer without VIF
  Critic input, then the Drift Detector applies the two-consecutive-Conflict
  rule.
- Saving a Journal Entry never reviews its open calendar week. Review cadence
  is Monday through Sunday, and the first partial week becomes eligible after
  its first Sunday.
- The VIF Critic remains offline research. Inspect may link to separate
  research reports, but it must not imply that VIF Critic Predictions produce
  user-facing Drift.
- Persona replay uses saved Weekly Drift Reviewer Decisions by default. It must
  identify their model contract, input hash, run provenance, and whether the
  source was replayed or generated live.
- AI-reviewed synthetic development evidence is not human validation or
  deployment approval. The source of every displayed decision must remain
  explicit in Inspect.

## 3. View Availability

| Session stage | Experience | Inspect |
|---|---|---|
| Active SVBWS card selection | Enabled | Disabled with “Available after all 11 questions” |
| Value summary before confirmation | Enabled | Enabled; shows the complete browser calculation and highest-scoring values |
| Confirmed Profile handoff | Enabled | Enabled; shows the Profile handoff and validation |
| Journal Entry draft or nudge check | Enabled | Enabled; follows the active Journal Entry event |
| Weekly Drift Detection or Coach Digest work | Enabled | Enabled; follows the active run |
| Persona replay | Enabled | Enabled; follows the selected saved run |

The disabled state must explain why it is unavailable. It must not look like a
broken control.

## 4. Shared Session Model

One client-side session store owns:

- the confirmed Profile and Core Values;
- ordered Journal Entries, including displayed nudges and responses;
- the assessment clock for manual Experience, when active;
- the selected persona scenario, week, Journal Entry, and trace event;
- Weekly Drift Reviewer Decisions grouped by calendar week;
- the current Drift Detector result;
- the Weekly Drift Detection output, Coach Digest status, and valid Coach
  Digest response when available;
- run status and retry state; and
- references to backend trace events.

Changing views changes presentation only. It must not repeat a model call,
reset onboarding, alter replay progress, or create a second copy of the
session.

Saved replay Drift states and Coach Digest responses have context-specific
Inspect actions. The live Journal Entry path retains one latest-run Inspect action.
A weekly result action switches to Inspect and focuses the weekly explanation.
It also selects and expands the event that produced the result. Other Inspect
actions focus the selected event. Returning to Experience restores the same
screen position and selection where practical.

## 5. Experience View

### 5.1 Manual onboarding

Preserve the complete React onboarding flow:

1. 11 randomized SVBWS groups with one Most and one Least choice;
2. label-free Core Value confirmation, with an exact two-value choice when
   more than two values share the highest score; and
3. first Journal Entry handoff.

Do not add backend telemetry to the SVBWS card screens. Profile JSON, raw
scores, Schwartz labels, and developer terminology remain hidden in
Experience.

### 5.2 Manual journaling

After Profile confirmation, Experience provides:

- a Journal Entry composer with the current Simulated time date;
- clear saving, nudge-checking, reviewing, complete, and failed states;
- one contextual nudge with reply and skip actions when the nudge decision
  requests it;
- an 800 ms pause after the saved Journal Entry appears, followed by one small
  horizontal Nudge reveal; reduced-motion mode keeps the pause without the
  horizontal movement;
- the anti-annoyance rule of no more than two displayed nudges in the previous
  three Journal Entries;
- a newest-first thread containing each Journal Entry plus any displayed nudge
  and response;
- guided **Write on the next day** and **Close week and review** actions after
  the newest Journal Entry is final;
- a contextual retry action after a retryable backend failure; and
- an edit action when no accepted submission response has returned, so a
  pending Journal Entry never leaves Experience without an enabled recovery
  control.

Manual Experience starts one assessment clock from the browser IANA timezone.
Python owns all later date changes. **Write on the next day** moves the date
forward by one day. **Close week and review** moves the date to the next Monday
and runs all due finalized weeks. Simulated dates never move backward. The
action is blocked while a displayed nudge needs a response or skip.

The newest Journal Entry card appears first in manual Experience. The stored
Journal Entries remain chronological by `t_index`. The Weekly Drift Reviewer,
Drift Detector, Coach Digest, and Inspect use that chronological order. Persona
replay keeps its existing chronological presentation.

Manual Experience allows explicit Journal Entry removal after confirmation.
Removing a Journal Entry or saving a nudge reply or skip advances the session
revision. Python recomputes the affected week plus any later weeks only when
those closed weeks were already reviewed. An open week remains unreviewed. The
saved browser state remains unchanged if synchronization fails, so the action
can be retried without losing user text. Removed Journal Entry `t_index` values
are not reused. Inspect retains their immutable submission events and marks
them as removed from the current Experience.

A Journal Entry must be held safely while the nudge check runs. A missing key,
refusal, invalid response, or request failure must not discard the Journal
Entry. Failure copy distinguishes text retained in the browser editor from a
Journal Entry accepted by the Python boundary and names the Experience service
rather than attributing transport or routing failures to a product component.

### 5.3 Persona simulation

Experience offers a curated **Try a demo persona** shortcut. Selecting a
persona loads its Profile, Core Values, Journal Entries, displayed nudges and
responses, saved Weekly Drift Reviewer Decisions, Drift states, and Coach
Digest responses.

Persona simulation is a week-by-week replay rather than an immediate dump of
the final state. Controls provide:

- manual next-step movement as the default;
- previous week;
- optional automatic replay and pause;
- direct selection of any already revealed week; and
- restart scenario.

Advancing a week reveals only the Journal Entries and results available by that
week. This preserves the temporal meaning of Drift and lets the professor see
the user experience change between Active Drift, No Active Drift, and
Insufficient Evidence. Historical Drift Records remain available after the
current state changes. Future weeks remain disabled. A separate **Jump to key moment** action
provides explicit fast navigation without making future results look available.

Manual next-step movement and automatic replay use the same sequence. Journal
Entries appear one at a time as compact excerpts. The Weekly Drift Detection
result appears only after the final Journal Entry. The next week then starts
empty. Automatic replay leaves enough time to read each step. Opening a Journal
Entry uses a desktop side panel or mobile bottom sheet so the timeline does not
reflow.

When a Journal Entry has a displayed Nudge, the Nudge appears 800 ms after the
Journal Entry. An early **Next step** action reveals that pending Nudge before
the replay advances. Automatic replay starts its next reading interval only
after the Nudge appears. Reduced-motion mode keeps the delay and removes the
horizontal movement.

When a saved Coach Digest response is present, it appears after the Weekly
Drift Detection result in the same scrollable column. It does not replace the
Weekly Drift Detection result.

The desktop Experience uses a fixed weekly workspace. The Journal Entry column
and Weekly Drift Detection column stay in the same viewport. Each column scrolls
internally when necessary. The phone Experience uses a Journal Entries and
Weekly Drift view control instead of two narrow columns. The current result
stays visible above that control. The active week stays centered in the week
rail.

Profile details remain collapsed by default and include a short Persona context.
The Persona header always names the selected Schwartz Core Values. State-change
evidence appears with the Weekly Drift Detection result. The first two Conflicts
show where Drift started. Later Conflicts show that Drift continued. No Active
Drift cites current Journal Entries and their Weekly Drift Reviewer Decisions.
Insufficient Evidence cites the blocking Journal Entry and its Abstain or
failed review status when available.

Each cited Weekly Drift Reviewer Decision provides its saved model name,
reasoning effort, parsed model output, and recorded justification. Desktop
shows these details when the evidence card is hovered or receives keyboard
focus. An **AI review** action opens the same details. On a phone, that action
opens a bottom sheet. The details are available for Active Drift, No Active
Drift, and Insufficient Evidence. The replay identifies itself as
saved synthetic evidence throughout. The staged reveal must not imply live
model inference.

Saved replay is the default because it is fast, deterministic, and free of
provider availability. A separate, clearly labelled **Re-run live** action may
exist in Inspect. Live results must never silently replace the saved reference
run.

### 5.4 User-facing results

Experience shows:

- the user's own Journal Entries, displayed nudges, and responses;
- an ambient per-Core-Value Drift state; and
- the Coach Digest response and question when valid, or a Coach Digest
  unavailable state.

Experience does not show the full Weekly Drift Reviewer prompt, provider
payload, validation record, identifiers, or hashes. Those belong in Inspect.
Experience does show the Weekly Drift Reviewer Decision, saved model contract,
parsed model output, and recorded justification beside each cited Journal
Entry. It does not claim that reasoning effort is a readable chain of thought.

## 6. Inspect View

### 6.1 Information hierarchy

Inspect opens on a readable calculation and event timeline, not a telemetry
dump. The first level answers:

1. What happened?
2. What component did it?
3. What result did it produce?
4. Does the event need attention because it is queued, running, refused,
   invalid, or failed?

Technical details show the duration, model contract, identifiers, hashes,
inputs, prompts, responses, and validation on demand.

For Persona replay, Inspect shows the selected week first. Filters select
Journal Entry events, Weekly Drift Reviewer events, or Drift Detector events.
The complete earlier Inspect history stays collapsed by default. Repeated
saved-run labels, reused-result labels, and zero-duration labels do not appear
on each event. Model details, run source, reasoning effort, identifiers, hashes,
and exact inputs remain under **Technical details**.

Before the backend event timeline, Inspect presents the completed browser-side
SVBWS calculation as a professor-facing explanation rather than developer
documentation. It shows:

- two aligned evidence columns containing the 11 recorded Most selections and
  11 recorded Least selections in presentation order;
- a separate totals table that lists each of the 11 SVBWS objects once and
  shows Most and Least totals of 11 recorded choices;
- a note after the totals table that explains why the published SVBWS keeps
  Universalism–Nature and Universalism–Social separate, and that the Profile
  merges them into one Universalism score;
- the exact Most-minus-Least calculation beside each score in the ten-value
  Profile table;
- the two-facet Universalism mean and the ten-value Profile transformation;
- the ten-value Profile rows in descending weight order, with canonical order
  retained for equal weights;
- the exact Schwartz value-to-Experience phrase mapping;
- every highest-score tie before confirmation and the resulting Core Values
  after confirmation;
- completeness, balanced-exposure, distinct-choice, and weight-total checks;
  and
- the explicit boundary that the deterministic calculation makes no model,
  reliability, confidence, diagnostic, or clinical claim.

On desktop, assessment Inspect uses a 240-pixel left rail. A sticky section map
links to Choices, Counts, the Universalism merge, the Profile, and Checks. The
map highlights the section at the reading position. Narrow screens keep the
single-column Inspect layout and do not show the rail.

The same 240-pixel section rail supports Profile confirmation, the first
Journal Entry handoff, manual Journal Entry work, saved Persona selection,
saved Persona replay, and saved-run Inspect. Each map links only to sections in
the current view. Content with a maximum width stays centered between the
section rail and the outer page edge. The active values questions retain the
compass because it shows assessment progress. Narrow screens do not show a
section rail.

This calculation is labelled **Calculation method** and **Deterministic · no
model**.
It is not fabricated as a Python trace event. Profile confirmation remains the
first Python event, preserving the React ownership of onboarding scoring and
the Python ownership of confirmed Profile validation.

### 6.2 Trace event types

Inspect represents these events when applicable:

1. `profile_confirmed`
   - Profile validation, Core Values, and Profile provenance.
2. `journal_entry_submitted`
   - Journal Entry date, text reference, ordering validation, and session ID.
3. `nudge_suppression_checked`
   - previous-three-entry window and whether the anti-annoyance rule suppressed
     a nudge.
4. `nudge_decided`
   - sanitized inputs, exact prompt, model, category, reason, response,
     validation, and latency.
5. `nudge_generated`
   - exact prompt, generated question, word-count validation, attempts, and
     latency.
6. `weekly_review_requested`
   - week boundaries, cumulative displayed Journal Entry history, Core Values,
     prompt, fixed model contract, and input hash.
7. `weekly_review_completed`
   - raw provider response, validation result, effective Weekly Drift Reviewer
     Decisions, response ID when available, attempts, and latency.
8. `drift_detected`
   - the ordered Weekly Drift Reviewer Decisions considered, the deterministic
     rule steps, and resulting Drift state.
9. `weekly_digest_built`
   - structured Weekly Drift Detection output fields, cited Journal Entries, and source Drift
     state.
10. `weekly_coach_generated`
    - exact prompt, model, response, narrative validation, and latency.

The Inspect copy must use the canonical component names above. In particular,
Weekly Drift Reviewer Decisions are not called predictions.

### 6.3 Required event fields

Each trace event contains:

| Field | Purpose |
|---|---|
| `event_id` | Stable identity within the session |
| `session_id` | Joins Experience and Inspect state |
| `parent_event_id` | Connects cause and effect |
| `event_type` | One of the event types above |
| `status` | queued, running, complete, reused, refused, invalid, or failed |
| `source` | saved replay or live run |
| `started_at` / `completed_at` | Timing and ordering |
| `duration_ms` | Latency shown in Inspect |
| `input_refs` | Profile, Journal Entry, week, or prior-event references |
| `model_contract` | Model and reasoning effort when a model is called |
| `prompt` | Exact rendered provider request, with trusted instructions and untrusted input data shown as separate messages |
| `raw_response` | Provider response before product transformation |
| `validation` | Schema and content validation result |
| `result_refs` | Resulting nudge, decisions, Drift, or Weekly Drift Detection output |
| `input_hash` | Idempotency and replay identity |
| `error` | Safe error class and message without secrets |

Prompt reuse or a cache hit may be recorded when implemented, but caching is
not required by this UI contract. Persistent, inspectable provenance is the
requirement.

## 7. Python Boundary

The React app calls a small Python HTTP API. The API framework is an
implementation detail; the request, response, and trace contracts are the
stable boundary.

The Python side owns:

- Profile validation and session creation;
- nudge decision and generation;
- calendar-week grouping, closed-week eligibility, and affected-week selection;
- Weekly Drift Reviewer calls and response validation;
- Drift Detector execution;
- Weekly Drift Detection output storage;
- Coach Digest generation after every Weekly Drift Detection result;
- forward-only assessment clock changes;
- idempotent retry behavior; and
- trace creation and retrieval.

The React side owns:

- onboarding interaction and local resumability;
- Experience and Inspect presentation;
- persona replay controls;
- failure-safe pending Journal Entry state;
- view selection and focused trace navigation; and
- accessible, responsive status and error presentation.

Provider keys and unredacted provider configuration stay on the Python side.

### 7.1 Version 1 contract

`experience-inspect-v1` defines five framework-neutral operations:

| Operation | Purpose |
|---|---|
| `create_session` | Validate a confirmed Profile, establish or resume in-memory shared session state, and synchronize one browser-held interaction or removal |
| `submit_journal_entry` | Append one ordered Journal Entry using an expected session revision |
| `advance_assessment_time` | Move an assessment-only clock forward by one day or to the next Monday |
| `load_scenario` | Load one deterministic saved persona scenario |
| `read_trace` | Retrieve typed trace events, optionally after a known event |

Python Pydantic models are the schema source. The checked-in JSON Schema and
canonical fixture are generated by
`uv run python -m src.demo.export_contract_schema`. React validates the same
fixture through `frontend/onboarding/src/demoContracts.ts`. The fixture covers
all 11 event types and complete, reused, refused, invalid, and failed results.

The following rules are part of the contract rather than a chosen HTTP
framework:

- `create_session`, `submit_journal_entry`, and `advance_assessment_time` carry
  a 64-character input hash
  as an idempotency key. Repeating the same key and input returns the stored
  result with `reused`; reusing the key for different input returns a safe
  conflict error before any model call.
- An existing session accepts only a one-revision browser update that either
  records one displayed nudge as answered or skipped, or removes one Journal
  Entry and its linked nudge. Python recomputes affected closed weeks that were
  already reviewed; broader state replacement is rejected. A same-revision
  resume must exactly match the current Journal Entries, nudges, and trace; it
  cannot silently replace or ignore divergent browser state.
- `submit_journal_entry` carries `expected_revision`. Python rejects a stale
  revision, duplicate Journal Entry identifier, duplicate `t_index`, or
  non-chronological Journal Entry before nudge or Weekly Drift Detection begins.
- `advance_assessment_time` carries `expected_revision`. Python rejects a
  backward date, an unanswered displayed nudge, or a close-week request without
  a finalized Journal Entry in the current week.
- Event order is represented by timestamps plus `parent_event_id`. Journal
  Entry order is represented by `t_index`; callers must not infer it from
  response array order alone.
- Provider secrets, authorization headers, and unredacted provider
  configuration never cross the boundary. Exact prompts and raw model
  responses may cross only after secret redaction. Errors expose a stable code,
  safe message, and retryable flag.
- Weekly Drift Detection events require `gpt-5.6-luna` with reasoning effort
  `low`. They contain Weekly Drift Reviewer Decisions. VIF Critic Predictions
  and their uncertainty fields are rejected by this contract.
- Saved replay and live results use the same payload shapes and differ through
  `source`. A saved result may use `reused`; caching remains optional.
- Version 1 is strict: unknown fields or incompatible values are rejected. The
  assessment clock is an optional, assessment-only extension. Browser sessions
  without it migrate with no assessment controls. Saved Persona bundles use a
  null clock and keep their existing behavior. Nested records can use their own
  version. The current Drift Detector record is `drift-detector-result-v2`, and
  React and Python both handle that exact version. A later incompatible change
  to the operation envelope requires a new Experience and Inspect contract
  version and explicit React and Python compatibility
  handling.

### 7.2 Live model trust boundary

The live Nudge, Weekly Drift Reviewer, and Coach Digest calls separate stable
Twinkl instructions from user-controlled data. OpenAI receives the stable rules
through its instruction field. Gemini receives the same rules through its
system-instruction field. Each provider receives Journal Entries, nudge
responses, preferred names, and current focus text as a separate JSON input.

The stable rules state that all JSON values are untrusted data. The model can
use this data only as evidence for the named task. It must not follow a command,
role, request, or delimiter inside the data. JSON serialization preserves text
that looks like a boundary without treating that text as a boundary.

Inspect stores one `live-prompt-boundary-v1` receipt that shows both provider
messages. The Nudge and Weekly Drift Reviewer prompt hashes cover this receipt.
The Coach Digest trace input hash continues to identify the structured Weekly
Drift Detection output. Response-schema validation, evidence validation, retry
behavior, and fail-closed behavior remain separate controls. Message
separation reduces prompt injection risk. It does not prove that a model will
always ignore an instruction-like phrase in user data.

## 8. Review Orchestration

For one manually submitted Journal Entry in the open week, the observable
sequence is:

```text
Journal Entry submitted
→ nudge suppression check
→ optional nudge decision and generation
→ optional user response or skip
→ Journal Entry finalized
→ wait for the calendar week to close
```

No Weekly Drift Reviewer, Drift Detector, Weekly Drift Detection output, or
Coach Digest response event is created for the open week. When a finalized
calendar week is due, the separate sequence is:

```text
closed Monday-through-Sunday week selected
→ Weekly Drift Reviewer runs with cumulative displayed history
→ response validated into Weekly Drift Reviewer Decisions
→ Drift Detector applies the deterministic rule
→ Weekly Drift Detection output is stored
→ Coach Digest runs for the stored result
→ a valid Coach Digest response is attached, or the result remains available
```

The nudge decision and question come from one structured
`gpt-5.6-luna` reasoning-effort-`none` call after the deterministic suppression
check. Inspect still records separate linked `nudge_decided` and
`nudge_generated` events: the provider prompt, raw response, model contract,
and latency belong to `nudge_decided`; question-length validation and the
effective displayed nudge belong to `nudge_generated`. A `no_nudge` decision
has no `nudge_generated` event.

The due-review caller supplies an `as_of` date already resolved in the user's
IANA timezone. A week is eligible only when its Sunday `week_end` is earlier
than `as_of`. Thus a first Journal Entry on Thursday is reviewed after Sunday,
not seven days later on the following Thursday. The first partial week is
eligible even when it contains only that Journal Entry. A displayed nudge must
be answered or skipped before its week is eligible.

The Python Experience service provides a due-review method for a scheduler or
host. The React POC uses the assessment clock and explicit close-week action.
It does not need a later Journal Entry to start due work. A production
background scheduler remains outside the capstone.

The backend may reuse an unchanged weekly result by input hash. Reuse must be
visible in Inspect and must return the same saved decisions and provenance. A
cache is an optimization, not a user-facing feature or a capstone result by
itself.

## 9. Persona Scenario Bundles

The capstone demo uses these five curated scenarios:

| Scenario | Persona | Core Values | Saved progression |
|---|---|---|---|
| No Active Drift | Meera Krishnamurthy, South Asian teacher, 45–54 | Achievement, Security | No Active Drift throughout |
| Active Drift | Wei Jun Chen, East Asian software engineer, 35–44 | Universalism | No Active Drift → Active Drift |
| Drift ended | Marc Vandenberghe, Western European manager, 45–54 | Power | No Active Drift → Active Drift → No Active Drift |
| Insufficient Evidence | Noor Haddad, Middle Eastern stay-at-home parent, 18–24 | Self-Direction, Tradition | Insufficient Evidence → No Active Drift → Active Drift → No Active Drift |
| Two Core Values | Lukas Vermeer, Western European software engineer, 25–34 | Self-Direction, Conformity | Conformity has No Active Drift while Self-Direction has Insufficient Evidence |

Lukas is the recommended professor walkthrough because his nine-week replay
shows displayed nudges and responses, two independent Core Value histories,
an ended Historical Drift Record, Abstain, Insufficient Evidence, and a grounded Coach Digest response in
the key week. The other four personas make each individual state easy to
demonstrate.

This menu covers seven Schwartz Core Values, four cultural backgrounds, four
age bands, and several work and family contexts. Selection favored coherent
week-by-week behavior over maximizing Core Value count: the reviewed
eight-value alternative began with Insufficient Evidence without a useful
No Active Drift progression.

Each saved scenario bundle contains or references:

- persona and Profile provenance;
- ordered Journal Entries, displayed nudges, and responses;
- calendar-week boundaries;
- rendered Weekly Drift Reviewer requests;
- raw responses and validation results;
- effective Weekly Drift Reviewer Decisions;
- Drift Detector results;
- Weekly Drift Detection outputs, Coach Digest event status, and valid Coach
  Digest responses when available;
- model contract, timestamps, response IDs when available, and input hashes;
  and
- a bundle manifest version, plus a content hash in the scenario catalog.

Scenario selection must be based on reviewed, reproducible behavior. Do not
rewrite Journal Entries or decisions merely to make the demonstration cleaner.
If a scenario is AI-reviewed synthetic development evidence, say so.

The checked-in files use Run 1 of the frozen `gpt-5.6-luna` reasoning-effort
`low` Weekly Drift Reviewer setup. Each onboarding Profile is a deterministic
projection from the synthetic persona's declared Core Values, not a claim that
the persona completed onboarding. Its provenance is
`synthetic_persona_projection`; the original React onboarding provenance
remains distinct. Generation metadata is retained only for Inspect nudge
provenance and is never supplied to the Weekly Drift Reviewer, Drift Detector,
Weekly Drift Detection, or Coach Digest.

The historical persona files preserve each displayed nudge's category, trigger,
text, and response, but not the original nudge provider prompt or raw response.
Saved nudge trace events therefore retain the available fields and leave the
unavailable provider fields null; they do not invent a receipt.

## 10. Privacy and Safety

- Inspect is a capstone and developer view, not a normal user destination.
- The default Persona replay uses synthetic personas. Manual Journal Entries
  are stored in browser storage for resume and in the matching in-memory Python
  session. Live work can send that text to the configured provider. Before the
  first manual Journal Entry, Experience requires acknowledgement of this data
  flow, the assessment-only scope, and the non-therapy boundary. Saved Persona
  replay does not require acknowledgement.
- Delete session removes the matching Python session, Inspect events, and
  request receipts before React clears browser storage. If Python deletion
  fails, React keeps the browser session and states that deletion was not
  confirmed. If browser removal fails after Python deletion, React keeps the
  current view and states that only the Python session was deleted.
- Delete session does not request deletion from the configured AI provider.
- The capstone POC does not add data export, production authentication,
  encryption infrastructure, or multi-user storage.
- Never display API keys, authorization headers, hidden environment values, or
  unrelated logs.
- Preserve the banned-term and value-leakage protections in generation and
  labeling work.
- Do not expose synthetic generation metadata to the Weekly Drift Reviewer,
  Drift Detector, Weekly Drift Detection output, or Coach Digest.
- Raw provider responses are visible only in Inspect and must be clearly
  separated from validated product results.

## 11. Responsive and Accessible Behavior

- Narrow-screen phones are the primary design and verification target. Start
  layout, interaction, and content decisions at the narrow viewport; treat
  wider layouts as progressive enhancements.
- The Experience/Inspect selector remains reachable at the top of every
  post-onboarding screen.
- On narrow screens, each view occupies the full screen; do not force a
  side-by-side debugger.
- Primary actions, Journal Entry composition, persona replay, Coach Digest
  reading, and event inspection remain usable without hover or precision
  pointer input.
- Persona replay controls and revealed week markers remain operable with touch
  and keyboard input; unrevealed weeks remain inert.
- A context-specific weekly Inspect action moves focus to the weekly
  explanation. It keeps the linked event selected and expanded. Other Inspect
  actions move focus to the selected event.
- Status changes and nudge availability use appropriate live-region behavior.
- Long prompts and responses wrap, preserve whitespace, and expand without
  horizontal page scrolling.
- Reduced-motion preferences disable automatic replay while keeping the
  explicit Previous and Next step controls.

## 12. Non-Goals

- Reimplementing onboarding in Shiny.
- Making the current Shiny Runtime Demo Review App the mobile-first product.
- Adding VIF Critic Predictions to the Weekly Drift Reviewer or user-facing
  Drift path.
- Presenting LLM-Judge labels as production decisions.
- Claiming human validation, a fresh final test, or deployment approval.
- Building production authentication, multi-tenant storage, notifications, or
  native mobile packaging in the first capstone demo slice.
- Turning every backend log line into Inspect content.

## 13. Professor Assessment Walkthrough

A release is assessment-ready when one uninterrupted walkthrough can:

1. complete or resume React onboarding and confirm Core Values;
2. submit a Journal Entry and observe a relevant nudge or a documented no-nudge
   decision;
3. switch to Inspect and view the exact nudge events without losing Experience
   state;
4. load an active-Drift persona scenario;
5. replay the persona week by week until the two consecutive Weekly Drift
   Reviewer Conflicts produce Drift;
6. inspect the exact weekly request, validated decisions, and deterministic
   Drift Detector steps;
7. return to Experience and read the Coach Digest response and reflective
   question;
8. demonstrate one ended Historical Drift Record or Insufficient Evidence; and
9. distinguish saved replay from an optional live run.

## 14. Implementation Order

1. Define the Python API, shared session, scenario bundle, and trace contracts.
2. Build deterministic scenario bundles with provenance checks.
3. Extend the React app with the shared Experience/Inspect shell.
4. Implement manual Journal Entry and nudge behavior through the Python API.
5. Integrate Weekly Drift Detection and Coach Digest behavior.
6. Implement persona replay in Experience.
7. Implement event-linked Inspect timelines and details.
8. Add end-to-end, accessibility, responsive, failure, and replay tests.
9. Prepare the professor walkthrough and update capstone documentation.

Contract work blocks integration work. Scenario replay and live Journal Entry
work may proceed in parallel after the contracts exist. Inspect must consume
real trace events rather than reconstructing backend behavior in the browser.

## 15. Tracked Implementation Work

The parent Beads epic is `twinkl-rklc`. The core P0 quality gate is complete.
The current Beads record controls the remaining feature-freeze and finalization
work. Each remaining feature must complete, move to future work, or leave the
capstone scope before the professor walkthrough starts.

| Beads issue | Priority | Scope | Blocked by |
|---|---:|---|---|
| `twinkl-rklc.1` | P0 | API, session, scenario, and trace contracts | — |
| `twinkl-rklc.2` | P0 | Deterministic persona scenario bundles | `.1` |
| `twinkl-rklc.3` | P0 | Shared React Experience/Inspect shell | `.1` |
| `twinkl-rklc.4` | P0 | Experience journaling and nudges | `.1`, `.3` |
| `twinkl-rklc.5` | P0 | Weekly Drift Detection and Coach Digest | `.1`, `.4` |
| `twinkl-rklc.6` | P0 | Week-by-week persona replay | `.2`, `.3`, `.5` |
| `twinkl-rklc.7` | P0 | Event-linked Inspect view | `.1`, `.3` |
| `twinkl-rklc.8` | P1 | Optional live rerun and visible reuse | `.5`, `.7` |
| `twinkl-rklc.9` | P0 | End-to-end demo quality gate | `.4`, `.5`, `.6`, `.7` |
| `twinkl-rklc.26` | P0 | Minimum privacy controls for manual journaling (complete) | — |
| `twinkl-rklc.27` | P0 | Current Coach Digest Validations and Coach Digest Evals results | — |
| `twinkl-rklc.28` | P1 | Coach Digest feedback and perceived accuracy | — |
| `twinkl-rklc.30` | P1 | Longitudinal Core Value history | — |
| `twinkl-rklc.10` | P4 | Professor walkthrough and capstone evidence | `.9`, `.26`, `.27`, `.28`, `.30` |

The P0 quality gate intentionally does not depend on optional live reruns. A
saved, deterministic replay must remain sufficient for the complete Persona
walkthrough. `twinkl-rklc.8` can therefore move to future work without blocking
the final walkthrough.

## 16. Verification Requirements

- Unit tests protect onboarding contracts, client state transitions, trace
  serialization, closed-week eligibility, affected-week selection, and
  deterministic replay.
- Contract tests verify React fixtures against Python request and response
  schemas.
- Integration tests cover successful, reused, refused, invalid, and failed
  model outcomes.
- End-to-end browser tests cover the professor walkthrough, open-week Journal
  Entries without weekly events, reply and skip, Journal Entry removal, active
  Drift that ends, Insufficient Evidence, and view-state preservation.
- Accessibility checks cover keyboard operation, focus, names, status updates,
  and reduced motion.
- Responsive checks treat representative narrow-screen phone viewports as the
  primary acceptance target and also cover representative desktop viewports.
- Saved scenario manifests are reproducible from their recorded inputs and
  reject mismatched hashes or model contracts.
