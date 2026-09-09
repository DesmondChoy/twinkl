# Experience and Inspect React App

## Status

North Star Moment application code covers onboarding and all five saved
Personas. After a closed-week review, it selects at most one exact quotation:
a pre-onset reminder during Active Drift, specific encouragement for a verified
current-week action without Active Drift, or a historical reminder for older
supportive writing. Insufficient Evidence and failed or unsuitable reviews
produce no card. The Coach Digest remains usable independently.
Saved replay uses Weekly Drift prompt v4 Run 1 and the full-history runtime
outcomes from the [completed targeted NSM update](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md).
The five selected Personas cover 27 reviewed weeks: 22 selected quotations,
three ineligible outcomes, and two completed reviews with no supportive source.
Original model receipts and synthetic source availability remain
inspectable. Live NSM uses a separately authorized US$1 allowance shared across
sessions and restarts; offline replay does not use that budget. The [methodology](../north_star/nsm_experiment_methodology.md)
discloses retained observations and the limits of AI assessment.

North Star Moment implementation, evaluation, and integration are complete
for the capstone (`twinkl-fz34` and `twinkl-fz34.8`). The final source-level
walkthrough, reconciliation of affected capstone documentation, and report
PDF/figure generation with visual verification belong to `twinkl-rklc.10`.
The integrated Coach Digest semantic concerns remain a separate follow-up in
`twinkl-rklc.39`.

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

The five deterministic Persona replays load into the shared React session
with immediate selected-week Journal Entries, explicit result review, previous-
and next-week navigation, restart, named jumps to key weeks, reduced-motion behavior,
no-future-data projection, and browser-side scenario hash verification. The
release quality gate is implemented. Every saved week has a Coach Digest whose
source hash matches that week's current Weekly Drift Detection output. The
[Persona replacement run](../../logs/experiments/reports/demo_persona_replacement_20260908/report.md)
adds ten responses for Lukas and Meera and retains 17 compatible responses for
Nisha, Noor, and Wei Jun from the
[September completion run](../../logs/experiments/reports/demo_coach_all_weeks_20260908/report.md).
All 27 use Luna at reasoning effort `none`, prompt `4.2`, and pass Coach Digest
Validations. These generation runs add no Coach Digest Evals or human
validation; the
[August evaluation manifest](../../logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json)
remains historical evidence for a different Persona roster and inputs.

| Persona | Weeks with source-compatible Coach Digests |
| --- | --- |
| Nisha Agarwal | 1–5 |
| Noor Haddad | 1–6 |
| Lukas Vetter | 1–5 |
| Wei Jun Chen | 1–6 |
| Meera Krishnamurthy | 1–5 |

The replacement uses exact saved weekly inputs, preserves compatible response
receipts, and validates one Coach event per week. It reuses Weekly Drift
Detection and full-history North Star Moment outcomes from the pinned studies.
The generation command and raw receipts are recorded with the replacement report.
The active replay files no longer include the two retired Personas; their
source datasets and historical experiment records remain intact.

The [integration validation](../../logs/experiments/reports/integrated_coach_validation_20260908/report.md)
records passing application checks and five AI editorial concerns in the
previous roster, tracked in `twinkl-rklc.39`. It remains historical evidence;
replacing two Personas does not resolve the retained Personas' concerns or
establish semantic agreement between the Coach Digest and North Star Moment.

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
- **Replay build boundary:** The frontend verifies catalogued scenario hashes
  and source-bound Coach Digest responses. The current replay uses Weekly Drift
  v4 Run 1 and saved full-history North Star Moment outcomes.
- **Deletion behavior:** After Profile confirmation, Delete session removes the
  matching in-memory Python session and request receipts before React clears
  browser storage. Before Profile confirmation, Start over clears browser-only
  progress.
- **Scope:** The deployment is for capstone assessment only. It is not
  deployment approval.
- **Excluded production controls:** Production authentication, multi-tenant
  persistence, and service-level commitments.

## 1. Purpose

The entry page introduces Twinkl with a full-width inner-compass banner and a
short product description adapted from the [README](../../README.md). A
top-right GitHub link opens the project repository in a new tab. Below the
introduction, Try the Demo and Try Onboarding lead into the two journeys;
the Twinkl home link returns to this page while preserving session progress.
The top-right action is **Choose another Persona** in saved replay Experience
and Inspect, returning to the five-Persona chooser. **Go home** remains available
in the chooser, onboarding, and manual Experience and Inspect.
The current Persona offers **Continue replay** in the picker to resume its
selected week after returning home.

The app presents the product experience and the AI architecture from the same
session. After a Persona replay opens, or during onboarding and the manual
Experience, a two-option control switches between:

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
- The VIF Critic (Offline) remains offline research. Inspect may link to separate
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
| Persona chooser | Enabled; shows the saved weekly comparison | No view switch; Inspect becomes available after opening a replay |
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
- the selected manual reviewed week, stored by its Monday date;
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
Inspect actions. The live Journal Entry path uses **Inspect latest activity**
for its latest recorded event. A displayed North Star Moment in either path
offers **Inspect this moment** for that quotation's exact event.
A weekly result action switches to Inspect and focuses the weekly explanation.
It also selects and expands the event that produced the result. Other Inspect
actions focus the selected event. Returning to Experience restores the same
screen position and selection where practical.

## 5. Experience View

Fresh visits open a two-choice screen. **Try the Demo** opens the current
five-Persona catalog; **Try Onboarding** opens the personal assessment. Existing
sessions resume their current flow. The Twinkl wordmark returns to the choice
without clearing progress, and choosing Onboarding resumes manual work or
starts a fresh personal Profile when leaving a synthetic Persona replay.
Loading a Persona over manual work retains the existing replacement confirmation.

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

The **Reviewed week** picker reads retained Weekly Drift Detection results and
their matching Coach Digest responses and North Star Moment records. Browsing
completed historical results does not call a model or change Simulated time;
the Journal Entry composer continues to use the current assessment date.
`selected_manual_week` stores the selected Monday, so adding or rebuilding
weeks does not shift the selection to a different array position. A null
selection follows the latest reviewed week. If a selected week is no longer
available after a source change, the view falls back to the latest remaining
result. Closing a new week selects its result.

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
If the accepted work is missing its Inspect trace, **Try loading Inspect again**
retrieves the trace only; it does not repeat submission or model work.

The live nudge question is brief, grounded in the writer's text, and open.
Its direct, sometimes blunt voice does not permit advice, character judgments,
invented motives, or an implied obligation to repair something. It may ask
about an apology the writer introduced, but must not assume an apology is
owed, wanted, or completed. The combined prompt retains its two-to-twelve-word
limit and can choose `no_nudge` when no useful neutral question is supported.

### 5.3 Persona simulation

The entry page offers **Try the Demo**. The chooser shows five selectable
Persona rows in the same order as the saved catalog: Nisha, Noor, Lukas,
Wei Jun, and Meera. Weekly states are visible without opening a guide, and
Meera's row labels the states for Self-Direction and Tradition separately.
A **See how Twinkl works** banner explains how to read saved Journal Entries,
see weekly results, and open Inspect to check which entries support them.
The header identifies the saved Persona demo without the onboarding Inspect hint.

On desktop, the comparison table spans the available content width. Compact
rows and spacing that adapts to the window height keep the selected Persona
panel visible without clipping content. Selecting a row updates the explanation, replay
length, and key week in the detail panel below the table. The key-week date
comes from the same catalogue field used by the replay shortcut. A single
**Start at week 1** action loads that Persona; selecting the current replay
instead offers **Continue replay** with its saved week.
On narrow screens, each Persona's weekly cells form a labelled grid, with the
selected details and the same action immediately below that Persona. The action loads the Profile,
Core Values, Journal Entries, displayed nudges and responses, saved Weekly
Drift Reviewer Decisions, Drift states, and Coach Digest responses.
**Choose another Persona** returns from saved replay Experience or Inspect to
the picker while preserving replay progress. Profile details contain only
the Persona context. The Twinkl wordmark always returns to the entry page;
**Go home** does so from the picker, onboarding, and manual Experience or Inspect.

The chooser and subsequent saved replay screens share Source Serif 4 headings
and Manrope body text, with a consistent heading scale, 16px body text and
actions, and 14.4px secondary labels. This includes Profile details, weekly
results, Inspect, and the Journal Entry and AI review drawers. Technical code
retains monospace. Shorter and narrow windows allow the replay page to scroll
vertically so the larger text stays accessible.

Persona simulation presents one week at a time. Each selected week immediately
shows all its saved Journal Entries as compact excerpts, with available nudges
and responses beneath their entries. **Review Weekly Drift Detection** opens
that week's saved result. The replay makes no timed or automatic result reveal.
**Next week** advances without requiring result review and is disabled only
at the final week. Every week is directly selectable without reviewing earlier weeks;
**Restart** returns to week one and clears replay progress.

On desktop, a quiet 170-pixel week navigator replaces the section-link banner.
It shows each week's number and dates, highlights the selected week, and reveals
states through the furthest reviewed week. Selecting an unreviewed week opens
its Journal Entries without revealing its outcome. **Restart** and the named key-week shortcut sit
below the list. The content area begins with a collapsed Profile row and a
compact selected-week heading beside **Next week**. At widths up to 900 pixels,
the weeks form a horizontally scrollable strip above the reading area.

Selecting a week projects only Journal Entries and evidence available by its
cutoff. All week markers remain selectable regardless of review progress. A named jump, such
as **Show Active Drift — week 4**, offers explicit navigation to a key week;
it opens that week's journals and still requires the review button to show its
result. The chooser's weekly comparison is derived from the same saved catalog
and labels weeks beyond a Persona's replay as unavailable.

The shared browser session preserves the selected week and furthest completed
week across Experience and Inspect and after reload. Existing saved step
progress remains readable, but does not hide Journal Entries. Initial load,
reload, returning from Inspect, week navigation, and restart open the journals
panel. Restart clears replay progress. Every week remains selectable regardless
of which results have been reviewed.
Sessions holding a retired Persona return safely to the current chooser rather
than attempting to load a removed bundle.

Opening a Journal Entry uses a desktop side panel or mobile bottom sheet so
the reading area does not reflow. Journal Entry and AI review dialogs keep
keyboard focus inside and make the background inactive until dismissal;
Escape, Close, or a backdrop click restores focus to the triggering control.
Saved nudges are available immediately with
their entries; manual journaling retains its separate delayed Nudge reveal.

When a saved Coach Digest response is present, it appears after the Weekly
Drift Detection result in the same reading area. It does not replace the
Weekly Drift Detection result. The short heading **Your weekly reflection**
precedes readable narrative paragraphs and links to the supporting Journal
Entries. Ellipsis-ended quotations expand through the source sentence only
when they match a unique occurrence in a cited Journal Entry available by the
week's cutoff. If inline expansion would interrupt the surrounding sentence,
the complete source quotation appears immediately below that paragraph.
The card discloses expansion; Inspect retains the original model response.
Ambiguous matches and ellipses present in the source remain
unchanged, with the complete Journal Entry available through its link.

The desktop and phone Experience use one expanded reading panel that grows with
its content. Journal Entries occupy the content width first, with **Review Weekly
Drift Detection** beside the panel heading (below it on phones). Revealing the
result hides the journals and places **Read Journal Entries** beside the result heading.
That button restores the journals and hides the result; **Read Weekly Drift
Detection** returns to the completed result without changing its evidence.
Every week change opens Journal Entries, including key-week jumps and returning
to a completed week, and returns the page to the top. The review button opens its result again.

The result uses two columns on wide desktops: a compact Drift state on the
left and one integrated Coach Digest on the right. Phones and narrower screens
stack these in the same order. The Coach Digest presents its original narrative,
then an optional validated North Star Moment passage, and ends with its one
original reflective question. No additional model rewrites the response.

**Inspect decision** beneath the Drift state opens the weekly explanation,
linked to the current week's Drift Detector event (or its saved Weekly Drift
Detection output when that event is unavailable). **Inspect this moment**
continues to focus the exact North Star Moment event. Inspect filters sit with
Recorded work, directly above the affected lists, and show matching current-week
and earlier-event counts beside the results.
Onboarding presents the passage without a North Star Moment label; Persona
replay adds a discreet attribution. Both paths provide **Inspect this moment**
for that exact backend event. Inspect distinguishes deterministic framing from
the exact source quotation and the AI assessment.

The result grows with its content and uses page scrolling. **Why this state**
keeps detailed evidence collapsed until requested; **Inspect decision** stays
with the Drift state. Missing or invalid Coach responses show their existing
status and suppress the passage. Missing, pending, failed, or unsuitable North
Star Moment results leave a valid Coach Digest intact, without an empty moment
card. Source links open the Journal Entry drawer without expanding the journals
panel. The active week stays centered in the horizontal week strip on narrow screens.

Profile details remain collapsed by default and include a short Persona context.
Each Core Value explanation names its state beside the Core Value. When a
Profile has two Core Values, the overall result explains that their states can
differ. A No Active Drift explanation for one Core Value does not contradict
Active Drift or Insufficient Evidence for the other Core Value.
The Persona header always names the selected Schwartz Core Values. State-change
evidence appears with the Weekly Drift Detection result. The first two Conflicts
show where Drift started. Later Conflicts show that Drift continued. No Active
Drift cites current Journal Entries and their Weekly Drift Reviewer Decisions.
When a successful Not Conflict decision ended the latest Drift, the explanation
also names and links that ending Journal Entry. It distinguishes an ending in
the selected week from an earlier Historical Drift Record, without implying
improvement or changing the current Drift state.
Insufficient Evidence cites the blocking Journal Entry and its Abstain or
failed review status when available.

Each cited Weekly Drift Reviewer Decision provides its saved model name,
reasoning effort, parsed model output, and recorded justification. Desktop
shows these details when the evidence card is hovered or receives keyboard
focus. An **AI review** action opens the same details. On a phone, that action
opens a bottom sheet. The details are available for Active Drift, No Active
Drift, and Insufficient Evidence. The replay identifies itself as
saved synthetic evidence throughout. The explicit result review must not imply live
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
  unavailable state;
- within a valid Coach Digest, at most one North Star Moment with the exact
  quotation, source date, and Journal Entry link when an eligible review selects
  a supportive action, followed by the original reflective question.

Experience does not show the full Weekly Drift Reviewer prompt, provider
payload, validation record, identifiers, or hashes. Those belong in Inspect.
Experience does show the Weekly Drift Reviewer Decision, saved model contract,
parsed model output, and recorded justification beside each cited Journal
Entry. It does not claim that reasoning effort is a readable chain of thought.

In manual Experience, Active Drift shows the triggering pair and current
Conflict evidence before other Journal Entry context. The triggering pair
comes from the Drift record's onset and confirmation indices, checked against
the available Weekly Drift Reviewer Decisions. Other context appears
separately and is not labeled supportive merely because a decision was Not
Conflict. Selecting evidence opens and focuses its Journal Entry.

Manual Coach Digest and North Star Moment work exposes progress and failure
states while keeping the Weekly Drift Detection result visible. Recorded
ineligible North Star Moment outcomes and completed reviews with no supportive
source receive an explanation without a quotation. Pending or failed review
does not fall back to an older quotation. **Retry moment review** appears only
when the current failure permits retry. **Retry Coach Digest** uses the stored
Weekly Drift Detection output; it does not advance Simulated time or repeat
the Weekly Drift Reviewer. A failed trace refresh after accepted retry work
instead offers **Try loading Inspect again**.
Invalid Coach responses remain unpublished and can also be retried. Prompt
4.3 makes the existing verbatim-quotation requirement explicit; validation
still checks each generated response. Until Coach Digest is available, the
Moment status explains that its review is waiting for the weekly reflection.

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
Live Coach Digest and North Star Moment durations measure the complete
operation; North Star Moment retains the recorded start of its pending work.

For Persona replay, Inspect shows the selected week first. Filters select
**Journal Entries**, **Weekly Drift Reviewer**, or **Weekly results**. The
last includes Drift Detector, structured Weekly Drift Detection output, Coach
Digest, and North Star Moment events.
The complete earlier Inspect history stays collapsed by default. Repeated
saved-run labels, reused-result labels, and zero-duration labels do not appear
on each event. Model details, run source, reasoning effort, identifiers, hashes,
and exact inputs remain under **Technical details**.

The weekly explanation follows the selected event's parent chain and recorded
week dates rather than taking the latest event from each component across the
session. This keeps earlier Coach Digest responses and North Star Moment
records attached to the Weekly Drift Detection run that produced them.
Opening the Profile calculation suppresses that weekly explanation and its
event filters until **View recorded events** is selected again.

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
Journal Entry handoff, manual Journal Entry work, and saved-run Inspect.
Saved Persona replay uses the week navigator described in section 5.3.
The Persona chooser uses an informational banner instead of a section map.
Each section map links only to sections in
the current view. Content with a maximum width stays centered between the
section rail and the outer page edge. The active values questions retain the
compass because it shows assessment progress. Narrow screens do not show a
section rail.

This calculation is labelled **Calculation method** and **Deterministic · no
model**.
After manual Journal Entries begin, **View Profile calculation** reopens it
from Inspect, and **View recorded events** returns to the event list. Saved
Persona Inspect explains the synthetic Profile projection and does not offer
this manual assessment control.
It is not fabricated as a Python trace event. Profile confirmation remains the
first Python event, preserving the React ownership of onboarding scoring and
the Python ownership of confirmed Profile validation.

### 6.2 Trace event types

Inspect represents these events when applicable:

1. `profile_confirmed`
   - Profile validation, Core Values, and Profile provenance.
2. `assessment_time_advanced`
   - next-day or close-week action, previous and current Simulated time, input
     hash, and validation.
3. `journal_entry_submitted`
   - Journal Entry date, text reference, ordering validation, and session ID.
4. `nudge_suppression_checked`
   - previous-three-entry window and whether the anti-annoyance rule suppressed
     a nudge.
5. `nudge_decided`
   - sanitized inputs, exact prompt, model, category, reason, response,
     validation, and latency.
6. `nudge_generated`
   - exact prompt, generated question, word-count validation, attempts, and
     latency.
7. `weekly_review_requested`
   - week boundaries, cumulative displayed Journal Entry history, Core Values,
     prompt, fixed model contract, and input hash.
8. `weekly_review_completed`
   - raw provider response, validation result, effective Weekly Drift Reviewer
     Decisions, response ID when available, attempts, and latency.
9. `drift_detected`
   - the ordered Weekly Drift Reviewer Decisions considered, the deterministic
     rule steps, and resulting Drift state.
10. `weekly_digest_built`
   - structured Weekly Drift Detection output fields, cited Journal Entries, and source Drift
     state.
11. `weekly_coach_generated`
    - exact prompt, model, response, narrative validation, and latency.
12. `nudge_response_recorded`
    - the user response, linked Journal Entry and nudge, and independent
      server-recorded availability timestamp used for North Star Moment sources.
13. `north_star_reviewed`
    - the frozen review record, source checks, selected quotation or no-card
      outcome, model receipts, usage, and retry status.

A `weekly_coach_generated` event with `complete` or `reused` status presents the
Coach Digest response as available. A `refused`, `invalid`, or `failed` status
presents it as unavailable without removing the Weekly Drift Detection result.

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
- North Star Moment source eligibility, bounded model review, and record reuse;
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

`experience-inspect-v1` defines eight framework-neutral operations:

| Operation | Purpose |
|---|---|
| `create_session` | Validate a confirmed Profile, establish or resume in-memory shared session state, and synchronize one browser-held interaction or removal |
| `submit_journal_entry` | Append one ordered Journal Entry using an expected session revision |
| `advance_assessment_time` | Move an assessment-only clock forward by one day or to the next Monday |
| `delete_session` | Remove one matching in-memory Python session, trace, and request receipts before browser state is cleared |
| `load_scenario` | Load one deterministic saved persona scenario |
| `read_trace` | Retrieve typed trace events, optionally after a known event |
| `review_north_star` | Review a frozen closed-week snapshot separately, reuse matching records, or retry a retryable NSM failure |
| `retry_coach` | Retry Coach Digest for stored live Weekly Drift Detection output without advancing time or repeating the Weekly Drift Reviewer |

Python Pydantic models are the schema source. The checked-in JSON Schema and
canonical fixture are generated by
`uv run python -m src.demo.export_contract_schema`. React validates the same
fixture through `frontend/onboarding/src/demoContracts.ts`. The canonical
fixture retains the original event examples; focused tests also cover
`north_star_reviewed` and `nudge_response_recorded`. The current saved NSM
records preserve completed v4 Run 1 experiment outcomes. Live NSM completion
updates its pending event, so clients
refresh the complete trace after a dedicated review rather than using a cursor
that would omit the updated event.

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
- `review_north_star` carries the session revision, reviewed Monday, and an
  explicit retry flag. The backend snapshots the corresponding saved weekly
  review, releases the session lock before token counting or model calls, and
  verifies the inputs again before publishing. Identical in-flight requests
  share one worker. That worker owns completion and cleanup even if its HTTP
  waiter disconnects. Source edits remove affected records; stale output is
  discarded. A restored pending event is reused rather than left running.
- `retry_coach` carries the expected session revision, reviewed Monday, and
  an idempotency key. It uses that week's latest stored live Weekly Drift
  Detection output, reuses a matching valid Coach Digest response when one
  exists, and otherwise runs Coach Digest alone. Repeating the same key returns
  the accepted response; reusing it with different inputs is rejected.
  Retrying an earlier week does not replace the session's latest weekly result.
  If the local service restarted, the browser can restore its complete saved
  session and trace before retrying. Idempotency receipts themselves remain
  in service memory and do not survive a restart.
- A server-timestamped `nudge_response_recorded` event proves response
  availability separately from its parent Journal Entry. Original submission
  identity, date, order, and content must match before NSM uses the source.
- NSM records bind the confirmed Profile content, owner, source window, weekly
  review, source text and availability, prompt and policy. Browser Profile
  hashes normalize integral floats and optional null preferred names for parity.
  Missing legacy records show no card until reviewed. No positive claim is
  inferred from No Active Drift.
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
→ React requests North Star Moment review for the frozen closed-week snapshot
```

`review_north_star` is a separate request. Its failure leaves Weekly Drift
Detection and a valid Coach Digest available. Matching completed records are
reused; eligible retryable failures expose an explicit retry. The live runtime
uses the pinned `config/evals/north_star_live_v1.json` policy and fixed
`logs/exports/demo_tool_runs/north_star/` ledger. The US$1 allowance is shared
across sessions and restarts, with two attempts at most per request, no SDK
retries, a 16,000-token input limit, and a US$0.25 per-attempt cap. It does not
recreate the removed integration experiment budget. Corrupt or changed ledger
policies and exhausted budgets fail closed. Saved replay retains its original
integration-policy receipts; validation accepts only the two repository-pinned
policies and requires one consistent policy across a record's reviews.
Saved Persona replay reads its prepared records without a provider or live budget.

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
| A Drift emerges | Nisha Agarwal, South Asian teacher, 18–24 | Universalism | No Active Drift in weeks 1–3 → Active Drift in week 4 → No Active Drift in week 5 |
| Six weeks without Active Drift | Noor Haddad, Middle Eastern stay-at-home parent, 18–24 | Self-Direction, Tradition | No Active Drift in all six weeks |
| Drift continues across weeks | Lukas Vetter, Western European software engineer, 25–34 | Universalism | Active Drift in weeks 1–4 → No Active Drift in week 5; this is one continuing Drift, not four separate Drifts |
| Insufficient Evidence | Wei Jun Chen, East Asian software engineer, 35–44 | Universalism | No Active Drift in weeks 1–4 → Insufficient Evidence in weeks 5–6 |
| Drift affects one Core Value | Meera Krishnamurthy, South Asian stay-at-home parent, 45–54 | Self-Direction, Tradition | Week 1 has Active Drift for Self-Direction and No Active Drift for Tradition; both have No Active Drift in weeks 2–5 |

Nisha is the recommended professor walkthrough: her five-week replay shows
how two consecutive Conflicts produce Active Drift and how a later Not Conflict
decision ends that pattern. Her key week starts on 3 March 2025. Lukas's
week starting 30 June shows that the same Drift remains active in week 4;
Wei Jun's week starting 30 June demonstrates a failed review that blocks a
current claim; Meera's week starting 10 November demonstrates contrasting
Core Value states. Noor's week starting 19 May closes the No Active Drift
baseline without implying that this state proves alignment.

This menu covers three Schwartz Core Values, four cultural backgrounds, four
age bands, 53 Journal Entries, and 27 reviewed weeks. Selection used saved
v4 Run 1 Drift patterns, coherent histories, completed NSM source coverage,
and compatibility with the existing displayed-nudge anti-annoyance rule.
It did not use AI evaluation grades to select successful-looking cards.

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
- full-history North Star Moment records, including no-card outcomes;
- model contract, timestamps, response IDs when available, and input hashes; and
- a bundle manifest version, plus a content hash in the scenario catalog.

Scenario selection must be based on reviewed, reproducible behavior. Do not
rewrite Journal Entries or decisions merely to make the demonstration cleaner.
If a scenario is AI-reviewed synthetic development evidence, say so.

The checked-in files use the definitions variant of prompt v4, Run 1, with
`gpt-5.6-luna` at reasoning effort `low`, from the [completed Core Value context
comparison](../../logs/experiments/reports/experiment_review_2026-09-07_twinkl_j3k7_core_value_definitions.md). Each onboarding Profile is a deterministic
projection from the synthetic persona's declared Core Values, not a claim that
the persona completed onboarding. Its provenance is
`synthetic_persona_projection`; the original React onboarding provenance
remains distinct. Generation metadata is retained only for Inspect nudge
provenance and is never supplied to the Weekly Drift Reviewer, Drift Detector,
Weekly Drift Detection, or Coach Digest.

The browser requests the scenario catalog and each bundle with
`cache: no-store`, then verifies the bundle against the catalogued SHA-256 hash
before it displays any saved Persona data. This keeps a deployment from reusing
an older browser-cached scenario while preserving the content-integrity check.

The historical persona files preserve each displayed nudge's category, trigger,
text, and response, but not the original nudge provider prompt or raw response.
Saved nudge trace events therefore retain the available fields and leave the
unavailable provider fields null; they do not invent a receipt.

### Reproduce the saved replay

From the repository root:

```sh
source .venv/bin/activate
uv run python -m scripts.export_demo_experiments
```

This command verifies the pinned completed North Star Moment experiment and
projects its full-history runtime selections and no-card outcomes into
`src/demo/north_star_replay_records.json`. It then writes the five scenario
bundles and their SHA-256 catalog in `frontend/onboarding/public/scenarios/`
using saved Weekly Drift v4 Run 1 requests, responses, attempts, and manifest.
Source-bound Coach Digest responses come from
`src/demo/coach_digest_responses.json`. The command takes no options and makes
no provider calls; it does not generate or evaluate replacement responses.

`uv run python -m src.demo.scenarios` exports only the bundles and catalog from
the existing compact records. After changing a contract, regenerate its JSON
Schema and canonical fixture with
`uv run python -m src.demo.export_contract_schema`. The [saved demo reproduction
record](../../logs/experiments/reports/demo_v4_run1_20260907/README.md) identifies
the source files and separate provider-backed Coach Digest generation workflow.

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
- Persona replay controls and all week markers remain operable with touch
  and keyboard input. Week selection and **Next week** open Journal Entries
  without revealing the result; explicit review reveals that selected result.
- Manual **Write** navigation appears only while the Journal Entry composer
  exists. It is omitted before the first-use notice is acknowledged and while
  a displayed Nudge awaits a response.
- A context-specific weekly Inspect action moves focus to the weekly
  explanation. It keeps the linked event selected and expanded. Other Inspect
  actions move focus to the selected event.
- Status changes and nudge availability use appropriate live-region behavior.
- Long prompts and responses wrap, preserve whitespace, and expand without
  horizontal page scrolling.
- Saved replay uses explicit Review Weekly Drift Detection and week navigation
  controls. Reduced-motion preferences suppress decorative entry movement.

## 12. Non-Goals

- Reimplementing onboarding in Shiny.
- Making the current Shiny Runtime Demo Review App the mobile-first product.
- Adding VIF Critic Predictions to the Weekly Drift Reviewer or user-facing
  Drift path.
- Presenting LLM-Judge VIF Labels as production decisions.
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

The React checks run from `frontend/onboarding/`:

```sh
npm test
npm run test:watch
npm run typecheck
npm run build
```

For controlled North Star Moment browser checks, run this Python boundary
instead of `src.demo.api:app`, alongside the React development server:

```sh
source .venv/bin/activate
uv run uvicorn scripts.demo_north_star_qc:app --port 8000
```

Select a response mode through its local control endpoint:

```sh
curl -X POST http://127.0.0.1:8000/qc/mode/failure
curl -X POST http://127.0.0.1:8000/qc/mode/success
```

| Mode | Controlled behavior |
|---|---|
| `success` | Selects the fixture's exact supportive quotation when the source contains it and Benevolence is the reviewed Core Value |
| `failure` | Returns a retryable provider-unavailable result |
| `omission` | Reviews sources without accepting a supportive action |
| `pending` | Holds North Star Moment completion until another mode releases it |
| `long` | Supports the long fixture quotation when its complete text appears in the source |

The fixture quotations are `QUOTE` and `LONG_QUOTE` in
[`scripts/demo_north_star_qc.py`](../../scripts/demo_north_star_qc.py). This
harness uses deterministic Weekly Drift Reviewer, nudge, Coach Digest, token
counting, and North Star Moment test doubles. Its budget and count files live
in a temporary directory, and it makes no paid provider calls. It is separate
from the deployed app and provides browser behavior evidence, not model-quality
or human-validation evidence.

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
