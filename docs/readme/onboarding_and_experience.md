# Onboarding and Experience

[Overview](../../README.md) | **Onboarding and Experience** | [Review Apps](review_apps.md) | [Research and Data](research_and_data.md) | [Status and Setup](status_and_setup.md)

---

## Onboarding — 🧪 Experimental React POC

The React app in [`frontend/onboarding/`](../../frontend/onboarding/) implements the published 11-group, six-object balanced SVBWS design. People tap or drag visually neutral cards into Most and Least boxes before a label-free Core Value summary and first Journal Entry handoff. Group and card order are randomized, raw BWS results remain separate from the ten-value Profile transformation, and there is no midpoint result or unsupported confidence field. This is a research-grounded pilot instrument, not a validated Twinkl instrument.

```sh
cd frontend/onboarding
npm install
npm run dev
```

The POC stores resumable progress and its confirmed Profile in the browser. The
manual Experience synchronizes the confirmed Profile and browser-held
interaction state with the in-memory Python boundary. A separate host can also
persist the Profile exposed by the callback or browser event, and the batch
runtime accepts saved Profile JSON with `--profile-path`. Production
multi-user storage and generalized persistence are outside the time-boxed
capstone.

## Experience and Inspect React App — 🚧 In Progress

The entry page offers **Try the Demo** for five saved synthetic Persona stories
and **Try Onboarding** for the personal assessment. Its introduction presents
the capstone's behavioral science, longitudinal reasoning, and experiments on
synthetic Journal Entries. A decorative compass responds to pointer movement,
stays static with reduced motion, and pauses when the browser tab is hidden.
The Twinkl wordmark returns to this page while preserving progress. Saved replay
Experience and Inspect use
**Choose another Persona** to return to the chooser; **Go home** remains in the
chooser, onboarding, and manual Experience and Inspect. **Continue replay**
resumes the current Persona from the picker.

The manual Experience submits Journal Entries to the versioned Python
boundary, supports displayed nudge reply and skip actions, reviews only closed
Monday-through-Sunday weeks, and keeps the Weekly Drift Detection result when
the Coach Digest cannot return a valid response. **Write on the next day** and
**Close week and review** advance Simulated time after the newest Journal Entry
is final. Journal Entry removal and saved nudge responses recompute affected
closed weeks that were already reviewed. Inspect reads the same Profile,
Journal Entries, Weekly Drift Reviewer Decisions, Drift state, Coach Digest
response, North Star Moment record, and trace events. Manual sessions retain
**View Profile calculation** in Inspect; saved Persona Profiles disclose their
synthetic projection.

**Reviewed week** opens a retained closed-week result, including its matching
Coach Digest and North Star Moment outcome. Browsing completed historical
results makes no model call; writing stays on the current Simulated time date.
The selection is stored by the week's Monday rather than its position in the
list. Active Drift presents the Journal Entries that started and continued the
conflict before other context. A Not Conflict decision does not label that
context as supportive.

Manual Experience shows progress and failure states for Coach Digest and
North Star Moment, as well as recorded ineligible or no-supportive-source
outcomes. **Retry Coach Digest** retries the saved Weekly Drift Detection
output without advancing time or repeating the Weekly Drift Reviewer. Each
explicit retry makes at most one generation attempt and uses matching prior
validation failures as repair instructions.
**Retry moment review** is available for an eligible retryable North Star
Moment failure. Historical results offer **Review moment** for an unreviewed
selection or **Resume moment review** for stored pending work. Browsing or
reloading those results does not start a review. Journal Entry source links
open a centered dialog whose dismissal restores focus and reading position.
If accepted work is missing its trace, **Try loading Inspect again** reloads
those details without repeating the accepted work.

**Inspect latest activity** opens the latest recorded event, while **Inspect
this moment** opens the event for the displayed quotation. A weekly explanation
uses the selected event's recorded run and dates, so results from different
weeks do not mix. The **Weekly results** filter includes Drift Detector,
structured Weekly Drift Detection output, Coach Digest, and North Star Moment
events. **View Profile calculation** shows the assessment without a weekly
summary. The live nudge prompt asks a brief, grounded, neutral open question
without advice, character judgment, or an implied obligation to apologize or
repair something.

Saved Persona replay shows the first Journal Entry immediately, followed by its
available nudge and response after 500 ms. Later Journal Entries and their
interactions follow at 500 ms intervals. **Review Weekly Drift Detection** can
open the saved result at any point; returning to Journal Entries shows the
whole week. **Next week** advances until the final week without requiring
review. Every week is directly selectable from the week navigator before
review; **Restart** and named key-week jumps provide additional navigation.
Week changes, reload, and returning from Inspect open Journal Entries first.
Results keep **Why this state** collapsed until requested, with source links
and per-Core-Value AI review details. Coach Digest
and North Star Moment cards have direct navigation when available.

The **Inspect** switch follows the visible replay panel: Journal Entries opens
**Recorded work** for saved Journal Entry and nudge processing, while the weekly
result opens its Weekly Drift Reviewer and Drift Detector explanation. Inspect
preserves this context across reloads. Expanded component events and the Profile
calculation include **Read the method** and **View implementation** links to a
fixed assessment reference revision; the recorded prompts and receipts remain
the evidence for each run.

The chooser compares Nisha (a Drift emerges), Noor (six weeks without Active
Drift), Lukas (one Drift continues across four weeks), Wei Jun (Insufficient
Evidence), and Meera (Drift affects Self-Direction while Tradition has No Active
Drift). The five selectable rows show weekly states by default, with separate
Core Value labels for Meera. Selecting a row updates its explanation, replay
length, and key week beside one action to start or continue that replay.
The **See how Twinkl works** guide explains the demo; the Experience/Inspect switch
appears after a replay opens. Retired Persona sessions return to this chooser.
Saved Weekly Drift prompt v4 Run 1 results cover 27 reviewed weeks and 53
Journal Entries. All 27 weeks have source-compatible Coach Digest responses.

North Star Moment displays at most one exact quotation within the Coach Digest,
after its narrative and reflective question, with a source date and Journal
Entry link. Active Drift uses a
supportive action from before onset; No Active Drift prefers a verified
current-week action and otherwise uses a historical reminder. Insufficient
Evidence produces no card. The saved replay preserves 22 selected quotations,
three ineligible outcomes, and two completed reviews without a supportive
source from the pinned full-history study. The replay's **With North Star
Moment** and **Without North Star Moment** controls switch the complete saved
Coach Digest response, including its reflective question. The default omits
the selected source context; the alternative includes it and the quotation.
When no valid comparison is available, the control explains the reason.
Toggling makes no provider call. Personal Experience keeps its original Coach
Digest response with an optional validated quotation. These are AI assessments
of synthetic writing without human review; human studies are closed outside
capstone scope. Live review uses the separate
`review_north_star` operation with a separately authorized US$1 allowance shared
across sessions and restarts. It fails closed when that allowance is exhausted
or its budget accounting is unavailable.

Saved Persona replay is deterministic and does not require a provider key. The
browser requests the scenario catalog and bundles with `cache: no-store`, then
verifies each bundle against its catalogued SHA-256 hash. Inspect presents both
completed and reused Coach Digest events as available responses. Refused,
invalid, and failed events remain unavailable.

Run the Python boundary from the repository root:

```sh
source .venv/bin/activate
uv run uvicorn src.demo.api:app --env-file .env --port 8000
```

Run the React development server in a second terminal:

```sh
cd frontend/onboarding
npm install
npm run dev
```

React checks are available through `npm test`, `npm run test:watch`,
`npm run typecheck`, and `npm run build`. Deterministic browser checks use the
built app, a controlled Python boundary, and Chromium at desktop and narrow
viewports. Run them from `frontend/onboarding/`:

```sh
npx playwright install chromium  # Once per Playwright browser version
npm run test:e2e
```

The Python environment must already be installed and port 8765 must be free.
The command builds the app and manages the test server; model calls use test
doubles. Desktop and laptop browsers are the primary verification target.
From the repository root, these commands regenerate the shared contracts and
saved experiment replay without provider calls:

```sh
source .venv/bin/activate
uv run python -m src.demo.export_contract_schema
uv run python -m scripts.export_demo_experiments
```

The replay export writes the compact North Star Moment records, five scenario
bundles, and catalog hashes from pinned experiment inputs. It preserves saved
Coach Digest responses only when their source hashes match the current inputs.
See the [Experience and Inspect guide](../../docs/demo/experience_inspect_app.md) for
the eight operations, assessment deployment, data boundary, and verification
workflow.

The [public assessment](https://onboarding-production-1dd2.up.railway.app/)
serves the React app and same-origin Python boundary. It allows anonymous access
for capstone assessment and can make paid provider calls during manual use. It
is not deployment approval and provides no production authentication,
multi-user storage, or service-level commitment.
