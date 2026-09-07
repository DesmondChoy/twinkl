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
and **Try Onboarding** for the personal assessment. The Twinkl wordmark and
**Go home** return to this page while preserving progress. **Continue replay**
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

Saved Persona replay shows all Journal Entries, displayed nudges, and responses
for the selected week immediately. **Review Weekly Drift Detection** opens the
result in the full weekly workspace; **Next week** becomes available after
review. **Previous**, **Restart**, reviewed week markers, and named key-week
jumps provide navigation. Week changes, reload, and returning from Inspect
open Journal Entries first. Results keep **Why this state** collapsed until
requested, with source links and per-Core-Value AI review details. Coach Digest
and North Star Moment cards have direct navigation when available.

The five Personas are Noor (No Active Drift), Nisha (Active Drift), Sook Yin
(an ended Historical Drift Record), Wei Jun (Insufficient Evidence), and Henrik
(independent Core Value states). Their saved Weekly Drift prompt v4 Run 1
results cover 27 reviewed weeks and 52 Journal Entries.
Saved Coach Digest responses cover one key week per Persona; the other 22
weeks retain Weekly Drift Detection output without a saved Coach Digest response.

North Star Moment displays at most one exact quotation beneath the Coach
Digest, with its source date and Journal Entry link. Active Drift uses a
supportive action from before onset; No Active Drift prefers a verified
current-week action and otherwise uses a historical reminder. Insufficient
Evidence produces no card. The saved replay preserves 23 selected quotations,
three ineligible outcomes, and one completed review without a supportive
source. These are AI assessments of synthetic writing, with human review
deferred. Live review uses the separate `review_north_star` operation and fails
closed without a finalized integration budget; the live budget is unavailable.

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
`npm run typecheck`, and `npm run build`. From the repository root, these
commands regenerate the shared contracts and saved experiment replay without
provider calls:

```sh
source .venv/bin/activate
uv run python -m src.demo.export_contract_schema
uv run python -m scripts.export_demo_experiments
```

The replay export writes the compact North Star Moment records, five scenario
bundles, and catalog hashes from pinned experiment inputs. It preserves saved
Coach Digest responses only when their source hashes match the current inputs.
See the [Experience and Inspect guide](../../docs/demo/experience_inspect_app.md) for
the seven operations, assessment deployment, data boundary, and verification
workflow.

The [public assessment](https://onboarding-production-1dd2.up.railway.app/)
serves the React app and same-origin Python boundary. It allows anonymous access
for capstone assessment and can make paid provider calls during manual use. It
is not deployment approval and provides no production authentication,
multi-user storage, or service-level commitment.
