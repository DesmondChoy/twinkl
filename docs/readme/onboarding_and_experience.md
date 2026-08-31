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

The React app also contains the manual Experience, saved Persona replay, and
Inspect. Manual Experience submits Journal Entries to the versioned Python
boundary, supports displayed nudge reply and skip actions, reviews only closed
Monday-through-Sunday weeks, and keeps the Weekly Drift Detection result when
the Coach Digest cannot return a valid response. Inspect reads the same Profile,
Journal Entries, Weekly Drift Reviewer Decisions, Drift state, Coach Digest
response, and trace events.

Saved Persona replay is deterministic and does not require a provider key. The
browser requests the scenario catalog and bundles with `cache: no-store`, then
verifies each bundle against its catalogued SHA-256 hash. Inspect presents both
completed and reused Coach Digest events as available responses. Refused,
invalid, and failed events remain unavailable.

Run the Python boundary from the repository root:

```sh
uv run uvicorn src.demo.api:app --port 8000
```

Run the React development server in a second terminal:

```sh
cd frontend/onboarding
npm install
npm run dev
```

React checks are available through `npm test`, `npm run typecheck`, and
`npm run build`. Regenerate the shared JSON Schema and canonical fixture with
`uv run python -m src.demo.export_contract_schema` after a contract change.
See the [Experience and Inspect guide](../../docs/demo/experience_inspect_app.md) for
the six operations, assessment deployment, data boundary, and verification
workflow.

The [public assessment](https://onboarding-production-1dd2.up.railway.app/)
serves the React app and same-origin Python boundary. It allows anonymous access
for capstone assessment and can make paid provider calls during manual use. It
is not deployment approval and provides no production authentication,
multi-user storage, or service-level commitment.
