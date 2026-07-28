# Twinkl onboarding

Standalone React proof of concept for the published Schwartz Values Best-Worst
Survey (SVBWS) onboarding flow. This is a research-grounded pilot instrument,
not a psychometrically validated Twinkl instrument. It produces a confirmed,
versioned Profile. A host can persist the Profile exposed by the handoff, and
the approved runtime imports its Core Values from saved JSON.

The assessment contains 11 randomized groups of six neutral cards from the
published balanced design. Each group uses six abstract backgrounds assigned
by randomized display position rather than value identity. People can tap,
drag, or use the keyboard to make Most and Least choices. Schwartz labels
remain internal. The Profile keeps raw
11-object BWS results separate from the ten-value product transformation, with
no midpoint result or confidence proxy. The 11th group advances directly to
the label-free Core Value summary. The final action opens the manual
Journal Entry flow. The React Experience passes the confirmed Profile and
ordered Journal Entries through the versioned Python boundary, applies the
anti-annoyance rule, and shows the resulting displayed nudge with reply or
skip actions. The same submission runs the fixed Weekly Drift Reviewer,
applies the Drift Detector, and shows a cited Weekly Digest. Inspect reads the
linked live trace events. Profile confirmation starts this trace when the
Python boundary is available; without it, Experience stays usable and Inspect
shows zero events instead of fixture events. Retryable failures include a retry
action. **Try demo** loads one of five saved synthetic personas into the same
React session and replays Journal Entries, displayed nudges and responses,
Drift, Weekly Digests, and Inspect events one week at a time. Previous, next,
play or pause, and restart controls preserve the selected week across
Experience and Inspect; reduced-motion preferences disable automatic
advancement. The browser verifies each scenario against the catalogued SHA-256
hash before displaying it. The Profile remains available through the
`onStartJournal` callback and
`twinkl:start-first-journal` browser event.

[`docs/onboarding/onboarding_spec.md`](../../docs/onboarding/onboarding_spec.md)
is the canonical workflow and evidence-boundary documentation. Background
generation provenance is in
[`public/card-backgrounds/README.md`](public/card-backgrounds/README.md).

## Run locally

```sh
source .venv/bin/activate
uv run uvicorn src.demo.api:app --port 8000
```

In a second terminal:

```sh
cd frontend/onboarding
npm install
npm run dev
```

## Checks

```sh
npm test
npm run build
```

## Railway deployment

Create a Railway service from this repository with:

- root directory: `/`
- config file path: `/frontend/onboarding/railway.json`
- branch: `main`

The repository root is required because the image builds React and includes
the existing `src.demo.api` Python boundary. Uvicorn serves the built React
files, the public `/health` route, and same-origin `/api/experience` requests
from one Railway process. The Docker build context excludes `.env`, Git data,
development caches, and unrelated experiment outputs.

Set `OPENAI_API_KEY` to enable live provider-backed Journal Entry work.
Onboarding and saved persona replay remain available without it; manual
provider work fails safely and retains the Journal Entry for editing or retry.
Remove the obsolete `TWINKL_DEMO_USERNAME` and `TWINKL_DEMO_PASSWORD`
variables; the deployment no longer reads them.
The public Railway URL has no username or password gate, so anyone with the URL
can trigger paid provider calls. Keep the deployment URL private when it is not
being demonstrated, and use provider-side usage limits appropriate for a
time-boxed capstone POC.

Build the same image locally from the repository root:

```sh
docker build -f frontend/onboarding/Dockerfile -t twinkl-experience .
docker run --rm -p 3000:3000 twinkl-experience
```

The React Experience stores unfinished progress in the browser. The local
Python boundary keeps the active session and idempotency receipts in memory,
so restarting it clears backend state. Before the next Journal Entry, React
restores the confirmed browser-held Journal Entries, nudges, and trace events
through the validated session request. Provider keys stay on the Python side.
Nudge reply and skip outcomes remain in the resumable browser session, while
the Python boundary records nudge generation events for Inspect. Saving either
outcome, or confirming Journal Entry removal, advances the session revision and
recomputes the affected week plus any later weeks. A failed synchronization
keeps the Journal Entry or response in the browser for a contextual retry.
Removed Journal Entry positions are not reused, and Inspect marks their
immutable submission events as removed from the current Experience.
Production authentication, multi-tenant storage, and generalized persistence
remain outside the time-boxed capstone.
