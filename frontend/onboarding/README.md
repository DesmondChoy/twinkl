# Twinkl React app

Shared React app for the Schwartz Values Best-Worst Survey (SVBWS) onboarding,
Experience, and Inspect. The onboarding phase is a research-grounded pilot
instrument, not a psychometrically validated Twinkl instrument. It produces a
confirmed, versioned Profile and synchronizes the Profile and browser-held
Experience state with the in-memory Python boundary. A separate host can also
persist the Profile exposed by the handoff, and the batch runtime imports its
Core Values from saved JSON.

The flow first asks what Twinkl should call the user, then presents 11
randomized groups of six neutral cards from the published balanced design.
Each group uses six abstract backgrounds assigned by randomized display
position rather than value identity. People can tap, drag, or use the keyboard
to make Most and Least choices. Schwartz labels remain internal. The Profile
stores the preferred name and keeps raw
11-object BWS results separate from the ten-value product transformation, with
no midpoint result or confidence proxy. The 11th group advances directly to
the label-free Core Value summary. A Profile has at most two Core Values. If
more than two values share the highest score, the user selects exactly two and
the Profile retains every tied value. The final action opens the
manual Journal Entry flow. The React Experience passes the confirmed Profile and
ordered Journal Entries through the versioned Python boundary, applies the
anti-annoyance rule, and shows the resulting displayed nudge with reply or
skip actions. Saving a Journal Entry does not review its open
Monday-through-Sunday week. Manual Experience starts one Simulated time date
from the browser timezone. After the newest Journal Entry is final, the user
can move to the next day or close the week. Closing the week moves to the next
Monday. It runs the fixed Weekly Drift Reviewer and applies the Drift Detector.
Coach Digest then runs for every Weekly Drift Detection result, including No
Active Drift. If Coach Digest cannot return a valid response, the Weekly Drift
Detection result remains available. The first partial week follows the same
rule. Inspect reads the live trace events. Profile confirmation starts this
trace when the Python boundary is available. Without it, Experience stays
usable and Inspect shows zero events instead of fixture events. Retryable
failures include a retry action. **Try demo** loads one of five saved synthetic
personas into the same
React session and replays Journal Entries, displayed nudges and responses,
Drift, Coach Digest responses, and Inspect events one week at a time. **Next step** is
the default. **Previous** returns to an earlier week. **Auto replay** and
**Pause replay** provide optional automatic replay. **Restart** and **Jump to
key moment** provide quick navigation. These controls preserve the selected
week across Experience and Inspect. Reduced-motion preferences disable Auto
replay. The browser verifies each scenario against the catalogued SHA-256 hash
before displaying it. The Profile remains available through the
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
npm run test:watch
npm run typecheck
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

The frontend build imports the exact Coach Digest evaluation manifest used by
the five saved Persona key weeks, so the Dockerfile copies that manifest before
`npm run build`. The browser requests the scenario catalog and bundles with
`cache: no-store`, then verifies each bundle against its catalogued SHA-256
hash.

Set `OPENAI_API_KEY` to enable live provider-backed Journal Entry work.
Onboarding and saved Persona replay remain available without it; manual
provider work fails safely and retains the Journal Entry for editing or retry.
`TWINKL_DEMO_USERNAME` and `TWINKL_DEMO_PASSWORD` are unused.
The public Railway URL has no username or password gate, so anyone with the URL
can trigger paid provider calls. Use provider-side usage limits appropriate for
a time-boxed capstone POC, and remove live keys when paid calls are not needed.

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
If an older confirmed Profile contains more than two Core Values, React keeps
the SVBWS responses, Journal Entries, draft text, and Simulated time. It asks
the user to choose two Core Values, starts a new Experience session, and clears
old Profile-dependent outputs.
Nudge reply and skip outcomes remain in the resumable browser session, while
the Python boundary records nudge generation events for Inspect. Saving either
outcome, or confirming Journal Entry removal, advances the session revision and
recomputes affected closed weeks that were already reviewed; an open week
remains unreviewed. The Python boundary owns forward-only Simulated time
changes. It stores the user's IANA timezone with the assessment clock. The
manual Experience shows Journal Entry cards newest first. Stored Journal
Entries, Weekly Drift Detection input, and Inspect events stay in chronological
order. A production background scheduler remains outside the capstone. A
failed synchronization keeps the Journal Entry or response in the browser for
a contextual retry.
Removed Journal Entry positions are not reused, and Inspect marks their
immutable submission events as removed from the current Experience. Production
authentication, multi-tenant storage, and generalized persistence remain
outside the time-boxed capstone.

Before the first manual Journal Entry, Experience explains browser storage,
temporary Python memory, AI provider processing, assessment-only use, and the
non-therapy boundary. Saved Persona replay does not require acknowledgement.
Delete session removes the matching Python session and request receipts before
React clears browser storage. If Python deletion fails, React keeps the browser
session and does not claim success. If browser removal fails after Python
deletion, React keeps the current view and states the partial result. Data
export and provider-side deletion remain outside this capstone POC.
