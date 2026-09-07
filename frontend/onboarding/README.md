# Twinkl React app

Shared React app for the Schwartz Values Best-Worst Survey (SVBWS) onboarding,
Experience, and Inspect. The onboarding phase is a research-grounded pilot
instrument, not a psychometrically validated Twinkl instrument. It produces a
confirmed, versioned Profile and synchronizes the Profile and browser-held
Experience state with the in-memory Python boundary. A separate host can also
persist the Profile exposed by the handoff, and the batch runtime imports its
Core Values from saved JSON.

Fresh visits start with two choices: **Try the Demo** opens the current saved
Persona catalog; **Try Onboarding** starts the personal assessment. The Twinkl
wordmark returns to this choice without clearing progress. Existing sessions
resume their current flow. Choosing Onboarding from a saved Persona starts a
new personal Profile without carrying over synthetic Journal Entries.

The onboarding flow asks what Twinkl should call the user, then presents 11
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
**Pause replay** provide optional automatic replay. **Restart** and named jumps
such as **Show Active Drift — week 4** in Nisha's replay provide quick navigation. The Persona
picker includes a collapsed professor guide mapping saved weeks to Drift
states. Each Core Value explanation names its own state. These controls preserve
the selected week, revealed step, and previously revealed weeks across Experience
and Inspect and after reload. Automatic replay pauses when Experience closes.
Reduced-motion preferences disable Auto replay. The browser verifies each
scenario against the catalogued SHA-256 hash
before displaying it. The Profile remains available through the
`onStartJournal` callback and
`twinkl:start-first-journal` browser event.

After manual Journal Entries begin, Inspect retains **View Profile calculation**
and **View recorded events** controls. The calculation uses the original 22
choices. Saved Persona Profiles are labelled as synthetic projections and do
not offer a fabricated completed assessment. The manual summary explains why
one or two Core Values are shown; the one-or-two scoring contract is unchanged.

[`docs/onboarding/onboarding_spec.md`](../../docs/onboarding/onboarding_spec.md)
is the canonical workflow and evidence-boundary documentation. Background
generation provenance is in
[`public/card-backgrounds/README.md`](public/card-backgrounds/README.md).

## North Star Moment

Both frontend paths display at most one North Star Moment beneath the Coach
Digest. Active Drift uses a supportive action from before onset. No Active
Drift prefers a verified action from the reviewed week, with specific
encouragement; older writing receives historical reminder wording. Neither an
omission nor a Not Conflict decision establishes alignment. Insufficient
Evidence omits the card.

The card preserves the exact quotation, source date and Journal Entry link.
Long quotations expand without rewriting the source. Inspect exposes the
record, source checks, model settings, usage and review outcome. All five
saved Personas support prepared per-week records; normal replay never calls a
provider. The replay uses Weekly Drift prompt v4 Run 1 and the full-history
runtime outcomes from the [completed targeted NSM update](../../logs/experiments/reports/north_star_v4_run1_20260907/report.md).
The five selected Personas cover 27 reviewed weeks: 23 selected quotations,
three ineligible outcomes, and one completed review with no supportive source.
Records preserve original model receipts and synthetic source availability;
they do not substitute AI evaluation grades for runtime decisions.

Manual onboarding calls `review_north_star` separately after the existing
closed-week response, keeping Weekly Drift Detection and valid Coach Digest
results available. It coalesces duplicate requests, reuses completed records,
shows bounded retry when permitted, and discards obsolete results after source
changes or deletion. Replies have separate server availability timestamps;
the current runtime excludes responses without an availability value. This
runtime behavior does not define the fresh experiment's synthetic-data policy.

Temporary token-counting timeouts, connection failures, and retryable HTTP
errors allow the same explicit retry. Invalid token receipts and over-limit
inputs remain terminal. Counting failures consume no generation attempt, and
retry reuses any earlier completed Core Value reviews.

The backend uses server-side `OPENAI_API_KEY`, Luna `low`, and the shared
[integration policy](../../config/evals/north_star_integration_v1.json): complete
inputs up to 16,000 tokens, no truncation, at most two attempts per model
request, US$0.25 per attempt and US$20 cumulative. These are retained runtime
settings, not an approved budget for the fresh experiment. Default live
provider receipts persist budget metadata. Full prompts, eligible source
text and raw responses remain in server session memory and browser-held
Inspect records; browser localStorage retains those records for resume.
Confirmed Delete session removes both copies. A storage-quota failure raises
the existing persistence warning; live history has no production storage
capacity guarantee. The reusable synthetic preparation code supports
source-disclosed raw outputs for reproduction. The saved replay links the
completed experiment's original receipts without making new provider calls.
The live runtime does not make a separate evaluation call.

Live NSM work is serialized in one worker thread. It uses the ignored
`logs/exports/demo_tool_runs/north_star/` directory and requires a finalized
integration budget. A missing or changed integration budget fails closed
before counting or provider work. The reset removed the old budget artifacts;
fresh budget setup remains pending, so live NSM generation is unavailable.
Saved replay does not require a live integration budget.

The [experiment methodology](../../docs/north_star/nsm_experiment_methodology.md)
records the completed comparison and targeted v4 Run 1 update. Its results are
AI assessments of synthetic histories, with human review deferred.

## Run locally

```sh
source .venv/bin/activate
uv run uvicorn src.demo.api:app --env-file .env --port 8000
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

The image includes the five current Personas' wrangled and synthetic source
files, plus the v4 definitions study's `requests.jsonl`, `responses.jsonl`,
`attempts.jsonl`, and `manifest.json`. The compact NSM replay records ship with
`src/`; the full experiment record is not needed in the deployed image. React
builds from the exported scenario bundles without importing the historical
August Coach Digest evaluation manifest.

The scenario exporter verifies saved Coach Digest provenance against the
current key-week Weekly Drift Detection output. The [September refresh](../../logs/experiments/reports/demo_v4_run1_20260907/report.md)
provides one accepted response for each of the five current key weeks, generated
with Luna at reasoning effort `none` and prompt `4.2`. All five passed Coach
Digest Validations; no new Coach Digest Evals or human review was performed.
Incompatible responses remain unavailable. The August Coach Digest evaluation
remains historical evidence. The browser requests
the scenario catalog and bundles with
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
