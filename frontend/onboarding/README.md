# Twinkl onboarding

Standalone React proof of concept for the Best-Worst Scaling onboarding flow.
It produces a confirmed, versioned Profile; `twinkl-1m8` owns durable
persistence and the Core Value handoff to the Weekly Drift Reviewer.

Each four-card set lets people tap or drag cards between a selection area and
Most or Least boxes, with keyboard controls and distinct illustrations for all
ten Schwartz values. The final action opens the first Journal Entry prompt and
exposes the confirmed Profile through an `onStartJournal` callback and a
`twinkl:start-first-journal` browser event.

## Run locally

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

This standalone POC stores unfinished progress and its confirmed Profile in
browser `localStorage` and makes no model or provider calls itself. The UI does
not present that implementation detail as a product privacy guarantee: deployed
Twinkl is expected to hand user data to LLM-backed services.
