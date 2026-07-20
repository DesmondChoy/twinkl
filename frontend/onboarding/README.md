# Twinkl onboarding

Standalone React proof of concept for the published Schwartz Values Best-Worst
Survey (SVBWS) onboarding flow. This is a research-grounded pilot instrument,
not a psychometrically validated Twinkl instrument. It produces a confirmed,
versioned Profile; `twinkl-1m8` owns durable persistence and the Core Value
handoff to the Weekly Drift Reviewer.

The assessment contains 11 randomized groups of six neutral cards from the
published balanced design. Each group uses six abstract backgrounds assigned
by randomized display position rather than value identity. People can tap,
drag, or use the keyboard to make Most and Least choices. Schwartz labels
remain internal. The Profile keeps raw
11-object BWS results separate from the ten-value product transformation, with
no midpoint result or confidence proxy. The final action opens the first
Journal Entry prompt and exposes the confirmed Profile through an
`onStartJournal` callback and a `twinkl:start-first-journal` browser event.

[`docs/onboarding/onboarding_spec.md`](../../docs/onboarding/onboarding_spec.md)
is the canonical workflow and evidence-boundary documentation. Background
generation provenance is in
[`public/card-backgrounds/README.md`](public/card-backgrounds/README.md).

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
