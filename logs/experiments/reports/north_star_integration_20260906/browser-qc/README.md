# North Star Moment browser checks — 6 September 2026

Actual Chromium browser checks used the React app at `http://127.0.0.1:5173/`.
Wide viewport: 1440 × 1050 CSS pixels. Narrow viewport: 390 × 844 CSS pixels.
All five saved Persona fixtures were loaded through the UI. Fresh manual
Profiles were created through all 11 Best-Worst Survey groups and confirmed
through the onboarding UI; no Profile was injected.

## Saved replay

All 36 closed weeks were opened and compared with their saved runtime records.
Every displayed quotation was exact, and every omitted card matched its record.
The corresponding automated projection tests check source/owner/Profile,
chronology, and absence of later-week events for every Persona/week.

| Persona | Reviewed weeks | Displayed cards | Treatments |
|---|---:|---:|---|
| Meera Krishnamurthy | 7 | 7 | 7 encouragement |
| Wei Jun Chen | 6 | 5 | 3 encouragement, 1 reminder, 1 reflection |
| Marc Vandenberghe | 8 | 7 | 3 encouragement, 3 reminders, 1 reflection |
| Noor Haddad | 6 | 4 | 4 encouragement |
| Lukas Vermeer | 9 | 7 | 6 encouragement, 1 reflection |

All five narrow layouts had a document width no greater than 390px. The
narrow replay uses its existing Journal Entries / Weekly Drift view switch
and scrollable result panel. Source navigation opened the original Journal
Entry drawer and focused its close button. Inspect exposed the selected
week's source availability, provider attempts, raw response, and validation.
Inspect navigation initially selected an older Coach Digest when the current
week had none; the fix now prioritizes that week's NSM event, then that week's
Coach Digest, Drift, and digest. Browser recheck confirmed the current NSM.

## Controlled onboarding

The backend was `scripts.demo_north_star_qc:app`, launched with:

```sh
source .venv/bin/activate
python -m uvicorn scripts.demo_north_star_qc:app --host 127.0.0.1 --port 8000
```

The harness uses the real Experience service, shared NSM runtime, contract
checks and frontend, with deterministic Weekly Drift Reviewer, nudge and
Coach responses and a controlled NSM provider. Its token counts and provider
responses are test doubles. These checks made zero paid calls and do not
measure model quality. No controlled text was sent to a real model.

The short supportive writing was:

> I helped my sister carry the groceries upstairs and stayed to cook dinner with her.

The complete 489-character expansion case is `LONG_QUOTE` in the harness.
The neutral omission text was: “I spent most of today moving between errands
and checking the weather. Nothing particular stands out yet.”

Verified through the UI:

- A closed week produced specific encouragement from the exact source,
  alongside No Active Drift and a valid Coach Digest.
- The source link focused the attributed Journal Entry; reload retained the
  result and source. Both wide and narrow layouts were inspected.
- A controlled provider failure retained the weekly result and Coach Digest;
  explicit retry restored exactly one card.
- Controlled no-supportive-writing output omitted the card. Empty history
  after removals showed no card. The neutral case also triggered the existing
  Coach validation failure because its deliberately fixed quotation was absent;
  Weekly Drift Detection remained available.
- A 489-character quotation expanded using Enter on the focused button.
  `aria-expanded` became true; all 489 characters were exact and the narrow
  document stayed within 390px. Saved selections were shorter than the 320
  character collapse threshold, so this control was exercised with the
  controlled manual source rather than altering saved evidence.
- Removing the selected completed source invalidated the affected card.
- Removing a source while NSM was blocked initially exposed a 409 trace
  mismatch: the server had an asynchronous pending event absent from the
  browser. The correction reconciles only server-owned NSM event identities,
  preserves exact writing-event/revision checks, and uses authoritative server
  events before mutation. Repeated browser removal succeeded; releasing the
  obsolete request returned a deliberate conflict and retained only the earlier
  valid card. Regression tests cover both pending and already-completed events.
- Deleting the session while NSM was blocked completed without waiting for
  the model. Releasing the old request and reloading left fresh onboarding,
  with no card or deleted writing restored.

The browser console contains the deliberately reproduced 409 failures and
expected obsolete-result conflicts. No JavaScript runtime exception or
horizontal overflow was observed. Automated tests additionally cover HTTP
disconnect cleanup, response availability, restored pending records, stale
hashes and forged-source rejection.

PNG files in this directory are browser captures. `onboarding-wide.png` and
`onboarding-narrow.png` show complete manual page layouts. Card captures are
viewport excerpts and may include the existing sticky header or scroll-panel
clipping; they are not claims that a long scrollable panel fits in one image.
The expansion capture shows keyboard focus and full expanded quotation text.
