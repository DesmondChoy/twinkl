## TL;DR

We need to settle on **one v1 definition of drift** so that several P0 tasks can move.
For v1, drift means repeated conflict with a value someone said matters to them — not one isolated journal entry.

My recommendation (EDA below):

> **Drift = a sustained conflict episode on a declared core value.**
> Specifically: a value the user explicitly said matters to them receives **2 conflict labels in a row**.

**React or comment by Wed 15 Jul 2026.** No objections by then → Option 1 is adopted and `twinkl-wq9p` proceeds against it.

Blocked on this decision: `twinkl-wq9p` (drift-trigger benchmark), `twinkl-a30f`, `twinkl-j0ck`, `twinkl-749`.

## How to read the numbers

- **Label:** every journal entry gets a score per value — `-1` (conflicts with the value), `0` (neutral), `+1` (aligns).
- **Core value:** a value the persona declared as important during onboarding. 204 personas, 292 core-value declarations.
- **Consensus labels are the EDA reference, not ground truth or the final test set.** They are majority votes over five Judge passes — more stable than the original single-pass labels, but still LLM-judged rather than human-validated. The current model was trained on single-pass labels, so part of any future benchmark gap will be label-regime shift rather than model error.

Full analysis: [EDA report](https://github.com/DesmondChoy/twinkl/blob/c70d5c1493b291303dd90606bec6a36f80c6474f/docs/drift/trajectory_eda.md) · [script](https://github.com/DesmondChoy/twinkl/blob/c70d5c1493b291303dd90606bec6a36f80c6474f/scripts/drift/trajectory_eda.py) · [impact table](https://github.com/DesmondChoy/twinkl/blob/c70d5c1493b291303dd90606bec6a36f80c6474f/docs/drift/tables/single_definition_impact_comparison.csv)

## What the data allows

**This dataset only supports short-window definitions.**

![Dataset structure](https://raw.githubusercontent.com/DesmondChoy/twinkl/c70d5c1493b291303dd90606bec6a36f80c6474f/docs/drift/figures/fig1_structure.png)

The median persona has 8 entries over a 26-day span, with 5 active journaling weeks. That is not enough history to evaluate definitions requiring 3+ weeks of sustained decline. For v1, we can test short entry-level patterns; longer fade or evolution concepts would require new arc-scripted data.

**For this corpus, two consecutive conflicts is the practical threshold.**

![Threshold sensitivity — personas flagged per required consecutive conflicts](https://raw.githubusercontent.com/DesmondChoy/twinkl/codex/drift-eda-definition-fixes/docs/drift/figures/fig4_persona_cliff.png)

One conflict anywhere on a core value flags 102/204 personas (50%), which is too broad for a Coach trigger. Separately, among 135 dip transitions where later recovery is observable, 74% recover at the next entry and 84% within two entries. Requiring two consecutive conflicts reduces coverage to 40 personas (19.6%); three leaves 20 across the full corpus, and four leaves 5. C=2 is therefore the only practical organic v1 threshold — but it still needs a separate held-out evaluation set.

Two supporting facts from the EDA report (F5, F6), stated without figures:

- **Core-value gating keeps v1 focused.** Ungated, 46.7% of persona×value trajectories are entirely neutral; on declared core values, that falls to 1.0%. It also matches the product promise: surface conflict with values the user explicitly said matter, not incidental movement on every dimension.
- **The signal is uneven across values.** Sustained conflict on core values ranges from Power (32%) and Universalism (28%) down to Tradition (0%). Any global definition over-represents Power/Hedonism/Universalism personas. I propose accepting this for v1 and reporting results per-value rather than tuning per-value thresholds.

## The three candidate definitions

All three anchor to a declared core value; impact under consensus labels:

| # | Definition | Impact | Main blind spot |
|---|---|---:|---|
| **1 — Sustained conflict** (recommended) | 2 consecutive `-1` labels | **40/204 = 19.6%** | conflicts that recur but never back-to-back |
| 2 — Conflict week | ≥2 `-1` entries in one calendar week | 32/204 = 15.7% | back-to-back conflicts straddling a week boundary |
| 3 — Unrecovered departure | `0/+1 → -1 → -1` | 30/204 = 14.7% | personas whose observed entries *start* in conflict |

(Single-pass label impacts — 24.0% / 19.1% / 17.2% — are in the impact CSV; same ranking.)

**At the persona-coverage level, the options overlap almost completely:** their union is 41 personas (20.1%), barely more than Option 1 alone (40). The definitions still differ in which episode and week they label, so the choice matters mainly for timing and explanation — not headline population coverage.

Four verified personas show where the lenses disagree:

| Persona (core value) | Pattern | Opt 1 | Opt 2 | Opt 3 |
|---|---|:---:|:---:|:---:|
| Layla (parent)\* — Hedonism | fine → two conflicts in one week → recovers → one late blip | ✅ | ✅ | ✅ |
| Lukas (teacher) — Power | conflicts and wins strictly alternate; two conflicts share a week | ❌ | ✅ | ❌ |
| Layla (artist)\* — Universalism | aligned → conflicts Sun + Mon (straddling a week boundary) → repairs | ✅ | ❌ | ✅ |
| Nate (entrepreneur) — Universalism | first two observed entries already in conflict → strong recovery | ✅ | ❌ | ❌ |

\* Two distinct synthetic personas share the name Layla — a generator name collision, not a typo.

Why Option 1:

- Option 2's calendar bins are delivery artifacts, not behavior — Layla (artist)'s Sunday+Monday episode is invisible to it.
- Option 2 also overreacts to messy-but-mixed weeks — Lukas had a genuine win between his two conflicts.
- Option 3 adds a prior aligned/neutral requirement. That makes it the cleanest definition of movement away from a baseline, but it excludes people whose journals start in conflict — exactly the rocky first week where journaling often begins.
- Option 1's own blind spot (Lukas-style alternating conflict) is an acceptable v1 trade for a Coach that must not overreact.

## Recommendation

Adopt **Option 1: Sustained Conflict Episode** for v1 and ship the end-to-end path. Layer split:

- **Development reference** — two consecutive consensus conflict labels on a declared core value.
- **Runtime detector** — accumulated soft conflict confidence over recent entries, rather than two hard verdicts.
- **Final evaluation** — a separate held-out scripted or adjudicated episode set; the organic label-derived corpus is for EDA and tuning.
- **Delivery** — a weekly Coach digest citing the entries. An episode is detected when the first `-1, -1` pair completes; later evidence in that digest week classifies it as open, unresolved, or recovered. Recovered episodes (9/41 flagged trajectories, 22%) get recovery framing rather than a warning, so detected coverage is not the same as warning volume.

Options 2 and 3 remain secondary analysis slices (digest QA; departure-from-alignment analysis), not the v1 definition.

## Open questions for buy-in

1. Are we comfortable defining v1 as **2 consecutive conflicts** on a declared core value? One conflict flags half the corpus; three leaves only 20 positive personas before any validation/test split. Should the two entries also have to occur within a maximum elapsed window, such as 7 days?
2. Do we accept the **per-value skew** (Power/Hedonism/Universalism over-represented, Tradition contributing zero cases) for v1, handled via per-value reporting rather than per-value thresholds?
3. Do we agree that episodes which have **recovered by digest time** get recovery framing ("you pulled it back"), not a drift warning — with the benchmark still counting them as detected?
4. Do we agree that the organic consensus corpus is for development and tuning, while final performance is measured on a separately held-out scripted or adjudicated set?
