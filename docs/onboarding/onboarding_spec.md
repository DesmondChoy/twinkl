# Onboarding Specification: SVBWS Values Assessment

## 1. Purpose and Evidence Boundary

Onboarding gives Twinkl an initial Profile before the user has written any
Journal Entries. The Profile contains:

1. raw Schwartz Values Best-Worst Survey (SVBWS) responses and scores;
2. a separately named ten-value product transformation;
3. Core Value descriptions confirmed by the user; and
4. one structured goal category.

The assessment implements the Case 1, or object-case, SVBWS design published by
Lee, Soutar, and Louviere (2008). It is a **research-grounded pilot
instrument**, not a psychometrically validated Twinkl instrument. Twinkl has
not established test-retest reliability, measurement invariance, criterion
validity, or calibrated uncertainty for this implementation.

The flow therefore makes no ratio-scale, reliability, confidence, diagnostic,
or clinical claim. It also makes no claim that the final product weights are
BWS preference shares.

### Scope

- Onboarding owns the 11 SVBWS tasks, goal selection, Profile confirmation,
  local resume, and first Journal Entry handoff.
- `twinkl-1m8` owns durable Profile storage and the Core Value handoff to the
  Weekly Drift Reviewer.
- The first Journal Entry editor and later Profile evolution are outside this
  specification.

## 2. Published SVBWS Contract

### 2.1 Experimental design

The assessment contains 11 value objects. Universalism is split into
Universalism–Nature and Universalism–Social for the BWS task. The published
balanced incomplete block design contains 11 groups of six objects:

- every object appears six times;
- every pair of objects appears together three times; and
- each task requires one distinct Most and Least choice.

The React app randomizes both group order and card order once per resumable
session. Randomization changes presentation only; each stored response retains
its canonical group number and displayed card order.

### 2.2 Value objects and card text

Schwartz labels are internal metadata. Users see only the published descriptor
triplets.

| Internal object | User-facing descriptor |
|---|---|
| Power | Social power, authority, wealth |
| Achievement | Successful, capable, ambitious |
| Hedonism | Pleasure, enjoying life, self-indulgent |
| Stimulation | Daring, a varied life, an exciting life |
| Self-Direction | Creativity, curious, freedom |
| Universalism–Nature | Protecting the environment, a world of beauty, unity with nature |
| Benevolence | Helpful, honest, forgiving |
| Tradition | Devout, accepting portion in life, humble |
| Conformity | Politeness, honouring parents & elders, obedient |
| Security | Clean, national & family security, social order |
| Universalism–Social | Equality, world at peace, social justice |

The published descriptor text remains the scored stimulus. Twinkl's visual and
copy treatment is a separately documented presentation adaptation in Section
3; it does not alter these words.

### 2.3 Balanced groups

| Group | Six internal objects |
|---|---|
| 1 | Achievement; Universalism–Nature; Benevolence; Tradition; Security; Universalism–Social |
| 2 | Power; Hedonism; Benevolence; Tradition; Conformity; Universalism–Social |
| 3 | Power; Achievement; Stimulation; Tradition; Conformity; Security |
| 4 | Achievement; Hedonism; Self-Direction; Conformity; Security; Universalism–Social |
| 5 | Power; Hedonism; Stimulation; Universalism–Nature; Security; Universalism–Social |
| 6 | Power; Achievement; Stimulation; Self-Direction; Benevolence; Universalism–Social |
| 7 | Power; Achievement; Hedonism; Self-Direction; Universalism–Nature; Tradition |
| 8 | Achievement; Hedonism; Stimulation; Universalism–Nature; Benevolence; Conformity |
| 9 | Hedonism; Stimulation; Self-Direction; Benevolence; Tradition; Security |
| 10 | Stimulation; Self-Direction; Universalism–Nature; Tradition; Conformity; Universalism–Social |
| 11 | Power; Self-Direction; Universalism–Nature; Benevolence; Conformity; Security |

These invariants are executable tests, not documentation-only claims.

## 3. Twinkl Presentation Adaptation

### 3.1 Neutral visual variety

Each task shows six distinct abstract “memory atlas” backgrounds. They provide
warmth and visual variety without assigning an illustration to a value:

- background `01` through `06` maps to randomized display position, never a
  BWS object key or Schwartz label;
- a card keeps its background when it moves between the selection area, Most,
  and Least;
- an object can receive a different background when it returns in another task
  because each task has an independently randomized card order;
- all six backgrounds use the same dimensions, typography, accent, image
  treatment, readability veil, and approximately matched palette, luminance,
  contrast, and complexity; and
- the images contain no people, places, recognizable objects, value symbols,
  words, or numbers.

The backgrounds are decorative presentation data. They are not stored in the
Profile and do not affect scoring. Position-based assignment prevents a stable
image-to-value association; it does not establish that the images have no
priming effect. This limitation is one reason the instrument identifier names
the implementation as a UI adaptation.

Generation prompts and asset provenance are recorded in
[`frontend/onboarding/public/card-backgrounds/README.md`](../../frontend/onboarding/public/card-backgrounds/README.md).

### 3.2 Warm framing without rewritten stimuli

The scored descriptor triplets remain verbatim from the published SVBWS and
are protected by a focused test. Twinkl changes only value-neutral framing:

- the heading invites reflection without implying a preferred answer;
- the introduction says there are no right answers and that more than one
  principle can matter;
- instructions retain the published comparison frame: Most and Least
  important as a guiding principle in the user's life; and
- prompts use conversational language for choosing, placing, and reconsidering
  a card.

The first task states that the standalone prototype saves choices only in the
current browser so the user can resume. This describes the current storage
boundary; it is not a confidentiality, encryption, or deployed-product privacy
guarantee.

### 3.3 Project-report rationale

The presentation balances respondent care with experimental control. Free
paraphrasing could change the meaning of a scored object, while value-specific
illustration could add a second stimulus whose appeal affects Most or Least
choices. Twinkl therefore preserves the published words, warms only the shared
instructions, and randomizes neutral backgrounds independently of value
identity. The design reduces systematic visual association but remains a
Twinkl-specific presentation adaptation requiring empirical validation.

## 4. User Flow

### 4.1 Direct entry and progress

The user enters the first randomized group directly. Progress reads
`Values · n of 11`, followed by `Your focus` and `Your compass`. There is no
welcome screen and no midpoint result.

The heading reads `What matters most as you find your way?` The introduction
says there are no right answers and that more than one principle can matter.
The first instruction explains that the user will see 11 groups, that each
choice concerns what matters Most and Least as a guide for the user's life, and
that some cards return. Later groups use a shorter version of the same
instruction.

### 4.2 Card interaction

Six cards begin in the selection area. The user places one card in `Most` and a
different card in `Least`.

- Pointer and touch dragging work between the selection area and both boxes.
- Tapping a card activates explicit Most and Least placement targets.
- A placed card can return to the selection area or move to the other box.
- Moving a card into an occupied box replaces the prior choice.
- `M` and `L` place a focused card; Backspace, Delete, or Arrow Down returns it.
- `Continue` is disabled until both distinct choices exist.
- Keyboard focus is visible and reduced-motion preferences are respected.

The app does not show preliminary results between groups. This avoids making
later answers contingent on an inferred ranking displayed earlier.

### 4.3 Goal selection

After all 11 groups, the user chooses one goal category:

| Key | Display text |
|---|---|
| `work_life_balance` | I'm stretched too thin between work and everything else |
| `life_transition` | I'm going through a career or life transition |
| `relationships` | I want to be more present for people I care about |
| `health_wellbeing` | I'm neglecting my health or wellbeing |
| `direction` | I feel stuck or unclear about my direction |
| `meaningful_work` | I want to make more room for what matters to me |

The goal does not change BWS scoring.

### 4.4 Profile confirmation

The summary shows friendly descriptions for every ten-value score tied for
highest, plus the chosen goal. It does not reveal Schwartz labels or numerical
scores. Every tied description has equal visual weight.

`Set my compass` confirms the displayed descriptions as the user's Core Values
and emits the Profile. The user is not asked to rank, promote, or demote the
internal Schwartz categories.

### 4.5 First Journal Entry handoff

The completed flow opens the generic first Journal Entry prompt and exposes the
confirmed Profile through:

- the `onStartJournal` callback; and
- the `twinkl:start-first-journal` browser event.

The standalone React POC stores its resumable session and confirmed Profile in
browser `localStorage`. The UI describes that current storage boundary without
presenting it as a privacy guarantee.

## 5. Scoring and Product Transformation

### 5.1 Raw BWS results

For each of the 11 objects `o`:

```text
net_count(o) = best_count(o) - worst_count(o)
score(o) = net_count(o) / appearances(o)
```

In a complete response, `appearances(o) = 6`, so every object score is in
`[-1, +1]`. The Profile retains appearances, Most counts, Least counts, net
counts, and scores under `bws_results`.

This is the individual count score used in the published SVBWS work. It is a
descriptive relative-importance score. Twinkl does not attach a reliability or
uncertainty estimate to it.

### 5.2 Ten-value Profile transformation

Twinkl's other components use ten Schwartz dimensions, so `value_profile`
performs a separate product transformation:

```text
score(universalism) =
  (score(universalism_nature) + score(universalism_social)) / 2

score(v) = bws_score(v)  # for the other nine values
```

The transformation then creates positive weights:

```text
shifted(v) = score(v) - min(all ten scores) + 1
weight(v) = shifted(v) / sum(all shifted scores)
```

The method identifier is
`mean_universalism_facets_then_shift_normalize_v1`. The weights preserve score
order and sum to one, but they are product features—not BWS utilities,
probabilities, ratio-scale measurements, or preference shares. A two-to-one
weight ratio has no psychometric interpretation.

`top_values` contains every value tied for the highest transformed score in
canonical order. The confirmation screen turns those displayed descriptions
into Core Values. Exact ties are never broken or truncated.

### 5.3 No confidence field

The Profile has no onboarding confidence field. Selecting the same object as
Most in one group and Least in another can follow a stable global ordering, and
score spread measures differentiation rather than reliability. Neither is used
as a confidence proxy.

## 6. Versioned Profile Contract

The canonical implementation is
[`frontend/onboarding/src/domain.ts`](../../frontend/onboarding/src/domain.ts).
The abbreviated shape is:

```json
{
  "schema_version": 2,
  "onboarding_version": "2.1.0",
  "instrument": "svbws_lee_soutar_louviere_2008_ui_adaptation_v2",
  "scoring_method": "best_minus_worst_divided_by_appearances_v1",
  "user_id": "uuid",
  "session_id": "uuid",
  "started_at": "2026-07-20T05:00:00.000Z",
  "timestamp": "2026-07-20T05:04:00.000Z",
  "bws_responses": [
    {
      "set_number": 1,
      "items": [
        "achievement",
        "universalism_nature",
        "benevolence",
        "tradition",
        "security",
        "universalism_social"
      ],
      "item_order_shown": ["tradition", "achievement", "security", "benevolence", "universalism_social", "universalism_nature"],
      "selected_best": "benevolence",
      "selected_worst": "tradition",
      "response_time_ms": 9200
    }
  ],
  "bws_results": {
    "appearances": {},
    "best_counts": {},
    "worst_counts": {},
    "net_counts": {},
    "scores": {}
  },
  "value_profile": {
    "method": "mean_universalism_facets_then_shift_normalize_v1",
    "scores": {},
    "weights": {},
    "top_values": ["benevolence"],
    "bottom_values": ["power"]
  },
  "top_values": ["benevolence"],
  "goal_category": "work_life_balance",
  "user_confirmed": true,
  "provenance": {
    "source": "react_onboarding_poc",
    "set_order_randomized": true,
    "card_order_randomized": true
  }
}
```

A confirmed Profile requires all 11 canonical groups exactly once, six valid
objects per response, distinct valid Most and Least choices, non-negative
integer response time, a valid goal, and explicit summary confirmation.
Validation rebuilds the Profile deterministically and rejects any mismatch.

The resumable browser session uses schema version `4` and storage key
`twinkl.onboarding.session.v4`. It stores the randomized group order and each
canonical group's randomized card order.

## 7. Integration Status

- The React POC creates and validates the Profile locally.
- Durable Profile storage is not implemented.
- `top_values` is not yet supplied to the Weekly Drift Reviewer or Drift
  Detector.
- `value_profile.weights` is not yet supplied to the VIF Critic.
- `goal_category` does not yet focus the Weekly Coach.

The current synthetic personas retain their explicit `core_values` until
`twinkl-1m8` implements the Profile handoff.

## 8. References

- Lee, J. A., Soutar, G. N., & Louviere, J. J. (2008). [The best-worst scaling
  approach: An alternative to Schwartz's Values Survey](https://doi.org/10.1080/00223890802107925).
  *Journal of Personality Assessment, 90*(4), 335–347.
- Lee, J. A., Soutar, G. N., & Louviere, J. J. [Prepublication method paper with
  task wording and count scoring](https://anzam.org/wp-content/uploads/pdf-manager/2446_LEE_JULIE_ANNE_RM-04.PDF).
- Bardi, A., Lee, J. A., Hofmann-Towfigh, N., & Soutar, G. (2009). [The
  structure of intraindividual value change](https://repository.royalholloway.ac.uk/file/2ec96770-e8f0-33d2-184d-0eade5384604/10/VChStructure_LastWordVersion.pdf).
  *Journal of Personality and Social Psychology, 97*(5), 913–929. Appendices B
  and C reproduce the design and one task.
