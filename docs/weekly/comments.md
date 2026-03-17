The existing documentation has been reviewed, and there are a couple of things that needed to be changed related to RAG and semantic similarity, which would be overkill at POC scale.

## What was changed

In [`556ac6b`](https://github.com/DesmondChoy/twinkl/commit/556ac6b), the Coach docs were narrowed to the actual POC scope for this issue. The main change is that the docs now specify the initial weekly digest should use **full-context prompting** over the user's complete journal history, not a RAG pipeline.

In the current synthetic corpus, personas have a **median of 8 entries (max 12)** — about 1.1k words of initial-entry text per persona, or about 1.3k if nudge/response turns are included. Retrieval is a future scaling concern rather than a current requirement.

The same commit also:
- **Moves cosine-similarity identity drift to a future phase** — the simpler crash/rut threshold checks are sufficient for the POC.
- **Scopes initial evaluation to Tier 1 automated checks only** — meta-judge LLM eval (Tier 2) and human calibration (Tier 3) have been deferred.

**Important caveat:** This is a docs/spec refinement, not an implementation claim. The Coach digest is still not implemented, and the crash/rut trigger code is also still missing. Additionally, the current Critic quality (median QWK ~0.362) is [not yet strong enough for fully reliable automated drift alerts](https://github.com/DesmondChoy/twinkl/blob/556ac6b/docs/evals/drift_detection_eval.md#blocking-dependencies).

## Proposed first-pass placement of the weekly digest in the architecture

```
                         ONLINE (User Inference)

  Journal Entries ──► Text Encoder (SBERT) ──► Critic (MLP + MC Dropout)
                                                       │
                                          ┌────────────┴────────────┐
                                          │                         │
                                   Per-dimension             Per-dimension
                                   alignment (μ)            uncertainty (σ)
                                          │                         │
                                          └────────────┬────────────┘
                                                       │
                                                       ▼
                                              Drift Detection
                                              (threshold rules)
                                                       │
                                    ┌──────────┬───────┴───────┬──────────┐
                                    │          │               │          │
                                  crash       rut           stable    high-σ
                                    │          │               │          │
                                    └──────────┴───────┬───────┴──────────┘
                                                       │
                             ┌─────────────────────────┼──────────────────────────┐
                             │              WEEKLY DIGEST GENERATOR (NEW)          │
                             │              LLM + Full-Context Prompting           │
                             │                                                    │
                             │  Inputs:                                           │
                             │   • Response mode (from drift detection)           │
                             │   • Full journal history (all entries in context)  │
                             │   • Critic scores + uncertainty                   │
                             │   • User value profile (w_u)                      │
                             │   • Schwartz value elaborations                   │
                             │                                                    │
                             │  Output:                                           │
                             │   • Reflective narrative digest                    │
                             └────────────────────────┬───────────────────────────┘
                                                      │
                                                      ▼
                                           Persist (Pydantic → parquet)
                                                      │
                                                      ▼
                                            Tier 1 validation checks
```

## How your comments are addressed

| Your comment | What the docs now specify / what can be leveraged | Relevant repo files |
|---|---|---|
| **"Link it to intelligent reasoning"** / **"Go beyond rule engine"** | The digest is specified as a two-layer design: deterministic rules over Critic outputs choose the Coach mode, and the LLM decides *what* to say by reasoning over the full journal history plus Critic scores. The rules are the activation gate — the intelligence is intended to come from the LLM's synthesis. 4 distinct response modes are already spec'd out (crash, rut, sustained alignment, high uncertainty) and can be treated as acceptance tests. Note: the trigger code itself has not yet been implemented. | [`docs/vif/example.md`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/docs/vif/example.md) — **start here**, 4 concrete Coach responses with exact phrasing and anti-patterns; [`docs/vif/04_uncertainty_logic.md`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/docs/vif/04_uncertainty_logic.md) — dual-trigger formulas; [`docs/evals/drift_detection_eval.md`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/docs/evals/drift_detection_eval.md) — full drift spec with starting thresholds (crash δ=0.5, rut τ=−0.4, C_min=3 weeks, ε=0.3) |
| **"Data schema to save the weekly digests to find some sort of pattern to allow anomaly (drift) detection"** | A reasonable first implementation would be to persist digests as structured data (e.g. Pydantic schema → parquet), following the same pattern already used for judge labels. This would enable longitudinal querying and future anomaly detection over saved digests. | [`src/judge/consolidate.py`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/src/judge/consolidate.py#L132-L220) — existing parquet persistence pattern (Pydantic validation, row-building, parquet write) that could be followed for digest storage |
| **"Wire outputs, give it to LLM"** / **"Not just regurgitating — value-adds with historical context"** | The Critic's `predict_with_uncertainty()` interface already produces per-dimension alignment scores + uncertainty (σ), exposed across all three critic variants (MLP, ordinal, BNN). With full-context prompting, the LLM would receive *all* of a persona's entries in a single prompt, giving it the complete longitudinal picture. The digest is framed as comparing lived behavior against declared values, not summarizing what happened. | [`src/vif/critic.py`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/src/vif/critic.py), [`src/vif/critic_ordinal.py`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/src/vif/critic_ordinal.py), [`src/vif/critic_bnn.py`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/src/vif/critic_bnn.py) — `predict_with_uncertainty()` interface; [`docs/vif/02_system_architecture.md`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/docs/vif/02_system_architecture.md) — Critic vs Coach separation and full-context prompting spec; [`docs/prd.md`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/docs/prd.md) — *prospective accountability* vs *retrospective summarization* comparison table; [`config/schwartz_values.yaml`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/config/schwartz_values.yaml) — value elaborations to inject into the prompt |
| **"Reflective feedback!"** | **Now specified in docs:** The Coach response modes are spec'd to be reflective, not prescriptive — cite specific entry content, connect to the user's declared values, ask open-ended questions. Anti-patterns (no judgment, no advice, no gamification) are documented. **Existing code guidance:** For structured LLM calls and output validation, the nudge generation module is a reusable pattern. For metadata hygiene (stripping generation metadata before LLM processing), the nudge decision module demonstrates the approach. | [`docs/vif/example.md`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/docs/vif/example.md) — Coach tone and reflective feedback examples (strongest tone anchor); [`prompts/judge_alignment.yaml`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/prompts/judge_alignment.yaml) — structured output and rationale conventions; [`src/nudge/generation.py`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/src/nudge/generation.py) — structured LLM calls and validation; [`src/nudge/decision.py`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/src/nudge/decision.py) — anti-metadata-leakage pattern |

## Suggested next steps

1. **Define the digest prompt template** — Use [`docs/vif/example.md`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/docs/vif/example.md) as the tone anchor and [`config/schwartz_values.yaml`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/config/schwartz_values.yaml) for value context injection. Feed in: response mode + full journal history + Critic scores + user value profile.
2. **Define a digest Pydantic schema + parquet persistence** — Follow the [`src/judge/consolidate.py`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/src/judge/consolidate.py) pattern so digests are saved and queryable for longitudinal analysis.
3. **Run Tier 1 automated checks on digest output** — Groundedness, non-circularity, length. Code sketches are in [`docs/evals/explanation_quality_eval.md`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/docs/evals/explanation_quality_eval.md).

**Dependency (separate issue):** Crash/rut trigger implementation — formulas are spec'd in [`docs/evals/drift_detection_eval.md`](https://github.com/DesmondChoy/twinkl/blob/556ac6b/docs/evals/drift_detection_eval.md), but the code doesn't exist yet. The digest generator can be built and tested with a hardcoded response mode initially, with trigger wiring added once the detection code lands.

**Out of scope for first pass:** RAG, cosine-sim identity drift, and Tier 2/3 evaluation.