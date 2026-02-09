# Nudge Selection: Ranking-Based Evaluation

> ⚠️ **STATUS: NOT IMPLEMENTING**
>
> This evaluation plan has been **archived as future work**. The nudging feature is justified through ecological validity (industry alignment) rather than quantitative ablation.
>
> **Reasons:**
> 1. **Nudging is industry-standard** — 5 of 7 leading AI journaling apps (Rosebud, Reflection, Entries, Life Note, Mindsera) use conversational follow-ups. No validation needed to justify a standard practice.
> 2. **Academic focus is on VIF training** — The core contribution is the Value Identity Function, not nudge selection quality.
> 3. **Current system is single-shot** — Building a candidate ranking system requires new infrastructure that's out of scope for this capstone.
>
> See: [Nudging Feature: Design Rationale & Industry Alignment](../synthetic_data/nudge_validation_plan.md)

---

## Goal

Add academically rigorous evaluation for the nudge/question selection system using standard recommender metrics (NDCG@k, Hit@k) while keeping implementation minimal for a time-boxed capstone.

---

## Part 1: Baseline Comparison

Three baselines to benchmark against, ordered by implementation effort:

| Baseline | Description | Implementation | Academic Value |
|----------|-------------|----------------|----------------|
| **Random** | Randomly shuffle candidates | 2 lines | Required — proves system beats chance |
| **Embedding Similarity** | Rank by cosine similarity between entry embedding and nudge embedding | ~15 lines (SBERT already in repo) | Shows value-conditioning adds signal beyond semantic similarity |
| **Category-Only** | Always pick first nudge of the detected category (no personalization) | ~5 lines | Isolates contribution of user context |

**Recommendation:** Implement all three — they're trivial and strengthen the evaluation story.

---

## Part 2: Extending Current System for Candidate Ranking

### Current State

- Single-shot: one nudge generated per entry
- No candidate pool, no scoring, no ranking

### Minimal Extension (~80 lines total)

#### 2.1 New Prompt: `prompts/nudge_candidates.yaml`

```yaml
name: nudge_candidates
description: Generate multiple nudge candidates for ranking evaluation
template: |
  Given this journal entry, generate exactly {{ num_candidates }} different follow-up nudges.

  Entry: {{ entry_content }}
  Category: {{ nudge_category }}

  For each nudge, provide:
  - nudge_text: The actual nudge (5-12 words)
  - reasoning: Why this nudge might help (1 sentence)
  - confidence: Your confidence this is the best nudge (0.0-1.0)

  Make candidates diverse — vary the angle, specificity, and emotional tone.

  Return as JSON array.
```

#### 2.2 Schema Addition (in notebook)

```python
class NudgeCandidate(BaseModel):
    nudge_text: str
    reasoning: str
    confidence: float  # LLM self-reported confidence

class CandidatePool(BaseModel):
    candidates: list[NudgeCandidate]
    ground_truth_ranking: list[int] | None  # For evaluation
```

#### 2.3 Generation Function

```python
async def generate_nudge_candidates(
    entry: JournalEntry,
    category: NudgeCategory,
    num_candidates: int = 5,
) -> list[NudgeCandidate]:
    # Single LLM call returning N candidates
    # More efficient than N separate calls
```

#### 2.4 Config Addition (`config/synthetic_data.yaml`)

```yaml
nudge:
  # ... existing config ...
  evaluation:
    num_candidates: 5
    generate_ground_truth: true
```

---

## Part 3: Evaluation Pipeline

### 3.1 Ground Truth Labeling

**Approach: LLM-as-Judge**

- Use GPT-4 to rate each candidate's relevance given entry + persona context
- 3-point scale: highly relevant (2), okay (1), irrelevant (0)
- ~10 lines of code; reuses existing Judge pattern
- **Limitation to note in report**: No human validation; acknowledge potential LLM bias

### 3.2 Metrics Implementation (~30 lines)

```python
def hit_at_k(ranked_candidates: list, relevant_set: set, k: int) -> float:
    """1 if any relevant item in top-k, else 0"""
    return float(any(c in relevant_set for c in ranked_candidates[:k]))

def ndcg_at_k(ranked_candidates: list, relevance_scores: dict, k: int) -> float:
    """Normalized Discounted Cumulative Gain"""
    dcg = sum(relevance_scores.get(c, 0) / log2(i + 2)
              for i, c in enumerate(ranked_candidates[:k]))
    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(r / log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0
```

### 3.3 Evaluation Script Structure

```python
# For each synthetic persona:
for persona in personas:
    for entry in persona.entries:
        # 1. Generate candidate pool
        candidates = await generate_nudge_candidates(entry, category, n=5)

        # 2. Get ground truth relevance (LLM-as-Judge)
        relevance = await judge_nudge_relevance(entry, candidates, persona)

        # 3. Rank using each method
        rankings = {
            "random": random_rank(candidates),
            "embedding": embedding_rank(entry, candidates),
            "category_only": category_rank(candidates, category),
            "twinkl": twinkl_rank(entry, candidates, persona),  # Your system
        }

        # 4. Compute metrics
        for method, ranking in rankings.items():
            results[method]["hit@1"].append(hit_at_k(ranking, relevant, 1))
            results[method]["hit@3"].append(hit_at_k(ranking, relevant, 3))
            results[method]["ndcg@3"].append(ndcg_at_k(ranking, relevance, 3))

# 5. Report mean +/- std for each metric/method
```

---

## Part 4: Implementation Roadmap

### Phase 1: Infrastructure (Day 1)

- [ ] Create `prompts/nudge_candidates.yaml`
- [ ] Add `NudgeCandidate` schema to notebook
- [ ] Implement `generate_nudge_candidates()` function
- [ ] Test on 2-3 entries to verify JSON output

### Phase 2: Baselines (Day 1-2)

- [ ] Implement `random_rank()` — shuffle candidates
- [ ] Implement `embedding_rank()` — SBERT cosine similarity
- [ ] Implement `category_rank()` — category match only

### Phase 3: Ground Truth (Day 2)

- [ ] Create `prompts/nudge_relevance_judge.yaml` for LLM-as-Judge
- [ ] Implement `judge_nudge_relevance()` function
- [ ] Run on ~200 entries × 5 candidates = 1000 candidate evaluations

### Phase 4: Metrics & Reporting (Day 2-3)

- [ ] Implement `hit_at_k()` and `ndcg_at_k()`
- [ ] Create evaluation notebook: `notebooks/nudge_ranking_eval.ipynb`
- [ ] Generate results table comparing all methods
- [ ] Add statistical significance tests (paired t-test or bootstrap)

### Phase 5: Documentation (Day 3)

- [ ] Add evaluation results to `docs/evals/`
- [ ] Update PRD with new metrics

---

## Expected Output

A results table like:

| Method | Hit@1 | Hit@3 | NDCG@3 |
|--------|-------|-------|--------|
| Random | 0.20 | 0.52 | 0.41 |
| Embedding Similarity | 0.35 | 0.68 | 0.55 |
| Category-Only | 0.42 | 0.71 | 0.58 |
| **Twinkl (value-aware)** | **0.58** | **0.84** | **0.72** |

This demonstrates the system's value-conditioning provides meaningful lift over baselines.

---

## Files to Create/Modify

| File | Action | Lines |
|------|--------|-------|
| `prompts/nudge_candidates.yaml` | Create | ~25 |
| `prompts/nudge_relevance_judge.yaml` | Create | ~30 |
| `prompts/__init__.py` | Modify | +2 |
| `notebooks/nudge_ranking_eval.ipynb` | Create | ~200 |
| `config/synthetic_data.yaml` | Modify | +5 |

**Total new code: ~260 lines** (achievable in 2-3 focused days)

---

## Cost Estimate (200 entries × 5 candidates)

| Step | LLM Calls | Estimated Cost (GPT-4o-mini) |
|------|-----------|------------------------------|
| Candidate generation | 200 | ~$0.50 |
| Relevance judging | 1000 | ~$1.50 |
| **Total** | 1200 | **~$2.00** |

*Using GPT-4o for judging would be ~$15 total if higher quality needed.*

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM-as-Judge bias | Note as limitation in report; use consistent prompting |
| Low candidate diversity | Add diversity prompt instructions; filter near-duplicates |
| Metric gaming | Use multiple metrics (Hit + NDCG); report confidence intervals |
| Time overrun | Phase 1-2 are MVP; Phases 3-5 can be scoped down if needed |

---

## Academic Justification

This approach is defensible because:

1. **Reproducible** — Synthetic data + deterministic metrics
2. **Comparable** — NDCG/Hit@k are standard in RecSys literature
3. **Ablative** — Baselines isolate contribution of each component
4. **Scalable** — Can run on 200+ entries without real users
