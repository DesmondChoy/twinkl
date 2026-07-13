# Habit Recommendations Feature (Future Work)

**Status:** Parked for post-capstone development
**Created:** 2026-01-18
**Rationale:** Excellent product feature, but adds scope risk during capstone validation phase

---

## Vision

After N Journal Entries, a recommendation engine suggests specific habits or goals to improve value convergence.

### Example Flow
1. User journals for 2-4 weeks
2. The Drift Detector finds Drift: "Two Journal Entries in a row show choices against your Achievement Core Value"
3. The recommendation engine suggests:
   - "Set a weekly goal to tackle one uncomfortable task"
   - "Track completion in daily check-ins"
4. User accepts/modifies the habit
5. Twinkl monitors whether behavior shifts toward stated values

---

## Why This Strengthens the Product

### 1. **Closed-Loop Intelligence**
- **Current state:** Passive detection ("Twinkl found Drift")
- **With recommendations:** Active guidance ("here's how to converge")
- Shows the recommendation engine can **intervene**, not just observe

### 2. **Demonstrates End-to-End Thinking**
- Goes beyond ML model → shows understanding of complete user journey
- Addresses the "so what?" question: detection is diagnostic, recommendations are therapeutic

### 3. **Measurable Impact**
- Can track: Did users who followed recommendations show faster convergence?
- Provides concrete evaluation metric (vs. subjective "does this tension feel right?")

### 4. **Differentiation**
- Most journaling apps stop at reflection
- Twinkl becomes a **behavior-change product**, not just a mirror

---

## Why NOT to Build This for Capstone

### 1. **Scope Risk**
- VIF validation is still in progress
- Adding recommendations means:
  - Second ML problem (recommendation model)
  - Second validation challenge (are recommendations effective?)
  - Diluted focus on core contribution

### 2. **Evaluation Without Real Users**
- **Capstone grading** (20% on Technical Paper) rewards rigorous evaluation
- Synthetic personas can't tell you if "meditate 10min/day" actually helps
- Would need either:
  - Real user study (not feasible in capstone timeline)
  - Simulated evaluation (less convincing to assessors)

### 3. **Rubric Alignment**
- IS Capstone emphasizes:
  - "Substantial depth, Technical achievement" (20% Technical Paper)
  - "Complexity of problem" (20% System Implementation)
- Rubric doesn't reward feature breadth
- **Better strategy:** Exceptional VIF with thorough uncertainty quantification + excellent documentation

### 4. **Premature Optimization**
- Don't know yet what recommendations would be most effective
- Need real usage data to understand:
  - Which Core Values and contexts most often produce Drift?
  - Which interventions users actually follow?
  - How to avoid recommendation fatigue?

---

## Design Sketch (For Future Reference)

### Architecture

```
┌─────────────────┐
│ Journal Entries │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   VIF Critic    │  ← Core capstone work
│ + Drift Detector│
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ Recommendation Engine   │  ← Future work
│ - Habit library         │
│ - Personalization       │
│ - Difficulty calibration│
└────────┬────────────────┘
         │
         ▼
┌─────────────────┐
│  User accepts/  │
│  modifies habit │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Twinkl monitors │
│ convergence     │
└─────────────────┘
```

### Recommendation Types

**Direct Value Alignment**
- Drift detected in *Benevolence* → Recommend: "Schedule weekly call with family member"
- Drift in *Self-Direction* → Recommend: "Block 2hrs/week for creative exploration"

**Behavioral Scaffolding**
- "Start small: 5min daily instead of 30min weekly"
- "Track yes/no completion, not intensity"

**Nudge Integration**
- Recommendations could surface during conversational nudges:
- Twinkl: "I notice you've avoided creative projects lately. Would it help to set a recurring goal?"
  - User: "Yes, but I don't know where to start"
  - Twinkl: "How about: 'Every Saturday morning, spend 30min on [X]'?"

### Data Requirements

**Habit Library**
- Map each Schwartz value → 5-10 common habits
- Source from:
  - Behavior change literature
  - Self-help books (Atomic Habits, Tiny Habits)
  - User-generated (if we have real users)

**Personalization Features**
- User's declared constraints (time availability, energy levels)
- Historical compliance rate (don't recommend 6am meditation to night owls)
- Drift frequency and duration (a new Drift → gentle suggestion, repeated or longer Drift → more assertive)

**Success Metrics**
- Recommendation acceptance rate
- Completion rate over 2 weeks
- Post-recommendation VIF convergence scores

---

## Technical Challenges

### 1. **Recommendation Quality**
- How to avoid generic advice? ("Just do more X!")
- Need specificity: "Call Mom on Sunday 7pm" > "Be more benevolent"

### 2. **Overfitting to Schwartz Values**
- Recommendations could feel formulaic if too rigidly tied to value labels
- Should emerge from **behavioral patterns**, not value categories

### 3. **Temporal Dynamics**
- When does the two-consecutive-Conflict rule identify real Drift rather than normal fluctuation?
- Don't want to recommend habit changes for temporary life events

### 4. **User Agency**
- The recommendation engine shouldn't feel prescriptive ("You must do X")
- Frame as **collaborative exploration**: "Would this help?"

---

## Evaluation Approach (If Built)

### Synthetic Evaluation (Weak)
- Generate synthetic users with known value profiles
- Simulate Drift scenarios
- Check if recommendations align with psychology literature
- **Limitation:** Can't validate whether users would actually follow through

### Real User Study (Strong, but requires time)
- Recruit 20-30 users for 6-week study
- Randomize: Control (VIF only) vs. Treatment (VIF + recommendations)
- Measure:
  - Recommendation acceptance rate
  - Behavioral compliance (self-reported)
  - VIF convergence scores over time
- **Limitation:** Not feasible within capstone timeline

### Expert Review (Moderate)
- Show recommendations to psychologists/behavior change experts
- Ask: "Would this plausibly help someone with this Drift pattern?"
- **Limitation:** Proxy for real user validation

---

## Capstone Strategy

### Don't Build It, But **Document It**

In the Technical Paper "Future Work" section:

> **Habit Recommendation Engine**
> While the current implementation focuses on the VIF Critic and Drift Detector, a natural extension would be a recommendation engine that suggests specific behavioral interventions when Drift occurs. This would require:
>
> 1. **Habit Library Construction:** Mapping Schwartz values to evidence-based behavioral practices from behavior change literature (Fogg, 2020; Clear, 2018)
> 2. **Personalization:** Calibrating recommendations based on user constraints, historical compliance, and Drift frequency or duration
> 3. **Longitudinal Validation:** Real-world study measuring whether users who follow recommendations exhibit faster convergence than control groups
>
> Preliminary design sketches (Appendix X) suggest integrating recommendations into the conversational nudge flow, framed as collaborative exploration rather than prescriptive advice. This would position Twinkl as a **behavior-change product** rather than purely a reflective tool.

This shows:
- ✅ You've thought through the full product arc
- ✅ You understand the technical/evaluation challenges
- ✅ You made a deliberate scoping decision (depth over breadth)
- ✅ You can discuss it during Q&A to show systems thinking

---

## When to Build This

**Signals that it's time:**
1. ✅ VIF is validated and performing well
2. ✅ You have 10+ real users providing feedback
3. ✅ Users explicitly ask: "Now what? How do I respond to this Drift?"
4. ✅ You've observed which Core Values and contexts most often produce Drift

**Don't build until:**
- ❌ VIF evaluation is incomplete
- ❌ You're still tuning uncertainty thresholds
- ❌ You don't have real usage data

---

## References for Future Implementation

**Behavior Change Literature:**
- Fogg, BJ. *Tiny Habits* (2020) - START SMALL principle
- Clear, James. *Atomic Habits* (2018) - Habit stacking, environment design
- Duhigg, Charles. *The Power of Habit* (2012) - Cue-routine-reward loops

**Recommendation Systems:**
- Collaborative filtering for personalization
- Contextual bandits for adaptive recommendation (if you have user feedback)

**Schwartz Values → Behaviors:**
- Would need to map each value dimension to concrete practices
- Example: *Achievement* → goal-setting frameworks, progress tracking
- Example: *Universalism* → volunteer work, donation habits, environmental choices

---

## Final Note

This is a **great idea for the product**. It's just not the right **capstone strategy**. Nail the VIF, write an exceptional technical paper, and save this for the "what's next?" conversation during your defense.

The assessors will be more impressed by:
- Rigorous VIF evaluation
- Thoughtful uncertainty quantification
- Well-documented limitations
- Clear future roadmap (which includes this!)

...than by a half-validated VIF + untested recommendations.

---

**Next Steps (Post-Capstone):**
1. Interview 5-10 potential users: "If an app detected you drifting from your values, what would be helpful?"
2. Build habit library (start with 3-5 habits per Schwartz dimension)
3. Prototype recommendation UI in conversational nudge flow
4. Run small user study (N=10) to validate acceptance/compliance
5. Iterate based on feedback
