# Synthetic Voice Pipeline: Text-to-Speech for VIF Training Data

> **Status:** Exploratory idea for generating labeled audio training data using TTS.
> **Relationship to A3D:** This pipeline could generate initial training data for the [A3D module](A3D.md) before real user recordings are available.

## 1. Concept Overview

**Idea:** Inject emotion tags into synthetic journal entries, convert to speech via TTS (ElevenLabs v3), extract acoustic features, and use these as additional VIF training features.

**Pipeline:**
```
Synthetic Text + [emotion tags] → TTS (ElevenLabs v3) → Audio → Feature Extraction → VIF State Vector
```

**Why this matters:** The VIF architecture already reserves `phi_audio(A_u,t)` in the state vector, but we have no audio training data. Synthetic TTS with emotion tags creates labeled ground truth for bootstrapping audio feature extractors.

## 2. ElevenLabs v3 Audio Tags

ElevenLabs v3 supports bracketed emotion/delivery tags:

| Category | Tags |
|----------|------|
| Emotional | `[sad]`, `[angry]`, `[excited]`, `[nervous]`, `[calm]`, `[frustrated]` |
| Reactions | `[laughs]`, `[sigh]`, `[gasps]`, `[gulps]`, `[whispers]`, `[clears throat]` |
| Delivery | `[hesitates]`, `[stammers]`, `[pauses]`, `[rushed]`, `[slows down]` |
| Pacing | `[pause]`, `[short pause]`, `[long pause]` |

**Example:**
```
[sorrowful] I couldn't sleep that night. [sigh] It just kept replaying in my mind.
```

## 3. Proposed Integration

### 3.1 Generation Phase (Extend Existing Pipeline)

Modify `journal_entry.yaml` prompt to optionally output emotion annotations:

```yaml
# New output field
emotional_delivery:
  overall_tone: "nervous"
  moments:
    - position: "start"
      tag: "[hesitates]"
    - position: "mid"
      tag: "[sigh]"
```

These annotations are metadata (not shown to user), used only for TTS generation.

### 3.2 TTS Conversion

```python
# Pseudocode
def generate_audio(entry_text: str, emotion_tags: list[EmotionMoment]) -> bytes:
    tagged_text = inject_tags(entry_text, emotion_tags)
    return elevenlabs.generate(
        text=tagged_text,
        model="eleven_v3",
        voice="selected_voice_id"
    )
```

### 3.3 Feature Extraction

Extract prosodic features using OpenSMILE or Librosa:

| Feature | What It Captures |
|---------|------------------|
| Jitter | Pitch instability (stress marker) |
| Shimmer | Amplitude variation |
| F0 variance | Pitch range (emotional arousal) |
| Speaking rate | Words per second |
| Pause ratio | Hesitation patterns |
| HNR | Harmonics-to-noise ratio |

### 3.4 VIF Integration Options

**Option A: Direct concatenation**
```python
phi_audio = [jitter, shimmer, f0_var, speaking_rate, pause_ratio, hnr]
s_u,t = Concat[phi_text, phi_audio, ...]
```

**Option B: Text-audio dissonance score**
```python
# Train alignment model: does audio emotion match text sentiment?
dissonance = TextAudioAligner(text_emb, audio_emb)
# High score = cognitive dissonance signal
```

---

## 4. Pros and Cons

### Pros

| Benefit | Explanation |
|---------|-------------|
| **Labeled ground truth** | Emotion tags = supervision signal. We *know* the audio was generated with `[sad]`. |
| **Controlled variability** | Systematically vary emotion × content × speaker × accent combinations. |
| **Cold-start solution** | Bootstrap A3D training before real user recordings exist. |
| **Dissonance training data** | Deliberately mismatch (happy text + `[sad]` voice) to train cognitive dissonance detection. |
| **Cost-efficient iteration** | Test feature extraction pipelines without recruiting users. |

### Cons

| Drawback | Explanation |
|----------|-------------|
| **Synthetic ≠ real** | TTS prosody is "too clean" — real stress has subtler, messier patterns. |
| **Distribution shift** | Model trained on synthetic audio may not generalize to real voice notes. |
| **Emotion tag limitations** | ElevenLabs tags are coarse; real emotions are continuous and blended. |
| **Cost at scale** | ~$0.24/1000 chars × 1M chars = ~$240 for full synthetic corpus (manageable but non-zero). |
| **Voice diversity** | Limited to ElevenLabs voice library; may not cover Singlish/Singaporean accents. |

---

## 5. Pitfalls to Avoid

### Overengineering Traps

1. **Don't build a full audio pipeline before validating the hypothesis**
   - First test: Does synthetic TTS actually produce measurably different prosody for `[sad]` vs `[excited]`?
   - Run OpenSMILE on 10 samples before building automation.

2. **Don't train VIF on audio features prematurely**
   - The text-only VIF isn't trained yet. Adding audio complexity before text baseline works = debugging nightmare.
   - Sequence: Text VIF works → Add audio → Validate improvement.

3. **Don't over-invest in synthetic-only training**
   - Synthetic data is for bootstrapping, not production. Plan for real user data from day one.
   - Keep synthetic/real data separate in training logs for later ablation studies.

4. **Don't conflate emotion tags with Schwartz values**
   - `[sad]` ≠ misaligned. Someone might be sadly accepting a value-aligned sacrifice.
   - Audio features inform *arousal/valence*, not alignment. Let VIF learn the mapping.

5. **Don't build custom TTS integration when API calls suffice**
   - ElevenLabs API is straightforward. Resist urge to wrap it in abstractions.
   - A simple Python script is fine for v0.

### Domain-Specific Risks

| Risk | Mitigation |
|------|------------|
| Metadata leakage | Don't let VIF see emotion tags directly; only extracted features. |
| Accent mismatch | Test with Singaporean users early; synthetic American accent may mislead. |
| Privacy theater | Don't claim "voice analysis" if it's really just TTS playback. Be honest in docs. |

---

## 6. Potential Next Steps

### Phase 0: Feasibility Spike (1-2 days)

- [ ] Generate 10 synthetic entries with varied emotion tags
- [ ] Run through ElevenLabs v3 API (manual, no automation)
- [ ] Extract features with OpenSMILE/Librosa
- [ ] Verify: Do `[sad]` and `[excited]` produce measurably different F0/jitter?
- [ ] **Gate:** If features don't differentiate, reconsider approach

### Phase 1: Minimal Pipeline (if Phase 0 passes)

- [ ] Add `emotional_delivery` field to `journal_entry.yaml` output schema
- [ ] Write simple script: `generate_audio.py` that takes entry + tags → WAV file
- [ ] Write `extract_features.py` using eGeMAPSv02 feature set
- [ ] Output: Parquet with `entry_id, emotion_tag, [acoustic_features...]`

### Phase 2: Integration with VIF Training

- [ ] Extend state vector construction to include `phi_audio`
- [ ] Train VIF with and without audio features
- [ ] Measure: Does audio improve alignment prediction? (Ablation study)

### Phase 3: Real User Validation

- [ ] Collect 5-10 real voice note recordings (with consent)
- [ ] Compare feature distributions: synthetic vs real
- [ ] Fine-tune or recalibrate if distribution shift is severe

---

## 7. Cost Estimate

| Item | Estimate |
|------|----------|
| ElevenLabs API (1M chars) | ~$240 |
| Compute (feature extraction) | Negligible (CPU) |
| Storage (WAV files, 500 entries × 30s × 16kHz) | ~2GB |
| Engineering time (Phase 0-1) | 2-3 days |

---

## 8. Open Questions

1. **Which ElevenLabs voice best approximates Singaporean English?** Need to test options.
2. **Should emotion tags be generated by the LLM or rule-based?** LLM gives natural variation; rules give control.
3. **How to handle entries that are emotionally neutral?** Skip TTS? Use `[calm]` default?
4. **Is dissonance detection (text vs audio mismatch) more valuable than raw audio features?** Worth testing both.

---

## References

- [ElevenLabs v3 Audio Tags Documentation](https://elevenlabs.io/blog/v3-audiotags)
- [A3D Module Spec](A3D.md) — Adaptive Acoustic Anomaly Detector
- [VIF State Vector Spec](../VIF/VIF_05_State_and_Data_Pipeline.md) — `phi_audio` placeholder
- [OpenSMILE eGeMAPSv02](https://github.com/audeering/opensmile) — Feature extraction
