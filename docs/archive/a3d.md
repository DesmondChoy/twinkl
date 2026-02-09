# ISS-Practice-Module: The Adaptive Acoustic Anomaly Detector (A3D)

> **Note:** This module is a **Semester 3 Project** (ISS-Practice-Module) within the same NUS Master of Technology in Intelligent Systems program. Once successfully implemented and validated, it will be fed back into the main Capstone project as the core "Intelligent Sensing" component. For program structure and module relationships, see [IS_Capstone_Slides.pdf](IS_Capstone_Slides.pdf). For the main project overview, see [PRD.md](PRD.md).

## 1. Component Overview & Pain Point
*   **Component:** A privacy-first, unsupervised anomaly detection system that learns a user's unique prosodic baseline (pitch, rhythm, jitter) to flag physiological deviations (stress, lethargy) without relying on "Universal Emotion" models that fail on Singlish accents.
*   **Pain Point Solved:** Users often self-deceive in text ("I'm fine" typed out looks the same whether calm or crying). Standard "Emotion AI" is brittle, failing on non-US accents and ignoring individual baselines.
*   **Core Innovation:** Shifting from **Supervised Classification** (User vs. Database) to **Unsupervised Anomaly Detection** (User vs. Self) with an **Active Learning Loop** (Human-in-the-loop ground truth).

## 2. Technology Stack & Architecture
*   **Input:** High-fidelity audio (Raw WAV/PCM preferred to avoid codec artifacts that destroy jitter/shimmer features).
*   **Sensing Layer (Signal Processing):**
    *   **Library:** `OpenSMILE` (C++) or `Librosa` (Python).
    *   **Feature Extraction:** Focus on physiological signals: **Jitter/Shimmer** (micro-tremors), **F0 Variance** (pitch range), and **Speaking Rate**.
    *   **Pre-processing:** **SNR (Signal-to-Noise Ratio) Gate**: Rejects samples with high background noise (wind, traffic) to prevent environmental noise from being misclassified as physiological stress.
*   **Pattern Recognition Layer:**
    *   **Algorithm:** **One-Class SVM** or **Isolation Forest**.
    *   **Training:** Starts with a "Generic Singaporean Baseline" (Hybrid Start) to overcome the cold-start problem, then weights the model towards the user's specific voice via feedback.

## 3. Data Strategy (Generation & Usage)
Since labeled "Singaporean Stress" datasets don't exist, we generate "Small Data":
*   **Dataset A (The Golden Baseline):** User records 20–30 natural entries (mix of calm/excited) to train the initial self-model.
*   **Dataset B (The Stress Test):** User performs **Physiological Induction** (e.g., burpees, breath holding) to generate genuine "High Arousal" samples for validation.
*   **Dataset C (The Imposter):** Recordings from other speakers to test system specificity.
*   **Dataset D (Reference):** RAVDESS/IEMOCAP used *only* for feature importance analysis (validating that Jitter correlates with arousal), not for training the live model.

## 4. Evaluation Strategy (Academic Verification)
*   **Metric 1: False Positive Rate (FPR) Decay:** Measuring the "Learning Curve." The % of normal entries incorrectly flagged as anomalies should drop asymptotically as the user provides feedback.
*   **Metric 2: Physiological Sensitivity (Recall):** Ability to flag the **Dataset B** (Physiologically Induced Stress) as anomalies. This validates the sensor detects biology, not just acting.
*   **Metric 3: Imposter Rejection (Specificity):** Feeding **Dataset C** into the model should trigger high anomaly scores, proving the model has learned a *personalized* identity.

## 5. Link to Capstone (Twinkl)
This module transforms Twinkl from a text-based journal to a **Multimodal Reasoning System**.
*   **The "Prosodic Truth" Signal:** The A3D outputs a vector (`Deviation_Score`, `Confidence`, `Signal_Driver`) to the central LLM.
*   **Reasoning Logic:** `IF Text_Sentiment == Positive AND Audio_Deviation == High THEN Trigger_Cognitive_Dissonance_Probe`.
*   **User Value:** It allows the system to gently challenge users ("You said you're excited, but your voice sounds heavy—are you forcing it?") based on biological data, not just semantic analysis.
