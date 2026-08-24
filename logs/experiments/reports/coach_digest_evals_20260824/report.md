# Coach Digest Evals Report

**Source:** AI evaluation scores, NOT human validation. Future human calibration of the AI review remains separate work.

- Evaluator model: `gpt-5.6-luna`
- Evaluator reasoning effort: `none`
- Same-model-review limitation: Luna-none generated and evaluated the responses. Correlated errors can make this AI review too favorable.
- Scored: 5
- Failed (no valid verdict): 0
- Flagged for human review (any dimension < 3): 0
- Reflective question open & relevant: 100%
- Paid API calls recorded: 5
- Input tokens: 6921
- Cached input tokens: 0
- Output tokens: 498
- Calculated published-rate cost: `$0.00227865`
- Total request latency: 11.042s
- Mean / median / maximum request latency: 2.208s / 2.213s / 2.429s
- Cost basis: response token usage and published standard-tier Luna rates ([source](https://developers.openai.com/api/docs/models/gpt-5.6-luna)); not a billing export

| Dimension | Mean | Target | Meets | % ≥ 4 |
| --- | --- | --- | --- | --- |
| correctness | 4.80 | ≥ 3.5 | ✅ | 100% |
| specificity | 5.00 | ≥ 3.5 | ✅ | 100% |
| non_prescriptive_tone | 5.00 | ≥ 3.5 | ✅ | 100% |
| tension_honesty | 4.60 | ≥ 3.5 | ✅ | 80% |

## Per-Response Scores

| Response | Correctness | Specificity | Non-prescriptive tone | Tension honesty | Question | Review flag |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 11de77e8:2025-10-19 | 5 | 5 | 5 | 5 | pass | no |
| 23d101f8:2025-09-21 | 4 | 5 | 5 | 3 | pass | no |
| 8f83c818:2025-07-06 | 5 | 5 | 5 | 5 | pass | no |
| 988d1a65:2025-03-23 | 5 | 5 | 5 | 5 | pass | no |
| 02fb94f3:2025-04-20 | 5 | 5 | 5 | 5 | pass | no |

## Evaluator Justifications

### 11de77e8:2025-10-19

The response accurately cites the tech lead offer, the quoted acceptance rationale, and others’ reactions, while correctly preserving the insufficient-evidence limit and asking an open, relevant question without prescribing action.

### 23d101f8:2025-09-21

The response accurately cites the 2025-09-19 teaching results and the 2025-09-12 job-posting evidence, remains reflective, and explicitly says no current tension is confirmed, but its claim that an unfinished question or contrast remains present risks implying a current tension despite both values being classified no_active_drift.

### 8f83c818:2025-07-06

The response accurately reflects the active Universalism drift, cites the July 1 and July 5 entries while appropriately connecting the June 25 “I said okay,” and offers a specific, non-directive question without claiming unsupported events or resolution.

### 988d1a65:2025-03-23

It accurately cites the client-call correction, sharper delivery, client noticing, and quiet Sunday details; appropriately describes active drift as ended with the documented not_conflict reason without claiming success, and asks an open, relevant question.

### 02fb94f3:2025-04-20

The response accurately quotes the journal entries, reflects the documented Self Direction and Tradition context, explicitly acknowledges the limited evidence under more_reflection_needed without claiming Drift, and asks an open, relevant question without prescribing action.

## Per-Call API Metrics

| Call | Input | Cached | Output | Latency | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| coach_eval:11de77e8:2025-10-19 | 1801 | 0 | 90 | 2.213s | $0.00055810 |
| coach_eval:23d101f8:2025-09-21 | 1711 | 0 | 118 | 1.987s | $0.00056920 |
| coach_eval:8f83c818:2025-07-06 | 972 | 0 | 100 | 2.429s | $0.00031440 |
| coach_eval:988d1a65:2025-03-23 | 1337 | 0 | 96 | 2.091s | $0.00044930 |
| coach_eval:02fb94f3:2025-04-20 | 1100 | 0 | 94 | 2.322s | $0.00038765 |
