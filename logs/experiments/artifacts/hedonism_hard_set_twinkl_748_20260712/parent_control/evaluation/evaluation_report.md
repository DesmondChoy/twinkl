# twinkl-748 Hedonism hard-set evaluation

> Codex-reviewed diagnostic evidence; not human validation or a promotion surface.

Frozen entries: 40 across 20 pairs.
MC samples per checkpoint: 50.

## Family summary

| Family | Exact accuracy | Recall -1 | Recall +1 | Strict pair | Directional | Mean margin | QWK (secondary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent | 0.525 [0.500, 0.525] | 0.050 [0.000, 0.050] | 1.000 [1.000, 1.000] | 0.050 [0.000, 0.050] | 0.650 [0.650, 0.650] | 0.080 [0.047, 0.086] | 0.050 [0.050, 0.050] |
| tail_sensitive_reference | 0.575 [0.550, 0.575] | 0.200 [0.200, 0.250] | 0.950 [0.850, 0.950] | 0.150 [0.100, 0.150] | 0.750 [0.600, 0.800] | 0.196 [0.163, 0.225] | 0.150 [0.127, 0.150] |

Directional accuracy uses `P(+1) - P(-1)` and succeeds only when the
positive member scores above its matched negative member. QWK is secondary
because this is a small, deliberately balanced diagnostic set.

## Per-run summary

| Run | Family | Exact | Recall -1 | Recall +1 | Strict pair | Directional | High-confidence errors |
|---|---|---:|---:|---:|---:|---:|---:|
| run_019 | incumbent | 0.525 | 0.050 | 1.000 | 0.050 | 0.650 | 19 |
| run_020 | incumbent | 0.525 | 0.050 | 1.000 | 0.050 | 0.650 | 18 |
| run_021 | incumbent | 0.500 | 0.000 | 1.000 | 0.000 | 0.650 | 18 |
| run_034 | tail_sensitive_reference | 0.550 | 0.250 | 0.850 | 0.100 | 0.800 | 10 |
| run_035 | tail_sensitive_reference | 0.575 | 0.200 | 0.950 | 0.150 | 0.600 | 7 |
| run_036 | tail_sensitive_reference | 0.575 | 0.200 | 0.950 | 0.150 | 0.750 | 5 |
