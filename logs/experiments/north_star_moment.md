# North Star Moment experiment log

**Latest NSM results to use:** the [targeted Weekly Drift v4 Run 1 update](reports/north_star_v4_run1_20260907/report.md), completed on 7 September 2026, is the current basis for NSM reporting and method comparisons. It supersedes the original v2 repeat-1 results for current reporting; the original record remains historical evidence. These results retain prior observations and remain AI assessments of synthetic histories.

The [original 501-week AI comparison](../../docs/north_star/nsm_experiment_methodology.md#completed-results)
is complete and retained in
[`north_star_20260906/nsm_experiment.json`](reports/north_star_20260906/nsm_experiment.json).
It used Weekly Drift v2 repeat 1 and found higher Card precision and Opportunity
recall for full eligible history than Nomic top-three retrieval in both
partitions. Human review remains deferred.

The [targeted v4 Run 1 update](reports/north_star_v4_run1_20260907/report.md)
is complete for both methods under `twinkl-fz34.14`. It preserves the original
105 Personas, 501 cases, 81/24 partition, and 20-case consistency sample. The
impact audit identified 38 affected cases across 22 Personas: 29 development
and nine final weeks. Both methods were reassessed on those contexts, with
individual receipts reused only when their complete request and experimental
identity matched. Observations for all 463 unchanged cases, including failures
and unresolved judgments, were retained; their recomputed grades are identical.

| Partition and metric | Nomic top-three retrieval + Luna-low | Full eligible history + Luna-low |
| --- | ---: | ---: |
| Development Card precision | 181/309 (58.58%) | 244/323 (75.54%) |
| Development Opportunity recall | 181/327 (55.35%) | 244/327 (74.62%) |
| Qualified final Card precision | 49/85 (57.65%) | 70/85 (82.35%) |
| Qualified final Opportunity recall | 49/86 (56.98%) | 70/86 (81.40%) |

Shared Luna-xhigh evaluation retained the paired exclusion rules: development
excludes 31 cases from Card precision and 35 from Opportunity recall; final
excludes ten and 11 respectively. Full eligible history remains higher on both
headline metrics in both partitions. The report contains the recomputed
Persona-bootstrap intervals and secondary diagnostics.

The update retained 2,952 request receipts and issued 190 new requests with
202 generation attempts. Six new requests ended unsuccessfully: two Nomic
runtime reviews and four shared reference reviews. Known incremental cost is
$0.90767, with five new attempts lacking usage receipts, so this is a lower
bound. The original experiment's known $10.84758 cost and five unknown-cost
attempts remain historical. Mixed retained/new runtime measurements are not
a fresh latency benchmark.

This is a targeted update with retained observations, not a wholly fresh
independent experiment. The original record, frozen methodology, receipts, and
correction history remain preserved. The separate report links the new
manifest, per-case differences, evidence lineage, complete failures and
denominators, and verification. The evidence is AI review of synthetic
histories; human review remains deferred. Saved replay integration uses the
full-history outcomes for five selected Personas and 27 reviewed weeks, with
original receipts and explicit no-card outcomes.
