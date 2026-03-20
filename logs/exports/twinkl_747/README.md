# twinkl-747 Audit Bundle

Status: completed on 2026-03-20.

This bundle was prepared by `scripts/journalling/twinkl_747_prepare_audit.py`.

## Files

- `sample_manifest.csv`: the deterministic 50-case audit sample.
- `prompts/<condition>.jsonl`: rendered judge prompts for `full_context`, `profile_only`, and `student_visible`.
- `results/<condition>_results.jsonl`: populated judge outputs for the completed audit run.
- `manual_review_workbook.csv`: answer sheet for human review.
- `manual_review_blind_packet.md`: text-only packet for the top 3 hard cases per hard dimension.
- `manual_review_reference.md`: rich-context reference for all sampled cases.
- `reachability_audit_report.md`: final audit summary and recommendation.
- `joined_results.csv`, `comparison_rows.csv`, `flip_summary.csv`: summarized outputs produced by the final report step.

## Result File Format

Each results file uses one JSON object per line:

```json
{"case_id":"security__013d8101__1","scores":{"self_direction":0,"stimulation":0,"hedonism":0,"achievement":0,"power":0,"security":1,"conformity":0,"tradition":0,"benevolence":1,"universalism":0},"rationales":{"security":"...","benevolence":"..."}}
```

Only `case_id` and `scores` are required by the summarizer. The `scores` object must contain all 10 Schwartz dimensions.

## Outcome

The completed audit recommends `change_distillation_target`, driven primarily by severe `security` unreachability and reproducibility drift in the sampled hard-dimension cases.
