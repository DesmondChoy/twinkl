# Running the pilot judge

Spawn one `nsm-impact-judge` subagent per task, each in a fresh context.
Pass no `model` override; the agent definition pins model and effort.
Give each subagent only these two paths:

- task: `logs/experiments/reports/nsm_impact_pilot_20260926/tasks/<task_id>.md`
- verdict: `logs/experiments/reports/nsm_impact_pilot_20260926/verdicts/<task_id>.json`

Never give a subagent the `sealed/` directory. Then run:

```sh
uv run python -m scripts.experiments.nsm_impact_pilot score --out logs/experiments/reports/nsm_impact_pilot_20260926
```

Tasks (44): task-01, task-02, task-03, task-04, task-05, task-06, task-07, task-08, task-09, task-10, task-11, task-12, task-13, task-14, task-15, task-16, task-17, task-18, task-19, task-20, task-21, task-22, task-23, task-24, task-25, task-26, task-27, task-28, task-29, task-30, task-31, task-32, task-33, task-34, task-35, task-36, task-37, task-38, task-39, task-40, task-41, task-42, task-43, task-44
