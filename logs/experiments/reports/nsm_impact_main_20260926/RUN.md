# Running the pilot judge

Spawn one `nsm-impact-judge` subagent per task, each in a fresh context.
Pass no `model` override; the agent definition pins model and effort.
Give each subagent only these two paths:

- task: `logs/experiments/reports/nsm_impact_main_20260926/tasks/<task_id>.md`
- verdict: `logs/experiments/reports/nsm_impact_main_20260926/verdicts/<task_id>.json`

Never give a subagent the `sealed/` directory. Then run:

```sh
uv run python -m scripts.experiments.nsm_impact_pilot score --out logs/experiments/reports/nsm_impact_main_20260926
```

Tasks (144): task-01, task-02, task-03, task-04, task-05, task-06, task-07, task-08, task-09, task-10, task-100, task-101, task-102, task-103, task-104, task-105, task-106, task-107, task-108, task-109, task-11, task-110, task-111, task-112, task-113, task-114, task-115, task-116, task-117, task-118, task-119, task-12, task-120, task-121, task-122, task-123, task-124, task-125, task-126, task-127, task-128, task-129, task-13, task-130, task-131, task-132, task-133, task-134, task-135, task-136, task-137, task-138, task-139, task-14, task-140, task-141, task-142, task-143, task-144, task-15, task-16, task-17, task-18, task-19, task-20, task-21, task-22, task-23, task-24, task-25, task-26, task-27, task-28, task-29, task-30, task-31, task-32, task-33, task-34, task-35, task-36, task-37, task-38, task-39, task-40, task-41, task-42, task-43, task-44, task-45, task-46, task-47, task-48, task-49, task-50, task-51, task-52, task-53, task-54, task-55, task-56, task-57, task-58, task-59, task-60, task-61, task-62, task-63, task-64, task-65, task-66, task-67, task-68, task-69, task-70, task-71, task-72, task-73, task-74, task-75, task-76, task-77, task-78, task-79, task-80, task-81, task-82, task-83, task-84, task-85, task-86, task-87, task-88, task-89, task-90, task-91, task-92, task-93, task-94, task-95, task-96, task-97, task-98, task-99
