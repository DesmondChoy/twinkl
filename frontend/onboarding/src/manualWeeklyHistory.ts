import type { TraceEventContract, WeeklyDriftReviewerDecisionContract } from "./demoContracts";
import type { ExperienceState } from "./session";
import { weeklyRunContext } from "./weeklyRun";

type JsonObject = Record<string, unknown>;

export interface ManualWeeklyResult {
  weekStart: string;
  weekEnd: string;
  digest: JsonObject;
  driftResult: JsonObject;
  decisions: WeeklyDriftReviewerDecisionContract[];
  events: TraceEventContract[];
  inspectEventId: string | null;
}

function object(value: unknown): JsonObject | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject : null;
}

function resultFor(
  digest: JsonObject,
  driftResult: JsonObject,
  decisions: WeeklyDriftReviewerDecisionContract[],
  events: TraceEventContract[],
): ManualWeeklyResult | null {
  if (typeof digest.week_start !== "string" || typeof digest.week_end !== "string") return null;
  const coach = [...events].reverse().find((event) => event.event_type === "weekly_coach_generated");
  const available = coach && ["complete", "reused"].includes(coach.status)
    && coach.validation?.valid !== false;
  return {
    weekStart: digest.week_start,
    weekEnd: digest.week_end,
    digest: coach ? {
      ...digest,
      coach_narrative: available ? object(coach.details.narrative) : null,
      validation: available ? object(coach.details.validation) : null,
    } : digest,
    driftResult, decisions, events,
    inspectEventId: coach?.event_id
      ?? [...events].reverse().find((event) => event.event_type === "drift_detected")?.event_id
      ?? [...events].reverse().find((event) => event.event_type === "weekly_digest_built")?.event_id
      ?? null,
  };
}

/** Reconstruct reviewed weeks from their retained run records; never run a model. */
export function manualWeeklyHistory(experience: ExperienceState): ManualWeeklyResult[] {
  const weeks = new Map<string, ManualWeeklyResult>();
  const hasCurrentWriting = (digest: JsonObject) => experience.journal_entries.some((entry) =>
    typeof digest.week_start === "string" && typeof digest.week_end === "string"
    && entry.date >= digest.week_start && entry.date <= digest.week_end);
  const byId = new Map(experience.trace_events.map((event) => [event.event_id, event]));
  for (const event of experience.trace_events) {
    if (event.event_type !== "weekly_digest_built") continue;
    const digest = object(event.details.digest);
    const run = weeklyRunContext(experience.trace_events, event.event_id);
    const drift = byId.get(event.parent_event_id ?? "")
      ?? [...(run?.events ?? [])].reverse().find((item) => item.event_type === "drift_detected");
    const driftResult = drift?.event_type === "drift_detected" ? object(drift.details.result) : null;
    if (!digest || !driftResult || !run || !hasCurrentWriting(digest)) continue;
    const decisions = Array.isArray(drift?.details.decisions)
      ? drift.details.decisions as WeeklyDriftReviewerDecisionContract[] : [];
    const result = resultFor(digest, driftResult, decisions, run.events);
    // A rebuilt week supersedes all earlier versions, including their Coach response.
    if (result) weeks.set(result.weekStart, result);
  }
  // The session result is also available while its trace is loading or on older clients.
  if (experience.weekly_digest && experience.drift_result && hasCurrentWriting(experience.weekly_digest)) {
    const weekStart = experience.weekly_digest.week_start;
    if (typeof weekStart === "string" && !weeks.has(weekStart)) {
      const events = weeks.size === 0 ? experience.trace_events : [];
      const result = resultFor(experience.weekly_digest, experience.drift_result,
        experience.weekly_reviewer_decisions, events);
      if (result) weeks.set(weekStart, result);
    }
  }
  return [...weeks.values()].sort((left, right) => left.weekStart.localeCompare(right.weekStart));
}
