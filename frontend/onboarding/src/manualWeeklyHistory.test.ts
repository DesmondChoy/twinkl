import { describe, expect, it } from "vitest";
import { canonicalInspectFixture } from "./inspectFixture";
import { createExperienceState } from "./session";
import type { TraceEventContract } from "./demoContracts";
import { manualWeeklyHistory } from "./manualWeeklyHistory";

function week(prefix: string, start: string, end: string, active: boolean): TraceEventContract[] {
  const template = canonicalInspectFixture.trace_events[0];
  const narrative = { weekly_mirror: `${prefix} reflection`, tension_explanation: `${prefix} context`,
    reflective_question: `${prefix} question?` };
  const record = (suffix: string, type: string, parent: string | null, details: Record<string, unknown>) => ({
    ...template, event_id: `${prefix}-${suffix}`, event_type: type,
    parent_event_id: parent ? `${prefix}-${parent}` : null, details,
  });
  return [
    record("request", "weekly_review_requested", null, { request: { week_start: start, week_end: end } }),
    record("review", "weekly_review_completed", "request", { receipt: { week_start: start, week_end: end } }),
    record("drift", "drift_detected", "review", { decisions: [], result: {
      delivery_state: active ? "active_drift" : "no_active_drift",
      core_value_states: { benevolence: active ? "active_drift" : "no_active_drift" },
    } }),
    record("digest", "weekly_digest_built", "drift", { digest: {
      week_start: start, week_end: end, evidence: [], coach_narrative: null,
    } }),
    record("coach", "weekly_coach_generated", "digest", { narrative, validation: { checks: [] } }),
    record("moment", "north_star_reviewed", "coach", { record: { week_start: start, week_end: end } }),
  ];
}

const first = week("first", "2026-09-07", "2026-09-13", false);
const second = week("second", "2026-09-14", "2026-09-20", true);
const entries = ["2026-09-09", "2026-09-15"].map((date, index) => ({
  journal_entry_id: `entry-${index}`, t_index: index, date,
  content: "Fictional journal sample.", nudge_response: null,
}));
const initial = { ...createExperienceState(), journal_entries: entries };

describe("manual weekly history", () => {
  it("reconstructs two different results without mixing their Coach or moment events", () => {
    const result = manualWeeklyHistory({ ...initial, trace_events: [...first, ...second] });
    expect(result.map((row) => row.weekStart)).toEqual(["2026-09-07", "2026-09-14"]);
    expect(result[0].digest.coach_narrative).toMatchObject({ weekly_mirror: "first reflection" });
    expect(result[0].driftResult.delivery_state).toBe("no_active_drift");
    expect(result[0].events.map((event) => event.event_id)).toEqual(first.map((event) => event.event_id));
    expect(result[1].digest.coach_narrative).toMatchObject({ weekly_mirror: "second reflection" });
    expect(result[1].driftResult.delivery_state).toBe("active_drift");
  });

  it("keeps a later historical Coach retry in its original week", () => {
    const retry = { ...first[4], event_id: "first-retry", details: {
      ...first[4].details, narrative: { weekly_mirror: "retried first week" },
    } };
    const result = manualWeeklyHistory({ ...initial, trace_events: [...first, ...second, retry] });
    expect(result[0].digest.coach_narrative).toEqual({ weekly_mirror: "retried first week" });
    expect(result[0].inspectEventId).toBe("first-retry");
    expect(result[1].inspectEventId).toBe("second-coach");
  });

  it("does not recover an obsolete response after rebuilding the same week", () => {
    const rebuilt = week("rebuilt", "2026-09-07", "2026-09-13", true).slice(0, 5);
    rebuilt[4] = { ...rebuilt[4], status: "failed", details: { narrative: null, validation: null } };
    const result = manualWeeklyHistory({ ...initial, trace_events: [...first, ...second, ...rebuilt] });
    expect(result).toHaveLength(2);
    expect(result[0].digest.coach_narrative).toBeNull();
    expect(result[0].driftResult.delivery_state).toBe("active_drift");
    expect(result[0].events.some((event) => event.event_id === "first-moment")).toBe(false);
  });

  it("preserves an accepted new week while its trace is unavailable without borrowing the old Coach", () => {
    const result = manualWeeklyHistory({ ...initial, trace_events: first,
      weekly_digest: { week_start: "2026-09-14", week_end: "2026-09-20",
        coach_narrative: { weekly_mirror: "accepted new week" } },
      drift_result: { delivery_state: "active_drift" },
    });
    expect(result[1].digest.coach_narrative).toEqual({ weekly_mirror: "accepted new week" });
    expect(result[1].events).toEqual([]);
    expect(result[1].inspectEventId).toBeNull();
  });

  it.each([{ remaining: [entries[1]] }, { remaining: [entries[0]] }, { remaining: [] }])("omits obsolete results for weeks whose writing was removed: %j", ({ remaining }) => {
    const result = manualWeeklyHistory({ ...initial, journal_entries: remaining, trace_events: [...first, ...second] });
    expect(result).toHaveLength(remaining.length);
    for (const row of result) {
      expect(remaining.some((entry) => entry.date >= row.weekStart && entry.date <= row.weekEnd)).toBe(true);
    }
  });
});
