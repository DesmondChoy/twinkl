import { describe, expect, it } from "vitest";
import { BWS_SETS, createProfile, type BwsResponse } from "./domain";
import {
  clearChoice,
  createSession,
  inspectRun,
  parseSession,
  setChoice,
  showView,
} from "./session";

describe("onboarding session", () => {
  const completeResponses = (): BwsResponse[] =>
    BWS_SETS.map((set) => ({
      set_number: set.setNumber,
      items: [...set.items],
      item_order_shown: [...set.items],
      selected_best: set.items[0],
      selected_worst: set.items[1],
      response_time_ms: 1_000,
    }));

  it("randomizes set order and every prescribed card order once, then round-trips", () => {
    const ids = ["user-1", "session-1"];
    const session = createSession(() => 0, new Date("2026-07-19T00:00:00.000Z"), () => ids.shift()!);
    expect(session.schema_version).toBe(7);
    expect(session.stage).toBe("set");
    expect(session.experience).toMatchObject({
      active_view: "experience",
      journal_started: false,
      selected_event_id: null,
      run_state: "idle",
    });
    expect(session.set_order).toHaveLength(11);
    expect(new Set(session.set_order)).toEqual(new Set(BWS_SETS.map((_, index) => index)));
    expect(session.displayed_orders).toHaveLength(11);
    session.displayed_orders.forEach((order, index) => {
      expect(new Set(order)).toEqual(new Set(BWS_SETS[index].items));
    });
    expect(parseSession(JSON.stringify(session))).toEqual(session);
  });

  it("sets Most and Least explicitly and keeps them distinct", () => {
    const session = createSession(() => 0.5);
    const first = session.displayed_orders[0][0];
    const second = session.displayed_orders[0][1];
    const withMost = setChoice(session, "most", first);
    expect(withMost.draft_best).toBe(first);
    const withBoth = setChoice(withMost, "least", second);
    expect(withBoth.draft_worst).toBe(second);
    const leastMovedToFirst = setChoice(withBoth, "least", first);
    expect(leastMovedToFirst.draft_best).toBeNull();
    expect(leastMovedToFirst.draft_worst).toBe(first);
    expect(clearChoice(leastMovedToFirst, "least").draft_worst).toBeNull();
  });

  it("rejects corrupted stored state", () => {
    expect(parseSession("not-json")).toBeNull();
    expect(parseSession(JSON.stringify({ schema_version: 3 }))).toBeNull();
    const session = createSession(() => 0.5);
    session.set_order[1] = session.set_order[0];
    expect(parseSession(JSON.stringify(session))).toBeNull();
  });

  it("migrates a version 4 onboarding session into the shared session", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.schema_version = 4;
    delete legacy.experience;
    const migrated = parseSession(JSON.stringify(legacy));
    expect(migrated?.schema_version).toBe(7);
    expect(migrated?.experience.active_view).toBe("experience");
    expect(migrated?.responses).toEqual(legacy.responses);
  });

  it("migrates version 5 drafts without carrying fixture trace state forward", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.schema_version = 5;
    legacy.experience.journal_draft = "A draft worth keeping.";
    legacy.experience.trace_event_ids = ["fixture-event"];
    legacy.experience.selected_event_id = "fixture-event";

    const migrated = parseSession(JSON.stringify(legacy));

    expect(migrated?.schema_version).toBe(7);
    expect(migrated?.experience.journal_draft).toBe("A draft worth keeping.");
    expect(migrated?.experience.trace_event_ids).toEqual([]);
    expect(migrated?.experience.trace_events).toEqual([]);
    expect(migrated?.experience.selected_event_id).toBeNull();
  });

  it("migrates version 6 goal and confirmed Profile state", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    const responses = completeResponses();
    const profile = createProfile({
      userId: legacy.user_id,
      sessionId: legacy.session_id,
      startedAt: legacy.started_at,
      completedAt: "2026-07-19T00:02:00.000Z",
      responses,
      userConfirmed: true,
    });
    legacy.schema_version = 6;
    legacy.stage = "complete";
    legacy.set_index = 10;
    legacy.responses = responses;
    legacy.goal_category = "direction";
    legacy.confirmed_profile = {
      ...profile,
      schema_version: 2,
      onboarding_version: "2.1.0",
      goal_category: "direction",
    };

    const migrated = parseSession(JSON.stringify(legacy));

    expect(migrated?.schema_version).toBe(7);
    expect(migrated?.confirmed_profile?.schema_version).toBe(3);
    expect(migrated?.confirmed_profile).not.toHaveProperty("goal_category");
  });

  it("routes a version 6 goal stage to the Core Value summary", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.schema_version = 6;
    legacy.stage = "goal";
    legacy.set_index = 10;
    legacy.responses = completeResponses();
    legacy.goal_category = null;

    const migrated = parseSession(JSON.stringify(legacy));

    expect(migrated?.stage).toBe("summary");
    expect(migrated).not.toHaveProperty("goal_category");
  });

  it("unlocks Inspect after all 11 questions and preserves event selection", () => {
    const session = createSession(() => 0.5);
    expect(showView(session, "inspect")).toBe(session);
    expect(inspectRun(session, "event-09")).toBe(session);

    session.stage = "summary";
    session.set_index = 10;
    session.responses = completeResponses();
    const scoreInspection = showView(session, "inspect");
    expect(scoreInspection.experience.active_view).toBe("inspect");
    expect(parseSession(JSON.stringify(scoreInspection))).toEqual(scoreInspection);

    session.confirmed_profile = {} as NonNullable<typeof session.confirmed_profile>;
    session.experience.trace_event_ids = ["event-09"];
    expect(inspectRun(session, "event-10")).toBe(session);
    const inspected = inspectRun(session, "event-09");
    expect(inspected.experience.active_view).toBe("inspect");
    expect(inspected.experience.selected_event_id).toBe("event-09");
    expect(showView(inspected, "experience").experience.selected_event_id).toBe("event-09");
  });

  it("rejects a response that does not match the randomized progress order", () => {
    const session = createSession(() => 0.5);
    session.set_index = 1;
    const wrongSet = BWS_SETS[session.set_order[1]];
    session.responses = [{
      set_number: wrongSet.setNumber,
      items: [...wrongSet.items],
      item_order_shown: [...session.displayed_orders[session.set_order[1]]],
      selected_best: wrongSet.items[0],
      selected_worst: wrongSet.items[1],
      response_time_ms: 1_000,
    }];
    expect(parseSession(JSON.stringify(session))).toBeNull();
  });
});
