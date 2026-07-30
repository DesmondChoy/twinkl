import { describe, expect, it } from "vitest";
import { BWS_SETS, createProfile } from "./domain";
import {
  clearChoice,
  createSession,
  inspectRun,
  parseSession,
  setChoice,
  showView,
} from "./session";

describe("onboarding session", () => {
  it("randomizes set order and every prescribed card order once, then round-trips", () => {
    const ids = ["user-1", "session-1"];
    const session = createSession(() => 0, new Date("2026-07-19T00:00:00.000Z"), () => ids.shift()!);
    expect(session.schema_version).toBe(6);
    expect(session.stage).toBe("name");
    expect(session.preferred_name).toBe("");
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
    expect(migrated?.schema_version).toBe(6);
    expect(migrated?.preferred_name).toBe("Friend");
    expect(migrated?.experience.active_view).toBe("experience");
    expect(migrated?.responses).toEqual(legacy.responses);
  });

  it("preserves a confirmed legacy Profile without inventing a stored name", () => {
    const session = createSession(() => 0.5);
    const responses = BWS_SETS.map((set, index) => ({
      set_number: set.setNumber,
      items: [...set.items],
      item_order_shown: [...set.items],
      selected_best: set.items[0],
      selected_worst: set.items[1],
      response_time_ms: 1_000 + index,
    }));
    session.schema_version = 6;
    session.stage = "complete";
    session.preferred_name = "Casey";
    session.responses = responses;
    session.goal_category = "direction";
    session.confirmed_profile = createProfile({
      userId: session.user_id,
      preferredName: session.preferred_name,
      sessionId: session.session_id,
      startedAt: session.started_at,
      completedAt: "2026-07-19T00:02:00.000Z",
      responses,
      goalCategory: session.goal_category,
      userConfirmed: true,
    });

    const legacy = JSON.parse(JSON.stringify(session));
    legacy.schema_version = 5;
    delete legacy.preferred_name;
    delete legacy.confirmed_profile.preferred_name;

    const migrated = parseSession(JSON.stringify(legacy));

    expect(migrated?.preferred_name).toBe("Friend");
    expect(migrated?.confirmed_profile?.preferred_name).toBeUndefined();
  });

  it("keeps Inspect unavailable before Profile confirmation and preserves event selection", () => {
    const session = createSession(() => 0.5);
    expect(showView(session, "inspect")).toBe(session);
    expect(inspectRun(session, "event-09")).toBe(session);

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
    session.stage = "set";
    session.preferred_name = "Casey";
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
