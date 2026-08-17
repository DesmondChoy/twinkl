import { describe, expect, it } from "vitest";
import {
  BWS_SETS,
  VALUE_ORDER,
  createProfile,
  scoreResponses,
  type BwsResponse,
} from "./domain";
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

  const tiedSelectedPairs = [
    ["achievement", "universalism_social"],
    ["power", "benevolence"],
    ["stimulation", "power"],
    ["hedonism", "conformity"],
    ["universalism_nature", "hedonism"],
    ["self_direction", "achievement"],
    ["tradition", "universalism_nature"],
    ["conformity", "stimulation"],
    ["security", "tradition"],
    ["universalism_social", "self_direction"],
    ["benevolence", "security"],
  ] as const;

  const tiedResponses = (): BwsResponse[] =>
    BWS_SETS.map((set, index) => ({
      set_number: set.setNumber,
      items: [...set.items],
      item_order_shown: [...set.items],
      selected_best: tiedSelectedPairs[index][0],
      selected_worst: tiedSelectedPairs[index][1],
      response_time_ms: 1_000,
    }));

  it("randomizes set order and every prescribed card order once, then round-trips", () => {
    const ids = ["user-1", "session-1"];
    const session = createSession(() => 0, new Date("2026-07-19T00:00:00.000Z"), () => ids.shift()!);
    expect(session.schema_version).toBe(10);
    expect(session.stage).toBe("name");
    expect(session.preferred_name).toBe("");
    expect(session.experience).toMatchObject({
      active_view: "experience",
      data_notice_acknowledged: false,
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
    legacy.stage = "set";
    delete legacy.preferred_name;
    delete legacy.experience;
    const migrated = parseSession(JSON.stringify(legacy));
    expect(migrated?.schema_version).toBe(10);
    expect(migrated?.preferred_name).toBe("Friend");
    expect(migrated?.experience.active_view).toBe("experience");
    expect(migrated?.responses).toEqual(legacy.responses);
  });

  it("migrates version 5 drafts without carrying fixture trace state forward", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.schema_version = 5;
    legacy.stage = "set";
    delete legacy.preferred_name;
    legacy.experience.journal_draft = "A draft worth keeping.";
    legacy.experience.trace_event_ids = ["fixture-event"];
    legacy.experience.selected_event_id = "fixture-event";

    const migrated = parseSession(JSON.stringify(legacy));

    expect(migrated?.schema_version).toBe(10);
    expect(migrated?.experience.journal_draft).toBe("A draft worth keeping.");
    expect(migrated?.experience.trace_event_ids).toEqual([]);
    expect(migrated?.experience.trace_events).toEqual([]);
    expect(migrated?.experience.selected_event_id).toBeNull();
  });

  it("migrates version 6 goal and confirmed Profile state", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.preferred_name = "Casey";
    const responses = completeResponses();
    const profile = createProfile({
      userId: legacy.user_id,
      preferredName: legacy.preferred_name,
      sessionId: legacy.session_id,
      startedAt: legacy.started_at,
      completedAt: "2026-07-19T00:02:00.000Z",
      responses,
      selectedTopValues: scoreResponses(responses, true).profile.top_values.slice(0, 2),
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

    expect(migrated?.schema_version).toBe(10);
    expect(migrated?.confirmed_profile?.schema_version).toBe(4);
    expect(migrated?.confirmed_profile).not.toHaveProperty("goal_category");
  });

  it("migrates a version 3 Profile with at most two Core Values", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.preferred_name = "Casey";
    const responses = completeResponses();
    const profile = createProfile({
      userId: legacy.user_id,
      preferredName: legacy.preferred_name,
      sessionId: legacy.session_id,
      startedAt: legacy.started_at,
      completedAt: "2026-07-19T00:02:00.000Z",
      responses,
      userConfirmed: true,
    });
    legacy.schema_version = 8;
    legacy.stage = "complete";
    legacy.set_index = 10;
    legacy.responses = responses;
    legacy.confirmed_profile = {
      ...profile,
      schema_version: 3,
      onboarding_version: "2.2.0",
    };

    const migrated = parseSession(JSON.stringify(legacy));

    expect(migrated?.confirmed_profile).toMatchObject({
      schema_version: 4,
      onboarding_version: "2.3.0",
      top_values: profile.top_values,
    });
  });

  it("preserves raw Experience data when a legacy tie requires reselection", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.preferred_name = "Casey";
    const responses = tiedResponses();
    const profile = createProfile({
      userId: legacy.user_id,
      preferredName: legacy.preferred_name,
      sessionId: legacy.session_id,
      startedAt: legacy.started_at,
      completedAt: "2026-07-19T00:02:00.000Z",
      responses,
      selectedTopValues: VALUE_ORDER.slice(0, 2),
      userConfirmed: true,
    });
    legacy.schema_version = 8;
    legacy.stage = "complete";
    legacy.set_index = 10;
    legacy.responses = responses;
    legacy.confirmed_profile = {
      ...profile,
      schema_version: 3,
      onboarding_version: "2.2.0",
      top_values: [...VALUE_ORDER],
    };
    legacy.experience = {
      ...legacy.experience,
      journal_started: true,
      journal_draft: "A draft worth preserving.",
      revision: 4,
      journal_entries: [{
        journal_entry_id: "entry-1",
        t_index: 0,
        date: "2026-07-20",
        content: "A saved Journal Entry.",
        nudge_response: null,
      }],
      nudges: [{ nudge_id: "old-nudge" }],
      weekly_reviewer_decisions: [{ verdict: "conflict" }],
      drift_result: { state: "active_drift" },
      weekly_digest: { text: "Old digest" },
      weekly_coach: { text: "Old Coach Digest" },
      assessment_clock: {
        mode: "simulated_assessment",
        current_date: "2026-07-21",
        timezone: "Asia/Singapore",
      },
      run_state: "complete",
      trace_event_ids: ["old-event"],
      trace_events: [{ event_id: "old-event" }],
      selected_event_id: "old-event",
    };

    const migrated = parseSession(JSON.stringify(legacy));

    expect(migrated).toMatchObject({
      schema_version: 10,
      session_id: `${legacy.session_id}:core-values-v2`,
      stage: "summary",
      selected_top_values: [],
      confirmed_profile: null,
    });
    expect(migrated?.responses).toEqual(responses);
    expect(migrated?.experience).toMatchObject({
      journal_started: true,
      journal_draft: "A draft worth preserving.",
      revision: 1,
      journal_entries: legacy.experience.journal_entries,
      nudges: [],
      weekly_reviewer_decisions: [],
      drift_result: null,
      weekly_digest: null,
      weekly_coach: null,
      assessment_clock: legacy.experience.assessment_clock,
      run_state: "idle",
      trace_event_ids: [],
      trace_events: [],
      selected_event_id: null,
    });
  });

  it("migrates version 7 sessions without an assessment clock", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.schema_version = 7;
    delete legacy.experience.assessment_clock;

    const migrated = parseSession(JSON.stringify(legacy));

    expect(migrated?.schema_version).toBe(10);
    expect(migrated?.experience.assessment_clock).toBeNull();
  });

  it("requires a new data notice acknowledgement after version 9 migration", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.schema_version = 9;
    delete legacy.experience.data_notice_acknowledged;

    const migrated = parseSession(JSON.stringify(legacy));

    expect(migrated?.schema_version).toBe(10);
    expect(migrated?.experience.data_notice_acknowledged).toBe(false);
  });

  it("preserves a legacy Profile without a stored preferred name", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.preferred_name = "Casey";
    const responses = completeResponses();
    legacy.schema_version = 6;
    legacy.stage = "complete";
    legacy.set_index = 10;
    legacy.responses = responses;
    legacy.confirmed_profile = createProfile({
      userId: legacy.user_id,
      preferredName: legacy.preferred_name,
      sessionId: legacy.session_id,
      startedAt: legacy.started_at,
      completedAt: "2026-07-19T00:02:00.000Z",
      responses,
      selectedTopValues: scoreResponses(responses, true).profile.top_values.slice(0, 2),
      userConfirmed: true,
    });
    delete legacy.preferred_name;
    delete legacy.confirmed_profile.preferred_name;

    const migrated = parseSession(JSON.stringify(legacy));

    expect(migrated?.preferred_name).toBe("Friend");
    expect(migrated?.confirmed_profile?.preferred_name).toBeUndefined();
  });

  it("routes a version 6 goal stage to the Core Value summary", () => {
    const legacy = JSON.parse(JSON.stringify(createSession(() => 0.5)));
    legacy.schema_version = 6;
    legacy.stage = "goal";
    delete legacy.preferred_name;
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
    session.preferred_name = "Casey";
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
