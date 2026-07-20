import { describe, expect, it } from "vitest";
import { BWS_SETS } from "./domain";
import { clearChoice, createSession, parseSession, setChoice } from "./session";

describe("onboarding session", () => {
  it("randomizes set order and every prescribed card order once, then round-trips", () => {
    const ids = ["user-1", "session-1"];
    const session = createSession(() => 0, new Date("2026-07-19T00:00:00.000Z"), () => ids.shift()!);
    expect(session.schema_version).toBe(4);
    expect(session.stage).toBe("set");
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
