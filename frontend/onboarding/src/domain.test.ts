import { describe, expect, it } from "vitest";
import {
  BWS_SETS,
  VALUE_ORDER,
  createProfile,
  scoreResponses,
  validateProfile,
  type BwsResponse,
} from "./domain";

const selectedPairs = [
  ["security", "benevolence"],
  ["stimulation", "universalism"],
  ["hedonism", "power"],
  ["achievement", "conformity"],
  ["security", "tradition"],
  ["self_direction", "security"],
] as const;

function completeResponses(): BwsResponse[] {
  return BWS_SETS.map((set, index) => ({
    set_number: set.setNumber,
    items: [...set.items],
    item_order_shown: [...set.items].reverse(),
    selected_best: selectedPairs[index][0],
    selected_worst: selectedPairs[index][1],
    response_time_ms: 1_000 + index,
  }));
}

describe("BWS scoring", () => {
  it("normalizes net choices by item exposure", () => {
    const scores = scoreResponses(completeResponses(), true);
    expect(scores.appearances.security).toBe(3);
    expect(scores.net_counts.security).toBe(1);
    expect(scores.scores.security).toBeCloseTo(1 / 3, 7);
    expect(scores.appearances.achievement).toBe(2);
    expect(scores.scores.achievement).toBe(0.5);
  });

  it("produces positive weights that sum exactly to one", () => {
    const scores = scoreResponses(completeResponses(), true);
    const weights = VALUE_ORDER.map((value) => scores.weights[value]);
    expect(weights.every((weight) => weight > 0)).toBe(true);
    expect(weights.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 12);
  });

  it("retains every exact highest-score tie in canonical order", () => {
    const responses = completeResponses();
    responses[0].selected_best = "self_direction";
    responses[0].selected_worst = "security";
    responses[2].selected_best = "self_direction";
    responses[2].selected_worst = "hedonism";
    responses[5].selected_best = "universalism";
    responses[5].selected_worst = "security";
    const scores = scoreResponses(responses, true);
    const ordered = scores.top_values.map((value) => VALUE_ORDER.indexOf(value));
    expect(ordered).toEqual([...ordered].sort((left, right) => left - right));
  });

  it("keeps all ten Core Values when every exposure-normalized score ties", () => {
    const pairs = [
      ["security", "self_direction"],
      ["stimulation", "conformity"],
      ["hedonism", "tradition"],
      ["conformity", "stimulation"],
      ["tradition", "hedonism"],
      ["self_direction", "security"],
    ] as const;
    const responses = completeResponses();
    responses.forEach((response, index) => {
      response.selected_best = pairs[index][0];
      response.selected_worst = pairs[index][1];
    });
    expect(scoreResponses(responses, true).top_values).toEqual(VALUE_ORDER);
  });

  it("rejects incomplete and invalid responses", () => {
    expect(() => scoreResponses(completeResponses().slice(0, 5), true)).toThrow(
      "all six",
    );
    const responses = completeResponses();
    responses[0].selected_worst = responses[0].selected_best;
    expect(() => scoreResponses(responses, true)).toThrow("distinct valid");
  });
});

describe("versioned profile", () => {
  it("requires confirmation and round-trips through deterministic validation", () => {
    const input = {
      userId: "user-1",
      sessionId: "session-1",
      startedAt: "2026-07-19T00:00:00.000Z",
      completedAt: "2026-07-19T00:02:00.000Z",
      responses: completeResponses(),
      goalCategory: "direction" as const,
    };
    expect(() => createProfile({ ...input, userConfirmed: false })).toThrow(
      "before confirmation",
    );
    const profile = createProfile({ ...input, userConfirmed: true });
    expect(validateProfile(JSON.parse(JSON.stringify(profile)))).toEqual(profile);
  });
});
