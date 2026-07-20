import { describe, expect, it } from "vitest";
import {
  BWS_OBJECTS,
  BWS_OBJECT_ORDER,
  BWS_SETS,
  VALUE_ORDER,
  createProfile,
  scoreResponses,
  validateProfile,
  type BwsResponse,
} from "./domain";

const selectedPairs = [
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

describe("published SVBWS design", () => {
  it("preserves the published descriptor triplets verbatim", () => {
    expect(BWS_OBJECT_ORDER.map((key) => BWS_OBJECTS[key].descriptor)).toEqual([
      "Social power, authority, wealth",
      "Successful, capable, ambitious",
      "Pleasure, enjoying life, self-indulgent",
      "Daring, a varied life, an exciting life",
      "Creativity, curious, freedom",
      "Protecting the environment, a world of beauty, unity with nature",
      "Helpful, honest, forgiving",
      "Devout, accepting portion in life, humble",
      "Politeness, honouring parents & elders, obedient",
      "Clean, national & family security, social order",
      "Equality, world at peace, social justice",
    ]);
  });

  it("has 11 six-object sets with balanced exposure and pair frequency", () => {
    expect(BWS_SETS).toHaveLength(11);
    expect(BWS_SETS.every((set) => set.items.length === 6)).toBe(true);

    const appearances = Object.fromEntries(BWS_OBJECT_ORDER.map((item) => [item, 0]));
    const pairCounts = new Map<string, number>();
    BWS_SETS.forEach((set) => {
      set.items.forEach((item) => {
        appearances[item] += 1;
      });
      set.items.forEach((left, leftIndex) => {
        set.items.slice(leftIndex + 1).forEach((right) => {
          const pair = [left, right].sort().join("|");
          pairCounts.set(pair, (pairCounts.get(pair) ?? 0) + 1);
        });
      });
    });

    expect(Object.values(appearances)).toEqual(Array(11).fill(6));
    expect(pairCounts.size).toBe(55);
    expect([...pairCounts.values()]).toEqual(Array(55).fill(3));
  });
});

describe("BWS scoring", () => {
  it("keeps raw 11-object scores separate from the ten-value Profile transformation", () => {
    const responses = completeResponses();
    responses[0].selected_best = "universalism_social";
    responses[0].selected_worst = "achievement";

    const results = scoreResponses(responses, true);
    expect(results.bws.appearances.achievement).toBe(6);
    expect(results.bws.scores.achievement).toBeCloseTo(-1 / 3, 7);
    expect(results.bws.scores.universalism_social).toBeCloseTo(1 / 3, 7);
    expect(results.bws.scores.universalism_nature).toBe(0);
    expect(results.profile.scores.universalism).toBeCloseTo(1 / 6, 7);
    expect(results.profile.method).toBe(
      "mean_universalism_facets_then_shift_normalize_v1",
    );
    expect(results).not.toHaveProperty("confidence");
  });

  it("produces positive product weights that sum exactly to one", () => {
    const responses = completeResponses();
    responses[0].selected_best = "universalism_social";
    responses[0].selected_worst = "achievement";
    const weights = VALUE_ORDER.map(
      (value) => scoreResponses(responses, true).profile.weights[value],
    );
    expect(weights.every((weight) => weight > 0)).toBe(true);
    expect(weights.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 12);
  });

  it("retains every exact highest-score tie in canonical order", () => {
    expect(scoreResponses(completeResponses(), true).profile.top_values).toEqual(
      VALUE_ORDER,
    );
  });

  it("rejects incomplete and invalid responses", () => {
    expect(() => scoreResponses(completeResponses().slice(0, 10), true)).toThrow(
      "all 11",
    );
    const responses = completeResponses();
    responses[0].selected_worst = responses[0].selected_best;
    expect(() => scoreResponses(responses, true)).toThrow("distinct valid");
  });
});

describe("versioned Profile", () => {
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
    expect(profile.onboarding_version).toBe("2.1.0");
    expect(profile.instrument).toBe(
      "svbws_lee_soutar_louviere_2008_ui_adaptation_v2",
    );
    expect(profile.bws_results.scores).toHaveProperty("universalism_nature");
    expect(profile.value_profile.scores).toHaveProperty("universalism");
    expect(profile).not.toHaveProperty("confidence");
    expect(validateProfile(JSON.parse(JSON.stringify(profile)))).toEqual(profile);
  });
});
