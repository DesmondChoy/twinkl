export const VALUE_ORDER = [
  "self_direction",
  "stimulation",
  "hedonism",
  "achievement",
  "power",
  "security",
  "conformity",
  "tradition",
  "benevolence",
  "universalism",
] as const;

export type ValueKey = (typeof VALUE_ORDER)[number];

export interface ValueDefinition {
  key: ValueKey;
  name: string;
  phrase: string;
}

export const VALUES: Record<ValueKey, ValueDefinition> = {
  self_direction: {
    key: "self_direction",
    name: "Self-Direction",
    phrase: "Having the freedom to choose my own path",
  },
  stimulation: {
    key: "stimulation",
    name: "Stimulation",
    phrase: "Seeking new experiences and challenges",
  },
  hedonism: {
    key: "hedonism",
    name: "Hedonism",
    phrase: "Enjoying life and having fun",
  },
  achievement: {
    key: "achievement",
    name: "Achievement",
    phrase: "Making progress toward something meaningful",
  },
  power: {
    key: "power",
    name: "Power",
    phrase: "Having influence over how things go",
  },
  security: {
    key: "security",
    name: "Security",
    phrase: "Feeling calm and secure in my life",
  },
  conformity: {
    key: "conformity",
    name: "Conformity",
    phrase: "Being someone others can count on to do the right thing",
  },
  tradition: {
    key: "tradition",
    name: "Tradition",
    phrase: "Honoring the customs and practices I was raised with",
  },
  benevolence: {
    key: "benevolence",
    name: "Benevolence",
    phrase: "Being there for the people closest to me",
  },
  universalism: {
    key: "universalism",
    name: "Universalism",
    phrase: "Making the world a fairer, better place",
  },
};

export interface BwsSet {
  setNumber: number;
  items: readonly ValueKey[];
}

export const BWS_SETS: readonly BwsSet[] = [
  {
    setNumber: 1,
    items: ["security", "self_direction", "achievement", "benevolence"],
  },
  {
    setNumber: 2,
    items: ["stimulation", "power", "conformity", "universalism"],
  },
  {
    setNumber: 3,
    items: ["hedonism", "tradition", "self_direction", "power"],
  },
  {
    setNumber: 4,
    items: ["achievement", "benevolence", "stimulation", "conformity"],
  },
  {
    setNumber: 5,
    items: ["security", "universalism", "hedonism", "tradition"],
  },
  {
    setNumber: 6,
    items: ["self_direction", "stimulation", "universalism", "security"],
  },
] as const;

export const GOALS = {
  work_life_balance: "I'm stretched too thin between work and everything else",
  life_transition: "I'm going through a career or life transition",
  relationships: "I want to be more present for people I care about",
  health_wellbeing: "I'm neglecting my health or wellbeing",
  direction: "I feel stuck or unclear about my direction",
  meaningful_work: "I want to make more room for what matters to me",
} as const;

export type GoalCategory = keyof typeof GOALS;

export interface BwsResponse {
  set_number: number;
  items: ValueKey[];
  item_order_shown: ValueKey[];
  selected_best: ValueKey;
  selected_worst: ValueKey;
  response_time_ms: number;
}

export type ValueVector = Record<ValueKey, number>;

export interface ScoreBundle {
  appearances: ValueVector;
  best_counts: ValueVector;
  worst_counts: ValueVector;
  net_counts: ValueVector;
  scores: ValueVector;
  weights: ValueVector;
  top_values: ValueKey[];
  bottom_values: ValueKey[];
  confidence: {
    consistent: boolean;
    spread: number;
    method: "response_consistency_population_spread_v1";
  };
}

export interface OnboardingProfile {
  schema_version: 1;
  onboarding_version: "1.0.0";
  scoring_method: "exposure_normalized_best_worst_v1";
  user_id: string;
  session_id: string;
  started_at: string;
  timestamp: string;
  bws_responses: BwsResponse[];
  value_scores: Omit<ScoreBundle, "top_values" | "bottom_values" | "confidence">;
  confidence: ScoreBundle["confidence"];
  top_values: ValueKey[];
  goal_category: GoalCategory;
  user_confirmed: true;
  provenance: {
    source: "react_onboarding_poc";
    card_order_randomized: true;
  };
}

const VALUE_SET = new Set<string>(VALUE_ORDER);

export function isValueKey(value: unknown): value is ValueKey {
  return typeof value === "string" && VALUE_SET.has(value);
}

export function valueVector(initial = 0): ValueVector {
  return Object.fromEntries(VALUE_ORDER.map((value) => [value, initial])) as ValueVector;
}

function round(value: number, digits = 8): number {
  const scale = 10 ** digits;
  return Math.round((value + Number.EPSILON) * scale) / scale;
}

function assertResponse(response: BwsResponse): void {
  const expected = BWS_SETS[response.set_number - 1];
  if (!expected) {
    throw new Error(`Unknown BWS set ${response.set_number}`);
  }
  const expectedItems = new Set(expected.items);
  if (
    response.items.length !== 4 ||
    response.item_order_shown.length !== 4 ||
    new Set(response.items).size !== 4 ||
    new Set(response.item_order_shown).size !== 4 ||
    response.items.some((item) => !expectedItems.has(item)) ||
    response.item_order_shown.some((item) => !expectedItems.has(item))
  ) {
    throw new Error(`BWS set ${response.set_number} must contain its prescribed items`);
  }
  if (
    !expectedItems.has(response.selected_best) ||
    !expectedItems.has(response.selected_worst) ||
    response.selected_best === response.selected_worst
  ) {
    throw new Error(`BWS set ${response.set_number} needs distinct valid Most and Least choices`);
  }
  if (!Number.isInteger(response.response_time_ms) || response.response_time_ms < 0) {
    throw new Error("Response time must be a non-negative integer");
  }
}

function assertResponses(responses: BwsResponse[], requireComplete: boolean): void {
  const seen = new Set<number>();
  responses.forEach((response) => {
    assertResponse(response);
    if (seen.has(response.set_number)) {
      throw new Error(`Duplicate BWS set ${response.set_number}`);
    }
    seen.add(response.set_number);
  });
  if (requireComplete && (responses.length !== BWS_SETS.length || seen.size !== BWS_SETS.length)) {
    throw new Error("A confirmed profile requires all six BWS responses");
  }
}

function normalizedWeights(scores: ValueVector): ValueVector {
  const minimum = Math.min(...VALUE_ORDER.map((value) => scores[value]));
  const shifted = VALUE_ORDER.map((value) => scores[value] - minimum + 1);
  const total = shifted.reduce((sum, value) => sum + value, 0);
  const weights = valueVector();
  let assigned = 0;
  VALUE_ORDER.forEach((value, index) => {
    if (index === VALUE_ORDER.length - 1) {
      weights[value] = round(1 - assigned);
      return;
    }
    weights[value] = round(shifted[index] / total);
    assigned += weights[value];
  });
  return weights;
}

function populationSpread(values: number[]): number {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return round(Math.sqrt(variance), 6);
}

export function scoreResponses(
  responses: BwsResponse[],
  requireComplete = false,
): ScoreBundle {
  if (responses.length === 0) {
    throw new Error("At least one BWS response is required");
  }
  assertResponses(responses, requireComplete);

  const appearances = valueVector();
  const bestCounts = valueVector();
  const worstCounts = valueVector();
  responses.forEach((response) => {
    response.items.forEach((value) => {
      appearances[value] += 1;
    });
    bestCounts[response.selected_best] += 1;
    worstCounts[response.selected_worst] += 1;
  });

  const netCounts = valueVector();
  VALUE_ORDER.forEach((value) => {
    netCounts[value] = bestCounts[value] - worstCounts[value];
  });

  const scores = valueVector();
  VALUE_ORDER.forEach((value) => {
    const exposure = appearances[value];
    scores[value] = exposure === 0 ? 0 : round(netCounts[value] / exposure);
  });

  const highest = Math.max(...VALUE_ORDER.map((value) => scores[value]));
  const lowest = Math.min(...VALUE_ORDER.map((value) => scores[value]));
  const tolerance = 1e-8;
  const topValues = VALUE_ORDER.filter((value) => Math.abs(scores[value] - highest) <= tolerance);
  const bottomValues = VALUE_ORDER.filter((value) => Math.abs(scores[value] - lowest) <= tolerance);
  const bestSelections = new Set(responses.map((response) => response.selected_best));
  const worstSelections = new Set(responses.map((response) => response.selected_worst));
  const consistent = [...bestSelections].every((value) => !worstSelections.has(value));

  return {
    appearances,
    best_counts: bestCounts,
    worst_counts: worstCounts,
    net_counts: netCounts,
    scores,
    weights: normalizedWeights(scores),
    top_values: topValues,
    bottom_values: bottomValues,
    confidence: {
      consistent,
      spread: populationSpread(VALUE_ORDER.map((value) => scores[value])),
      method: "response_consistency_population_spread_v1",
    },
  };
}

export interface CreateProfileInput {
  userId: string;
  sessionId: string;
  startedAt: string;
  completedAt: string;
  responses: BwsResponse[];
  goalCategory: GoalCategory;
  userConfirmed: boolean;
}

export function createProfile(input: CreateProfileInput): OnboardingProfile {
  if (!input.userConfirmed) {
    throw new Error("An onboarding profile cannot be emitted before confirmation");
  }
  if (!(input.goalCategory in GOALS)) {
    throw new Error("A valid goal category is required");
  }
  const scores = scoreResponses(input.responses, true);
  const { top_values, bottom_values: _bottomValues, confidence, ...valueScores } = scores;
  return {
    schema_version: 1,
    onboarding_version: "1.0.0",
    scoring_method: "exposure_normalized_best_worst_v1",
    user_id: input.userId,
    session_id: input.sessionId,
    started_at: input.startedAt,
    timestamp: input.completedAt,
    bws_responses: input.responses,
    value_scores: valueScores,
    confidence,
    top_values,
    goal_category: input.goalCategory,
    user_confirmed: true,
    provenance: {
      source: "react_onboarding_poc",
      card_order_randomized: true,
    },
  };
}

export function validateProfile(value: unknown): OnboardingProfile {
  if (!value || typeof value !== "object") {
    throw new Error("Profile must be an object");
  }
  const profile = value as Partial<OnboardingProfile>;
  if (
    profile.schema_version !== 1 ||
    profile.onboarding_version !== "1.0.0" ||
    profile.scoring_method !== "exposure_normalized_best_worst_v1" ||
    profile.user_confirmed !== true ||
    !Array.isArray(profile.bws_responses) ||
    typeof profile.goal_category !== "string" ||
    typeof profile.user_id !== "string" ||
    typeof profile.session_id !== "string" ||
    typeof profile.started_at !== "string" ||
    typeof profile.timestamp !== "string"
  ) {
    throw new Error("Profile is missing required versioned fields");
  }
  const rebuilt = createProfile({
    userId: profile.user_id,
    sessionId: profile.session_id,
    startedAt: profile.started_at,
    completedAt: profile.timestamp,
    responses: profile.bws_responses,
    goalCategory: profile.goal_category as GoalCategory,
    userConfirmed: true,
  });
  if (JSON.stringify(rebuilt) !== JSON.stringify(profile)) {
    throw new Error("Profile contents do not match the deterministic scoring contract");
  }
  return rebuilt;
}
