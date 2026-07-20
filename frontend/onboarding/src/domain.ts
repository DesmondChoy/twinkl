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

export const BWS_OBJECT_ORDER = [
  "power",
  "achievement",
  "hedonism",
  "stimulation",
  "self_direction",
  "universalism_nature",
  "benevolence",
  "tradition",
  "conformity",
  "security",
  "universalism_social",
] as const;

export type BwsObjectKey = (typeof BWS_OBJECT_ORDER)[number];

export interface BwsObjectDefinition {
  key: BwsObjectKey;
  value: ValueKey;
  descriptor: string;
}

// Descriptor triplets from Lee, Soutar, and Louviere (2008). The two
// Universalism facets remain distinct until the Profile transformation.
export const BWS_OBJECTS: Record<BwsObjectKey, BwsObjectDefinition> = {
  power: {
    key: "power",
    value: "power",
    descriptor: "Social power, authority, wealth",
  },
  achievement: {
    key: "achievement",
    value: "achievement",
    descriptor: "Successful, capable, ambitious",
  },
  hedonism: {
    key: "hedonism",
    value: "hedonism",
    descriptor: "Pleasure, enjoying life, self-indulgent",
  },
  stimulation: {
    key: "stimulation",
    value: "stimulation",
    descriptor: "Daring, a varied life, an exciting life",
  },
  self_direction: {
    key: "self_direction",
    value: "self_direction",
    descriptor: "Creativity, curious, freedom",
  },
  universalism_nature: {
    key: "universalism_nature",
    value: "universalism",
    descriptor: "Protecting the environment, a world of beauty, unity with nature",
  },
  benevolence: {
    key: "benevolence",
    value: "benevolence",
    descriptor: "Helpful, honest, forgiving",
  },
  tradition: {
    key: "tradition",
    value: "tradition",
    descriptor: "Devout, accepting portion in life, humble",
  },
  conformity: {
    key: "conformity",
    value: "conformity",
    descriptor: "Politeness, honouring parents & elders, obedient",
  },
  security: {
    key: "security",
    value: "security",
    descriptor: "Clean, national & family security, social order",
  },
  universalism_social: {
    key: "universalism_social",
    value: "universalism",
    descriptor: "Equality, world at peace, social justice",
  },
};

export interface BwsSet {
  setNumber: number;
  items: readonly BwsObjectKey[];
}

// Lee, Soutar, and Louviere's 11 x 6 balanced incomplete block design: every
// object appears in six sets and every pair appears together in three sets.
export const BWS_SETS: readonly BwsSet[] = [
  {
    setNumber: 1,
    items: [
      "achievement",
      "universalism_nature",
      "benevolence",
      "tradition",
      "security",
      "universalism_social",
    ],
  },
  {
    setNumber: 2,
    items: ["power", "hedonism", "benevolence", "tradition", "conformity", "universalism_social"],
  },
  {
    setNumber: 3,
    items: ["power", "achievement", "stimulation", "tradition", "conformity", "security"],
  },
  {
    setNumber: 4,
    items: ["achievement", "hedonism", "self_direction", "conformity", "security", "universalism_social"],
  },
  {
    setNumber: 5,
    items: ["power", "hedonism", "stimulation", "universalism_nature", "security", "universalism_social"],
  },
  {
    setNumber: 6,
    items: ["power", "achievement", "stimulation", "self_direction", "benevolence", "universalism_social"],
  },
  {
    setNumber: 7,
    items: ["power", "achievement", "hedonism", "self_direction", "universalism_nature", "tradition"],
  },
  {
    setNumber: 8,
    items: ["achievement", "hedonism", "stimulation", "universalism_nature", "benevolence", "conformity"],
  },
  {
    setNumber: 9,
    items: ["hedonism", "stimulation", "self_direction", "benevolence", "tradition", "security"],
  },
  {
    setNumber: 10,
    items: ["stimulation", "self_direction", "universalism_nature", "tradition", "conformity", "universalism_social"],
  },
  {
    setNumber: 11,
    items: ["power", "self_direction", "universalism_nature", "benevolence", "conformity", "security"],
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
  items: BwsObjectKey[];
  item_order_shown: BwsObjectKey[];
  selected_best: BwsObjectKey;
  selected_worst: BwsObjectKey;
  response_time_ms: number;
}

export type BwsObjectVector = Record<BwsObjectKey, number>;
export type ValueVector = Record<ValueKey, number>;

export interface RawBwsScores {
  appearances: BwsObjectVector;
  best_counts: BwsObjectVector;
  worst_counts: BwsObjectVector;
  net_counts: BwsObjectVector;
  scores: BwsObjectVector;
}

export interface ProfileTransform {
  method: "mean_universalism_facets_then_shift_normalize_v1";
  scores: ValueVector;
  weights: ValueVector;
  top_values: ValueKey[];
  bottom_values: ValueKey[];
}

export interface ScoreBundle {
  bws: RawBwsScores;
  profile: ProfileTransform;
}

export interface OnboardingProfile {
  schema_version: 2;
  onboarding_version: "2.1.0";
  instrument: "svbws_lee_soutar_louviere_2008_ui_adaptation_v2";
  scoring_method: "best_minus_worst_divided_by_appearances_v1";
  user_id: string;
  session_id: string;
  started_at: string;
  timestamp: string;
  bws_responses: BwsResponse[];
  bws_results: RawBwsScores;
  value_profile: ProfileTransform;
  top_values: ValueKey[];
  goal_category: GoalCategory;
  user_confirmed: true;
  provenance: {
    source: "react_onboarding_poc";
    set_order_randomized: true;
    card_order_randomized: true;
  };
}

const BWS_OBJECT_SET = new Set<string>(BWS_OBJECT_ORDER);

export function isBwsObjectKey(value: unknown): value is BwsObjectKey {
  return typeof value === "string" && BWS_OBJECT_SET.has(value);
}

export function bwsObjectVector(initial = 0): BwsObjectVector {
  return Object.fromEntries(BWS_OBJECT_ORDER.map((value) => [value, initial])) as BwsObjectVector;
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
    response.items.length !== 6 ||
    response.item_order_shown.length !== 6 ||
    new Set(response.items).size !== 6 ||
    new Set(response.item_order_shown).size !== 6 ||
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
    throw new Error("A confirmed Profile requires all 11 BWS responses");
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

function transformForProfile(bwsScores: BwsObjectVector): ProfileTransform {
  const scores = valueVector();
  VALUE_ORDER.forEach((value) => {
    if (value === "universalism") {
      scores[value] = round(
        (bwsScores.universalism_nature + bwsScores.universalism_social) / 2,
      );
      return;
    }
    scores[value] = bwsScores[value];
  });
  const highest = Math.max(...VALUE_ORDER.map((value) => scores[value]));
  const lowest = Math.min(...VALUE_ORDER.map((value) => scores[value]));
  const tolerance = 1e-8;
  return {
    method: "mean_universalism_facets_then_shift_normalize_v1",
    scores,
    weights: normalizedWeights(scores),
    top_values: VALUE_ORDER.filter((value) => Math.abs(scores[value] - highest) <= tolerance),
    bottom_values: VALUE_ORDER.filter((value) => Math.abs(scores[value] - lowest) <= tolerance),
  };
}

export function scoreResponses(
  responses: BwsResponse[],
  requireComplete = false,
): ScoreBundle {
  if (responses.length === 0) {
    throw new Error("At least one BWS response is required");
  }
  assertResponses(responses, requireComplete);

  const appearances = bwsObjectVector();
  const bestCounts = bwsObjectVector();
  const worstCounts = bwsObjectVector();
  responses.forEach((response) => {
    response.items.forEach((value) => {
      appearances[value] += 1;
    });
    bestCounts[response.selected_best] += 1;
    worstCounts[response.selected_worst] += 1;
  });

  const netCounts = bwsObjectVector();
  const scores = bwsObjectVector();
  BWS_OBJECT_ORDER.forEach((value) => {
    netCounts[value] = bestCounts[value] - worstCounts[value];
    scores[value] = appearances[value] === 0 ? 0 : round(netCounts[value] / appearances[value]);
  });

  return {
    bws: {
      appearances,
      best_counts: bestCounts,
      worst_counts: worstCounts,
      net_counts: netCounts,
      scores,
    },
    profile: transformForProfile(scores),
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
    throw new Error("An onboarding Profile cannot be emitted before confirmation");
  }
  if (!(input.goalCategory in GOALS)) {
    throw new Error("A valid goal category is required");
  }
  const results = scoreResponses(input.responses, true);
  return {
    schema_version: 2,
    onboarding_version: "2.1.0",
    instrument: "svbws_lee_soutar_louviere_2008_ui_adaptation_v2",
    scoring_method: "best_minus_worst_divided_by_appearances_v1",
    user_id: input.userId,
    session_id: input.sessionId,
    started_at: input.startedAt,
    timestamp: input.completedAt,
    bws_responses: input.responses,
    bws_results: results.bws,
    value_profile: results.profile,
    top_values: results.profile.top_values,
    goal_category: input.goalCategory,
    user_confirmed: true,
    provenance: {
      source: "react_onboarding_poc",
      set_order_randomized: true,
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
    profile.schema_version !== 2 ||
    profile.onboarding_version !== "2.1.0" ||
    profile.instrument !== "svbws_lee_soutar_louviere_2008_ui_adaptation_v2" ||
    profile.scoring_method !== "best_minus_worst_divided_by_appearances_v1" ||
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
