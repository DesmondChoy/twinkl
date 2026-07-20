import {
  BWS_SETS,
  GOALS,
  type BwsResponse,
  type GoalCategory,
  type OnboardingProfile,
  type ValueKey,
  isValueKey,
  scoreResponses,
  validateProfile,
} from "./domain";

export const SESSION_STORAGE_KEY = "twinkl.onboarding.session.v2";

export type OnboardingStage =
  | "set"
  | "mirror"
  | "goal"
  | "summary"
  | "complete";

export interface OnboardingSession {
  schema_version: 2;
  user_id: string;
  session_id: string;
  started_at: string;
  stage: OnboardingStage;
  set_index: number;
  stage_started_at_ms: number;
  displayed_orders: ValueKey[][];
  responses: BwsResponse[];
  draft_best: ValueKey | null;
  draft_worst: ValueKey | null;
  goal_category: GoalCategory | null;
  confirmed_profile: OnboardingProfile | null;
}

function shuffled<T>(items: readonly T[], random: () => number): T[] {
  const result = [...items];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [result[index], result[swapIndex]] = [result[swapIndex], result[index]];
  }
  return result;
}

export function createSession(
  random: () => number = Math.random,
  now: Date = new Date(),
  makeId: () => string = () => crypto.randomUUID(),
): OnboardingSession {
  return {
    schema_version: 2,
    user_id: makeId(),
    session_id: makeId(),
    started_at: now.toISOString(),
    stage: "set",
    set_index: 0,
    stage_started_at_ms: now.getTime(),
    displayed_orders: BWS_SETS.map((set) => shuffled(set.items, random)),
    responses: [],
    draft_best: null,
    draft_worst: null,
    goal_category: null,
    confirmed_profile: null,
  };
}

function isDisplayedOrders(value: unknown): value is ValueKey[][] {
  return (
    Array.isArray(value) &&
    value.length === BWS_SETS.length &&
    value.every(
      (order, index) =>
        Array.isArray(order) &&
        order.length === 4 &&
        new Set(order).size === 4 &&
        order.every(
          (item) => isValueKey(item) && BWS_SETS[index].items.includes(item),
        ),
    )
  );
}

export function parseSession(raw: string | null): OnboardingSession | null {
  if (!raw) return null;
  try {
    const session = JSON.parse(raw) as Partial<OnboardingSession>;
    if (
      session.schema_version !== 2 ||
      typeof session.user_id !== "string" ||
      typeof session.session_id !== "string" ||
      typeof session.started_at !== "string" ||
      !["set", "mirror", "goal", "summary", "complete"].includes(
        session.stage ?? "",
      ) ||
      !Number.isInteger(session.set_index) ||
      session.set_index! < 0 ||
      session.set_index! >= BWS_SETS.length ||
      typeof session.stage_started_at_ms !== "number" ||
      !isDisplayedOrders(session.displayed_orders) ||
      !Array.isArray(session.responses) ||
      !(session.draft_best === null || isValueKey(session.draft_best)) ||
      !(session.draft_worst === null || isValueKey(session.draft_worst)) ||
      !(
        session.confirmed_profile === null ||
        (typeof session.confirmed_profile === "object" && session.confirmed_profile !== undefined)
      ) ||
      !(
        session.goal_category === null ||
        (typeof session.goal_category === "string" && session.goal_category in GOALS)
      )
    ) {
      return null;
    }
    if (session.responses.length > 0) {
      scoreResponses(session.responses);
    }
    if (session.stage === "complete") {
      validateProfile(session.confirmed_profile);
    }
    return session as OnboardingSession;
  } catch {
    return null;
  }
}

export function loadOrCreateSession(): OnboardingSession {
  return parseSession(localStorage.getItem(SESSION_STORAGE_KEY)) ?? createSession();
}

export function persistSession(session: OnboardingSession): void {
  localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  localStorage.removeItem(SESSION_STORAGE_KEY);
}

export function setChoice(
  session: OnboardingSession,
  choice: "most" | "least",
  value: ValueKey,
): OnboardingSession {
  if (choice === "most") {
    return {
      ...session,
      draft_best: value,
      draft_worst: session.draft_worst === value ? null : session.draft_worst,
    };
  }
  return {
    ...session,
    draft_best: session.draft_best === value ? null : session.draft_best,
    draft_worst: value,
  };
}

export function clearChoice(
  session: OnboardingSession,
  choice: "most" | "least",
): OnboardingSession {
  return choice === "most"
    ? { ...session, draft_best: null }
    : { ...session, draft_worst: null };
}
