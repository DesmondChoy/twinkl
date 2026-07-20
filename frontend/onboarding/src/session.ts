import {
  BWS_SETS,
  GOALS,
  type BwsObjectKey,
  type BwsResponse,
  type GoalCategory,
  type OnboardingProfile,
  isBwsObjectKey,
  scoreResponses,
  validateProfile,
} from "./domain";

export const SESSION_STORAGE_KEY = "twinkl.onboarding.session.v4";

export type OnboardingStage = "set" | "goal" | "summary" | "complete";

export interface OnboardingSession {
  schema_version: 4;
  user_id: string;
  session_id: string;
  started_at: string;
  stage: OnboardingStage;
  set_index: number;
  set_order: number[];
  stage_started_at_ms: number;
  displayed_orders: BwsObjectKey[][];
  responses: BwsResponse[];
  draft_best: BwsObjectKey | null;
  draft_worst: BwsObjectKey | null;
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
    schema_version: 4,
    user_id: makeId(),
    session_id: makeId(),
    started_at: now.toISOString(),
    stage: "set",
    set_index: 0,
    set_order: shuffled(BWS_SETS.map((_, index) => index), random),
    stage_started_at_ms: now.getTime(),
    displayed_orders: BWS_SETS.map((set) => shuffled(set.items, random)),
    responses: [],
    draft_best: null,
    draft_worst: null,
    goal_category: null,
    confirmed_profile: null,
  };
}

function isSetOrder(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    value.length === BWS_SETS.length &&
    new Set(value).size === BWS_SETS.length &&
    value.every(
      (index) => Number.isInteger(index) && index >= 0 && index < BWS_SETS.length,
    )
  );
}

function isDisplayedOrders(value: unknown): value is BwsObjectKey[][] {
  return (
    Array.isArray(value) &&
    value.length === BWS_SETS.length &&
    value.every(
      (order, index) =>
        Array.isArray(order) &&
        order.length === 6 &&
        new Set(order).size === 6 &&
        order.every(
          (item) => isBwsObjectKey(item) && BWS_SETS[index].items.includes(item),
        ),
    )
  );
}

export function parseSession(raw: string | null): OnboardingSession | null {
  if (!raw) return null;
  try {
    const session = JSON.parse(raw) as Partial<OnboardingSession>;
    if (
      session.schema_version !== 4 ||
      typeof session.user_id !== "string" ||
      typeof session.session_id !== "string" ||
      typeof session.started_at !== "string" ||
      !["set", "goal", "summary", "complete"].includes(session.stage ?? "") ||
      !Number.isInteger(session.set_index) ||
      session.set_index! < 0 ||
      session.set_index! >= BWS_SETS.length ||
      !isSetOrder(session.set_order) ||
      typeof session.stage_started_at_ms !== "number" ||
      !isDisplayedOrders(session.displayed_orders) ||
      !Array.isArray(session.responses) ||
      !(session.draft_best === null || isBwsObjectKey(session.draft_best)) ||
      !(session.draft_worst === null || isBwsObjectKey(session.draft_worst)) ||
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
    const setIndex = session.set_index as number;
    const setOrder = session.set_order as number[];
    const currentItems = new Set(BWS_SETS[setOrder[setIndex]].items);
    if (
      (session.draft_best !== null && !currentItems.has(session.draft_best)) ||
      (session.draft_worst !== null && !currentItems.has(session.draft_worst)) ||
      (session.draft_best !== null && session.draft_best === session.draft_worst)
    ) {
      return null;
    }
    if (session.responses.length > 0) {
      scoreResponses(session.responses);
    }
    if (
      (session.stage === "set" && session.responses.length !== setIndex) ||
      (session.stage !== "set" && session.responses.length !== BWS_SETS.length)
    ) {
      return null;
    }
    if (session.stage === "set") {
      const expectedCompletedSets = new Set(
        setOrder
          .slice(0, setIndex)
          .map((canonicalIndex) => BWS_SETS[canonicalIndex].setNumber),
      );
      if (
        session.responses.some(
          (response) => !expectedCompletedSets.has(response.set_number),
        )
      ) {
        return null;
      }
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
  value: BwsObjectKey,
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
