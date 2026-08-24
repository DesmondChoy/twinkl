import type {
  ExperienceInspectFixtureContract,
  ExperienceSessionContract,
  NudgeInteractionContract,
  ScenarioBundleContract,
  ScenarioDeliveryState,
  TraceEventContract,
} from "./demoContracts";
import { validateExperienceInspectFixture } from "./demoContracts";

const HASH_PATTERN = /^[0-9a-f]{64}$/;
const SCENARIO_FILE_PATTERN = /^[a-z0-9-]+\.json$/;

export interface ScenarioCatalogItem {
  scenario_id: string;
  file: string;
  content_sha256: string;
  persona_id: string;
  persona_name: string;
  age: string;
  profession: string;
  culture: string;
  core_values: string[];
  role:
    | "no_active_drift"
    | "active_drift"
    | "drift_ended"
    | "insufficient_evidence"
    | "two_core_values";
  progression: ScenarioDeliveryState[];
  summary: string;
  recommended: boolean;
}

export interface ScenarioCatalog {
  schema_version: "scenario-catalog-v1";
  evidence_source: "ai_reviewed_synthetic_development";
  scenarios: ScenarioCatalogItem[];
}

export interface LoadedScenario {
  catalogItem: ScenarioCatalogItem;
  fixture: ExperienceInspectFixtureContract;
}

export interface ScenarioWeekProjection {
  session: ExperienceSessionContract;
  events: TraceEventContract[];
}

function record(value: unknown, name: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function text(value: unknown, name: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${name} must be a non-empty string`);
  }
  return value;
}

function textList(value: unknown, name: string): string[] {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  return value.map((item, index) => text(item, `${name}[${index}]`));
}

function validateCatalogItem(
  value: unknown,
  index: number,
): ScenarioCatalogItem {
  const name = `catalog.scenarios[${index}]`;
  const item = record(value, name);
  const scenarioId = text(item.scenario_id, `${name}.scenario_id`);
  const file = text(item.file, `${name}.file`);
  const contentSha256 = text(
    item.content_sha256,
    `${name}.content_sha256`,
  );
  if (!SCENARIO_FILE_PATTERN.test(file) || file !== `${scenarioId}.json`) {
    throw new Error(`${name}.file is incompatible`);
  }
  if (!HASH_PATTERN.test(contentSha256)) {
    throw new Error(`${name}.content_sha256 is incompatible`);
  }
  const role = text(item.role, `${name}.role`);
  if (![
    "no_active_drift",
    "active_drift",
    "drift_ended",
    "insufficient_evidence",
    "two_core_values",
  ].includes(role)) {
    throw new Error(`${name}.role is incompatible`);
  }
  if (!Array.isArray(item.progression) || item.progression.length === 0) {
    throw new Error(`${name}.progression must not be empty`);
  }
  const progression = item.progression.map((state, stateIndex) => {
    if (!["active_drift", "no_active_drift", "insufficient_evidence"].includes(
      String(state),
    )) {
      throw new Error(`${name}.progression[${stateIndex}] is incompatible`);
    }
    return state as ScenarioDeliveryState;
  });
  if (typeof item.recommended !== "boolean") {
    throw new Error(`${name}.recommended must be a boolean`);
  }
  return {
    scenario_id: scenarioId,
    file,
    content_sha256: contentSha256,
    persona_id: text(item.persona_id, `${name}.persona_id`),
    persona_name: text(item.persona_name, `${name}.persona_name`),
    age: text(item.age, `${name}.age`),
    profession: text(item.profession, `${name}.profession`),
    culture: text(item.culture, `${name}.culture`),
    core_values: textList(item.core_values, `${name}.core_values`),
    role: role as ScenarioCatalogItem["role"],
    progression,
    summary: text(item.summary, `${name}.summary`),
    recommended: item.recommended,
  };
}

export function validateScenarioCatalog(value: unknown): ScenarioCatalog {
  const catalog = record(value, "catalog");
  if (
    catalog.schema_version !== "scenario-catalog-v1" ||
    catalog.evidence_source !== "ai_reviewed_synthetic_development" ||
    !Array.isArray(catalog.scenarios) ||
    catalog.scenarios.length === 0
  ) {
    throw new Error("Saved persona catalog is incompatible");
  }
  const scenarios = catalog.scenarios.map(validateCatalogItem);
  if (new Set(scenarios.map((item) => item.scenario_id)).size !== scenarios.length) {
    throw new Error("Saved persona catalog repeats a scenario");
  }
  if (scenarios.filter((item) => item.recommended).length !== 1) {
    throw new Error("Saved persona catalog must recommend one scenario");
  }
  return {
    schema_version: "scenario-catalog-v1",
    evidence_source: "ai_reviewed_synthetic_development",
    scenarios,
  };
}

async function fetchResponse(path: string): Promise<Response> {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("The saved persona replay could not be loaded.");
  }
  return response;
}

async function fetchJson(path: string): Promise<unknown> {
  const response = await fetchResponse(path);
  return response.json();
}

function hexDigest(bytes: ArrayBuffer): string {
  return [...new Uint8Array(bytes)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function fetchVerifiedJson(
  path: string,
  expectedSha256: string,
): Promise<unknown> {
  const response = await fetchResponse(path);
  const bytes = await response.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  if (hexDigest(digest) !== expectedSha256) {
    throw new Error("The saved persona replay failed its integrity check.");
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}

export async function loadScenarioCatalog(): Promise<ScenarioCatalog> {
  return validateScenarioCatalog(await fetchJson("/scenarios/index.json"));
}

export async function loadSavedScenario(
  catalogItem: ScenarioCatalogItem,
): Promise<LoadedScenario> {
  const fixture = validateExperienceInspectFixture(
    await fetchVerifiedJson(
      `/scenarios/${catalogItem.file}`,
      catalogItem.content_sha256,
    ),
  );
  if (
    fixture.scenario.scenario_id !== catalogItem.scenario_id ||
    fixture.scenario.persona_id !== catalogItem.persona_id ||
    fixture.scenario.weeks.length !== catalogItem.progression.length
  ) {
    throw new Error("The saved persona replay does not match its catalog.");
  }
  return { catalogItem, fixture };
}

export async function loadSavedScenarioById(
  scenarioOrPersonaId: string,
): Promise<LoadedScenario> {
  const catalog = await loadScenarioCatalog();
  const item = catalog.scenarios.find(
    (scenario) =>
      scenario.scenario_id === scenarioOrPersonaId ||
      scenario.persona_id === scenarioOrPersonaId,
  );
  if (!item) throw new Error("The selected saved persona is unavailable.");
  return loadSavedScenario(item);
}

function eventResult(
  events: TraceEventContract[],
  eventIds: Set<string>,
  eventType: string,
  field: string,
): Record<string, unknown> {
  const event = events.find(
    (candidate) =>
      eventIds.has(candidate.event_id) && candidate.event_type === eventType,
  );
  const value = event?.details[field];
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`Saved replay week is missing ${eventType}.`);
  }
  return value as Record<string, unknown>;
}

export function projectScenarioWeek(
  fixture: ExperienceInspectFixtureContract,
  weekIndex: number,
): ScenarioWeekProjection {
  const scenario: ScenarioBundleContract = fixture.scenario;
  if (
    !Number.isInteger(weekIndex) ||
    weekIndex < 0 ||
    weekIndex >= scenario.weeks.length
  ) {
    throw new Error("Unknown saved replay week.");
  }
  const selectedWeek = scenario.weeks[weekIndex];
  const visibleWeeks = scenario.weeks.slice(0, weekIndex + 1);
  const visibleJournalEntryIds = new Set(
    visibleWeeks.flatMap((week) => week.journal_entry_ids),
  );
  const visibleEventIds = new Set(
    visibleWeeks.flatMap((week) => week.event_ids),
  );
  const currentEventIds = new Set(selectedWeek.event_ids);
  const events = fixture.trace_events.filter((event) =>
    visibleEventIds.has(event.event_id),
  );
  const journalEntries = scenario.journal_entries.filter((entry) =>
    visibleJournalEntryIds.has(entry.journal_entry_id),
  );
  const nudges = events.flatMap((event) => {
    if (event.event_type !== "nudge_generated") return [];
    const nudge = event.details.nudge;
    return nudge === null || typeof nudge !== "object" || Array.isArray(nudge)
      ? []
      : [nudge as NudgeInteractionContract];
  });
  const driftResult = eventResult(
    fixture.trace_events,
    currentEventIds,
    "drift_detected",
    "result",
  );
  const weeklyDigest = eventResult(
    fixture.trace_events,
    currentEventIds,
    "weekly_digest_built",
    "digest",
  );
  const digestEvent = events.find(
    (event) =>
      currentEventIds.has(event.event_id) &&
      event.event_type === "weekly_digest_built",
  );
  const traceEventIds = events.map((event) => event.event_id);

  return {
    session: {
      schema_version: "experience-inspect-v1",
      session_id: scenario.profile.session_id,
      revision: journalEntries.length,
      profile: scenario.profile,
      journal_entries: journalEntries,
      nudges,
      weekly_reviewer_decisions:
        scenario.weekly_reviewer_decisions.filter(
          (decision) => decision.week_end <= selectedWeek.week_end,
        ),
      drift_result: driftResult,
      weekly_digest: weeklyDigest,
      assessment_clock: null,
      trace_event_ids: traceEventIds,
      selection: {
        view: "experience",
        selected_week: selectedWeek.week_id,
        selected_journal_entry_id: null,
        selected_event_id: null,
      },
      updated_at:
        digestEvent?.completed_at ??
        digestEvent?.started_at ??
        `${selectedWeek.week_end}T23:59:59Z`,
    },
    events,
  };
}
