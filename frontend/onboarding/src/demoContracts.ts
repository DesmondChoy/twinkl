import {
  type OnboardingProfile,
  validateProfile,
} from "./domain";

export const EXPERIENCE_INSPECT_CONTRACT_VERSION = "experience-inspect-v1" as const;

const EVENT_TYPES = new Set([
  "profile_confirmed",
  "journal_entry_submitted",
  "nudge_suppression_checked",
  "nudge_decided",
  "nudge_generated",
  "weekly_review_requested",
  "weekly_review_completed",
  "drift_detected",
  "weekly_digest_built",
  "weekly_coach_generated",
]);
const EVENT_STATUSES = new Set([
  "queued",
  "running",
  "complete",
  "reused",
  "refused",
  "invalid",
  "failed",
]);
const EVENT_SOURCES = new Set(["saved_replay", "live_run"]);
const REVIEW_VERDICTS = new Set(["conflict", "not_conflict", "abstain"]);
const REVIEW_STATUSES = new Set(["ok", "refusal", "invalid", "error"]);
const OPERATIONS = new Set([
  "create_session",
  "submit_journal_entry",
  "load_scenario",
  "read_trace",
]);
const HASH_PATTERN = /^[0-9a-f]{64}$/;

type JsonObject = Record<string, unknown>;

export interface WeeklyDriftReviewerDecisionContract {
  persona_id: string;
  week_start: string;
  week_end: string;
  t_index: number;
  date: string;
  core_value: string;
  verdict: "conflict" | "not_conflict" | "abstain";
  confidence: "low" | "medium" | "high" | null;
  reason_code: string | null;
  evidence_quote: string;
  review_status: "ok" | "refusal" | "invalid" | "error";
}

export interface TraceEventContract extends JsonObject {
  schema_version: typeof EXPERIENCE_INSPECT_CONTRACT_VERSION;
  event_id: string;
  session_id: string;
  parent_event_id: string | null;
  event_type: string;
  status: string;
  source: string;
  started_at: string;
  completed_at: string | null;
  duration_ms: number | null;
  input_refs: unknown[];
  model_contract: JsonObject | null;
  prompt: string | null;
  raw_response: unknown;
  validation: JsonObject | null;
  result_refs: unknown[];
  input_hash: string;
  error: JsonObject | null;
  details: JsonObject;
}

export interface ExperienceSessionContract extends JsonObject {
  schema_version: typeof EXPERIENCE_INSPECT_CONTRACT_VERSION;
  session_id: string;
  revision: number;
  profile: OnboardingProfile;
  journal_entries: JsonObject[];
  nudges: JsonObject[];
  weekly_reviewer_decisions: WeeklyDriftReviewerDecisionContract[];
  drift_result: JsonObject | null;
  weekly_digest: JsonObject | null;
  trace_event_ids: string[];
  selection: JsonObject;
  updated_at: string;
}

export interface ExperienceInspectFixtureContract extends JsonObject {
  schema_version: typeof EXPERIENCE_INSPECT_CONTRACT_VERSION;
  session: ExperienceSessionContract;
  scenario: JsonObject;
  requests: JsonObject[];
  responses: JsonObject[];
  trace_events: TraceEventContract[];
}

function object(value: unknown, name: string): JsonObject {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${name} must be an object`);
  }
  return value as JsonObject;
}

function array(value: unknown, name: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  return value;
}

function string(value: unknown, name: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`${name} must be a non-empty string`);
  }
  return value;
}

function nullableString(value: unknown, name: string): string | null {
  if (value === null) return null;
  return string(value, name);
}

function integer(value: unknown, name: string): number {
  if (!Number.isInteger(value) || (value as number) < 0) {
    throw new Error(`${name} must be a non-negative integer`);
  }
  return value as number;
}

function boolean(value: unknown, name: string): boolean {
  if (typeof value !== "boolean") throw new Error(`${name} must be a boolean`);
  return value;
}

function exactKeys(value: JsonObject, expected: readonly string[], name: string): void {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  if (JSON.stringify(actual) !== JSON.stringify(wanted)) {
    throw new Error(`${name} has incompatible fields`);
  }
}

function version(value: JsonObject, name: string): void {
  if (value.schema_version !== EXPERIENCE_INSPECT_CONTRACT_VERSION) {
    throw new Error(`${name} has an incompatible schema_version`);
  }
}

function stringArray(value: unknown, name: string): string[] {
  return array(value, name).map((item, index) => string(item, `${name}[${index}]`));
}

function validateModelContract(value: unknown, name: string): JsonObject {
  const contract = object(value, name);
  exactKeys(contract, ["provider", "model", "reasoning_effort"], name);
  string(contract.provider, `${name}.provider`);
  string(contract.model, `${name}.model`);
  if (contract.reasoning_effort !== null) {
    string(contract.reasoning_effort, `${name}.reasoning_effort`);
  }
  return contract;
}

function validateLunaLow(value: unknown, name: string): void {
  const contract = validateModelContract(value, name);
  if (contract.model !== "gpt-5.6-luna" || contract.reasoning_effort !== "low") {
    throw new Error(`${name} must preserve the Luna-low Weekly Drift Reviewer contract`);
  }
}

function validateSafeError(value: unknown, name: string): JsonObject {
  const error = object(value, name);
  exactKeys(error, ["code", "message", "retryable"], name);
  string(error.code, `${name}.code`);
  const message = string(error.message, `${name}.message`).toLowerCase();
  if (message.includes("sk-") || message.includes("api_key=") || message.includes("authorization:")) {
    throw new Error(`${name} contains provider secrets`);
  }
  boolean(error.retryable, `${name}.retryable`);
  return error;
}

function validateJournalEntry(value: unknown, name: string): JsonObject {
  const entry = object(value, name);
  exactKeys(entry, ["journal_entry_id", "t_index", "date", "content", "nudge_response"], name);
  string(entry.journal_entry_id, `${name}.journal_entry_id`);
  integer(entry.t_index, `${name}.t_index`);
  string(entry.date, `${name}.date`);
  string(entry.content, `${name}.content`);
  if (entry.nudge_response !== null) string(entry.nudge_response, `${name}.nudge_response`);
  return entry;
}

function validateDecision(value: unknown, name: string): WeeklyDriftReviewerDecisionContract {
  const decision = object(value, name);
  exactKeys(
    decision,
    [
      "persona_id",
      "week_start",
      "week_end",
      "t_index",
      "date",
      "core_value",
      "verdict",
      "confidence",
      "reason_code",
      "evidence_quote",
      "review_status",
    ],
    name,
  );
  string(decision.persona_id, `${name}.persona_id`);
  string(decision.week_start, `${name}.week_start`);
  string(decision.week_end, `${name}.week_end`);
  integer(decision.t_index, `${name}.t_index`);
  string(decision.date, `${name}.date`);
  string(decision.core_value, `${name}.core_value`);
  if (!REVIEW_VERDICTS.has(String(decision.verdict))) {
    throw new Error(`${name}.verdict is incompatible`);
  }
  if (decision.confidence !== null && !["low", "medium", "high"].includes(String(decision.confidence))) {
    throw new Error(`${name}.confidence is incompatible`);
  }
  if (decision.reason_code !== null) string(decision.reason_code, `${name}.reason_code`);
  if (typeof decision.evidence_quote !== "string") {
    throw new Error(`${name}.evidence_quote must be a string`);
  }
  if (!REVIEW_STATUSES.has(String(decision.review_status))) {
    throw new Error(`${name}.review_status is incompatible`);
  }
  return decision as unknown as WeeklyDriftReviewerDecisionContract;
}

function validateNudge(value: unknown, name: string): JsonObject {
  const nudge = object(value, name);
  exactKeys(
    nudge,
    ["nudge_id", "journal_entry_id", "outcome", "category", "reason", "text", "response"],
    name,
  );
  string(nudge.nudge_id, `${name}.nudge_id`);
  string(nudge.journal_entry_id, `${name}.journal_entry_id`);
  if (!["suppressed", "no_nudge", "displayed", "skipped", "answered"].includes(String(nudge.outcome))) {
    throw new Error(`${name}.outcome is incompatible`);
  }
  for (const field of ["category", "reason", "text", "response"] as const) {
    if (nudge[field] !== null) string(nudge[field], `${name}.${field}`);
  }
  return nudge;
}

function validateSession(value: unknown, name: string): ExperienceSessionContract {
  const session = object(value, name);
  exactKeys(
    session,
    [
      "schema_version",
      "session_id",
      "revision",
      "profile",
      "journal_entries",
      "nudges",
      "weekly_reviewer_decisions",
      "drift_result",
      "weekly_digest",
      "trace_event_ids",
      "selection",
      "updated_at",
    ],
    name,
  );
  version(session, name);
  const sessionId = string(session.session_id, `${name}.session_id`);
  integer(session.revision, `${name}.revision`);
  const profile = validateProfile(session.profile);
  if (profile.session_id !== sessionId) throw new Error(`${name} and Profile session_id differ`);
  const entries = array(session.journal_entries, `${name}.journal_entries`).map((entry, index) =>
    validateJournalEntry(entry, `${name}.journal_entries[${index}]`),
  );
  const indices = entries.map((entry) => entry.t_index as number);
  if (new Set(indices).size !== indices.length || indices.some((item, index) => index > 0 && item <= indices[index - 1])) {
    throw new Error(`${name}.journal_entries must have unique chronological t_index`);
  }
  array(session.nudges, `${name}.nudges`).forEach((nudge, index) =>
    validateNudge(nudge, `${name}.nudges[${index}]`),
  );
  array(session.weekly_reviewer_decisions, `${name}.weekly_reviewer_decisions`).forEach(
    (decision, index) => validateDecision(decision, `${name}.weekly_reviewer_decisions[${index}]`),
  );
  if (session.drift_result !== null) {
    const drift = object(session.drift_result, `${name}.drift_result`);
    if (drift.schema_version !== "drift-detector-result-v1") {
      throw new Error(`${name}.drift_result has an incompatible schema_version`);
    }
  }
  if (session.weekly_digest !== null) object(session.weekly_digest, `${name}.weekly_digest`);
  const eventIds = stringArray(session.trace_event_ids, `${name}.trace_event_ids`);
  if (new Set(eventIds).size !== eventIds.length) throw new Error(`${name}.trace_event_ids repeat`);
  object(session.selection, `${name}.selection`);
  string(session.updated_at, `${name}.updated_at`);
  return session as ExperienceSessionContract;
}

function validateEventDetails(event: JsonObject, name: string): void {
  const details = object(event.details, `${name}.details`);
  switch (event.event_type) {
    case "profile_confirmed":
      exactKeys(details, ["profile"], `${name}.details`);
      validateProfile(details.profile);
      break;
    case "journal_entry_submitted":
      exactKeys(details, ["journal_entry", "ordering_valid"], `${name}.details`);
      validateJournalEntry(details.journal_entry, `${name}.details.journal_entry`);
      boolean(details.ordering_valid, `${name}.details.ordering_valid`);
      break;
    case "nudge_suppression_checked":
      exactKeys(details, ["previous_entry_ids", "window_size", "max_nudges", "suppressed"], `${name}.details`);
      stringArray(details.previous_entry_ids, `${name}.details.previous_entry_ids`);
      if (details.window_size !== 3 || details.max_nudges !== 2) {
        throw new Error(`${name}.details changes the anti-annoyance contract`);
      }
      boolean(details.suppressed, `${name}.details.suppressed`);
      break;
    case "nudge_decided": {
      exactKeys(details, ["should_nudge", "category", "reason"], `${name}.details`);
      const shouldNudge = boolean(details.should_nudge, `${name}.details.should_nudge`);
      if (shouldNudge !== (details.category !== null)) {
        throw new Error(`${name}.details category disagrees with should_nudge`);
      }
      break;
    }
    case "nudge_generated":
      exactKeys(details, ["nudge", "word_count", "attempts"], `${name}.details`);
      integer(details.attempts, `${name}.details.attempts`);
      if ((details.nudge === null) !== (details.word_count === null)) {
        throw new Error(`${name}.details nudge and word_count must be present together`);
      }
      if (details.nudge !== null) {
        validateNudge(details.nudge, `${name}.details.nudge`);
        integer(details.word_count, `${name}.details.word_count`);
      }
      break;
    case "weekly_review_requested": {
      exactKeys(details, ["request"], `${name}.details`);
      const request = object(details.request, `${name}.details.request`);
      exactKeys(
        request,
        [
          "persona_id",
          "week_start",
          "week_end",
          "core_values",
          "history",
          "current_t_indices",
          "prompt",
          "prompt_sha256",
          "runtime_text_sha256",
        ],
        `${name}.details.request`,
      );
      break;
    }
    case "weekly_review_completed": {
      exactKeys(details, ["receipt"], `${name}.details`);
      const receipt = object(details.receipt, `${name}.details.receipt`);
      exactKeys(
        receipt,
        [
          "schema_version",
          "created_at",
          "persona_id",
          "week_start",
          "week_end",
          "core_values",
          "current_t_indices",
          "prompt_name",
          "prompt_version",
          "prompt_sha256",
          "runtime_text_sha256",
          "requested_model",
          "reasoning_effort",
          "status",
          "attempts",
          "latency_seconds",
          "resolved_model",
          "response_id",
          "usage",
          "refusal",
          "validation_error",
          "error_type",
          "error",
          "assessments",
          "decisions",
        ],
        `${name}.details.receipt`,
      );
      if (receipt.schema_version !== "weekly-drift-reviewer-receipt-v1") {
        throw new Error(`${name}.details.receipt has an incompatible schema_version`);
      }
      array(receipt.decisions, `${name}.details.receipt.decisions`).forEach((decision, index) =>
        validateDecision(decision, `${name}.details.receipt.decisions[${index}]`),
      );
      break;
    }
    case "drift_detected":
      exactKeys(details, ["decisions", "steps", "result"], `${name}.details`);
      array(details.decisions, `${name}.details.decisions`).forEach((decision, index) =>
        validateDecision(decision, `${name}.details.decisions[${index}]`),
      );
      if (object(details.result, `${name}.details.result`).schema_version !== "drift-detector-result-v1") {
        throw new Error(`${name}.details.result has an incompatible schema_version`);
      }
      break;
    case "weekly_digest_built":
      exactKeys(details, ["digest", "cited_journal_entry_ids"], `${name}.details`);
      object(details.digest, `${name}.details.digest`);
      stringArray(details.cited_journal_entry_ids, `${name}.details.cited_journal_entry_ids`);
      break;
    case "weekly_coach_generated":
      exactKeys(details, ["narrative", "validation"], `${name}.details`);
      object(details.narrative, `${name}.details.narrative`);
      object(details.validation, `${name}.details.validation`);
      break;
    default:
      throw new Error(`${name}.event_type is incompatible`);
  }
}

function validateTraceEvent(value: unknown, name: string): TraceEventContract {
  const event = object(value, name);
  exactKeys(
    event,
    [
      "schema_version",
      "event_id",
      "session_id",
      "parent_event_id",
      "event_type",
      "status",
      "source",
      "started_at",
      "completed_at",
      "duration_ms",
      "input_refs",
      "model_contract",
      "prompt",
      "raw_response",
      "validation",
      "result_refs",
      "input_hash",
      "error",
      "details",
    ],
    name,
  );
  version(event, name);
  string(event.event_id, `${name}.event_id`);
  string(event.session_id, `${name}.session_id`);
  nullableString(event.parent_event_id, `${name}.parent_event_id`);
  if (!EVENT_TYPES.has(String(event.event_type))) throw new Error(`${name}.event_type is incompatible`);
  if (!EVENT_STATUSES.has(String(event.status))) throw new Error(`${name}.status is incompatible`);
  if (!EVENT_SOURCES.has(String(event.source))) throw new Error(`${name}.source is incompatible`);
  string(event.started_at, `${name}.started_at`);
  const terminal = ["complete", "reused", "refused", "invalid", "failed"].includes(String(event.status));
  if (terminal) {
    string(event.completed_at, `${name}.completed_at`);
    integer(event.duration_ms, `${name}.duration_ms`);
  } else if (event.completed_at !== null || event.duration_ms !== null) {
    throw new Error(`${name} is incomplete but has completion timing`);
  }
  array(event.input_refs, `${name}.input_refs`);
  array(event.result_refs, `${name}.result_refs`);
  if (!HASH_PATTERN.test(String(event.input_hash))) throw new Error(`${name}.input_hash is incompatible`);
  const failed = ["refused", "invalid", "failed"].includes(String(event.status));
  if (failed) validateSafeError(event.error, `${name}.error`);
  else if (event.error !== null) throw new Error(`${name} succeeded but contains an error`);
  if (event.status === "invalid") {
    const validation = object(event.validation, `${name}.validation`);
    if (validation.valid !== false) throw new Error(`${name} invalid status lacks failed validation`);
  }
  if (["weekly_review_requested", "weekly_review_completed"].includes(String(event.event_type))) {
    validateLunaLow(event.model_contract, `${name}.model_contract`);
  } else if (event.model_contract !== null) {
    validateModelContract(event.model_contract, `${name}.model_contract`);
  }
  validateEventDetails(event, name);
  return event as TraceEventContract;
}

function validateScenario(value: unknown, name: string): JsonObject {
  const scenario = object(value, name);
  exactKeys(
    scenario,
    [
      "schema_version",
      "scenario_id",
      "title",
      "description",
      "source",
      "persona_id",
      "profile",
      "journal_entries",
      "weekly_reviewer_decisions",
      "drift_result",
      "weekly_digest",
      "weeks",
      "trace_event_ids",
      "manifest",
    ],
    name,
  );
  version(scenario, name);
  if (scenario.source !== "saved_replay") throw new Error(`${name}.source must be saved_replay`);
  validateProfile(scenario.profile);
  array(scenario.journal_entries, `${name}.journal_entries`).forEach((entry, index) =>
    validateJournalEntry(entry, `${name}.journal_entries[${index}]`),
  );
  array(scenario.weekly_reviewer_decisions, `${name}.weekly_reviewer_decisions`).forEach(
    (decision, index) => validateDecision(decision, `${name}.weekly_reviewer_decisions[${index}]`),
  );
  if (object(scenario.drift_result, `${name}.drift_result`).schema_version !== "drift-detector-result-v1") {
    throw new Error(`${name}.drift_result has an incompatible schema_version`);
  }
  object(scenario.weekly_digest, `${name}.weekly_digest`);
  array(scenario.weeks, `${name}.weeks`);
  const manifest = object(scenario.manifest, `${name}.manifest`);
  exactKeys(
    manifest,
    ["bundle_version", "created_at", "input_hash", "source_files", "model_contract", "prompt_sha256"],
    `${name}.manifest`,
  );
  if (manifest.bundle_version !== "scenario-bundle-v1") {
    throw new Error(`${name}.manifest has an incompatible bundle_version`);
  }
  if (!HASH_PATTERN.test(String(manifest.input_hash)) || !HASH_PATTERN.test(String(manifest.prompt_sha256))) {
    throw new Error(`${name}.manifest has an incompatible hash`);
  }
  stringArray(manifest.source_files, `${name}.manifest.source_files`);
  validateLunaLow(manifest.model_contract, `${name}.manifest.model_contract`);
  stringArray(scenario.trace_event_ids, `${name}.trace_event_ids`);
  return scenario;
}

function validateRequest(value: unknown, name: string): JsonObject {
  const request = object(value, name);
  version(request, name);
  if (!OPERATIONS.has(String(request.operation))) throw new Error(`${name}.operation is incompatible`);
  string(request.request_id, `${name}.request_id`);
  const requestKeys: Record<string, string[]> = {
    create_session: ["schema_version", "operation", "request_id", "idempotency_key", "profile"],
    submit_journal_entry: [
      "schema_version",
      "operation",
      "request_id",
      "idempotency_key",
      "session_id",
      "expected_revision",
      "journal_entry",
    ],
    load_scenario: ["schema_version", "operation", "request_id", "scenario_id"],
    read_trace: ["schema_version", "operation", "request_id", "session_id", "after_event_id"],
  };
  exactKeys(request, requestKeys[String(request.operation)], name);
  if (["create_session", "submit_journal_entry"].includes(String(request.operation))) {
    if (!HASH_PATTERN.test(String(request.idempotency_key))) {
      throw new Error(`${name}.idempotency_key is incompatible`);
    }
  }
  if (request.operation === "create_session") validateProfile(request.profile);
  if (request.operation === "submit_journal_entry") {
    string(request.session_id, `${name}.session_id`);
    integer(request.expected_revision, `${name}.expected_revision`);
    validateJournalEntry(request.journal_entry, `${name}.journal_entry`);
  }
  return request;
}

function validateResponse(value: unknown, name: string): JsonObject {
  const response = object(value, name);
  version(response, name);
  string(response.request_id, `${name}.request_id`);
  if (response.operation === "error") {
    exactKeys(
      response,
      ["schema_version", "operation", "requested_operation", "request_id", "status", "error"],
      name,
    );
    if (!OPERATIONS.has(String(response.requested_operation)) || response.status !== "error") {
      throw new Error(`${name} has an incompatible error response`);
    }
    validateSafeError(response.error, `${name}.error`);
    return response;
  }
  if (!OPERATIONS.has(String(response.operation)) || response.status !== "ok") {
    throw new Error(`${name} has an incompatible success response`);
  }
  const responseKeys: Record<string, string[]> = {
    create_session: ["schema_version", "operation", "request_id", "status", "session"],
    submit_journal_entry: [
      "schema_version",
      "operation",
      "request_id",
      "status",
      "session",
      "event_ids",
    ],
    load_scenario: [
      "schema_version",
      "operation",
      "request_id",
      "status",
      "session",
      "scenario",
      "event_ids",
    ],
    read_trace: ["schema_version", "operation", "request_id", "status", "session_id", "events"],
  };
  exactKeys(response, responseKeys[String(response.operation)], name);
  if (["create_session", "submit_journal_entry", "load_scenario"].includes(String(response.operation))) {
    validateSession(response.session, `${name}.session`);
  }
  if (response.operation === "load_scenario") validateScenario(response.scenario, `${name}.scenario`);
  if (["submit_journal_entry", "load_scenario"].includes(String(response.operation))) {
    stringArray(response.event_ids, `${name}.event_ids`);
  }
  if (response.operation === "read_trace") {
    string(response.session_id, `${name}.session_id`);
    array(response.events, `${name}.events`).forEach((event, index) =>
      validateTraceEvent(event, `${name}.events[${index}]`),
    );
  }
  return response;
}

export function validateExperienceInspectFixture(value: unknown): ExperienceInspectFixtureContract {
  const fixture = object(value, "fixture");
  exactKeys(fixture, ["schema_version", "session", "scenario", "requests", "responses", "trace_events"], "fixture");
  version(fixture, "fixture");
  const session = validateSession(fixture.session, "fixture.session");
  const scenario = validateScenario(fixture.scenario, "fixture.scenario");
  const requests = array(fixture.requests, "fixture.requests").map((request, index) =>
    validateRequest(request, `fixture.requests[${index}]`),
  );
  const responses = array(fixture.responses, "fixture.responses").map((response, index) =>
    validateResponse(response, `fixture.responses[${index}]`),
  );
  const traceEvents = array(fixture.trace_events, "fixture.trace_events").map((event, index) =>
    validateTraceEvent(event, `fixture.trace_events[${index}]`),
  );
  const knownIds = new Set(traceEvents.map((event) => event.event_id));
  for (const event of traceEvents) {
    if (event.session_id !== session.session_id) throw new Error("Trace event session_id differs from the session");
    if (event.parent_event_id !== null && !knownIds.has(event.parent_event_id)) {
      throw new Error("Trace parent_event_id does not reference a known event");
    }
  }
  for (const eventId of session.trace_event_ids) {
    if (!knownIds.has(eventId)) throw new Error("Session references an unknown trace event");
  }
  for (const eventId of stringArray(scenario.trace_event_ids, "fixture.scenario.trace_event_ids")) {
    if (!knownIds.has(eventId)) throw new Error("Scenario references an unknown trace event");
  }
  return {
    schema_version: EXPERIENCE_INSPECT_CONTRACT_VERSION,
    session,
    scenario,
    requests,
    responses,
    trace_events: traceEvents,
  };
}
