import type { OnboardingProfile } from "./domain";
import {
  EXPERIENCE_INSPECT_CONTRACT_VERSION,
  type AssessmentTimeAdvancedResponseContract,
  type AssessmentClockContract,
  type ExperienceApiResponseContract,
  type ExperienceResumeStateContract,
  type JournalEntryContract,
  type JournalEntrySubmittedResponseContract,
  type SessionCreatedResponseContract,
  type TraceReadResponseContract,
  type TraceEventContract,
  validateExperienceApiResponse,
} from "./demoContracts";

const EXPERIENCE_API_PATH = "/api/experience";

export class ExperienceApiError extends Error {
  readonly code: string;
  readonly retryable: boolean;

  constructor(message: string, code = "request_failed", retryable = true) {
    super(message);
    this.name = "ExperienceApiError";
    this.code = code;
    this.retryable = retryable;
  }
}

function requestId(): string {
  return crypto.randomUUID();
}

function browserTimeZone(): string {
  return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
}

async function sha256(value: unknown): Promise<string> {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function buildProfileReselectionResumeState(
  profile: OnboardingProfile,
  journalEntries: JournalEntryContract[],
  assessmentClock: AssessmentClockContract | null,
): Promise<ExperienceResumeStateContract> {
  const completedAt = profile.timestamp;
  const profileEventId = `profile-reselection:${profile.session_id}:profile`;
  const traceEvents: TraceEventContract[] = [
    {
      schema_version: EXPERIENCE_INSPECT_CONTRACT_VERSION,
      event_id: profileEventId,
      session_id: profile.session_id,
      parent_event_id: null,
      event_type: "profile_confirmed",
      status: "complete",
      source: "live_run",
      started_at: completedAt,
      completed_at: completedAt,
      duration_ms: 0,
      input_refs: [],
      model_contract: null,
      prompt: null,
      raw_response: null,
      validation: {
        valid: true,
        schema_name: "profile-reselection-v1",
        errors: [],
      },
      result_refs: [{ kind: "profile", id: profile.session_id }],
      input_hash: await sha256({ operation: "profile_reselection", profile }),
      error: null,
      details: { profile },
    },
  ];
  for (const entry of journalEntries) {
    const parentEventId = traceEvents.at(-1)!.event_id;
    const eventId = `profile-reselection:${profile.session_id}:journal:${entry.journal_entry_id}`;
    traceEvents.push({
      schema_version: EXPERIENCE_INSPECT_CONTRACT_VERSION,
      event_id: eventId,
      session_id: profile.session_id,
      parent_event_id: parentEventId,
      event_type: "journal_entry_submitted",
      status: "complete",
      source: "live_run",
      started_at: completedAt,
      completed_at: completedAt,
      duration_ms: 0,
      input_refs: [{ kind: "profile", id: profile.session_id }],
      model_contract: null,
      prompt: null,
      raw_response: null,
      validation: {
        valid: true,
        schema_name: "journal-entry-preserved-v1",
        errors: [],
      },
      result_refs: [{ kind: "journal_entry", id: entry.journal_entry_id }],
      input_hash: await sha256({
        operation: "preserve_journal_entry",
        session_id: profile.session_id,
        entry,
      }),
      error: null,
      details: { journal_entry: entry, ordering_valid: true },
    });
  }
  return {
    session_id: profile.session_id,
    revision: journalEntries.length,
    journal_entries: journalEntries,
    nudges: [],
    assessment_clock: assessmentClock,
    trace_events: traceEvents,
  };
}

async function postExperience(
  payload: Record<string, unknown>,
): Promise<ExperienceApiResponseContract> {
  let response: Response;
  try {
    response = await fetch(EXPERIENCE_API_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ExperienceApiError(
      "The Experience service could not be reached.",
    );
  }

  let parsed: ExperienceApiResponseContract;
  try {
    parsed = validateExperienceApiResponse(await response.json());
  } catch {
    if (!response.ok) {
      const retryable =
        response.status === 408
        || response.status === 429
        || response.status >= 500;
      throw new ExperienceApiError(
        "The Experience service did not accept this request.",
        `http_${response.status}`,
        retryable,
      );
    }
    throw new ExperienceApiError(
      "The Experience service returned an unreadable response.",
      "invalid_response",
      false,
    );
  }
  if (parsed.operation === "error") {
    throw new ExperienceApiError(
      parsed.error.message,
      parsed.error.code,
      parsed.error.retryable,
    );
  }
  return parsed;
}

export async function createExperienceSession(
  profile: OnboardingProfile,
  resumeState: ExperienceResumeStateContract | null = null,
  assessmentTimezone: string | null = browserTimeZone(),
): Promise<SessionCreatedResponseContract> {
  const idempotencyKey = await sha256({
    operation: "create_session",
    profile,
    assessment_timezone: assessmentTimezone,
    resume_state: resumeState,
  });
  const response = await postExperience({
    schema_version: EXPERIENCE_INSPECT_CONTRACT_VERSION,
    operation: "create_session",
    request_id: requestId(),
    idempotency_key: idempotencyKey,
    profile,
    ...(assessmentTimezone === null
      ? {}
      : { assessment_timezone: assessmentTimezone }),
    ...(resumeState === null ? {} : { resume_state: resumeState }),
  });
  if (response.operation !== "create_session") {
    throw new ExperienceApiError(
      "The Experience session returned the wrong result.",
      "unexpected_operation",
      false,
    );
  }
  return response;
}

export async function advanceAssessmentTime(
  {
    sessionId,
    expectedRevision,
    action,
  }: {
    sessionId: string;
    expectedRevision: number;
    action: "next_day" | "close_week";
  },
): Promise<AssessmentTimeAdvancedResponseContract> {
  const idempotencyKey = await sha256({
    operation: "advance_assessment_time",
    session_id: sessionId,
    expected_revision: expectedRevision,
    action,
  });
  const response = await postExperience({
    schema_version: EXPERIENCE_INSPECT_CONTRACT_VERSION,
    operation: "advance_assessment_time",
    request_id: requestId(),
    idempotency_key: idempotencyKey,
    session_id: sessionId,
    expected_revision: expectedRevision,
    action,
  });
  if (response.operation !== "advance_assessment_time") {
    throw new ExperienceApiError(
      "Changing simulated time returned the wrong result.",
      "unexpected_operation",
      false,
    );
  }
  return response;
}

export async function journalIdempotencyKey(
  sessionId: string,
  entry: JournalEntryContract,
): Promise<string> {
  return sha256({
    operation: "submit_journal_entry",
    session_id: sessionId,
    journal_entry: entry,
  });
}

export async function submitJournalEntry(
  {
    sessionId,
    expectedRevision,
    entry,
    idempotencyKey,
  }: {
    sessionId: string;
    expectedRevision: number;
    entry: JournalEntryContract;
    idempotencyKey: string;
  },
): Promise<JournalEntrySubmittedResponseContract> {
  const response = await postExperience({
    schema_version: EXPERIENCE_INSPECT_CONTRACT_VERSION,
    operation: "submit_journal_entry",
    request_id: requestId(),
    idempotency_key: idempotencyKey,
    session_id: sessionId,
    expected_revision: expectedRevision,
    journal_entry: entry,
  });
  if (response.operation !== "submit_journal_entry") {
    throw new ExperienceApiError(
      "Saving the Journal Entry returned the wrong result.",
      "unexpected_operation",
      false,
    );
  }
  return response;
}

export async function readExperienceTrace(
  sessionId: string,
): Promise<TraceReadResponseContract> {
  const response = await postExperience({
    schema_version: EXPERIENCE_INSPECT_CONTRACT_VERSION,
    operation: "read_trace",
    request_id: requestId(),
    session_id: sessionId,
    after_event_id: null,
  });
  if (response.operation !== "read_trace") {
    throw new ExperienceApiError(
      "Inspect returned the wrong result.",
      "unexpected_operation",
      false,
    );
  }
  return response;
}
