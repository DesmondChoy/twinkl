import type { OnboardingProfile } from "./domain";
import {
  EXPERIENCE_INSPECT_CONTRACT_VERSION,
  type ExperienceApiResponseContract,
  type ExperienceResumeStateContract,
  type JournalEntryContract,
  type JournalEntrySubmittedResponseContract,
  type SessionCreatedResponseContract,
  type TraceReadResponseContract,
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

async function sha256(value: unknown): Promise<string> {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
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
      "The Journal Entry is safe here, but the nudge check could not start.",
    );
  }

  let parsed: ExperienceApiResponseContract;
  try {
    parsed = validateExperienceApiResponse(await response.json());
  } catch {
    throw new ExperienceApiError(
      "The Journal Entry is safe here, but the nudge check returned an unreadable result.",
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
): Promise<SessionCreatedResponseContract> {
  const idempotencyKey = await sha256({
    operation: "create_session",
    profile,
    resume_state: resumeState,
  });
  const response = await postExperience({
    schema_version: EXPERIENCE_INSPECT_CONTRACT_VERSION,
    operation: "create_session",
    request_id: requestId(),
    idempotency_key: idempotencyKey,
    profile,
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
