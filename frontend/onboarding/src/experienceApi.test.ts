import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { canonicalInspectFixture } from "./inspectFixture";
import {
  createExperienceSession,
  ExperienceApiError,
  readExperienceTrace,
  submitJournalEntry,
} from "./experienceApi";

const profile = canonicalInspectFixture.session.profile;
const entry = canonicalInspectFixture.session.journal_entries[0];
const fetchMock = vi.fn();

function jsonResponse(value: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => value,
  } as Response;
}

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Experience API client", () => {
  it("sends browser-held state when creating a resumable session", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        schema_version: "experience-inspect-v1",
        operation: "create_session",
        request_id: "create-response",
        status: "ok",
        session: canonicalInspectFixture.session,
      }),
    );
    const resumeState = {
      session_id: profile.session_id,
      revision: canonicalInspectFixture.session.revision,
      journal_entries: canonicalInspectFixture.session.journal_entries,
      nudges: canonicalInspectFixture.session.nudges,
      trace_events: canonicalInspectFixture.trace_events,
    };

    const response = await createExperienceSession(profile, resumeState);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const request = JSON.parse(String(init.body));

    expect(response.operation).toBe("create_session");
    expect(request.operation).toBe("create_session");
    expect(request.resume_state).toEqual(resumeState);
    expect(request.idempotency_key).toMatch(/^[0-9a-f]{64}$/);
  });

  it("maps safe API errors without exposing an invalid success response", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        schema_version: "experience-inspect-v1",
        operation: "error",
        requested_operation: "submit_journal_entry",
        request_id: "submit-response",
        status: "error",
        error: {
          code: "journal_order_conflict",
          message: "The Journal Entry is based on an older session revision.",
          retryable: false,
        },
      }),
    );

    await expect(
      submitJournalEntry({
        sessionId: profile.session_id,
        expectedRevision: 2,
        entry,
        idempotencyKey: "a".repeat(64),
      }),
    ).rejects.toEqual(
      expect.objectContaining<Partial<ExperienceApiError>>({
        code: "journal_order_conflict",
        retryable: false,
      }),
    );
  });

  it("reports an HTTP routing failure without blaming the nudge check", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({}, 405));

    await expect(
      createExperienceSession(profile),
    ).rejects.toEqual(
      expect.objectContaining<Partial<ExperienceApiError>>({
        code: "http_405",
        message: "The Experience service did not accept this request.",
        retryable: false,
      }),
    );
  });

  it("validates linked trace responses and sends a null initial cursor", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        schema_version: "experience-inspect-v1",
        operation: "read_trace",
        request_id: "trace-response",
        status: "ok",
        session_id: profile.session_id,
        events: canonicalInspectFixture.trace_events,
      }),
    );

    const response = await readExperienceTrace(profile.session_id);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const request = JSON.parse(String(init.body));

    expect(response.events).toHaveLength(
      canonicalInspectFixture.trace_events.length,
    );
    expect(request).toMatchObject({
      operation: "read_trace",
      session_id: profile.session_id,
      after_event_id: null,
    });
  });
});
