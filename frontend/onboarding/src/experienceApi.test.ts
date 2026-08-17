import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { canonicalInspectFixture } from "./inspectFixture";
import {
  advanceAssessmentTime,
  buildProfileReselectionResumeState,
  createExperienceSession,
  deleteExperienceSession,
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
  it("rebuilds a factual resume state after Core Value reselection", async () => {
    const resumeState = await buildProfileReselectionResumeState(
      profile,
      [entry],
      canonicalInspectFixture.session.assessment_clock,
    );

    expect(resumeState).toMatchObject({
      session_id: profile.session_id,
      revision: 1,
      journal_entries: [entry],
      nudges: [],
      assessment_clock: canonicalInspectFixture.session.assessment_clock,
    });
    expect(resumeState.trace_events).toHaveLength(2);
    expect(resumeState.trace_events[0]).toMatchObject({
      event_type: "profile_confirmed",
      parent_event_id: null,
      details: { profile },
    });
    expect(resumeState.trace_events[1]).toMatchObject({
      event_type: "journal_entry_submitted",
      parent_event_id: resumeState.trace_events[0].event_id,
      details: { journal_entry: entry, ordering_valid: true },
    });
    expect(
      resumeState.trace_events.every((event) =>
        /^[0-9a-f]{64}$/.test(event.input_hash)),
    ).toBe(true);
  });

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
      assessment_clock: canonicalInspectFixture.session.assessment_clock,
      trace_events: canonicalInspectFixture.trace_events,
    };

    const response = await createExperienceSession(profile, resumeState);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const request = JSON.parse(String(init.body));

    expect(response.operation).toBe("create_session");
    expect(request.operation).toBe("create_session");
    expect(request.assessment_timezone).toBe(
      Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC",
    );
    expect(request.resume_state).toEqual(resumeState);
    expect(request.idempotency_key).toMatch(/^[0-9a-f]{64}$/);
  });

  it("advances Simulated time with a stable retry key", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        schema_version: "experience-inspect-v1",
        operation: "advance_assessment_time",
        request_id: "advance-response",
        status: "ok",
        session: canonicalInspectFixture.session,
        event_ids: ["event-15"],
      }),
    );

    const response = await advanceAssessmentTime({
      sessionId: profile.session_id,
      expectedRevision: canonicalInspectFixture.session.revision,
      action: "next_day",
    });
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const request = JSON.parse(String(init.body));

    expect(response.operation).toBe("advance_assessment_time");
    expect(request).toMatchObject({
      operation: "advance_assessment_time",
      session_id: profile.session_id,
      expected_revision: canonicalInspectFixture.session.revision,
      action: "next_day",
    });
    expect(request.idempotency_key).toMatch(/^[0-9a-f]{64}$/);
  });

  it("deletes one Experience session", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        schema_version: "experience-inspect-v1",
        operation: "delete_session",
        request_id: "delete-response",
        status: "ok",
        session_id: profile.session_id,
        deleted: true,
      }),
    );

    const response = await deleteExperienceSession(profile.session_id);
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const request = JSON.parse(String(init.body));

    expect(response).toMatchObject({
      operation: "delete_session",
      session_id: profile.session_id,
      deleted: true,
    });
    expect(request).toMatchObject({
      operation: "delete_session",
      session_id: profile.session_id,
    });
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
