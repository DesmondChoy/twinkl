import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { canonicalInspectFixture } from "./inspectFixture";
import { createExperienceState } from "./session";
import useNorthStarReview from "./useNorthStarReview";
import { ExperienceApiError } from "./experienceApi";
import { northStarProfileRef } from "./northStar";

const api = vi.hoisted(() => ({ createExperienceSession: vi.fn(), reviewNorthStar: vi.fn(), readExperienceTrace: vi.fn() }));
vi.mock("./experienceApi", async (importOriginal) => ({
  ...await importOriginal<typeof import("./experienceApi")>(), ...api,
}));
const profile = canonicalInspectFixture.session.profile;
const experience = { ...createExperienceState(), ...canonicalInspectFixture.session,
  trace_events: canonicalInspectFixture.trace_events,
};

beforeEach(() => {
  vi.resetAllMocks();
  api.reviewNorthStar.mockResolvedValue({ session: canonicalInspectFixture.session });
  api.createExperienceSession.mockResolvedValue({ session: canonicalInspectFixture.session });
  api.readExperienceTrace.mockResolvedValue({ session_id: profile.session_id, events: [] });
});

describe("independent North Star Moment lifecycle", () => {
  it("requests one review after weekly completion and updates only trace state", async () => {
    const updateExperience = vi.fn();
    const { rerender } = renderHook((busy) => useNorthStarReview({
      profile, experience, updateExperience, enabled: true, busy,
    }), { initialProps: true });
    expect(api.reviewNorthStar).not.toHaveBeenCalled();
    rerender(false);
    await waitFor(() => expect(updateExperience).toHaveBeenCalledTimes(1));
    expect(api.reviewNorthStar).toHaveBeenCalledWith({
      sessionId: profile.session_id, expectedRevision: experience.revision,
      weekStart: experience.weekly_digest?.week_start, retry: false,
    });
    expect(api.createExperienceSession).not.toHaveBeenCalled();
    expect(Object.keys(updateExperience.mock.calls[0][0]).sort())
      .toEqual(["revision", "trace_event_ids", "trace_events"]);
    rerender(false);
    expect(api.reviewNorthStar).toHaveBeenCalledTimes(1);
  });

  it("makes no live request for saved replay", async () => {
    renderHook(() => useNorthStarReview({ profile, experience, updateExperience: vi.fn(), enabled: false, busy: false }));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 20)); });
    expect(api.reviewNorthStar).not.toHaveBeenCalled();
  });

  it.each(["complete", "failed", "not_eligible", "pending"])(
    "resumes pending work but does not automatically rerun a %s record",
    async (status) => {
      const record = {
        schema_version: "north-star-record-v1", session_id: profile.session_id,
        owner_id: profile.user_id, profile_ref: await northStarProfileRef(profile),
        week_start: experience.weekly_digest?.week_start,
        week_end: experience.weekly_digest?.week_end,
        input_hash: "c".repeat(64), status, selected: null, sources: [],
        retryable: status === "failed",
      };
      const event = { ...canonicalInspectFixture.trace_events[0],
        event_id: "north-star-current", event_type: "north_star_reviewed",
        input_hash: record.input_hash, details: { record },
      };
      const current = { ...experience, trace_events: [...experience.trace_events, event] };
      renderHook(() => useNorthStarReview({ profile, experience: current, updateExperience: vi.fn(), enabled: true, busy: false }));
      if (status === "pending") {
        await waitFor(() => expect(api.reviewNorthStar).toHaveBeenCalledTimes(1));
      } else {
        await act(async () => { await new Promise((resolve) => setTimeout(resolve, 20)); });
        expect(api.reviewNorthStar).not.toHaveBeenCalled();
      }
    },
  );

  it("restores complete browser-held state only when the Python session is missing", async () => {
    api.reviewNorthStar.mockRejectedValueOnce(new ExperienceApiError("Missing", "session_not_found", false));
    const current = { ...experience,
      trace_event_ids: experience.trace_events.map((event) => event.event_id),
    };
    const updateExperience = vi.fn();
    renderHook(() => useNorthStarReview({ profile, experience: current, updateExperience, enabled: true, busy: false }));
    await waitFor(() => expect(updateExperience).toHaveBeenCalledTimes(1));
    expect(api.reviewNorthStar).toHaveBeenCalledTimes(2);
    expect(api.createExperienceSession.mock.calls[0][1]).toMatchObject({
      session_id: profile.session_id, revision: experience.revision,
      trace_events: experience.trace_events,
    });
  });

  it("retains the weekly result and bounds explicit retries after a network error", async () => {
    api.reviewNorthStar.mockRejectedValue(new ExperienceApiError("Unavailable"));
    const updateExperience = vi.fn();
    const { result } = renderHook(() => useNorthStarReview({ profile, experience, updateExperience, enabled: true, busy: false }));
    await waitFor(() => expect(result.current.failed).toBe(true));
    expect(updateExperience).not.toHaveBeenCalled();
    expect(api.reviewNorthStar).toHaveBeenCalledTimes(1);
    act(() => result.current.retry());
    await waitFor(() => expect(api.reviewNorthStar).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(result.current.retryable).toBe(false));
    expect(api.reviewNorthStar.mock.calls[1][0].retry).toBe(true);
  });

  it("discards a late review response after Journal Entry state changes", async () => {
    let resolve!: (value: unknown) => void;
    api.reviewNorthStar.mockReturnValue(new Promise((done) => { resolve = done; }));
    const updateExperience = vi.fn();
    const { rerender } = renderHook((current) => useNorthStarReview({
      profile, experience: current, updateExperience, enabled: true, busy: false,
    }), { initialProps: experience });
    await waitFor(() => expect(api.reviewNorthStar).toHaveBeenCalledTimes(1));
    rerender({ ...experience, revision: experience.revision + 1, weekly_digest: null });
    await act(async () => { resolve({ session: canonicalInspectFixture.session }); });
    expect(updateExperience).not.toHaveBeenCalled();
    expect(api.readExperienceTrace).not.toHaveBeenCalled();
  });

  it("retrieves the result after a concurrent date change discarded the first response", async () => {
    let resolve!: (value: unknown) => void;
    api.reviewNorthStar.mockReturnValueOnce(new Promise((done) => { resolve = done; }));
    const updateExperience = vi.fn();
    const { rerender } = renderHook(({ current, busy }) => useNorthStarReview({
      profile, experience: current, updateExperience, enabled: true, busy,
    }), { initialProps: { current: experience, busy: false } });
    await waitFor(() => expect(api.reviewNorthStar).toHaveBeenCalledTimes(1));
    rerender({ current: experience, busy: true });
    await act(async () => { resolve({ session: canonicalInspectFixture.session }); });
    expect(updateExperience).not.toHaveBeenCalled();
    const advanced = { ...experience, revision: experience.revision + 1 };
    api.reviewNorthStar.mockResolvedValue({ session: { ...canonicalInspectFixture.session, revision: advanced.revision } });
    rerender({ current: advanced, busy: false });
    await waitFor(() => expect(updateExperience).toHaveBeenCalledTimes(1));
    expect(api.reviewNorthStar).toHaveBeenCalledTimes(2);
    expect(api.reviewNorthStar.mock.calls[1][0].expectedRevision).toBe(advanced.revision);
  });

  it("retrieves a discarded response when a failed mutation leaves the same revision", async () => {
    let resolve!: (value: unknown) => void;
    api.reviewNorthStar.mockReturnValueOnce(new Promise((done) => { resolve = done; }));
    const updateExperience = vi.fn();
    const { rerender } = renderHook((busy) => useNorthStarReview({
      profile, experience, updateExperience, enabled: true, busy,
    }), { initialProps: false });
    await waitFor(() => expect(api.reviewNorthStar).toHaveBeenCalledTimes(1));
    rerender(true);
    await act(async () => { resolve({ session: canonicalInspectFixture.session }); });
    rerender(false);
    await waitFor(() => expect(updateExperience).toHaveBeenCalledTimes(1));
    expect(api.reviewNorthStar).toHaveBeenCalledTimes(2);
  });

  it("does not restore or publish the session after deletion disables the review", async () => {
    let reject!: (reason: unknown) => void;
    api.reviewNorthStar.mockReturnValueOnce(new Promise((_resolve, fail) => { reject = fail; }));
    const current = { ...experience, trace_event_ids: experience.trace_events.map((event) => event.event_id) };
    const updateExperience = vi.fn();
    const { rerender } = renderHook((enabled) => useNorthStarReview({
      profile, experience: current, updateExperience, enabled, busy: false,
    }), { initialProps: true });
    await waitFor(() => expect(api.reviewNorthStar).toHaveBeenCalledTimes(1));
    rerender(false);
    await act(async () => { reject(new ExperienceApiError("Missing", "session_not_found", false)); });
    expect(api.createExperienceSession).not.toHaveBeenCalled();
    expect(updateExperience).not.toHaveBeenCalled();
    expect(api.reviewNorthStar).toHaveBeenCalledTimes(1);
  });
});
