import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import WeeklyExperience from "./WeeklyExperience";
import { canonicalInspectFixture } from "./inspectFixture";
import { northStarProfileRef, type NorthStarRecord } from "./northStar";
import type { JournalEntryContract, TraceEventContract, WeeklyDriftReviewerDecisionContract } from "./demoContracts";
import type { OnboardingProfile } from "./domain";

const profile: OnboardingProfile = {
  ...canonicalInspectFixture.session.profile, top_values: ["benevolence"],
};
const entries: JournalEntryContract[] = [
  { journal_entry_id: "neutral", t_index: 0, date: "2026-06-30", content: "I took a walk after work.", nudge_response: null },
  { journal_entry_id: "supportive", t_index: 1, date: "2026-07-01", content: "I cooked dinner for my sister.", nudge_response: null },
  { journal_entry_id: "onset", t_index: 2, date: "2026-07-07", content: "I ignored my sister's call to finish work.", nudge_response: null },
  { journal_entry_id: "confirmation", t_index: 3, date: "2026-07-08", content: "I cancelled our dinner to work late again.", nudge_response: null },
  { journal_entry_id: "continued", t_index: 4, date: "2026-07-09", content: "I missed another family meal for work.", nudge_response: null },
];
const narrative = {
  weekly_mirror: "Work took time from being with your family.",
  tension_explanation: "This repeated across several days.",
  reflective_question: "What stands out when you read these moments together?",
};
const digest = {
  week_start: "2026-07-06", week_end: "2026-07-12", coach_narrative: narrative,
  evidence: [entries[0], entries[1], entries[4]].map((entry) => ({
    date: entry.date, t_index: entry.t_index, excerpt: entry.content, dimensions: ["benevolence"],
  })),
};
const driftResult = {
  delivery_state: "active_drift", core_value_states: { benevolence: "active_drift" },
  drifts: [{ core_value: "benevolence", onset_t_index: 2, confirmation_t_index: 3,
    onset_date: entries[2].date, end_t_index: 4, termination_reason: null }],
};
const decisions: WeeklyDriftReviewerDecisionContract[] = entries.map((entry) => ({
  persona_id: profile.user_id, week_start: entry.t_index < 2 ? "2026-06-29" : digest.week_start,
  week_end: entry.t_index < 2 ? "2026-07-05" : digest.week_end,
  t_index: entry.t_index, date: entry.date, core_value: "benevolence",
  verdict: entry.t_index < 2 ? "not_conflict" : "conflict", confidence: "high",
  reason_code: null, evidence_quote: entry.content, review_status: "ok",
}));
const props = {
  profile, journalEntries: entries, weeklyReviewerDecisions: decisions,
  driftResult, weeklyDigest: digest, traceEvents: [], inspectRun: vi.fn(),
};

async function momentEvent(overrides: Partial<NorthStarRecord> = {}): Promise<TraceEventContract> {
  const entry = entries[1];
  const record: NorthStarRecord = {
    schema_version: "north-star-record-v1", session_id: profile.session_id,
    owner_id: profile.user_id, profile_ref: await northStarProfileRef(profile),
    week_start: digest.week_start, week_end: digest.week_end,
    cutoff_at: "2026-07-13T00:00:00Z", input_hash: "a".repeat(64),
    status: "complete", mode: "reflection", reason: "supportive_action_selected",
    core_value: "benevolence", value_phrase: "Being there for the people closest to me",
    selected: { entry_id: entry.journal_entry_id, t_index: entry.t_index, date: entry.date,
      quote_source: "journal_entry", evidence_quote: entry.content },
    sources: [{ owner_id: profile.user_id, entry_id: entry.journal_entry_id, t_index: entry.t_index,
      date: entry.date, journal_entry: entry.content, nudge_response: null,
      available_at: "2026-07-01T18:00:00Z", response_available_at: null }],
    onset_t_index: 2, onset_date: entries[2].date, onset_available_at: "2026-07-07T18:00:00Z",
    reviews: [], attempts: 1, retryable: false, ...overrides,
  };
  return {
    ...canonicalInspectFixture.trace_events[0], event_id: "current-moment",
    event_type: "north_star_reviewed", session_id: profile.session_id,
    input_hash: record.input_hash, details: { record },
  };
}

describe("manual weekly response", () => {
  it("places the triggering conflict pair and current conflict before separate background context", async () => {
    const user = userEvent.setup();
    const selectJournalEntry = vi.fn();
    render(<WeeklyExperience {...props} selectJournalEntry={selectJournalEntry} />);
    const conflict = screen.getByRole("region", { name: "Journal Entries behind this Drift" });
    const context = screen.getByRole("region", { name: "Other Journal Entry context" });
    expect(within(conflict).getAllByRole("link").map((link) => link.getAttribute("href")))
      .toEqual(["#journal-entry-onset", "#journal-entry-confirmation", "#journal-entry-continued"]);
    expect(conflict.compareDocumentPosition(context) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(within(context).getByText(entries[0].content)).toBeTruthy();
    expect(within(context).getByText(entries[1].content)).toBeTruthy();
    expect(context.textContent).not.toMatch(/supportive|aligned|Not Conflict/);
    expect(within(context).getByText("These entries provide context; they do not establish this Drift.")).toBeTruthy();
    expect(screen.getByText("Selecting evidence opens and focuses its Journal Entry.")).toBeTruthy();
    await user.click(within(conflict).getAllByRole("link")[0]);
    expect(selectJournalEntry).toHaveBeenCalledWith(entries[2].journal_entry_id);
  });

  it("does not promote failed reviews or later entries into the current conflict evidence", () => {
    render(<WeeklyExperience {...props} weeklyReviewerDecisions={[
      ...decisions.map((decision) => decision.t_index === 4
        ? { ...decision, review_status: "error" as const } : decision),
      { ...decisions[4], t_index: 5, date: "2026-07-13" },
    ]} journalEntries={[...entries, { ...entries[4], journal_entry_id: "future", t_index: 5, date: "2026-07-13" }]} />);
    const conflict = screen.getByRole("region", { name: "Journal Entries behind this Drift" });
    expect(within(conflict).getAllByRole("link")).toHaveLength(2);
    expect(within(conflict).queryByText(entries[4].content)).toBeNull();
  });

  it("offers contextual Inspect for a verified personal moment", async () => {
    const user = userEvent.setup();
    const inspectRun = vi.fn();
    const event = await momentEvent();
    render(<WeeklyExperience {...props} traceEvents={[event]} inspectRun={inspectRun} />);
    const inspect = await screen.findByRole("button", { name: "Inspect this moment" });
    expect(screen.getByRole("heading", { name: "A past moment in your own words" })).toBeTruthy();
    await user.click(inspect);
    expect(inspectRun).toHaveBeenCalledWith(event.event_id);
  });

  it.each(["pending", "failed"])("explains %s moment review while preserving the weekly result", async (state) => {
    const user = userEvent.setup();
    const retry = vi.fn();
    const event = await momentEvent();
    const { rerender } = render(<WeeklyExperience {...props} traceEvents={[event]} />);
    await screen.findByRole("heading", { name: "A past moment in your own words" });
    rerender(<WeeklyExperience {...props} traceEvents={[event]} northStarReview={{
      pending: state === "pending", failed: state === "failed", retryable: true, retry,
    }} />);
    expect(screen.queryByRole("heading", { name: "A past moment in your own words" })).toBeNull();
    expect(screen.getByRole("heading", { name: "A repeated conflict surfaced." })).toBeTruthy();
    expect(screen.getByText(narrative.reflective_question)).toBeTruthy();
    const status = screen.getByRole("region", { name: "Moment review" });
    expect(status.textContent).toContain(state === "pending" ? "Looking for a moment" : "could not be prepared");
    if (state === "failed") {
      await user.click(within(status).getByRole("button", { name: "Retry moment review" }));
      expect(retry).toHaveBeenCalledOnce();
    } else expect(within(status).queryByRole("button")).toBeNull();
  });

  it.each([
    ["pending", "awaiting_ai_review", "Looking for a moment in your writing"],
    ["failed", "budget_unavailable", "A moment from your writing could not be prepared"],
    ["not_eligible", "no_eligible_writing", "No eligible writing was available"],
    ["not_eligible", "insufficient_evidence", "A moment is not shown while this week's evidence is insufficient"],
    ["complete", "no_supportive_source", "The review did not identify a supportive action"],
  ] as const)("explains the recorded %s / %s outcome without showing an old quote", async (status, reason, message) => {
    const oldEvent = await momentEvent();
    const latest = { ...await momentEvent({ status, reason, selected: null }), event_id: "no-moment" };
    render(<WeeklyExperience {...props} traceEvents={[oldEvent, latest]} />);
    expect(await screen.findByText(new RegExp(message))).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "A past moment in your own words" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Retry moment review" })).toBeNull();
    expect(screen.getByText(narrative.reflective_question)).toBeTruthy();
  });

  it("removes the old moment when a newer review fails source binding", async () => {
    const older = await momentEvent();
    const { rerender } = render(<WeeklyExperience {...props} traceEvents={[older]} />);
    await screen.findByRole("heading", { name: "A past moment in your own words" });
    const newer = { ...await momentEvent(), event_id: "invalid-source" };
    (newer.details.record as NorthStarRecord).sources[0].journal_entry = "Changed writing.";
    rerender(<WeeklyExperience {...props} traceEvents={[older, newer]} />);
    expect(screen.queryByRole("heading", { name: "A past moment in your own words" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Inspect this moment" })).toBeNull();
  });

  it("keeps Drift visible when Coach Digest fails and calls only the supplied retry", async () => {
    const user = userEvent.setup();
    const retry = vi.fn();
    const failedDigest = { ...digest, coach_narrative: null };
    const { rerender } = render(<WeeklyExperience {...props} weeklyDigest={failedDigest}
      coachReview={{ pending: false, retryable: true, retry, error: "The response timed out." }} />);
    expect(screen.getByRole("heading", { name: "A repeated conflict surfaced." })).toBeTruthy();
    expect(screen.getByText("The response timed out.")).toBeTruthy();
    expect(screen.getByRole("status", { name: "Moment review status" }).textContent)
      .toBe("A moment from your writing can be reviewed after your weekly reflection is ready.");
    expect(screen.queryByRole("button", { name: "Retry moment review" })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Retry Coach Digest" }));
    expect(retry).toHaveBeenCalledOnce();
    rerender(<WeeklyExperience {...props} weeklyDigest={failedDigest}
      coachReview={{ pending: true, retryable: true, retry }} />);
    expect(screen.getByRole("heading", { name: "Preparing your weekly reflection…" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Retry Coach Digest" })).toBeNull();
    expect(screen.getByRole("heading", { name: "A repeated conflict surfaced." })).toBeTruthy();
  });

  it("keeps a valid Drift result visible without a digest and hides retry when unavailable", () => {
    render(<WeeklyExperience {...props} weeklyDigest={null}
      coachReview={{ pending: false, retryable: false, retry: vi.fn() }} />);
    expect(screen.getByRole("heading", { name: "A repeated conflict surfaced." })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Your weekly response could not be prepared." })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Retry Coach Digest" })).toBeNull();
  });

  it("does not replace a recorded moment outcome with a waiting-for-Coach message", async () => {
    const event = await momentEvent({ status: "complete", selected: null, reason: "no_supportive_source" });
    render(<WeeklyExperience {...props} weeklyDigest={{ ...digest, coach_narrative: null }} traceEvents={[event]} />);
    expect(await screen.findByText("The review did not identify a supportive action in the eligible writing.")).toBeTruthy();
    expect(screen.queryByText("A moment from your writing can be reviewed after your weekly reflection is ready.")).toBeNull();
  });

  it("does not display a waiting status before there is a Drift result", () => {
    const { container } = render(<WeeklyExperience {...props} driftResult={null} weeklyDigest={null} />);
    expect(container.textContent).toBe("");
  });
});
