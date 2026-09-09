import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import CoachDigestCard from "./CoachDigestCard";
import WeeklyExperience from "./WeeklyExperience";
import InspectView from "./InspectView";
import ReplayTimeline from "./ReplayTimeline";
import { canonicalInspectFixture } from "./inspectFixture";
import { northStarFraming, northStarProfileRef, type NorthStarRecord } from "./northStar";
import type { TraceEventContract } from "./demoContracts";

const profile = canonicalInspectFixture.session.profile;
const quote = "I left work on time and cooked dinner for my sister.";
const entry = {
  journal_entry_id: "supportive-entry", t_index: 0, date: "2026-07-01",
  content: `I had a busy day. ${quote}`, nudge_response: null,
};
const narrative = {
  weekly_mirror: "You described a demanding week at work.",
  tension_explanation: "Time at work competed with being there for people close to you.",
  reflective_question: "What would make room for the people closest to you next week?",
};
const digest = { week_start: "2026-07-06", week_end: "2026-07-12", coach_narrative: narrative };
const driftResult = {
  delivery_state: "active_drift", core_value_states: { benevolence: "active_drift" },
  drifts: [{ core_value: "benevolence", onset_t_index: 2, onset_date: "2026-07-07", termination_reason: null }],
};

async function momentEvent(overrides: Partial<NorthStarRecord> = {}): Promise<TraceEventContract> {
  const record: NorthStarRecord = {
    schema_version: "north-star-record-v1", session_id: profile.session_id,
    owner_id: profile.user_id, profile_ref: await northStarProfileRef(profile),
    week_start: digest.week_start, week_end: digest.week_end,
    cutoff_at: "2026-07-13T00:00:00Z", input_hash: "a".repeat(64),
    status: "complete", mode: "reflection", reason: "supportive_action_selected",
    core_value: "benevolence", value_phrase: "Being there for the people closest to me",
    selected: { entry_id: entry.journal_entry_id, t_index: entry.t_index, date: entry.date, quote_source: "journal_entry", evidence_quote: quote },
    sources: [{ owner_id: profile.user_id, entry_id: entry.journal_entry_id, t_index: entry.t_index, date: entry.date,
      journal_entry: entry.content, nudge_response: null, available_at: "2026-07-01T18:00:00Z", response_available_at: null }],
    reviews: [{ decision: "supportive", reason_code: "observable_choice" }], attempts: 1, retryable: false,
    onset_t_index: 2, onset_date: "2026-07-07", onset_available_at: "2026-07-07T18:00:00Z",
    ...overrides,
  };
  return {
    ...canonicalInspectFixture.trace_events[0], event_id: "moment-current",
    event_type: "north_star_reviewed", session_id: profile.session_id,
    input_hash: record.input_hash, details: { record },
  };
}

function card(event: TraceEventContract, extra: Partial<Parameters<typeof CoachDigestCard>[0]> = {}) {
  return <CoachDigestCard weeklyDigest={digest} headingId="coach-title" journalEntries={[entry]}
    northStar={{ profile, driftResult, traceEvents: [event] }} {...extra} />;
}

describe("Coach Digest composition", () => {
  it.each(["personal", "demo"] as const)("keeps the %s moment inside the response and the existing question last", async (presentation) => {
    const user = userEvent.setup();
    const event = await momentEvent();
    const onOpenEntry = vi.fn();
    const inspectMoment = vi.fn();
    const original = JSON.stringify({ digest, event });
    render(card(event, {
      onOpenEntry,
      northStar: { profile, driftResult, traceEvents: [event], presentation, inspectMoment },
    }));
    const quotation = await screen.findByText(quote);
    const coach = screen.getByRole("complementary", { name: "Your weekly reflection" });
    const introduction = screen.getByText(northStarFraming("reflection")!);
    const question = within(coach).getByText(narrative.reflective_question);
    expect(document.querySelectorAll(".coach-digest")).toHaveLength(1);
    expect(coach.contains(quotation)).toBe(true);
    expect(within(coach).getByText(narrative.weekly_mirror)).toBeTruthy();
    const tension = within(coach).getByText(narrative.tension_explanation);
    expect(tension.compareDocumentPosition(introduction) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(introduction.compareDocumentPosition(quotation) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(quotation.compareDocumentPosition(question) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(coach.lastElementChild).toBe(question);
    expect(coach.querySelectorAll(".coach-digest__question")).toHaveLength(1);
    expect(quotation.textContent).toBe(quote);
    expect(within(coach).getByRole("heading", { level: 3, name: "A past moment in your own words" })).toBeTruthy();
    if (presentation === "demo") {
      expect(within(coach).getByText("North Star Moment")).toBeTruthy();
    } else {
      expect(within(coach).queryByText(/North Star Moment/)).toBeNull();
    }
    await user.click(within(coach).getByRole("button", { name: "Inspect this moment" }));
    expect(inspectMoment).toHaveBeenCalledWith(event.event_id);
    await user.click(within(coach).getByRole("link", { name: /^Open Journal Entry/ }));
    expect(onOpenEntry).toHaveBeenCalledWith(entry);
    expect(JSON.stringify({ digest, event })).toBe(original);
  });

  it.each([null, {}, { ...narrative, reflective_question: " " }])(
    "omits the whole composition when the Coach Digest is unavailable or incomplete: %j",
    async (coachNarrative) => {
      const event = await momentEvent();
      render(card(event, { weeklyDigest: { ...digest, coach_narrative: coachNarrative } }));
      await act(async () => {});
      expect(document.querySelector(".coach-digest")).toBeNull();
      expect(document.querySelector(".north-star-moment")).toBeNull();
    },
  );

  it.each(["failed", "pending", "not_eligible"] as const)(
    "keeps the response and question when the newest moment is %s",
    async (status) => {
      const older = await momentEvent();
      const { rerender } = render(card(older));
      await screen.findByText(quote);
      const newer = { ...await momentEvent({ status, selected: null }), event_id: "newer-moment" };
      rerender(card(newer, { northStar: { profile, driftResult, traceEvents: [older, newer] } }));
      expect(screen.queryByText(quote)).toBeNull();
      expect(screen.getByText(narrative.weekly_mirror)).toBeTruthy();
      expect(screen.getByText(narrative.reflective_question)).toBeTruthy();
      expect(document.querySelector(".north-star-status")).toBeNull();
    },
  );

  it("does not fall back to an older quotation after a newer review fails source binding", async () => {
    const older = await momentEvent();
    const { rerender } = render(card(older));
    await screen.findByText(quote);
    const newer = { ...await momentEvent(), event_id: "newer-moment" };
    const record = newer.details.record as NorthStarRecord;
    record.sources[0].journal_entry = "A different version of this Journal Entry.";
    rerender(card(newer, { northStar: { profile, driftResult, traceEvents: [older, newer] } }));
    expect(screen.queryByText(quote)).toBeNull();
    expect(screen.getByText(narrative.reflective_question)).toBeTruthy();
  });

  it("waits for a new moment review after the same week's digest is rebuilt", async () => {
    const older = await momentEvent();
    const { rerender } = render(card(older));
    await screen.findByText(quote);
    const rebuiltDigest: TraceEventContract = {
      ...canonicalInspectFixture.trace_events[0], event_type: "weekly_digest_built",
      event_id: "rebuilt-digest", session_id: profile.session_id, details: { digest },
    };
    rerender(card(older, { northStar: { profile, driftResult, traceEvents: [older, rebuiltDigest] } }));
    expect(screen.queryByText(quote)).toBeNull();
    expect(screen.getByText(narrative.reflective_question)).toBeTruthy();
    const current = { ...await momentEvent(), event_id: "review-after-rebuild" };
    rerender(card(current, { northStar: { profile, driftResult, traceEvents: [older, rebuiltDigest, current] } }));
    expect(await screen.findByText(quote)).toBeTruthy();
  });

  it.each(["profile", "week", "source", "drift"] as const)(
    "removes a displayed quotation immediately when its %s changes",
    async (change) => {
      const event = await momentEvent();
      const { rerender } = render(card(event));
      await screen.findByText(quote);
      rerender(card(event, {
        weeklyDigest: change === "week" ? { ...digest, week_start: "2026-07-13", week_end: "2026-07-19" } : digest,
        journalEntries: change === "source" ? [{ ...entry, content: "Changed entry text." }] : [entry],
        northStar: {
          profile: change === "profile" ? { ...profile, preferred_name: "A changed Profile" } : profile,
          driftResult: change === "drift" ? { delivery_state: "insufficient_evidence" } : driftResult,
          traceEvents: [event],
        },
      }));
      expect(screen.queryByText(quote)).toBeNull();
      expect(screen.getByText(narrative.reflective_question)).toBeTruthy();
    },
  );

  it("keeps the personal response while showing moment review progress and recovery", async () => {
    const event = await momentEvent();
    const selectJournalEntry = vi.fn();
    const props = {
      profile, journalEntries: [entry], weeklyReviewerDecisions: [], driftResult, weeklyDigest: digest,
      traceEvents: [event], inspectRun: vi.fn(), selectJournalEntry,
    };
    const { rerender } = render(<WeeklyExperience {...props} />);
    const quotation = await screen.findByText(quote);
    expect(quotation.closest(".coach-digest")).not.toBeNull();
    expect(screen.queryByText(/North Star Moment/)).toBeNull();
    await userEvent.click(screen.getByRole("link", { name: /^Open Journal Entry/ }));
    expect(selectJournalEntry).toHaveBeenCalledWith(entry.journal_entry_id);

    for (const pending of [true, false]) {
      rerender(<WeeklyExperience {...props} northStarReview={{
        pending, failed: !pending, retryable: true, retry: vi.fn(),
      }} />);
      expect(screen.queryByText(quote)).toBeNull();
      expect(screen.getByText(narrative.reflective_question)).toBeTruthy();
      expect(screen.queryByText(/North Star Moment/)).toBeNull();
      expect(Boolean(screen.queryByRole("button", { name: "Retry moment review" }))).toBe(!pending);
    }
  });

  it("Inspect preserves the selected event's exact quote, framing, and source checks", async () => {
    const user = userEvent.setup();
    const selected = await momentEvent();
    const later = { ...await momentEvent({ mode: "reminder" }), event_id: "later-moment" };
    render(<InspectView events={[selected, later]} currentWeekEventIds={[selected.event_id, later.event_id]}
      selectedEventId={selected.event_id} traceLabel="Saved Persona replay" onReturn={() => undefined} />);
    const details = screen.getByTestId(`trace-details-${selected.event_id}`);
    expect(screen.getByLabelText("Event 1: North Star Moment reviewed").getAttribute("aria-current")).toBe("true");
    expect(within(details).getByLabelText("Deterministic introduction").textContent).toBe(northStarFraming("reflection"));
    expect(within(details).getByLabelText("Exact selected quotation").textContent).toBe(quote);
    expect(within(details).getByText(entry.journal_entry_id)).toBeTruthy();
    await user.click(within(details).getByText("Source checks and AI assessment"));
    expect(within(details).getByLabelText("AI assessment").textContent).toContain("observable_choice");
    expect(within(details).getByLabelText("Source checks").textContent).toContain("2026-07-07T18:00:00Z");
    expect(within(details).getByText(/not human validation/)).toBeTruthy();
  });

  it("opens a quoted Nudge response in the replay source drawer and restores focus on close", async () => {
    const user = userEvent.setup();
    const event = await momentEvent();
    const record = event.details.record as NorthStarRecord;
    record.selected!.quote_source = "nudge_response";
    const sourceEntry = { ...entry, content: "I wanted to record a busy day.", nudge_response: quote };
    record.sources[0] = { ...record.sources[0], journal_entry: sourceEntry.content,
      nudge_response: quote, response_available_at: "2026-07-01T19:00:00Z" };
    render(<ReplayTimeline profile={profile} week={{
      week_id: "current-week", week_start: digest.week_start, week_end: digest.week_end,
      journal_entry_ids: [], event_ids: [event.event_id], expected_delivery_state: "active_drift",
    }} journalEntries={[]} nudges={[]} reviewedJournalEntries={[sourceEntry]}
      weeklyReviewerDecisions={[]} reviewTraceEvents={[event]} selectedJournalEntryId={null}
      cumulativeEntryCount={1} resultVisible onRevealResult={vi.fn()} driftResult={driftResult}
      weeklyDigest={digest} inspectRun={vi.fn()} inspectEventId={event.event_id}
      onSelectJournalEntry={vi.fn()} />);
    const sourceLink = await screen.findByRole("link", { name: /^Open response in Journal Entry/ });
    await user.click(sourceLink);
    const drawer = screen.getByRole("dialog");
    expect(within(drawer).getByText(sourceEntry.content)).toBeTruthy();
    expect(within(drawer).getByRole("heading", { name: "Response to the Nudge" })).toBeTruthy();
    expect(within(drawer).getByText(quote).textContent).toBe(quote);
    expect(screen.queryByRole("heading", { name: "Journal Entries" })).toBeNull();
    await user.click(within(drawer).getByRole("button", { name: "Close Journal Entry" }));
    await waitFor(() => expect(document.activeElement).toBe(sourceLink));
    expect(sourceLink.closest(".coach-digest")).not.toBeNull();
  });
});
