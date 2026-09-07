import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import NorthStarMoment from "./NorthStarMoment";
import { canonicalInspectFixture } from "./inspectFixture";
import { northStarProfileRef, type NorthStarRecord } from "./northStar";
import type { TraceEventContract } from "./demoContracts";

const profile = canonicalInspectFixture.session.profile;
const quote = "I left work on time and cooked dinner for my sister.";
const entry = {
  journal_entry_id: "supportive-entry", t_index: 0, date: "2026-07-01",
  content: `I had a busy day. ${quote}`, nudge_response: null,
};
const digest = { week_start: "2026-07-06", week_end: "2026-07-12" };
const activeDrift = {
  delivery_state: "active_drift", core_value_states: { benevolence: "active_drift" },
  drifts: [{ core_value: "benevolence", onset_t_index: 2, onset_date: "2026-07-07", termination_reason: null }],
};

async function event(overrides: Partial<NorthStarRecord> = {}): Promise<TraceEventContract> {
  const record: NorthStarRecord = {
    schema_version: "north-star-record-v1", session_id: profile.session_id,
    owner_id: profile.user_id, profile_ref: await northStarProfileRef(profile),
    ...digest, cutoff_at: "2026-07-13T00:00:00Z", input_hash: "a".repeat(64),
    status: "complete", mode: "reflection", reason: "selected",
    core_value: "benevolence", value_phrase: "Being there for the people closest to me",
    selected: { entry_id: entry.journal_entry_id, t_index: entry.t_index, date: entry.date, quote_source: "journal_entry", evidence_quote: quote },
    sources: [{ owner_id: profile.user_id, entry_id: entry.journal_entry_id, t_index: entry.t_index, date: entry.date,
      journal_entry: entry.content, nudge_response: null, available_at: "2026-07-01T18:00:00Z", response_available_at: null }],
    reviews: [], attempts: 1, retryable: false,
    onset_t_index: 2, onset_date: "2026-07-07", onset_available_at: "2026-07-07T18:00:00Z",
    ...overrides,
  };
  return {
    ...canonicalInspectFixture.trace_events[0], event_id: "north-star-1",
    event_type: "north_star_reviewed", session_id: profile.session_id,
    input_hash: record.input_hash, details: { record },
  };
}

async function renderCard(overrides: Partial<NorthStarRecord> = {}, props: Record<string, unknown> = {}) {
  const traceEvent = await event(overrides);
  render(<NorthStarMoment profile={profile} journalEntries={[entry]} weeklyDigest={digest}
    driftResult={activeDrift} traceEvents={[traceEvent]} {...props} />);
  // Profile binding is asynchronous; wait for its own effect before checking omissions.
  await new Promise((resolve) => setTimeout(resolve, 20));
}

describe("North Star Moment", () => {
  it("shows the exact quotation immediately and links to its original Journal Entry", async () => {
    const user = userEvent.setup();
    const selectJournalEntry = vi.fn();
    await renderCard({}, { selectJournalEntry });
    expect(await screen.findByText("North Star Moment")).toBeTruthy();
    expect(screen.getByText(/earlier action expressed a priority/)).toBeTruthy();
    expect(screen.getByText(quote).tagName).toBe("BLOCKQUOTE");
    expect(screen.getByRole("heading", { name: "A past moment in your own words" })).toBeTruthy();
    await user.click(screen.getByRole("link", { name: /Open Journal Entry/ }));
    expect(selectJournalEntry).toHaveBeenCalledWith(entry.journal_entry_id);
  });

  it("expands long quotations without changing their exact text", async () => {
    const user = userEvent.setup();
    const longQuote = `${quote} `.repeat(10).trim();
    const eventWithLongQuote = await event();
    const record = eventWithLongQuote.details.record as NorthStarRecord;
    record.selected!.evidence_quote = longQuote;
    record.sources[0].journal_entry = longQuote;
    render(<NorthStarMoment profile={profile} journalEntries={[{ ...entry, content: longQuote }]}
      weeklyDigest={digest} driftResult={activeDrift} traceEvents={[eventWithLongQuote]} />);
    const button = await screen.findByRole("button", { name: "Expand quotation" });
    const blockquote = screen.getByText(longQuote);
    expect(blockquote.className).toContain("collapsed");
    expect(button.getAttribute("aria-controls")).toBe(blockquote.id);
    await user.click(button);
    expect(button.getAttribute("aria-expanded")).toBe("true");
    expect(blockquote.textContent).toBe(longQuote);
    expect(blockquote.className).not.toContain("collapsed");
  });

  it("encourages a verified current-week action without claiming overall alignment", async () => {
    await renderCard({ mode: "encouragement", week_start: "2026-06-29", week_end: "2026-07-05" }, {
      weeklyDigest: { week_start: "2026-06-29", week_end: "2026-07-05" },
      driftResult: { delivery_state: "no_active_drift", core_value_states: {} },
    });
    expect(await screen.findByText(/one way you put this priority into practice/)).toBeTruthy();
    expect(screen.queryByText("This earlier writing is a reference point for your Core Value.")).toBeNull();
  });

  it("frames an older example as a reminder when no conflict is active", async () => {
    await renderCard({ mode: "reminder" }, { driftResult: { delivery_state: "no_active_drift" } });
    expect(await screen.findByText(/earlier action is a reminder/)).toBeTruthy();
    expect(screen.queryByText(/one way you put this priority into practice/)).toBeNull();
  });

  it.each([
    { status: "failed" }, { status: "pending" }, { selected: null },
    { profile_ref: "b".repeat(64) }, { owner_id: "another-person" },
    { week_start: "2026-06-29" }, { cutoff_at: "2026-06-30T00:00:00Z" },
    { mode: "encouragement" }, { sources: [] },
  ] as Partial<NorthStarRecord>[])("omits unsafe or unavailable record %j", async (overrides) => {
    await renderCard(overrides);
    expect(screen.queryByText("North Star Moment")).toBeNull();
  });

  it("omits a quotation whose current source was changed", async () => {
    await renderCard({}, { journalEntries: [{ ...entry, content: "I stayed late instead." }] });
    expect(screen.queryByText("North Star Moment")).toBeNull();
  });

  it("omits a source at or after the current conflict onset", async () => {
    await renderCard({}, { driftResult: { ...activeDrift, drifts: [{
      core_value: "benevolence", onset_t_index: 0, onset_date: entry.date, termination_reason: null,
    }] } });
    expect(screen.queryByText("North Star Moment")).toBeNull();
  });

  it("keeps simulated Journal Entry dates separate from server availability timestamps", async () => {
    const selectedEvent = await event();
    const record = selectedEvent.details.record as NorthStarRecord;
    record.cutoff_at = "2026-06-03T00:00:00Z";
    record.onset_available_at = "2026-06-02T00:00:00Z";
    record.sources[0].available_at = "2026-06-01T00:00:00Z";
    render(<NorthStarMoment profile={profile} journalEntries={[entry]} weeklyDigest={digest}
      driftResult={activeDrift} traceEvents={[selectedEvent]} />);
    expect(await screen.findByText(quote)).toBeTruthy();
  });

  it("allows a same-day action recorded before the conflict began", async () => {
    const selectedEvent = await event();
    const record = selectedEvent.details.record as NorthStarRecord;
    const sourceEntry = { ...entry, date: "2026-07-07" };
    record.selected!.date = sourceEntry.date;
    record.sources[0].date = sourceEntry.date;
    record.sources[0].available_at = "2026-07-07T09:00:00Z";
    render(<NorthStarMoment profile={profile} journalEntries={[sourceEntry]} weeklyDigest={digest}
      driftResult={activeDrift} traceEvents={[selectedEvent]} />);
    expect(await screen.findByText(quote)).toBeTruthy();
  });

  it("does not turn insufficient evidence into encouragement", async () => {
    await renderCard({ mode: "encouragement", week_start: "2026-06-29", week_end: "2026-07-05" }, {
      weeklyDigest: { week_start: "2026-06-29", week_end: "2026-07-05" },
      driftResult: { delivery_state: "insufficient_evidence" },
    });
    expect(screen.queryByText("North Star Moment")).toBeNull();
  });

  it("rejects a Nudge response that was recorded after the conflict began", async () => {
    const selectedEvent = await event();
    const record = selectedEvent.details.record as NorthStarRecord;
    record.selected!.quote_source = "nudge_response";
    record.sources[0].nudge_response = quote;
    record.sources[0].response_available_at = "2026-07-08T09:00:00Z";
    render(<NorthStarMoment profile={profile} journalEntries={[{ ...entry, nudge_response: quote }]}
      weeklyDigest={digest} driftResult={activeDrift} traceEvents={[selectedEvent]} />);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByText("North Star Moment")).toBeNull();
  });
});
