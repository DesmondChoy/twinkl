import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import InspectView from "./InspectView";
import { canonicalInspectFixture } from "./inspectFixture";
import type { TraceEventContract } from "./demoContracts";
import { displayWeekRange } from "./displayFormatters";
import { weeklyRunContext } from "./weeklyRun";
import styles from "./styles.css?raw";

const events = canonicalInspectFixture.trace_events;

function weeklyTrace(prefix: string, start: string, end: string): TraceEventContract[] {
  const rows = structuredClone(events.slice(6, 11));
  const eventIds = new Map(rows.map((event) => [event.event_id, `${prefix}:${event.event_type}`]));
  rows.forEach((event) => {
    event.event_id = eventIds.get(event.event_id)!;
    event.parent_event_id = eventIds.get(event.parent_event_id ?? "") ?? null;
    for (const ref of [...event.input_refs, ...event.result_refs] as { id: string }[]) {
      ref.id = eventIds.get(ref.id) ?? `${prefix}:${ref.id}`;
    }
    for (const key of ["request", "receipt", "digest"]) {
      const value = event.details[key] as Record<string, unknown> | undefined;
      if (value) Object.assign(value, { week_start: start, week_end: end });
    }
  });
  rows.push({
    ...rows[4], event_id: `${prefix}:north_star_reviewed`, event_type: "north_star_reviewed",
    parent_event_id: rows[4].event_id, input_refs: [{ kind: "week", id: `${prefix}:${start}` }],
    result_refs: [], model_contract: null,
    details: { record: { status: "unavailable", reason: `${prefix}_moment`, week_start: start, week_end: end, selected: null } },
  });
  return rows;
}

describe("Inspect view", () => {
  it.each([true, false])("distinguishes saved nudge history from an enforced spacing decision (would suppress: %s)", (suppressed) => {
    const event = structuredClone(events.find((item) => item.event_type === "nudge_suppression_checked")!);
    event.details.policy_applied = false;
    event.details.suppressed = suppressed;
    render(<InspectView events={[event]} selectedEventId={event.event_id}
      traceLabel="Saved Persona replay" onReturn={() => undefined} />);
    expect(screen.getByText(`Saved nudge history; current spacing rule would ${suppressed ? "suppress" : "allow"} a nudge`)).toBeTruthy();
    expect(screen.queryByText("Nudge suppressed by the anti-annoyance rule")).toBeNull();
  });
  it("shows an honest empty state without rendering fixture events", () => {
    render(
      <InspectView
        events={[]}
        emptyMessage="Profile validation is still in progress."
        selectedEventId={null}
        traceLabel="Current Experience session"
        onReturn={() => undefined}
      />,
    );

    expect(screen.getByText("0 recorded events")).toBeTruthy();
    expect(screen.getByText("Recorded work")).toBeTruthy();
    expect(
      screen.getByText(/Open Technical details/i),
    ).toBeTruthy();
    expect(screen.getByRole("status")).toHaveProperty(
      "textContent",
      "Profile validation is still in progress.",
    );
    expect(screen.queryByRole("list")).toBeNull();
    expect(screen.queryByText("Journal Entry submitted")).toBeNull();
  });

  it("renders every trace event type and terminal state from the contract fixture", () => {
    render(
      <InspectView
        events={events}
        selectedEventId={null}
        traceLabel="Canonical contract fixture"
        onReturn={() => undefined}
      />,
    );

    expect(screen.getAllByRole("listitem")).toHaveLength(15);
    [
      "Profile confirmed",
      "Journal Entry submitted",
      "Nudge suppression checked",
      "Nudge decided",
      "Nudge generated",
      "Weekly review requested",
      "Weekly review completed",
      "Drift checked",
      "Weekly Drift Detection output stored",
      "Coach Digest response generated",
      "Simulated time changed",
    ].forEach((label) => expect(screen.getAllByText(label).length).toBeGreaterThan(0));
    ["Refused", "Invalid", "Failed"].forEach((status) =>
      expect(screen.getAllByText(status).length).toBeGreaterThan(0));
    expect(screen.queryByText("Reused")).toBeNull();
    expect(screen.queryByText("Saved replay")).toBeNull();
    expect(screen.queryByText("0 ms")).toBeNull();
    expect(screen.queryByText("Live run")).toBeNull();
    expect(screen.getByText("Canonical contract fixture")).toBeTruthy();
  });

  it("shows a focused weekly explanation before the complete event history", () => {
    render(
      <InspectView
        events={events}
        currentWeekEventIds={events.slice(6, 11).map((event) => event.event_id)}
        selectedEventId="event-09"
        traceLabel="Canonical contract fixture"
        onReturn={() => undefined}
      />,
    );

    const selectedSummary = screen.getByLabelText(
      "Event 9: Drift checked",
    );
    expect(screen.getByRole("heading", {
      name: "How Twinkl reached this result.",
    })).toBeTruthy();
    expect(screen.getByText(/not human validation/i)).toBeTruthy();
    expect(screen.getAllByText("Weekly Drift Reviewer").length)
      .toBeGreaterThan(0);
    expect(screen.getAllByText("Drift Detector").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Coach Digest").length).toBeGreaterThan(0);
    expect(selectedSummary.getAttribute("aria-current")).toBe("true");
    expect(selectedSummary.closest("details")?.open).toBe(true);
    expect(screen.getByText(/Event 09 · Drift Detector/)).toBeTruthy();
    expect(
      screen.getByText("Complete Inspect history").closest("details")?.open,
    ).toBe(false);
  });

  it("shows exact model evidence on demand and keeps source labels explicit", () => {
    render(
      <InspectView
        events={events}
        selectedEventId="event-08"
        traceLabel="Canonical contract fixture"
        onReturn={() => undefined}
      />,
    );

    expect(
      screen.getByLabelText(
        "Event 8: Weekly review completed",
      ).closest("details")?.open,
    ).toBe(true);
    expect(screen.getByLabelText("Model contract").textContent).toContain(
      "gpt-5.6-luna",
    );
    expect(screen.getByLabelText("Exact rendered prompt").textContent).toContain(
      "Benevolence Conflict",
    );
    expect(screen.getByLabelText("Raw provider response").textContent).toContain(
      "evidence_quote",
    );
    expect(screen.getByLabelText("Validation").textContent).toContain(
      "weekly_review_completed-v1",
    );
    expect(screen.getByLabelText("Effective result").textContent).toContain(
      "response-demo",
    );
  });

  it("shows a reused Coach Digest response as available", () => {
    const coachEvent = events.find(
      (event) => event.event_type === "weekly_coach_generated",
    )!;
    const reusedCoachEvent = { ...coachEvent, status: "reused" as const };

    render(
      <InspectView
        events={[reusedCoachEvent]}
        currentWeekEventIds={[reusedCoachEvent.event_id]}
        selectedEventId={reusedCoachEvent.event_id}
        traceLabel="Saved Persona replay"
        onReturn={() => undefined}
      />,
    );

    expect(
      screen.getAllByText("Coach Digest response and question ready").length,
    ).toBeGreaterThan(0);
    expect(screen.queryByText("Coach Digest response unavailable")).toBeNull();
  });

  it.each([
    "weekly_review_requested", "weekly_review_completed", "drift_detected",
    "weekly_digest_built", "weekly_coach_generated", "north_star_reviewed",
  ])("keeps a historical %s linked to its own weekly result", (eventType) => {
    const first = weeklyTrace("first", "2026-07-06", "2026-07-12");
    const later = weeklyTrace("later", "2026-07-13", "2026-07-19");
    later[0].parent_event_id = first.at(-1)!.event_id;
    later[2].details.result = { delivery_state: "no_active_drift", drifts: [] };
    later[4].status = "failed";
    const selectedEventId = first.find((event) => event.event_type === eventType)!.event_id;
    render(<InspectView events={[...first, ...later]} selectedEventId={selectedEventId}
      traceLabel="Current Experience session" onReturn={() => undefined} />);

    const summary = within(screen.getByRole("region", { name: "How Twinkl reached this result." }));
    expect(summary.getByText(`Week: ${displayWeekRange("2026-07-06", "2026-07-12")}`)).toBeTruthy();
    expect(summary.getByText("Active Drift · 1 Drift confirmed")).toBeTruthy();
    expect(summary.getByText("Coach Digest response and question ready")).toBeTruthy();
    expect(summary.getByText("Unavailable · first moment")).toBeTruthy();
    expect(summary.queryByText("No Active Drift · 0 Drifts confirmed")).toBeNull();
    expect(summary.queryByText("Coach Digest response unavailable")).toBeNull();
    expect(summary.queryByText("Unavailable · later moment")).toBeNull();
    expect(document.querySelector('[aria-current="true"]')?.closest("details")?.open).toBe(true);
  });

  it.each([true, false])("binds a later Coach retry to its original digest (parent link: %s)", (hasParent) => {
    const first = weeklyTrace("first", "2026-07-06", "2026-07-12");
    const later = weeklyTrace("later", "2026-07-13", "2026-07-19");
    later[2].details.result = { delivery_state: "no_active_drift", drifts: [] };
    const retry: TraceEventContract = {
      ...first[4], event_id: "retry-coach", status: "failed",
      parent_event_id: hasParent ? first[3].event_id : null,
    };
    const trace = [...first, ...later, retry];
    expect(weeklyRunContext(trace, retry.event_id)?.events.map((event) => event.event_id))
      .toEqual([...first.map((event) => event.event_id), retry.event_id]);
    render(<InspectView events={trace} selectedEventId={retry.event_id}
      traceLabel="Current Experience session" onReturn={() => undefined} />);

    const summary = within(screen.getByRole("region", { name: "How Twinkl reached this result." }));
    expect(summary.getByText(`Week: ${displayWeekRange("2026-07-06", "2026-07-12")}`)).toBeTruthy();
    expect(summary.getByText("Active Drift · 1 Drift confirmed")).toBeTruthy();
    expect(summary.getByText("Coach Digest response unavailable")).toBeTruthy();
  });

  it("keeps assessment calculation separate from a previously focused weekly event and filter", async () => {
    const user = userEvent.setup();
    const profile = canonicalInspectFixture.session.profile;
    const props = { events, selectedEventId: "event-11", traceLabel: "Current Experience session", onReturn: () => undefined };
    const view = render(<InspectView {...props} />);
    await user.click(screen.getByRole("button", { name: "Weekly results" }));
    view.rerender(<InspectView {...props} onboarding={{
      confirmedValues: profile.top_values, responses: profile.bws_responses,
      scores: { bws: profile.bws_results, profile: profile.value_profile },
      setOrder: Array.from({ length: 11 }, (_, index) => index),
    }} />);

    expect(screen.queryByRole("region", { name: "How Twinkl reached this result." })).toBeNull();
    expect(screen.queryByTestId("inspect-selection")).toBeNull();
    expect(document.activeElement).toBe(screen.getByRole("heading", { name: "See how each trade-off shaped this Profile." }));
    expect(within(screen.getByRole("list", { name: "Recorded events" })).getAllByRole("listitem"))
      .toHaveLength(events.length);
  });

  it("places filters beside event history and reports matches without changing the focused explanation", async () => {
    const user = userEvent.setup();
    render(
      <InspectView
        events={events}
        currentWeekEventIds={events.slice(6, 11).map((event) => event.event_id)}
        selectedEventId="event-09"
        traceLabel="Canonical contract fixture"
        onReturn={() => undefined}
      />,
    );

    const history = screen.getByRole("region", { name: "Current week first." });
    const filters = within(history).getByRole("navigation", { name: "Filter Inspect events" });
    const count = within(history).getByRole("status", { name: "Filtered event count" });
    const explanation = screen.getByRole("region", { name: "How Twinkl reached this result." });
    const explanationText = explanation.textContent;
    expect(explanation.compareDocumentPosition(filters) & Node.DOCUMENT_POSITION_FOLLOWING)
      .toBeTruthy();
    expect(count.textContent).toBe("5 of 5 current week events · 10 of 10 earlier events");

    await user.click(within(filters).getByRole("button", {
      name: "Weekly Drift Reviewer",
    }));

    expect(
      within(screen.getByRole("list", { name: "Current week events" }))
        .getAllByRole("listitem"),
    ).toHaveLength(2);
    expect(
      screen.getByRole("button", { name: "Weekly Drift Reviewer" })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    expect(count.textContent).toBe("2 of 5 current week events · 1 of 10 earlier events");
    expect(explanation.textContent).toBe(explanationText);
    const earlierHistory = screen.getByText("Complete Inspect history").closest("details")!;
    expect(earlierHistory.open).toBe(false);
    await user.click(screen.getByText("Complete Inspect history"));
    expect(within(screen.getByRole("list", { name: "Earlier events" }))
      .getAllByRole("listitem")).toHaveLength(1);

    await user.click(within(filters).getByRole("button", { name: "Journal Entries" }));
    expect(count.textContent).toBe("0 of 5 current week events · 8 of 10 earlier events");
    expect(screen.getByText("This week has no Journal Entries events.")).toBeTruthy();
    expect(within(screen.getByRole("list", { name: "Earlier events" }))
      .getAllByRole("listitem")).toHaveLength(8);

    await user.click(within(filters).getByRole("button", { name: "All steps" }));
    expect(count.textContent).toBe("5 of 5 current week events · 10 of 10 earlier events");
    expect(within(screen.getByRole("list", { name: "Current week events" }))
      .getAllByRole("listitem")).toHaveLength(5);
  });

  it("reports filtered recorded events when there is no selected week", async () => {
    const user = userEvent.setup();
    render(
      <InspectView
        events={events}
        selectedEventId={null}
        traceLabel="Current Experience session"
        onReturn={() => undefined}
      />,
    );

    const count = screen.getByRole("status", { name: "Filtered event count" });
    expect(count.textContent).toBe("15 of 15 recorded events");
    await user.click(screen.getByRole("button", { name: "Weekly results" }));
    expect(count.textContent).toBe("3 of 15 recorded events");
    expect(within(screen.getByRole("list", { name: "Recorded events" }))
      .getAllByRole("listitem")).toHaveLength(3);
    expect(screen.queryByText("Complete Inspect history")).toBeNull();
  });

  it("redacts sensitive fields before rendering provider data", () => {
    const event = {
      ...events[7],
      raw_response: {
        result: "safe content",
        authorization: "Bearer visible-secret",
        nested: {
          api_key: "visible-api-key",
          headers: { cookie: "visible-cookie" },
        },
      },
    };
    render(
      <InspectView
        events={[event]}
        selectedEventId={event.event_id}
        traceLabel="Redaction fixture"
        onReturn={() => undefined}
      />,
    );

    const response = screen.getByLabelText("Raw provider response").textContent ?? "";
    expect(response).toContain("safe content");
    expect(response).toContain("[redacted]");
    expect(response).not.toContain("visible-secret");
    expect(response).not.toContain("visible-api-key");
    expect(response).not.toContain("visible-cookie");
  });

  it("marks submitted Journal Entries removed from the current Experience", () => {
    const submitted = events.find(
      (event) => event.event_type === "journal_entry_submitted",
    )!;
    render(
      <InspectView
        events={[submitted]}
        currentJournalEntryIds={[]}
        selectedEventId={null}
        traceLabel="Current Experience session"
        onReturn={() => undefined}
      />,
    );

    expect(
      screen.getByText(/Removed from current Experience/),
    ).toBeTruthy();
  });

  it("keeps the phone layout single-column and wraps long trace content", () => {
    expect(styles).toContain("@media (max-width: 620px)");
    expect(styles).toMatch(
      /\.trace-event__details\s*\{[\s\S]*?display:\s*block;/,
    );
    expect(styles).toMatch(
      /\.inspect-detail pre\s*\{[\s\S]*?white-space:\s*pre-wrap;[\s\S]*?overflow-wrap:\s*anywhere;/,
    );
    expect(styles).toMatch(
      /\.journal-thread__entry\s*\{[\s\S]*?overflow-wrap:\s*anywhere;/,
    );
  });

  it("centers bounded content in the right column", () => {
    expect(styles).toMatch(
      /\.stage--inspect\s*\{[\s\S]*?margin:\s*0 auto;/,
    );
    expect(styles).toMatch(/\.stage\s*\{[\s\S]*?margin:\s*auto;/);
    expect(styles).toMatch(
      /\.persona-picker\s*\{[\s\S]*?margin-inline:\s*auto;/,
    );
    expect(styles).toMatch(
      /\.persona-replay\s*\{[\s\S]*?margin-inline:\s*auto;/,
    );
    expect(styles).toMatch(
      /\.journal-experience\s*\{[\s\S]*?margin-inline:\s*auto;/,
    );
  });
});
