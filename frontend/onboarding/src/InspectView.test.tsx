import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import InspectView from "./InspectView";
import { canonicalInspectFixture } from "./inspectFixture";
import styles from "./styles.css?raw";

const events = canonicalInspectFixture.trace_events;

describe("Inspect view", () => {
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

    expect(screen.getAllByRole("listitem")).toHaveLength(14);
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
      "Weekly Coach response generated",
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
    expect(screen.getAllByText("Weekly Coach").length).toBeGreaterThan(0);
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

  it("filters the current week by component", async () => {
    const user = userEvent.setup();
    render(
      <InspectView
        events={events}
        currentWeekEventIds={events.slice(6, 11).map((event) => event.event_id)}
        selectedEventId={null}
        traceLabel="Canonical contract fixture"
        onReturn={() => undefined}
      />,
    );

    await user.click(screen.getByRole("button", {
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
