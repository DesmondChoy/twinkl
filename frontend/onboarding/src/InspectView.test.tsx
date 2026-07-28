import { render, screen } from "@testing-library/react";
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

    expect(screen.getByText("0 trace events")).toBeTruthy();
    expect(screen.getByText("No backend events yet")).toBeTruthy();
    expect(
      screen.getByText(/exact prompt when applicable/i),
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
      "Weekly Digest built",
      "Weekly Coach generated",
    ].forEach((label) => expect(screen.getAllByText(label).length).toBeGreaterThan(0));
    ["Complete", "Reused", "Refused", "Invalid", "Failed"].forEach((status) =>
      expect(screen.getAllByText(status).length).toBeGreaterThan(0));
    expect(screen.getAllByText("Saved replay").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Live run").length).toBeGreaterThan(0);
    expect(screen.getByText("Canonical contract fixture")).toBeTruthy();
  });

  it("opens and focuses the event linked from Experience", () => {
    render(
      <InspectView
        events={events}
        selectedEventId="event-09"
        traceLabel="Canonical contract fixture"
        onReturn={() => undefined}
      />,
    );

    const selectedSummary = screen.getByLabelText(
      "Event 9: Drift checked, Complete, Saved replay",
    );
    expect(document.activeElement).toBe(selectedSummary);
    expect(selectedSummary.getAttribute("aria-current")).toBe("true");
    expect(selectedSummary.closest("details")?.open).toBe(true);
    expect(screen.getByTestId("trace-details-event-09")).toBeTruthy();
    expect(screen.getByText("After event 08")).toBeTruthy();
    expect(screen.getByText(/Event 09 · Drift Detector/)).toBeTruthy();
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
        "Event 8: Weekly review completed, Reused, Saved replay",
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
});
