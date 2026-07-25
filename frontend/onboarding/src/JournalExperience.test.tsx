import { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type {
  ExperienceSessionContract,
  JournalEntryContract,
  NudgeInteractionContract,
  TraceEventContract,
} from "./demoContracts";
import JournalExperience from "./JournalExperience";
import { canonicalInspectFixture } from "./inspectFixture";
import { createExperienceState, type ExperienceState } from "./session";

const api = vi.hoisted(() => ({
  createExperienceSession: vi.fn(),
  journalIdempotencyKey: vi.fn(),
  readExperienceTrace: vi.fn(),
  submitJournalEntry: vi.fn(),
}));

vi.mock("./experienceApi", async (importOriginal) => {
  const original = await importOriginal<typeof import("./experienceApi")>();
  return {
    ...original,
    ...api,
  };
});

const profile = canonicalInspectFixture.session.profile;

function entry(content = "A quiet walk helped me hear my own thoughts."): JournalEntryContract {
  return {
    journal_entry_id: "manual-entry-1",
    t_index: 0,
    date: "2026-07-25",
    content,
    nudge_response: null,
  };
}

function nudge(
  outcome: NudgeInteractionContract["outcome"] = "displayed",
  journalEntryId = "manual-entry-1",
): NudgeInteractionContract {
  return {
    nudge_id: `nudge-${journalEntryId}`,
    journal_entry_id: journalEntryId,
    outcome,
    category: outcome === "displayed" ? "elaboration" : null,
    reason: outcome === "displayed" ? "The meaning remains unexplored." : null,
    text: outcome === "displayed" ? "What felt clearest during that walk?" : null,
    response: null,
  };
}

function session(
  journalEntries: JournalEntryContract[],
  nudges: NudgeInteractionContract[],
): ExperienceSessionContract {
  return {
    ...canonicalInspectFixture.session,
    session_id: profile.session_id,
    profile,
    revision: journalEntries.length,
    journal_entries: journalEntries,
    nudges,
    weekly_reviewer_decisions: [],
    drift_result: null,
    weekly_digest: null,
    trace_event_ids: ["profile-1", "journal-1", "suppression-1", "decision-1"],
    selection: {
      view: "experience",
      selected_week: null,
      selected_journal_entry_id: null,
      selected_event_id: null,
    },
    updated_at: "2026-07-25T08:30:00Z",
  };
}

function decisionEvent(
  status: TraceEventContract["status"],
  journalEntryId = "manual-entry-1",
): TraceEventContract {
  return {
    ...canonicalInspectFixture.trace_events.find(
      (event) => event.event_type === "nudge_decided",
    )!,
    event_id: `decision-${status}`,
    session_id: profile.session_id,
    status,
    input_refs: [{ kind: "journal_entry", id: journalEntryId }],
  };
}

function trace(events: TraceEventContract[]) {
  return {
    schema_version: "experience-inspect-v1" as const,
    operation: "read_trace" as const,
    request_id: "trace-request",
    status: "ok" as const,
    session_id: profile.session_id,
    events,
  };
}

function Harness({
  initial = createExperienceState(),
  inspectRun = vi.fn(),
}: {
  initial?: ExperienceState;
  inspectRun?: (eventId: string) => void;
}) {
  const [experience, setExperience] = useState(initial);
  return (
    <JournalExperience
      profile={profile}
      experience={experience}
      updateExperience={(patch) =>
        setExperience((current) => ({ ...current, ...patch }))
      }
      inspectRun={inspectRun}
    />
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  api.createExperienceSession.mockResolvedValue({
    operation: "create_session",
  });
  api.journalIdempotencyKey.mockResolvedValue("a".repeat(64));
});

describe("manual Journal Entry Experience", () => {
  it("holds the Journal Entry, shows one nudge, and saves a reply", async () => {
    const savedEntry = entry();
    let submittedId = "";
    api.submitJournalEntry.mockImplementation(async (
      args: { entry: JournalEntryContract },
    ) => {
      submittedId = args.entry.journal_entry_id;
      return {
        operation: "submit_journal_entry",
        session: session([args.entry], [nudge("displayed", submittedId)]),
      };
    });
    api.readExperienceTrace.mockImplementation(async () =>
      trace([decisionEvent("complete", submittedId)]),
    );
    const user = userEvent.setup();
    render(<Harness />);

    await user.type(
      screen.getByRole("textbox", { name: "First Journal Entry" }),
      savedEntry.content,
    );
    await user.click(screen.getByRole("button", { name: "Save Journal Entry" }));

    const nudgeHeading = await screen.findByRole("heading", {
      name: "What felt clearest during that walk?",
    });
    expect(nudgeHeading).toBeTruthy();
    await waitFor(() => expect(document.activeElement).toBe(nudgeHeading));
    expect(screen.getByRole("status").textContent).toContain(
      "A follow-up question is ready.",
    );
    expect(screen.getByRole("status").querySelector("button")).toBeNull();
    await user.type(
      screen.getByRole("textbox", { name: "Your response" }),
      "I stopped performing for anyone else.",
    );
    await user.click(screen.getByRole("button", { name: "Save response" }));

    expect(screen.getByText("I stopped performing for anyone else.")).toBeTruthy();
    expect(screen.getAllByText(savedEntry.content)).toHaveLength(1);
    expect(api.submitJournalEntry).toHaveBeenCalledTimes(1);
  });

  it("allows the displayed nudge to be skipped", async () => {
    const savedEntry = entry();
    let submittedId = "";
    api.submitJournalEntry.mockImplementation(async (
      args: { entry: JournalEntryContract },
    ) => {
      submittedId = args.entry.journal_entry_id;
      return {
        operation: "submit_journal_entry",
        session: session([args.entry], [nudge("displayed", submittedId)]),
      };
    });
    api.readExperienceTrace.mockImplementation(async () =>
      trace([decisionEvent("complete", submittedId)]),
    );
    const user = userEvent.setup();
    render(<Harness />);

    await user.type(
      screen.getByRole("textbox", { name: "First Journal Entry" }),
      savedEntry.content,
    );
    await user.click(screen.getByRole("button", { name: "Save Journal Entry" }));
    await user.click(await screen.findByRole("button", { name: "Skip for now" }));

    expect(screen.getByText("Follow-up skipped.")).toBeTruthy();
    expect(screen.getByRole("textbox", { name: "Journal Entry" })).toBeTruthy();
  });

  it("retains earlier nudge replies after later Journal Entries", async () => {
    const serverEntries: JournalEntryContract[] = [];
    let submittedId = "";
    api.submitJournalEntry.mockImplementation(async (
      args: { entry: JournalEntryContract },
    ) => {
      submittedId = args.entry.journal_entry_id;
      serverEntries.push(args.entry);
      return {
        operation: "submit_journal_entry",
        session: session(
          [...serverEntries],
          serverEntries.map((item) =>
            nudge("displayed", item.journal_entry_id),
          ),
        ),
      };
    });
    api.readExperienceTrace.mockImplementation(async () =>
      trace([decisionEvent("complete", submittedId)]),
    );
    const user = userEvent.setup();
    render(<Harness />);

    await user.type(
      screen.getByRole("textbox", { name: "First Journal Entry" }),
      "A quiet walk helped me hear my own thoughts.",
    );
    await user.click(screen.getByRole("button", { name: "Save Journal Entry" }));
    await user.type(
      await screen.findByRole("textbox", { name: "Your response" }),
      "I stopped performing for anyone else.",
    );
    await user.click(screen.getByRole("button", { name: "Save response" }));

    await user.type(
      screen.getByRole("textbox", { name: "Journal Entry" }),
      "Planning a small project made the afternoon disappear.",
    );
    await user.click(screen.getByRole("button", { name: "Save Journal Entry" }));

    expect(
      await screen.findByText("I stopped performing for anyone else."),
    ).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Skip for now" }));
    expect(screen.getByText("I stopped performing for anyone else.")).toBeTruthy();
    expect(screen.getByText("Follow-up skipped.")).toBeTruthy();
  });

  it("shows no question when the anti-annoyance rule suppresses it", async () => {
    const savedEntry = entry();
    api.submitJournalEntry.mockImplementation(async (
      args: { entry: JournalEntryContract },
    ) => ({
      operation: "submit_journal_entry",
      session: session(
        [args.entry],
        [nudge("suppressed", args.entry.journal_entry_id)],
      ),
    }));
    api.readExperienceTrace.mockResolvedValue(trace([]));
    const user = userEvent.setup();
    render(<Harness />);

    await user.type(
      screen.getByRole("textbox", { name: "First Journal Entry" }),
      savedEntry.content,
    );
    await user.click(screen.getByRole("button", { name: "Save Journal Entry" }));

    await waitFor(() => expect(screen.getByText("Saved.")).toBeTruthy());
    expect(screen.queryByText("One question")).toBeNull();
  });

  it("retries only the failed nudge step without duplicating the Journal Entry", async () => {
    const savedEntry = entry();
    let submittedEntry: JournalEntryContract | null = null;
    api.submitJournalEntry
      .mockImplementationOnce(async (args: { entry: JournalEntryContract }) => {
        submittedEntry = args.entry;
        return {
          operation: "submit_journal_entry",
          session: session([args.entry], []),
        };
      })
      .mockImplementationOnce(async (args: { entry: JournalEntryContract }) => {
        submittedEntry = args.entry;
        return {
          operation: "submit_journal_entry",
          session: session(
            [args.entry],
            [nudge("no_nudge", args.entry.journal_entry_id)],
          ),
        };
      });
    api.readExperienceTrace
      .mockImplementationOnce(async () =>
        trace([decisionEvent("failed", submittedEntry!.journal_entry_id)]),
      )
      .mockImplementationOnce(async () =>
        trace([decisionEvent("complete", submittedEntry!.journal_entry_id)]),
      );
    const user = userEvent.setup();
    render(<Harness />);

    await user.type(
      screen.getByRole("textbox", { name: "First Journal Entry" }),
      savedEntry.content,
    );
    await user.click(screen.getByRole("button", { name: "Save Journal Entry" }));
    await user.click(
      await screen.findByRole("button", {
        name: "Try the follow-up check again",
      }),
    );

    await waitFor(() => expect(api.submitJournalEntry).toHaveBeenCalledTimes(2));
    expect(screen.getAllByText(savedEntry.content)).toHaveLength(1);
    expect(screen.queryByRole("button", {
      name: "Try the follow-up check again",
    })).toBeNull();
  });

  it("keeps transport-failed text only in the composer until retry succeeds", async () => {
    const savedEntry = entry();
    let submittedEntry: JournalEntryContract | null = null;
    api.submitJournalEntry
      .mockRejectedValueOnce(new Error("Connection lost"))
      .mockImplementationOnce(async (args: { entry: JournalEntryContract }) => {
        submittedEntry = args.entry;
        return {
          operation: "submit_journal_entry",
          session: session(
            [args.entry],
            [nudge("no_nudge", args.entry.journal_entry_id)],
          ),
        };
      });
    api.readExperienceTrace.mockImplementation(async () =>
      trace([decisionEvent("complete", submittedEntry!.journal_entry_id)]),
    );
    const user = userEvent.setup();
    render(<Harness />);

    await user.type(
      screen.getByRole("textbox", { name: "First Journal Entry" }),
      savedEntry.content,
    );
    await user.click(screen.getByRole("button", { name: "Save Journal Entry" }));

    expect(
      await screen.findByRole("button", {
        name: "Try the follow-up check again",
      }),
    ).toBeTruthy();
    expect(screen.getByDisplayValue(savedEntry.content)).toBeTruthy();
    expect(screen.queryByRole("region", { name: "Moment by moment." })).toBeNull();
    expect(screen.getByRole("status").querySelector("button")).toBeNull();

    await user.click(
      screen.getByRole("button", { name: "Try the follow-up check again" }),
    );

    await waitFor(() => expect(api.submitJournalEntry).toHaveBeenCalledTimes(2));
    expect(screen.getAllByText(savedEntry.content)).toHaveLength(1);
    expect(screen.queryByDisplayValue(savedEntry.content)).toBeNull();
  });

  it("locks synchronously before hashing to prevent double submission", async () => {
    let resolveKey: (value: string) => void = () => undefined;
    api.journalIdempotencyKey.mockReturnValueOnce(
      new Promise<string>((resolve) => {
        resolveKey = resolve;
      }),
    );
    let submittedId = "";
    api.submitJournalEntry.mockImplementation(async (
      args: { entry: JournalEntryContract },
    ) => {
      submittedId = args.entry.journal_entry_id;
      return {
        operation: "submit_journal_entry",
        session: session(
          [args.entry],
          [nudge("no_nudge", args.entry.journal_entry_id)],
        ),
      };
    });
    api.readExperienceTrace.mockImplementation(async () =>
      trace([decisionEvent("complete", submittedId)]),
    );
    render(<Harness />);

    fireEvent.change(
      screen.getByRole("textbox", { name: "First Journal Entry" }),
      { target: { value: "One submission, even after two fast clicks." } },
    );
    const save = screen.getByRole("button", { name: "Save Journal Entry" });
    fireEvent.click(save);
    fireEvent.click(save);

    expect(api.journalIdempotencyKey).toHaveBeenCalledTimes(1);
    resolveKey("a".repeat(64));
    await waitFor(() => expect(api.submitJournalEntry).toHaveBeenCalledTimes(1));
  });

  it("shows ambient Drift and cited Weekly Digest evidence", async () => {
    const inspectRun = vi.fn();
    const initial: ExperienceState = {
      ...createExperienceState(),
      journal_started: true,
      revision: canonicalInspectFixture.session.revision,
      journal_entries: canonicalInspectFixture.session.journal_entries,
      nudges: canonicalInspectFixture.session.nudges,
      weekly_reviewer_decisions:
        canonicalInspectFixture.session.weekly_reviewer_decisions,
      drift_result: canonicalInspectFixture.session.drift_result,
      weekly_digest: canonicalInspectFixture.session.weekly_digest,
      run_state: "complete",
      trace_event_ids: canonicalInspectFixture.session.trace_event_ids,
      trace_events: canonicalInspectFixture.trace_events,
    };
    const user = userEvent.setup();
    render(<Harness initial={initial} inspectRun={inspectRun} />);

    expect(
      screen.getByRole("heading", { name: "Your week in view." }),
    ).toBeTruthy();
    expect(screen.getByText("Benevolence")).toBeTruthy();
    expect(screen.getAllByText("Active Drift").length).toBeGreaterThan(0);
    expect(
      screen.getAllByText(
        "Cancelled dinner with my sister to stay at work.",
      ).length,
    ).toBeGreaterThan(1);
    const citation = screen.getByRole("link", {
      name: "Open Journal Entry from 2026-07-06",
    });
    expect(citation.getAttribute("href")).toContain("journal-entry-");
    await user.click(citation);
    const journalEntry = document.querySelector(citation.getAttribute("href")!);
    expect(journalEntry?.getAttribute("aria-current")).toBe("true");
    expect(screen.queryByText("Raw provider response")).toBeNull();
    expect(screen.queryByText("Validation result")).toBeNull();

    await user.click(screen.getByRole("button", { name: "Inspect Drift run" }));
    expect(inspectRun).toHaveBeenLastCalledWith("event-09");
    await user.click(
      screen.getByRole("button", { name: "Inspect Weekly Digest run" }),
    );
    expect(inspectRun).toHaveBeenLastCalledWith("event-10");
  });

  it("does not call an unavailable weekly review No Drift", () => {
    const savedEntry = entry();
    const initial: ExperienceState = {
      ...createExperienceState(),
      journal_started: true,
      journal_entries: [savedEntry],
      weekly_reviewer_decisions: [{
        persona_id: "persona-1",
        week_start: "2026-07-20",
        week_end: "2026-07-26",
        t_index: savedEntry.t_index,
        date: savedEntry.date,
        core_value: profile.top_values[0],
        verdict: "abstain",
        confidence: null,
        reason_code: "provider_error",
        evidence_quote: "",
        review_status: "error",
      }],
      drift_result: {
        persona_id: "persona-1",
        delivery_state: "stable",
        core_value_states: { [profile.top_values[0]]: "stable" },
      },
      weekly_digest: {
        week_start: "2026-07-20",
        week_end: "2026-07-26",
        mode_rationale:
          "The Weekly Drift Reviewer could not return usable evidence for this week.",
        evidence: [],
      },
    };
    render(<Harness initial={initial} />);

    expect(screen.getAllByText("Review unavailable")).toHaveLength(
      profile.top_values.length + 1,
    );
    expect(screen.queryByText("No Drift")).toBeNull();
  });
});
