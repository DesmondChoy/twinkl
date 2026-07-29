import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App, { AppErrorBoundary } from "./App";
import type {
  JournalEntrySubmittedResponseContract,
  SessionCreatedResponseContract,
  TraceEventContract,
  TraceReadResponseContract,
} from "./demoContracts";
import {
  createExperienceSession,
  ExperienceApiError,
  readExperienceTrace,
  submitJournalEntry,
} from "./experienceApi";
import { canonicalInspectFixture } from "./inspectFixture";
import { SESSION_STORAGE_KEY } from "./session";

vi.mock("./experienceApi", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./experienceApi")>();
  return {
    ...actual,
    createExperienceSession: vi.fn(),
    readExperienceTrace: vi.fn(),
    submitJournalEntry: vi.fn(),
  };
});

vi.stubGlobal("confirm", () => true);

const profileEvents = new Map<string, TraceEventContract>();

function answerSet() {
  const first = screen.getAllByTestId("value-card").find((card) => card.dataset.location === "pool")!;
  fireEvent.click(first);
  const second = screen.getAllByTestId("value-card").find((card) => card.dataset.location === "pool")!;
  fireEvent.click(second);
  act(() => vi.advanceTimersByTime(1_000));
}

beforeEach(() => {
  localStorage.clear();
  profileEvents.clear();
  vi.mocked(createExperienceSession).mockReset();
  vi.mocked(readExperienceTrace).mockReset();
  vi.mocked(submitJournalEntry).mockReset();
  vi.mocked(createExperienceSession).mockImplementation(async (profile) => {
    const fixtureEvent = canonicalInspectFixture.trace_events.find(
      (event) => event.event_type === "profile_confirmed",
    )!;
    const event: TraceEventContract = {
      ...fixtureEvent,
      event_id: `profile-${profile.session_id}`,
      session_id: profile.session_id,
      parent_event_id: null,
      input_refs: [],
      result_refs: [{ kind: "profile", id: profile.session_id }],
      details: { profile },
    };
    profileEvents.set(profile.session_id, event);
    return {
      schema_version: canonicalInspectFixture.schema_version,
      operation: "create_session",
      request_id: "create-profile",
      status: "ok",
      session: {
        ...canonicalInspectFixture.session,
        session_id: profile.session_id,
        revision: 0,
        profile,
        journal_entries: [],
        nudges: [],
        weekly_reviewer_decisions: [],
        drift_result: null,
        weekly_digest: null,
        trace_event_ids: [event.event_id],
        selection: { view: "experience" },
      },
    } as SessionCreatedResponseContract;
  });
  vi.mocked(readExperienceTrace).mockImplementation(async (sessionId) => ({
    schema_version: canonicalInspectFixture.schema_version,
    operation: "read_trace",
    request_id: "read-profile",
    status: "ok",
    session_id: sessionId,
    events: profileEvents.has(sessionId)
      ? [profileEvents.get(sessionId)!]
      : [],
  } as TraceReadResponseContract));
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("onboarding app", () => {
  it("keeps Start over reachable after an unexpected render failure", () => {
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    const Broken = () => {
      throw new Error("render failed");
    };

    render(
      <AppErrorBoundary>
        <Broken />
      </AppErrorBoundary>,
    );

    expect(
      screen.getByRole("heading", {
        name: "This saved view could not be restored.",
      }),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: "Start over" })).toBeTruthy();
    consoleError.mockRestore();
  });

  it("keeps Experience usable when browser storage rejects a write", async () => {
    const storageWrite = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new DOMException("Storage full", "QuotaExceededError");
      });

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveProperty(
      "textContent",
      expect.stringContaining("Progress could not be saved"),
    );
    expect(screen.getByLabelText("Values · 1 of 11")).toBeTruthy();
    storageWrite.mockRestore();
  });

  it("opens directly on the first six-card set without Schwartz labels", () => {
    render(<App />);
    const progress = screen.getByRole("progressbar", {
      name: "Values · 1 of 11",
    });
    expect(progress.getAttribute("aria-valuenow")).toBe("1");
    expect(progress.getAttribute("aria-valuemax")).toBe("11");
    expect(
      screen.getByRole("heading", {
        name: "What matters most as you find your way?",
      }).getAttribute("aria-describedby"),
    ).toBe("assessment-progress");
    expect(screen.getByLabelText("Values · 1 of 11")).toBeTruthy();
    expect(screen.getAllByTestId("value-card")).toHaveLength(6);
    expect(screen.getByText("Next step")).toBeTruthy();
    expect(screen.getByText(/across 11 groups/i)).toBeTruthy();
    expect(screen.queryByText(/saved only in this browser/i)).toBeNull();
    expect(screen.queryByText("Hedonism")).toBeNull();
    expect(screen.queryByText("Universalism")).toBeNull();
    const inspect = screen.getByRole("button", { name: /inspect/i });
    expect(inspect.getAttribute("aria-disabled")).toBe("true");
    expect(screen.getByText("Available after all 11 questions")).toBeTruthy();
    fireEvent.click(inspect);
    expect(
      screen.queryByRole("heading", {
        name: "See how each trade-off shaped this Profile.",
      }),
    ).toBeNull();
    expect(screen.getByTestId("drop-most").classList.contains("drop-box--guided")).toBe(true);
    expect(screen.getByTestId("drop-least").classList.contains("drop-box--guided")).toBe(false);
  });

  it("places two taps in Most then Least and advances after a one-second review", () => {
    vi.useFakeTimers();
    render(<App />);
    act(() => vi.advanceTimersByTime(400));
    const first = screen.getAllByTestId("value-card")[0];
    const firstPhrase = first.querySelector(".value-card__phrase")!.textContent!;
    fireEvent.click(first);
    expect(screen.getByTestId("drop-most").querySelector('[data-location="most"]')).toBeTruthy();
    expect(screen.getByTestId("drop-most").textContent).toContain(firstPhrase);
    expect(screen.getByTestId("drop-least").classList.contains("drop-box--guided")).toBe(true);
    expect(screen.getByText(/now choose least/i)).toBeTruthy();
    expect(screen.getByTestId("selection-area").querySelectorAll('[data-location="pool"]')).toHaveLength(5);

    act(() => vi.advanceTimersByTime(350));
    const second = screen.getAllByTestId("value-card").find((card) => card.dataset.location === "pool")!;
    const secondPhrase = second.querySelector(".value-card__phrase")!.textContent!;
    fireEvent.click(second);
    expect(screen.getByTestId("drop-most").textContent).toContain(firstPhrase);
    expect(screen.getByTestId("drop-least").textContent).toContain(secondPhrase);
    expect(screen.getByTestId("drop-least").querySelector('[data-location="least"]')).toBeTruthy();
    expect(screen.getByText(/take a moment to review/i)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Continue" })).toBeNull();
    expect(screen.getAllByTestId("value-card").every((card) => card.getAttribute("aria-disabled") === "true")).toBe(true);

    act(() => vi.advanceTimersByTime(999));
    expect(screen.getByLabelText("Values · 1 of 11")).toBeTruthy();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByLabelText("Values · 2 of 11")).toBeTruthy();
    expect(screen.getAllByTestId("value-card")).toHaveLength(6);
    const stored = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY)!);
    expect(stored.responses[0].response_time_ms).toBe(750);
  });

  it("moves cards into both boxes and back with the keyboard", async () => {
    const user = userEvent.setup();
    render(<App />);
    const first = screen.getAllByTestId("value-card")[0];
    const firstValue = first.dataset.value;
    first.focus();
    await user.keyboard("m");
    const mostCard = screen.getByTestId("drop-most").querySelector<HTMLElement>('[data-location="most"]')!;
    expect(mostCard).toBeTruthy();
    expect(document.activeElement).toBe(mostCard);
    mostCard.focus();
    await user.keyboard("{Backspace}");
    expect(screen.getByTestId("drop-most").textContent).toContain("Tap a card first");
    const returnedCard = screen.getByTestId("selection-area").querySelector<HTMLElement>(
      `[data-value="${firstValue}"][data-location="pool"]`,
    );
    expect(document.activeElement).toBe(returnedCard);

    returnedCard!.focus();
    await user.keyboard("m");
    const poolCard = screen.getAllByTestId("value-card").find((card) => card.dataset.location === "pool")!;
    poolCard.focus();
    await user.keyboard("l");
    const leastCard = screen.getByTestId("drop-least").querySelector<HTMLElement>('[data-location="least"]')!;
    expect(leastCard).toBeTruthy();
    expect(document.activeElement).toBe(leastCard);
    expect(screen.getByText(/take a moment to review/i)).toBeTruthy();
  });

  it("moves cards into both boxes and back with pointer dragging", () => {
    render(<App />);
    const most = screen.getByTestId("drop-most");
    const least = screen.getByTestId("drop-least");
    const selection = screen.getByTestId("selection-area");
    vi.spyOn(most, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: 200,
      bottom: 120,
      width: 200,
      height: 120,
      toJSON: () => ({}),
    });
    vi.spyOn(least, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 500,
      top: 500,
      left: 0,
      right: 200,
      bottom: 620,
      width: 200,
      height: 120,
      toJSON: () => ({}),
    });
    vi.spyOn(selection, "getBoundingClientRect").mockReturnValue({
      x: 0,
      y: 200,
      top: 200,
      left: 0,
      right: 400,
      bottom: 500,
      width: 400,
      height: 300,
      toJSON: () => ({}),
    });
    const card = screen.getAllByTestId("value-card")[0];
    fireEvent.pointerDown(card, { pointerId: 1, clientX: 300, clientY: 300 });
    fireEvent.pointerMove(card, { pointerId: 1, clientX: 100, clientY: 60 });
    fireEvent.pointerUp(card, { pointerId: 1, clientX: 100, clientY: 60 });
    expect(most.querySelector('[data-location="most"]')).toBeTruthy();
    const placed = most.querySelector<HTMLElement>('[data-location="most"]')!;
    fireEvent.pointerDown(placed, { pointerId: 2, clientX: 100, clientY: 60 });
    fireEvent.pointerMove(placed, { pointerId: 2, clientX: 100, clientY: 300 });
    fireEvent.pointerUp(placed, { pointerId: 2, clientX: 100, clientY: 300 });
    expect(most.querySelector('[data-location="most"]')).toBeNull();
    expect(screen.getAllByTestId("value-card")).toHaveLength(6);
    const nextCard = screen.getAllByTestId("value-card")[1];
    fireEvent.pointerDown(nextCard, { pointerId: 3, clientX: 100, clientY: 300 });
    fireEvent.pointerMove(nextCard, { pointerId: 3, clientX: 100, clientY: 560 });
    fireEvent.pointerUp(nextCard, { pointerId: 3, clientX: 100, clientY: 560 });
    expect(least.querySelector('[data-location="least"]')).toBeTruthy();
  });

  it("keeps touch movement separate from direct tap placement", () => {
    render(<App />);
    const card = screen.getAllByTestId("value-card")[0];
    fireEvent.pointerDown(card, {
      pointerId: 4,
      pointerType: "touch",
      clientX: 120,
      clientY: 300,
    });
    expect(card.classList.contains("value-card--dragging")).toBe(false);
    fireEvent.pointerMove(card, {
      pointerId: 4,
      pointerType: "touch",
      clientX: 120,
      clientY: 120,
    });
    fireEvent.pointerUp(card, {
      pointerId: 4,
      pointerType: "touch",
      clientX: 120,
      clientY: 120,
    });
    expect(screen.getByTestId("drop-most").querySelector('[data-location="most"]')).toBeNull();
    expect(screen.getByTestId("drop-least").querySelector('[data-location="least"]')).toBeNull();

    fireEvent.click(card);
    expect(screen.getByTestId("drop-most").querySelector('[data-location="most"]')).toBeTruthy();
    expect(screen.getByTestId("drop-least").querySelector('[data-location="least"]')).toBeNull();
  });

  it("uses six distinct position-bound backgrounds with the same accent", () => {
    render(<App />);
    expect(screen.queryByText(/card 0[1-6]/i)).toBeNull();
    const cards = screen.getAllByTestId("value-card");
    const backgrounds = cards.map((card) =>
      card.style.getPropertyValue("--card-background-image"),
    );
    const positions = cards.map((card) => card.dataset.backgroundPosition).sort();
    expect(new Set(backgrounds).size).toBe(6);
    expect(positions).toEqual(["0", "1", "2", "3", "4", "5"]);
    const accents = cards
      .map((card) => card.getAttribute("style")?.match(/--card-accent:\s*([^;]+)/)?.[1]);
    expect(new Set(accents).size).toBe(1);
  });

  it("completes the phase-aware flow and hands the Profile to the first Journal Entry", async () => {
    vi.useFakeTimers();
    const onStartJournal = vi.fn();
    const { unmount } = render(<App onStartJournal={onStartJournal} />);
    for (let setNumber = 1; setNumber <= 11; setNumber += 1) {
      const progress = screen.getByRole("progressbar", {
        name: `Values · ${setNumber} of 11`,
      });
      expect(progress.getAttribute("aria-valuenow")).toBe(String(setNumber));
      expect(progress.getAttribute("aria-valuemax")).toBe("11");
      if (setNumber === 11) {
        expect(
          (progress.querySelector(".progress__track span") as HTMLElement)
            .style.width,
        ).toBe("100%");
      }
      answerSet();
      expect(screen.queryByRole("heading", { name: "A pattern is beginning to appear." })).toBeNull();
    }
    expect(screen.getByLabelText("Your compass")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "What sits at the center." })).toBeTruthy();
    expect(screen.queryByText("What brought you here right now?")).toBeNull();
    expect(screen.queryByText(/^0[1-9]$/)).toBeNull();
    expect(
      screen.getByText(
        "This result reflects the Most and Least choices you made most consistently across all 11 groups.",
      ),
    ).toBeTruthy();
    const summaryInspect = screen.getByRole("button", { name: "Inspect" });
    expect(summaryInspect.getAttribute("aria-disabled")).toBe("false");
    fireEvent.click(summaryInspect);
    expect(
      screen.getByRole("heading", {
        name: "Begin with the recorded choices.",
      }),
    ).toBeTruthy();
    expect(screen.getByText("Calculation method")).toBeTruthy();
    expect(screen.getByText("Deterministic · no model")).toBeTruthy();
    expect(
      screen.getByRole("region", {
        name: "Recorded Most and Least selections",
      }),
    ).toBeTruthy();
    expect(screen.getByText(/Confirm the result in Experience/)).toBeTruthy();
    expect(screen.queryByText("Profile confirmed")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Return to Experience" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm my compass" }));
    expect(screen.getByRole("heading", { name: "Your compass is ready." })).toBeTruthy();
    expect(
      screen.getByRole("region", { name: "Your Core Values" }),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: "Inspect" }).getAttribute("aria-disabled")).toBe("false");
    expect(screen.queryByRole("button", { name: /start again/i })).toBeNull();
    expect(screen.queryByText(/profile JSON/i)).toBeNull();
    await act(async () => undefined);
    expect(createExperienceSession).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Inspect" }));
    expect(screen.getByText("11 of 11 questions complete")).toBeTruthy();
    expect(screen.getByText("22 recorded selections")).toBeTruthy();
    expect(screen.getByText("Python validation recorded")).toBeTruthy();
    expect(screen.getByText("Profile confirmed")).toBeTruthy();
    expect(screen.queryByText("Canonical contract fixture")).toBeNull();
    expect(screen.queryByText("Journal Entry submitted")).toBeNull();
    expect(screen.queryByText("Nudge suppression checked")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Return to Experience" }));
    fireEvent.click(screen.getByRole("button", { name: "Start my first Journal Entry" }));
    expect(onStartJournal).toHaveBeenCalledTimes(1);
    expect(onStartJournal.mock.calls[0][0].user_confirmed).toBe(true);
    const journalHeading = screen.getByRole("heading", {
      name: "When did you feel most like yourself?",
    });
    expect(journalHeading).toBeTruthy();
    expect(
      screen.getByRole("region", { name: "Your Core Values" }),
    ).toBeTruthy();
    expect(document.activeElement).toBe(journalHeading);
    const journal = screen.getByRole("textbox", { name: "First Journal Entry" });
    fireEvent.change(journal, { target: { value: "A quiet walk helped me think clearly." } });
    expect(screen.queryByLabelText("Your compass")).toBeNull();
    const inspect = screen.getByRole("button", { name: "Inspect" });
    inspect.focus();
    vi.useRealTimers();
    const user = userEvent.setup();
    await user.keyboard("{Enter}");
    expect(
      screen.getByRole("heading", {
        name: "See how each trade-off shaped this Profile.",
      }),
    ).toBeTruthy();
    expect(document.activeElement).toBe(
      screen.getByRole("heading", {
        name: "See how each trade-off shaped this Profile.",
      }),
    );
    expect(screen.getByText("Python validation recorded")).toBeTruthy();
    expect(screen.queryByText("Journal Entry submitted")).toBeNull();
    expect(screen.queryByText("Nudge decided")).toBeNull();
    expect(onStartJournal).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByRole("button", { name: "Experience" }));
    expect((screen.getByRole("textbox", { name: "First Journal Entry" }) as HTMLTextAreaElement).value)
      .toBe("A quiet walk helped me think clearly.");
    const stored = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY)!);
    expect(stored.confirmed_profile.user_confirmed).toBe(true);
    expect(stored.experience.active_view).toBe("experience");
    expect(stored.experience.journal_started).toBe(true);
    expect(stored.experience.journal_draft).toBe("A quiet walk helped me think clearly.");
    expect(stored.confirmed_profile.bws_responses).toHaveLength(11);
    expect(stored.confirmed_profile.bws_results.scores).toHaveProperty(
      "universalism_nature",
    );
    expect(stored.confirmed_profile.value_profile.scores).toHaveProperty(
      "universalism",
    );
    expect(stored.confirmed_profile).not.toHaveProperty("goal_category");
    expect(stored.confirmed_profile).not.toHaveProperty("confidence");

    fireEvent.click(screen.getByRole("button", { name: "Inspect" }));
    unmount();
    render(<App onStartJournal={onStartJournal} />);
    expect(
      screen.getByRole("heading", {
        name: "See how each trade-off shaped this Profile.",
      }),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Experience" }));
    expect((screen.getByRole("textbox", { name: "First Journal Entry" }) as HTMLTextAreaElement).value)
      .toBe("A quiet walk helped me think clearly.");
    expect(onStartJournal).toHaveBeenCalledTimes(1);
  });

  it("keeps an open-week Journal Entry and Inspect at the same event boundary", async () => {
    vi.useFakeTimers();
    let openWeekEvents: TraceEventContract[] = [];
    vi.mocked(readExperienceTrace).mockImplementation(async (sessionId) => ({
      schema_version: canonicalInspectFixture.schema_version,
      operation: "read_trace",
      request_id: "read-open-week",
      status: "ok",
      session_id: sessionId,
      events: openWeekEvents.length > 0
        ? openWeekEvents
        : profileEvents.has(sessionId)
          ? [profileEvents.get(sessionId)!]
          : [],
    }));
    vi.mocked(submitJournalEntry).mockImplementation(async ({
      sessionId,
      entry,
    }) => {
      const profileEvent = profileEvents.get(sessionId)!;
      const journalTemplate = canonicalInspectFixture.trace_events.find(
        (event) => event.event_type === "journal_entry_submitted",
      )!;
      const suppressionTemplate = canonicalInspectFixture.trace_events.find(
        (event) => event.event_type === "nudge_suppression_checked",
      )!;
      const decisionTemplate = canonicalInspectFixture.trace_events.find(
        (event) => event.event_type === "nudge_decided",
      )!;
      const journalEvent: TraceEventContract = {
        ...journalTemplate,
        event_id: "journal-open-week",
        session_id: sessionId,
        parent_event_id: profileEvent.event_id,
        input_refs: [{ kind: "profile", id: sessionId }],
        result_refs: [{ kind: "journal_entry", id: entry.journal_entry_id }],
        details: { journal_entry: entry, ordering_valid: true },
      };
      const suppressionEvent: TraceEventContract = {
        ...suppressionTemplate,
        event_id: "suppression-open-week",
        session_id: sessionId,
        parent_event_id: journalEvent.event_id,
        input_refs: [{
          kind: "journal_entry",
          id: entry.journal_entry_id,
        }],
        details: {
          previous_entry_ids: [],
          window_size: 3,
          max_nudges: 2,
          suppressed: false,
        },
      };
      const decisionEvent: TraceEventContract = {
        ...decisionTemplate,
        event_id: "decision-open-week",
        session_id: sessionId,
        parent_event_id: suppressionEvent.event_id,
        input_refs: [{
          kind: "journal_entry",
          id: entry.journal_entry_id,
        }],
        details: {
          should_nudge: false,
          category: null,
          reason: "No follow-up question was useful.",
        },
      };
      openWeekEvents = [
        profileEvent,
        journalEvent,
        suppressionEvent,
        decisionEvent,
      ];
      return {
        schema_version: canonicalInspectFixture.schema_version,
        operation: "submit_journal_entry",
        request_id: "submit-open-week",
        status: "ok",
        session: {
          ...canonicalInspectFixture.session,
          session_id: sessionId,
          revision: 1,
          profile: profileEvent.details.profile,
          journal_entries: [entry],
          nudges: [{
            nudge_id: "nudge-open-week",
            journal_entry_id: entry.journal_entry_id,
            outcome: "no_nudge",
            category: null,
            reason: null,
            text: null,
            response: null,
          }],
          weekly_reviewer_decisions: [],
          drift_result: null,
          weekly_digest: null,
          trace_event_ids: openWeekEvents.map((event) => event.event_id),
          selection: { view: "experience" },
        },
        event_ids: openWeekEvents.slice(1).map((event) => event.event_id),
      } as JournalEntrySubmittedResponseContract;
    });

    render(<App />);
    for (let setNumber = 1; setNumber <= 11; setNumber += 1) {
      answerSet();
    }
    fireEvent.click(screen.getByRole("button", { name: "Confirm my compass" }));
    await act(async () => undefined);
    fireEvent.click(
      screen.getByRole("button", { name: "Start my first Journal Entry" }),
    );

    vi.useRealTimers();
    const user = userEvent.setup();
    const content = "I took a quiet walk and left my phone behind.";
    await user.type(
      screen.getByRole("textbox", { name: "First Journal Entry" }),
      content,
    );
    await user.click(
      screen.getByRole("button", { name: "Save Journal Entry" }),
    );

    expect(await screen.findByText(content)).toBeTruthy();
    expect(
      screen.queryByRole("heading", { name: "Your week in view." }),
    ).toBeNull();

    await user.click(screen.getByRole("button", { name: "Inspect" }));

    const trace = screen.getByRole("list", { name: "Trace events" });
    expect(within(trace).getAllByRole("listitem")).toHaveLength(4);
    [
      "Profile confirmed",
      "Journal Entry submitted",
      "Nudge suppression checked",
      "Nudge decided",
    ].forEach((label) => expect(screen.getByText(label)).toBeTruthy());
    [
      "Weekly review requested",
      "Weekly review completed",
      "Drift checked",
      "Weekly Digest built",
    ].forEach((label) => expect(screen.queryByText(label)).toBeNull());
  });

  it("never substitutes fixture events when Profile trace loading fails", async () => {
    vi.useFakeTimers();
    vi.mocked(createExperienceSession).mockRejectedValueOnce(
      new ExperienceApiError("Profile trace unavailable."),
    );
    render(<App />);
    for (let setNumber = 1; setNumber <= 11; setNumber += 1) {
      answerSet();
    }

    fireEvent.click(screen.getByRole("button", { name: "Confirm my compass" }));
    await act(async () => undefined);
    expect(
      screen.getByRole("button", { name: "Start my first Journal Entry" }),
    ).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Inspect" }));
    expect(screen.getByText("Python validation unavailable")).toBeTruthy();
    expect(screen.queryByText("Canonical contract fixture")).toBeNull();
    expect(screen.queryByText("Journal Entry submitted")).toBeNull();
    expect(screen.queryByText("Nudge decided")).toBeNull();
    expect(screen.getByRole("status")).toHaveProperty(
      "textContent",
      expect.stringContaining("Inspect trace could not be loaded"),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Retry Profile validation" }),
    );
    await act(async () => undefined);
    expect(screen.getByText("Python validation recorded")).toBeTruthy();
    expect(screen.getByText("Profile confirmed")).toBeTruthy();
  });

  it("does not carry a Profile trace failure into the Journal Entry status", async () => {
    vi.useFakeTimers();
    vi.mocked(createExperienceSession).mockRejectedValueOnce(
      new ExperienceApiError("Profile trace unavailable."),
    );
    render(<App />);
    for (let setNumber = 1; setNumber <= 11; setNumber += 1) {
      answerSet();
    }

    fireEvent.click(screen.getByRole("button", { name: "Confirm my compass" }));
    await act(async () => undefined);
    fireEvent.click(
      screen.getByRole("button", { name: "Start my first Journal Entry" }),
    );

    expect(
      screen.getByRole("textbox", { name: "First Journal Entry" }),
    ).toBeTruthy();
    expect(screen.getByRole("status").textContent).toBe("");
    expect(screen.queryByText(/Inspect trace could not be loaded/)).toBeNull();
  });

  it("does not block the Journal Entry while Profile synchronization is pending", async () => {
    vi.useFakeTimers();
    vi.mocked(createExperienceSession).mockImplementationOnce(
      () => new Promise(() => undefined),
    );
    render(<App />);
    for (let setNumber = 1; setNumber <= 11; setNumber += 1) {
      answerSet();
    }

    fireEvent.click(screen.getByRole("button", { name: "Confirm my compass" }));
    await act(async () => undefined);
    expect(createExperienceSession).toHaveBeenCalledTimes(1);
    fireEvent.click(
      screen.getByRole("button", { name: "Start my first Journal Entry" }),
    );

    const editor = screen.getByRole("textbox", {
      name: "First Journal Entry",
    }) as HTMLTextAreaElement;
    expect(editor.disabled).toBe(false);
    expect(screen.getByRole("status").textContent).toBe("");
  });
});
