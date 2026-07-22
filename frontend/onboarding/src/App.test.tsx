import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { SESSION_STORAGE_KEY } from "./session";

vi.stubGlobal("confirm", () => true);

function answerSet() {
  const first = screen.getAllByTestId("value-card").find((card) => card.dataset.location === "pool")!;
  fireEvent.click(first);
  const second = screen.getAllByTestId("value-card").find((card) => card.dataset.location === "pool")!;
  fireEvent.click(second);
  act(() => vi.advanceTimersByTime(1_000));
}

afterEach(() => vi.useRealTimers());

describe("onboarding app", () => {
  it("opens directly on the first six-card set without Schwartz labels", () => {
    render(<App />);
    expect(screen.getByLabelText("Values · 1 of 11")).toBeTruthy();
    expect(screen.getAllByTestId("value-card")).toHaveLength(6);
    expect(screen.getByText("Next step")).toBeTruthy();
    expect(screen.getByText(/across 11 groups/i)).toBeTruthy();
    expect(screen.queryByText(/saved only in this browser/i)).toBeNull();
    expect(screen.queryByText("Hedonism")).toBeNull();
    expect(screen.queryByText("Universalism")).toBeNull();
    const inspect = screen.getByRole("button", { name: /inspect/i });
    expect(inspect.getAttribute("aria-disabled")).toBe("true");
    expect(screen.getByText("Available after Profile confirmation")).toBeTruthy();
    fireEvent.click(inspect);
    expect(screen.queryByRole("heading", { name: "The trail starts here." })).toBeNull();
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
      expect(screen.getByLabelText(`Values · ${setNumber} of 11`)).toBeTruthy();
      answerSet();
      expect(screen.queryByRole("heading", { name: "A pattern is beginning to appear." })).toBeNull();
    }
    const goal = screen.getByRole("radio", {
      name: "I feel stuck or unclear about my direction",
    });
    expect(screen.getByLabelText("Your focus")).toBeTruthy();
    fireEvent.click(goal);
    fireEvent.click(screen.getByRole("button", { name: "See my compass" }));
    expect(screen.getByLabelText("Your compass")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "What sits at the center." })).toBeTruthy();
    expect(screen.queryByText(/^0[1-9]$/)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Set my compass" }));
    expect(screen.getByRole("heading", { name: "Your compass is ready." })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Inspect" }).getAttribute("aria-disabled")).toBe("false");
    expect(screen.queryByRole("button", { name: /start again/i })).toBeNull();
    expect(screen.queryByText(/profile JSON/i)).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Start my first Journal Entry" }));
    expect(onStartJournal).toHaveBeenCalledTimes(1);
    expect(onStartJournal.mock.calls[0][0].user_confirmed).toBe(true);
    expect(screen.getByRole("heading", { name: "When did you feel most like yourself?" })).toBeTruthy();
    const journal = screen.getByRole("textbox", { name: "First Journal Entry" });
    fireEvent.change(journal, { target: { value: "A quiet walk helped me think clearly." } });
    expect(screen.queryByLabelText("Your compass")).toBeNull();
    const inspect = screen.getByRole("button", { name: "Inspect" });
    inspect.focus();
    vi.useRealTimers();
    const user = userEvent.setup();
    await user.keyboard("{Enter}");
    expect(screen.getByRole("heading", { name: "The trail starts here." })).toBeTruthy();
    expect(document.activeElement).toBe(screen.getByRole("heading", { name: "The trail starts here." }));
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
    expect(stored.confirmed_profile).not.toHaveProperty("confidence");

    fireEvent.click(screen.getByRole("button", { name: "Inspect" }));
    unmount();
    render(<App onStartJournal={onStartJournal} />);
    expect(screen.getByRole("heading", { name: "The trail starts here." })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Experience" }));
    expect((screen.getByRole("textbox", { name: "First Journal Entry" }) as HTMLTextAreaElement).value)
      .toBe("A quiet walk helped me think clearly.");
    expect(onStartJournal).toHaveBeenCalledTimes(1);
  });
});
