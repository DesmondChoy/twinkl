import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

vi.stubGlobal("confirm", () => true);

async function answerSet(user: ReturnType<typeof userEvent.setup>) {
  const first = screen.getAllByTestId("value-card").find((card) => card.dataset.location === "pool")!;
  first.focus();
  await user.keyboard("m");
  const second = screen.getAllByTestId("value-card").find((card) => card.dataset.location === "pool")!;
  second.focus();
  await user.keyboard("l");
  await user.click(screen.getByRole("button", { name: "Continue" }));
}

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
  });

  it("lets touch users tap a card, choose a box, and reconsider", async () => {
    const user = userEvent.setup();
    render(<App />);
    const first = screen.getAllByTestId("value-card")[0];
    const firstPhrase = first.querySelector(".value-card__phrase")!.textContent!;
    await user.click(first);
    expect(first.getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText(/where does this principle sit/i)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: `Place ${firstPhrase} in Most` }));
    expect(screen.getByTestId("drop-most").querySelector('[data-location="most"]')).toBeTruthy();

    const second = screen.getAllByTestId("value-card").find((card) => card.dataset.location === "pool")!;
    const secondPhrase = second.querySelector(".value-card__phrase")!.textContent!;
    await user.click(second);
    await user.click(screen.getByRole("button", { name: `Place ${secondPhrase} in Least` }));
    expect(screen.getByTestId("drop-least").querySelector('[data-location="least"]')).toBeTruthy();
    expect(screen.getByRole("button", { name: "Continue" }).hasAttribute("disabled")).toBe(false);

    const placedMost = screen.getByTestId("drop-most").querySelector<HTMLElement>('[data-location="most"]')!;
    await user.click(placedMost);
    expect(screen.getByTestId("drop-most").textContent).toContain("Drop a card here");
    expect(screen.getByRole("button", { name: "Continue" }).hasAttribute("disabled")).toBe(true);
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
    const poolCard = screen.getAllByTestId("value-card").find((card) => card.dataset.location === "pool")!;
    poolCard.focus();
    await user.keyboard("l");
    const leastCard = screen.getByTestId("drop-least").querySelector<HTMLElement>('[data-location="least"]')!;
    expect(leastCard).toBeTruthy();
    expect(document.activeElement).toBe(leastCard);
    expect(screen.getByRole("button", { name: "Continue" }).hasAttribute("disabled")).toBe(false);
    mostCard.focus();
    await user.keyboard("{Backspace}");
    expect(screen.getByTestId("drop-most").textContent).toContain("Drop a card here");
    expect(screen.getByRole("button", { name: "Continue" }).hasAttribute("disabled")).toBe(true);
    const returnedCard = screen.getByTestId("selection-area").querySelector<HTMLElement>(
      `[data-value="${firstValue}"][data-location="pool"]`,
    );
    expect(document.activeElement).toBe(returnedCard);
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
    const user = userEvent.setup();
    const onStartJournal = vi.fn();
    render(<App onStartJournal={onStartJournal} />);
    for (let setNumber = 1; setNumber <= 11; setNumber += 1) {
      expect(screen.getByLabelText(`Values · ${setNumber} of 11`)).toBeTruthy();
      await answerSet(user);
      expect(screen.queryByRole("heading", { name: "A pattern is beginning to appear." })).toBeNull();
    }
    const goal = screen.getByRole("radio", {
      name: "I feel stuck or unclear about my direction",
    });
    expect(screen.getByLabelText("Your focus")).toBeTruthy();
    fireEvent.click(goal);
    await user.click(screen.getByRole("button", { name: "See my compass" }));
    expect(screen.getByLabelText("Your compass")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "What sits at the center." })).toBeTruthy();
    expect(screen.queryByText(/^0[1-9]$/)).toBeNull();
    await user.click(screen.getByRole("button", { name: "Set my compass" }));
    expect(screen.getByRole("heading", { name: "Your compass is ready." })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /start again/i })).toBeNull();
    expect(screen.queryByText(/profile JSON/i)).toBeNull();
    await user.click(screen.getByRole("button", { name: "Start my first Journal Entry" }));
    expect(onStartJournal).toHaveBeenCalledTimes(1);
    expect(onStartJournal.mock.calls[0][0].user_confirmed).toBe(true);
    expect(screen.getByRole("heading", { name: "When did you feel most like yourself?" })).toBeTruthy();
    expect(screen.getByRole("textbox", { name: "First Journal Entry" })).toBeTruthy();
    expect(screen.queryByLabelText("Your compass")).toBeNull();
    await waitFor(() => {
      const stored = JSON.parse(localStorage.getItem("twinkl.onboarding.session.v4")!);
      expect(stored.confirmed_profile.user_confirmed).toBe(true);
      expect(stored.confirmed_profile.bws_responses).toHaveLength(11);
      expect(stored.confirmed_profile.bws_results.scores).toHaveProperty(
        "universalism_nature",
      );
      expect(stored.confirmed_profile.value_profile.scores).toHaveProperty(
        "universalism",
      );
      expect(stored.confirmed_profile).not.toHaveProperty("confidence");
    });
  });
});
