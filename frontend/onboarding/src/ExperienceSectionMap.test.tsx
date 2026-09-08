import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ExperienceSectionMap, {
  type ExperienceSectionMapView,
} from "./ExperienceSectionMap";

const EXPECTED_LINKS: Partial<Record<ExperienceSectionMapView, string[]>> = {
  summary: ["Profile", "Confirm"],
  complete: ["Profile", "Journal Entry"],
  journal: ["Prompt", "Write"],
  inspect: ["Summary", "Recorded work"],
};

describe("Experience section map", () => {
  it("explains the Persona demo without inactive navigation links", () => {
    render(<ExperienceSectionMap view="persona-picker" />);
    expect(screen.queryByRole("navigation")).toBeNull();
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.getByRole("heading", { name: "See how Twinkl works" })).toBeTruthy();
    expect(screen.getByText("Read a Persona’s saved Journal Entries.")).toBeTruthy();
    expect(screen.getByText("See the weekly results.")).toBeTruthy();
    expect(screen.getByText("Open Inspect to check which entries support those results.")).toBeTruthy();
  });

  it.each(Object.entries(EXPECTED_LINKS))(
    "shows the %s sections",
    (view, labels) => {
      render(
        <ExperienceSectionMap hasJournalComposer view={view as ExperienceSectionMapView} />,
      );

      const navigation = screen.getByRole("navigation", {
        name: "Experience sections",
      });
      expect(within(navigation).getAllByRole("link")).toHaveLength(
        labels.length,
      );
      labels.forEach((label) => {
        expect(
          within(navigation).getByRole("link", { name: label }),
        ).toBeTruthy();
      });
    },
  );

  it("adds the Journal Entries link after the first saved entry", () => {
    render(<ExperienceSectionMap hasJournalEntries view="journal" />);

    expect(
      screen.getByRole("link", { name: "Journal Entries" }).getAttribute(
        "href",
      ),
    ).toBe("#journal-thread-title");
  });

  it("shows Write only while the Journal Entry composer is available", () => {
    const { rerender } = render(
      <ExperienceSectionMap hasJournalEntries hasWeeklyResult view="journal" />,
    );
    expect(screen.queryByRole("link", { name: "Write" })).toBeNull();
    expect(screen.getByRole("link", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.getByRole("link", { name: "Weekly Drift" })).toBeTruthy();

    rerender(<ExperienceSectionMap hasJournalComposer hasJournalEntries hasWeeklyResult view="journal" />);
    expect(screen.getByRole("link", { name: "Write" }).getAttribute("href"))
      .toBe("#experience-journal-compose");

    rerender(<ExperienceSectionMap hasJournalEntries hasWeeklyResult view="journal" />);
    expect(screen.queryByRole("link", { name: "Write" })).toBeNull();
  });

  it("adds Weekly Drift only when the result is present", () => {
    render(<ExperienceSectionMap hasWeeklyResult view="journal" />);

    expect(
      screen.getByRole("link", { name: "Weekly Drift" }).getAttribute("href"),
    ).toBe("#experience-weekly");
  });
});
