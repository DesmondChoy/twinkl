import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ExperienceSectionMap, {
  type ExperienceSectionMapView,
} from "./ExperienceSectionMap";

const EXPECTED_LINKS: Record<ExperienceSectionMapView, string[]> = {
  summary: ["Profile", "Confirm"],
  complete: ["Profile", "Journal Entry"],
  journal: ["Prompt", "Write"],
  "persona-picker": ["Introduction", "Personas", "Evidence"],
  "persona-replay": ["Persona", "Week", "Journal Entries", "Weekly Drift"],
  inspect: ["Summary", "Recorded work"],
};

describe("Experience section map", () => {
  it.each(Object.entries(EXPECTED_LINKS))(
    "shows the %s sections",
    (view, labels) => {
      render(
        <ExperienceSectionMap view={view as ExperienceSectionMapView} />,
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

  it("adds Weekly Drift only when the result is present", () => {
    render(<ExperienceSectionMap hasWeeklyResult view="journal" />);

    expect(
      screen.getByRole("link", { name: "Weekly Drift" }).getAttribute("href"),
    ).toBe("#experience-weekly");
  });
});
