import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import AssessmentSectionMap, {
  ASSESSMENT_SECTIONS,
} from "./AssessmentSectionMap";

describe("assessment section map", () => {
  it("links every assessment section and follows the reading position", () => {
    render(
      <>
        <AssessmentSectionMap />
        {ASSESSMENT_SECTIONS.map((section) => (
          <section id={section.id} key={section.id} />
        ))}
      </>,
    );

    ASSESSMENT_SECTIONS.forEach((section) => {
      expect(
        screen.getByRole("link", { name: section.label }).getAttribute("href"),
      ).toBe(`#${section.id}`);
    });
    expect(
      screen.getByRole("link", { name: "Choices" }).getAttribute("aria-current"),
    ).toBe("location");

    const sectionTops = [-900, -620, -300, 80, 520];
    ASSESSMENT_SECTIONS.forEach((section, index) => {
      const target = document.getElementById(section.id)!;
      vi.spyOn(target, "getBoundingClientRect").mockReturnValue({
        top: sectionTops[index],
        height: 240,
      } as DOMRect);
    });

    fireEvent.scroll(window);

    expect(
      screen.getByRole("link", { name: "Profile" }).getAttribute("aria-current"),
    ).toBe("location");
    expect(
      screen.getByRole("link", { name: "Choices" }).getAttribute("aria-current"),
    ).toBeNull();
  });
});
