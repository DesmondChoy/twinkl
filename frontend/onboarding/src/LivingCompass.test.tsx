import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import LivingCompass, {
  compassMotionEnabled,
} from "./LivingCompass";

function renderCompass(
  milestone: number,
  currentQuestionIndex: number | null,
  mostSelected = false,
  leastSelected = false,
) {
  const result = render(
    <LivingCompass
      currentQuestionIndex={currentQuestionIndex}
      leastSelected={leastSelected}
      milestone={milestone}
      mostSelected={mostSelected}
    />,
  );
  const compass = result.container.querySelector(".living-compass")!;
  return { ...result, compass };
}

function getNeedleAngle(compass: Element) {
  const needle = compass.querySelector<SVGGElement>(".living-compass__needle")!;
  return Number.parseFloat(needle.style.getPropertyValue("--needle-angle"));
}

describe("LivingCompass", () => {
  it("disables needle motion for reduced-motion preferences", () => {
    expect(compassMotionEnabled(true)).toBe(false);
    expect(compassMotionEnabled(false)).toBe(true);
  });

  it("keeps its canvas decorative and shows a WebGL fallback", () => {
    const { compass } = renderCompass(0, null);

    expect(compass.getAttribute("aria-hidden")).toBe("true");
    expect(compass.classList.contains("living-compass--fallback")).toBe(true);
    expect(
      compass.querySelector("canvas")?.getAttribute("aria-hidden"),
    ).toBe("true");
    expect(
      compass.querySelectorAll(".living-compass__segment"),
    ).toHaveLength(11);
    expect(
      compass.querySelectorAll(".living-compass__north-star"),
    ).toHaveLength(1);
    expect(compass.querySelectorAll(".living-compass__needle")).toHaveLength(1);
  });

  it.each([
    ["name", 0, null, 0, 0, 38],
    ["first question", 1, 0, 0, 1, 41],
    ["middle question", 6, 5, 5, 1, 23.727],
    ["final question", 11, 10, 10, 1, 6.455],
    ["completed Profile", 12, null, 11, 0, 0],
  ])(
    "shows progress for the %s state",
    (
      _,
      milestone,
      currentQuestionIndex,
      completedCount,
      currentCount,
      needleAngle,
    ) => {
      const { compass } = renderCompass(milestone, currentQuestionIndex);

      expect(
        compass.querySelectorAll(".living-compass__segment--complete"),
      ).toHaveLength(completedCount);
      expect(
        compass.querySelectorAll(".living-compass__segment--current"),
      ).toHaveLength(currentCount);
      expect(getNeedleAngle(compass)).toBeCloseTo(needleAngle, 2);
    },
  );

  it("guides the needle toward north as draft choices settle", () => {
    const noChoice = renderCompass(1, 0);
    const mostOnly = renderCompass(1, 0, true);
    const both = renderCompass(1, 0, true, true);

    expect(getNeedleAngle(noChoice.compass)).toBeCloseTo(41);
    expect(getNeedleAngle(mostOnly.compass)).toBeCloseTo(39.4);
    expect(getNeedleAngle(both.compass)).toBeCloseTo(38);
    expect(
      both.compass.querySelector(".living-compass__needle--settled"),
    ).not.toBeNull();
  });

  it.each([
    ["Most only", true, false, 1, 0],
    ["Least only", false, true, 0, 1],
    ["Most and Least", true, true, 1, 1],
  ])(
    "shows opposing anchors for %s",
    (_, mostSelected, leastSelected, mostCount, leastCount) => {
      const { compass } = renderCompass(
        1,
        0,
        mostSelected,
        leastSelected,
      );

      expect(
        compass.querySelectorAll(
          ".living-compass__anchor--most.living-compass__anchor--selected",
        ),
      ).toHaveLength(mostCount);
      expect(
        compass.querySelectorAll(
          ".living-compass__anchor--least.living-compass__anchor--selected",
        ),
      ).toHaveLength(leastCount);
      expect(
        compass.querySelectorAll(".living-compass__anchor--settled"),
      ).toHaveLength(mostSelected && leastSelected ? 2 : 0);
    },
  );
});
