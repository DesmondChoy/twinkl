import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import LivingCompass, {
  compassMotionEnabled,
  getCompassSegmentLayout,
  getCompassMotionTuning,
  getStarGlowFrame,
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

  it("reduces swing energy as calibration progresses", () => {
    const early = getCompassMotionTuning(0, false);
    const middle = getCompassMotionTuning(4, false);
    const late = getCompassMotionTuning(8, false);
    const settled = getCompassMotionTuning(0, true);

    expect(early.kickDegreesPerSecond).toBeGreaterThan(
      middle.kickDegreesPerSecond,
    );
    expect(middle.kickDegreesPerSecond).toBeGreaterThan(
      late.kickDegreesPerSecond,
    );
    expect(early.damping).toBeLessThan(middle.damping);
    expect(middle.damping).toBeLessThan(late.damping);
    expect(settled.kickDegreesPerSecond).toBeLessThan(
      early.kickDegreesPerSecond,
    );
    expect(getCompassMotionTuning(null, false).kickDegreesPerSecond).toBe(0);
  });

  it("centers its north gap and advances segments clockwise", () => {
    const first = getCompassSegmentLayout(0);
    const second = getCompassSegmentLayout(1);
    const last = getCompassSegmentLayout(10);
    const firstEnd = first.thetaStart + first.thetaLength;
    const lastStart = last.thetaStart + Math.PI * 2;

    expect((firstEnd + lastStart) / 2).toBeCloseTo(Math.PI / 2, 6);
    expect(second.thetaStart).toBeLessThan(first.thetaStart);
    expect(second.svgRotationDegrees).toBeGreaterThan(
      first.svgRotationDegrees,
    );
    expect(first.svgDashLength).toBeCloseTo(8.056, 3);
  });

  it("keeps the completed north star glow pulsing", () => {
    const start = getStarGlowFrame(0);
    const peak = getStarGlowFrame(900);
    const nextCycle = getStarGlowFrame(1_800);
    const nextPeak = getStarGlowFrame(2_700);

    expect(start.active).toBe(true);
    expect(peak.opacity).toBeGreaterThan(start.opacity);
    expect(peak.scale).toBeGreaterThan(start.scale);
    expect(nextCycle.active).toBe(true);
    expect(nextCycle.opacity).toBeCloseTo(start.opacity);
    expect(nextCycle.scale).toBeCloseTo(start.scale);
    expect(nextPeak.opacity).toBeCloseTo(peak.opacity);
    expect(nextPeak.scale).toBeCloseTo(peak.scale);
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
    expect(compass.querySelectorAll(".living-compass__anchor")).toHaveLength(0);
    const segments = compass.querySelectorAll<SVGCircleElement>(
      ".living-compass__segment",
    );
    expect(segments[0]?.style.getPropertyValue("--segment-angle")).toBe(
      `${getCompassSegmentLayout(0).svgRotationDegrees}deg`,
    );
    expect(segments[0]?.getAttribute("stroke-dasharray")).toBe(
      `${getCompassSegmentLayout(0).svgDashLength} ${100 - getCompassSegmentLayout(0).svgDashLength}`,
    );
  });

  it.each([
    ["name", 0, null, 0, 0, 180],
    ["first question", 1, 0, 0, 1, -90],
    ["middle question", 6, 5, 5, 1, 60],
    ["final question", 11, 10, 10, 1, -2],
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

  it("keeps its heading while draft choices settle", () => {
    const noChoice = renderCompass(1, 0);
    const mostOnly = renderCompass(1, 0, true);
    const both = renderCompass(1, 0, true, true);

    expect(getNeedleAngle(noChoice.compass)).toBeCloseTo(-90);
    expect(getNeedleAngle(mostOnly.compass)).toBeCloseTo(-90);
    expect(getNeedleAngle(both.compass)).toBeCloseTo(-90);
    expect(
      both.compass.querySelector(".living-compass__needle--settled"),
    ).not.toBeNull();
  });

  it.each([
    ["Most only", true, false, 1, 0],
    ["Least only", false, true, 0, 1],
    ["Most and Least", true, true, 1, 1],
  ])(
    "shows opposing needle halves for %s",
    (_, mostSelected, leastSelected, mostCount, leastCount) => {
      const { compass } = renderCompass(
        1,
        0,
        mostSelected,
        leastSelected,
      );

      expect(
        compass.querySelectorAll(
          ".living-compass__needle-half--north.living-compass__needle-half--active",
        ),
      ).toHaveLength(mostCount);
      expect(
        compass.querySelectorAll(
          ".living-compass__needle-half--south.living-compass__needle-half--active",
        ),
      ).toHaveLength(leastCount);
    },
  );
});
