import { describe, expect, it } from "vitest";
import activeReplay from "../public/scenarios/active-nisha.json";
import { savedCoachComparison, type CoachComparison } from "./coachComparison";
import { validateExperienceInspectFixture, type TraceEventContract } from "./demoContracts";

function comparisonEvent(): TraceEventContract {
  const fixture = validateExperienceInspectFixture(structuredClone(activeReplay));
  return fixture.trace_events.find((event) => event.event_type === "weekly_coach_generated"
    && event.details.comparison)!;
}

function pair(event: TraceEventContract): CoachComparison {
  return event.details.comparison as CoachComparison;
}

describe("comparison prompt version receipts", () => {
  it("keeps the historical saved comparison readable", () => {
    const event = comparisonEvent();
    expect(pair(event).without_north_star.prompt_version).toBe("1.0");
    expect(savedCoachComparison(event)).not.toBeNull();
  });

  it("requires the new voice check on both new-version arms", () => {
    const event = comparisonEvent();
    const { without_north_star: without, with_north_star: withMoment } = pair(event);
    without.prompt_version = "1.1";
    withMoment.prompt_version = "1.1";
    expect(savedCoachComparison(event)).toBeNull();
    for (const arm of [without, withMoment]) {
      (arm.validation.checks as unknown[]).push({
        name: "natural_reflection_voice", passed: true, details: "Voice checked.",
      });
    }
    expect(savedCoachComparison(event)).not.toBeNull();
    (withMoment.validation.checks as { name: string; passed: boolean }[])
      .find((check) => check.name === "natural_reflection_voice")!.passed = false;
    expect(savedCoachComparison(event)).toBeNull();
  });

  it("rejects mixed or unknown prompt versions", () => {
    const event = comparisonEvent();
    pair(event).with_north_star.prompt_version = "1.1";
    (pair(event).with_north_star.validation.checks as unknown[]).push({
      name: "natural_reflection_voice", passed: true,
    });
    expect(savedCoachComparison(event)).toBeNull();
    for (const arm of [pair(event).without_north_star, pair(event).with_north_star]) {
      arm.prompt_version = "2.0";
    }
    expect(savedCoachComparison(event)).toBeNull();
  });
});
