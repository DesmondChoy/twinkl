import activeReplayJson from "../public/scenarios/active-wei-jun.json";
import recoveredReplayJson from "../public/scenarios/recovered-marc.json";
import scenarioCatalogJson from "../public/scenarios/index.json";
import stableReplayJson from "../public/scenarios/stable-meera.json";
import twoValuesReplayJson from "../public/scenarios/two-values-lukas.json";
import uncertainReplayJson from "../public/scenarios/uncertain-noor.json";
import judgeSampleManifest from "../../../logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json";
import { describe, expect, it } from "vitest";
import { validateExperienceInspectFixture } from "./demoContracts";
import {
  projectScenarioWeek,
  validateScenarioCatalog,
} from "./scenarioReplay";

describe("saved persona replay", () => {
  const fixture = validateExperienceInspectFixture(activeReplayJson);

  it("loads the five checked-in scenario roles with one recommendation", () => {
    const catalog = validateScenarioCatalog(scenarioCatalogJson);

    expect(catalog.scenarios).toHaveLength(5);
    expect(catalog.scenarios.filter((item) => item.recommended)).toHaveLength(1);
    expect(new Set(catalog.scenarios.map((item) => item.role))).toEqual(
      new Set([
        "no_active_drift",
        "active_drift",
        "drift_ended",
        "insufficient_evidence",
        "two_core_values",
      ]),
    );
  });

  it("projects only information available through the selected week", () => {
    const first = projectScenarioWeek(fixture, 0);
    const final = projectScenarioWeek(
      fixture,
      fixture.scenario.weeks.length - 1,
    );
    const futureEntries = fixture.scenario.journal_entries.filter(
      (entry) =>
        !first.session.journal_entries.some(
          (visible) => visible.journal_entry_id === entry.journal_entry_id,
        ),
    );
    const firstJson = JSON.stringify(first);

    expect(first.session.selection.selected_week).toBe(
      fixture.scenario.weeks[0].week_id,
    );
    expect(first.events.every((event) => event.source === "saved_replay")).toBe(
      true,
    );
    expect(
      futureEntries.every((entry) => !firstJson.includes(entry.content)),
    ).toBe(true);
    expect(final.session.journal_entries).toEqual(
      fixture.scenario.journal_entries,
    );
    expect(final.session.drift_result?.delivery_state).toBe("active_drift");
  });

  it("is deterministic and rejects week boundaries", () => {
    expect(projectScenarioWeek(fixture, 2)).toEqual(
      projectScenarioWeek(fixture, 2),
    );
    expect(() => projectScenarioWeek(fixture, -1)).toThrow(
      "Unknown saved replay week",
    );
    expect(() =>
      projectScenarioWeek(fixture, fixture.scenario.weeks.length),
    ).toThrow("Unknown saved replay week");
  });

  it.each([
    ["two-values-lukas", twoValuesReplayJson, "2025-10-13"],
    ["stable-meera", stableReplayJson, "2025-09-15"],
    ["active-wei-jun", activeReplayJson, "2025-06-30"],
    ["recovered-marc", recoveredReplayJson, "2025-03-17"],
    ["uncertain-noor", uncertainReplayJson, "2025-04-14"],
  ])(
    "reveals the evaluated Coach Digest only in the %s key week",
    (scenarioId, scenarioJson, weekStart) => {
      const scenarioFixture = validateExperienceInspectFixture(scenarioJson);
      const keyWeekIndex = scenarioFixture.scenario.weeks.findIndex(
        (week) => week.week_start === weekStart,
      );
      const manifestEntry = judgeSampleManifest.find(
        (entry) => entry.provenance.scenario_id === scenarioId,
      );

      expect(keyWeekIndex).toBeGreaterThanOrEqual(0);
      expect(manifestEntry).toBeTruthy();
      for (let index = 0; index < keyWeekIndex; index += 1) {
        const earlier = projectScenarioWeek(scenarioFixture, index);
        expect(earlier.session.weekly_digest?.coach_narrative).toBeNull();
        expect(JSON.stringify(earlier)).not.toContain(
          manifestEntry!.narrative.weekly_mirror,
        );
      }

      const keyWeek = projectScenarioWeek(scenarioFixture, keyWeekIndex);
      expect(keyWeek.session.weekly_digest?.coach_narrative).toEqual(
        manifestEntry!.narrative,
      );
      const coachEvents = keyWeek.events.filter(
        (event) => event.event_type === "weekly_coach_generated",
      );
      expect(coachEvents).toHaveLength(1);
      expect(coachEvents[0].details.narrative).toEqual(
        manifestEntry!.narrative,
      );
    },
  );

  it.each([
    activeReplayJson,
    recoveredReplayJson,
    stableReplayJson,
    twoValuesReplayJson,
    uncertainReplayJson,
  ])("matches the checked-in Python first-week projection", (scenarioJson) => {
    const scenarioFixture = validateExperienceInspectFixture(scenarioJson);

    expect(projectScenarioWeek(scenarioFixture, 0).session).toEqual(
      scenarioFixture.session,
    );
  });
});
