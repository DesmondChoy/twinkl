import activeReplayJson from "../public/scenarios/active-wei-jun.json";
import recoveredReplayJson from "../public/scenarios/recovered-marc.json";
import scenarioCatalogJson from "../public/scenarios/index.json";
import stableReplayJson from "../public/scenarios/stable-meera.json";
import twoValuesReplayJson from "../public/scenarios/two-values-lukas.json";
import uncertainReplayJson from "../public/scenarios/uncertain-noor.json";
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
        "stable",
        "active_drift",
        "recovered_drift",
        "uncertain",
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
    expect(final.session.drift_result?.delivery_state).toBe("active");
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
