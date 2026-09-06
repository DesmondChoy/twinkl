import activeReplayJson from "../public/scenarios/active-wei-jun.json";
import recoveredReplayJson from "../public/scenarios/recovered-marc.json";
import scenarioCatalogJson from "../public/scenarios/index.json";
import stableReplayJson from "../public/scenarios/stable-meera.json";
import twoValuesReplayJson from "../public/scenarios/two-values-lukas.json";
import uncertainReplayJson from "../public/scenarios/uncertain-noor.json";
import judgeSampleManifest from "../../../logs/experiments/reports/coach_digest_sample_20260824/judge_sample_manifest.json";
import { afterEach, describe, expect, it, vi } from "vitest";
import { validateExperienceInspectFixture } from "./demoContracts";
import { currentNorthStarEvent, displayableNorthStarSelection, northStarProfileRef } from "./northStar";
import { BWS_SETS } from "./domain";
import { createExperienceState, createSession, parseSession, persistSession, SESSION_STORAGE_KEY, type OnboardingSession } from "./session";
import {
  loadScenarioCatalog,
  projectScenarioWeek,
  validateScenarioCatalog,
} from "./scenarioReplay";

afterEach(() => {
  vi.unstubAllGlobals();
});

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

  it("does not reuse cached scenario files", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(scenarioCatalogJson), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await loadScenarioCatalog();

    expect(fetchMock).toHaveBeenCalledWith("/scenarios/index.json", {
      cache: "no-store",
    });
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

  it.each([
    ["Wei Jun", activeReplayJson], ["Marc", recoveredReplayJson],
    ["Meera", stableReplayJson], ["Lukas", twoValuesReplayJson],
    ["Noor", uncertainReplayJson],
  ])("preserves %s replay without precomputed North Star Moment results", async (_name, scenarioJson) => {
    const saved = validateExperienceInspectFixture(scenarioJson);
    const profile = saved.scenario.profile;
    const profileRef = await northStarProfileRef(profile);
    for (let index = 0; index < saved.scenario.weeks.length; index += 1) {
      const projected = projectScenarioWeek(saved, index);
      const result = currentNorthStarEvent({
        events: projected.events, profile, profileRef,
        weeklyDigest: projected.session.weekly_digest,
        journalEntries: projected.session.journal_entries,
      });
      expect(result, `${saved.scenario.scenario_id} week ${index + 1}`).toBeNull();
      expect(projected.events.some((event) => event.event_type === "north_star_reviewed")).toBe(false);
      expect(displayableNorthStarSelection(result, profile,
        projected.session.journal_entries, projected.session.drift_result!)).toBeNull();
      expect(projected.session.weekly_digest).not.toBeNull();
      const futureEventIds = new Set(saved.scenario.weeks.slice(index + 1).flatMap((week) => week.event_ids));
      expect(projected.events.some((event) => futureEventIds.has(event.event_id))).toBe(false);
    }
    const finalIndex = saved.scenario.weeks.length - 1;
    const final = projectScenarioWeek(saved, finalIndex);
    const session: OnboardingSession = {
      ...createSession(), user_id: profile.user_id, session_id: profile.session_id,
      preferred_name: profile.preferred_name ?? "Friend", started_at: profile.started_at,
      stage: "complete", set_index: BWS_SETS.length - 1,
      set_order: BWS_SETS.map((_, index) => index),
      displayed_orders: BWS_SETS.map((set) => [...profile.bws_responses.find((response) => response.set_number === set.setNumber)!.item_order_shown]),
      responses: profile.bws_responses,
      selected_top_values: profile.value_profile.top_values.length > 2 ? [...profile.top_values] : [],
      confirmed_profile: profile,
      experience: {
        ...createExperienceState(), journal_started: true, revision: final.session.revision,
        journal_entries: final.session.journal_entries, nudges: final.session.nudges,
        selected_persona_id: saved.scenario.persona_id, selected_week: finalIndex,
        weekly_reviewer_decisions: final.session.weekly_reviewer_decisions,
        drift_result: final.session.drift_result, weekly_digest: final.session.weekly_digest,
        run_state: "complete", trace_events: final.events, trace_event_ids: final.session.trace_event_ids,
      },
    };
    expect(persistSession(session)).toBe(true);
    const restored = parseSession(localStorage.getItem(SESSION_STORAGE_KEY));
    expect(restored?.experience.trace_events).toEqual(final.events);
    expect(restored?.confirmed_profile).toEqual(profile);
  });
});
