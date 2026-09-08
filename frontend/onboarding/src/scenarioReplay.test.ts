import activeReplayJson from "../public/scenarios/active-nisha.json";
import persistentReplayJson from "../public/scenarios/persistent-lukas.json";
import scenarioCatalogJson from "../public/scenarios/index.json";
import stableReplayJson from "../public/scenarios/stable-noor.json";
import twoValuesReplayJson from "../public/scenarios/two-values-meera.json";
import uncertainReplayJson from "../public/scenarios/uncertain-wei-jun.json";
import savedCoachResponses from "../../../src/demo/coach_digest_responses.json";
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

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) =>
      `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

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
        "persistent_drift",
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

  it.each(["missing_value", "unknown_value", "missing_week", "invalid_state", "wrong_aggregate"])(
    "rejects a catalog whose Core Value progression has %s", (mutation) => {
      const invalid = structuredClone(scenarioCatalogJson);
      const meera = invalid.scenarios.find((item) => item.scenario_id === "two-values-meera")!;
      const states = meera.core_value_progression as unknown as Record<string, string[]>;
      if (mutation === "missing_value") delete states.tradition;
      if (mutation === "unknown_value") states.unknown = [...states.tradition];
      if (mutation === "missing_week") states.tradition.pop();
      if (mutation === "invalid_state") states.tradition[0] = "drifting";
      if (mutation === "wrong_aggregate") states.self_direction[0] = "no_active_drift";
      expect(() => validateScenarioCatalog(invalid)).toThrow(/progression/);
    },
  );

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
    expect(final.session.drift_result?.delivery_state).toBe("no_active_drift");
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

  it("preserves strict microsecond chronology for same-day North Star Moment sources", async () => {
    const saved = validateExperienceInspectFixture(persistentReplayJson);
    const projected = projectScenarioWeek(saved, 1);
    const profile = saved.scenario.profile;
    const result = currentNorthStarEvent({
      events: projected.events, profile, profileRef: await northStarProfileRef(profile),
      weeklyDigest: projected.session.weekly_digest, journalEntries: projected.session.journal_entries,
    })!;
    const display = (candidate: typeof result) => displayableNorthStarSelection(
      candidate, profile, projected.session.journal_entries, projected.session.drift_result!,
    );
    expect(result.record.mode).toBe("reflection");
    expect(display(result)).toEqual(result.record.selected);
    const source = result.record.sources.find((item) => item.entry_id === result.record.selected!.entry_id)!;
    const availableAt = source.available_at as string;
    expect(display({ ...result, record: { ...result.record, onset_available_at: availableAt } })).toBeNull();

    const sameDay = structuredClone(result);
    sameDay.record.onset_available_at = "2025-06-11T00:00:00.000003Z";
    sameDay.record.sources[0].available_at = "2025-06-11T00:00:00.000002Z";
    expect(display(sameDay)).toEqual(sameDay.record.selected);
    const laterSource = structuredClone(sameDay);
    laterSource.record.sources[0].available_at = "2025-06-11T00:00:00.000004Z";
    expect(display(laterSource)).toBeNull();

    const beyondCutoff = structuredClone(result);
    beyondCutoff.record.cutoff_at = "2025-06-11T00:00:00.000001Z";
    beyondCutoff.record.sources[0].available_at = "2025-06-11T00:00:00.000002Z";
    expect(display(beyondCutoff)).toBeNull();
  });

  it.each([
    ["two-values-meera", twoValuesReplayJson],
    ["stable-noor", stableReplayJson],
    ["active-nisha", activeReplayJson],
    ["persistent-lukas", persistentReplayJson],
    ["uncertain-wei-jun", uncertainReplayJson],
  ])(
    "projects a source-bound Coach Digest for every %s week without future responses",
    async (scenarioId, scenarioJson) => {
      const scenarioFixture = validateExperienceInspectFixture(scenarioJson);
      for (let index = 0; index < scenarioFixture.scenario.weeks.length; index += 1) {
        const week = scenarioFixture.scenario.weeks[index];
        const savedResponse = Object.values(savedCoachResponses.responses).find(
          (response) => response.scenario_id === scenarioId && response.week_start === week.week_start,
        )!;
        expect(savedResponse).toBeTruthy();
        const projected = projectScenarioWeek(scenarioFixture, index);
        const currentEventIds = new Set(scenarioFixture.scenario.weeks[index].event_ids);
        const coachEvents = projected.events.filter((event) => event.event_type === "weekly_coach_generated");
        const currentCoachEvents = coachEvents.filter((event) => currentEventIds.has(event.event_id));
        const digestEvent = projected.events.find((event) =>
          currentEventIds.has(event.event_id)
          && event.event_type === "weekly_digest_built");
        expect(coachEvents).toHaveLength(index + 1);
        expect(projected.session.weekly_digest?.coach_narrative).toEqual(savedResponse.narrative);
        expect(digestEvent?.details.coach_unavailable_reason).toBeNull();
        expect(currentCoachEvents).toHaveLength(1);
        expect(currentCoachEvents[0]).toMatchObject({
          source: "saved_replay", model_contract: savedResponse.generation.model_contract,
          prompt: savedResponse.generation.prompt, raw_response: savedResponse.generation.raw_output,
          details: { narrative: savedResponse.narrative },
        });
        const input = Object.fromEntries(Object.entries(projected.session.weekly_digest!)
          .filter(([key]) => !["coach_narrative", "validation"].includes(key)));
        const hash = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonicalJson(input)));
        expect(Array.from(new Uint8Array(hash), (byte) => byte.toString(16).padStart(2, "0")).join(""))
          .toBe(savedResponse.generation.weekly_drift_input_sha256);
        expect(digestEvent?.event_id).toBe(savedResponse.generation.weekly_digest_event_id);
        const futureEventIds = new Set(scenarioFixture.scenario.weeks.slice(index + 1)
          .flatMap((futureWeek) => futureWeek.event_ids));
        expect(projected.events.some((event) => futureEventIds.has(event.event_id))).toBe(false);
      }
    },
  );

  it.each([
    activeReplayJson,
    persistentReplayJson,
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
    ["Nisha", activeReplayJson], ["Lukas Vetter", persistentReplayJson],
    ["Noor", stableReplayJson], ["Meera", twoValuesReplayJson],
    ["Wei Jun", uncertainReplayJson],
  ])("preserves %s completed experiment evidence without future-week leakage", async (_name, scenarioJson) => {
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
      expect(result, `${saved.scenario.scenario_id} week ${index + 1}`).not.toBeNull();
      expect(result!.event.source).toBe("saved_replay");
      expect(result!.record.experiment).toMatchObject({
        source_path: "logs/experiments/reports/north_star_v4_run1_20260907/nsm_experiment.json",
        method: "full_history",
        case_id: `${saved.scenario.persona_id}:week:${saved.scenario.weeks[index].week_start}`,
      });
      const selection = displayableNorthStarSelection(result, profile,
        projected.session.journal_entries, projected.session.drift_result!);
      expect(selection).toEqual(result!.record.selected);
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
