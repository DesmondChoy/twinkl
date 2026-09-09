import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import activeReplayJson from "../public/scenarios/active-nisha.json";
import activeReplayRaw from "../public/scenarios/active-nisha.json?raw";
import persistentReplayJson from "../public/scenarios/persistent-lukas.json";
import scenarioCatalogJson from "../public/scenarios/index.json";
import stableReplayJson from "../public/scenarios/stable-noor.json";
import twoValuesReplayJson from "../public/scenarios/two-values-meera.json";
import twoValuesReplayRaw from "../public/scenarios/two-values-meera.json?raw";
import uncertainReplayJson from "../public/scenarios/uncertain-wei-jun.json";
import App from "./App";
import {
  PersonaReplayExperience,
  PersonaReplayPicker,
} from "./PersonaReplay";
import {
  validateExperienceInspectFixture,
  type ExperienceInspectFixtureContract,
} from "./demoContracts";
import {
  projectScenarioWeek,
  validateScenarioCatalog,
  type LoadedScenario,
  type ScenarioCatalogItem,
} from "./scenarioReplay";
import {
  createSession,
  createExperienceState,
  parseSession,
  SESSION_STORAGE_KEY,
  type ExperienceState,
} from "./session";
import type { NorthStarRecord } from "./northStar";
import { savedCoachComparison, type CoachComparisonNarrative } from "./coachComparison";

const fixture = validateExperienceInspectFixture(activeReplayJson);
const catalog = validateScenarioCatalog(scenarioCatalogJson);
const catalogItem = catalog.scenarios.find(
  (item) => item.scenario_id === fixture.scenario.scenario_id,
)!;
const loaded: LoadedScenario = { catalogItem, fixture };

function scenarioResponse(raw = activeReplayRaw) {
  return {
    ok: true,
    arrayBuffer: async () => new TextEncoder().encode(raw).buffer,
  };
}

function enterPreferredName(name = "Casey") {
  const onboarding = screen.queryByRole("button", { name: "Try Onboarding" });
  if (onboarding) fireEvent.click(onboarding);
  fireEvent.change(screen.getByRole("textbox", { name: "Preferred name" }), {
    target: { value: name },
  });
  fireEvent.click(screen.getByRole("button", { name: "Continue" }));
}

function experienceForWeek(
  weekIndex: number,
  scenarioFixture: ExperienceInspectFixtureContract = fixture,
  item: ScenarioCatalogItem = catalogItem,
): ExperienceState {
  const projection = projectScenarioWeek(scenarioFixture, weekIndex);
  return {
    ...createExperienceState(),
    journal_started: true,
    revision: projection.session.revision,
    journal_entries: projection.session.journal_entries,
    nudges: projection.session.nudges,
    selected_persona_id: item.persona_id,
    selected_week: weekIndex,
    weekly_reviewer_decisions:
      projection.session.weekly_reviewer_decisions,
    drift_result: projection.session.drift_result,
    weekly_digest: projection.session.weekly_digest,
    run_state: "complete",
    trace_event_ids: projection.session.trace_event_ids,
    trace_events: projection.events,
  };
}

function ScenarioReplayHarness({
  scenarioJson = activeReplayJson,
  inspectRun = () => undefined,
}: {
  scenarioJson?: unknown;
  inspectRun?: (eventId: string) => void;
}) {
  const scenarioFixture = validateExperienceInspectFixture(scenarioJson);
  const item = catalog.scenarios.find(
    (candidate) => candidate.scenario_id === scenarioFixture.scenario.scenario_id,
  )!;
  const [weekIndex, setWeekIndex] = useState(0);
  const [experience, setExperience] = useState(() =>
    experienceForWeek(0, scenarioFixture, item)
  );
  const changeWeek = (nextWeek: number) => {
    setWeekIndex(nextWeek);
    setExperience(experienceForWeek(nextWeek, scenarioFixture, item));
  };
  return (
    <PersonaReplayExperience
      loaded={{ catalogItem: item, fixture: scenarioFixture }}
      weekIndex={weekIndex}
      profile={scenarioFixture.scenario.profile}
      experience={experience}
      updateExperience={(patch) =>
        setExperience((current) => ({ ...current, ...patch }))
      }
      inspectRun={inspectRun}
      onWeekChange={changeWeek}
    />
  );
}

function ReplayHarness() {
  return <ScenarioReplayHarness />;
}

function matchMedia(matches: boolean) {
  vi.stubGlobal(
    "matchMedia",
    vi.fn().mockImplementation(() => ({
      matches,
      media: "(prefers-reduced-motion: reduce)",
      onchange: null,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      addListener: vi.fn(),
      removeListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  );
}

function personaCard(name: string): HTMLElement {
  const card = screen.getByRole("radio", { name }).closest("tbody");
  if (!card) throw new Error(`Persona card not found: ${name}`);
  return card;
}

async function startMeeraReplay() {
  matchMedia(false);
  vi.stubGlobal("fetch", vi.fn().mockImplementation((input: string) => {
    if (input === "/scenarios/index.json") {
      return Promise.resolve({ ok: true, json: async () => scenarioCatalogJson });
    }
    return Promise.resolve(input === "/scenarios/two-values-meera.json"
      ? scenarioResponse(twoValuesReplayRaw)
      : { ok: false });
  }));
  const user = userEvent.setup();
  const view = render(<App />);
  await user.click(screen.getByRole("button", { name: /Try (?:the )?demo/i }));
  await screen.findByRole("radio", { name: "Meera Krishnamurthy" });
  await user.click(screen.getByRole("radio", { name: "Meera Krishnamurthy" }));
  await user.click(screen.getByRole("button", { name: "Start at week 1" }));
  await screen.findByRole("heading", { name: "Meera Krishnamurthy", level: 1 });
  return { user, view };
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("persona replay", () => {
  it("compares all five Personas in catalog order and selects a whole row with one action", async () => {
    matchMedia(false);
    vi.stubGlobal("CSS", { escape: (value: string) => value });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => scenarioCatalogJson }));
    const user = userEvent.setup();
    render(<PersonaReplayPicker onLoad={() => true} />);
    await screen.findByRole("radio", { name: "Nisha Agarwal" });
    expect(screen.getAllByRole("radio").map((radio) => radio.getAttribute("aria-label")))
      .toEqual(["Nisha Agarwal", "Noor Haddad", "Lukas Vetter", "Wei Jun Chen", "Meera Krishnamurthy"]);
    expect(screen.getAllByRole("button", { name: "Start at week 1" })).toHaveLength(1);
    expect(screen.queryByText("Professor guide: states by week")).toBeNull();
    const lukasRow = personaCard("Lukas Vetter");
    expect(within(lukasRow).getAllByRole("cell").map((cell) => cell.textContent))
      .toEqual(["Active Drift", "Active Drift", "Active Drift", "Active Drift", "No Active Drift", "—Replay ends before this week"]);
    await user.click(within(lukasRow).getAllByRole("cell")[2]);
    expect(screen.getByRole("heading", { name: "Lukas Vetter" })).toBeTruthy();
    expect((screen.getByRole("radio", { name: "Lukas Vetter" }) as HTMLInputElement).checked).toBe(true);
    expect(screen.getByText(/The same Drift stays active through weeks 1–4/)).toBeTruthy();
    screen.getByRole("radio", { name: "Lukas Vetter" }).focus();
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("heading", { name: "Wei Jun Chen" })).toBeTruthy();
    await user.keyboard("{ArrowDown}");
    expect(screen.getByRole("heading", { name: "Meera Krishnamurthy" })).toBeTruthy();
    const meeraRows = within(personaCard("Meera Krishnamurthy")).getAllByRole("row");
    expect(meeraRows).toHaveLength(2);
    expect(within(meeraRows[0]).getByRole("rowheader", { name: "Self-Direction" })).toBeTruthy();
    expect(within(meeraRows[0]).getAllByRole("cell")[0].textContent).toBe("Active Drift");
    expect(within(meeraRows[1]).getByRole("rowheader", { name: "Tradition" })).toBeTruthy();
    expect(within(meeraRows[1]).getAllByRole("cell")[0].textContent).toBe("No Active Drift");
    expect(screen.getByText("Choosing her own path")).toBeTruthy();
    expect(screen.getAllByRole("button", { name: "Start at week 1" })).toHaveLength(1);
  });

  it("places the one selected detail and action beneath its row on narrow screens", async () => {
    matchMedia(true);
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => scenarioCatalogJson }));
    const user = userEvent.setup();
    render(<PersonaReplayPicker onLoad={() => true} />);
    await user.click(await screen.findByRole("radio", { name: "Meera Krishnamurthy" }));
    const row = personaCard("Meera Krishnamurthy");
    expect(within(row).getByRole("heading", { name: "Meera Krishnamurthy" })).toBeTruthy();
    expect(screen.getAllByRole("button", { name: "Start at week 1" })).toHaveLength(1);
    expect(within(row).getByRole("button", { name: "Start at week 1" })).toBeTruthy();
    expect(screen.getAllByRole("radio")).toHaveLength(5);
  });

  it("uses the catalogued key week in both chooser guidance and replay navigation", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    const saved = validateExperienceInspectFixture(uncertainReplayJson);
    const originalItem = catalog.scenarios.find((item) => item.scenario_id === saved.scenario.scenario_id)!;
    // Moving the catalogue key must update both controls without new copy or data fields.
    const item = { ...originalItem, key_week_start: saved.scenario.weeks[4].week_start };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({
      ...scenarioCatalogJson,
      scenarios: catalog.scenarios.map((candidate) => candidate.scenario_id === item.scenario_id ? item : candidate),
    }) }));
    const picker = render(<PersonaReplayPicker onLoad={() => true} />);
    await user.click(await screen.findByRole("radio", { name: "Wei Jun Chen" }));
    expect(screen.getByRole("region", { name: "Wei Jun Chen" }).querySelector("time")
      ?.getAttribute("datetime")).toBe(saved.scenario.weeks[4].week_start);
    picker.unmount();

    const changeWeek = vi.fn();
    render(<PersonaReplayExperience loaded={{ catalogItem: item, fixture: saved }} weekIndex={0}
      profile={saved.scenario.profile} experience={experienceForWeek(0, saved, item)}
      updateExperience={() => undefined} inspectRun={() => undefined} onWeekChange={changeWeek} />);
    await user.click(screen.getByRole("button", { name: "Show Insufficient Evidence — week 5" }));
    expect(changeWeek).toHaveBeenCalledWith(4);
  });

  it("keeps revealed later weeks after returning from Inspect and reloading an earlier week", async () => {
    const { user, view } = await startMeeraReplay();
    for (let week = 0; week < 2; week += 1) {
      await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
      await user.click(screen.getByRole("button", { name: "Next week" }));
    }
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    await user.click(screen.getByRole("button", { name: "Show week 2: no active drift" }));
    await user.click(screen.getByRole("button", { name: "Inspect" }));
    expect(screen.queryByRole("button", { name: "View Profile calculation" })).toBeNull();
    expect(screen.getByText(/This Persona Profile is a synthetic projection/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Return to Experience" }));
    expect(screen.getByText("Week 2 of 5")).toBeTruthy();
    expect((screen.getByRole("button", {
      name: "Show week 3: no active drift",
    }) as HTMLButtonElement).disabled).toBe(false);
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    const states = document.querySelectorAll(".state-change > header");
    expect(Array.from(states, (header) => header.textContent)).toEqual([
      "Self-DirectionNo Active Drift", "TraditionNo Active Drift",
    ]);

    view.unmount();
    render(<App />);
    await screen.findByText("Week 2 of 5");
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    expect((screen.getByRole("button", {
      name: "Show week 3: no active drift",
    }) as HTMLButtonElement).disabled).toBe(false);
    await user.click(screen.getByRole("button", { name: "Show week 3: no active drift" }));
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    expect(Array.from(document.querySelectorAll(".state-change > header"),
      (header) => header.textContent)).toEqual([
      "Self-DirectionNo Active Drift", "TraditionNo Active Drift",
    ]);
  });

  it("restores all journals and requires review again after Inspect and reload", async () => {
    const { user, view } = await startMeeraReplay();
    const firstWeekCount = twoValuesReplayJson.scenario.weeks[0].journal_entry_ids.length;
    expect(screen.getAllByRole("button", { name: /Open Journal Entry/ })).toHaveLength(firstWeekCount);
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    await user.click(screen.getByRole("button", { name: "Inspect" }));
    await user.click(screen.getByRole("button", { name: "Return to Experience" }));
    expect(screen.getAllByRole("button", { name: /Open Journal Entry/ })).toHaveLength(firstWeekCount);
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    expect((screen.getByRole("button", { name: "Next week" }) as HTMLButtonElement).disabled).toBe(false);

    view.unmount();
    const restored = render(<App />);
    await screen.findByRole("button", { name: /Open Journal Entry 1/ });
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    await user.click(screen.getByRole("button", { name: "Next week" }));
    expect(screen.getByText("Week 2 of 5")).toBeTruthy();
    const secondWeekCount = twoValuesReplayJson.scenario.weeks[1].journal_entry_ids.length;
    expect(screen.getAllByRole("button", { name: /Open Journal Entry/ })).toHaveLength(secondWeekCount);

    restored.unmount();
    render(<App />);
    await screen.findByText("Week 2 of 5");
    expect(screen.getAllByRole("button", { name: /Open Journal Entry/ })).toHaveLength(secondWeekCount);
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
  });

  it("loads a selected persona from the saved catalog", async () => {
    const onLoad = vi.fn<(loaded: LoadedScenario) => boolean>(() => true);
    const activeCatalog = {
      ...scenarioCatalogJson,
      scenarios: scenarioCatalogJson.scenarios
        .filter((item) => item.scenario_id === "active-nisha")
        .map((item) => ({ ...item, recommended: true })),
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => activeCatalog,
      })
      .mockResolvedValueOnce(scenarioResponse());
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();

    render(<PersonaReplayPicker onLoad={onLoad} />);
    await user.click(
      await screen.findByRole("button", { name: "Start at week 1" }),
    );

    await waitFor(() => expect(onLoad).toHaveBeenCalledTimes(1));
    expect(onLoad.mock.calls[0][0].fixture.scenario.source).toBe(
      "saved_replay",
    );
  });

  it("rejects saved persona content that does not match the catalog hash", async () => {
    const onLoad = vi.fn<(loaded: LoadedScenario) => boolean>(() => true);
    const activeCatalog = {
      ...scenarioCatalogJson,
      scenarios: scenarioCatalogJson.scenarios
        .filter((item) => item.scenario_id === "active-nisha")
        .map((item) => ({ ...item, recommended: true })),
    };
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: async () => activeCatalog,
        })
        .mockResolvedValueOnce(
          scenarioResponse(`${activeReplayRaw} `),
        ),
    );
    const user = userEvent.setup();

    render(<PersonaReplayPicker onLoad={onLoad} />);
    await user.click(
      await screen.findByRole("button", { name: "Start at week 1" }),
    );

    expect((await screen.findByRole("alert")).textContent).toContain(
      "could not be loaded",
    );
    expect(onLoad).not.toHaveBeenCalled();
  });

  it("keeps the current persona selected when reopening the picker", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => scenarioCatalogJson,
      }),
    );

    render(
      <PersonaReplayPicker
        currentPersonaId={catalogItem.persona_id}
        onLoad={() => true}
      />,
    );

    await screen.findByRole("radio", { name: "Nisha Agarwal" });
    const current = personaCard("Nisha Agarwal");
    expect(within(current).getByText("Saved progress")).toBeTruthy();
    expect(
      (screen.getByRole("button", {
        name: "Continue replay · Week 1",
      }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it("offers a retry when the persona catalog fails to load", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce({
        ok: true,
        json: async () => scenarioCatalogJson,
      });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();

    render(<PersonaReplayPicker onLoad={() => true} />);
    await user.click(
      await screen.findByRole("button", { name: "Try loading again" }),
    );

    expect(await screen.findByRole("radio", { name: "Meera Krishnamurthy" }))
      .toBeTruthy();
    expect(
      screen.getAllByRole("radio")[0].getAttribute("aria-label"),
    ).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it.each([
    ["two-values-meera", twoValuesReplayJson],
    ["stable-noor", stableReplayJson],
    ["active-nisha", activeReplayJson],
    ["persistent-lukas", persistentReplayJson],
    ["uncertain-wei-jun", uncertainReplayJson],
  ])("immediately shows every selected-week Journal Entry for %s", (_scenarioId, scenarioJson) => {
    matchMedia(false);
    render(<ScenarioReplayHarness scenarioJson={scenarioJson} />);
    const week = scenarioJson.scenario.weeks[0];
    expect(screen.getAllByRole("button", { name: /Open Journal Entry/ }))
      .toHaveLength(week.journal_entry_ids.length);
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    expect(screen.queryByRole("button", { name: "Next step" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Auto replay|Pause replay/ })).toBeNull();
    expect((screen.getByRole("button", { name: "Next week" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it.each([0, 1, 2])("ignores legacy partial reveal stage %s when displaying a saved week", (stage) => {
    matchMedia(false);
    const experience = experienceForWeek(0);
    experience.replay_progress = {
      scenario_id: catalogItem.scenario_id, week_index: 0,
      reveal_stage: stage, furthest_completed_week: -1,
    };
    render(<PersonaReplayExperience loaded={loaded} weekIndex={0}
      profile={fixture.scenario.profile} experience={experience}
      updateExperience={() => undefined} inspectRun={() => undefined}
      onWeekChange={() => undefined} />);
    expect(screen.getAllByRole("button", { name: /Open Journal Entry/ })).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Review Weekly Drift Detection" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
  });

  it("advances without review and returns to journals for new weeks and restart", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);
    expect(screen.queryByRole("button", { name: "Previous" })).toBeNull();
    expect((screen.getByRole("button", { name: "Show week 2, outcome hidden" }) as HTMLButtonElement).disabled).toBe(false);
    expect((screen.getByRole("button", { name: "Next week" }) as HTMLButtonElement).disabled).toBe(false);
    await user.click(screen.getByRole("button", { name: "Next week" }));
    expect(screen.getByText("Week 2 of 5")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Show week 1, outcome hidden" }));
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    expect(screen.getByRole("heading", { name: "No Active Drift" })).toBeTruthy();
    expect(screen.getByRole("heading", {
      name: "Drift Detection (End of Week) (based on 2 Journal Entries through Feb 16)",
    })).toBeTruthy();
    document.documentElement.scrollTop = 640;
    document.body.scrollTop = 640;
    await user.click(screen.getByRole("button", { name: "Next week" }));
    expect(screen.getByText("Week 2 of 5")).toBeTruthy();
    expect(document.documentElement.scrollTop).toBe(0);
    expect(document.body.scrollTop).toBe(0);
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    expect((screen.getByRole("button", { name: "Next week" }) as HTMLButtonElement).disabled).toBe(false);
    await user.click(screen.getByRole("button", { name: "Show week 1: no active drift" }));
    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    document.documentElement.scrollTop = 640;
    document.body.scrollTop = 640;
    await user.click(screen.getByRole("button", { name: "Restart" }));
    expect(document.documentElement.scrollTop).toBe(0);
    expect(document.body.scrollTop).toBe(0);
    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    expect(screen.getAllByRole("button", { name: /Open Journal Entry/ })).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Review Weekly Drift Detection" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    expect(screen.getByRole("listitem", { name: "Week 1, not yet replayed" })).toBeTruthy();
  });

  it("allows sequential navigation without review and stops at the final week", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);
    for (let week = 2; week <= fixture.scenario.weeks.length; week += 1) {
      await user.click(screen.getByRole("button", { name: "Next week" }));
      expect(screen.getByText(`Week ${week} of ${fixture.scenario.weeks.length}`)).toBeTruthy();
      expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
      expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    }
    expect((screen.getByRole("button", { name: "Next week" }) as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole("button", { name: "Next week" }));
    expect(screen.getByText(`Week ${fixture.scenario.weeks.length} of ${fixture.scenario.weeks.length}`)).toBeTruthy();
  });

  it.each([false, true])("keeps the saved week unchanged without timer-driven replay (reduced motion: %s)", (reducedMotion) => {
    matchMedia(reducedMotion);
    vi.useFakeTimers();
    render(<ReplayHarness />);
    expect(screen.getAllByRole("button", { name: /Open Journal Entry/ })).toHaveLength(2);
    act(() => vi.advanceTimersByTime(60_000));
    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    expect(screen.getAllByRole("button", { name: /Open Journal Entry/ })).toHaveLength(2);
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
  });

  it("reviews the week with keyboard input without revealing future outcomes", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);
    screen.getByRole("button", { name: "Review Weekly Drift Detection" }).focus();
    await user.keyboard("{Enter}");
    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    const resultHeading = screen.getByRole("heading", { name: /^Drift Detection \(End of Week\)/ });
    await waitFor(() => expect(document.activeElement).toBe(resultHeading));
    expect(screen.getByRole("listitem", { name: "Week 5, not yet replayed" })).toBeTruthy();
    expect(screen.queryByRole("listitem", { name: "Week 5: No Active Drift" })).toBeNull();
  });

  it("preserves the journal reading position within a week and resets it for another week", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);
    const journalViewport = document.querySelector<HTMLDivElement>(
      ".replay-column--entries .replay-column__scroll",
    )!;
    const scrollTo = vi.fn();
    Object.defineProperty(journalViewport, "scrollTo", { configurable: true, value: scrollTo });
    journalViewport.scrollTop = 240;

    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    await user.click(screen.getByRole("button", { name: "Read Journal Entries" }));
    expect(journalViewport.scrollTop).toBe(240);
    expect(scrollTo).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Read Weekly Drift Detection" }));
    await user.click(screen.getByRole("button", { name: "Next week" }));
    expect(screen.getByText("Week 2 of 5")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(scrollTo).toHaveBeenCalledWith({ top: 0 });
  });

  it("shows a longer Journal Entry preview and opens the complete text in a dialog", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);

    const entry = experienceForWeek(0).journal_entries[0];
    const entryButton = screen.getByRole("button", {
      name: /Open Journal Entry 1/,
    });
    const words = entry.content.trim().split(/\s+/);
    expect(entryButton.querySelector(".replay-entry__excerpt")?.textContent).toBe(
      words.length > 50 ? `${words.slice(0, 50).join(" ")}…` : entry.content.trim(),
    );
    expect(entryButton.getAttribute("aria-haspopup")).toBe("dialog");

    await user.click(entryButton);
    const dialog = screen.getByRole("dialog");
    expect(
      dialog.querySelector(".replay-entry-drawer__content")?.textContent,
    ).toBe(entry.content);
    expect(document.body.style.overflow).toBe("hidden");
    expect(document.activeElement).toBe(
      within(dialog).getByRole("button", { name: "Close Journal Entry" }),
    );

    await user.click(
      within(dialog).getByRole("button", { name: "Close Journal Entry" }),
    );
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.body.style.overflow).toBe("");
    await waitFor(() => expect(document.activeElement).toBe(entryButton));
  });

  it("lets people revisit weeks without revealing future outcomes first", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);

    const futureWeek = screen.getByRole("button", {
      name: "Show week 4, outcome hidden",
    }) as HTMLButtonElement;
    expect(futureWeek.disabled).toBe(false);
    await user.click(futureWeek);
    expect(screen.getByText("Week 4 of 5")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    await user.click(
      screen.getByRole("button", {
        name: "Show week 1: no active drift",
      }),
    );

    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    expect(
      (screen.getByRole("button", {
        name: "Restart",
      }) as HTMLButtonElement).disabled,
    ).toBe(false);
    await user.click(
      screen.getByRole("button", {
        name: "Show week 3: no active drift",
      }),
    );
    expect(screen.getByText("Week 3 of 5")).toBeTruthy();
    expect(screen.getByRole("button", {
      name: "Show week 4: active drift",
    })).toBeTruthy();
  });

  it.each([
    ["Nisha", activeReplayJson],
    ["Noor", stableReplayJson],
    ["Lukas", persistentReplayJson],
    ["Wei Jun", uncertainReplayJson],
    ["Meera", twoValuesReplayJson],
  ])("lets every %s week open directly without reviewing another week", async (_name, scenarioJson) => {
    matchMedia(false);
    const user = userEvent.setup();
    const saved = validateExperienceInspectFixture(scenarioJson);
    const weeks = saved.scenario.weeks;
    render(<ScenarioReplayHarness scenarioJson={scenarioJson} />);

    for (let index = weeks.length - 1; index >= 0; index -= 1) {
      const weekButton = screen.getByRole("button", {
        name: `Show week ${index + 1}, outcome hidden`,
      }) as HTMLButtonElement;
      expect(weekButton.disabled).toBe(false);
      await user.click(weekButton);
      expect(screen.getByText(`Week ${index + 1} of ${weeks.length}`)).toBeTruthy();
      expect(weekButton.getAttribute("aria-current")).toBe("step");
      expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
      expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
      for (const entryId of weeks[index].journal_entry_ids) {
        expect(document.getElementById(`replay-entry-button-${entryId}`)).not.toBeNull();
      }
      for (const laterWeek of weeks.slice(index + 1)) {
        for (const entryId of laterWeek.journal_entry_ids) {
          expect(document.getElementById(`replay-entry-button-${entryId}`)).toBeNull();
        }
      }
      expect(within(screen.getByRole("navigation", { name: "Replay weeks" }))
        .queryAllByRole("listitem", { name: /^Week \d+: / })).toHaveLength(0);
    }

    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    expect(screen.getByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeTruthy();
    await user.click(screen.getByRole("button", { name: /^Show week 1:/ }));
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
  });

  it("opens an earlier unreviewed week after a key-week jump without revealing outcomes", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);

    await user.click(screen.getByRole("button", {
      name: "Show Active Drift — week 4",
    }));
    const previousWeek = screen.getByRole("button", {
      name: "Show week 3, outcome hidden",
    }) as HTMLButtonElement;
    expect(previousWeek.disabled).toBe(false);
    document.documentElement.scrollTop = 640;
    document.body.scrollTop = 640;
    await user.click(previousWeek);
    expect(document.documentElement.scrollTop).toBe(0);
    expect(document.body.scrollTop).toBe(0);

    expect(screen.getByText("Week 3 of 5")).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    for (const entryId of fixture.scenario.weeks[2].journal_entry_ids) {
      expect(document.getElementById(`replay-entry-button-${entryId}`)).not.toBeNull();
    }
    for (const entryId of fixture.scenario.weeks[3].journal_entry_ids) {
      expect(document.getElementById(`replay-entry-button-${entryId}`)).toBeNull();
    }
    const weekNavigation = screen.getByRole("navigation", { name: "Replay weeks" });
    expect(within(weekNavigation).queryAllByRole("listitem", {
      name: /^Week \d+: /,
    })).toHaveLength(0);
    for (const week of [4, 5]) {
      expect((within(weekNavigation).getByRole("button", {
        name: `Show week ${week}, outcome hidden`,
      }) as HTMLButtonElement).disabled).toBe(false);
    }
    expect((screen.getByRole("button", { name: "Next week" }) as HTMLButtonElement).disabled).toBe(false);
  });

  it.each([
    ["two-values-meera", twoValuesReplayJson],
    ["stable-noor", stableReplayJson],
    ["active-nisha", activeReplayJson],
    ["persistent-lukas", persistentReplayJson],
    ["uncertain-wei-jun", uncertainReplayJson],
  ])(
    "shows the fresh Coach Digest and saved North Star Moment evidence for %s",
    async (scenarioId, scenarioJson) => {
      matchMedia(false);
      const user = userEvent.setup();
      render(<ScenarioReplayHarness scenarioJson={scenarioJson} />);
      expect(screen.queryByText("No saved Coach Digest for this result")).toBeNull();
      expect(screen.queryByRole("heading", { name: "Coach Digest" })).toBeNull();
      await user.click(screen.getByRole("button", {
        name: /^Show .+ — week \d+$/,
      }));
      expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
      expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
      const selectedFixture = validateExperienceInspectFixture(scenarioJson);
      const selectedItem = catalog.scenarios.find((item) => item.scenario_id === scenarioId)!;
      const selectedWeek = selectedFixture.scenario.weeks.find((week) => week.week_start === selectedItem.key_week_start)!;
      const coachEvent = selectedFixture.trace_events.find((event) => selectedWeek.event_ids.includes(event.event_id)
        && event.event_type === "weekly_coach_generated")!;
      const savedNarrative = coachEvent.details.narrative as CoachComparisonNarrative;
      expect(screen.queryAllByRole("button", { name: /Open Journal Entry/ }))
        .toHaveLength(selectedWeek.journal_entry_ids.length);
      await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));

      expect(screen.queryByText("No saved Coach Digest for this result")).toBeNull();
      expect(screen.getByText("Why this state")).toBeTruthy();
      expect(screen.getByRole("heading", { name: "Coach Digest" })).toBeTruthy();
      const tension = document.querySelector(".coach-digest__tension");
      expect(tension?.textContent).toContain(
        savedNarrative.tension_explanation.split("...")[0],
      );
      expect(tension?.textContent).not.toMatch(/(?:\.{3}|…)[”"]/);
      expect(document.querySelector(".coach-digest__question")?.textContent)
        .toBe(savedNarrative.reflective_question);
      const saved = validateExperienceInspectFixture(scenarioJson);
      const item = catalog.scenarios.find((candidate) => candidate.scenario_id === scenarioId)!;
      const keyWeek = saved.scenario.weeks.find((week) => week.week_start === item.key_week_start)!;
      const record = saved.trace_events.find((event) =>
        keyWeek.event_ids.includes(event.event_id) && event.event_type === "north_star_reviewed"
      )!.details.record as NorthStarRecord;
      expect(document.querySelector(".north-star-moment")).toBeNull();
      const toggle = screen.getByRole("button", { name: "With North Star Moment" });
      if (record.selected) {
        const comparison = savedCoachComparison(coachEvent);
        expect(comparison).not.toBeNull();
        await waitFor(() => expect(toggle.hasAttribute("disabled")).toBe(false));
        await user.click(toggle);
        expect(document.querySelector(".coach-digest__question")?.textContent)
          .toBe(comparison!.with_north_star.narrative.reflective_question);
        await waitFor(() => expect(document.querySelector(".north-star-moment blockquote")?.textContent)
          .toBe(record.selected!.evidence_quote));
        await user.click(screen.getByRole("link", { name: /^Open (response in )?Journal Entry ·/ }));
        const source = saved.scenario.journal_entries.find((entry) => entry.journal_entry_id === record.selected!.entry_id)!;
        expect(screen.getByRole("dialog").querySelector(".replay-entry-drawer__content")?.textContent)
          .toBe(source.content);
        expect(screen.queryByRole("heading", { name: "Journal Entries" })).toBeNull();
        await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Close Journal Entry" }));
        expect(screen.getByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeTruthy();
        expect(screen.queryByRole("heading", { name: "Journal Entries" })).toBeNull();
      } else {
        expect(toggle.hasAttribute("disabled")).toBe(false);
        const reflection = () => ["mirror", "tension", "question"].map((part) =>
          document.querySelector(`.coach-digest__${part}`)?.textContent);
        const baseline = reflection();
        await user.click(toggle);
        expect(await screen.findByRole("heading", { name: "Why there’s no North Star Moment" })).toBeTruthy();
        expect(reflection()).toEqual(baseline);
        await user.click(screen.getByRole("button", { name: "Hide explanation" }));
        expect(screen.queryByRole("heading", { name: "Why there’s no North Star Moment" })).toBeNull();
        expect(reflection()).toEqual(baseline);
        expect(["insufficient_evidence", "no_eligible_writing"]).toContain(record.reason);
        expect(document.querySelector(".north-star-moment")).toBeNull();
        expect(document.querySelector(".replay-result-column--north-star")).toBeNull();
        expect(screen.queryByText("No moment for this week")).toBeNull();
        expect(screen.queryByRole("button", { name: "Inspect this moment" })).toBeNull();
        expect(screen.queryByRole("button", { name: "North Star Moment" })).toBeNull();
      }
      const firstState = saved.scenario.weeks[0].expected_delivery_state === "active_drift" ? "active drift" : "no active drift";
      await user.click(screen.getByRole("button", { name: `Show week 1: ${firstState}` }));
      await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
      const firstWeekDigest = projectScenarioWeek(saved, 0).session.weekly_digest;
      if (firstWeekDigest?.coach_narrative) {
        expect(screen.getByRole("heading", { name: "Coach Digest" })).toBeTruthy();
        expect(screen.queryByText("No saved Coach Digest for this result")).toBeNull();
      } else {
        expect(screen.getByText("No saved Coach Digest for this result")).toBeTruthy();
        expect(document.querySelector(".coach-digest__question")).toBeNull();
        expect(document.querySelector(".north-star-moment")).toBeNull();
        expect(screen.queryByRole("heading", { name: "Coach Digest" })).toBeNull();
      }
      expect(screen.queryByRole("button", { name: "Coach Digest" })).toBeNull();
    },
  );

  it("clamps an out-of-range restored week to the final week", () => {
    matchMedia(false);
    render(
      <PersonaReplayExperience
        loaded={loaded}
        weekIndex={20}
        profile={fixture.scenario.profile}
        experience={experienceForWeek(4)}
        updateExperience={() => undefined}
        inspectRun={() => undefined}
          onWeekChange={() => undefined}
      />,
    );

    expect(screen.getByText("Week 5 of 5")).toBeTruthy();
    expect(screen.getByRole("heading", { name: /Mar 10–16, 2025/i })).toBeTruthy();
  });

  it.each([
    ["stable-noor", stableReplayJson, "No Active Drift"],
    ["active-nisha", activeReplayJson, "No Active Drift"],
    ["persistent-lukas", persistentReplayJson, "No Active Drift"],
    ["uncertain-wei-jun", uncertainReplayJson, "Insufficient Evidence"],
    ["two-values-meera", twoValuesReplayJson, "No Active Drift"],
  ])("renders the final %s progression", async (scenarioId, scenarioJson, label) => {
    matchMedia(false);
    const user = userEvent.setup();
    const scenarioFixture = validateExperienceInspectFixture(scenarioJson);
    const item = catalog.scenarios.find(
      (candidate) => candidate.scenario_id === scenarioId,
    )!;
    const weekIndex = scenarioFixture.scenario.weeks.length - 1;

    render(
      <PersonaReplayExperience
        loaded={{ catalogItem: item, fixture: scenarioFixture }}
        weekIndex={weekIndex}
        profile={scenarioFixture.scenario.profile}
        experience={experienceForWeek(weekIndex, scenarioFixture, item)}
        updateExperience={() => undefined}
        inspectRun={() => undefined}
          onWeekChange={() => undefined}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    await user.click(screen.getByText("Why this state"));
    expect(screen.getAllByRole("button", { name: "AI review" }).length)
      .toBeGreaterThan(0);
    expect(screen.queryByText("Raw provider response")).toBeNull();
    expect(screen.queryByText("Validation result")).toBeNull();
  });

  it("shows the saved nudge beneath its Journal Entry", () => {
    matchMedia(false);
    const scenarioFixture =
      validateExperienceInspectFixture(twoValuesReplayJson);
    const item = catalog.scenarios.find(
      (candidate) => candidate.scenario_id === "two-values-meera",
    )!;

    render(
      <PersonaReplayExperience
        loaded={{ catalogItem: item, fixture: scenarioFixture }}
        weekIndex={1}
        profile={scenarioFixture.scenario.profile}
        experience={experienceForWeek(1, scenarioFixture, item)}
        updateExperience={() => undefined}
        inspectRun={() => undefined}
          onWeekChange={() => undefined}
      />,
    );

    const nudge = screen.getByLabelText("Nudge for Journal Entry 2");
    const entry = screen.getByRole("button", {
      name: "Open Journal Entry 2 from Nov 23",
    });

    expect(nudge.textContent).toContain(
      "Which part of the day stuck with you more?",
    );
    expect(entry.closest("li")?.contains(nudge)).toBe(true);
  });

  it("does not show a nudge when the selected week has none", () => {
    matchMedia(false);
    const scenarioFixture =
      validateExperienceInspectFixture(activeReplayJson);
    const item = catalog.scenarios.find(
      (candidate) => candidate.scenario_id === "active-nisha",
    )!;

    render(
      <PersonaReplayExperience
        loaded={{ catalogItem: item, fixture: scenarioFixture }}
        weekIndex={2}
        profile={scenarioFixture.scenario.profile}
        experience={experienceForWeek(2, scenarioFixture, item)}
        updateExperience={() => undefined}
        inspectRun={() => undefined}
          onWeekChange={() => undefined}
      />,
    );

    expect(screen.queryByLabelText(/Nudge for Journal Entry/)).toBeNull();
  });

  it("shows Schwartz Core Value names without repeating value phrases", () => {
    matchMedia(false);
    const scenarioFixture =
      validateExperienceInspectFixture(twoValuesReplayJson);
    const item = catalog.scenarios.find(
      (candidate) => candidate.scenario_id === "two-values-meera",
    )!;
    const weekIndex = scenarioFixture.scenario.weeks.length - 1;

    render(
      <PersonaReplayExperience
        loaded={{ catalogItem: item, fixture: scenarioFixture }}
        weekIndex={weekIndex}
        profile={scenarioFixture.scenario.profile}
        experience={experienceForWeek(weekIndex, scenarioFixture, item)}
        updateExperience={() => undefined}
        inspectRun={() => undefined}
          onWeekChange={() => undefined}
      />,
    );

    const values = document.querySelector(".replay-persona__value");
    expect(values?.textContent).toContain("Schwartz Core Values");
    expect(values?.textContent).toContain("Self-Direction · Tradition");
    expect(screen.queryByText("Having the freedom to choose my own path"))
      .toBeNull();
    expect(screen.queryByText(
      "Being someone others can count on to do the right thing",
    )).toBeNull();
  });

  it("places the moment inside Coach Digest before its question and links to the exact review event", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    const inspectRun = vi.fn();
    render(<ScenarioReplayHarness inspectRun={inspectRun} />);
    await user.click(screen.getByRole("button", { name: "Show Active Drift — week 4" }));
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    const toggle = screen.getByRole("button", { name: "With North Star Moment" });
    expect(document.querySelector(".north-star-moment")).toBeNull();
    await waitFor(() => expect(toggle.hasAttribute("disabled")).toBe(false));
    await user.click(toggle);
    await screen.findByRole("heading", { name: "A past moment in your own words" });
    const columns = document.querySelectorAll<HTMLElement>(".replay-result-columns > section");
    expect(columns).toHaveLength(2);
    expect(within(columns[0]).getByRole("article", { name: "Active Drift" })).toBeTruthy();
    expect(within(columns[0]).getByRole("button", { name: "Inspect decision" })).toBeTruthy();
    expect(within(columns[1]).getByRole("heading", { name: "Coach Digest" })).toBeTruthy();
    const momentHeading = within(columns[1]).getByRole("heading", { level: 4, name: "A past moment in your own words" });
    const coach = momentHeading.closest(".coach-digest")!;
    expect(coach.querySelector(".coach-digest__tension")?.nextElementSibling)
      .toBe(coach.querySelector(".coach-digest__question"));
    expect(within(coach as HTMLElement).getByText("North Star Moment")).toBeTruthy();
    expect(screen.queryByRole("navigation", { name: "Weekly reflection sections" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Coach Digest" })).toBeNull();
    expect(screen.queryByRole("button", { name: "North Star Moment" })).toBeNull();
    expect(screen.getByText("Week 4 of 5")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Journal Entries" })).toBeNull();
    const expectedEvent = experienceForWeek(3).trace_events.find((event) =>
      event.event_type === "north_star_reviewed"
      && (event.details.record as NorthStarRecord).week_start === fixture.scenario.weeks[3].week_start)!;
    await user.click(within(coach as HTMLElement).getByRole("button", { name: "Inspect this moment" }));
    expect(inspectRun).toHaveBeenCalledWith(expectedEvent.event_id);
  });

  it("keeps Weekly Drift Detection and an available Coach Digest together", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    const inspectRun = vi.fn();
    const scenarioFixture =
      validateExperienceInspectFixture(twoValuesReplayJson);
    const item = catalog.scenarios.find(
      (candidate) => candidate.scenario_id === "two-values-meera",
    )!;
    const weekIndex = scenarioFixture.scenario.weeks.findIndex((week) => week.week_start === item.key_week_start);
    const experience = experienceForWeek(weekIndex, scenarioFixture, item);
    const narrative = experience.weekly_digest!.coach_narrative as CoachComparisonNarrative;

    render(
      <PersonaReplayExperience
        loaded={{ catalogItem: item, fixture: scenarioFixture }}
        weekIndex={weekIndex}
        profile={scenarioFixture.scenario.profile}
        experience={experience}
        updateExperience={() => undefined}
        inspectRun={inspectRun}
          onWeekChange={() => undefined}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    const result = screen.getByRole("article", {
      name: "Active Drift",
    });
    const coachHeading = screen.getByRole("heading", {
      name: "Coach Digest",
    });
    expect(screen.getByText(narrative.weekly_mirror)).toBeTruthy();
    const coachCard = coachHeading.closest(".coach-digest--replay");
    const resultColumns = result.closest(".replay-result-columns");

    expect(coachCard).not.toBeNull();
    expect(resultColumns?.contains(coachCard)).toBe(true);
    expect(result.closest(".replay-result-column--state")).not.toBeNull();
    expect(coachCard?.closest(".replay-result-column--coach")).not.toBeNull();
    expect(
      result.compareDocumentPosition(coachCard!)
      & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(within(result).getByText("Why this state")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Inspect decision" }));
    const driftEvent = [...experience.trace_events]
      .reverse()
      .find((event) => event.event_type === "drift_detected");
    expect(driftEvent).toBeTruthy();
    expect(inspectRun).toHaveBeenCalledWith(driftEvent!.event_id);
    expect(screen.queryByText("No saved Coach Digest for this result")).toBeNull();
  });

  it.each([
    { scenarioId: "uncertain-wei-jun", scenarioJson: uncertainReplayJson, eventType: "drift_detected" },
    { scenarioId: "two-values-meera", scenarioJson: twoValuesReplayJson, eventType: "weekly_digest_built" },
  ])("focuses the current $scenarioId week through $eventType instead of earlier events", async ({ scenarioId, scenarioJson, eventType }) => {
    matchMedia(false);
    const user = userEvent.setup();
    const inspectRun = vi.fn();
    const saved = validateExperienceInspectFixture(scenarioJson);
    const item = catalog.scenarios.find((candidate) => candidate.scenario_id === scenarioId)!;
    const weekIndex = saved.scenario.weeks.length - 1;
    const currentIds = new Set(saved.scenario.weeks[weekIndex].event_ids);
    const experience = experienceForWeek(weekIndex, saved, item);
    // Exercise each priority fallback while retaining earlier higher-priority events.
    const higherPriorityTypes = eventType === "weekly_digest_built" ? ["drift_detected"] : [];
    experience.trace_events = experience.trace_events.filter((event) =>
      !currentIds.has(event.event_id) || !higherPriorityTypes.includes(event.event_type));
    const expected = experience.trace_events.find((event) =>
      currentIds.has(event.event_id) && event.event_type === eventType);
    expect(expected).toBeTruthy();
    expect(experience.trace_events.some((event) => event.event_type === "north_star_reviewed"
      && !currentIds.has(event.event_id))).toBe(true);
    render(<PersonaReplayExperience loaded={{ catalogItem: item, fixture: saved }}
      weekIndex={weekIndex} profile={saved.scenario.profile} experience={experience}
      updateExperience={() => undefined} inspectRun={inspectRun}
      onWeekChange={() => undefined} />);
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    await user.click(screen.getByRole("button", { name: "Inspect decision" }));
    expect(inspectRun).toHaveBeenCalledWith(expected!.event_id);
    expect(currentIds.has(inspectRun.mock.calls[0][0])).toBe(true);
  });

  it("labels the Core Value state and keeps AI review evidence beside each decision", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    const weekIndex = 3;

    render(
      <PersonaReplayExperience
        loaded={loaded}
        weekIndex={weekIndex}
        profile={fixture.scenario.profile}
        experience={experienceForWeek(weekIndex)}
        updateExperience={() => undefined}
        inspectRun={() => undefined}
          onWeekChange={() => undefined}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    const result = screen.getByRole("article", { name: "Active Drift" });
    expect(within(result).getByRole("heading", { name: "Active Drift" })).toBeTruthy();
    const explanation = within(result).getByText("Why this state").closest("details")!;
    expect(explanation.open).toBe(false);
    await user.click(within(result).getByText("Why this state"));
    expect(explanation.open).toBe(true);
    expect(result.querySelector(".state-change > header")).toBeNull();
    expect(within(result).queryByText(
      "Making the world a fairer, better place",
    )).toBeNull();

    const reviewButton = within(result).getAllByRole("button", {
      name: "AI review",
    })[0];
    fireEvent.mouseEnter(reviewButton.closest(".state-change__evidence")!);
    expect(screen.getByRole("tooltip").textContent).toContain("gpt-5.6-luna");
    fireEvent.mouseLeave(reviewButton.closest(".state-change__evidence")!);
    expect(screen.queryByRole("tooltip")).toBeNull();

    await user.click(reviewButton);

    const dialog = screen.getByRole("dialog", { name: "AI review details" });
    expect(within(dialog).getByText("gpt-5.6-luna")).toBeTruthy();
    expect(within(dialog).getByText("low")).toBeTruthy();
    expect(within(dialog).getByText("Recorded model output")).toBeTruthy();
    expect(within(dialog).getByText("Recorded justification")).toBeTruthy();
    expect(dialog.textContent).toContain("direct_behavior_or_choice");

    await user.click(within(dialog).getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("dialog", { name: "AI review details" }))
      .toBeNull();
    await waitFor(() => expect(document.activeElement).toBe(reviewButton));
  });

  it.each([
    [
      "active-nisha",
      activeReplayJson,
      "No active Drift is confirmed at this cutoff.",
      "Not Conflict",
    ],
    [
      "persistent-lukas",
      persistentReplayJson,
      "No active Drift is confirmed at this cutoff.",
      "Not Conflict",
    ],
    [
      "uncertain-wei-jun",
      uncertainReplayJson,
      "blocked recent Conflict evidence.",
      "Abstain",
    ],
  ])(
    "explains the final state change for %s",
    (scenarioId, scenarioJson, reason, decision) => {
      matchMedia(false);
      const scenarioFixture = validateExperienceInspectFixture(scenarioJson);
      const item = catalog.scenarios.find(
        (candidate) => candidate.scenario_id === scenarioId,
      )!;
      const weekIndex = scenarioFixture.scenario.weeks.length - 1;

      render(
        <PersonaReplayExperience
          loaded={{ catalogItem: item, fixture: scenarioFixture }}
          weekIndex={weekIndex}
          profile={scenarioFixture.scenario.profile}
          experience={experienceForWeek(weekIndex, scenarioFixture, item)}
          updateExperience={() => undefined}
          inspectRun={() => undefined}
              onWeekChange={() => undefined}
        />,
      );

      fireEvent.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
      fireEvent.click(screen.getByText("Why this state"));
      expect(screen.getAllByText((content) => content.includes(reason)).length)
        .toBeGreaterThan(0);
      expect(screen.getAllByText(decision).length).toBeGreaterThan(0);
    },
  );

  it("shows one reading panel and reveals only the current week's result", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);

    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));

    const resultHeading = screen.getByRole("heading", { name: /^Drift Detection \(End of Week\)/ });
    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Journal Entries" })).toBeNull();
    expect(screen.queryByRole("button", { name: /Open Journal Entry/ })).toBeNull();
    await waitFor(() => expect(document.activeElement).toBe(resultHeading));
    expect((screen.getByRole("button", { name: "Show week 2, outcome hidden" }) as HTMLButtonElement).disabled)
      .toBe(false);

    await user.click(screen.getByRole("button", { name: "Read Journal Entries" }));
    expect(screen.getByRole("heading", { name: "Journal Entries" })).toBeTruthy();
    await waitFor(() => expect(document.activeElement)
      .toBe(screen.getByRole("heading", { name: "Journal Entries" })));
    expect(screen.getAllByRole("button", { name: /Open Journal Entry/ })).toHaveLength(2);
    expect(screen.queryByRole("heading", { name: /^Drift Detection \(End of Week\)/ })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Read Weekly Drift Detection" }));
    expect(screen.getByRole("heading", { name: "No Active Drift" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Journal Entries" })).toBeNull();
    expect((screen.getByRole("button", { name: "Show week 2, outcome hidden" }) as HTMLButtonElement).disabled)
      .toBe(false);
  });

  it("confirms before a saved persona replaces manual progress", async () => {
    matchMedia(false);
    vi.useFakeTimers();
    render(<App />);
    enterPreferredName();
    const first = screen.getAllByTestId("value-card")[0];
    fireEvent.click(first);
    fireEvent.click(
      screen.getAllByTestId("value-card").find(
        (card) => card.dataset.location === "pool",
      )!,
    );
    act(() => vi.advanceTimersByTime(1_000));
    expect(screen.getByLabelText("Values · 2 of 11")).toBeTruthy();
    vi.useRealTimers();
    const confirm = vi.fn(() => false);
    vi.stubGlobal("confirm", confirm);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((input: string) => {
        if (input === "/scenarios/index.json") {
          return Promise.resolve({
            ok: true,
            json: async () => scenarioCatalogJson,
          });
        }
        return Promise.resolve(scenarioResponse());
      }),
    );
    const user = userEvent.setup();

    await user.click(screen.getByRole("button", { name: "Go home" }));
    await user.click(screen.getByRole("button", { name: /Try (?:the )?demo/i }));
    await screen.findByRole("radio", { name: "Nisha Agarwal" });
    expect(screen.queryByRole("navigation", { name: "Experience sections" })).toBeNull();
    expect(screen.getByRole("heading", { name: "See how Twinkl works" })).toBeTruthy();
    await user.click(
      screen.getByRole("button", {
        name: "Start at week 1",
      }),
    );

    expect(confirm).toHaveBeenCalledWith(
      "Load this saved Persona and replace your current progress?",
    );
    expect(
      screen.getByRole("heading", {
        name: "Explore five synthetic Personas, each with a different Drift profile.",
      }),
    ).toBeTruthy();
    expect(
      (screen.getByRole("button", {
        name: "Start at week 1",
      }) as HTMLButtonElement).disabled,
    ).toBe(false);
    const stored = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY)!);
    expect(stored.responses).toHaveLength(1);
    expect(stored.experience.selected_persona_id).toBeNull();
  });

  it("re-projects and clamps a restored saved replay", async () => {
    matchMedia(false);
    const profile = fixture.scenario.profile;
    const saved = createSession(() => 0.5);
    saved.user_id = profile.user_id;
    saved.preferred_name = profile.preferred_name ?? "Friend";
    saved.session_id = profile.session_id;
    saved.started_at = profile.started_at;
    saved.stage = "complete";
    saved.set_index = 10;
    saved.set_order = Array.from({ length: 11 }, (_, index) => index);
    saved.displayed_orders = Array.from(
      { length: 11 },
      (_, index) =>
        profile.bws_responses.find(
          (response) => response.set_number === index + 1,
        )!.item_order_shown,
    );
    saved.responses = profile.bws_responses;
    saved.confirmed_profile = profile;
    saved.experience = {
      ...experienceForWeek(0),
      selected_week: 20,
      journal_entries: [],
      trace_event_ids: [],
      trace_events: [],
    };
    expect(parseSession(JSON.stringify(saved))).not.toBeNull();
    localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(saved));
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((input: string) => {
        if (input === "/scenarios/index.json") {
          return Promise.resolve({
            ok: true,
            json: async () => scenarioCatalogJson,
          });
        }
        return Promise.resolve(scenarioResponse());
      }),
    );

    render(<App />);

    expect(await screen.findByText("Week 5 of 5")).toBeTruthy();
    await waitFor(() => {
      const stored = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY)!);
      expect(stored.experience.selected_week).toBe(4);
      expect(stored.experience.journal_entries).toHaveLength(
        fixture.scenario.journal_entries.length,
      );
    });
  });

  it("preserves persona, week, Journal Entry, and event across Inspect", async () => {
    matchMedia(false);
    const fetchMock = vi.fn().mockImplementation((input: string) => {
      if (input === "/scenarios/index.json") {
        return Promise.resolve({
          ok: true,
          json: async () => scenarioCatalogJson,
        });
      }
      if (input === "/scenarios/active-nisha.json") {
        return Promise.resolve(scenarioResponse());
      }
      return Promise.resolve({ ok: false, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: /Try (?:the )?demo/i }));
    await screen.findByRole("radio", { name: "Nisha Agarwal" });
    document.documentElement.scrollTop = 640;
    document.body.scrollTop = 640;
    await user.click(
      screen.getByRole("button", {
        name: "Start at week 1",
      }),
    );
    expect(
      await screen.findByRole("heading", {
        name: "Nisha Agarwal",
        level: 1,
      }),
    ).toBeTruthy();
    expect(screen.queryByRole("navigation", { name: "Experience sections" })).toBeNull();
    const weekNavigation = screen.getByRole("navigation", { name: "Replay weeks" });
    expect(within(weekNavigation).getAllByRole("button", { name: /^Show week/ })).toHaveLength(5);
    expect(within(weekNavigation).getByText("Feb 10–16, 2025")).toBeTruthy();
    expect(document.documentElement.scrollTop).toBe(0);
    expect(document.body.scrollTop).toBe(0);
    await user.click(screen.getByRole("button", {
      name: "Show Active Drift — week 4",
    }));
    expect(screen.getByText("Week 4 of 5")).toBeTruthy();
    const selectedEntryId =
      fixture.scenario.weeks[3].journal_entry_ids[0];
    const entryButton = screen.getByRole("button", {
      name: /Open Journal Entry 1/,
    });
    await user.click(entryButton);
    expect(entryButton.getAttribute("aria-current")).toBe("true");
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Close Journal Entry",
      }),
    );
    expect(
      screen.queryByRole("button", { name: "Inspect this run" }),
    ).toBeNull();
    await user.click(screen.getByRole("button", { name: "Review Weekly Drift Detection" }));
    await user.click(
      screen.getByRole("button", { name: "Inspect decision" }),
    );
    expect(
      screen.getByRole("heading", {
        name: "How Twinkl reached this result.",
      }),
    ).toBeTruthy();
    const inspectSections = screen.getByRole("navigation", {
      name: "Experience sections",
    });
    ["Summary", "Recorded work"].forEach((label) => {
      expect(
        within(inspectSections).getByRole("link", { name: label }),
      ).toBeTruthy();
    });
    expect(screen.getAllByText("Technical details").length).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", { name: "Experience" }));
    expect(screen.getByText("Week 4 of 5")).toBeTruthy();
    expect(
      screen.getByRole("button", { name: /Open Journal Entry 1/ })
        .getAttribute("aria-current"),
    ).toBe("true");

    const stored = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY)!);
    expect(stored.experience.selected_persona_id).toBe(
      catalogItem.persona_id,
    );
    expect(stored.experience.selected_week).toBe(3);
    expect(stored.experience.selected_entry_id).toBe(selectedEntryId);
    expect(stored.experience.selected_event_id).not.toBeNull();
    expect(parseSession(JSON.stringify(stored))).not.toBeNull();

    await user.click(screen.getByRole("button", { name: "Show week 3: no active drift" }));
    expect(screen.getByText("Week 3 of 5")).toBeTruthy();
    await waitFor(() => {
      const changedWeek = JSON.parse(
        localStorage.getItem(SESSION_STORAGE_KEY)!,
      );
      expect(changedWeek.experience.selected_entry_id).toBeNull();
      expect(changedWeek.experience.selected_event_id).toBeNull();
    });
  });
});
