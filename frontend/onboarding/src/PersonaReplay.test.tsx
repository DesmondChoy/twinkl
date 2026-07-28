import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import activeReplayJson from "../public/scenarios/active-wei-jun.json";
import activeReplayRaw from "../public/scenarios/active-wei-jun.json?raw";
import recoveredReplayJson from "../public/scenarios/recovered-marc.json";
import scenarioCatalogJson from "../public/scenarios/index.json";
import stableReplayJson from "../public/scenarios/stable-meera.json";
import twoValuesReplayJson from "../public/scenarios/two-values-lukas.json";
import uncertainReplayJson from "../public/scenarios/uncertain-noor.json";
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
import { journalEntryAnchorId } from "./journalEntryAnchor";

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

function ReplayHarness() {
  const [weekIndex, setWeekIndex] = useState(0);
  const [experience, setExperience] = useState(() => experienceForWeek(0));
  const changeWeek = (nextWeek: number) => {
    setWeekIndex(nextWeek);
    setExperience(experienceForWeek(nextWeek));
  };
  return (
    <PersonaReplayExperience
      loaded={loaded}
      weekIndex={weekIndex}
      profile={fixture.scenario.profile}
      experience={experience}
      updateExperience={(patch) =>
        setExperience((current) => ({ ...current, ...patch }))
      }
      inspectRun={() => undefined}
      onChoosePersona={() => undefined}
      onWeekChange={changeWeek}
    />
  );
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

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("persona replay", () => {
  it("loads a selected persona from the saved catalog", async () => {
    const onLoad = vi.fn<(loaded: LoadedScenario) => boolean>(() => true);
    const activeCatalog = {
      ...scenarioCatalogJson,
      scenarios: scenarioCatalogJson.scenarios
        .filter((item) => item.scenario_id === "active-wei-jun")
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

    render(<PersonaReplayPicker onBack={() => undefined} onLoad={onLoad} />);
    await user.click(
      await screen.findByRole("button", { name: "Replay Wei Jun Chen" }),
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
        .filter((item) => item.scenario_id === "active-wei-jun")
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

    render(<PersonaReplayPicker onBack={() => undefined} onLoad={onLoad} />);
    await user.click(
      await screen.findByRole("button", { name: "Replay Wei Jun Chen" }),
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
        onBack={() => undefined}
        onLoad={() => true}
      />,
    );

    const current = await screen.findByRole("radio", {
      name: /Wei Jun Chen.*Current/i,
    });
    expect((current as HTMLInputElement).checked).toBe(true);
    expect(
      (screen.getByRole("button", {
        name: "Current replay",
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

    render(<PersonaReplayPicker onBack={() => undefined} onLoad={() => true} />);
    await user.click(
      await screen.findByRole("button", { name: "Try loading again" }),
    );

    expect(await screen.findByRole("radio", { name: /Lukas Vermeer/i }))
      .toBeTruthy();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("supports explicit boundaries, restart, and automatic playback", () => {
    matchMedia(false);
    vi.useFakeTimers();
    render(<ReplayHarness />);

    expect(screen.getByText("Week 1 of 6")).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: "Previous" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("Week 2 of 6")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Play" }));
    expect(screen.getByRole("button", { name: "Pause" })).toBeTruthy();
    act(() => vi.advanceTimersByTime(3_999));
    expect(screen.getByText("Week 2 of 6")).toBeTruthy();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByText("Week 3 of 6")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    fireEvent.click(screen.getByRole("button", { name: "Restart scenario" }));
    expect(screen.getByText("Week 1 of 6")).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: "Show week 3: no drift" }),
    ).toBeNull();
    expect(
      screen.getByRole("listitem", {
        name: "Week 3, not yet replayed",
      }),
    ).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: "Previous" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it("disables automatic playback while keeping explicit controls", async () => {
    matchMedia(true);
    const user = userEvent.setup();
    render(<ReplayHarness />);

    expect(
      (screen.getByRole("button", { name: "Play" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(screen.getByText(/automatic replay is off/i)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByText("Week 2 of 6")).toBeTruthy();
  });

  it("advances with keyboard input", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);
    const next = screen.getByRole("button", { name: "Next" });

    next.focus();
    await user.keyboard("{Enter}");

    expect(screen.getByText("Week 2 of 6")).toBeTruthy();
    expect(document.activeElement).toBe(next);
    expect(
      screen.getByRole("listitem", {
        name: "Week 6, not yet replayed",
      }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("listitem", {
        name: "Week 6: Active Drift",
      }),
    ).toBeNull();
  });

  it("lets people revisit revealed weeks while future weeks stay inert", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);

    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(
      screen.getByRole("button", {
        name: "Show week 1: no drift",
      }),
    );

    expect(screen.getByText("Week 1 of 6")).toBeTruthy();
    expect(
      (screen.getByRole("button", {
        name: "Restart scenario",
      }) as HTMLButtonElement).disabled,
    ).toBe(false);
    await user.click(
      screen.getByRole("button", {
        name: "Show week 3: no drift",
      }),
    );
    expect(screen.getByText("Week 3 of 6")).toBeTruthy();
    expect(
      screen.queryByRole("button", {
        name: /show week 6/i,
      }),
    ).toBeNull();
  });

  it("clamps an out-of-range restored week to the final week", () => {
    matchMedia(false);
    render(
      <PersonaReplayExperience
        loaded={loaded}
        weekIndex={20}
        profile={fixture.scenario.profile}
        experience={experienceForWeek(5)}
        updateExperience={() => undefined}
        inspectRun={() => undefined}
        onChoosePersona={() => undefined}
        onWeekChange={() => undefined}
      />,
    );

    expect(screen.getByText("Week 6 of 6")).toBeTruthy();
    expect(screen.getByRole("heading", { name: /Jul 6, 2025/i })).toBeTruthy();
  });

  it.each([
    ["stable-meera", stableReplayJson, "No Drift"],
    ["active-wei-jun", activeReplayJson, "Active Drift"],
    ["recovered-marc", recoveredReplayJson, "Recovered Drift"],
    ["uncertain-noor", uncertainReplayJson, "Uncertain"],
    ["two-values-lukas", twoValuesReplayJson, "Mixed"],
  ])("renders the final %s progression", (scenarioId, scenarioJson, label) => {
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
        onChoosePersona={() => undefined}
        onWeekChange={() => undefined}
      />,
    );

    expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    expect(screen.queryByText("Raw provider response")).toBeNull();
    expect(screen.queryByText("Validation result")).toBeNull();
  });

  it("names the Core Value states behind a Mixed week", () => {
    matchMedia(false);
    const scenarioFixture =
      validateExperienceInspectFixture(twoValuesReplayJson);
    const item = catalog.scenarios.find(
      (candidate) => candidate.scenario_id === "two-values-lukas",
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
        onChoosePersona={() => undefined}
        onWeekChange={() => undefined}
      />,
    );

    expect(
      screen.getByText(
        "Self-Direction is Uncertain; Conformity shows Recovered Drift.",
      ),
    ).toBeTruthy();
  });

  it("confirms before a saved persona replaces manual progress", async () => {
    matchMedia(false);
    vi.useFakeTimers();
    render(<App />);
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

    await user.click(screen.getByRole("button", { name: "Try demo" }));
    await user.click(
      await screen.findByRole("radio", { name: /Wei Jun Chen/i }),
    );
    await user.click(
      screen.getByRole("button", { name: "Replay Wei Jun Chen" }),
    );

    expect(confirm).toHaveBeenCalledWith(
      "Load this saved persona and replace your current progress?",
    );
    expect(
      screen.getByRole("heading", {
        name: "Follow a life week by week.",
      }),
    ).toBeTruthy();
    expect(
      (screen.getByRole("button", {
        name: "Replay Wei Jun Chen",
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

    expect(await screen.findByText("Week 6 of 6")).toBeTruthy();
    await waitFor(() => {
      const stored = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY)!);
      expect(stored.experience.selected_week).toBe(5);
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
      if (input === "/scenarios/active-wei-jun.json") {
        return Promise.resolve(scenarioResponse());
      }
      return Promise.resolve({ ok: false, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Try demo" }));
    await user.click(
      await screen.findByRole("radio", { name: /Wei Jun Chen/i }),
    );
    await user.click(
      screen.getByRole("button", { name: "Replay Wei Jun Chen" }),
    );
    expect(
      await screen.findByRole("heading", {
        name: "Wei Jun Chen, week by week.",
      }),
    ).toBeTruthy();
    for (let week = 1; week < 6; week += 1) {
      await user.click(screen.getByRole("button", { name: "Next" }));
    }
    expect(screen.getByText("Week 6 of 6")).toBeTruthy();
    const citation = screen.getAllByRole("link", {
      name: /Open Journal Entry/,
    })[0];
    await user.click(citation);
    const selectedEntryId = citation.getAttribute("href")!.slice(1);
    expect(document.getElementById(selectedEntryId)?.getAttribute("aria-current"))
      .toBe("true");
    expect(
      screen.queryByRole("button", { name: "Inspect this run" }),
    ).toBeNull();
    await user.click(
      screen.getByRole("button", { name: "Inspect Weekly Digest run" }),
    );
    expect(
      screen.getByRole("heading", {
        name: "Follow the work, event by event.",
      }),
    ).toBeTruthy();
    expect(screen.getAllByText("Saved replay").length).toBeGreaterThan(0);
    await user.click(screen.getByRole("button", { name: "Experience" }));
    expect(screen.getByText("Week 6 of 6")).toBeTruthy();
    expect(document.getElementById(selectedEntryId)?.getAttribute("aria-current"))
      .toBe("true");

    const stored = JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY)!);
    expect(stored.experience.selected_persona_id).toBe(
      catalogItem.persona_id,
    );
    expect(stored.experience.selected_week).toBe(5);
    expect(journalEntryAnchorId(stored.experience.selected_entry_id)).toBe(
      selectedEntryId,
    );
    expect(stored.experience.selected_event_id).not.toBeNull();
    expect(parseSession(JSON.stringify(stored))).not.toBeNull();
  });
});
