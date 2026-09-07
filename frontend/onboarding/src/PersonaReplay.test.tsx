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
import recoveredReplayJson from "../public/scenarios/ended-sook-yin.json";
import scenarioCatalogJson from "../public/scenarios/index.json";
import stableReplayJson from "../public/scenarios/stable-noor.json";
import twoValuesReplayJson from "../public/scenarios/two-values-henrik.json";
import twoValuesReplayRaw from "../public/scenarios/two-values-henrik.json?raw";
import uncertainReplayJson from "../public/scenarios/uncertain-wei-jun.json";
import savedCoachResponses from "../../../src/demo/coach_digest_responses.json";
import App from "./App";
import styles from "./styles.css?raw";
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
import { northStarProfileRef, type NorthStarRecord } from "./northStar";

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
}: {
  scenarioJson?: unknown;
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
      inspectRun={() => undefined}
      onChoosePersona={() => undefined}
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
  const card = screen.getByText(name).closest("article");
  if (!card) throw new Error(`Persona card not found: ${name}`);
  return card;
}

async function startHenrikReplay() {
  matchMedia(false);
  vi.stubGlobal("fetch", vi.fn().mockImplementation((input: string) => {
    if (input === "/scenarios/index.json") {
      return Promise.resolve({ ok: true, json: async () => scenarioCatalogJson });
    }
    return Promise.resolve(input === "/scenarios/two-values-henrik.json"
      ? scenarioResponse(twoValuesReplayRaw)
      : { ok: false });
  }));
  const user = userEvent.setup();
  const view = render(<App />);
  await user.click(screen.getByRole("button", { name: /Try (?:the )?demo/i }));
  await screen.findByText("Henrik Larsson");
  await user.click(within(personaCard("Henrik Larsson")).getByRole("button", {
    name: "Start at week 1",
  }));
  await screen.findByRole("heading", { name: "Henrik Larsson", level: 1 });
  return { user, view };
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("persona replay", () => {
  it("keeps revealed later weeks after returning from Inspect and reloading an earlier week", async () => {
    const { user, view } = await startHenrikReplay();
    await user.click(screen.getByRole("button", {
      name: "Show independent Core Value states — week 3",
    }));
    await user.click(screen.getByRole("button", { name: "Show week 2: no active drift" }));
    await user.click(screen.getByRole("button", { name: "Inspect" }));
    expect(screen.queryByRole("button", { name: "View Profile calculation" })).toBeNull();
    expect(screen.getByText(/This Persona Profile is a synthetic projection/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Return to Experience" }));
    expect(screen.getByText("Week 2 of 6")).toBeTruthy();
    expect((screen.getByRole("button", {
      name: "Show week 3: insufficient evidence",
    }) as HTMLButtonElement).disabled).toBe(false);
    const states = document.querySelectorAll(".state-change > header");
    expect(Array.from(states, (header) => header.textContent)).toEqual([
      "StimulationNo Active Drift", "SecurityNo Active Drift",
    ]);

    view.unmount();
    render(<App />);
    await screen.findByText("Week 2 of 6");
    expect((screen.getByRole("button", {
      name: "Show week 3: insufficient evidence",
    }) as HTMLButtonElement).disabled).toBe(false);
    await user.click(screen.getByRole("button", { name: "Show week 3: insufficient evidence" }));
    expect(Array.from(document.querySelectorAll(".state-change > header"),
      (header) => header.textContent)).toEqual([
      "StimulationInsufficient Evidence", "SecurityNo Active Drift",
    ]);
  });

  it("preserves partial first-week and later-week steps across Inspect and reload", async () => {
    const { user, view } = await startHenrikReplay();
    await user.click(screen.getByRole("button", { name: "Next step" }));
    await user.click(screen.getByRole("button", { name: "Inspect" }));
    await user.click(screen.getByRole("button", { name: "Return to Experience" }));
    expect(screen.getByRole("button", { name: /Open Journal Entry 1/ })).toBeTruthy();
    expect(document.querySelector(".replay-result")).toBeNull();

    view.unmount();
    const restored = render(<App />);
    await screen.findByRole("button", { name: /Open Journal Entry 1/ });
    expect(document.querySelector(".replay-result")).toBeNull();
    for (let step = 0; step < 10 && !document.querySelector(".replay-result"); step += 1) {
      await user.click(screen.getByRole("button", { name: "Next step" }));
    }
    expect(document.querySelector(".replay-result")).not.toBeNull();
    await user.click(screen.getByRole("button", { name: "Next step" }));
    expect(screen.getByText("Week 2 of 6")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Inspect" }));
    await user.click(screen.getByRole("button", { name: "Return to Experience" }));
    expect(screen.queryByRole("button", { name: /Open Journal Entry/ })).toBeNull();
    expect(document.querySelector(".replay-result")).toBeNull();

    restored.unmount();
    render(<App />);
    await screen.findByText("Week 2 of 6");
    expect(screen.queryByRole("button", { name: /Open Journal Entry/ })).toBeNull();
    expect(document.querySelector(".replay-result")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Next step" }));
    expect(screen.getByRole("button", { name: /Open Journal Entry 1/ })).toBeTruthy();
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

    render(<PersonaReplayPicker onBack={() => undefined} onLoad={onLoad} />);
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

    render(<PersonaReplayPicker onBack={() => undefined} onLoad={onLoad} />);
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
        onBack={() => undefined}
        onLoad={() => true}
      />,
    );

    await screen.findByText("Nisha Agarwal");
    const current = personaCard("Nisha Agarwal");
    expect(within(current).getByText("Current")).toBeTruthy();
    expect(
      (within(current).getByRole("button", {
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

    expect(await screen.findByText("Henrik Larsson"))
      .toBeTruthy();
    expect(
      within(screen.getAllByRole("article")[0]).getByText("Nisha Agarwal"),
    ).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("uses manual steps to reveal Journal Entries, then the result", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);

    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "No Active Drift" })).toBeNull();
    expect(screen.queryByRole("button", {
      name: /Open Journal Entry 1/,
    })).toBeNull();
    expect(
      (screen.getByRole("button", { name: "Previous" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);

    await user.click(screen.getByRole("button", { name: "Next step" }));
    expect(screen.getByRole("button", {
      name: /Open Journal Entry 1/,
    })).toBeTruthy();
    expect(screen.queryByLabelText("Nudge for Journal Entry 1")).toBeNull();
    expect(screen.queryByRole("heading", { name: "No Active Drift" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Next step" }));
    expect(screen.getByRole("button", { name: /Open Journal Entry 2/ })).toBeTruthy();
    expect(screen.queryByLabelText("Nudge for Journal Entry 2")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Next step" }));
    expect(screen.getByLabelText("Nudge for Journal Entry 2")).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "No Active Drift" })).toBeNull();

    await user.click(screen.getByRole("button", { name: "Next step" }));
    expect(screen.getByRole("heading", { name: "No Active Drift" })).toBeTruthy();
    expect(screen.getByRole("heading", {
      name: "Weekly Drift Detection (based on 2 Journal Entries through Feb 16)",
    })).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Next step" }));
    expect(screen.getByText("Week 2 of 5")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Restart" }));
    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: "Show week 1: no active drift" }),
    ).toBeNull();
    expect(
      screen.getByRole("listitem", {
        name: "Week 1, not yet replayed",
      }),
    ).toBeTruthy();
  });

  it("paces automatic replay for reading", () => {
    matchMedia(false);
    vi.useFakeTimers();
    render(<ReplayHarness />);

    fireEvent.click(screen.getByRole("button", { name: "Auto replay" }));
    expect(screen.getByRole("button", { name: "Pause replay" })).toBeTruthy();
    act(() => vi.advanceTimersByTime(3_599));
    expect(screen.queryByRole("button", {
      name: /Open Journal Entry 1/,
    })).toBeNull();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByRole("button", {
      name: /Open Journal Entry 1/,
    })).toBeTruthy();
    expect(screen.queryByLabelText("Nudge for Journal Entry 1")).toBeNull();
    act(() => vi.advanceTimersByTime(3_599));
    expect(screen.queryByRole("button", { name: /Open Journal Entry 2/ })).toBeNull();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByRole("button", { name: /Open Journal Entry 2/ })).toBeTruthy();
    act(() => vi.advanceTimersByTime(799));
    expect(screen.queryByLabelText("Nudge for Journal Entry 2")).toBeNull();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByLabelText("Nudge for Journal Entry 2")).toBeTruthy();
    act(() => vi.advanceTimersByTime(3_199));
    expect(screen.queryByRole("heading", { name: "No Active Drift" })).toBeNull();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByRole("heading", { name: "No Active Drift" })).toBeTruthy();
    act(() => vi.advanceTimersByTime(5_999));
    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    act(() => vi.advanceTimersByTime(1));
    expect(screen.getByText("Week 2 of 5")).toBeTruthy();
  });

  it("disables automatic playback while keeping explicit controls", async () => {
    matchMedia(true);
    const user = userEvent.setup();
    render(<ReplayHarness />);

    expect(
      (screen.getByRole("button", { name: "Auto replay" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(screen.getByText(/automatic replay is off/i)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Next step" }));
    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    expect(screen.getByRole("button", {
      name: /Open Journal Entry 1/,
    })).toBeTruthy();
  });

  it("advances with keyboard input", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);
    const next = screen.getByRole("button", { name: "Next step" });

    next.focus();
    await user.keyboard("{Enter}");

    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
    expect(screen.getByRole("button", {
      name: /Open Journal Entry 1/,
    })).toBeTruthy();
    expect(document.activeElement).toBe(next);
    expect(
      screen.getByRole("listitem", {
        name: "Week 5, not yet replayed",
      }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("listitem", {
        name: "Week 5: No Active Drift",
      }),
    ).toBeNull();
  });

  it("keeps Journal Entries compact and opens the full text in a dialog", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    render(<ReplayHarness />);

    await user.click(screen.getByRole("button", { name: "Next step" }));
    const entry = experienceForWeek(0).journal_entries[0];
    const entryButton = screen.getByRole("button", {
      name: /Open Journal Entry 1/,
    });
    expect(entryButton.textContent).not.toContain(entry.content);

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
    expect(futureWeek.disabled).toBe(true);
    await user.click(screen.getByRole("button", {
      name: "Show Active Drift — week 4",
    }));
    expect(screen.getByText("Week 4 of 5")).toBeTruthy();
    await user.click(
      screen.getByRole("button", {
        name: "Show week 1: no active drift",
      }),
    );

    expect(screen.getByText("Week 1 of 5")).toBeTruthy();
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
    ["two-values-henrik", twoValuesReplayJson],
    ["stable-noor", stableReplayJson],
    ["active-nisha", activeReplayJson],
    ["ended-sook-yin", recoveredReplayJson],
    ["uncertain-wei-jun", uncertainReplayJson],
  ])(
    "shows the fresh Coach Digest and saved North Star Moment evidence for %s",
    async (scenarioId, scenarioJson) => {
      matchMedia(false);
      const user = userEvent.setup();
      const savedResponse = Object.values(savedCoachResponses.responses).find(
        (response) => response.scenario_id === scenarioId,
      )!;

      render(<ScenarioReplayHarness scenarioJson={scenarioJson} />);
      expect(screen.queryByText("No saved Coach Digest for this result")).toBeNull();
      expect(screen.queryByRole("heading", { name: "Your weekly reflection" })).toBeNull();
      await user.click(screen.getByRole("button", {
        name: /^Show .+ — week \d+$/,
      }));

      expect(screen.queryByText("No saved Coach Digest for this result")).toBeNull();
      expect(screen.getByText("Why this state")).toBeTruthy();
      expect(screen.getByRole("heading", { name: "Your weekly reflection" })).toBeTruthy();
      const tension = document.querySelector(".coach-digest > p:nth-of-type(3)");
      expect(tension?.textContent).toContain(
        savedResponse.narrative.tension_explanation.split("...")[0],
      );
      expect(tension?.textContent).not.toMatch(/(?:\.{3}|…)[”"]/);
      expect(document.querySelector(".coach-digest__question")?.textContent)
        .toBe(savedResponse.narrative.reflective_question);
      const saved = validateExperienceInspectFixture(scenarioJson);
      const item = catalog.scenarios.find((candidate) => candidate.scenario_id === scenarioId)!;
      const keyWeek = saved.scenario.weeks.find((week) => week.week_start === item.key_week_start)!;
      const record = saved.trace_events.find((event) =>
        keyWeek.event_ids.includes(event.event_id) && event.event_type === "north_star_reviewed"
      )!.details.record as NorthStarRecord;
      if (record.selected) {
        await waitFor(() => expect(document.querySelector(".north-star-moment blockquote")?.textContent)
          .toBe(record.selected!.evidence_quote));
        await user.click(screen.getByRole("link", { name: /^Open (response in )?Journal Entry ·/ }));
        const source = saved.scenario.journal_entries.find((entry) => entry.journal_entry_id === record.selected!.entry_id)!;
        expect(screen.getByRole("dialog").querySelector(".replay-entry-drawer__content")?.textContent)
          .toBe(source.content);
        await user.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Close Journal Entry" }));
      } else {
        expect(record.reason).toBe("insufficient_evidence");
        expect(document.querySelector(".north-star-moment")).toBeNull();
      }
      await user.click(screen.getByRole("button", { name: "Show week 1: no active drift" }));
      expect(screen.getByText("No saved Coach Digest for this result")).toBeTruthy();
      expect(document.querySelector(".coach-digest__question")).toBeNull();
      expect(screen.queryByRole("heading", { name: "Your weekly reflection" })).toBeNull();
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
        onChoosePersona={() => undefined}
        onWeekChange={() => undefined}
      />,
    );

    expect(screen.getByText("Week 5 of 5")).toBeTruthy();
    expect(screen.getByRole("heading", { name: /Mar 10–16, 2025/i })).toBeTruthy();
  });

  it.each([
    ["stable-noor", stableReplayJson, "No Active Drift"],
    ["active-nisha", activeReplayJson, "No Active Drift"],
    ["ended-sook-yin", recoveredReplayJson, "No Active Drift"],
    ["uncertain-wei-jun", uncertainReplayJson, "Insufficient Evidence"],
    ["two-values-henrik", twoValuesReplayJson, "No Active Drift"],
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
      (candidate) => candidate.scenario_id === "two-values-henrik",
    )!;

    render(
      <PersonaReplayExperience
        loaded={{ catalogItem: item, fixture: scenarioFixture }}
        weekIndex={1}
        profile={scenarioFixture.scenario.profile}
        experience={experienceForWeek(1, scenarioFixture, item)}
        updateExperience={() => undefined}
        inspectRun={() => undefined}
        onChoosePersona={() => undefined}
        onWeekChange={() => undefined}
      />,
    );

    const nudge = screen.getByLabelText("Nudge for Journal Entry 2");
    const entry = screen.getByRole("button", {
      name: "Open Journal Entry 2 from Feb 14",
    });

    expect(nudge.textContent).toContain(
      "When you say splitting, which half feels more like you?",
    );
    expect(entry.closest("li")?.contains(nudge)).toBe(true);
    expect(nudge.classList.contains("nudge-reveal")).toBe(true);
  });

  it("does not show a nudge when the selected week has none", () => {
    matchMedia(false);
    const scenarioFixture =
      validateExperienceInspectFixture(twoValuesReplayJson);
    const item = catalog.scenarios.find(
      (candidate) => candidate.scenario_id === "two-values-henrik",
    )!;

    render(
      <PersonaReplayExperience
        loaded={{ catalogItem: item, fixture: scenarioFixture }}
        weekIndex={2}
        profile={scenarioFixture.scenario.profile}
        experience={experienceForWeek(2, scenarioFixture, item)}
        updateExperience={() => undefined}
        inspectRun={() => undefined}
        onChoosePersona={() => undefined}
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
      (candidate) => candidate.scenario_id === "two-values-henrik",
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

    const values = document.querySelector(".replay-persona__value");
    expect(values?.textContent).toContain("Schwartz Core Values");
    expect(values?.textContent).toContain("Stimulation · Security");
    expect(screen.queryByText("Having the freedom to choose my own path"))
      .toBeNull();
    expect(screen.queryByText(
      "Being someone others can count on to do the right thing",
    )).toBeNull();
  });

  it("keeps Weekly Drift Detection and an available Coach Digest together", async () => {
    matchMedia(false);
    const user = userEvent.setup();
    const inspectRun = vi.fn();
    const scenarioFixture =
      validateExperienceInspectFixture(twoValuesReplayJson);
    const item = catalog.scenarios.find(
      (candidate) => candidate.scenario_id === "two-values-henrik",
    )!;
    const weekIndex = scenarioFixture.scenario.weeks.findIndex((week) => week.week_start === item.key_week_start);
    const experience = experienceForWeek(weekIndex, scenarioFixture, item);
    const narrative = Object.values(savedCoachResponses.responses).find(
      (response) => response.scenario_id === item.scenario_id,
    )!.narrative;

    render(
      <PersonaReplayExperience
        loaded={{ catalogItem: item, fixture: scenarioFixture }}
        weekIndex={weekIndex}
        profile={scenarioFixture.scenario.profile}
        experience={experience}
        updateExperience={() => undefined}
        inspectRun={inspectRun}
        onChoosePersona={() => undefined}
        onWeekChange={() => undefined}
      />,
    );

    const result = screen.getByRole("article", {
      name: "Insufficient Evidence",
    });
    const coachHeading = screen.getByRole("heading", {
      name: "Your weekly reflection",
    });
    expect(screen.getByText(narrative.weekly_mirror)).toBeTruthy();
    const coachCard = coachHeading.closest(".coach-digest--replay");
    const resultScroll = result.closest(".replay-column__scroll--result");

    expect(coachCard).not.toBeNull();
    expect(resultScroll?.contains(coachCard)).toBe(true);
    expect(
      result.compareDocumentPosition(coachCard!)
      & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(within(result).getByText("Why this state")).toBeTruthy();

    await user.click(screen.getByRole("button", { name: "Inspect decision" }));
    const northStarEvent = [...experience.trace_events]
      .reverse()
      .find((event) => event.event_type === "north_star_reviewed");
    expect(northStarEvent).toBeTruthy();
    expect(inspectRun).toHaveBeenCalledWith(northStarEvent!.event_id);
    expect(screen.queryByText("No saved Coach Digest for this result")).toBeNull();
  });

  it.each([
    { scenarioId: "uncertain-wei-jun", scenarioJson: uncertainReplayJson, eventType: "north_star_reviewed" },
    { scenarioId: "uncertain-wei-jun", scenarioJson: uncertainReplayJson, eventType: "drift_detected" },
    { scenarioId: "two-values-henrik", scenarioJson: twoValuesReplayJson, eventType: "weekly_digest_built" },
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
    const higherPriorityTypes = eventType === "drift_detected"
      ? ["north_star_reviewed", "weekly_coach_generated"]
      : eventType === "weekly_digest_built"
        ? ["north_star_reviewed", "weekly_coach_generated", "drift_detected"]
        : [];
    experience.trace_events = experience.trace_events.filter((event) =>
      !currentIds.has(event.event_id) || !higherPriorityTypes.includes(event.event_type));
    if (eventType === "north_star_reviewed") {
      const week = saved.scenario.weeks[weekIndex];
      const baseEvent = experience.trace_events.at(-1)!;
      const record: NorthStarRecord = {
        schema_version: "north-star-record-v1", session_id: saved.session.session_id,
        owner_id: saved.scenario.profile.user_id, profile_ref: await northStarProfileRef(saved.scenario.profile),
        week_start: week.week_start, week_end: week.week_end, cutoff_at: baseEvent.started_at,
        input_hash: "a".repeat(64), status: "failed", mode: null, reason: "budget_unavailable",
        core_value: null, value_phrase: null, selected: null, sources: [], reviews: [],
        onset_t_index: null, onset_date: null, onset_available_at: null, attempts: 0, retryable: false,
      };
      const event = { ...baseEvent, event_id: `${scenarioId}:test-north-star`,
        event_type: "north_star_reviewed" as const, input_hash: record.input_hash, details: { record } };
      experience.trace_events = experience.trace_events.filter((candidate) =>
        !currentIds.has(candidate.event_id) || candidate.event_type !== "north_star_reviewed");
      experience.trace_events.push(event);
      experience.trace_event_ids.push(event.event_id);
      saved.trace_events.push(event);
      week.event_ids.push(event.event_id);
      currentIds.add(event.event_id);
    }
    const expected = experience.trace_events.find((event) =>
      currentIds.has(event.event_id) && event.event_type === eventType);
    expect(expected).toBeTruthy();
    expect(experience.trace_events.some((event) => event.event_type === "north_star_reviewed"
      && !currentIds.has(event.event_id))).toBe(true);
    render(<PersonaReplayExperience loaded={{ catalogItem: item, fixture: saved }}
      weekIndex={weekIndex} profile={saved.scenario.profile} experience={experience}
      updateExperience={() => undefined} inspectRun={inspectRun}
      onChoosePersona={() => undefined} onWeekChange={() => undefined} />);
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
        onChoosePersona={() => undefined}
        onWeekChange={() => undefined}
      />,
    );

    const result = screen.getByRole("article", { name: "Active Drift" });
    expect(within(result).getByRole("heading", { name: "Active Drift" })).toBeTruthy();
    expect(result.querySelector(".state-change > header")?.textContent)
      .toBe("UniversalismActive Drift");
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
      "ended-sook-yin",
      recoveredReplayJson,
      "No active Drift is confirmed at this cutoff.",
      "Conflict",
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
          onChoosePersona={() => undefined}
          onWeekChange={() => undefined}
        />,
      );

      expect(screen.getAllByText((content) => content.includes(reason)).length)
        .toBeGreaterThan(0);
      expect(screen.getAllByText(decision).length).toBeGreaterThan(0);
    },
  );

  it("uses a fixed two-column workspace with internal scrolling", () => {
    expect(styles).toMatch(
      /\.replay-workspace\s*\{[\s\S]*?grid-template-columns:/,
    );
    expect(styles).toMatch(
      /\.replay-column__scroll\s*\{[\s\S]*?overflow-y:\s*auto;/,
    );
    expect(styles).toMatch(
      /\.app-shell--journal\.app-shell--saved-persona\s*\{[\s\S]*?overflow:\s*hidden;/,
    );
    expect(styles).toContain(".replay-workspace__switch");
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

    await user.click(screen.getByRole("button", { name: /Try (?:the )?demo/i }));
    await screen.findByText("Nisha Agarwal");
    const pickerSections = screen.getByRole("navigation", {
      name: "Experience sections",
    });
    ["Introduction", "Personas", "Evidence"].forEach((label) => {
      expect(
        within(pickerSections).getByRole("link", { name: label }),
      ).toBeTruthy();
    });
    await user.click(
      within(personaCard("Nisha Agarwal")).getByRole("button", {
        name: "Start at week 1",
      }),
    );

    expect(confirm).toHaveBeenCalledWith(
      "Load this saved Persona and replace your current progress?",
    );
    expect(
      screen.getByRole("heading", {
        name: "Choose what you want to observe.",
      }),
    ).toBeTruthy();
    expect(
      (within(personaCard("Nisha Agarwal")).getByRole("button", {
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
    await screen.findByText("Nisha Agarwal");
    document.documentElement.scrollTop = 640;
    document.body.scrollTop = 640;
    await user.click(
      within(personaCard("Nisha Agarwal")).getByRole("button", {
        name: "Start at week 1",
      }),
    );
    expect(
      await screen.findByRole("heading", {
        name: "Nisha Agarwal",
        level: 1,
      }),
    ).toBeTruthy();
    const replaySections = screen.getByRole("navigation", {
      name: "Experience sections",
    });
    ["Persona", "Week", "Journal Entries", "Weekly Drift"].forEach(
      (label) => {
        expect(
          within(replaySections).getByRole("link", { name: label }),
        ).toBeTruthy();
      },
    );
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

    await user.click(screen.getByRole("button", { name: "Previous" }));
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
