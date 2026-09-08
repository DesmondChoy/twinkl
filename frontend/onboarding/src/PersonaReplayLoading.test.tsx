import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import activeReplayJson from "../public/scenarios/active-nisha.json";
import scenarioCatalogJson from "../public/scenarios/index.json";
import App from "./App";
import { validateExperienceInspectFixture } from "./demoContracts";
import {
  loadSavedScenario,
  validateScenarioCatalog,
  type LoadedScenario,
} from "./scenarioReplay";
import { parseSession, SESSION_STORAGE_KEY } from "./session";

vi.mock("./scenarioReplay", async (importOriginal) => ({
  ...await importOriginal<typeof import("./scenarioReplay")>(),
  loadSavedScenario: vi.fn(),
}));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("leaving a loading Persona replay", () => {
  it.each(["wordmark", "home"])(
    "preserves new onboarding after leaving through %s before the replay loads",
    async (exit) => {
      const user = userEvent.setup();
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
        new Response(JSON.stringify(scenarioCatalogJson)),
      ));
      let finishLoading!: (loaded: LoadedScenario) => void;
      const pendingReplay = new Promise<LoadedScenario>((resolve) => {
        finishLoading = resolve;
      });
      vi.mocked(loadSavedScenario).mockReturnValue(pendingReplay);
      render(<App />);

      await user.click(screen.getByRole("button", { name: "Try the Demo" }));
      const starts = await screen.findAllByRole("button", { name: "Start at week 1" });
      await user.click(starts[0]);
      expect(screen.getByRole("button", { name: "Loading saved replay…" })).toBeTruthy();
      await user.click(exit === "wordmark"
        ? screen.getByRole("link", { name: "Twinkl home" })
        : screen.getByRole("button", { name: "Go home" }));
      await user.click(screen.getByRole("button", { name: "Try Onboarding" }));
      fireEvent.change(screen.getByRole("textbox", { name: "Preferred name" }), {
        target: { value: "Casey" },
      });
      await user.click(screen.getByRole("button", { name: "Continue" }));
      fireEvent.click(screen.getAllByTestId("value-card")[0]);
      const before = parseSession(localStorage.getItem(SESSION_STORAGE_KEY))!;

      const fixture = validateExperienceInspectFixture(activeReplayJson);
      const catalogItem = validateScenarioCatalog(scenarioCatalogJson).scenarios.find(
        (item) => item.persona_id === fixture.scenario.persona_id,
      )!;
      await act(async () => {
        finishLoading({ fixture, catalogItem });
        await pendingReplay;
      });

      expect(screen.getByLabelText("Values · 1 of 11")).toBeTruthy();
      expect(parseSession(localStorage.getItem(SESSION_STORAGE_KEY))).toEqual(before);
      expect(screen.queryByText("Nisha Agarwal · saved replay")).toBeNull();
    },
  );
});
