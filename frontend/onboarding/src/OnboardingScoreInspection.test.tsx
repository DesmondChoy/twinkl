import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import OnboardingScoreInspection from "./OnboardingScoreInspection";
import { VALUES, VALUE_ORDER, type ScoreBundle } from "./domain";
import { canonicalInspectFixture } from "./inspectFixture";

const profile = canonicalInspectFixture.session.profile;
const scores: ScoreBundle = {
  bws: profile.bws_results,
  profile: profile.value_profile,
};
const setOrder = Array.from({ length: 11 }, (_, index) => index);

describe("onboarding score inspection", () => {
  it("explains the complete browser calculation without leading with JSON", () => {
    render(
      <OnboardingScoreInspection
        confirmed={false}
        responses={profile.bws_responses}
        scores={scores}
        setOrder={setOrder}
      />,
    );

    expect(
      screen.getByRole("heading", {
        name: "Begin with the recorded choices.",
      }),
    ).toBeTruthy();
    expect(screen.getByText("Calculation method")).toBeTruthy();
    expect(screen.getByText("Deterministic · no model")).toBeTruthy();
    expect(
      screen.getByRole("list", { name: "SVBWS calculation steps" }),
    ).toBeTruthy();

    const mostSelections = screen.getByRole("region", {
      name: "Most",
    });
    const leastSelections = screen.getByRole("region", {
      name: "Least",
    });
    expect(within(mostSelections).getAllByRole("listitem")).toHaveLength(11);
    expect(within(leastSelections).getAllByRole("listitem")).toHaveLength(11);
    expect(within(mostSelections).queryByText(/× overall/)).toBeNull();
    expect(within(leastSelections).queryByText(/× overall/)).toBeNull();

    const selectionTotals = screen.getByRole("table", {
      name: "Most and Least totals by value",
    });
    expect(selectionTotals.querySelectorAll("tbody tr")).toHaveLength(11);
    expect(
      within(selectionTotals).getByRole("row", {
        name: "Benevolence 6 0",
      }),
    ).toBeTruthy();
    expect(
      within(selectionTotals).getByRole("row", {
        name: "Total choices 11 11",
      }),
    ).toBeTruthy();
    expect(screen.getByText("Why two Universalism objects?")).toBeTruthy();
    expect(
      screen.getByText(
        /The Profile averages them back into one Universalism score/,
      ),
    ).toBeTruthy();

    const profileMapping = screen.getByRole("region", {
      name: "Ten-value Profile scores and Experience mapping",
    });
    expect(
      profileMapping.querySelectorAll(".score-table__calculation"),
    ).toHaveLength(10);
    profile.top_values.forEach((value) => {
      expect(
        within(profileMapping).getByText(VALUES[value].phrase),
      ).toBeTruthy();
    });
    const expectedValueOrder = [...VALUE_ORDER]
      .sort((left, right) => {
        const weightDifference =
          scores.profile.weights[right] - scores.profile.weights[left];
        return (
          weightDifference ||
          VALUE_ORDER.indexOf(left) - VALUE_ORDER.indexOf(right)
        );
      })
      .map((value) => VALUES[value].name);
    const displayedValueOrder = within(profileMapping)
      .getAllByRole("row")
      .slice(1)
      .map((row) => within(row).getByRole("rowheader").textContent);
    expect(displayedValueOrder).toEqual(expectedValueOrder);
    expect(screen.getByText("No confidence score is inferred.")).toBeTruthy();
    expect(screen.queryByText("bws_results")).toBeNull();
  });

  it("distinguishes highest-scoring values from confirmed Core Values", () => {
    const { rerender } = render(
      <OnboardingScoreInspection
        confirmed={false}
        responses={profile.bws_responses}
        scores={scores}
        setOrder={setOrder}
      />,
    );

    expect(screen.getAllByText("Highest").length).toBe(
      profile.top_values.length,
    );

    rerender(
      <OnboardingScoreInspection
        confirmed
        responses={profile.bws_responses}
        scores={scores}
        setOrder={setOrder}
      />,
    );

    expect(screen.getAllByText("Core Value").length).toBe(
      profile.top_values.length,
    );
  });
});
