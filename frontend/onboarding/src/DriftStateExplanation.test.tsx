import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import twoValuesReplayJson from "../public/scenarios/two-values-meera.json";
import DriftStateExplanation from "./DriftStateExplanation";
import { validateExperienceInspectFixture } from "./demoContracts";
import { projectScenarioWeek } from "./scenarioReplay";

const fixture = validateExperienceInspectFixture(twoValuesReplayJson);

function propsForWeek(weekIndex: number) {
  const projection = projectScenarioWeek(fixture, weekIndex);
  const week = fixture.scenario.weeks[weekIndex];
  return {
    profile: { ...fixture.scenario.profile, top_values: ["self_direction" as const] },
    journalEntries: projection.session.journal_entries,
    weeklyReviewerDecisions: projection.session.weekly_reviewer_decisions,
    reviewTraceEvents: projection.events,
    weekStart: week.week_start,
    weekEnd: week.week_end,
    driftResult: projection.session.drift_result,
    onOpenEntry: vi.fn(),
  };
}

describe("ended Drift explanation", () => {
  it("names the ending Not Conflict decision and opens its source without duplicating it", () => {
    const props = propsForWeek(1);
    render(<DriftStateExplanation {...props} />);

    expect(screen.getByText("Drift ended here.")).toBeTruthy();
    expect(screen.getByText(/Weekly Drift Reviewer's Not Conflict decision/).textContent)
      .toContain("ended the earlier Conflict run");
    const source = screen.getByRole("button", {
      name: /Read Journal Entry from (?=.*\b20\b)(?=.*Nov)(?=.*2025)/,
    });
    expect(within(source).getByText("Not Conflict")).toBeTruthy();
    fireEvent.click(source);
    expect(props.onOpenEntry).toHaveBeenCalledWith(
      props.journalEntries.find((entry) => entry.t_index === 2),
    );
    expect(screen.queryByText(/This does not prove a positive change/)).toBeNull();
  });

  it("distinguishes an earlier ending from the selected week's decisions", () => {
    render(<DriftStateExplanation {...propsForWeek(2)} />);

    expect(screen.getByText("Earlier Drift ended here.")).toBeTruthy();
    expect(screen.queryByText("Drift ended here.")).toBeNull();
    expect(screen.getByText("This week's decisions.")).toBeTruthy();
    expect(screen.getByRole("button", {
      name: /Read Journal Entry from (?=.*\b20\b)(?=.*Nov)(?=.*2025)/,
    })).toBeTruthy();
    expect(screen.getByRole("button", {
      name: /Read Journal Entry from (?=.*\b24\b)(?=.*Nov)(?=.*2025)/,
    })).toBeTruthy();
  });

  it("does not expose a later ending while Drift is still active", () => {
    render(<DriftStateExplanation {...propsForWeek(0)} />);

    expect(screen.queryByText("Active Drift")).toBeNull();
    expect(screen.getByText("Drift started here.")).toBeTruthy();
    expect(screen.queryByText(/Drift ended here/)).toBeNull();
    expect(screen.queryByRole("button", {
      name: /Read Journal Entry from (?=.*\b20\b)(?=.*Nov)(?=.*2025)/,
    })).toBeNull();
  });

  it.each(["abstain", "failed_review", "missing_decision"])(
    "does not claim a Not Conflict ending from %s evidence",
    (evidence) => {
      const props = propsForWeek(1);
      props.weeklyReviewerDecisions = props.weeklyReviewerDecisions.flatMap((decision) => {
        if (decision.t_index !== 2) return [decision];
        if (evidence === "missing_decision") return [];
        return [{
          ...decision,
          verdict: evidence === "abstain" ? "abstain" as const : decision.verdict,
          review_status: evidence === "failed_review" ? "error" as const : decision.review_status,
        }];
      });
      render(<DriftStateExplanation {...props} />);

      expect(screen.queryByText(/Drift ended here/)).toBeNull();
      expect(screen.queryByText(/ended the earlier Conflict run/)).toBeNull();
    },
  );
});
