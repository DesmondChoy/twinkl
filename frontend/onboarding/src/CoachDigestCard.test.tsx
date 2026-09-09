import { render, screen, cleanup, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import activeReplay from "../public/scenarios/active-nisha.json";
import persistentReplay from "../public/scenarios/persistent-lukas.json";
import stableReplay from "../public/scenarios/stable-noor.json";
import twoValuesReplay from "../public/scenarios/two-values-meera.json";
import uncertainReplay from "../public/scenarios/uncertain-wei-jun.json";
import catalog from "../public/scenarios/index.json";
import CoachDigestCard from "./CoachDigestCard";
import { expandCoachQuotations } from "./coachQuotes";
import { validateExperienceInspectFixture } from "./demoContracts";
import { projectScenarioWeek } from "./scenarioReplay";
import type { NorthStarRecord } from "./northStar";
import { savedCoachComparison } from "./coachComparison";

afterEach(cleanup);

describe("Coach Digest quotations", () => {
  it("restores exact source text through the sentence boundary", () => {
    expect(expandCoachQuotations('You said “five more...”', [
      "But I thought, five more minutes won't matter, and I stood in line anyway. The laksa was good.",
    ]).text).toBe('You said “five more minutes won\'t matter, and I stood in line anyway.”');
    expect(expandCoachQuotations('You wrote “five more...”.', [
      "five more minutes would be fine.",
    ]).text).toBe('You wrote “five more minutes would be fine.”');
  });

  it("leaves ambiguous, missing, and author-written ellipses intact", () => {
    const narrative = 'You said “five more...”';
    expect(expandCoachQuotations(narrative, ["five more minutes.", "five more days."]).text)
      .toBe(narrative);
    expect(expandCoachQuotations(narrative, ["An unrelated Journal Entry."]).text)
      .toBe(narrative);
    expect(expandCoachQuotations(narrative, ["I thought five more... then stopped."]).text)
      .toBe(narrative);
    expect(expandCoachQuotations('“he went...”', ["She went to lunch."]).text)
      .toBe('“he went...”');
    expect(expandCoachQuotations('“we take...”', ["we takeover the shop."]).text)
      .toBe('“we take...”');
    expect(expandCoachQuotations(narrative, ["I thought five more ... then stopped."]).text)
      .toBe(narrative);
  });

  it("keeps a quoted fragment within continuing narration and shows its full source separately", () => {
    const source = "Afterward, we didn't talk about anything real.";
    const display = expandCoachQuotations(
      'You said “we didn\'t talk about...” what might have mattered.', [source],
    );
    expect(display.text).toBe('You said “we didn\'t talk about” what might have mattered.');
    expect(display.fullQuotations).toEqual([{
      quotation: "we didn't talk about anything real.", source,
    }]);
  });

  it("requires linked evidence before the cutoff and keeps saved output intact", async () => {
    const entry = {
      journal_entry_id: "entry-1", t_index: 1, date: "2025-02-11",
      content: "I thought five more minutes would be fine.", nudge_response: null,
    };
    const digest = {
      week_end: "2025-02-10",
      evidence: [{ date: entry.date, t_index: 1, excerpt: "I thought five more..." }],
      coach_narrative: {
        weekly_mirror: 'You said “five more...”',
        tension_explanation: "A moment to enjoy lunch.",
        reflective_question: "What made this possible?",
      },
    };
    const original = JSON.stringify(digest);
    const onOpenEntry = vi.fn();
    const { rerender } = render(<CoachDigestCard weeklyDigest={digest}
      headingId="coach-title" journalEntries={[entry]} onOpenEntry={onOpenEntry} />);
    expect(screen.getByText('You said “five more...”')).toBeTruthy();
    expect(screen.queryByRole("navigation")).toBeNull();

    rerender(<CoachDigestCard weeklyDigest={{ ...digest, week_end: entry.date }}
      headingId="coach-title" journalEntries={[entry]} onOpenEntry={onOpenEntry} />);
    expect(screen.getByRole("heading", { name: "Your weekly reflection" })).toBeTruthy();
    expect(screen.getByText('You said “five more minutes would be fine.”')).toBeTruthy();
    await userEvent.click(screen.getByRole("link"));
    expect(onOpenEntry).toHaveBeenCalledWith(entry);
    expect(JSON.stringify(digest)).toBe(original);

    rerender(<CoachDigestCard weeklyDigest={{ ...digest, week_end: entry.date }}
      headingId="coach-title" journalEntries={[{ ...entry, content: "A changed entry.",
        nudge_response: entry.content }]} />);
    expect(screen.getByText('You said “five more...”')).toBeTruthy();

    rerender(<CoachDigestCard weeklyDigest={{ ...digest, week_end: entry.date, evidence: [] }}
      headingId="coach-title" journalEntries={[entry]} />);
    expect(screen.getByText('You said “five more...”')).toBeTruthy();
  });

  it.each([activeReplay, persistentReplay, stableReplay, twoValuesReplay, uncertainReplay])(
    "shows complete quotations for each current catalog key week ($scenario.scenario_id)",
    (raw) => {
      const fixture = validateExperienceInspectFixture(raw);
      const item = catalog.scenarios.find((row) => row.scenario_id === fixture.scenario.scenario_id)!;
      const weekIndex = fixture.scenario.weeks.findIndex((week) =>
        week.week_start === item.key_week_start);
      const { session } = projectScenarioWeek(fixture, weekIndex);
      const original = JSON.stringify(session.weekly_digest);
      const { container } = render(<CoachDigestCard weeklyDigest={session.weekly_digest}
        headingId="coach-title" journalEntries={session.journal_entries} />);
      expect(screen.getByRole("heading", { name: "Your weekly reflection" })).toBeTruthy();
      expect(container.textContent).not.toMatch(/(?:\.{3}|…)[”"]/);
      expect(JSON.stringify(session.weekly_digest)).toBe(original);
      expect(screen.getByText("Something to reflect on")).toBeTruthy();
      expect(screen.queryByText("Read the supporting Journal Entries")).toBeNull();
    },
  );

  it.each([activeReplay, persistentReplay, stableReplay, twoValuesReplay, uncertainReplay])(
    "offers exactly the frozen selected weeks across $scenario.scenario_id and restores the baseline on each click",
    async (raw) => {
      const fixture = validateExperienceInspectFixture(raw);
      for (let index = 0; index < fixture.scenario.weeks.length; index += 1) {
        const { session } = projectScenarioWeek(fixture, index);
        const week = fixture.scenario.weeks[index];
        const events = fixture.trace_events.filter((event) => week.event_ids.includes(event.event_id));
        const moment = events.find((event) => event.event_type === "north_star_reviewed")!;
        const record = moment.details.record as NorthStarRecord;
        const coach = events.find((event) => event.event_type === "weekly_coach_generated")!;
        const pair = savedCoachComparison(coach);
        const view = render(<CoachDigestCard weeklyDigest={session.weekly_digest}
          headingId="coach-title" journalEntries={session.journal_entries}
          northStar={{ profile: session.profile, driftResult: session.drift_result!, traceEvents: events, presentation: "demo" }} />);
        const toggle = screen.getByRole("button", { name: "With North Star Moment" });
        expect(document.querySelector(".north-star-moment")).toBeNull();
        if (record.selected) {
          expect(pair, `${fixture.scenario.scenario_id} ${week.week_start}`).not.toBeNull();
          await waitFor(() => expect(toggle.hasAttribute("disabled")).toBe(false));
          const baseline = document.querySelector(".coach-digest__question")!.textContent;
          await userEvent.click(toggle);
          await waitFor(() => expect(document.querySelector(".north-star-moment blockquote")!.textContent)
            .toBe(record.selected!.evidence_quote));
          expect(document.querySelector(".coach-digest__question")!.textContent)
            .toBe(pair!.with_north_star.narrative.reflective_question);
          await userEvent.click(screen.getByRole("button", { name: "Without North Star Moment" }));
          expect(document.querySelector(".coach-digest__question")!.textContent).toBe(baseline);
          expect(document.querySelector(".north-star-moment")).toBeNull();
        } else {
          expect(pair).toBeNull();
          expect(toggle.hasAttribute("disabled")).toBe(true);
          expect(screen.getByText("No North Star Moment for this week.")).toBeTruthy();
        }
        view.unmount();
      }
    },
  );
});
