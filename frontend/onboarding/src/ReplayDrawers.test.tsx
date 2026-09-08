import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import DriftStateExplanation from "./DriftStateExplanation";
import ReplayTimeline from "./ReplayTimeline";
import { canonicalInspectFixture } from "./inspectFixture";
import type { ScenarioWeekContract } from "./demoContracts";

const profile = canonicalInspectFixture.session.profile;
const entry = {
  journal_entry_id: "drawer-entry", t_index: 0, date: "2026-07-01",
  content: "I made time to cook dinner for my sister.", nudge_response: null,
};
const week: ScenarioWeekContract = {
  week_id: "week-1", week_start: "2026-06-29", week_end: "2026-07-05",
  journal_entry_ids: [entry.journal_entry_id], event_ids: [],
  expected_delivery_state: "no_active_drift",
};

function renderDrawer(kind: "Journal Entry" | "AI review") {
  const backgroundAction = vi.fn();
  const rendered = render(
    <>
      <button onClick={backgroundAction}>Inspect this moment</button>
      {kind === "Journal Entry" ? (
        <ReplayTimeline
          profile={profile} week={week} journalEntries={[entry]} nudges={[]}
          reviewedJournalEntries={[entry]} weeklyReviewerDecisions={[]}
          reviewTraceEvents={[]} selectedJournalEntryId={null} cumulativeEntryCount={1}
          resultVisible={false} onRevealResult={() => undefined} driftResult={null}
          weeklyDigest={null} inspectRun={() => undefined} inspectEventId={null}
          onSelectJournalEntry={() => undefined}
        />
      ) : (
        <DriftStateExplanation
          profile={profile} journalEntries={[entry]} reviewTraceEvents={[]}
          weekStart={week.week_start} weekEnd={week.week_end}
          driftResult={{ core_value_states: {} }} onOpenEntry={() => undefined}
          weeklyReviewerDecisions={[{
            persona_id: "drawer-persona", week_start: week.week_start,
            week_end: week.week_end, t_index: entry.t_index, date: entry.date,
            core_value: profile.top_values[0], verdict: "not_conflict", confidence: "high",
            reason_code: "direct_aligned_or_neutral_behavior", evidence_quote: entry.content,
            review_status: "ok",
          }]}
        />
      )}
    </>,
  );
  const trigger = screen.getByRole("button", {
    name: kind === "Journal Entry" ? /^Open Journal Entry/ : "AI review",
  });
  return { ...rendered, trigger, backgroundAction };
}

describe.each(["Journal Entry", "AI review"] as const)("%s modal drawer", (kind) => {
  it("keeps keyboard focus inside and disables the background until Escape restores focus", async () => {
    const user = userEvent.setup();
    const { container, trigger, backgroundAction } = renderDrawer(kind);
    const backgroundButton = screen.getByRole("button", { name: "Inspect this moment" });
    document.body.style.overflow = "auto";

    await user.click(trigger);
    const dialog = screen.getByRole("dialog");
    const close = within(dialog).getByRole("button", { name: /^Close/ });
    expect(document.activeElement).toBe(close);
    expect(container.hasAttribute("inert")).toBe(true);
    expect(document.body.style.overflow).toBe("hidden");

    await user.tab({ shift: true });
    expect(document.activeElement).toBe(close);
    await user.tab();
    expect(document.activeElement).toBe(close);
    backgroundButton.focus();
    expect(document.activeElement).toBe(close);
    expect(backgroundAction).not.toHaveBeenCalled();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() => expect(document.activeElement).toBe(trigger));
    expect(container.hasAttribute("inert")).toBe(false);
    expect(document.body.style.overflow).toBe("auto");
    await user.click(backgroundButton);
    expect(backgroundAction).toHaveBeenCalledOnce();
    document.body.style.overflow = "";
  });

  it.each(["Close", "backdrop"])("restores interaction and trigger focus after %s", async (method) => {
    const user = userEvent.setup();
    const { container, trigger } = renderDrawer(kind);
    await user.click(trigger);
    const dialog = screen.getByRole("dialog");
    if (method === "Close") {
      await user.click(within(dialog).getByRole("button", { name: /^Close/ }));
    } else {
      fireEvent.mouseDown(dialog.parentElement!);
    }
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(container.hasAttribute("inert")).toBe(false);
    await waitFor(() => expect(document.activeElement).toBe(trigger));
  });

  it("cleans up on navigation without enabling a previously inert background", async () => {
    const user = userEvent.setup();
    const { container, trigger, unmount } = renderDrawer(kind);
    const unavailable = document.createElement("div");
    unavailable.setAttribute("inert", "");
    document.body.append(unavailable);
    try {
      await user.click(trigger);
      expect(container.hasAttribute("inert")).toBe(true);
      unmount();
      expect(container.hasAttribute("inert")).toBe(false);
      expect(unavailable.hasAttribute("inert")).toBe(true);
      expect(document.body.style.overflow).toBe("");
    } finally {
      unavailable.remove();
    }
  });
});
