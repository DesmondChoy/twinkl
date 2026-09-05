import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import CoachDigestCard from "./CoachDigestCard";
import DriftStateExplanation from "./DriftStateExplanation";
import type { OnboardingProfile } from "./domain";
import type {
  JournalEntryContract,
  NudgeInteractionContract,
  ScenarioDeliveryState,
  ScenarioWeekContract,
  TraceEventContract,
  WeeklyDriftReviewerDecisionContract,
} from "./demoContracts";
import { isDisplayableNudge } from "./nudgeReveal";

type JsonObject = Record<string, unknown>;

interface ReplayTimelineProps {
  profile: OnboardingProfile;
  week: ScenarioWeekContract;
  journalEntries: JournalEntryContract[];
  nudges: NudgeInteractionContract[];
  reviewedJournalEntries: JournalEntryContract[];
  weeklyReviewerDecisions: WeeklyDriftReviewerDecisionContract[];
  reviewTraceEvents: TraceEventContract[];
  selectedJournalEntryId: string | null;
  cumulativeEntryCount: number;
  visibleEntryCount: number;
  visibleNudgeEntryIds: ReadonlySet<string>;
  pendingNudgeEntryId: string | null;
  resultVisible: boolean;
  playing: boolean;
  driftResult: JsonObject | null;
  weeklyDigest: JsonObject | null;
  inspectRun: (eventId: string) => void;
  inspectEventId: string | null;
  onSelectJournalEntry: (journalEntryId: string) => void;
}

function object(value: unknown): JsonObject | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function replayStateLabel(state: string): string {
  switch (state) {
    case "active_drift":
      return "Active Drift";
    case "insufficient_evidence":
      return "Insufficient Evidence";
    default:
      return "No Active Drift";
  }
}

function stateExplanation(state: ScenarioDeliveryState): string {
  switch (state) {
    case "active_drift":
      return "A repeated conflict with a Core Value is now active.";
    case "insufficient_evidence":
      return (
        "A review failure prevents a current claim, or an Abstain or gap "
        + "blocks recent Conflict evidence."
      );
    default:
      return "No active Drift is confirmed at this cutoff.";
  }
}

function excerpt(content: string, wordLimit = 20): string {
  const words = content.trim().split(/\s+/);
  return words.length > wordLimit
    ? `${words.slice(0, wordLimit).join(" ")}…`
    : content.trim();
}

function displayEntryDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
  }).format(new Date(`${value}T00:00:00`));
}

function evidenceUsesEarlierWeek(
  weeklyDigest: JsonObject | null,
  weekStart: string,
): boolean {
  const evidence = weeklyDigest?.evidence;
  return Array.isArray(evidence) && evidence.some((item) => {
    const row = object(item);
    return typeof row?.date === "string" && row.date < weekStart;
  });
}

interface JournalEntryDrawerProps {
  entry: JournalEntryContract | null;
  onClose: () => void;
}

function JournalEntryDrawer({
  entry,
  onClose,
}: JournalEntryDrawerProps) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!entry) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus({ preventScroll: true });
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", closeOnEscape);
    };
  }, [entry, onClose]);

  if (!entry) return null;

  return createPortal(
    <div
      className="replay-entry-drawer"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <aside
        className="replay-entry-drawer__panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="replay-entry-drawer-title"
      >
        <header>
          <div>
            <p className="eyebrow">Journal Entry</p>
            <h2 id="replay-entry-drawer-title">
              {displayEntryDate(entry.date)}
            </h2>
          </div>
          <button
            className="replay-entry-drawer__close"
            ref={closeRef}
            type="button"
            aria-label="Close Journal Entry"
            onClick={onClose}
          >
            Close
          </button>
        </header>
        <p className="replay-entry-drawer__content">{entry.content}</p>
      </aside>
    </div>,
    document.body,
  );
}

export default function ReplayTimeline({
  profile,
  week,
  journalEntries,
  nudges,
  reviewedJournalEntries,
  weeklyReviewerDecisions,
  reviewTraceEvents,
  selectedJournalEntryId,
  cumulativeEntryCount,
  visibleEntryCount,
  visibleNudgeEntryIds,
  pendingNudgeEntryId,
  resultVisible,
  playing,
  driftResult,
  weeklyDigest,
  inspectRun,
  inspectEventId,
  onSelectJournalEntry,
}: ReplayTimelineProps) {
  const [openEntry, setOpenEntry] = useState<JournalEntryContract | null>(null);
  const [mobilePanel, setMobilePanel] = useState<"entries" | "result">(
    "entries",
  );
  const visibleEntries = journalEntries.slice(0, visibleEntryCount);
  const nudgeByEntryId = useMemo(
    () => new Map(
      nudges
        .filter(isDisplayableNudge)
        .map((nudge) => [nudge.journal_entry_id, nudge]),
    ),
    [nudges],
  );
  const state =
    (driftResult?.delivery_state as ScenarioDeliveryState | undefined)
    ?? week.expected_delivery_state;
  const hasEarlierEvidence = evidenceUsesEarlierWeek(
    weeklyDigest,
    week.week_start,
  );
  const status = useMemo(() => {
    if (resultVisible) return `${replayStateLabel(state)} revealed.`;
    if (pendingNudgeEntryId) {
      const entryNumber = visibleEntries.findIndex(
        (entry) => entry.journal_entry_id === pendingNudgeEntryId,
      ) + 1;
      return `Journal Entry ${entryNumber} shown. A Nudge follows.`;
    }
    if (visibleEntries.length < journalEntries.length) {
      if (playing) {
        return `Replaying saved Journal Entry ${visibleEntries.length + 1} of ${journalEntries.length}.`;
      }
      return visibleEntries.length === 0
        ? `${journalEntries.length} saved Journal ${
          journalEntries.length === 1 ? "Entry is" : "Entries are"
        } ready.`
        : `Paused after saved Journal Entry ${visibleEntries.length} of ${journalEntries.length}.`;
    }
    return playing
      ? "Journal Entries replayed. Weekly Drift Detection is next."
      : "Paused before Weekly Drift Detection.";
  }, [
    journalEntries.length,
    playing,
    pendingNudgeEntryId,
    resultVisible,
    state,
    visibleEntries.length,
  ]);

  useEffect(() => {
    setOpenEntry(null);
    setMobilePanel("entries");
  }, [week.week_id]);

  useEffect(() => {
    if (resultVisible) setMobilePanel("result");
  }, [resultVisible]);

  const openJournalEntry = (entry: JournalEntryContract) => {
    onSelectJournalEntry(entry.journal_entry_id);
    setOpenEntry(entry);
  };

  return (
    <>
      <section
        className="replay-workspace"
        aria-label={`Week workspace with ${journalEntries.length} Journal Entries`}
      >
        <div className="replay-workspace__mobile-controls">
          <div className="replay-workspace__mobile-result">
            <span>Weekly Drift Detection</span>
            <strong className={`replay-state replay-state--${state}`}>
              {resultVisible ? replayStateLabel(state) : "Not reviewed"}
            </strong>
          </div>
          <div
            className="replay-workspace__switch"
            role="group"
            aria-label="Weekly workspace view"
          >
            <button
              type="button"
              aria-pressed={mobilePanel === "entries"}
              onClick={() => setMobilePanel("entries")}
            >
              Journal Entries
            </button>
            <button
              type="button"
              aria-pressed={mobilePanel === "result"}
              onClick={() => setMobilePanel("result")}
            >
              Weekly Drift
            </button>
          </div>
        </div>

        <section
          className="replay-column replay-column--entries"
          data-mobile-visible={mobilePanel === "entries" ? "true" : "false"}
          aria-labelledby="replay-entries-title"
        >
          <header className="replay-column__header">
            <div>
              <p className="eyebrow">This week</p>
              <h2 id="replay-entries-title">Journal Entries</h2>
            </div>
            <span>
              {visibleEntries.length} of {journalEntries.length}
            </span>
          </header>
          <div
            className="replay-timeline__status"
            role="status"
            aria-live="polite"
          >
            <span aria-hidden="true" />
            {status}
          </div>
          <div className="replay-column__scroll">
            <ol className="replay-entry-list">
              {visibleEntries.map((entry, index) => {
                const nudge = nudgeByEntryId.get(entry.journal_entry_id);
                return (
                  <li
                    className="replay-entry replay-entry--arriving"
                    key={entry.journal_entry_id}
                  >
                    <span className="replay-entry__number" aria-hidden="true">
                      {index + 1}
                    </span>
                    <button
                      id={`replay-entry-button-${entry.journal_entry_id}`}
                      type="button"
                      onClick={() => openJournalEntry(entry)}
                      aria-current={
                        selectedJournalEntryId === entry.journal_entry_id
                          ? "true"
                          : undefined
                      }
                      aria-label={`Open Journal Entry ${index + 1} from ${displayEntryDate(entry.date)}`}
                    >
                      <span className="replay-entry__meta">
                        Journal Entry {index + 1}
                        <time dateTime={entry.date}>{displayEntryDate(entry.date)}</time>
                      </span>
                      <span className="replay-entry__excerpt">
                        {excerpt(entry.content)}
                      </span>
                      <span className="replay-entry__open" aria-hidden="true">
                        Read
                      </span>
                    </button>
                    {nudge?.text
                    && visibleNudgeEntryIds.has(entry.journal_entry_id) ? (
                      <aside
                        className="replay-entry__nudge nudge-reveal"
                        aria-label={`Nudge for Journal Entry ${index + 1}`}
                      >
                        <span className="replay-entry__nudge-label">Nudge</span>
                        <p>{nudge.text}</p>
                        {nudge.outcome === "answered" && nudge.response ? (
                          <div className="replay-entry__response">
                            <span>Response</span>
                            <p>{nudge.response}</p>
                          </div>
                        ) : null}
                        {nudge.outcome === "skipped" ? (
                          <small>Follow-up skipped.</small>
                        ) : null}
                      </aside>
                    ) : null}
                  </li>
                );
              })}
            </ol>

            {journalEntries.length === 0 && resultVisible ? (
              <p className="replay-timeline__empty">No Journal Entries this week.</p>
            ) : null}
          </div>
        </section>

        <aside
          className="replay-column replay-column--result"
          data-mobile-visible={mobilePanel === "result" ? "true" : "false"}
          aria-labelledby="replay-result-column-title"
        >
          <header className="replay-column__header">
            <div>
              <p className="eyebrow">This week</p>
              <h2 id="replay-result-column-title">
                Weekly Drift Detection{" "}
                <span className="replay-column__basis">
                  (based on {cumulativeEntryCount} Journal{" "}
                  {cumulativeEntryCount === 1 ? "Entry" : "Entries"} through{" "}
                  {displayEntryDate(week.week_end)}
                  {hasEarlierEvidence ? "; includes earlier weeks" : ""})
                </span>
              </h2>
            </div>
          </header>
          <div className="replay-column__scroll replay-column__scroll--result">
            {resultVisible ? (
              <article
                className={`replay-result replay-result--${state}`}
                aria-labelledby="replay-result-title"
              >
                <div className="replay-result__body">
                  <header>
                    <h3 id="replay-result-title">{replayStateLabel(state)}</h3>
                  </header>
                  <p>{stateExplanation(state)}</p>
                  {profile.top_values.length > 1 ? (
                    <p>
                      This overall result combines {profile.top_values.length} Core Values.
                      Each has its own state below; they can differ.
                    </p>
                  ) : null}
                  <section
                    className="replay-result__details"
                    aria-labelledby="state-change-title"
                  >
                    <h4 id="state-change-title">Why this state</h4>
                    <DriftStateExplanation
                      profile={profile}
                      journalEntries={reviewedJournalEntries}
                      weeklyReviewerDecisions={weeklyReviewerDecisions}
                      reviewTraceEvents={reviewTraceEvents}
                      weekStart={week.week_start}
                      weekEnd={week.week_end}
                      driftResult={driftResult}
                      onOpenEntry={openJournalEntry}
                    />
                  </section>
                </div>
              </article>
            ) : (
              <div className="replay-result-placeholder" aria-live="polite">
                Weekly Drift Detection appears after the final Journal Entry.
              </div>
            )}
            {resultVisible ? (
              <CoachDigestCard
                weeklyDigest={weeklyDigest}
                headingId="replay-coach-digest-title"
                headingLevel={3}
                className="coach-digest--replay"
              />
            ) : null}
            {resultVisible && inspectEventId ? (
              <button
                className="inspect-run-link replay-column__inspect"
                type="button"
                onClick={() => inspectRun(inspectEventId)}
              >
                Inspect decision
              </button>
            ) : null}
          </div>
        </aside>
      </section>

      <JournalEntryDrawer
        entry={openEntry}
        onClose={() => {
          const closingEntry = openEntry;
          setOpenEntry(null);
          window.requestAnimationFrame?.(() => {
            if (closingEntry) {
              document
                .getElementById(
                  `replay-entry-button-${closingEntry.journal_entry_id}`,
                )
                ?.focus({ preventScroll: true });
            }
          });
        }}
      />
    </>
  );
}
