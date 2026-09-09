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
import useModalFocus from "./useModalFocus";

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
  resultVisible: boolean;
  onRevealResult: () => void;
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

function excerpt(content: string, wordLimit = 50): string {
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

interface JournalEntryDialogProps {
  entry: JournalEntryContract | null;
  onClose: () => void;
}

function JournalEntryDialog({
  entry,
  onClose,
}: JournalEntryDialogProps) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  useModalFocus(entry !== null, overlayRef, closeRef, onClose);

  if (!entry) return null;

  return createPortal(
    <div
      ref={overlayRef}
      className="replay-entry-drawer"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
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
        {entry.nudge_response ? (
          <section className="replay-entry-drawer__response" aria-labelledby="replay-entry-response-title">
            <h3 id="replay-entry-response-title">Response to the Nudge</h3>
            <p className="replay-entry-drawer__content">{entry.nudge_response}</p>
          </section>
        ) : null}
      </section>
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
  resultVisible,
  onRevealResult,
  driftResult,
  weeklyDigest,
  inspectRun,
  inspectEventId,
  onSelectJournalEntry,
}: ReplayTimelineProps) {
  const [openEntry, setOpenEntry] = useState<JournalEntryContract | null>(null);
  const openEntryTriggerRef = useRef<HTMLElement | null>(null);
  const [panel, setPanel] = useState<"entries" | "result">(
    resultVisible ? "result" : "entries",
  );
  const resultHeadingRef = useRef<HTMLHeadingElement>(null);
  const entriesHeadingRef = useRef<HTMLHeadingElement>(null);
  const entriesScrollRef = useRef<HTMLDivElement>(null);
  const coachNarrative = object(weeklyDigest?.coach_narrative);
  const hasCoach = ["weekly_mirror", "tension_explanation", "reflective_question"]
    .every((key) =>
      typeof coachNarrative?.[key] === "string"
      && coachNarrative[key].trim().length > 0,
    );
  const showingResult = resultVisible && panel === "result";
  const visibleEntries = journalEntries;
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
  const digestEvent = reviewTraceEvents.find((event) =>
    week.event_ids.includes(event.event_id)
    && event.event_type === "weekly_digest_built"
  );
  const coachUnavailableReason = digestEvent?.details.coach_unavailable_reason;
  const status = resultVisible
    ? `${replayStateLabel(state)} reviewed.`
    : "Read the Journal Entries, then review Weekly Drift Detection.";

  useEffect(() => {
    setOpenEntry(null);
    entriesScrollRef.current?.scrollTo?.({ top: 0 });
  }, [week.week_id]);

  useEffect(() => {
    setPanel(resultVisible ? "result" : "entries");
  }, [week.week_id, resultVisible]);

  useEffect(() => {
    if (showingResult) {
      resultHeadingRef.current?.focus({ preventScroll: true });
    }
  }, [showingResult, week.week_id]);

  const openJournalEntry = (entry: JournalEntryContract) => {
    openEntryTriggerRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement : null;
    onSelectJournalEntry(entry.journal_entry_id);
    setOpenEntry(entry);
  };

  return (
    <>
      <section
        id="replay-workspace"
        className={`replay-workspace${showingResult ? " replay-workspace--result" : ""}`}
        aria-label={`Week workspace with ${journalEntries.length} Journal Entries`}
      >
        <section
          id="replay-journals"
          className="replay-column replay-column--entries"
          hidden={showingResult}
          aria-labelledby="replay-entries-title"
        >
          <header className="replay-column__header">
            <div>
              <h2 id="replay-entries-title" ref={entriesHeadingRef} tabIndex={-1}>
                Journal Entries
              </h2>
            </div>
            <div className="replay-column__actions">
              <span className="replay-column__count">
                {journalEntries.length} {journalEntries.length === 1 ? "entry" : "entries"}
              </span>
              <button
                className="button button--primary"
                type="button"
                aria-controls="replay-weekly-result"
                onClick={() => {
                  setPanel("result");
                  if (!resultVisible) onRevealResult();
                }}
              >
                {resultVisible ? "Read Weekly Drift Detection" : "Review Weekly Drift Detection"}
              </button>
            </div>
          </header>
          <div
            className="replay-timeline__status"
            role="status"
            aria-live="polite"
          >
            <span aria-hidden="true" />
            {status}
          </div>
          <div className="replay-column__scroll" ref={entriesScrollRef}>
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
                      aria-haspopup="dialog"
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
                        Read entry
                      </span>
                    </button>
                    {nudge?.text ? (
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

            {journalEntries.length === 0 ? (
              <p className="replay-timeline__empty">No Journal Entries this week.</p>
            ) : null}
          </div>
        </section>

        <aside
          id="replay-weekly-result"
          className="replay-column replay-column--result"
          hidden={!showingResult}
          aria-labelledby="replay-result-column-title"
        >
          <header className="replay-column__header">
            <div>
              <h2 id="replay-result-column-title" ref={resultHeadingRef} tabIndex={-1}>
                Drift Detection (End of Week){" "}
                <span className="replay-column__basis">
                  (based on {cumulativeEntryCount} Journal{" "}
                  {cumulativeEntryCount === 1 ? "Entry" : "Entries"} through{" "}
                  {displayEntryDate(week.week_end)}
                  {hasEarlierEvidence ? "; includes earlier weeks" : ""})
                </span>
              </h2>
            </div>
            <button
              className="button button--quiet"
              type="button"
              aria-controls="replay-journals"
              onClick={() => {
                setPanel("entries");
                window.requestAnimationFrame?.(() =>
                  entriesHeadingRef.current?.focus({ preventScroll: true }),
                );
              }}
            >
              Read Journal Entries
            </button>
          </header>
          <div className="replay-result-columns">
            <section className="replay-result-column replay-result-column--state" aria-label="Drift state">
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
                    <details
                      className="replay-result__details"
                      key={week.week_id}
                    >
                      <summary>Why this state</summary>
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
                    </details>
                  </div>
                </article>
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
            </section>
            <section className="replay-result-column replay-result-column--coach" aria-label="Coach Digest">
              {resultVisible ? (
                <CoachDigestCard
                  weeklyDigest={weeklyDigest}
                  headingId="replay-coach-digest-title"
                  headingLevel={3}
                  className="coach-digest--replay"
                  journalEntries={reviewedJournalEntries}
                  onOpenEntry={openJournalEntry}
                  northStar={driftResult ? {
                    profile,
                    driftResult,
                    traceEvents: reviewTraceEvents,
                    presentation: "demo",
                    inspectMoment: inspectRun,
                  } : undefined}
                />
              ) : null}
              {resultVisible && !hasCoach ? (
                <aside className="coach-digest coach-digest--replay" aria-labelledby="replay-coach-unavailable-title">
                  <p className="eyebrow">Coach Digest</p>
                  <h3 id="replay-coach-unavailable-title">No saved Coach Digest for this result</h3>
                  <p>
                    {typeof coachUnavailableReason === "string" && coachUnavailableReason.startsWith("Historical Coach Digest omitted:")
                      ? "The earlier Coach Digest was based on different Weekly Drift Detection results and is omitted from this replay."
                      : "A Coach Digest response has not been saved for this replay week."}
                  </p>
                </aside>
              ) : null}
            </section>
          </div>
        </aside>
      </section>

      <JournalEntryDialog
        entry={openEntry}
        onClose={() => {
          const closingEntry = openEntry;
          setOpenEntry(null);
          window.requestAnimationFrame?.(() => {
            if (openEntryTriggerRef.current?.isConnected) {
              openEntryTriggerRef.current.focus({ preventScroll: true });
              return;
            }
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
