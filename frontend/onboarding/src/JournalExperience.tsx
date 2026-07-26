import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import CoreValueReminder from "./CoreValueReminder";
import type { OnboardingProfile } from "./domain";
import type {
  JournalEntryContract,
  NudgeInteractionContract,
  TraceEventContract,
} from "./demoContracts";
import {
  createExperienceSession,
  ExperienceApiError,
  journalIdempotencyKey,
  readExperienceTrace,
  submitJournalEntry,
} from "./experienceApi";
import { journalEntryAnchorId } from "./journalEntryAnchor";
import WeeklyExperience from "./WeeklyExperience";
import type {
  ExperienceState,
  PendingJournalSubmission,
} from "./session";

interface JournalExperienceProps {
  profile: OnboardingProfile;
  experience: ExperienceState;
  updateExperience: (patch: Partial<ExperienceState>) => void;
  inspectRun: (eventId: string) => void;
  headingRef?: RefObject<HTMLHeadingElement | null>;
  mode?: "manual" | "saved_replay";
}

const UNSAVED_JOURNAL_ERROR =
  "Your Journal Entry is still in this editor, but the Experience service could not process it.";
const INSPECT_TRACE_ERROR =
  "Your Journal Entry is saved. Its Inspect trace could not be loaded.";

function localDate(): string {
  const now = new Date();
  const month = String(now.getMonth() + 1).padStart(2, "0");
  const day = String(now.getDate()).padStart(2, "0");
  return `${now.getFullYear()}-${month}-${day}`;
}

function displayDate(value: string): string {
  const parsed = new Date(`${value}T00:00:00`);
  return new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(parsed);
}

function latestDecisionFor(
  events: TraceEventContract[],
  journalEntryId: string,
): TraceEventContract | null {
  return (
    [...events].reverse().find((event) => {
      if (event.event_type !== "nudge_decided") return false;
      return event.input_refs.some((reference) => {
        if (reference === null || typeof reference !== "object") return false;
        return (reference as Record<string, unknown>).id === journalEntryId;
      });
    }) ?? null
  );
}

function mergeJournalEntries(
  remote: JournalEntryContract[],
  local: JournalEntryContract[],
): JournalEntryContract[] {
  const localById = new Map(
    local.map((entry) => [entry.journal_entry_id, entry]),
  );
  return remote.map((entry) => ({
    ...entry,
    nudge_response:
      localById.get(entry.journal_entry_id)?.nudge_response ??
      entry.nudge_response,
  }));
}

function mergeNudges(
  remote: NudgeInteractionContract[],
  local: NudgeInteractionContract[],
): NudgeInteractionContract[] {
  const localById = new Map(local.map((nudge) => [nudge.nudge_id, nudge]));
  return remote.map((nudge) => {
    const previous = localById.get(nudge.nudge_id);
    return previous?.outcome === "answered" || previous?.outcome === "skipped"
      ? {
          ...nudge,
          outcome: previous.outcome,
          response: previous.response,
        }
      : nudge;
  });
}

function nextJournalTIndex(
  journalEntries: JournalEntryContract[],
  traceEvents: TraceEventContract[],
): number {
  const indices = journalEntries.map((entry) => entry.t_index);
  traceEvents.forEach((event) => {
    if (event.event_type !== "journal_entry_submitted") return;
    const candidate = event.details.journal_entry;
    if (candidate === null || typeof candidate !== "object") return;
    const tIndex = (candidate as Record<string, unknown>).t_index;
    if (typeof tIndex === "number" && Number.isInteger(tIndex)) {
      indices.push(tIndex);
    }
  });
  return Math.max(-1, ...indices) + 1;
}

function statusCopy(experience: ExperienceState): string | null {
  switch (experience.run_state) {
    case "saving":
      return "Saving your Journal Entry…";
    case "checking_nudge":
      return "Saved. Looking for one useful follow-up…";
    case "running":
      return "Updating your week…";
    case "awaiting_response":
      return experience.error_message ?? "A follow-up question is ready.";
    case "complete":
      return "Saved.";
    case "refused":
    case "invalid":
      return "Saved. No follow-up question this time.";
    case "failed":
      return experience.error_message ?? UNSAVED_JOURNAL_ERROR;
    default:
      return null;
  }
}

export default function JournalExperience({
  profile,
  experience,
  updateExperience,
  inspectRun,
  headingRef,
  mode = "manual",
}: JournalExperienceProps) {
  const submissionLockRef = useRef(false);
  const nudgeHeadingRef = useRef<HTMLHeadingElement>(null);
  const threadHeadingRef = useRef<HTMLHeadingElement>(null);
  const [removingEntryId, setRemovingEntryId] = useState<string | null>(null);
  const activeNudge = useMemo(
    () =>
      [...experience.nudges]
        .reverse()
        .find((nudge) => nudge.outcome === "displayed") ?? null,
    [experience.nudges],
  );
  const isBusy = ["saving", "checking_nudge", "running"].includes(
    experience.run_state,
  );
  const isAwaitingResponse =
    experience.run_state === "awaiting_response" && activeNudge !== null;
  const pendingJournalEntrySaved =
    experience.pending_submission !== null
    && experience.journal_entries.some(
      (entry) =>
        entry.journal_entry_id
        === experience.pending_submission?.entry.journal_entry_id,
    );
  const copy = statusCopy(experience);

  useEffect(() => {
    if (isAwaitingResponse) {
      nudgeHeadingRef.current?.focus({ preventScroll: true });
    }
  }, [isAwaitingResponse]);

  const runSubmission = async (pending: PendingJournalSubmission) => {
    const traceEventIds = new Set(
      experience.trace_events.map((event) => event.event_id),
    );
    const hasCompleteTrace =
      experience.trace_event_ids.length === experience.trace_events.length
      && experience.trace_event_ids.every((eventId) =>
        traceEventIds.has(eventId));
    const canResumeSession =
      experience.revision > 0
      && experience.trace_events.length > 0
      && hasCompleteTrace;
    let acceptedResponse: Awaited<
      ReturnType<typeof submitJournalEntry>
    > | null = null;
    updateExperience({
      run_state: "saving",
      retryable: false,
      error_message: null,
    });
    try {
      await createExperienceSession(
        profile,
        canResumeSession
          ? {
              session_id: profile.session_id,
              revision: experience.revision,
              journal_entries: experience.journal_entries,
              nudges: experience.nudges,
              trace_events: experience.trace_events,
            }
          : null,
      );
      updateExperience({ run_state: "checking_nudge" });
      acceptedResponse = await submitJournalEntry({
        sessionId: profile.session_id,
        expectedRevision: pending.expected_revision,
        entry: pending.entry,
        idempotencyKey: pending.idempotency_key,
      });
      const trace = await readExperienceTrace(profile.session_id);
      const nudge =
        acceptedResponse.session.nudges.find(
          (item) => item.journal_entry_id === pending.entry.journal_entry_id,
        ) ?? null;
      const decision = latestDecisionFor(
        trace.events,
        pending.entry.journal_entry_id,
      );
      const failed = decision?.status === "failed";
      const runState =
        nudge?.outcome === "displayed"
          ? "awaiting_response"
          : failed
            ? "failed"
            : decision?.status === "refused"
              ? "refused"
              : decision?.status === "invalid"
                ? "invalid"
                : "complete";

      updateExperience({
        revision: acceptedResponse.session.revision,
        journal_draft: "",
        journal_entries: mergeJournalEntries(
          acceptedResponse.session.journal_entries,
          experience.journal_entries,
        ),
        nudges: mergeNudges(
          acceptedResponse.session.nudges,
          experience.nudges,
        ),
        weekly_reviewer_decisions:
          acceptedResponse.session.weekly_reviewer_decisions,
        drift_result: acceptedResponse.session.drift_result,
        weekly_digest: acceptedResponse.session.weekly_digest,
        pending_submission: failed ? pending : null,
        run_state: runState,
        retryable: failed,
        error_message: failed
          ? "Your Journal Entry is saved. The follow-up check could not finish."
          : null,
        trace_event_ids: acceptedResponse.session.trace_event_ids,
        trace_events: trace.events,
      });
    } catch (error) {
      const apiError =
        error instanceof ExperienceApiError
          ? error
          : new ExperienceApiError(
              "The Experience service could not process this request.",
            );
      if (acceptedResponse) {
        updateExperience({
          revision: acceptedResponse.session.revision,
          journal_draft: "",
          journal_entries: mergeJournalEntries(
            acceptedResponse.session.journal_entries,
            experience.journal_entries,
          ),
          nudges: mergeNudges(
            acceptedResponse.session.nudges,
            experience.nudges,
          ),
          weekly_reviewer_decisions:
            acceptedResponse.session.weekly_reviewer_decisions,
          drift_result: acceptedResponse.session.drift_result,
          weekly_digest: acceptedResponse.session.weekly_digest,
          pending_submission: apiError.retryable ? pending : null,
          run_state: "failed",
          retryable: apiError.retryable,
          error_message:
            INSPECT_TRACE_ERROR,
          trace_event_ids: acceptedResponse.session.trace_event_ids,
        });
        return;
      }
      updateExperience({
        run_state: "failed",
        retryable: apiError.retryable,
        error_message: UNSAVED_JOURNAL_ERROR,
        pending_submission: pending,
      });
    }
  };

  const saveJournalEntry = async () => {
    const content = experience.journal_draft.trim();
    if (
      !content ||
      isBusy ||
      isAwaitingResponse ||
      experience.pending_submission !== null ||
      submissionLockRef.current
    ) {
      return;
    }
    submissionLockRef.current = true;
    try {
      const entry: JournalEntryContract = {
        journal_entry_id: crypto.randomUUID(),
        t_index: nextJournalTIndex(
          experience.journal_entries,
          experience.trace_events,
        ),
        date: localDate(),
        content,
        nudge_response: null,
      };
      const pending: PendingJournalSubmission = {
        entry,
        expected_revision: experience.revision,
        idempotency_key: await journalIdempotencyKey(
          profile.session_id,
          entry,
        ),
      };
      updateExperience({ pending_submission: pending });
      await runSubmission(pending);
    } finally {
      submissionLockRef.current = false;
    }
  };

  const retrySubmission = async () => {
    if (!experience.pending_submission || submissionLockRef.current) return;
    submissionLockRef.current = true;
    try {
      await runSubmission(experience.pending_submission);
    } finally {
      submissionLockRef.current = false;
    }
  };

  const editPendingSubmission = () => {
    updateExperience({
      pending_submission: null,
      run_state: "idle",
      retryable: false,
      error_message: null,
    });
  };

  const synchronizeBrowserState = async (
    journalEntries: JournalEntryContract[],
    nudges: NudgeInteractionContract[],
  ) => {
    const currentState = {
      session_id: profile.session_id,
      revision: experience.revision,
      journal_entries: experience.journal_entries,
      nudges: experience.nudges,
      trace_events: experience.trace_events,
    };
    const updatedState = {
      session_id: profile.session_id,
      revision: experience.revision + 1,
      journal_entries: journalEntries,
      nudges,
      trace_events: experience.trace_events,
    };
    let restoreError: unknown = null;
    try {
      await createExperienceSession(profile, currentState);
    } catch (error) {
      restoreError = error;
    }
    let response;
    try {
      response = await createExperienceSession(profile, updatedState);
    } catch (error) {
      throw restoreError ?? error;
    }
    if (
      response.session.revision !== updatedState.revision
      || JSON.stringify(response.session.journal_entries)
        !== JSON.stringify(updatedState.journal_entries)
      || JSON.stringify(response.session.nudges)
        !== JSON.stringify(updatedState.nudges)
    ) {
      throw new ExperienceApiError(
        "The Experience update did not match the requested Journal Entry state.",
        "session_conflict",
        false,
      );
    }
    const trace = await readExperienceTrace(profile.session_id);
    return { response, trace };
  };

  const finishNudge = async (outcome: "skipped" | "answered") => {
    if (!activeNudge) return;
    const response =
      outcome === "answered"
        ? experience.nudge_response_draft.trim()
        : null;
    if (outcome === "answered" && !response) return;
    const nudges = experience.nudges.map((nudge) =>
      nudge.nudge_id === activeNudge.nudge_id
        ? { ...nudge, outcome, response }
        : nudge,
    );
    const entries = experience.journal_entries.map((entry) =>
      entry.journal_entry_id === activeNudge.journal_entry_id
        ? { ...entry, nudge_response: response }
        : entry,
    );
    updateExperience({ run_state: "running", error_message: null });
    try {
      const synchronized = await synchronizeBrowserState(entries, nudges);
      updateExperience({
        revision: synchronized.response.session.revision,
        journal_entries: synchronized.response.session.journal_entries,
        nudges: synchronized.response.session.nudges,
        weekly_reviewer_decisions:
          synchronized.response.session.weekly_reviewer_decisions,
        drift_result: synchronized.response.session.drift_result,
        weekly_digest: synchronized.response.session.weekly_digest,
        nudge_response_draft: "",
        run_state: "complete",
        retryable: false,
        error_message: null,
        trace_event_ids: synchronized.response.session.trace_event_ids,
        trace_events: synchronized.trace.events,
      });
    } catch {
      updateExperience({
        run_state: "awaiting_response",
        retryable: false,
        error_message:
          "Your response is still here, but this week could not update. Try again.",
      });
    }
  };

  const removeJournalEntry = async (journalEntryId: string) => {
    if (
      isBusy ||
      isAwaitingResponse ||
      removingEntryId !== null ||
      !window.confirm(
        "Remove this Journal Entry and update the affected weekly results?",
      )
    ) {
      return;
    }
    const journalEntries = experience.journal_entries.filter(
      (entry) => entry.journal_entry_id !== journalEntryId,
    );
    const nudges = experience.nudges.filter(
      (nudge) => nudge.journal_entry_id !== journalEntryId,
    );
    setRemovingEntryId(journalEntryId);
    updateExperience({ run_state: "running", error_message: null });
    try {
      const synchronized = await synchronizeBrowserState(journalEntries, nudges);
      updateExperience({
        revision: synchronized.response.session.revision,
        journal_entries: synchronized.response.session.journal_entries,
        nudges: synchronized.response.session.nudges,
        weekly_reviewer_decisions:
          synchronized.response.session.weekly_reviewer_decisions,
        drift_result: synchronized.response.session.drift_result,
        weekly_digest: synchronized.response.session.weekly_digest,
        selected_entry_id:
          experience.selected_entry_id === journalEntryId
            ? null
            : experience.selected_entry_id,
        run_state: "complete",
        retryable: false,
        error_message: null,
        trace_event_ids: synchronized.response.session.trace_event_ids,
        trace_events: synchronized.trace.events,
      });
      threadHeadingRef.current?.focus({ preventScroll: true });
    } catch {
      updateExperience({
        run_state: "failed",
        retryable: false,
        error_message:
          "That Journal Entry is still here. Removing it could not finish.",
      });
    } finally {
      setRemovingEntryId(null);
    }
  };

  const inspectLatest = () => {
    const eventId = experience.trace_event_ids.at(-1);
    if (eventId) inspectRun(eventId);
  };

  return (
    <div className="journal-experience">
      {mode === "manual" ? <header className="journal-experience__header">
        <p className="eyebrow">
          {experience.journal_entries.length === 0
            ? "First Journal Entry"
            : "Journal Entries"}
        </p>
        <h1 ref={headingRef} tabIndex={-1}>
          {experience.journal_entries.length === 0
            ? "When did you feel most like yourself?"
            : "What has been taking up space today?"}
        </h1>
        <p className="lede" id="journal-entry-help">
          Write one real moment. Twinkl may ask one brief question if there is
          a useful thread to follow.
        </p>
        <CoreValueReminder profile={profile} />
      </header> : null}

      {mode === "manual" && !isAwaitingResponse ? (
        <form
          className="journal-composer"
          onSubmit={(event) => {
            event.preventDefault();
            void saveJournalEntry();
          }}
        >
          <label htmlFor="journal-entry">
            {experience.journal_entries.length === 0
              ? "First Journal Entry"
              : "Journal Entry"}
          </label>
          <textarea
            id="journal-entry"
            aria-describedby="journal-entry-help journal-status"
            placeholder="Start with the moment…"
            value={experience.journal_draft}
            disabled={isBusy || experience.pending_submission !== null}
            onChange={(event) =>
              updateExperience({ journal_draft: event.target.value })
            }
          />
          <div className="journal-composer__actions">
            <span>
              {experience.journal_draft.trim().length}{" "}
              {experience.journal_draft.trim().length === 1
                ? "character"
                : "characters"}
            </span>
            <button
              className="button button--primary"
              type="submit"
              disabled={
                !experience.journal_draft.trim() ||
                isBusy ||
                experience.pending_submission !== null
              }
            >
              {isBusy ? "Saving…" : "Save Journal Entry"}
            </button>
          </div>
        </form>
      ) : null}

      {mode === "manual" && activeNudge && isAwaitingResponse ? (
        <section className="nudge-reply" aria-labelledby="nudge-question">
          <p className="eyebrow">One question</p>
          <h2 id="nudge-question" ref={nudgeHeadingRef} tabIndex={-1}>
            {activeNudge.text}
          </h2>
          <label htmlFor="nudge-response">Your response</label>
          <textarea
            id="nudge-response"
            value={experience.nudge_response_draft}
            placeholder="Answer in your own words…"
            onChange={(event) =>
              updateExperience({ nudge_response_draft: event.target.value })
            }
          />
          <div className="nudge-reply__actions">
            <button
              className="button button--quiet"
              type="button"
              onClick={() => void finishNudge("skipped")}
            >
              Skip for now
            </button>
            <button
              className="button button--primary"
              type="button"
              disabled={!experience.nudge_response_draft.trim()}
              onClick={() => void finishNudge("answered")}
            >
              Save response
            </button>
          </div>
        </section>
      ) : null}

      {mode === "manual" ? <div
        className={`journal-status journal-status--${experience.run_state}`}
        id="journal-status"
        role="status"
        aria-live="polite"
      >
        {copy ? <p>{copy}</p> : null}
      </div> : null}
      <div className={`journal-status-actions${
        mode === "saved_replay" ? " journal-status-actions--replay" : ""
      }`}>
        {experience.run_state === "failed" && experience.retryable &&
        experience.pending_submission ? (
          <button
            className="button button--quiet"
            type="button"
            onClick={() => void retrySubmission()}
          >
            {pendingJournalEntrySaved
              ? experience.error_message === INSPECT_TRACE_ERROR
                ? "Try loading Inspect again"
                : "Try the follow-up check again"
              : "Try saving again"}
          </button>
        ) : null}
        {experience.run_state === "failed" &&
        experience.pending_submission &&
        !pendingJournalEntrySaved ? (
          <button
            className="button button--quiet"
            type="button"
            onClick={editPendingSubmission}
          >
            Edit Journal Entry
          </button>
        ) : null}
        {experience.trace_event_ids.length > 0 &&
        !isBusy &&
        !isAwaitingResponse ? (
          <button
            className="inspect-run-link"
            type="button"
            onClick={inspectLatest}
          >
            Inspect this run
          </button>
        ) : null}
      </div>

      {experience.journal_entries.length > 0 ? (
        <section className="journal-thread" aria-labelledby="journal-thread-title">
          <div className="journal-thread__heading">
            <p className="eyebrow">Your thread</p>
            <h2
              id="journal-thread-title"
              ref={threadHeadingRef}
              tabIndex={-1}
            >
              Moment by moment.
            </h2>
          </div>
          <ol>
            {experience.journal_entries.map((entry, position) => {
              const nudge: NudgeInteractionContract | null =
                experience.nudges.find(
                  (item) => item.journal_entry_id === entry.journal_entry_id,
                ) ?? null;
              return (
                <li
                  id={journalEntryAnchorId(entry.journal_entry_id)}
                  key={entry.journal_entry_id}
                  aria-current={
                    experience.selected_entry_id === entry.journal_entry_id
                      ? "true"
                      : undefined
                  }
                >
                  <div className="journal-thread__meta">
                    <time dateTime={entry.date}>{displayDate(entry.date)}</time>
                    {mode === "manual" ? (
                      <button
                        className="journal-thread__remove"
                        type="button"
                        aria-label={`Remove Journal Entry ${position + 1} of ${experience.journal_entries.length} from ${displayDate(entry.date)}`}
                        disabled={isBusy || isAwaitingResponse}
                        onClick={() =>
                          void removeJournalEntry(entry.journal_entry_id)}
                      >
                        {removingEntryId === entry.journal_entry_id
                          ? "Removing…"
                          : "Remove"}
                      </button>
                    ) : null}
                  </div>
                  <p className="journal-thread__entry">{entry.content}</p>
                  {nudge?.text && ["displayed", "answered", "skipped"].includes(
                    nudge.outcome,
                  ) ? (
                    <div className="journal-thread__exchange">
                      <p>{nudge.text}</p>
                      {nudge.outcome === "answered" && nudge.response ? (
                        <p>{nudge.response}</p>
                      ) : null}
                    </div>
                  ) : null}
                  {nudge?.outcome === "skipped" ? (
                    <p className="journal-thread__skipped">
                      Follow-up skipped.
                    </p>
                  ) : null}
                </li>
              );
            })}
          </ol>
        </section>
      ) : null}

      <WeeklyExperience
        profile={profile}
        journalEntries={experience.journal_entries}
        weeklyReviewerDecisions={experience.weekly_reviewer_decisions}
        driftResult={experience.drift_result}
        weeklyDigest={experience.weekly_digest}
        traceEvents={experience.trace_events}
        inspectRun={inspectRun}
        selectJournalEntry={(journalEntryId) =>
          updateExperience({ selected_entry_id: journalEntryId })
        }
      />
    </div>
  );
}
