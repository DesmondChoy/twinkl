import { useEffect, useMemo, useRef, type RefObject } from "react";
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
}

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

function statusCopy(experience: ExperienceState): string | null {
  switch (experience.run_state) {
    case "saving":
      return "Saving your Journal Entry…";
    case "checking_nudge":
      return "Saved. Looking for one useful follow-up…";
    case "awaiting_response":
      return "A follow-up question is ready.";
    case "complete":
      return "Saved.";
    case "refused":
    case "invalid":
      return "Saved. No follow-up question this time.";
    case "failed":
      return experience.error_message ??
        "Your Journal Entry is safe here, but the follow-up check could not finish.";
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
}: JournalExperienceProps) {
  const submissionLockRef = useRef(false);
  const nudgeHeadingRef = useRef<HTMLHeadingElement>(null);
  const activeNudge = useMemo(
    () =>
      [...experience.nudges]
        .reverse()
        .find((nudge) => nudge.outcome === "displayed") ?? null,
    [experience.nudges],
  );
  const isBusy = ["saving", "checking_nudge"].includes(experience.run_state);
  const isAwaitingResponse =
    experience.run_state === "awaiting_response" && activeNudge !== null;
  const copy = statusCopy(experience);

  useEffect(() => {
    if (isAwaitingResponse) {
      nudgeHeadingRef.current?.focus({ preventScroll: true });
    }
  }, [isAwaitingResponse]);

  const runSubmission = async (pending: PendingJournalSubmission) => {
    updateExperience({
      run_state: "saving",
      retryable: false,
      error_message: null,
    });
    try {
      await createExperienceSession(
        profile,
        experience.revision > 0 && experience.trace_events.length > 0
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
      const response = await submitJournalEntry({
        sessionId: profile.session_id,
        expectedRevision: pending.expected_revision,
        entry: pending.entry,
        idempotencyKey: pending.idempotency_key,
      });
      const trace = await readExperienceTrace(profile.session_id);
      const nudge =
        response.session.nudges.find(
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
        revision: response.session.revision,
        journal_draft: "",
        journal_entries: mergeJournalEntries(
          response.session.journal_entries,
          experience.journal_entries,
        ),
        nudges: mergeNudges(response.session.nudges, experience.nudges),
        weekly_reviewer_decisions:
          response.session.weekly_reviewer_decisions,
        drift_result: response.session.drift_result,
        weekly_digest: response.session.weekly_digest,
        pending_submission: failed ? pending : null,
        run_state: runState,
        retryable: failed,
        error_message: failed
          ? "Your Journal Entry is saved. The follow-up check could not finish."
          : null,
        trace_event_ids: response.session.trace_event_ids,
        trace_events: trace.events,
      });
    } catch (error) {
      const apiError =
        error instanceof ExperienceApiError
          ? error
          : new ExperienceApiError(
              "Your Journal Entry is safe here, but the follow-up check could not finish.",
            );
      updateExperience({
        run_state: "failed",
        retryable: apiError.retryable,
        error_message: apiError.message,
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
        t_index:
          experience.journal_entries.length === 0
            ? 0
            : experience.journal_entries.at(-1)!.t_index + 1,
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

  const finishNudge = (outcome: "skipped" | "answered") => {
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
    updateExperience({
      journal_entries: entries,
      nudges,
      nudge_response_draft: "",
      run_state: "complete",
      retryable: false,
      error_message: null,
    });
  };

  const inspectLatest = () => {
    const eventId = experience.trace_event_ids.at(-1);
    if (eventId) inspectRun(eventId);
  };

  return (
    <div className="journal-experience">
      <header className="journal-experience__header">
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
      </header>

      {!isAwaitingResponse ? (
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
            <span>{experience.journal_draft.trim().length} characters</span>
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

      {activeNudge && isAwaitingResponse ? (
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
              onClick={() => finishNudge("skipped")}
            >
              Skip for now
            </button>
            <button
              className="button button--primary"
              type="button"
              disabled={!experience.nudge_response_draft.trim()}
              onClick={() => finishNudge("answered")}
            >
              Save response
            </button>
          </div>
        </section>
      ) : null}

      <div
        className={`journal-status journal-status--${experience.run_state}`}
        id="journal-status"
        role="status"
        aria-live="polite"
      >
        {copy ? <p>{copy}</p> : null}
      </div>
      <div className="journal-status-actions">
        {experience.run_state === "failed" && experience.retryable &&
        experience.pending_submission ? (
          <button
            className="button button--quiet"
            type="button"
            onClick={() => void retrySubmission()}
          >
            Try the follow-up check again
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
            <h2 id="journal-thread-title">Moment by moment.</h2>
          </div>
          <ol>
            {experience.journal_entries.map((entry) => {
              const nudge: NudgeInteractionContract | null =
                experience.nudges.find(
                  (item) => item.journal_entry_id === entry.journal_entry_id,
                ) ?? null;
              return (
                <li
                  id={journalEntryAnchorId(entry.journal_entry_id)}
                  key={entry.journal_entry_id}
                >
                  <time dateTime={entry.date}>{displayDate(entry.date)}</time>
                  <p className="journal-thread__entry">{entry.content}</p>
                  {nudge?.outcome === "answered" ? (
                    <div className="journal-thread__exchange">
                      <p>{nudge.text}</p>
                      <p>{nudge.response}</p>
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
      />
    </div>
  );
}
