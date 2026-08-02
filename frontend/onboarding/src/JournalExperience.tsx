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
  advanceAssessmentTime,
  createExperienceSession,
  ExperienceApiError,
  journalIdempotencyKey,
  readExperienceTrace,
  submitJournalEntry,
} from "./experienceApi";
import { journalEntryAnchorId } from "./journalEntryAnchor";
import { NUDGE_REVEAL_DELAY_MS } from "./nudgeReveal";
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
  showWeeklySummary?: boolean;
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

function isoDate(value: Date): string {
  return value.toISOString().slice(0, 10);
}

function addDays(value: string, days: number): string {
  const parsed = new Date(`${value}T00:00:00Z`);
  parsed.setUTCDate(parsed.getUTCDate() + days);
  return isoDate(parsed);
}

function mondayFor(value: string): string {
  const parsed = new Date(`${value}T00:00:00Z`);
  const offset = (parsed.getUTCDay() + 6) % 7;
  return addDays(value, -offset);
}

function nextMonday(value: string): string {
  const parsed = new Date(`${value}T00:00:00Z`);
  const weekday = (parsed.getUTCDay() + 6) % 7;
  return addDays(value, 7 - weekday);
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

function latestEventId(
  events: TraceEventContract[],
  eventType: string,
): string | null {
  return [...events]
    .reverse()
    .find((event) => event.event_type === eventType)?.event_id ?? null;
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
      return "Updating saved details…";
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
  showWeeklySummary = true,
}: JournalExperienceProps) {
  const submissionLockRef = useRef(false);
  const nudgeHeadingRef = useRef<HTMLHeadingElement>(null);
  const threadHeadingRef = useRef<HTMLHeadingElement>(null);
  const latestEntryRef = useRef<HTMLLIElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const previousLatestEntryIdRef = useRef(
    experience.journal_entries.at(-1)?.journal_entry_id ?? null,
  );
  const [removingEntryId, setRemovingEntryId] = useState<string | null>(null);
  const [timeAction, setTimeAction] = useState<
    "next_day" | "close_week" | null
  >(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [revealedNudgeId, setRevealedNudgeId] = useState<string | null>(null);
  const replayWeekStart =
    typeof experience.weekly_digest?.week_start === "string"
      ? experience.weekly_digest.week_start
      : null;
  const replayWeekEnd =
    typeof experience.weekly_digest?.week_end === "string"
      ? experience.weekly_digest.week_end
      : null;
  const currentWeekEntries = useMemo(
    () =>
      mode === "saved_replay" && replayWeekStart && replayWeekEnd
        ? experience.journal_entries.filter(
            (entry) =>
              entry.date >= replayWeekStart && entry.date <= replayWeekEnd,
          )
        : experience.journal_entries,
    [
      experience.journal_entries,
      mode,
      replayWeekEnd,
      replayWeekStart,
    ],
  );
  const earlierEntries = useMemo(
    () =>
      mode === "saved_replay"
        ? experience.journal_entries.filter(
            (entry) => !currentWeekEntries.includes(entry),
          )
        : [],
    [currentWeekEntries, experience.journal_entries, mode],
  );
  const displayedCurrentWeekEntries = useMemo(
    () =>
      mode === "manual"
        ? [...currentWeekEntries].reverse()
        : currentWeekEntries,
    [currentWeekEntries, mode],
  );
  const replayInspectEventId =
    latestEventId(experience.trace_events, "drift_detected")
    ?? latestEventId(experience.trace_events, "weekly_digest_built");
  const activeNudge = useMemo(
    () =>
      [...experience.nudges]
        .reverse()
        .find((nudge) => nudge.outcome === "displayed") ?? null,
    [experience.nudges],
  );
  const assessmentDate = experience.assessment_clock?.current_date ?? null;
  const assessmentWeekStart = assessmentDate
    ? mondayFor(assessmentDate)
    : null;
  const assessmentWeekEnd = assessmentWeekStart
    ? addDays(assessmentWeekStart, 6)
    : null;
  const assessmentWeekEntries = useMemo(
    () => assessmentWeekStart && assessmentWeekEnd
      ? experience.journal_entries.filter(
          (entry) =>
            entry.date >= assessmentWeekStart
            && entry.date <= assessmentWeekEnd,
        )
      : [],
    [
      assessmentWeekEnd,
      assessmentWeekStart,
      experience.journal_entries,
    ],
  );
  const latestEntry = experience.journal_entries.at(-1) ?? null;
  const isBusy = ["saving", "checking_nudge", "running"].includes(
    experience.run_state,
  );
  const isAwaitingResponse =
    experience.run_state === "awaiting_response" && activeNudge !== null;
  const isActiveNudgeVisible =
    isAwaitingResponse && revealedNudgeId === activeNudge?.nudge_id;
  const pendingJournalEntrySaved =
    experience.pending_submission !== null
    && experience.journal_entries.some(
      (entry) =>
        entry.journal_entry_id
        === experience.pending_submission?.entry.journal_entry_id,
    );
  const copy = timeAction === "next_day"
    ? "Moving to the next day…"
    : timeAction === "close_week"
      ? "Closing the week and preparing its review…"
      : isAwaitingResponse && !isActiveNudgeVisible
        ? "Saved."
        : statusCopy(experience);

  useEffect(() => {
    if (!isAwaitingResponse || !activeNudge) return;
    const nudgeId = activeNudge.nudge_id;
    const timer = window.setTimeout(
      () => setRevealedNudgeId(nudgeId),
      NUDGE_REVEAL_DELAY_MS,
    );
    return () => window.clearTimeout(timer);
  }, [activeNudge, isAwaitingResponse]);

  useEffect(() => {
    if (isActiveNudgeVisible) {
      nudgeHeadingRef.current?.focus({ preventScroll: true });
      nudgeHeadingRef.current?.scrollIntoView?.({ block: "center" });
    }
  }, [isActiveNudgeVisible]);

  useEffect(() => {
    if (mode !== "manual") return;
    const latestEntryId =
      experience.journal_entries.at(-1)?.journal_entry_id ?? null;
    if (
      latestEntryId !== null
      && latestEntryId !== previousLatestEntryIdRef.current
      && !isActiveNudgeVisible
    ) {
      latestEntryRef.current?.focus({ preventScroll: true });
      latestEntryRef.current?.scrollIntoView?.({ block: "center" });
    }
    previousLatestEntryIdRef.current = latestEntryId;
  }, [experience.journal_entries, isActiveNudgeVisible, mode]);

  useEffect(() => {
    if (mode === "saved_replay") setHistoryOpen(false);
  }, [mode, replayWeekEnd, replayWeekStart]);

  useEffect(() => {
    if (
      mode !== "saved_replay"
      || !experience.selected_entry_id
      || !earlierEntries.some(
        (entry) =>
          entry.journal_entry_id === experience.selected_entry_id,
      )
    ) {
      return;
    }
    setHistoryOpen(true);
    const focusSelectedEntry = () => {
      const target = document.getElementById(
        journalEntryAnchorId(experience.selected_entry_id!),
      );
      target?.focus({ preventScroll: true });
      target?.scrollIntoView?.({ block: "center" });
    };
    if (window.requestAnimationFrame) {
      window.requestAnimationFrame(focusSelectedEntry);
    } else {
      focusSelectedEntry();
    }
  }, [
    earlierEntries,
    experience.selected_entry_id,
    mode,
  ]);

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
              assessment_clock: experience.assessment_clock,
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
        assessment_clock: acceptedResponse.session.assessment_clock,
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
          assessment_clock: acceptedResponse.session.assessment_clock,
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
        date: experience.assessment_clock?.current_date ?? localDate(),
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
      assessment_clock: experience.assessment_clock,
      trace_events: experience.trace_events,
    };
    const updatedState = {
      session_id: profile.session_id,
      revision: experience.revision + 1,
      journal_entries: journalEntries,
      nudges,
      assessment_clock: experience.assessment_clock,
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
        assessment_clock: synchronized.response.session.assessment_clock,
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
          "Your response is still here, but the saved Journal Entry could not update. Try again.",
      });
    }
  };

  const changeAssessmentTime = async (
    action: "next_day" | "close_week",
  ) => {
    if (
      experience.assessment_clock === null
      || isBusy
      || isAwaitingResponse
      || timeAction !== null
    ) {
      return;
    }
    let acceptedResponse: Awaited<
      ReturnType<typeof advanceAssessmentTime>
    > | null = null;
    setTimeAction(action);
    updateExperience({
      run_state: "running",
      retryable: false,
      error_message: null,
    });
    try {
      acceptedResponse = await advanceAssessmentTime({
        sessionId: profile.session_id,
        expectedRevision: experience.revision,
        action,
      });
      const trace = await readExperienceTrace(profile.session_id);
      updateExperience({
        revision: acceptedResponse.session.revision,
        journal_entries: acceptedResponse.session.journal_entries,
        nudges: acceptedResponse.session.nudges,
        weekly_reviewer_decisions:
          acceptedResponse.session.weekly_reviewer_decisions,
        drift_result: acceptedResponse.session.drift_result,
        weekly_digest: acceptedResponse.session.weekly_digest,
        assessment_clock: acceptedResponse.session.assessment_clock,
        run_state: "complete",
        retryable: false,
        error_message: null,
        trace_event_ids: acceptedResponse.session.trace_event_ids,
        trace_events: trace.events,
      });
      const focusTarget = () => {
        if (action === "next_day") {
          composerRef.current?.focus({ preventScroll: true });
          composerRef.current?.scrollIntoView?.({ block: "center" });
          return;
        }
        const weeklyHeading = document.getElementById("weekly-view-title");
        weeklyHeading?.focus({ preventScroll: true });
        weeklyHeading?.scrollIntoView?.({ block: "start" });
      };
      window.requestAnimationFrame?.(focusTarget);
    } catch (error) {
      const apiError = error instanceof ExperienceApiError
        ? error
        : new ExperienceApiError(
            "Simulated time could not change.",
            "assessment_time_failed",
            true,
          );
      if (acceptedResponse !== null) {
        updateExperience({
          revision: acceptedResponse.session.revision,
          journal_entries: acceptedResponse.session.journal_entries,
          nudges: acceptedResponse.session.nudges,
          weekly_reviewer_decisions:
            acceptedResponse.session.weekly_reviewer_decisions,
          drift_result: acceptedResponse.session.drift_result,
          weekly_digest: acceptedResponse.session.weekly_digest,
          assessment_clock: acceptedResponse.session.assessment_clock,
          run_state: "failed",
          retryable: apiError.retryable,
          error_message:
            "The date changed, but its Inspect details could not be loaded.",
          trace_event_ids: acceptedResponse.session.trace_event_ids,
        });
      } else {
        updateExperience({
          run_state: "failed",
          retryable: apiError.retryable,
          error_message: apiError.message,
        });
      }
    } finally {
      setTimeAction(null);
    }
  };

  const removeJournalEntry = async (journalEntryId: string) => {
    if (
      isBusy ||
      isAwaitingResponse ||
      removingEntryId !== null ||
      !window.confirm(
        "Remove this Journal Entry? Any reviewed later weeks will update.",
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
        assessment_clock: synchronized.response.session.assessment_clock,
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

  const renderJournalThread = (
    entries: JournalEntryContract[],
    id: string,
    eyebrow: string,
    title: string,
    focusHeading = false,
  ) => (
    <section className="journal-thread" aria-labelledby={id}>
      <div className="journal-thread__heading">
        <p className="eyebrow">{eyebrow}</p>
        <h2
          id={id}
          ref={focusHeading ? threadHeadingRef : undefined}
          tabIndex={focusHeading ? -1 : undefined}
        >
          {title}
        </h2>
      </div>
      <ol>
        {entries.map((entry) => {
          const position = experience.journal_entries.findIndex(
            (candidate) =>
              candidate.journal_entry_id === entry.journal_entry_id,
          );
          const nudge: NudgeInteractionContract | null =
            experience.nudges.find(
              (item) => item.journal_entry_id === entry.journal_entry_id,
            ) ?? null;
          const showNudge = nudge !== null
            && nudge.text
            && ["displayed", "answered", "skipped"].includes(nudge.outcome)
            && (
              mode !== "manual"
              || nudge.outcome !== "displayed"
              || nudge.nudge_id !== activeNudge?.nudge_id
              || isActiveNudgeVisible
            );
          const decision = latestDecisionFor(
            experience.trace_events,
            entry.journal_entry_id,
          );
          const entryFinalized = nudge !== null
            ? nudge.outcome !== "displayed"
            : ["complete", "refused", "invalid"].includes(
                decision?.status ?? "",
              );
          const isLatestEntry =
            entry.journal_entry_id === latestEntry?.journal_entry_id;
          const showTimeActions =
            mode === "manual"
            && assessmentDate !== null
            && entry.date === assessmentDate
            && isLatestEntry
            && entryFinalized
            && !isBusy
            && !isAwaitingResponse;
          return (
            <li
              id={journalEntryAnchorId(entry.journal_entry_id)}
              key={entry.journal_entry_id}
              ref={mode === "manual" && isLatestEntry ? latestEntryRef : undefined}
              tabIndex={-1}
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
              {showNudge ? (
                <div className={`journal-thread__exchange${
                  nudge.nudge_id === activeNudge?.nudge_id
                    ? " nudge-reveal"
                    : ""
                }`}>
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
              {showTimeActions && assessmentWeekStart && assessmentWeekEnd ? (
                <div className="simulated-time-actions">
                  <p>
                    Choose when you want to write again, or close this week and
                    review {assessmentWeekEntries.length} finalized {assessmentWeekEntries.length === 1
                      ? "Journal Entry"
                      : "Journal Entries"}.
                  </p>
                  <div>
                    <button
                      className="button button--quiet"
                      type="button"
                      onClick={() => void changeAssessmentTime("next_day")}
                    >
                      Write on the next day
                    </button>
                    <button
                      className="button button--primary"
                      type="button"
                      aria-describedby="close-week-effect"
                      onClick={() => void changeAssessmentTime("close_week")}
                    >
                      Close week and review
                    </button>
                  </div>
                  <small id="close-week-effect">
                    Moves to {displayDate(nextMonday(assessmentDate))}. Weekly
                    Drift Detection will review {displayDate(assessmentWeekStart)}–{displayDate(assessmentWeekEnd)},
                    then Coach Digest will run.
                  </small>
                </div>
              ) : null}
            </li>
          );
        })}
      </ol>
    </section>
  );

  const weeklyExperience = (
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
      showInspectAction={mode === "manual"}
    />
  );

  return (
    <div className="journal-experience">
      {mode === "manual" ? (
        <header
          className="journal-experience__header"
          id="experience-journal-prompt"
        >
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
            Write one real moment. Twinkl may ask one brief question if there
            is a useful thread to follow.
          </p>
          <CoreValueReminder profile={profile} />
        </header>
      ) : null}

      {mode === "manual" && experience.assessment_clock ? (
        <aside className="simulated-time" aria-label="Simulated time">
          <div>
            <p className="eyebrow">Simulated time</p>
            <time dateTime={experience.assessment_clock.current_date}>
              {displayDate(experience.assessment_clock.current_date)}
            </time>
          </div>
          <p>New Journal Entries use this date in the current session.</p>
        </aside>
      ) : null}

      {mode === "manual" && !isAwaitingResponse ? (
        <form
          className="journal-composer"
          id="experience-journal-compose"
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
            ref={composerRef}
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

      {mode === "manual" && activeNudge && isActiveNudgeVisible ? (
        <section
          className="nudge-reply nudge-reveal"
          id="experience-journal-compose"
          aria-labelledby="nudge-question"
        >
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
        {mode === "manual" &&
        experience.trace_event_ids.length > 0 &&
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

      {mode === "saved_replay" && showWeeklySummary ? weeklyExperience : null}

      {displayedCurrentWeekEntries.length > 0
        ? renderJournalThread(
            displayedCurrentWeekEntries,
            "journal-thread-title",
            mode === "saved_replay" ? "Current week" : "Your thread",
            mode === "saved_replay"
              ? "This week’s Journal Entries"
              : "Moment by moment.",
            mode === "manual",
          )
        : null}

      {mode === "saved_replay" && earlierEntries.length > 0 ? (
        <details
          className="journal-history"
          open={historyOpen}
          onToggle={(event) => setHistoryOpen(event.currentTarget.open)}
        >
          <summary>Earlier Journal Entries · {earlierEntries.length}</summary>
          {renderJournalThread(
            earlierEntries,
            "earlier-journal-thread-title",
            "Earlier history",
            "Earlier Journal Entries",
          )}
        </details>
      ) : null}

      {mode === "saved_replay" && showWeeklySummary && replayInspectEventId ? (
        <button
          className="button button--primary replay-inspect-action"
          type="button"
          onClick={() => inspectRun(replayInspectEventId)}
        >
          See how this was decided
        </button>
      ) : null}

      {mode === "manual" ? weeklyExperience : null}
    </div>
  );
}
