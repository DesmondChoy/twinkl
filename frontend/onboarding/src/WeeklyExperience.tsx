import {
  VALUES,
  type OnboardingProfile,
  type ValueKey,
} from "./domain";
import CoachDigestCard from "./CoachDigestCard";
import {
  displayWeekRange,
} from "./displayFormatters";
import type {
  JournalEntryContract,
  TraceEventContract,
  WeeklyDriftReviewerDecisionContract,
} from "./demoContracts";
import { journalEntryAnchorId } from "./journalEntryAnchor";
import {
  currentNorthStarEvent,
  displayableNorthStarSelection,
  useNorthStarProfileRef,
} from "./northStar";

type JsonObject = Record<string, unknown>;
type DeliveryState =
  | "active_drift"
  | "no_active_drift"
  | "insufficient_evidence";

interface WeeklyExperienceProps {
  profile: OnboardingProfile;
  journalEntries: JournalEntryContract[];
  weeklyReviewerDecisions: WeeklyDriftReviewerDecisionContract[];
  driftResult: JsonObject | null;
  weeklyDigest: JsonObject | null;
  traceEvents: TraceEventContract[];
  inspectRun: (eventId: string) => void;
  selectJournalEntry?: (journalEntryId: string) => void;
  showInspectAction?: boolean;
  northStarReview?: {
    pending: boolean;
    failed: boolean;
    retryable: boolean;
    retry: () => void;
  };
  coachReview?: {
    pending: boolean;
    retryable: boolean;
    retry: () => void;
    error?: string | null;
  };
}

interface DigestEvidence {
  date: string;
  tIndex: number;
  excerpt: string;
  dimensions: string[];
}

function object(value: unknown): JsonObject | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function stateLabel(state: DeliveryState): string {
  switch (state) {
    case "active_drift":
      return "Active Drift";
    case "insufficient_evidence":
      return "Insufficient evidence";
    default:
      return "No Active Drift";
  }
}

function deliveryState(value: unknown): DeliveryState {
  return ["active_drift", "no_active_drift", "insufficient_evidence"].includes(String(value))
    ? value as DeliveryState
    : "no_active_drift";
}

function stateExplanation(state: DeliveryState): string {
  switch (state) {
    case "active_drift":
      return "Two consecutive Journal Entries went against this priority.";
    case "insufficient_evidence":
      return (
        "A review failure prevented a current claim, or an Abstain or "
        + "Journal Entry gap blocked recent Conflict evidence."
      );
    default:
      return "No active Drift is confirmed now. This does not prove a positive change.";
  }
}

function stateHeading(state: DeliveryState): string {
  switch (state) {
    case "active_drift":
      return "A repeated conflict surfaced.";
    case "insufficient_evidence":
      return "Not enough evidence yet.";
    default:
      return "No active Drift now.";
  }
}

function coreValuePhrase(coreValue: string): string {
  return VALUES[coreValue as ValueKey]?.phrase ?? coreValue;
}

function citationExcerpt(excerpt: string, limit = 96): string {
  const compact = excerpt.replace(/\s+/g, " ").trim();
  return compact.length > limit
    ? `${compact.slice(0, limit - 1).trimEnd()}…`
    : compact;
}

function displayEvidenceDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
  }).format(new Date(`${value}T00:00:00`));
}

function latestEventId(
  events: TraceEventContract[],
  eventType: string,
): string | null {
  return [...events]
    .reverse()
    .find((event) => event.event_type === eventType)?.event_id ?? null;
}

function digestEvidence(digest: JsonObject): DigestEvidence[] {
  if (!Array.isArray(digest.evidence)) return [];
  return digest.evidence.flatMap((value) => {
    const row = object(value);
    if (
      row === null ||
      typeof row.date !== "string" ||
      !Number.isInteger(row.t_index) ||
      typeof row.excerpt !== "string"
    ) {
      return [];
    }
    return [{
      date: row.date,
      tIndex: Number(row.t_index),
      excerpt: row.excerpt,
      dimensions: Array.isArray(row.dimensions)
        ? row.dimensions.filter(
            (dimension): dimension is string => typeof dimension === "string",
          )
        : [],
    }];
  });
}

export default function WeeklyExperience({
  profile,
  journalEntries,
  weeklyReviewerDecisions,
  driftResult,
  weeklyDigest,
  traceEvents,
  inspectRun,
  selectJournalEntry,
  showInspectAction = true,
  northStarReview,
  coachReview,
}: WeeklyExperienceProps) {
  const profileRef = useNorthStarProfileRef(profile);
  if (driftResult === null) return null;

  const rawStates = object(driftResult.core_value_states) ?? {};
  const aggregateState = deliveryState(driftResult.delivery_state);
  const evidence = digestEvidence(weeklyDigest ?? {});
  const entriesByIndex = new Map(
    journalEntries.map((entry) => [entry.t_index, entry]),
  );
  const coachEventId = latestEventId(traceEvents, "weekly_coach_generated");
  const narrative = object(weeklyDigest?.coach_narrative);
  const hasCoach = ["weekly_mirror", "tension_explanation", "reflective_question"]
    .every((key) => typeof narrative?.[key] === "string"
      && narrative[key].trim().length > 0);
  const coachPending = coachReview?.pending ?? false;
  const coachUnavailable = !hasCoach && !coachPending;
  const driftEventId = latestEventId(traceEvents, "drift_detected");
  const digestEventId = latestEventId(traceEvents, "weekly_digest_built");
  const inspectEventId = coachEventId ?? driftEventId ?? digestEventId;
  const weekStart =
    typeof weeklyDigest?.week_start === "string" ? weeklyDigest.week_start : null;
  const weekEnd =
    typeof weeklyDigest?.week_end === "string" ? weeklyDigest.week_end : null;
  const northStar = currentNorthStarEvent({
    events: traceEvents, profile, profileRef, weeklyDigest, journalEntries,
  });
  const selectedMoment = displayableNorthStarSelection(northStar, profile, journalEntries, driftResult);
  const momentPending = northStarReview?.pending || northStar?.record.status === "pending";
  const momentFailed = northStarReview?.failed || northStar?.record.status === "failed";
  let momentStatus: string | null = null;
  if (momentPending) {
    momentStatus = "Looking for a moment in your writing that expressed one of your priorities…";
  } else if (momentFailed) {
    momentStatus = "A moment from your writing could not be prepared. Your weekly result remains available.";
  } else if (northStar?.record.status === "not_eligible") {
    momentStatus = northStar.record.reason === "insufficient_evidence"
      ? "A moment is not shown while this week's evidence is insufficient."
      : northStar.record.reason === "no_eligible_writing"
        ? "No eligible writing was available for a moment in this week's reflection."
        : "A moment could not be prepared from the available weekly evidence.";
  } else if (northStar?.record.status === "complete" && !selectedMoment) {
    momentStatus = northStar.record.selected === null
      ? "The review did not identify a supportive action in the eligible writing."
      : "The selected moment could not be verified against your current Journal Entries.";
  } else if (coachUnavailable && !northStar) {
    momentStatus = "A moment from your writing can be reviewed after your weekly reflection is ready.";
  }
  const drifts = Array.isArray(driftResult.drifts)
    ? driftResult.drifts.map(object).filter((row) => row !== null) : [];
  const valueEvidence = profile.top_values.map((coreValue) => {
    const state = deliveryState(rawStates[coreValue]);
    const context = evidence.filter((row) => (weekEnd === null || row.date <= weekEnd)
      && (row.dimensions.length === 0
        ? profile.top_values.length === 1 : row.dimensions.includes(coreValue)));
    const activeDrift = [...drifts].reverse().find((row) => row.core_value === coreValue
      && row.termination_reason == null && !Number.isInteger(row.termination_t_index));
    const startingIndices = new Set([activeDrift?.onset_t_index, activeDrift?.confirmation_t_index]
      .filter((index): index is number => typeof index === "number" && Number.isInteger(index)));
    const conflict = state !== "active_drift" ? [] : weeklyReviewerDecisions.flatMap((decision) => {
      const entry = entriesByIndex.get(decision.t_index);
      if (decision.core_value !== coreValue || decision.review_status !== "ok"
        || decision.verdict !== "conflict" || !entry || entry.date !== decision.date
        || (weekEnd !== null && entry.date > weekEnd)
        || (typeof activeDrift?.onset_t_index === "number" && entry.t_index < activeDrift.onset_t_index)
        || (!startingIndices.has(entry.t_index)
          && !(weekStart !== null && weekEnd !== null && entry.date >= weekStart && entry.date <= weekEnd))) {
        return [];
      }
      return [{
        date: entry.date, tIndex: entry.t_index, dimensions: [coreValue],
        excerpt: decision.evidence_quote.trim() && entry.content.includes(decision.evidence_quote)
          ? decision.evidence_quote : entry.content,
      }];
    }).sort((left, right) =>
      Number(!startingIndices.has(left.tIndex)) - Number(!startingIndices.has(right.tIndex))
      || left.tIndex - right.tIndex);
    const conflictIndices = new Set(conflict.map((row) => row.tIndex));
    return {
      coreValue, state, conflict,
      context: context.filter((row) => !conflictIndices.has(row.tIndex))
        .sort((left, right) => left.tIndex - right.tIndex),
    };
  });
  const renderEvidence = (rows: DigestEvidence[], coreValue: string) => (
    <ol className="weekly-evidence">
      {rows.map((row) => {
        const entry = entriesByIndex.get(row.tIndex);
        return (
          <li key={`${coreValue}-${row.date}-${row.tIndex}`}>
            {weekStart !== null && row.date < weekStart ? <small>Earlier week</small> : null}
            {entry ? (
              <a href={`#${journalEntryAnchorId(entry.journal_entry_id)}`}
                onClick={() => selectJournalEntry?.(entry.journal_entry_id)}>
                <span>{displayEvidenceDate(row.date)}</span>
                <q>{citationExcerpt(row.excerpt)}</q>
              </a>
            ) : <blockquote>{row.excerpt}</blockquote>}
          </li>
        );
      })}
    </ol>
  );
  const reviewUnavailable =
    weekStart !== null &&
    weekEnd !== null &&
    weeklyReviewerDecisions.some(
      (decision) =>
        decision.week_start === weekStart && decision.week_end === weekEnd,
    ) &&
    weeklyReviewerDecisions
      .filter(
        (decision) =>
          decision.week_start === weekStart && decision.week_end === weekEnd,
      )
      .every((decision) => decision.review_status !== "ok");
  return (
    <section
      className="weekly-workspace"
      id="experience-weekly"
      aria-labelledby="weekly-view-title"
    >
      <div
        className={`weekly-experience weekly-experience--${aggregateState}`}
      >
        <header className="weekly-experience__header">
          <div>
            <p className="eyebrow">What Twinkl noticed</p>
            <h2 id="weekly-view-title" tabIndex={-1}>
              {stateHeading(aggregateState)}
            </h2>
          </div>
          <span
            className={`weekly-experience__state weekly-experience__state--${aggregateState}`}
          >
            {reviewUnavailable
              ? "Review unavailable"
              : stateLabel(aggregateState)}
          </span>
        </header>

        {weekStart && weekEnd ? (
          <p className="weekly-experience__dates">
            {displayWeekRange(weekStart, weekEnd)}
          </p>
        ) : null}

        {reviewUnavailable ? (
          <p className="weekly-experience__summary">
            The Weekly Drift Reviewer could not return usable evidence for this
            week.
          </p>
        ) : (
          <ul
            className="weekly-experience__values"
            aria-label="Current Drift by Core Value"
          >
            {valueEvidence.map(({ coreValue, state, conflict, context }) => {
              return (
                <li
                  className={`weekly-value weekly-value--${state}`}
                  key={coreValue}
                >
                  <div className="weekly-value__heading">
                    <span>{coreValuePhrase(coreValue)}</span>
                    <strong>{stateLabel(state)}</strong>
                  </div>
                  <p>{stateExplanation(state)}</p>
                  {conflict.length > 0 ? (
                    <section aria-label="Journal Entries behind this Drift">
                      <h3>Journal Entries behind this Drift</h3>
                      {renderEvidence(conflict, coreValue)}
                    </section>
                  ) : null}
                  {context.length > 0 ? (
                    <section aria-label={state === "active_drift" ? "Other Journal Entry context" : "Journal Entry context"}>
                      <h3>{state === "active_drift" ? "Other Journal Entry context" : "Journal Entry context"}</h3>
                      {state === "active_drift" ? (
                        <p>These entries provide context; they do not establish this Drift.</p>
                      ) : null}
                      {renderEvidence(context, coreValue)}
                    </section>
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}

        {valueEvidence.some((value) => value.conflict.length > 0 || value.context.length > 0) ? (
          <p className="weekly-experience__evidence-note">
            Selecting evidence opens and focuses its Journal Entry.
          </p>
        ) : null}
      </div>

      <div className="weekly-workspace__response">
        <CoachDigestCard
          weeklyDigest={coachPending ? null : weeklyDigest}
          headingId="weekly-coach-title"
          journalEntries={journalEntries}
          northStar={momentPending || momentFailed ? undefined : {
            profile, driftResult, traceEvents, inspectMoment: inspectRun,
          }}
          onOpenEntry={selectJournalEntry ? (entry) => {
            selectJournalEntry(entry.journal_entry_id);
            const target = document.getElementById(journalEntryAnchorId(entry.journal_entry_id));
            target?.focus({ preventScroll: true });
            target?.scrollIntoView?.({ block: "center", behavior: "smooth" });
          } : undefined}
        />

        {coachPending || coachUnavailable ? (
          <aside
            className="coach-digest coach-digest--unavailable"
            aria-labelledby="weekly-coach-unavailable-title"
          >
            <p className="eyebrow">Coach Digest</p>
            <h2 id="weekly-coach-unavailable-title">
              {coachPending ? "Preparing your weekly reflection…" : "Your weekly response could not be prepared."}
            </h2>
            <p>
              The Weekly Drift Detection result remains available.
            </p>
            {coachUnavailable && coachReview?.error ? <p>{coachReview.error}</p> : null}
            {coachUnavailable && coachReview?.retryable ? (
              <button className="button button--quiet" type="button" onClick={coachReview.retry}>
                Retry Coach Digest
              </button>
            ) : null}
          </aside>
        ) : null}
        {momentStatus ? (
          <section className="north-star-status" aria-label="Moment review">
            <p role="status" aria-label="Moment review status">{momentStatus}</p>
            {momentFailed && !momentPending && northStarReview?.retryable ? (
              <button className="button button--quiet" type="button" onClick={northStarReview.retry}>
                Retry moment review
              </button>
            ) : null}
          </section>
        ) : null}
      </div>

      {showInspectAction && inspectEventId ? (
        <button
          className="button button--primary weekly-workspace__inspect"
          type="button"
          onClick={() => inspectRun(inspectEventId)}
        >
          See how this was decided
        </button>
      ) : null}
    </section>
  );
}
