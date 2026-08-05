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
}: WeeklyExperienceProps) {
  if (driftResult === null || weeklyDigest === null) return null;

  const rawStates = object(driftResult.core_value_states) ?? {};
  const aggregateState = deliveryState(driftResult.delivery_state);
  const evidence = digestEvidence(weeklyDigest);
  const entriesByIndex = new Map(
    journalEntries.map((entry) => [entry.t_index, entry]),
  );
  const coachEventId = latestEventId(traceEvents, "weekly_coach_generated");
  const coachEvent = [...traceEvents]
    .reverse()
    .find((event) => event.event_type === "weekly_coach_generated") ?? null;
  const coachUnavailable = coachEvent !== null
    && coachEvent.status !== "complete"
    && object(weeklyDigest.coach_narrative) === null;
  const driftEventId = latestEventId(traceEvents, "drift_detected");
  const digestEventId = latestEventId(traceEvents, "weekly_digest_built");
  const inspectEventId = coachEventId ?? driftEventId ?? digestEventId;
  const weekStart =
    typeof weeklyDigest.week_start === "string" ? weeklyDigest.week_start : null;
  const weekEnd =
    typeof weeklyDigest.week_end === "string" ? weeklyDigest.week_end : null;
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
            {profile.top_values.map((coreValue) => {
              const state = deliveryState(rawStates[coreValue]);
              const stateEvidence = evidence.filter((row) =>
                row.dimensions.length === 0
                  ? profile.top_values.length === 1
                  : row.dimensions.includes(coreValue),
              );
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
                  {stateEvidence.length > 0 ? (
                    <ol className="weekly-evidence">
                      {stateEvidence.map((row) => {
                        const entry = entriesByIndex.get(row.tIndex);
                        const earlierWeek =
                          weekStart !== null && row.date < weekStart;
                        return (
                          <li key={`${coreValue}-${row.date}-${row.tIndex}`}>
                            {earlierWeek ? <small>Earlier week</small> : null}
                            {entry ? (
                              <a
                                href={`#${journalEntryAnchorId(entry.journal_entry_id)}`}
                                onClick={() =>
                                  selectJournalEntry?.(entry.journal_entry_id)
                                }
                              >
                                <span>{displayEvidenceDate(row.date)}</span>
                                <q>{citationExcerpt(row.excerpt)}</q>
                              </a>
                            ) : (
                              <blockquote>{row.excerpt}</blockquote>
                            )}
                          </li>
                        );
                      })}
                    </ol>
                  ) : null}
                </li>
              );
            })}
          </ul>
        )}

        {evidence.length > 0 ? (
          <p className="weekly-experience__evidence-note">
            Selecting evidence opens and focuses that Journal Entry below.
          </p>
        ) : null}
      </div>

      <CoachDigestCard
        weeklyDigest={weeklyDigest}
        headingId="weekly-coach-title"
      />

      {coachUnavailable ? (
        <aside
          className="coach-digest coach-digest--unavailable"
          aria-labelledby="weekly-coach-unavailable-title"
        >
          <p className="eyebrow">Coach Digest</p>
          <h2 id="weekly-coach-unavailable-title">
            Your weekly response could not be prepared.
          </h2>
          <p>
            The Weekly Drift Detection result above remains available.
          </p>
        </aside>
      ) : null}

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
