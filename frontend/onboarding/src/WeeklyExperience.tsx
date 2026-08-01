import {
  VALUES,
  type OnboardingProfile,
  type ValueKey,
} from "./domain";
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
type DeliveryState = "stable" | "active" | "recovered" | "uncertain" | "mixed";

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
    case "active":
      return "Active Drift";
    case "recovered":
      return "Recovered Drift";
    case "uncertain":
      return "Uncertain";
    case "mixed":
      return "Mixed";
    default:
      return "No Drift";
  }
}

function deliveryState(value: unknown): DeliveryState {
  return ["active", "recovered", "uncertain", "mixed"].includes(String(value))
    ? value as DeliveryState
    : "stable";
}

function stateExplanation(state: DeliveryState): string {
  switch (state) {
    case "active":
      return "Two consecutive Journal Entries went against this priority.";
    case "recovered":
      return "A later Journal Entry ended the earlier pattern.";
    case "uncertain":
      return "The latest Journal Entry was unclear, so Twinkl did not claim a current pattern.";
    case "mixed":
      return "The Core Values have different current states.";
    default:
      return "No repeated conflict was found this week.";
  }
}

function stateHeading(state: DeliveryState): string {
  switch (state) {
    case "active":
      return "A repeated conflict surfaced.";
    case "recovered":
      return "The earlier pattern eased.";
    case "uncertain":
      return "Not enough evidence yet.";
    case "mixed":
      return "Two priorities moved differently.";
    default:
      return "No repeated conflict this week.";
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
  const driftEventId = latestEventId(traceEvents, "drift_detected");
  const digestEventId = latestEventId(traceEvents, "weekly_digest_built");
  const inspectEventId = driftEventId ?? digestEventId;
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
  const coachNarrative = object(weeklyDigest.coach_narrative);
  const weeklyMirror =
    typeof coachNarrative?.weekly_mirror === "string"
      ? coachNarrative.weekly_mirror
      : null;
  const tensionExplanation =
    typeof coachNarrative?.tension_explanation === "string"
      ? coachNarrative.tension_explanation
      : null;
  const reflectiveQuestion =
    typeof coachNarrative?.reflective_question === "string"
      ? coachNarrative.reflective_question
      : null;

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
            <h2 id="weekly-view-title">{stateHeading(aggregateState)}</h2>
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

      {weeklyMirror || tensionExplanation || reflectiveQuestion ? (
        <aside className="coach-digest" aria-labelledby="weekly-coach-title">
          <p className="eyebrow">Weekly Coach</p>
          <h2 id="weekly-coach-title">{weeklyMirror}</h2>
          {tensionExplanation ? <p>{tensionExplanation}</p> : null}
          {reflectiveQuestion ? (
            <p className="coach-digest__question">{reflectiveQuestion}</p>
          ) : null}
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
