import type { OnboardingProfile } from "./domain";
import {
  displayCoreValue,
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
}

interface DigestEvidence {
  date: string;
  tIndex: number;
  excerpt: string;
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

function coreValueStateSummary(coreValue: string, state: DeliveryState): string {
  const label = displayCoreValue(coreValue);
  switch (state) {
    case "active":
      return `${label} has Active Drift`;
    case "recovered":
      return `${label} shows Recovered Drift`;
    case "uncertain":
      return `${label} is Uncertain`;
    case "mixed":
      return `${label} has Mixed Drift states`;
    default:
      return `${label} has No Drift`;
  }
}

function citationExcerpt(excerpt: string): string {
  const compact = excerpt.replace(/\s+/g, " ").trim();
  return compact.length > 56 ? `${compact.slice(0, 55).trimEnd()}…` : compact;
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
  const rationale =
    typeof weeklyDigest.mode_rationale === "string"
      ? weeklyDigest.mode_rationale
      : null;
  const summary = reviewUnavailable
    ? "The Weekly Drift Reviewer could not return usable evidence for this week."
    : aggregateState === "mixed"
      ? `${profile.top_values
          .map((coreValue) =>
            coreValueStateSummary(
              coreValue,
              deliveryState(rawStates[coreValue]),
            ),
          )
          .join("; ")}.`
      : rationale;

  return (
    <section className="weekly-experience" aria-labelledby="weekly-digest-title">
      <header className="weekly-experience__header">
        <div>
          <p className="eyebrow">Weekly Digest</p>
          <h2 id="weekly-digest-title">Your week in view.</h2>
        </div>
        <span className={`weekly-experience__state weekly-experience__state--${aggregateState}`}>
          {reviewUnavailable ? "Review unavailable" : stateLabel(aggregateState)}
        </span>
      </header>

      {weekStart && weekEnd ? (
        <p className="weekly-experience__dates">
          {displayWeekRange(weekStart, weekEnd)}
        </p>
      ) : null}

      <ul className="weekly-experience__values" aria-label="Core Value Drift">
        {profile.top_values.map((coreValue) => {
          const state = deliveryState(rawStates[coreValue]);
          return (
            <li key={coreValue}>
              <span>{displayCoreValue(coreValue)}</span>
              <strong>
                {reviewUnavailable ? "Review unavailable" : stateLabel(state)}
              </strong>
            </li>
          );
        })}
      </ul>

      {summary ? <p className="weekly-experience__summary">{summary}</p> : null}

      {evidence.length > 0 ? (
        <div className="weekly-evidence">
          <h3>Journal Entries behind this view</h3>
          <ol>
            {evidence.map((row) => {
              const entry = entriesByIndex.get(row.tIndex);
              return (
                <li key={`${row.date}-${row.tIndex}`}>
                  <blockquote>{row.excerpt}</blockquote>
                  {entry ? (
                    <a
                      href={`#${journalEntryAnchorId(entry.journal_entry_id)}`}
                      onClick={() =>
                        selectJournalEntry?.(entry.journal_entry_id)
                      }
                    >
                      Open Journal Entry “{citationExcerpt(row.excerpt)}” from{" "}
                      {row.date}
                    </a>
                  ) : null}
                </li>
              );
            })}
          </ol>
        </div>
      ) : null}

      <div className="weekly-experience__actions">
        {driftEventId ? (
          <button
            className="inspect-run-link"
            type="button"
            onClick={() => inspectRun(driftEventId)}
          >
            Inspect Drift run
          </button>
        ) : null}
        {digestEventId ? (
          <button
            className="inspect-run-link"
            type="button"
            onClick={() => inspectRun(digestEventId)}
          >
            Inspect Weekly Digest run
          </button>
        ) : null}
      </div>
    </section>
  );
}
