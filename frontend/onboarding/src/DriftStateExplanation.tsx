import {
  VALUES,
  type OnboardingProfile,
  type ValueKey,
} from "./domain";
import type {
  JournalEntryContract,
  WeeklyDriftReviewerDecisionContract,
} from "./demoContracts";

type JsonObject = Record<string, unknown>;
type CoreValueState = "stable" | "active" | "recovered" | "uncertain";

interface DriftStateExplanationProps {
  profile: OnboardingProfile;
  journalEntries: JournalEntryContract[];
  weeklyReviewerDecisions: WeeklyDriftReviewerDecisionContract[];
  driftResult: JsonObject | null;
  onOpenEntry: (entry: JournalEntryContract) => void;
}

interface DriftRecord extends JsonObject {
  core_value: string;
  onset_t_index: number;
  confirmation_t_index: number;
  end_t_index: number;
  termination_t_index?: number;
  termination_verdict?: string;
}

function object(value: unknown): JsonObject | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function coreValueState(value: unknown): CoreValueState {
  return ["active", "recovered", "uncertain"].includes(String(value))
    ? value as CoreValueState
    : "stable";
}

function stateLabel(state: CoreValueState): string {
  switch (state) {
    case "active":
      return "Active Drift";
    case "recovered":
      return "Recovered Drift";
    case "uncertain":
      return "Uncertain";
    default:
      return "No Drift";
  }
}

function decisionLabel(
  decision: WeeklyDriftReviewerDecisionContract | undefined,
): string {
  switch (decision?.verdict) {
    case "conflict":
      return "Conflict";
    case "not_conflict":
      return "Not Conflict";
    case "abstain":
      return "Abstain";
    default:
      return "Decision unavailable";
  }
}

function abstainReason(reasonCode: string | null | undefined): string {
  switch (reasonCode) {
    case "feeling_or_intent_only":
      return "The Journal Entry states a feeling or intent, but no clear action.";
    case "external_constraint":
      return "An external constraint prevents a clear decision about the person's choice.";
    case "missing_text":
      return "The Journal Entry does not contain enough text for a clear decision.";
    default:
      return "The Journal Entry does not support a clear decision.";
  }
}

function entryExcerpt(entry: JournalEntryContract, limit = 170): string {
  const compact = entry.content.replace(/\s+/g, " ").trim();
  return compact.length > limit
    ? `${compact.slice(0, limit - 1).trimEnd()}…`
    : compact;
}

function displayEntryDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(new Date(`${value}T00:00:00`));
}

function driftRecords(value: unknown): DriftRecord[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const row = object(item);
    if (
      row === null
      || typeof row.core_value !== "string"
      || !Number.isInteger(row.onset_t_index)
      || !Number.isInteger(row.confirmation_t_index)
      || !Number.isInteger(row.end_t_index)
    ) {
      return [];
    }
    return [row as DriftRecord];
  });
}

interface DecisionEvidenceProps {
  entry: JournalEntryContract;
  decision: WeeklyDriftReviewerDecisionContract | undefined;
  onOpenEntry: (entry: JournalEntryContract) => void;
}

function DecisionEvidence({
  entry,
  decision,
  onOpenEntry,
}: DecisionEvidenceProps) {
  return (
    <button
      className="state-change__entry"
      type="button"
      onClick={() => onOpenEntry(entry)}
      aria-label={`Read Journal Entry from ${displayEntryDate(entry.date)}`}
    >
      <span className="state-change__entry-meta">
        <time dateTime={entry.date}>{displayEntryDate(entry.date)}</time>
        <strong className={`review-decision review-decision--${
          decision?.verdict ?? "unavailable"
        }`}>
          {decisionLabel(decision)}
        </strong>
      </span>
      <q>{entryExcerpt(entry)}</q>
      <span className="state-change__read">Read Journal Entry</span>
    </button>
  );
}

export default function DriftStateExplanation({
  profile,
  journalEntries,
  weeklyReviewerDecisions,
  driftResult,
  onOpenEntry,
}: DriftStateExplanationProps) {
  if (driftResult === null) return null;

  const entriesByIndex = new Map(
    journalEntries.map((entry) => [entry.t_index, entry]),
  );
  const states = object(driftResult.core_value_states) ?? {};
  const drifts = driftRecords(driftResult.drifts);
  const decisionFor = (tIndex: number, coreValue: string) =>
    weeklyReviewerDecisions.find(
      (decision) =>
        decision.t_index === tIndex && decision.core_value === coreValue,
    );

  return (
    <div className="state-change-list" aria-label="Reason for each current state">
      {profile.top_values.map((coreValue) => {
        const state = coreValueState(states[coreValue]);
        const drift = [...drifts]
          .reverse()
          .find((item) => item.core_value === coreValue);
        const startIndices = drift
          ? [drift.onset_t_index, drift.confirmation_t_index]
          : [];
        const continuedIndices = drift
          ? Array.from(
              {
                length: Math.max(
                  0,
                  drift.end_t_index - drift.confirmation_t_index,
                ),
              },
              (_, index) => drift.confirmation_t_index + index + 1,
            ).filter(
              (tIndex) =>
                decisionFor(tIndex, coreValue)?.verdict === "conflict",
            )
          : [];
        const terminationIndex =
          drift && Number.isInteger(drift.termination_t_index)
            ? Number(drift.termination_t_index)
            : null;
        const terminationEntry = terminationIndex === null
          ? null
          : entriesByIndex.get(terminationIndex) ?? null;

        return (
          <section
            className={`state-change state-change--${state}`}
            key={coreValue}
          >
            <header>
              <span>{VALUES[coreValue as ValueKey]?.phrase ?? coreValue}</span>
              <strong>{stateLabel(state)}</strong>
            </header>

            {state === "stable" || !drift ? (
              <p className="state-change__summary">
                No two consecutive Weekly Drift Reviewer Conflicts were found.
              </p>
            ) : null}

            {state === "active" ? (
              <>
                <p className="state-change__marker">Drift started here.</p>
                <div className="state-change__evidence-pair">
                  {startIndices.flatMap((tIndex) => {
                    const entry = entriesByIndex.get(tIndex);
                    return entry
                      ? [(
                          <DecisionEvidence
                            entry={entry}
                            decision={decisionFor(tIndex, coreValue)}
                            onOpenEntry={onOpenEntry}
                            key={`${coreValue}-${tIndex}`}
                          />
                        )]
                      : [];
                  })}
                </div>
                {continuedIndices.length > 0 ? (
                  <>
                    <p className="state-change__marker">Drift continued.</p>
                    <div className="state-change__evidence-pair">
                      {continuedIndices.flatMap((tIndex) => {
                        const entry = entriesByIndex.get(tIndex);
                        return entry
                          ? [(
                              <DecisionEvidence
                                entry={entry}
                                decision={decisionFor(tIndex, coreValue)}
                                onOpenEntry={onOpenEntry}
                                key={`${coreValue}-${tIndex}`}
                              />
                            )]
                          : [];
                      })}
                    </div>
                  </>
                ) : null}
              </>
            ) : null}

            {state === "recovered" && terminationEntry ? (
              <>
                <p className="state-change__summary">
                  The Weekly Drift Reviewer marked this Journal Entry as Not
                  Conflict. The Drift Detector ended the earlier Drift.
                </p>
                <DecisionEvidence
                  entry={terminationEntry}
                  decision={decisionFor(terminationEntry.t_index, coreValue)}
                  onOpenEntry={onOpenEntry}
                />
              </>
            ) : null}

            {state === "uncertain" && terminationEntry ? (
              <>
                <p className="state-change__summary">
                  {abstainReason(
                    decisionFor(terminationEntry.t_index, coreValue)
                      ?.reason_code,
                  )} The Weekly Drift Reviewer abstained, so the Drift Detector
                  did not claim a new state.
                </p>
                <DecisionEvidence
                  entry={terminationEntry}
                  decision={decisionFor(terminationEntry.t_index, coreValue)}
                  onOpenEntry={onOpenEntry}
                />
              </>
            ) : null}
          </section>
        );
      })}
    </div>
  );
}
