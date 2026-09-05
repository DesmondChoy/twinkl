import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { createPortal } from "react-dom";
import {
  VALUES,
  type OnboardingProfile,
  type ValueKey,
} from "./domain";
import type {
  JournalEntryContract,
  TraceEventContract,
  WeeklyDriftReviewerDecisionContract,
} from "./demoContracts";

type JsonObject = Record<string, unknown>;
type CoreValueState =
  | "active_drift"
  | "no_active_drift"
  | "insufficient_evidence";

interface DriftStateExplanationProps {
  profile: OnboardingProfile;
  journalEntries: JournalEntryContract[];
  weeklyReviewerDecisions: WeeklyDriftReviewerDecisionContract[];
  reviewTraceEvents: TraceEventContract[];
  weekStart: string;
  weekEnd: string;
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

interface ReviewEvidence {
  decision: WeeklyDriftReviewerDecisionContract | undefined;
  event: TraceEventContract | undefined;
  output: JsonObject;
}

interface ReviewSelection {
  evidence: ReviewEvidence;
  buttonId: string;
}

interface ReviewPreviewState {
  evidence: ReviewEvidence;
  bounds: DOMRect;
}

function object(value: unknown): JsonObject | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function coreValueState(value: unknown): CoreValueState {
  return ["active_drift", "no_active_drift", "insufficient_evidence"].includes(
    String(value),
  )
    ? value as CoreValueState
    : "no_active_drift";
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

function reasonLabel(reasonCode: string | null | undefined): string {
  switch (reasonCode) {
    case "direct_behavior_or_choice":
      return "The Journal Entry contains a direct behavior or choice.";
    case "direct_aligned_or_neutral_behavior":
      return "The Journal Entry shows aligned or neutral behavior.";
    case "feeling_or_intent_only":
      return "The Journal Entry states a feeling or intent, but no clear action.";
    case "external_constraint":
      return "An external constraint prevents a clear decision.";
    case "missing_text":
      return "The Journal Entry does not contain enough text for a clear decision.";
    case "ambiguous":
      return "The Journal Entry is too ambiguous for a clear decision.";
    default:
      return "No separate justification was recorded.";
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

function reviewEvidenceFor(
  decision: WeeklyDriftReviewerDecisionContract | undefined,
  traceEvents: TraceEventContract[],
): ReviewEvidence {
  const event = decision
    ? traceEvents.find((candidate) => {
        if (candidate.event_type !== "weekly_review_completed") return false;
        const receipt = object(candidate.details.receipt);
        return (
          receipt?.week_start === decision.week_start
          && receipt.week_end === decision.week_end
        );
      })
    : undefined;
  const rawResponse = object(event?.raw_response);
  const parsed = object(rawResponse?.parsed);
  const assessments = Array.isArray(parsed?.assessments)
    ? parsed.assessments
    : [];
  const recordedOutput = decision
    ? assessments.find((item) => {
        const assessment = object(item);
        return (
          assessment?.t_index === decision.t_index
          && assessment.dimension === decision.core_value
        );
      })
    : null;
  const output = object(recordedOutput) ?? {
    verdict: decision?.verdict ?? "not_recorded",
    confidence: decision?.confidence ?? null,
    reason_code: decision?.reason_code ?? null,
    evidence_quote: decision?.evidence_quote ?? "",
  };
  return { decision, event, output };
}

function modelField(
  event: TraceEventContract | undefined,
  field: "model" | "reasoning_effort",
): string {
  const value = event?.model_contract?.[field];
  return typeof value === "string" ? value : "Not recorded";
}

function ReviewDetails({ evidence }: { evidence: ReviewEvidence }) {
  const { decision, event, output } = evidence;
  return (
    <div className="review-evidence">
      <dl className="review-evidence__contract">
        <div>
          <dt>Model</dt>
          <dd><code>{modelField(event, "model")}</code></dd>
        </div>
        <div>
          <dt>Reasoning effort</dt>
          <dd><code>{modelField(event, "reasoning_effort")}</code></dd>
        </div>
      </dl>
      <section>
        <h4>Recorded model output</h4>
        <pre><code>{JSON.stringify(output, null, 2)}</code></pre>
      </section>
      <section>
        <h4>Recorded justification</h4>
        <p>{reasonLabel(decision?.reason_code)}</p>
        {decision?.evidence_quote ? (
          <q>{decision.evidence_quote}</q>
        ) : null}
      </section>
      <p className="review-evidence__source">
        Saved Weekly Drift Reviewer evidence · not human validation
      </p>
    </div>
  );
}

function ReviewPreview({ preview }: { preview: ReviewPreviewState }) {
  const width = Math.min(380, window.innerWidth - 24);
  const left = preview.bounds.left > width + 24
    ? preview.bounds.left - width - 12
    : Math.min(window.innerWidth - width - 12, preview.bounds.right + 12);
  const top = Math.max(
    12,
    Math.min(preview.bounds.top - 32, window.innerHeight - 452),
  );
  const style: CSSProperties = { left, top, width };

  return createPortal(
    <aside
      className="review-evidence-preview"
      role="tooltip"
      style={style}
    >
      <p className="eyebrow">AI review details</p>
      <ReviewDetails evidence={preview.evidence} />
    </aside>,
    document.body,
  );
}

interface ReviewDetailsDrawerProps {
  selection: ReviewSelection | null;
  onClose: () => void;
}

function ReviewDetailsDrawer({
  selection,
  onClose,
}: ReviewDetailsDrawerProps) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!selection) return;
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
  }, [onClose, selection]);

  if (!selection) return null;

  return createPortal(
    <div
      className="review-evidence-drawer"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <aside
        className="review-evidence-drawer__panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="review-evidence-drawer-title"
      >
        <header>
          <div>
            <p className="eyebrow">Weekly Drift Reviewer Decision</p>
            <h2 id="review-evidence-drawer-title">AI review details</h2>
          </div>
          <button
            className="replay-entry-drawer__close"
            ref={closeRef}
            type="button"
            onClick={onClose}
          >
            Close
          </button>
        </header>
        <ReviewDetails evidence={selection.evidence} />
      </aside>
    </div>,
    document.body,
  );
}

interface DecisionEvidenceProps {
  entry: JournalEntryContract;
  decision: WeeklyDriftReviewerDecisionContract | undefined;
  reviewEvidence: ReviewEvidence;
  onOpenEntry: (entry: JournalEntryContract) => void;
  onOpenReview: (selection: ReviewSelection) => void;
  onPreview: (preview: ReviewPreviewState) => void;
  onClosePreview: () => void;
}

function DecisionEvidence({
  entry,
  decision,
  reviewEvidence,
  onOpenEntry,
  onOpenReview,
  onPreview,
  onClosePreview,
}: DecisionEvidenceProps) {
  const reviewButtonId = `review-${entry.journal_entry_id.replace(
    /[^a-zA-Z0-9_-]/g,
    "-",
  )}-${decision?.core_value ?? "unknown"}`;
  return (
    <article
      className="state-change__evidence"
      onMouseEnter={(event) =>
        onPreview({
          evidence: reviewEvidence,
          bounds: event.currentTarget.getBoundingClientRect(),
        })
      }
      onMouseLeave={onClosePreview}
      onFocus={(event) =>
        onPreview({
          evidence: reviewEvidence,
          bounds: event.currentTarget.getBoundingClientRect(),
        })
      }
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) {
          onClosePreview();
        }
      }}
    >
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
      <button
        className="state-change__review-action"
        id={reviewButtonId}
        type="button"
        onClick={() => {
          onClosePreview();
          onOpenReview({
            evidence: reviewEvidence,
            buttonId: reviewButtonId,
          });
        }}
      >
        AI review
      </button>
    </article>
  );
}

export default function DriftStateExplanation({
  profile,
  journalEntries,
  weeklyReviewerDecisions,
  reviewTraceEvents,
  weekStart,
  weekEnd,
  driftResult,
  onOpenEntry,
}: DriftStateExplanationProps) {
  const [preview, setPreview] = useState<ReviewPreviewState | null>(null);
  const [openReview, setOpenReview] = useState<ReviewSelection | null>(null);
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
  const closeReview = () => {
    const closing = openReview;
    setOpenReview(null);
    window.requestAnimationFrame?.(() => {
      if (closing) {
        document
          .getElementById(closing.buttonId)
          ?.focus({ preventScroll: true });
      }
    });
  };
  const renderDecisionEvidence = (
    entry: JournalEntryContract,
    decision: WeeklyDriftReviewerDecisionContract | undefined,
    key: string,
  ) => (
    <DecisionEvidence
      entry={entry}
      decision={decision}
      reviewEvidence={reviewEvidenceFor(decision, reviewTraceEvents)}
      onOpenEntry={onOpenEntry}
      onOpenReview={setOpenReview}
      onPreview={setPreview}
      onClosePreview={() => setPreview(null)}
      key={key}
    />
  );

  return (
    <>
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
          const currentDecisions = weeklyReviewerDecisions.filter(
            (decision) =>
              decision.core_value === coreValue
              && decision.week_start === weekStart
              && decision.week_end === weekEnd,
          );

          return (
            <section
              className={`state-change state-change--${state}`}
              key={coreValue}
            >
              <header>
                <span>{VALUES[coreValue as ValueKey]?.name ?? coreValue}</span>
                <strong>
                  {state === "active_drift" ? "Active Drift"
                    : state === "insufficient_evidence" ? "Insufficient Evidence"
                      : "No Active Drift"}
                </strong>
              </header>

              {state === "no_active_drift" ? (
                <>
                  <p className="state-change__summary">
                    No active Drift is confirmed at this cutoff. This does not
                    prove a positive change.
                  </p>
                  {currentDecisions.length > 0 ? (
                    <div className="state-change__evidence-pair">
                      {currentDecisions.flatMap((decision) => {
                        const entry = entriesByIndex.get(decision.t_index);
                        return entry
                          ? [renderDecisionEvidence(
                              entry,
                              decision,
                              `${coreValue}-${decision.t_index}`,
                            )]
                          : [];
                      })}
                    </div>
                  ) : null}
                </>
              ) : null}

              {state === "active_drift" && drift ? (
                <>
                  <p className="state-change__marker">Drift started here.</p>
                  <div className="state-change__evidence-pair">
                    {startIndices.flatMap((tIndex) => {
                      const entry = entriesByIndex.get(tIndex);
                      return entry
                        ? [renderDecisionEvidence(
                            entry,
                            decisionFor(tIndex, coreValue),
                            `${coreValue}-${tIndex}`,
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
                            ? [renderDecisionEvidence(
                                entry,
                                decisionFor(tIndex, coreValue),
                                `${coreValue}-${tIndex}`,
                              )]
                            : [];
                        })}
                      </div>
                    </>
                  ) : null}
                </>
              ) : null}

              {state === "insufficient_evidence" ? (
                <>
                  <p className="state-change__summary">
                    A review failure prevented a current claim, or an Abstain
                    or Journal Entry gap blocked recent Conflict evidence.
                  </p>
                  <div className="state-change__evidence-pair">
                    {currentDecisions.flatMap((decision) => {
                      const entry = entriesByIndex.get(decision.t_index);
                      return entry
                        ? [renderDecisionEvidence(
                            entry,
                            decision,
                            `${coreValue}-${decision.t_index}`,
                          )]
                        : [];
                    })}
                  </div>
                </>
              ) : null}
            </section>
          );
        })}
      </div>
      {preview ? <ReviewPreview preview={preview} /> : null}
      <ReviewDetailsDrawer selection={openReview} onClose={closeReview} />
    </>
  );
}
