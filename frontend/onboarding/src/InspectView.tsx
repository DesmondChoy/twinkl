import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { TraceEventContract } from "./demoContracts";
import type { BwsResponse, ScoreBundle, ValueKey } from "./domain";
import OnboardingScoreInspection from "./OnboardingScoreInspection";
import InspectReferences, { type InspectReferenceKey } from "./InspectReferences";
import { northStarFraming } from "./northStar";
import { displayWeekRange } from "./displayFormatters";
import { isWeeklyEvent, weeklyRunContext } from "./weeklyRun";
import { coachPromptParts, savedCoachComparison, type CoachComparison } from "./coachComparison";
import "./northStarInspect.css";

type JsonRecord = Record<string, unknown>;

interface EventPresentation {
  label: string;
  component: string;
}

interface InspectViewProps {
  events: TraceEventContract[];
  currentWeekEventIds?: string[];
  currentJournalEntryIds?: string[];
  emptyActionLabel?: string;
  emptyMessage?: string;
  onboarding?: {
    confirmedValues: ValueKey[] | null;
    responses: BwsResponse[];
    scores: ScoreBundle;
    setOrder: number[];
  };
  onEmptyAction?: () => void;
  onToggleCalculation?: () => void;
  syntheticProfile?: boolean;
  selectedEventId: string | null;
  traceLabel: string;
  onReturn: () => void;
}

const EVENT_PRESENTATION: Record<string, EventPresentation> = {
  profile_confirmed: {
    label: "Profile confirmed",
    component: "Profile validation",
  },
  assessment_time_advanced: {
    label: "Simulated time changed",
    component: "Experience session clock",
  },
  journal_entry_submitted: {
    label: "Journal Entry submitted",
    component: "Journal Entry intake",
  },
  nudge_suppression_checked: {
    label: "Nudge suppression checked",
    component: "Nudge suppression rule",
  },
  nudge_decided: {
    label: "Nudge decided",
    component: "Nudge runtime",
  },
  nudge_generated: {
    label: "Nudge generated",
    component: "Nudge runtime",
  },
  weekly_review_requested: {
    label: "Weekly review requested",
    component: "Weekly Drift Reviewer",
  },
  weekly_review_completed: {
    label: "Weekly review completed",
    component: "Weekly Drift Reviewer",
  },
  drift_detected: {
    label: "Drift checked",
    component: "Drift Detector",
  },
  weekly_digest_built: {
    label: "Weekly Drift Detection output stored",
    component: "Weekly Drift Detection",
  },
  weekly_coach_generated: {
    label: "Coach Digest response generated",
    component: "Coach Digest",
  },
  north_star_reviewed: {
    label: "North Star Moment reviewed",
    component: "North Star Moment",
  },
  nudge_response_recorded: {
    label: "Nudge response recorded",
    component: "Journal Entry response",
  },
};

const STATUS_LABELS: Record<string, string> = {
  queued: "Queued",
  running: "Running",
  complete: "Complete",
  reused: "Reused",
  refused: "Refused",
  invalid: "Invalid",
  failed: "Failed",
};

const SOURCE_LABELS: Record<string, string> = {
  saved_replay: "Saved replay",
  live_run: "Live run",
};

const EVENT_REFERENCES: Partial<Record<TraceEventContract["event_type"], InspectReferenceKey>> = {
  nudge_decided: "nudge",
  weekly_review_completed: "reviewer",
  drift_detected: "detector",
  weekly_coach_generated: "coach",
  north_star_reviewed: "northStar",
};

type InspectFilter = "all" | "journal" | "reviewer" | "detector";

const FILTER_LABELS: Record<InspectFilter, string> = {
  all: "All steps",
  journal: "Journal Entries",
  reviewer: "Weekly Drift Reviewer",
  detector: "Weekly results",
};


function eventMatchesFilter(
  event: TraceEventContract,
  filter: InspectFilter,
): boolean {
  if (filter === "all") return true;
  if (filter === "journal") {
    return [
      "journal_entry_submitted",
      "assessment_time_advanced",
      "nudge_suppression_checked",
      "nudge_decided",
      "nudge_generated",
      "nudge_response_recorded",
    ].includes(event.event_type);
  }
  if (filter === "reviewer") {
    return [
      "weekly_review_requested",
      "weekly_review_completed",
    ].includes(event.event_type);
  }
  return [
    "drift_detected",
    "weekly_digest_built",
    "weekly_coach_generated",
    "north_star_reviewed",
  ].includes(event.event_type);
}

const SENSITIVE_KEYS = new Set([
  "access_token",
  "api_key",
  "authorization",
  "client_secret",
  "cookie",
  "env",
  "environment",
  "headers",
  "openai_api_key",
  "anthropic_api_key",
  "password",
  "proxy_authorization",
  "refresh_token",
  "secret",
  "set_cookie",
  "x_api_key",
]);

function record(value: unknown): JsonRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : null;
}

function array(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function string(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function normalizedKey(key: string): string {
  return key.toLowerCase().replaceAll("-", "_");
}

function redactSensitiveFields(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redactSensitiveFields);
  const item = record(value);
  if (!item) return value;
  return Object.fromEntries(
    Object.entries(item).map(([key, child]) => [
      key,
      SENSITIVE_KEYS.has(normalizedKey(key))
        ? "[redacted]"
        : redactSensitiveFields(child),
    ]),
  );
}

function titleCase(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function countLabel(
  count: number,
  singular: string,
  plural = `${singular}s`,
): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

function eventSummary(
  event: TraceEventContract,
  currentJournalEntryIds: Set<string> | null,
): string {
  const details = record(event.details) ?? {};
  const safeError = record(event.error);
  const errorMessage = string(safeError?.message);
  if (errorMessage) return errorMessage;

  switch (event.event_type) {
    case "north_star_reviewed": {
      const result = record(details.record);
      const selected = record(result?.selected);
      const reason = string(result?.reason)?.replaceAll("_", " ") ?? "No reason recorded";
      return selected && result?.status === "complete"
        ? `${titleCase(string(result.mode) ?? "selected")} · ${string(selected.date) ?? "date unavailable"} · ${string(result.value_phrase) ?? string(result.core_value) ?? "Core Value"}`
        : `${titleCase(string(result?.status) ?? "unavailable")} · ${reason}`;
    }
    case "nudge_response_recorded":
      return "Recorded when this response became available for later review";
    case "profile_confirmed": {
      const profile = record(details.profile);
      const coreValues = array(profile?.top_values);
      return `${countLabel(coreValues.length, "Core Value")} confirmed`;
    }
    case "journal_entry_submitted": {
      const entry = record(details.journal_entry);
      const tIndex = typeof entry?.t_index === "number" ? entry.t_index : null;
      const journalEntryId = string(entry?.journal_entry_id);
      const savedLabel = tIndex === null
        ? "Journal Entry saved"
        : `Journal Entry ${tIndex + 1} saved`;
      return journalEntryId
        && currentJournalEntryIds
        && !currentJournalEntryIds.has(journalEntryId)
        ? `${savedLabel} · Removed from current Experience`
        : savedLabel;
    }
    case "nudge_suppression_checked":
      if (details.policy_applied === false) {
        return `Saved nudge history; current spacing rule would ${details.suppressed === true ? "suppress" : "allow"} a nudge`;
      }
      return details.suppressed === true
        ? "Nudge suppressed by the anti-annoyance rule"
        : "Nudge allowed after the suppression check";
    case "nudge_decided": {
      if (details.should_nudge !== true) return "No nudge requested";
      const category = string(details.category);
      return category ? `Nudge requested · ${titleCase(category)}` : "Nudge requested";
    }
    case "nudge_generated": {
      const nudge = record(details.nudge);
      return string(nudge?.text) ?? "Reflective question ready";
    }
    case "weekly_review_requested": {
      const request = record(details.request);
      return `${countLabel(array(request?.history).length, "Journal Entry", "Journal Entries")} sent for review`;
    }
    case "weekly_review_completed": {
      const receipt = record(details.receipt);
      const decisions = array(receipt?.decisions);
      const conflicts = decisions.filter(
        (decision) => record(decision)?.verdict === "conflict",
      ).length;
      return conflicts > 0
        ? `${countLabel(decisions.length, "Weekly Drift Reviewer Decision")} · ${countLabel(conflicts, "Conflict")}`
        : countLabel(decisions.length, "Weekly Drift Reviewer Decision");
    }
    case "drift_detected": {
      const result = record(details.result);
      const deliveryState = string(result?.delivery_state);
      const driftCount = array(result?.drifts).length;
      return `${deliveryState ? titleCase(deliveryState) : "No active"} · ${countLabel(driftCount, "Drift")} confirmed`;
    }
    case "weekly_digest_built": {
      const digest = record(details.digest);
      const responseMode = string(digest?.response_mode);
      return responseMode
        ? `Weekly Drift Detection output ready · ${titleCase(responseMode)}`
        : "Weekly Drift Detection output ready";
    }
    case "weekly_coach_generated":
      return ["complete", "reused"].includes(event.status)
        ? "Coach Digest response and question ready"
        : "Coach Digest response unavailable";
    case "assessment_time_advanced": {
      const action = string(details.action);
      const currentDate = string(details.current_date);
      return action === "close_week"
        ? `Week closed · moved to ${currentDate ?? "the next Monday"}`
        : `Moved to ${currentDate ?? "the next day"}`;
    }
    default:
      return STATUS_LABELS[event.status] ?? titleCase(event.status);
  }
}

function eventPresentation(eventType: string): EventPresentation {
  return EVENT_PRESENTATION[eventType] ?? {
    label: titleCase(eventType),
    component: "Trace event",
  };
}

function formatDuration(durationMs: number | null): string {
  if (durationMs === null) return "Duration pending";
  if (durationMs < 1_000) return `${durationMs} ms`;
  return `${(durationMs / 1_000).toFixed(2)} s`;
}

function JsonBlock({ label, value }: { label: string; value: unknown }) {
  return (
    <section className="inspect-detail inspect-detail--wide">
      <h3>{label}</h3>
      <pre aria-label={label}>
        <code>{JSON.stringify(redactSensitiveFields(value), null, 2)}</code>
      </pre>
    </section>
  );
}

function TextBlock({ label, value }: { label: string; value: string }) {
  return (
    <section className="inspect-detail inspect-detail--wide">
      <h3>{label}</h3>
      <pre aria-label={label}>
        <code>{value}</code>
      </pre>
    </section>
  );
}

function TraceFacts({ event }: { event: TraceEventContract }) {
  return (
    <dl className="trace-facts">
      <div>
        <dt>Event ID</dt>
        <dd><code>{event.event_id}</code></dd>
      </div>
      <div>
        <dt>Parent event</dt>
        <dd><code>{event.parent_event_id ?? "Root event"}</code></dd>
      </div>
      <div>
        <dt>Started</dt>
        <dd><code>{event.started_at}</code></dd>
      </div>
      <div>
        <dt>Completed</dt>
        <dd><code>{event.completed_at ?? "Pending"}</code></dd>
      </div>
      <div>
        <dt>Duration</dt>
        <dd><code>{formatDuration(event.duration_ms)}</code></dd>
      </div>
      <div>
        <dt>Session ID</dt>
        <dd><code>{event.session_id}</code></dd>
      </div>
      <div>
        <dt>Input hash</dt>
        <dd><code>{event.input_hash}</code></dd>
      </div>
    </dl>
  );
}

function parsedRecord(value: unknown): JsonRecord | null {
  if (typeof value !== "string") return record(value);
  try {
    return record(JSON.parse(value));
  } catch {
    return null;
  }
}

interface NorthStarReviewEvidence {
  request: JsonRecord;
  input: JsonRecord | null;
  attempts: JsonRecord[];
  receipt: JsonRecord;
  label: string;
}

function northStarReviewEvidence(result: JsonRecord): NorthStarReviewEvidence[] {
  const experiment = record(result.experiment);
  const reviews = array(experiment ? experiment.receipts : result.reviews);
  return reviews.flatMap((value, index) => {
    const receipt = record(value);
    const request = record(experiment ? receipt?.request : receipt?.provider_request);
    if (!receipt || !request) return [];
    const input = parsedRecord(request.prompt);
    return [{
      request,
      input,
      receipt,
      attempts: array(experiment ? receipt.attempts : receipt.provider_attempts)
        .flatMap((attempt) => record(attempt) ? [record(attempt)!] : []),
      label: string(input?.user_phrase) ?? string(receipt.value_phrase)
        ?? string(input?.core_value) ?? `Core Value ${index + 1}`,
    }];
  });
}

function NorthStarAssessment({ review, selected }: {
  review: NorthStarReviewEvidence;
  selected: JsonRecord | null;
}) {
  const lastAttempt = review.attempts.at(-1);
  const response = parsedRecord(lastAttempt?.raw_text) ?? record(lastAttempt?.parsed_output);
  const assessments = array(response?.results).flatMap((item) => record(item) ? [record(item)!] : []);
  return (
    <section className="nsm-inspect__review">
      <h5>{review.label}</h5>
      <p>{lastAttempt
        ? `Last recorded attempt: ${titleCase(string(lastAttempt.status) ?? "unknown")}.`
        : "No provider attempt is recorded for this prepared request."}</p>
      {lastAttempt && lastAttempt.status !== "completed" ? (
        <p>This attempt did not produce an accepted response. Any response below is recorded evidence, not an accepted selection.</p>
      ) : null}
      {assessments.length === 0 ? <p>No readable source assessment is available in the recorded response.</p> : null}
      {assessments.map((assessment, index) => {
        const isSelected = selected !== null && selected.entry_id === assessment.entry_id
          && selected.evidence_quote === assessment.evidence_quote;
        return (
          <details className="nsm-inspect__source" key={`${String(assessment.entry_id)}:${index}`} open={isSelected}>
            <summary>
              <span>{string(assessment.entry_id) ?? `Source ${index + 1}`}</span>
              <span>{isSelected ? "Selected quotation" : titleCase(string(assessment.reason_code) ?? "No reason recorded")}</span>
            </summary>
            <dl className="nsm-inspect__assessments">
              <div><dt>What did the writer do?</dt><dd>{string(assessment.action_assessment) ?? "Not recorded"}</dd></div>
              <div><dt>How does the action relate to this Core Value?</dt><dd>{string(assessment.value_assessment) ?? "Not recorded"}</dd></div>
              <div><dt>Does this source also show opposing behavior?</dt><dd>{string(assessment.conflict_assessment) ?? "Not recorded"}</dd></div>
              <div><dt>AI reason</dt><dd>{titleCase(string(assessment.reason_code) ?? "Not recorded")}</dd></div>
            </dl>
            {string(assessment.evidence_quote) ? <blockquote>{string(assessment.evidence_quote)}</blockquote> : null}
          </details>
        );
      })}
      <details className="inspect-technical">
        <summary>Every provider attempt and exact response</summary>
        {review.attempts.map((attempt, index) => (
          <div key={index}>
            {typeof attempt.raw_text === "string" ? (
              <TextBlock label={`Exact response · ${review.label} · attempt ${index + 1}`} value={attempt.raw_text} />
            ) : null}
            <JsonBlock label={`Provider receipt · ${review.label} · attempt ${index + 1}`} value={attempt} />
          </div>
        ))}
        {array(review.receipt.validation_errors).length > 0 ? (
          <JsonBlock label={`Response validation errors · ${review.label}`} value={review.receipt.validation_errors} />
        ) : null}
      </details>
    </section>
  );
}

function CoachComparisonInspection({ comparison }: { comparison: CoachComparison }) {
  const common = coachPromptParts(comparison.without_north_star.base_prompt)!;
  return (
    <section className="inspect-coach-comparison" aria-label="Coach Digest prompt and response comparison">
      <h3>Coach Digest with and without North Star Moment</h3>
      <p>Two saved responses for the same Persona and week. Between the initial requests, only <code>north_star_context</code> changes. Opening Inspect or switching the reflection makes no provider call.</p>
      <p>AI-generated synthetic demonstration evidence; these differences do not establish user benefit.</p>
      <details className="inspect-technical">
        <summary>Shared instructions and unchanged weekly input</summary>
        <TextBlock label="Exact common initial instructions" value={common.instructions} />
        <JsonBlock label="Unchanged weekly input" value={Object.fromEntries(Object.entries(common.input)
          .filter(([key]) => key !== "north_star_context"))} />
      </details>
      {([ ["without_north_star", "Without North Star Moment"], ["with_north_star", "With North Star Moment"] ] as const)
        .map(([key, label]) => {
          const arm = comparison[key];
          const input = coachPromptParts(arm.prompt)!.input;
          const retried = arm.base_prompt !== arm.prompt;
          return (
            <section className="inspect-coach-comparison__arm" aria-label={label} key={key}>
              <h4>{label}</h4>
              <JsonBlock label={`${label}: north_star_context`} value={input.north_star_context} />
              <details className="inspect-technical" open>
                <summary>{label}: exact prompt and associated response</summary>
                <TextBlock label={`${label}: exact accepted prompt`} value={arm.prompt} />
                {retried ? (
                  <details className="inspect-technical">
                    <summary>Initial request and repair feedback</summary>
                    <p>The accepted response below belongs to the accepted prompt above, which includes repair feedback.</p>
                    <TextBlock label={`${label}: exact initial prompt`} value={arm.base_prompt} />
                    <JsonBlock label={`${label}: repair requirements`} value={arm.repair_requirements} />
                  </details>
                ) : null}
                <section className="inspect-coach-comparison__response" aria-label={`${label}: associated response`}>
                  <h5>Associated Coach Digest response</h5>
                  <p>{arm.narrative.weekly_mirror}</p>
                  <p>{arm.narrative.tension_explanation}</p>
                  <p>{arm.narrative.reflective_question}</p>
                </section>
                <TextBlock label={`${label}: exact raw provider response`} value={arm.raw_output} />
              </details>
              <details className="inspect-technical">
                <summary>{label}: validation and generation receipt</summary>
                <JsonBlock label={`${label}: validation`} value={arm.validation} />
                <JsonBlock label={`${label}: generation receipt`} value={Object.fromEntries(Object.entries(arm)
                  .filter(([field]) => !["narrative", "validation", "base_prompt", "prompt", "raw_output"].includes(field)))} />
              </details>
            </section>
          );
        })}
    </section>
  );
}

function NorthStarInspection({ event, comparison }: { event: TraceEventContract; comparison: CoachComparison | null }) {
  const result = record(event.details.record);
  if (!result) return null;
  const selected = result.status === "complete" && event.validation?.valid !== false
    ? record(result.selected) : null;
  const selectedSource = array(result.sources).map(record)
    .find((source) => source?.entry_id === selected?.entry_id);
  const experiment = record(result.experiment);
  const reviews = northStarReviewEvidence(result);
  const framing = selected ? northStarFraming(result.mode) : null;
  const outcome = selected ? "An exact quotation was selected."
    : result.status === "complete" && result.reason === "no_supportive_source"
      ? "The review completed without a suitable quotation."
      : result.status === "not_eligible" ? "This weekly result was not eligible for a North Star Moment."
        : result.status === "pending" ? "No completed North Star Moment review is available."
          : "No accepted North Star Moment is available from this record.";
  return (
    <section className="inspect-north-star nsm-inspect" aria-labelledby={`moment-inspect-${event.event_id}`}>
      <h3 id={`moment-inspect-${event.event_id}`}>How this North Star Moment was derived</h3>
      <p className="nsm-inspect__outcome">{outcome}</p>
      <p>{experiment
        ? "Saved replay: these are the original recorded provider requests and responses. Opening Inspect does not call a model."
        : "These requests and responses belong to this Experience session. Opening Inspect does not run another review."}</p>
      <dl className="trace-facts">
        <div><dt>Eligibility outcome</dt><dd>{titleCase(string(result.reason) ?? "unavailable")}</dd></div>
        <div><dt>Recorded status</dt><dd>{titleCase(string(result.status) ?? "unavailable")}</dd></div>
        <div><dt>Mode</dt><dd>{string(result.mode) ?? "No moment selected"}</dd></div>
        <div><dt>Core Value</dt><dd>{string(result.value_phrase) ?? string(result.core_value) ?? "None"}</dd></div>
        {selected ? (
          <>
            <div><dt>Journal Entry</dt><dd><code>{string(selected.entry_id)}</code> · {string(selected.date)}</dd></div>
            <div><dt>Quote source</dt><dd>{titleCase(string(selected.quote_source) ?? "unavailable")}</dd></div>
          </>
        ) : null}
      </dl>
      {typeof selected?.evidence_quote === "string" ? (
        <blockquote aria-label="Exact selected quotation">{selected.evidence_quote}</blockquote>
      ) : null}

      <section className="nsm-inspect__step">
        <h4>1. Choose eligible writing</h4>
        <p>The application chooses the Core Value and source window before the AI review. Active Drift uses writing from before its onset; otherwise, eligible writing runs through the reviewed week. Sources stay in newest-first order, with no embedding ranking.</p>
        <p>The model receives the Core Value phrase, its approved definition, and the complete eligible Journal Entries and user nudge responses in each request below. Persona biographies and Weekly Drift Reviewer Decisions are not semantic evidence for this review.</p>
        <p>Review cutoff: <code>{string(result.cutoff_at) ?? "Not recorded"}</code>{result.onset_date ? ` · Active Drift onset: ${String(result.onset_date)}` : ""}</p>
        {reviews.length === 0 ? (
          <p>{result.status === "not_eligible"
            ? "No source-review request was needed for this ineligible result."
            : "No source-review request is preserved in this record."}</p>
        ) : null}
        {reviews.map((review, index) => {
          const sources = array(review.input?.sources).flatMap((source) => record(source) ? [record(source)!] : []);
          return (
            <details className="inspect-technical" key={index}>
              <summary>{review.label} · {countLabel(sources.length, "Journal Entry", "Journal Entries")} · full request writing</summary>
              <p>{review.attempts.length > 0 ? "Writing in the recorded provider request." : "Prepared writing; no provider attempt is recorded."}</p>
              <p><strong>Approved definition:</strong> {string(review.input?.approved_definition) ?? "Not readable in the recorded request"}</p>
              {sources.map((source, sourceIndex) => (
                <section className="nsm-inspect__writing" key={`${String(source.entry_id)}:${sourceIndex}`}>
                  <h5>{sourceIndex + 1}. {string(source.entry_id) ?? "Source identifier unavailable"}</h5>
                  <p className="nsm-inspect__source-label">Journal Entry</p>
                  <p className="nsm-inspect__full-text">{string(source.journal_entry) ?? "No Journal Entry text supplied."}</p>
                  {typeof source.nudge_response === "string" ? (
                    <><p className="nsm-inspect__source-label">User nudge response</p><p className="nsm-inspect__full-text">{source.nudge_response}</p></>
                  ) : null}
                </section>
              ))}
            </details>
          );
        })}
      </section>

      <section className="nsm-inspect__step">
        <h4>2. Read the exact prompts</h4>
        <p>The system prompt asks for factual assessments and an exact quotation for each suitable source. The user message supplies the writing; the response schema defines the required output.</p>
        {reviews.length === 0 ? <p>No prompt is preserved because this record contains no source-review request.</p> : null}
        {reviews.map((review, index) => {
          const attempt = review.attempts.at(-1);
          return (
            <details className="inspect-technical" key={index}>
              <summary>{review.label} · exact prompts and response schema</summary>
              <dl className="trace-facts">
                <div><dt>Requested model</dt><dd>{string(attempt?.requested_model) ?? "No provider attempt recorded"}</dd></div>
                <div><dt>Actual model</dt><dd>{string(attempt?.actual_model) ?? "Not recorded"}</dd></div>
                <div><dt>Reasoning effort</dt><dd>{string(attempt?.reasoning_effort) ?? "Not recorded"}</dd></div>
                <div><dt>Recorded attempts</dt><dd>{review.attempts.length}</dd></div>
              </dl>
              {typeof review.request.system === "string" ? <TextBlock label={`Exact system prompt · ${review.label}`} value={review.request.system} /> : <p>The system prompt is missing.</p>}
              {typeof review.request.prompt === "string" ? <TextBlock label={`Exact user message · ${review.label}`} value={review.request.prompt} /> : <p>The user message is missing.</p>}
              <JsonBlock label={`Exact response schema · ${review.label}`} value={review.request.schema ?? null} />
              <JsonBlock label={`Complete input token receipt · ${review.label}`} value={review.receipt.count_receipt ?? review.receipt.input_receipt ?? null} />
            </details>
          );
        })}
      </section>

      <section className="nsm-inspect__step">
        <h4>3. Follow the AI assessment</h4>
        <p>The recorded model response answers three questions for every source. These are the runtime AI assessments used to derive the Moment, not human validation or a separate benchmark evaluator’s verdict.</p>
        {reviews.length === 0 ? <p>No AI source assessment is recorded for this result.</p> : null}
        {reviews.map((review, index) => <NorthStarAssessment key={index} review={review}
          selected={review.input?.core_value === result.core_value ? selected : null} />)}
      </section>

      <section className="nsm-inspect__step">
        <h4>4. Apply the selection rule and source checks</h4>
        <p>The model assesses sources; application code chooses the quotation. An observable choice becomes eligible, ambiguous or insufficient writing produces an abstention, and other reasons exclude the source. Same-value opposing behavior excludes the entire source.</p>
        <p>For Active Drift, the application reviews the Core Value with the longest current run, using confirmed Profile order to break ties. Otherwise, it reviews Core Values in confirmed Profile order and prefers a current-week supportive source. It then takes the first eligible source in that value and source order. It does not rank quotations as “best.”</p>
        {selected ? <p>The selected source is <code>{string(selected.entry_id)}</code> for {string(result.value_phrase) ?? string(result.core_value)}.</p> : <p>No quotation from this record is added to the Coach Digest.</p>}
        <p>Code checks source membership, identity, chronology, complete responses, and exact continuous quotation matching. It cannot prove the AI’s interpretation is correct.</p>
        <details className="inspect-technical">
          <summary>Recorded application validation and source checks</summary>
          <p>These are recorded check results and source bindings. A valid record can describe an intentional no-card outcome.</p>
          <JsonBlock label="Source checks" value={{
            validation: event.validation,
            evidence: result.validation_evidence ?? [],
            owner_id: result.owner_id,
            profile_ref: result.profile_ref,
            input_hash: result.input_hash,
            week_start: result.week_start,
            week_end: result.week_end,
            cutoff_at: result.cutoff_at,
            source_available_at: selected?.quote_source === "nudge_response"
              ? selectedSource?.response_available_at : selectedSource?.available_at,
            onset_available_at: result.onset_available_at,
            source_ids: result.source_ids,
            retryable: result.retryable,
          }} />
        </details>
        {framing ? (
          <div className="nsm-inspect__composition">
            <h5>Where it appears in Coach Digest</h5>
            <p>{comparison
              ? "The saved demo starts with the response generated without North Star Moment. Switching to the with-context response replaces both narrative paragraphs and its reflective question, and displays this exact quotation after the complete response. The selected quotation and its full source were supplied to the with-context request; both prompts and responses are recorded in the Coach Digest comparison."
              : "The full selected quotation is inserted as a North Star Moment passage after the Coach Digest narrative and reflective question. Its wording is preserved; the selected text is not sent back to rewrite the narrative."}</p>
            <p>Introduction: {framing}</p>
          </div>
        ) : null}
      </section>
    </section>
  );
}

function EventDetails({ event, comparison }: { event: TraceEventContract; comparison: CoachComparison | null }) {
  const reference = event.event_type === "weekly_coach_generated" && comparison
    ? "coachComparison" : EVENT_REFERENCES[event.event_type];
  const disclosure = (
    label: string,
    content: ReactNode,
  ) => (
    <details className="inspect-technical">
      <summary>{label}</summary>
      {content}
    </details>
  );

  return (
    <div
      className="trace-event__details"
      data-testid={`trace-details-${event.event_id}`}
    >
      <h2 className="sr-only">
        {eventPresentation(event.event_type).label} details
      </h2>
      {event.error !== null ? (
        <JsonBlock label="Safe error" value={event.error} />
      ) : null}
      {reference ? <InspectReferences reference={reference} /> : null}
      {event.event_type === "north_star_reviewed" ? <NorthStarInspection event={event} comparison={comparison} /> : null}
      {event.event_type === "weekly_coach_generated" && comparison ? <CoachComparisonInspection comparison={comparison} /> : null}
      <details className="inspect-technical inspect-technical--group">
        <summary>Technical details</summary>
        <p className="inspect-technical__help">
          These fields identify the recorded run and support reproduction.
        </p>
        <TraceFacts event={event} />
        {event.input_refs.length > 0 ? (
          disclosure(
            "Input references",
            <JsonBlock label="Input references" value={event.input_refs} />,
          )
        ) : null}
        {event.model_contract !== null ? (
          disclosure(
            "Model contract",
            <JsonBlock label="Model contract" value={event.model_contract} />,
          )
        ) : null}
        {event.prompt !== null ? (
          disclosure(
            "Prompt",
            <TextBlock label="Exact rendered prompt" value={event.prompt} />,
          )
        ) : null}
        {event.raw_response !== null ? (
          disclosure(
            "Raw response",
            <JsonBlock label="Raw provider response" value={event.raw_response} />,
          )
        ) : null}
        {event.validation !== null ? (
          disclosure(
            "Validation",
            <JsonBlock label="Validation" value={event.validation} />,
          )
        ) : null}
        {disclosure(
          "Effective result",
          <JsonBlock label="Effective result" value={event.details} />,
        )}
        {event.result_refs.length > 0 ? (
          disclosure(
            "Result references",
            <JsonBlock label="Result references" value={event.result_refs} />,
          )
        ) : null}
      </details>
    </div>
  );
}

export default function InspectView({
  events,
  currentWeekEventIds,
  currentJournalEntryIds,
  emptyActionLabel,
  emptyMessage = "No backend work has been recorded for this Experience yet.",
  onboarding,
  onEmptyAction,
  onToggleCalculation,
  syntheticProfile = false,
  selectedEventId,
  traceLabel,
  onReturn,
}: InspectViewProps) {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const eventRefs = useRef(new Map<string, HTMLElement>());
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(
    () => new Set(selectedEventId ? [selectedEventId] : []),
  );
  const [activeFilter, setActiveFilter] = useState<InspectFilter>("all");
  const [pendingEventFocus, setPendingEventFocus] = useState<string | null>(null);
  const eventNumbers = useMemo(
    () => new Map(events.map((event, index) => [event.event_id, index + 1])),
    [events],
  );
  const coachComparisons = useMemo(() => new Map(events.flatMap((event) => {
    const comparison = savedCoachComparison(event);
    return comparison ? [[event.event_id, comparison] as const] : [];
  })), [events]);
  const currentJournalEntryIdSet = useMemo(
    () => currentJournalEntryIds
      ? new Set(currentJournalEntryIds)
      : null,
    [currentJournalEntryIds],
  );
  const selectedEvent = !onboarding && selectedEventId
    ? events.find((event) => event.event_id === selectedEventId) ?? null
    : null;
  const currentWeekEventIdSet = useMemo(
    () => !onboarding && currentWeekEventIds ? new Set(currentWeekEventIds) : null,
    [currentWeekEventIds, onboarding],
  );
  const currentEvents = useMemo(
    () => currentWeekEventIdSet
      ? events.filter((event) => currentWeekEventIdSet.has(event.event_id))
      : events,
    [currentWeekEventIdSet, events],
  );
  const historyEvents = useMemo(
    () => currentWeekEventIdSet
      ? events.filter((event) => !currentWeekEventIdSet.has(event.event_id))
      : [],
    [currentWeekEventIdSet, events],
  );
  const filteredCurrentEvents = currentEvents.filter((event) =>
    eventMatchesFilter(event, onboarding ? "all" : activeFilter)
  );
  const filteredHistoryEvents = historyEvents.filter((event) =>
    eventMatchesFilter(event, onboarding ? "all" : activeFilter)
  );
  const focusedWeeklyEvent = selectedEvent && isWeeklyEvent(selectedEvent)
    ? selectedEvent
    : currentWeekEventIdSet
      ? [...currentEvents].reverse().find(isWeeklyEvent)
      : undefined;
  const focusedContext = useMemo(() => focusedWeeklyEvent
    ? weeklyRunContext(events, focusedWeeklyEvent.event_id) : null, [events, focusedWeeklyEvent]);
  const weeklyFocus = !onboarding && focusedContext !== null;
  const latestWeeklyEvent = (eventType: string) =>
    (selectedEvent?.event_type === eventType ? selectedEvent : null)
    ?? [...(focusedContext?.events ?? [])].reverse().find((event) => event.event_type === eventType)
    ?? null;
  const reviewerEvent = latestWeeklyEvent("weekly_review_completed");
  const driftEvent = latestWeeklyEvent("drift_detected");
  const coachEvent = latestWeeklyEvent("weekly_coach_generated");
  const northStarEvent = selectedEvent?.event_type === "north_star_reviewed"
    ? selectedEvent : latestWeeklyEvent("north_star_reviewed");
  const reviewerModel = record(reviewerEvent?.model_contract);
  const reviewerModelName = string(reviewerModel?.model);
  const reviewerEffort = string(reviewerModel?.reasoning_effort);

  useEffect(() => {
    if (onboarding || !selectedEventId || !eventRefs.current.has(selectedEventId)) {
      headingRef.current?.focus({ preventScroll: true });
      return;
    }
    if (weeklyFocus && selectedEvent?.event_type !== "north_star_reviewed") {
      headingRef.current?.focus({ preventScroll: true });
      return;
    }
    setExpandedEvents((current) => {
      if (current.has(selectedEventId)) return current;
      return new Set([...current, selectedEventId]);
    });
    const target = eventRefs.current.get(selectedEventId);
    target?.focus({ preventScroll: true });
    target?.scrollIntoView?.({ block: "center" });
  }, [onboarding, selectedEventId, selectedEvent?.event_type, weeklyFocus]);

  const setEventExpanded = (eventId: string, open: boolean) => {
    setExpandedEvents((current) => {
      if (current.has(eventId) === open) return current;
      const next = new Set(current);
      if (open) next.add(eventId);
      else next.delete(eventId);
      return next;
    });
  };

  useEffect(() => {
    if (!pendingEventFocus) return;
    const target = eventRefs.current.get(pendingEventFocus);
    const history = target?.closest<HTMLDetailsElement>(".inspect-history");
    if (history) history.open = true;
    target?.focus({ preventScroll: true });
    target?.scrollIntoView?.({ block: "start" });
    setPendingEventFocus(null);
  }, [pendingEventFocus]);

  const renderTimeline = (
    displayedEvents: TraceEventContract[],
    label: string,
  ) => (
    <ol className="inspect-timeline" aria-label={label}>
      {displayedEvents.map((event) => {
        const index = (eventNumbers.get(event.event_id) ?? 1) - 1;
        const presentation = eventPresentation(event.event_type);
        const status = STATUS_LABELS[event.status] ?? titleCase(event.status);
        const parentNumber = event.parent_event_id
          ? eventNumbers.get(event.parent_event_id)
          : null;
        const isSelected = !onboarding && event.event_id === selectedEventId;
        const isExpanded = expandedEvents.has(event.event_id);
        const showStatus = !["complete", "reused"].includes(event.status);
        const comparison = event.event_type === "weekly_coach_generated" ? coachComparisons.get(event.event_id) ?? null
          : event.event_type === "north_star_reviewed" ? [...coachComparisons.values()].find((pair) =>
            pair.north_star_input_hash === event.input_hash && pair.persona_id === record(event.details.record)?.owner_id
            && pair.week_start === record(event.details.record)?.week_start
            && pair.week_end === record(event.details.record)?.week_end) ?? null : null;
        return (
          <li className="trace-event" key={event.event_id}>
            <span className="trace-event__node" aria-hidden="true">
              {String(index + 1).padStart(2, "0")}
            </span>
            <details
              className="trace-event__card"
              data-selected={isSelected ? "true" : undefined}
              open={isExpanded}
              onToggle={(toggleEvent) =>
                setEventExpanded(event.event_id, toggleEvent.currentTarget.open)}
            >
              <summary
                ref={(node) => {
                  if (node) eventRefs.current.set(event.event_id, node);
                  else eventRefs.current.delete(event.event_id);
                }}
                aria-current={isSelected ? "true" : undefined}
                aria-label={`Event ${index + 1}: ${presentation.label}`}
              >
                <span className="trace-event__copy">
                  <span className="trace-event__component">
                    {presentation.component}
                  </span>
                  <span className="trace-event__name">{presentation.label}</span>
                  <span className="trace-event__result">
                    {eventSummary(event, currentJournalEntryIdSet)}
                  </span>
                </span>
                <span className="trace-event__aside">
                  {showStatus ? (
                    <span className={`trace-chip trace-chip--status-${event.status}`}>
                      {status}
                    </span>
                  ) : null}
                  <span className="trace-event__disclosure" aria-hidden="true">+</span>
                </span>
                <span className="trace-event__parent">
                  {parentNumber
                    ? `After event ${String(parentNumber).padStart(2, "0")}`
                    : "First recorded step"}
                </span>
              </summary>
              {isExpanded ? <EventDetails event={event} comparison={comparison} /> : null}
            </details>
          </li>
        );
      })}
    </ol>
  );

  return (
    <div
      className={`stage stage--inspect${
        weeklyFocus ? " stage--inspect-weekly" : ""
      }`}
    >
      <div className="inspect-intro" id="inspect-overview-section">
        <div>
          <p className="eyebrow">
            {onboarding ? "Assessment evidence" : "Inspect"}
          </p>
          <h1 ref={headingRef} tabIndex={-1}>
            {onboarding
              ? "See how each trade-off shaped this Profile."
              : "Follow the work, step by step."}
          </h1>
          <p className="lede">
            {onboarding
              ? "The assessment recorded one Most and one Least card in each question. Below, those 22 choices are followed into Schwartz scores, the ten-value Profile, and the exact phrases shown in Experience."
              : weeklyFocus
                ? "The focused result comes first. The complete event history follows."
                : "Each row is one recorded step. Open Technical details for exact inputs, prompts, and validation."}
          </p>
          {onToggleCalculation ? (
            <button className="inspect-run-link" type="button" onClick={onToggleCalculation}>
              {onboarding ? "View recorded events" : "View Profile calculation"}
            </button>
          ) : null}
          {syntheticProfile ? (
            <p>
              This Persona Profile is a synthetic projection. It does not
              represent a completed SVBWS assessment.
            </p>
          ) : null}
        </div>
        <button className="button button--quiet" type="button" onClick={onReturn}>
          Return to Experience
        </button>
      </div>

      <div
        className="inspect-overview"
        aria-label={onboarding ? "Assessment summary" : "Inspect summary"}
      >
        {onboarding ? (
          <>
            <span>{onboarding.responses.length} of 11 questions complete</span>
            <span>{onboarding.responses.length * 2} recorded selections</span>
            <span>
              {events.length > 0
                ? "Python validation recorded"
                : onboarding.confirmedValues !== null
                  ? "Python validation unavailable"
                  : "Python validation follows confirmation"}
            </span>
          </>
        ) : (
          <>
            <span>{traceLabel}</span>
            <span>{countLabel(events.length, "recorded event")}</span>
            <span>{currentWeekEventIdSet ? "Current week first" : "Recorded work"}</span>
          </>
        )}
      </div>

      {weeklyFocus ? (
        <section className="inspect-focus" aria-labelledby="inspect-focus-title">
          <div className="inspect-focus__heading">
            <div>
              <p className="eyebrow">Focused Inspect</p>
              <h2 id="inspect-focus-title">
                How Twinkl reached this result.
              </h2>
              {focusedContext?.week ? (
                <p>Week: {displayWeekRange(focusedContext.week.start, focusedContext.week.end)}</p>
              ) : null}
            </div>
            <details className="inspect-focus__technical">
              <summary>Technical details</summary>
              <p>
                {SOURCE_LABELS[
                  selectedEvent?.source ?? reviewerEvent?.source ?? ""
                ] ?? "Recorded run"}
                {reviewerModelName ? ` · ${reviewerModelName}` : ""}
                {reviewerEffort ? ` · reasoning effort ${reviewerEffort}` : ""}
                {reviewerEvent
                  ? ` · ${formatDuration(reviewerEvent.duration_ms)}`
                  : ""}
              </p>
            </details>
          </div>
          <p className="inspect-focus__evidence">
            {syntheticProfile
              ? "AI-reviewed synthetic development evidence · not human validation"
              : "AI-reviewed Journal Entries · not human validation"}
          </p>
          <ol className="inspect-focus__steps">
            <li>
              <span aria-hidden="true">1</span>
              <div>
                <strong>Weekly Drift Reviewer</strong>
                <p>
                  {reviewerEvent
                    ? eventSummary(reviewerEvent, currentJournalEntryIdSet)
                    : "No Weekly Drift Reviewer result is available."}
                </p>
                <small>Reviewed cumulative Journal Entry history.</small>
              </div>
            </li>
            <li>
              <span aria-hidden="true">2</span>
              <div>
                <strong>Drift Detector</strong>
                <p>
                  {driftEvent
                    ? eventSummary(driftEvent, currentJournalEntryIdSet)
                    : "No Drift Detector result is available."}
                </p>
                <small>Applied each Core Value rule independently.</small>
              </div>
            </li>
            {coachEvent ? (
              <li>
                <span aria-hidden="true">3</span>
                <div>
                  <strong>Coach Digest</strong>
                  <p>{eventSummary(coachEvent, currentJournalEntryIdSet)}</p>
                  <small>
                    Uses Weekly Drift Detection output to create a response and
                    question.
                  </small>
                  {savedCoachComparison(coachEvent) ? (
                    <button className="inspect-run-link" type="button" onClick={() => {
                      setActiveFilter("all");
                      setEventExpanded(coachEvent.event_id, true);
                      setPendingEventFocus(coachEvent.event_id);
                    }}>Inspect Coach Digest comparison</button>
                  ) : null}
                </div>
              </li>
            ) : null}
            {northStarEvent ? (
              <li>
                <span aria-hidden="true">{coachEvent ? "4" : "3"}</span>
                <div>
                  <strong>North Star Moment</strong>
                  <p>{eventSummary(northStarEvent, currentJournalEntryIdSet)}</p>
                  <small>Follow the full writing, exact prompts, AI assessments, and application selection checks.</small>
                  <button className="inspect-run-link" type="button" onClick={() => {
                    setActiveFilter("all");
                    setEventExpanded(northStarEvent.event_id, true);
                    setPendingEventFocus(northStarEvent.event_id);
                  }}>Inspect North Star Moment</button>
                </div>
              </li>
            ) : null}
          </ol>
          <p className="inspect-focus__more">
            The complete event history remains available below.
          </p>
        </section>
      ) : null}

      {onboarding ? (
        <OnboardingScoreInspection
          confirmedValues={onboarding.confirmedValues}
          responses={onboarding.responses}
          scores={onboarding.scores}
          setOrder={onboarding.setOrder}
        />
      ) : null}

      {selectedEvent ? (
        <div className="inspect-selection" data-testid="inspect-selection">
          <small>Focused from Experience</small>
          <p>
            Event {String(eventNumbers.get(selectedEvent.event_id)).padStart(2, "0")} ·{" "}
            {eventPresentation(selectedEvent.event_type).component}
          </p>
        </div>
      ) : !onboarding && selectedEventId ? (
        <div className="inspect-selection inspect-selection--missing" role="status">
          <small>Linked event unavailable</small>
          <p><code>{selectedEventId}</code> is not present in this trace.</p>
        </div>
      ) : null}

      <section
        className="backend-trace"
        id="inspect-events-section"
        aria-labelledby="backend-trace-title"
      >
        {onboarding ? (
          <header className="backend-trace__heading">
            <div>
              <p className="eyebrow">Python boundary</p>
              <h2 id="backend-trace-title">Validation and later work.</h2>
            </div>
            <p>
              Profile confirmation starts the Python Experience session. Later
              events show model calls, validation, and deterministic
              product logic.
            </p>
          </header>
        ) : currentWeekEventIdSet ? (
          <header className="backend-trace__heading">
            <div>
              <p className="eyebrow">Selected week</p>
              <h2 id="backend-trace-title">Current week first.</h2>
            </div>
            <p>
              Use the filters to follow Journal Entries, the Weekly Drift
              Reviewer, or weekly results including Coach Digest and North Star Moment.
            </p>
          </header>
        ) : (
          <h2 className="sr-only" id="backend-trace-title">
            Recorded events
          </h2>
        )}

        {!onboarding && events.length > 0 ? (
          <>
            <nav className="inspect-filters" aria-label="Filter Inspect events">
              {(Object.keys(FILTER_LABELS) as InspectFilter[]).map((filter) => (
                <button
                  type="button"
                  aria-pressed={activeFilter === filter}
                  onClick={() => setActiveFilter(filter)}
                  key={filter}
                >
                  {FILTER_LABELS[filter]}
                </button>
              ))}
            </nav>
            <p className="inspect-filter-count" role="status" aria-label="Filtered event count">
              {filteredCurrentEvents.length} of {countLabel(
                currentEvents.length,
                currentWeekEventIdSet ? "current week event" : "recorded event",
              )}
              {historyEvents.length > 0
                ? ` · ${filteredHistoryEvents.length} of ${countLabel(historyEvents.length, "earlier event")}`
                : ""}
            </p>
          </>
        ) : null}

        {events.length === 0 ? (
          <div className="inspect-empty">
            <p role="status">{emptyMessage}</p>
            {emptyActionLabel && onEmptyAction ? (
              <button
                className="button button--quiet"
                type="button"
                onClick={onEmptyAction}
              >
                {emptyActionLabel}
              </button>
            ) : null}
          </div>
        ) : (
          <>
            {filteredCurrentEvents.length > 0 ? (
              renderTimeline(
                filteredCurrentEvents,
                currentWeekEventIdSet ? "Current week events" : "Recorded events",
              )
            ) : (
              <p className="inspect-filter-empty" role="status">
                This week has no {FILTER_LABELS[activeFilter]} events.
              </p>
            )}
            {historyEvents.length > 0 ? (
              <details className="inspect-history">
                <summary>
                  <span>Complete Inspect history</span>
                  <small>{countLabel(filteredHistoryEvents.length, "event")}</small>
                </summary>
                {filteredHistoryEvents.length > 0 ? (
                  renderTimeline(filteredHistoryEvents, "Earlier events")
                ) : (
                  <p className="inspect-filter-empty">
                    Earlier weeks have no {FILTER_LABELS[activeFilter]} events.
                  </p>
                )}
              </details>
            ) : null}
          </>
        )}
      </section>
    </div>
  );
}
