import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { TraceEventContract } from "./demoContracts";
import type { BwsResponse, ScoreBundle } from "./domain";
import OnboardingScoreInspection from "./OnboardingScoreInspection";

type JsonRecord = Record<string, unknown>;

interface EventPresentation {
  label: string;
  component: string;
}

interface InspectViewProps {
  events: TraceEventContract[];
  currentJournalEntryIds?: string[];
  emptyActionLabel?: string;
  emptyMessage?: string;
  onboarding?: {
    confirmed: boolean;
    responses: BwsResponse[];
    scores: ScoreBundle;
    setOrder: number[];
  };
  onEmptyAction?: () => void;
  selectedEventId: string | null;
  traceLabel: string;
  onReturn: () => void;
}

const EVENT_PRESENTATION: Record<string, EventPresentation> = {
  profile_confirmed: {
    label: "Profile confirmed",
    component: "Profile validation",
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
    label: "Coach Digest generated",
    component: "Coach Digest",
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
      return "Coach Digest response and question ready";
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

function EventDetails({ event }: { event: TraceEventContract }) {
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
    </div>
  );
}

export default function InspectView({
  events,
  currentJournalEntryIds,
  emptyActionLabel,
  emptyMessage = "No backend work has been recorded for this Experience yet.",
  onboarding,
  onEmptyAction,
  selectedEventId,
  traceLabel,
  onReturn,
}: InspectViewProps) {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const eventRefs = useRef(new Map<string, HTMLElement>());
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(
    () => new Set(selectedEventId ? [selectedEventId] : []),
  );
  const eventNumbers = useMemo(
    () => new Map(events.map((event, index) => [event.event_id, index + 1])),
    [events],
  );
  const currentJournalEntryIdSet = useMemo(
    () => currentJournalEntryIds
      ? new Set(currentJournalEntryIds)
      : null,
    [currentJournalEntryIds],
  );
  const selectedEvent = selectedEventId
    ? events.find((event) => event.event_id === selectedEventId) ?? null
    : null;
  const sourceLabels = [...new Set(events.map((event) => SOURCE_LABELS[event.source]))];
  const weeklyFocus =
    selectedEvent?.event_type === "drift_detected"
    || selectedEvent?.event_type === "weekly_digest_built"
    || selectedEvent?.event_type === "weekly_coach_generated";
  const latestWeeklyEvent = (eventType: string) =>
    [...events].reverse().find((event) => event.event_type === eventType)
    ?? null;
  const reviewerEvent = latestWeeklyEvent("weekly_review_completed");
  const driftEvent = latestWeeklyEvent("drift_detected");
  const coachEvent = latestWeeklyEvent("weekly_coach_generated");
  const reviewerModel = record(reviewerEvent?.model_contract);
  const reviewerModelName = string(reviewerModel?.model);
  const reviewerEffort = string(reviewerModel?.reasoning_effort);

  useEffect(() => {
    if (!selectedEventId || !eventRefs.current.has(selectedEventId)) {
      headingRef.current?.focus({ preventScroll: true });
      return;
    }
    if (weeklyFocus) {
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
  }, [selectedEventId, weeklyFocus]);

  const setEventExpanded = (eventId: string, open: boolean) => {
    setExpandedEvents((current) => {
      if (current.has(eventId) === open) return current;
      const next = new Set(current);
      if (open) next.add(eventId);
      else next.delete(eventId);
      return next;
    });
  };

  return (
    <div
      className={`stage stage--inspect${
        weeklyFocus ? " stage--inspect-weekly" : ""
      }`}
    >
      <div className="inspect-intro">
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
                : "Each row is one trace event. Open it to see the inputs, exact prompt when applicable, validation, and effective result."}
          </p>
        </div>
        <button className="button button--quiet" type="button" onClick={onReturn}>
          Return to Experience
        </button>
      </div>

      <div
        className="inspect-overview"
        aria-label={onboarding ? "Assessment summary" : "Trace summary"}
      >
        {onboarding ? (
          <>
            <span>{onboarding.responses.length} of 11 questions complete</span>
            <span>{onboarding.responses.length * 2} recorded selections</span>
            <span>
              {events.length > 0
                ? "Python validation recorded"
                : onboarding.confirmed
                  ? "Python validation unavailable"
                  : "Python validation follows confirmation"}
            </span>
          </>
        ) : (
          <>
            <span>{traceLabel}</span>
            <span>{countLabel(events.length, "trace event")}</span>
            <span>
              {sourceLabels.length > 0
                ? sourceLabels.join(" + ")
                : "No backend events yet"}
            </span>
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
            </div>
            <div className="inspect-focus__provenance" aria-label="Run provenance">
              <span>
                {SOURCE_LABELS[selectedEvent?.source ?? ""] ?? "Recorded run"}
              </span>
              {reviewerModelName ? <span>{reviewerModelName}</span> : null}
              {reviewerEffort ? <span>Reasoning {reviewerEffort}</span> : null}
              {reviewerEvent ? (
                <span>{formatDuration(reviewerEvent.duration_ms)}</span>
              ) : null}
            </div>
          </div>
          <p className="inspect-focus__evidence">
            AI-reviewed synthetic development evidence · not human validation
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
            <li>
              <span aria-hidden="true">3</span>
              <div>
                <strong>Coach Digest</strong>
                <p>
                  {coachEvent
                    ? eventSummary(coachEvent, currentJournalEntryIdSet)
                    : "No Coach Digest response was generated for this saved replay."}
                </p>
                <small>
                  Produces the optional user-facing response when available.
                </small>
              </div>
            </li>
          </ol>
          <p className="inspect-focus__more">
            The complete event history remains available below.
          </p>
        </section>
      ) : null}

      {onboarding ? (
        <OnboardingScoreInspection
          confirmed={onboarding.confirmed}
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
      ) : selectedEventId ? (
        <div className="inspect-selection inspect-selection--missing" role="status">
          <small>Linked event unavailable</small>
          <p><code>{selectedEventId}</code> is not present in this trace.</p>
        </div>
      ) : null}

      <section
        className="backend-trace"
        aria-labelledby={onboarding ? "backend-trace-title" : undefined}
      >
        {onboarding ? (
          <header className="backend-trace__heading">
            <div>
              <p className="eyebrow">Python boundary</p>
              <h2 id="backend-trace-title">Validation and later work.</h2>
            </div>
            <p>
              Profile confirmation starts the Python Experience session. Later
              trace events show model calls, validation, and deterministic
              product logic.
            </p>
          </header>
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
          <ol className="inspect-timeline" aria-label="Trace events">
            {events.map((event, index) => {
              const presentation = eventPresentation(event.event_type);
              const status = STATUS_LABELS[event.status] ?? titleCase(event.status);
              const source = SOURCE_LABELS[event.source] ?? titleCase(event.source);
              const parentNumber = event.parent_event_id
                ? eventNumbers.get(event.parent_event_id)
                : null;
              const isSelected = event.event_id === selectedEventId;
              const isExpanded = expandedEvents.has(event.event_id);
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
                      aria-label={`Event ${index + 1}: ${presentation.label}, ${status}, ${source}`}
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
                        <span className={`trace-chip trace-chip--status-${event.status}`}>
                          {status}
                        </span>
                        <span className={`trace-chip trace-chip--source-${event.source}`}>
                          {source}
                        </span>
                        <span className="trace-event__duration">
                          {formatDuration(event.duration_ms)}
                        </span>
                        <span className="trace-event__disclosure" aria-hidden="true">+</span>
                      </span>
                      <span className="trace-event__parent">
                        {parentNumber
                          ? `After event ${String(parentNumber).padStart(2, "0")}`
                          : "Trace root"}
                      </span>
                    </summary>
                    {isExpanded ? <EventDetails event={event} /> : null}
                  </details>
                </li>
              );
            })}
          </ol>
        )}
      </section>
    </div>
  );
}
