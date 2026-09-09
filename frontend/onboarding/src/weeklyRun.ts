import type { TraceEventContract } from "./demoContracts";

const WEEKLY_EVENT_TYPES = new Set([
  "weekly_review_requested", "weekly_review_completed", "drift_detected",
  "weekly_digest_built", "weekly_coach_generated", "north_star_reviewed",
]);

function resourceRefs(values: unknown[]): { kind: string; id: string }[] {
  return values.flatMap((value) => {
    if (value === null || typeof value !== "object") return [];
    const ref = value as Record<string, unknown>;
    return typeof ref.kind === "string" && typeof ref.id === "string"
      ? [{ kind: ref.kind, id: ref.id }] : [];
  });
}

function recordedWeek(event: TraceEventContract): { start: string; end: string } | null {
  for (const key of ["digest", "record", "receipt", "request"]) {
    const raw = event.details[key];
    const value = raw !== null && typeof raw === "object" && !Array.isArray(raw)
      ? raw as Record<string, unknown> : null;
    if (typeof value?.week_start === "string" && typeof value.week_end === "string") {
      return { start: value.week_start, end: value.week_end };
    }
  }
  const start = resourceRefs(event.input_refs).find((ref) => ref.kind === "week")?.id.match(/\d{4}-\d{2}-\d{2}$/)?.[0];
  if (!start) return null;
  const end = new Date(`${start}T00:00:00Z`);
  if (!Number.isFinite(end.getTime())) return null;
  end.setUTCDate(end.getUTCDate() + 6);
  return { start, end: end.toISOString().slice(0, 10) };
}

export function weeklyRunContext(events: TraceEventContract[], selectedEventId: string | null) {
  const selected = events.find((event) => event.event_id === selectedEventId);
  if (!selected || !isWeeklyEvent(selected)) return null;
  const weeklyEvents = events.filter((event) => WEEKLY_EVENT_TYPES.has(event.event_type));
  const byId = new Map(weeklyEvents.map((event) => [event.event_id, event]));
  const parentOf = (event: TraceEventContract) => {
    const parent = event.parent_event_id ? byId.get(event.parent_event_id) : undefined;
    if (parent) return parent;
    const preceding = weeklyEvents.slice(0, weeklyEvents.indexOf(event)).reverse();
    const expectedParent: Record<string, string> = {
      weekly_review_completed: "weekly_review_requested",
      drift_detected: "weekly_review_completed",
      weekly_digest_built: "drift_detected",
      weekly_coach_generated: "weekly_digest_built",
    };
    return preceding.find((candidate) =>
      candidate.event_type === expectedParent[event.event_type]
      && resourceRefs(event.input_refs).some((input) =>
        (input.kind === "event" && input.id === candidate.event_id)
        || resourceRefs(candidate.result_refs).some((output) =>
          input.kind === output.kind && input.id === output.id)));
  };
  const lineage = (event: TraceEventContract) => {
    const chain: TraceEventContract[] = [];
    let current: TraceEventContract | undefined = event;
    while (current && !chain.some((ancestor) => ancestor.event_id === current!.event_id)) {
      chain.push(current);
      if (current.event_type === "weekly_review_requested") break;
      current = parentOf(current);
    }
    return chain;
  };
  const selectedChain = lineage(selected);
  const anchor = selectedChain.find((event) => event.event_type === "weekly_review_requested");
  const week = selectedChain.map(recordedWeek).find((value) => value !== null) ?? null;
  return {
    week,
    events: weeklyEvents.filter((event) => {
      if (event.event_id === selected.event_id) return true;
      const chain = lineage(event);
      if (anchor) return chain.some((ancestor) => ancestor.event_id === anchor.event_id);
      const candidateWeek = chain.map(recordedWeek).find((value) => value !== null);
      return week !== null && candidateWeek?.start === week.start && candidateWeek.end === week.end;
    }),
  };
}

export function isWeeklyEvent(event: TraceEventContract): boolean {
  return WEEKLY_EVENT_TYPES.has(event.event_type);
}
