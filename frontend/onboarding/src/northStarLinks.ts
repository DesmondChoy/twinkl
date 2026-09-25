import type { CoachComparison } from "./coachComparison";
import annotations from "./northStarLinks.json";

type NarrativeField = "weekly_mirror" | "tension_explanation" | "reflective_question";

export interface NorthStarLinks {
  spans: Record<NarrativeField, string[]>;
}

/**
 * Display-only annotation of the phrases in a saved with-context response that draw on
 * its North Star Moment. Bound to the exact pair so an edited response loses its links.
 */
export function northStarLinks(pair: CoachComparison): NorthStarLinks | null {
  const entry = annotations.links.find((link) =>
    link.scenario_id === pair.scenario_id && link.week_start === pair.week_start
    && link.north_star_input_hash === pair.north_star_input_hash
    && link.with_response_sha256 === pair.with_north_star.response_sha256);
  if (!entry) return null;
  const spans: Record<NarrativeField, string[]> = {
    weekly_mirror: [], tension_explanation: [], reflective_question: [],
  };
  for (const span of entry.spans) {
    if (span.field in spans && pair.with_north_star.narrative[span.field as NarrativeField].includes(span.text)) {
      spans[span.field as NarrativeField].push(span.text);
    }
  }
  return { spans };
}

/** Splits text into plain and linked runs, marking the first occurrence of each span. */
export function linkedRuns(text: string, spans: string[]): { text: string; linked: boolean }[] {
  const ranges = spans.map((span) => [text.indexOf(span), span.length] as const)
    .filter(([start]) => start >= 0)
    .sort((left, right) => left[0] - right[0]);
  const runs: { text: string; linked: boolean }[] = [];
  let cursor = 0;
  for (const [start, length] of ranges) {
    if (start < cursor) continue;
    if (start > cursor) runs.push({ text: text.slice(cursor, start), linked: false });
    runs.push({ text: text.slice(start, start + length), linked: true });
    cursor = start + length;
  }
  if (cursor < text.length) runs.push({ text: text.slice(cursor), linked: false });
  return runs;
}
