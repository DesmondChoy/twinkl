import { useEffect, useState } from "react";
import type { OnboardingProfile } from "./domain";
import type { JournalEntryContract, TraceEventContract } from "./demoContracts";

type JsonObject = Record<string, unknown>;

export interface NorthStarSelection {
  entry_id: string;
  t_index: number;
  date: string;
  quote_source: "journal_entry" | "nudge_response";
  evidence_quote: string;
}

export interface NorthStarRecord extends JsonObject {
  schema_version: "north-star-record-v1";
  session_id: string;
  owner_id: string;
  profile_ref: string;
  week_start: string;
  week_end: string;
  cutoff_at: string;
  input_hash: string;
  status: "pending" | "complete" | "failed" | "not_eligible";
  mode: "reflection" | "encouragement" | "reminder" | null;
  reason: string;
  core_value: string | null;
  value_phrase: string | null;
  selected: NorthStarSelection | null;
  sources: JsonObject[];
  onset_t_index: number | null;
  onset_date: string | null;
  onset_available_at: string | null;
  reviews: unknown[];
  attempts: number;
  retryable: boolean;
}

function object(value: unknown): JsonObject | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const item = object(value);
  if (item) {
    return `{${Object.keys(item).sort().map((key) =>
      `${JSON.stringify(key)}:${canonicalJson(item[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

export async function northStarProfileRef(profile: OnboardingProfile): Promise<string> {
  const hash = await crypto.subtle.digest(
    "SHA-256", new TextEncoder().encode(canonicalJson({
      ...profile, preferred_name: profile.preferred_name ?? null,
    })),
  );
  return Array.from(new Uint8Array(hash))
    .map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export function useNorthStarProfileRef(profile: OnboardingProfile): string | null {
  const profileJson = JSON.stringify(profile);
  const [resolved, setResolved] = useState<{ key: string; hash: string } | null>(null);
  useEffect(() => {
    let current = true;
    void northStarProfileRef(JSON.parse(profileJson) as OnboardingProfile)
      .then((hash) => { if (current) setResolved({ key: profileJson, hash }); })
      .catch(() => { if (current) setResolved(null); });
    return () => { current = false; };
  }, [profileJson]);
  return resolved?.key === profileJson ? resolved.hash : null;
}

export function currentNorthStarEvent({
  events, profile, profileRef, weeklyDigest, journalEntries,
}: {
  events: TraceEventContract[];
  profile: OnboardingProfile;
  profileRef: string | null;
  weeklyDigest: JsonObject | null;
  journalEntries: JournalEntryContract[];
}): { event: TraceEventContract; record: NorthStarRecord } | null {
  if (!profileRef || !weeklyDigest) return null;
  for (const event of [...events].reverse()) {
    if (event.event_type !== "north_star_reviewed" || event.session_id !== profile.session_id) continue;
    const record = object(event.details.record);
    if (!record || record.schema_version !== "north-star-record-v1"
      || record.session_id !== profile.session_id || record.owner_id !== profile.user_id
      || record.profile_ref !== profileRef || record.week_start !== weeklyDigest.week_start
      || record.week_end !== weeklyDigest.week_end || record.input_hash !== event.input_hash) continue;
    if (!Array.isArray(record.sources)) continue;
    const sources = record.sources;
    const sourcesMatch = sources.every((value) => {
      const source = object(value);
      if (!source || source.owner_id !== profile.user_id) return false;
      const entry = journalEntries.find((candidate) => candidate.journal_entry_id === source.entry_id);
      return entry && entry.t_index === source.t_index && entry.date === source.date
        && entry.content === source.journal_entry
        && (source.nudge_response === null || entry.nudge_response === source.nudge_response);
    });
    if (!sourcesMatch) continue;
    return { event, record: record as NorthStarRecord };
  }
  return null;
}

export function displayableNorthStarSelection(
  result: { event: TraceEventContract; record: NorthStarRecord } | null,
  profile: OnboardingProfile,
  entries: JournalEntryContract[],
  driftResult: JsonObject,
): NorthStarSelection | null {
  if (!result) return null;
  const { event, record } = result;
  const selected = object(record.selected);
  const selectedCoreValue = record.core_value;
  if (!["complete", "reused"].includes(event.status) || record.status !== "complete"
    || event.validation?.valid === false
    || !selected || !selectedCoreValue || !profile.top_values.includes(selectedCoreValue as typeof profile.top_values[number])
    || typeof selected.entry_id !== "string" || !Number.isInteger(selected.t_index)
    || typeof selected.date !== "string" || typeof selected.evidence_quote !== "string"
    || selected.evidence_quote.trim().length === 0
    || !["journal_entry", "nudge_response"].includes(String(selected.quote_source))) return null;
  const entry = entries.find((candidate) => candidate.journal_entry_id === selected.entry_id);
  if (!entry || entry.date !== selected.date || entry.t_index !== selected.t_index
    || !/^\d{4}-\d{2}-\d{2}$/.test(entry.date) || !Number.isFinite(Date.parse(`${entry.date}T00:00:00Z`))
    || entry.date > record.week_end || !Number.isFinite(Date.parse(record.cutoff_at))) return null;
  const quoteSource = selected.quote_source === "journal_entry" ? entry.content : entry.nudge_response;
  if (!quoteSource?.includes(selected.evidence_quote)) return null;
  const source = record.sources?.find((candidate) => candidate.entry_id === entry.journal_entry_id);
  if (!source || source.owner_id !== profile.user_id) return null;
  const availableAt = selected.quote_source === "nudge_response" ? source.response_available_at : source.available_at;
  if (typeof availableAt !== "string" || !Number.isFinite(Date.parse(availableAt))
    || Date.parse(availableAt) > Date.parse(record.cutoff_at)) return null;
  const states = object(driftResult.core_value_states);
  const active = driftResult.delivery_state === "active_drift";
  if (record.mode === "reflection") {
    if (!active || states?.[selectedCoreValue] !== "active_drift") return null;
    const drifts = Array.isArray(driftResult.drifts) ? driftResult.drifts : [];
    const onset = drifts.map(object).find((drift) => drift?.core_value === selectedCoreValue
      && drift.termination_reason === null);
    if (!onset || typeof onset.onset_t_index !== "number" || typeof onset.onset_date !== "string"
      || entry.t_index >= onset.onset_t_index || entry.date > onset.onset_date
      || record.onset_t_index !== onset.onset_t_index || record.onset_date !== onset.onset_date
      || typeof record.onset_available_at !== "string" || !Number.isFinite(Date.parse(record.onset_available_at))
      || Date.parse(availableAt) >= Date.parse(record.onset_available_at)) return null;
  } else if (record.mode === "encouragement") {
    if (driftResult.delivery_state !== "no_active_drift" || entry.date < record.week_start) return null;
  } else if (record.mode === "reminder") {
    if (driftResult.delivery_state !== "no_active_drift" || entry.date >= record.week_start) return null;
  } else return null;
  return selected as unknown as NorthStarSelection;
}
