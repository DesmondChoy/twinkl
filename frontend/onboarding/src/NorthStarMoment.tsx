import { useId, useState } from "react";
import { VALUES, type OnboardingProfile, type ValueKey } from "./domain";
import type { JournalEntryContract, TraceEventContract } from "./demoContracts";
import { journalEntryAnchorId } from "./journalEntryAnchor";
import { currentNorthStarEvent, displayableNorthStarSelection, northStarFraming, useNorthStarProfileRef } from "./northStar";

export default function NorthStarMoment({
  profile, journalEntries, weeklyDigest, driftResult, traceEvents, selectJournalEntry, openJournalEntry,
  headingLevel = 3, presentation = "personal", inspectMoment,
}: {
  profile: OnboardingProfile;
  journalEntries: JournalEntryContract[];
  weeklyDigest: Record<string, unknown>;
  driftResult: Record<string, unknown>;
  traceEvents: TraceEventContract[];
  selectJournalEntry?: (journalEntryId: string) => void;
  openJournalEntry?: (entry: JournalEntryContract) => void;
  headingLevel?: 3 | 4;
  presentation?: "personal" | "demo";
  inspectMoment?: (eventId: string) => void;
}) {
  const id = useId();
  const [expandedEventId, setExpandedEventId] = useState<string | null>(null);
  const profileRef = useNorthStarProfileRef(profile);
  const result = currentNorthStarEvent({ events: traceEvents, profile, profileRef, weeklyDigest, journalEntries });
  const selected = displayableNorthStarSelection(result, profile, journalEntries, driftResult);
  if (!result || !selected) return null;
  const { record } = result;
  const valuePhrase = VALUES[record.core_value as ValueKey]?.phrase;
  if (!valuePhrase) return null;
  const Heading = headingLevel === 3 ? "h3" : "h4";
  const date = new Intl.DateTimeFormat(undefined, {
    day: "numeric", month: "short", year: "numeric",
  }).format(new Date(`${selected.date}T00:00:00`));
  const longQuote = selected.evidence_quote.length > 320;
  const expanded = expandedEventId === result.event.event_id;
  const framing = northStarFraming(record.mode);
  return (
    <section className={`north-star-moment north-star-moment--${presentation}`} aria-labelledby={`${id}-title`}>
      {presentation === "demo" ? <p className="eyebrow">North Star Moment</p> : null}
      <Heading id={`${id}-title`}>{record.mode === "encouragement"
        ? "A moment in your own words"
        : "A past moment in your own words"}</Heading>
      <p className="north-star-moment__value">{valuePhrase}</p>
      <p>{framing}</p>
      <blockquote id={`${id}-quote`} className={longQuote && !expanded ? "north-star-moment__quote--collapsed" : undefined}>
        {selected.evidence_quote}
      </blockquote>
      {longQuote ? (
        <button className="button button--quiet" type="button"
          aria-expanded={expanded} aria-controls={`${id}-quote`}
          onClick={() => setExpandedEventId(expanded ? null : result.event.event_id)}>
          {expanded ? "Collapse quotation" : "Expand quotation"}
        </button>
      ) : null}
      <a href={`#${journalEntryAnchorId(selected.entry_id)}`}
        onClick={(event) => {
          const entry = journalEntries.find((candidate) => candidate.journal_entry_id === selected.entry_id);
          if (openJournalEntry && entry) {
            event.preventDefault();
            openJournalEntry(entry);
          } else selectJournalEntry?.(selected.entry_id);
        }}>
        Open {selected.quote_source === "nudge_response" ? "response in " : ""}Journal Entry · {date}
      </a>
      {record.mode !== "encouragement" ? (
        <p className="north-star-moment__reference">This earlier writing is a reference point for your Core Value.</p>
      ) : null}
      {inspectMoment ? (
        <button className="inspect-run-link north-star-moment__inspect" type="button"
          onClick={() => inspectMoment(result.event.event_id)}>
          Inspect this moment
        </button>
      ) : null}
    </section>
  );
}
