import { useId, useState } from "react";
import { VALUES, type OnboardingProfile, type ValueKey } from "./domain";
import type { JournalEntryContract, TraceEventContract } from "./demoContracts";
import { journalEntryAnchorId } from "./journalEntryAnchor";
import { currentNorthStarEvent, displayableNorthStarSelection, useNorthStarProfileRef } from "./northStar";

export default function NorthStarMoment({
  profile, journalEntries, weeklyDigest, driftResult, traceEvents, selectJournalEntry, openJournalEntry, headingLevel = 2,
}: {
  profile: OnboardingProfile;
  journalEntries: JournalEntryContract[];
  weeklyDigest: Record<string, unknown>;
  driftResult: Record<string, unknown>;
  traceEvents: TraceEventContract[];
  selectJournalEntry?: (journalEntryId: string) => void;
  openJournalEntry?: (entry: JournalEntryContract) => void;
  headingLevel?: 2 | 3;
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
  const Heading = headingLevel === 2 ? "h2" : "h3";
  const date = new Intl.DateTimeFormat(undefined, {
    day: "numeric", month: "short", year: "numeric",
  }).format(new Date(`${selected.date}T00:00:00`));
  const longQuote = selected.evidence_quote.length > 320;
  const expanded = expandedEventId === result.event.event_id;
  const framing = record.mode === "reflection"
    ? "This earlier action expressed a priority that has felt harder to make room for in the week reviewed."
    : record.mode === "encouragement"
      ? "This action from the week reviewed is one way you put this priority into practice."
      : "This earlier action is a reminder of how you have expressed this priority. It offers perspective on the week reviewed.";
  return (
    <aside className="north-star-moment" aria-labelledby={`${id}-title`}>
      <p className="eyebrow">North Star Moment</p>
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
    </aside>
  );
}
