import { Fragment, useEffect, useState, type MouseEvent } from "react";
import type { JournalEntryContract, TraceEventContract } from "./demoContracts";
import type { OnboardingProfile } from "./domain";
import NorthStarMoment from "./NorthStarMoment";
import { expandCoachQuotations } from "./coachQuotes";
import { journalEntryAnchorId } from "./journalEntryAnchor";
import { currentNorthStarEvent, displayableNorthStarSelection, useNorthStarProfileRef } from "./northStar";
import { comparisonMatchesMoment, savedCoachComparison } from "./coachComparison";

type JsonObject = Record<string, unknown>;

interface CoachDigestCardProps {
  weeklyDigest: JsonObject | null;
  headingId: string;
  headingLevel?: 2 | 3;
  className?: string;
  journalEntries?: JournalEntryContract[];
  onOpenEntry?: (entry: JournalEntryContract) => void;
  northStar?: {
    profile: OnboardingProfile;
    driftResult: JsonObject;
    traceEvents: TraceEventContract[];
    presentation?: "personal" | "demo";
    inspectMoment?: (eventId: string) => void;
  };
}

function object(value: unknown): JsonObject | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonObject
    : null;
}

function nonEmptyText(value: unknown): string | null {
  return typeof value === "string" && value.trim().length > 0
    ? value
    : null;
}

export default function CoachDigestCard({
  weeklyDigest,
  headingId,
  headingLevel = 2,
  className,
  journalEntries = [],
  onOpenEntry,
  northStar,
}: CoachDigestCardProps) {
  const demo = northStar?.presentation === "demo";
  const profileRef = useNorthStarProfileRef(demo ? northStar.profile : null);
  const moment = demo ? currentNorthStarEvent({
    events: northStar.traceEvents, profile: northStar.profile, profileRef, weeklyDigest, journalEntries,
  }) : null;
  const selected = demo ? displayableNorthStarSelection(moment, northStar.profile, journalEntries, northStar.driftResult) : null;
  const coachEvent = demo ? [...northStar.traceEvents].reverse().find((event) =>
    event.event_type === "weekly_coach_generated" && event.session_id === northStar.profile.session_id) ?? null : null;
  const savedPair = savedCoachComparison(coachEvent);
  const pair = savedPair && moment && selected && comparisonMatchesMoment(savedPair, moment, selected)
    ? savedPair : null;
  const recordedMoment = demo ? [...northStar.traceEvents].reverse().find((event) => {
    const record = object(event.details.record);
    return record !== null && event.event_type === "north_star_reviewed" && event.session_id === northStar.profile.session_id
      && record.week_start === weeklyDigest?.week_start && record.week_end === weeklyDigest?.week_end;
  }) : null;
  const hasRecordedSelection = object(object(recordedMoment?.details.record)?.selected) !== null;
  const comparisonKey = pair ? `${northStar!.profile.session_id}:${profileRef}:${pair.week_start}:${pair.week_end}:${moment!.event.event_id}:${coachEvent!.event_id}` : null;
  const [activeComparison, setActiveComparison] = useState<string | null>(null);
  useEffect(() => { setActiveComparison(null); }, [comparisonKey]);
  const withMoment = pair !== null && comparisonKey !== null && activeComparison === comparisonKey;
  const baseline = object(weeklyDigest?.coach_narrative);
  const narrative = pair ? pair[withMoment ? "with_north_star" : "without_north_star"].narrative : baseline;
  const weeklyMirror = nonEmptyText(narrative?.weekly_mirror);
  const tensionExplanation = nonEmptyText(narrative?.tension_explanation);
  const reflectiveQuestion = nonEmptyText(narrative?.reflective_question);

  if (!weeklyMirror || !tensionExplanation || !reflectiveQuestion
    || !["weekly_mirror", "tension_explanation", "reflective_question"].every((key) => nonEmptyText(baseline?.[key]))) return null;

  const Heading = headingLevel === 2 ? "h2" : "h3";
  const classes = ["coach-digest", className].filter(Boolean).join(" ");
  const cutoff = typeof weeklyDigest?.week_end === "string"
    ? weeklyDigest.week_end : null;
  const evidence = Array.isArray(weeklyDigest?.evidence)
    ? weeklyDigest.evidence.map(object).filter((row) => row !== null) : [];
  const sourceEntries = journalEntries.filter((entry) =>
    cutoff !== null && entry.date <= cutoff && evidence.some((row) =>
      row.t_index === entry.t_index && row.date === entry.date,
    ),
  );
  const compact = (text: string) => text.replace(/\s+/g, " ").trim();
  const sources = sourceEntries.filter((entry) => evidence.some((row) =>
    row.t_index === entry.t_index && row.date === entry.date
    && typeof row.excerpt === "string"
    && compact(row.excerpt.replace(/(?:\.{3}|…)\s*$/, "")).length > 0
    && compact(entry.content).includes(compact(row.excerpt.replace(/(?:\.{3}|…)\s*$/, ""))),
  )).map((entry) => entry.content);
  const paragraphs = [weeklyMirror, tensionExplanation, reflectiveQuestion];
  const expanded = paragraphs.map((text) => expandCoachQuotations(text, sources));
  const openEntry = (event: MouseEvent<HTMLAnchorElement>, entry: JournalEntryContract) => {
    if (onOpenEntry) {
      event.preventDefault();
      onOpenEntry(entry);
    }
  };
  const entryDate = (entry: JournalEntryContract) => new Intl.DateTimeFormat(undefined, {
    day: "numeric", month: "short",
  }).format(new Date(`${entry.date}T00:00:00`));
  const fullQuotations = (paragraph: typeof expanded[number]) =>
    paragraph.fullQuotations.map((row, quoteIndex) => {
      const entry = sourceEntries.find((item) => item.content === row.source)!;
      return (
        <figure className="coach-digest__full-quote" key={quoteIndex}>
          <figcaption>
            Full quotation from{" "}
            <a href={`#${journalEntryAnchorId(entry.journal_entry_id)}`}
              onClick={(event) => openEntry(event, entry)}>{entryDate(entry)}</a>
          </figcaption>
          <blockquote>{row.quotation}</blockquote>
        </figure>
      );
    });

  return (
    <aside className={classes} aria-labelledby={headingId}>
      <div className="coach-digest__header">
        <p className="eyebrow">Coach Digest</p>
        {demo ? (
          <button className="coach-digest__comparison-toggle" type="button"
            disabled={!pair}
            aria-describedby={`${headingId}-comparison-status`}
            onClick={() => setActiveComparison(withMoment ? null : comparisonKey)}>
            {withMoment ? "Without North Star Moment" : "With North Star Moment"}
          </button>
        ) : null}
      </div>
      <Heading id={headingId}>Your weekly reflection</Heading>
      {demo ? (
        <div className="coach-digest__comparison-status" id={`${headingId}-comparison-status`} role="status">
          <p>Showing: {withMoment ? "With" : "Without"} North Star Moment</p>
          {!pair ? <p>{hasRecordedSelection
            ? "Comparison unavailable for this week." : "No North Star Moment for this week."}</p> : null}
        </div>
      ) : null}
      {expanded.slice(0, 2).map((paragraph, index) => (
        <Fragment key={index}>
          <p className={index === 0 ? "coach-digest__mirror" : "coach-digest__tension"}>{paragraph.text}</p>
          {fullQuotations(paragraph)}
        </Fragment>
      ))}
      {sourceEntries.length > 0 ? (
        <nav className="coach-digest__sources" aria-label="Coach Digest Journal Entries">
          <ul>
            {sourceEntries.map((entry) => (
              <li key={entry.journal_entry_id}>
                <a
                  href={`#${journalEntryAnchorId(entry.journal_entry_id)}`}
                  onClick={(event) => openEntry(event, entry)}
                >
                  {entryDate(entry)}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      ) : null}
      {northStar && (!demo || withMoment) ? (
        <NorthStarMoment
          {...northStar}
          weeklyDigest={weeklyDigest!}
          journalEntries={journalEntries}
          openJournalEntry={onOpenEntry}
          headingLevel={headingLevel === 2 ? 3 : 4}
        />
      ) : null}
      {fullQuotations(expanded[2])}
      <p className="coach-digest__question-label">Something to reflect on</p>
      <p className="coach-digest__question">{expanded[2].text}</p>
    </aside>
  );
}
