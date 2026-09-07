import { Fragment, type MouseEvent } from "react";
import type { JournalEntryContract } from "./demoContracts";
import { expandCoachQuotations } from "./coachQuotes";
import { journalEntryAnchorId } from "./journalEntryAnchor";

type JsonObject = Record<string, unknown>;

interface CoachDigestCardProps {
  weeklyDigest: JsonObject | null;
  headingId: string;
  headingLevel?: 2 | 3;
  className?: string;
  journalEntries?: JournalEntryContract[];
  onOpenEntry?: (entry: JournalEntryContract) => void;
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
}: CoachDigestCardProps) {
  const narrative = object(weeklyDigest?.coach_narrative);
  const weeklyMirror = nonEmptyText(narrative?.weekly_mirror);
  const tensionExplanation = nonEmptyText(narrative?.tension_explanation);
  const reflectiveQuestion = nonEmptyText(narrative?.reflective_question);

  if (!weeklyMirror || !tensionExplanation || !reflectiveQuestion) return null;

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
  const hasExpandedQuotes = expanded.some((row, index) => row.text !== paragraphs[index]);
  const openEntry = (event: MouseEvent<HTMLAnchorElement>, entry: JournalEntryContract) => {
    if (onOpenEntry) {
      event.preventDefault();
      onOpenEntry(entry);
    }
  };
  const entryDate = (entry: JournalEntryContract) => new Intl.DateTimeFormat(undefined, {
    day: "numeric", month: "short",
  }).format(new Date(`${entry.date}T00:00:00`));

  return (
    <aside className={classes} aria-labelledby={headingId}>
      <p className="eyebrow">Coach Digest</p>
      <Heading id={headingId}>Your weekly reflection</Heading>
      {expanded.map((paragraph, index) => (
        <Fragment key={index}>
          <p className={index === 2 ? "coach-digest__question" : undefined}>{paragraph.text}</p>
          {paragraph.fullQuotations.map((row, quoteIndex) => {
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
          })}
        </Fragment>
      ))}
      {sourceEntries.length > 0 ? (
        <nav className="coach-digest__sources" aria-label="Coach Digest Journal Entries">
          <p>Read the supporting Journal Entries</p>
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
          {hasExpandedQuotes ? (
            <small>Truncated quotations are expanded from these Journal Entries.</small>
          ) : null}
        </nav>
      ) : null}
    </aside>
  );
}
