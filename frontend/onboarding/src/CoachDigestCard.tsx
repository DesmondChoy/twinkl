import { Fragment, useEffect, useState, type MouseEvent } from "react";
import type { JournalEntryContract, TraceEventContract } from "./demoContracts";
import type { OnboardingProfile } from "./domain";
import NorthStarMoment from "./NorthStarMoment";
import { expandCoachQuotations } from "./coachQuotes";
import { journalEntryAnchorId } from "./journalEntryAnchor";
import { currentNorthStarEvent, displayableNorthStarSelection, useNorthStarProfileRef } from "./northStar";
import { comparisonMatchesMoment, northStarAbsenceExplanation, savedCoachComparison } from "./coachComparison";
import { linkedRuns, northStarLinks } from "./northStarLinks";
import { VALUES, type ValueKey } from "./domain";

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

// Plain-language reason a saved moment qualified, keyed by its selection mode.
const MOMENT_BASIS: Record<string, string> = {
  reflection: "Active Drift this week, so the moment comes from writing before the Drift began.",
  encouragement: "No Active Drift, so the moment comes from this week’s writing.",
  reminder: "No Active Drift and no eligible action this week, so the moment comes from earlier writing.",
};

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
  const comparisonKey = pair ? `${northStar!.profile.session_id}:${profileRef}:${pair.week_start}:${pair.week_end}:${moment!.event.event_id}:${coachEvent!.event_id}` : null;
  const [activeComparison, setActiveComparison] = useState<string | null>(null);
  useEffect(() => { setActiveComparison(null); }, [comparisonKey]);
  const revealed = pair !== null && comparisonKey !== null && activeComparison === comparisonKey;
  const explanationKey = JSON.stringify([northStar?.profile, weeklyDigest?.week_start, weeklyDigest?.week_end]);
  const [expandedExplanation, setExpandedExplanation] = useState<string | null>(null);
  useEffect(() => { setExpandedExplanation(null); }, [explanationKey, comparisonKey]);
  const showExplanation = !pair && expandedExplanation === explanationKey;
  const explanation = northStarAbsenceExplanation(moment);
  const baseline = object(weeklyDigest?.coach_narrative);
  const narrative = pair ? pair.without_north_star.narrative : baseline;
  const weeklyMirror = nonEmptyText(narrative?.weekly_mirror);
  const tensionExplanation = nonEmptyText(narrative?.tension_explanation);
  const reflectiveQuestion = nonEmptyText(narrative?.reflective_question);

  if (!weeklyMirror || !tensionExplanation || !reflectiveQuestion
    || !["weekly_mirror", "tension_explanation", "reflective_question"].every((key) => nonEmptyText(baseline?.[key]))) return null;

  const Heading = headingLevel === 2 ? "h2" : "h3";
  const ExplanationHeading = headingLevel === 2 ? "h3" : "h4";
  const classes = ["coach-digest", demo ? "coach-digest--comparison" : null, className].filter(Boolean).join(" ");
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
  const expand = (paragraphs: string[]) => paragraphs.map((text) => expandCoachQuotations(text, sources));
  const expanded = expand([weeklyMirror, tensionExplanation, reflectiveQuestion]);
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
              aria-haspopup={onOpenEntry ? "dialog" : undefined}
              onClick={(event) => openEntry(event, entry)}>{entryDate(entry)}</a>
          </figcaption>
          <blockquote>{row.quotation}</blockquote>
        </figure>
      );
    });
  const northStarMoment = northStar ? (
    <NorthStarMoment
      {...northStar}
      weeklyDigest={weeklyDigest!}
      journalEntries={journalEntries}
      openJournalEntry={onOpenEntry}
      headingLevel={headingLevel === 2 ? 3 : 4}
    />
  ) : null;
  const links = pair ? northStarLinks(pair) : null;
  const linkedSpans = links ? [links.spans.weekly_mirror, links.spans.tension_explanation, links.spans.reflective_question] : null;
  const linkCount = linkedSpans?.reduce((total, spans) => total + spans.length, 0) ?? 0;
  // Each paragraph is one aligned row in the side-by-side comparison.
  const comparisonCells = (paragraphs: typeof expanded, spans?: string[][]) => paragraphs.map((paragraph, index) => (
    <div className={`coach-compare__cell coach-compare__cell--${["mirror", "tension", "question"][index]}`} key={index}>
      {index === 2 ? fullQuotations(paragraph) : null}
      <p className={`coach-digest__${["mirror", "tension", "question"][index]}`}>
        {spans ? linkedRuns(paragraph.text, spans[index]).map((run, runIndex) => run.linked
          ? <mark className="coach-compare__link" key={runIndex}>{run.text}</mark>
          : <Fragment key={runIndex}>{run.text}</Fragment>) : paragraph.text}
      </p>
      {index < 2 ? fullQuotations(paragraph) : null}
    </div>
  ));
  const momentEntry = selected ? journalEntries.find((entry) => entry.journal_entry_id === selected.entry_id) : undefined;
  const momentDate = selected ? new Intl.DateTimeFormat(undefined, {
    day: "numeric", month: "short", year: "numeric",
  }).format(new Date(`${selected.date}T00:00:00`)) : null;
  const momentValue = moment ? VALUES[moment.record.core_value as ValueKey] : undefined;
  const momentBasis = moment?.record.mode ? MOMENT_BASIS[moment.record.mode] : undefined;
  const addedInput = pair && moment && selected ? (
    <div className="coach-compare__cell coach-compare__cell--input coach-compare__input--added">
      <p className="coach-compare__input-label">Coach Digest input</p>
      <p>The same input, plus one North Star Moment:</p>
      <blockquote className="coach-compare__input-quote">{selected.evidence_quote}</blockquote>
      <p className="coach-compare__input-meta">
        {momentValue ? `${momentValue.name} · ` : ""}
        {selected.quote_source === "nudge_response" ? "Nudge response" : "Journal Entry"}, {momentDate}
      </p>
      {momentBasis ? (
        <p className="coach-compare__input-meta">
          Why this moment: {momentBasis} An AI review accepted it as an action supporting this Core Value.
        </p>
      ) : null}
      {links ? (
        <p className="coach-compare__input-meta coach-compare__input-links">
          {linkCount > 0
            ? "Highlighted phrases in the response draw on this moment."
            : "The response does not directly reference this moment."}
        </p>
      ) : null}
      <div className="coach-compare__input-actions">
        <a href={`#${journalEntryAnchorId(selected.entry_id)}`}
          aria-haspopup={onOpenEntry ? "dialog" : undefined}
          onClick={(event) => { if (momentEntry) openEntry(event, momentEntry); }}>
          Open {selected.quote_source === "nudge_response" ? "response in " : ""}Journal Entry · {momentDate}
        </a>
        {northStar?.inspectMoment ? (
          <button className="inspect-run-link" type="button"
            onClick={() => northStar.inspectMoment!(moment.event.event_id)}>
            Inspect this moment
          </button>
        ) : null}
      </div>
    </div>
  ) : null;
  const withColumnId = `${headingId}-with-north-star`;
  const open = pair ? revealed : showExplanation;

  return (
    <aside className={classes} aria-labelledby={headingId}>
      <div className="coach-digest__header">
        <Heading className="eyebrow" id={headingId}>Coach Digest</Heading>
        {demo ? (
          <button className="coach-digest__comparison-toggle" type="button"
            aria-describedby={`${headingId}-comparison-status`}
            aria-expanded={open}
            aria-controls={withColumnId}
            onClick={() => {
              if (pair) setActiveComparison(revealed ? null : comparisonKey);
              else setExpandedExplanation(showExplanation ? null : explanationKey);
            }}>
            {open ? "Hide comparison" : "Compare with North Star Moment"}
          </button>
        ) : null}
      </div>
      {demo ? (
        <div className="coach-digest__comparison-status sr-only"
          id={`${headingId}-comparison-status`} role="status">
          <p>Showing: {revealed ? "Without and with North Star Moment, side by side" : "Without North Star Moment"}</p>
        </div>
      ) : null}
      {demo ? (
        <div className={`coach-compare${open ? " coach-compare--revealed" : ""}`}>
          <section className="coach-compare__column coach-compare__column--without"
            aria-labelledby={`${headingId}-without-label`}>
            <p className="coach-compare__label" id={`${headingId}-without-label`}>Without North Star Moment</p>
            <div className="coach-compare__cell coach-compare__cell--input">
              <p className="coach-compare__input-label">Coach Digest input</p>
              <p>This week’s Journal Entries and Weekly Drift Detection result.</p>
            </div>
            {comparisonCells(expanded)}
          </section>
          {/* Rendered while hidden so both columns share row heights and the left text never shifts. */}
          <section className="coach-compare__column coach-compare__column--with" id={withColumnId}
            aria-labelledby={`${headingId}-with-label`} aria-hidden={!open} inert={!open}>
            <p className="coach-compare__label" id={`${headingId}-with-label`}>With North Star Moment</p>
            {pair ? (
              <>
                {addedInput}
                {comparisonCells(expand(["weekly_mirror", "tension_explanation", "reflective_question"]
                  .map((key) => pair.with_north_star.narrative[key] as string)), linkedSpans ?? undefined)}
              </>
            ) : (
              <section className="coach-compare__cell coach-compare__cell--explanation coach-digest__comparison-explanation"
                id={`${headingId}-comparison-explanation`}
                aria-labelledby={`${headingId}-explanation-title`}>
                <ExplanationHeading id={`${headingId}-explanation-title`}>{explanation.title}</ExplanationHeading>
                <p>{explanation.reason}</p>
                <p>The original reflection is unchanged.</p>
                {moment && northStar.inspectMoment ? (
                  <button className="inspect-run-link" type="button"
                    onClick={() => northStar.inspectMoment!(moment.event.event_id)}>
                    View review in Inspect
                  </button>
                ) : null}
              </section>
            )}
          </section>
          <div className="coach-compare__placeholder" aria-hidden="true">
            <p>{pair
              ? "The version with a North Star Moment appears here."
              : "There is no North Star Moment version this week. Compare to see why."}</p>
          </div>
        </div>
      ) : (
        <>
          {expanded.slice(0, 2).map((paragraph, index) => (
            <Fragment key={index}>
              <p className={index === 0 ? "coach-digest__mirror" : "coach-digest__tension"}>{paragraph.text}</p>
              {fullQuotations(paragraph)}
            </Fragment>
          ))}
          {fullQuotations(expanded[2])}
          <p className="coach-digest__question">{expanded[2].text}</p>
          {northStarMoment}
        </>
      )}
      {sourceEntries.length > 0 ? (
        <nav className="coach-digest__sources" aria-label="Coach Digest Journal Entries">
          <ul>
            {sourceEntries.map((entry) => (
              <li key={entry.journal_entry_id}>
                <a
                  href={`#${journalEntryAnchorId(entry.journal_entry_id)}`}
                  aria-haspopup={onOpenEntry ? "dialog" : undefined}
                  onClick={(event) => openEntry(event, entry)}
                >
                  {entryDate(entry)}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      ) : null}
    </aside>
  );
}
