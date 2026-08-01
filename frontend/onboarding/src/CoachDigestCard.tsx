type JsonObject = Record<string, unknown>;

interface CoachDigestCardProps {
  weeklyDigest: JsonObject | null;
  headingId: string;
  headingLevel?: 2 | 3;
  className?: string;
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
}: CoachDigestCardProps) {
  const narrative = object(weeklyDigest?.coach_narrative);
  const weeklyMirror = nonEmptyText(narrative?.weekly_mirror);
  const tensionExplanation = nonEmptyText(narrative?.tension_explanation);
  const reflectiveQuestion = nonEmptyText(narrative?.reflective_question);

  if (!weeklyMirror || !tensionExplanation || !reflectiveQuestion) return null;

  const Heading = headingLevel === 2 ? "h2" : "h3";
  const classes = ["coach-digest", className].filter(Boolean).join(" ");

  return (
    <aside className={classes} aria-labelledby={headingId}>
      <p className="eyebrow">Coach Digest</p>
      <Heading id={headingId}>{weeklyMirror}</Heading>
      <p>{tensionExplanation}</p>
      <p className="coach-digest__question">{reflectiveQuestion}</p>
    </aside>
  );
}
