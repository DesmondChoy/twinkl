import { useMemo } from "react";
import SectionMap, { type SectionMapItem } from "./SectionMap";

export type ExperienceSectionMapView =
  | "summary"
  | "complete"
  | "journal"
  | "persona-picker"
  | "inspect";

interface ExperienceSectionMapProps {
  hasJournalComposer?: boolean;
  hasJournalEntries?: boolean;
  hasWeeklyResult?: boolean;
  view: ExperienceSectionMapView;
}

interface SectionMapConfiguration {
  description: string;
  eyebrow: string;
  sections: readonly SectionMapItem[];
  title: string;
}

const MAPS: Record<
  Exclude<ExperienceSectionMapView, "journal" | "persona-picker">,
  SectionMapConfiguration
> = {
  summary: {
    eyebrow: "Experience trail",
    title: "Review your Profile.",
    description: "Move from the result to confirmation.",
    sections: [
      {
        id: "experience-profile",
        label: "Profile",
        description: "Highest-scoring values",
      },
      {
        id: "experience-confirm",
        label: "Confirm",
        description: "Save this Profile",
      },
    ],
  },
  complete: {
    eyebrow: "Experience trail",
    title: "Start the Journal.",
    description: "Carry the confirmed Profile into one real moment.",
    sections: [
      {
        id: "experience-ready",
        label: "Profile",
        description: "Confirmed Core Values",
      },
      {
        id: "experience-journal-handoff",
        label: "Journal Entry",
        description: "Continue into Experience",
      },
    ],
  },
  inspect: {
    eyebrow: "Inspect trail",
    title: "Follow the work.",
    description: "Move from the selected run to its recorded events.",
    sections: [
      {
        id: "inspect-overview-section",
        label: "Summary",
        description: "Selected run",
      },
      {
        id: "inspect-events-section",
        label: "Recorded work",
        description: "Filters, events, and details",
      },
    ],
  },
};

const JOURNAL_MAPS: Record<
  "empty" | "withEntries" | "withResult" | "withEntriesAndResult",
  SectionMapConfiguration
> = {
  empty: {
    eyebrow: "Experience trail",
    title: "Follow your week.",
    description: "Move from writing to the weekly result.",
    sections: [
      {
        id: "experience-journal-prompt",
        label: "Prompt",
        description: "One moment to notice",
      },
      {
        id: "experience-journal-compose",
        label: "Write",
        description: "Save a Journal Entry",
      },
    ],
  },
  withEntries: {
    eyebrow: "Experience trail",
    title: "Follow your week.",
    description: "Move from writing to the weekly result.",
    sections: [
      {
        id: "experience-journal-prompt",
        label: "Prompt",
        description: "One moment to notice",
      },
      {
        id: "experience-journal-compose",
        label: "Write",
        description: "Save a Journal Entry",
      },
      {
        id: "journal-thread-title",
        label: "Journal Entries",
        description: "Your recorded moments",
      },
    ],
  },
  withResult: {
    eyebrow: "Experience trail",
    title: "Follow your week.",
    description: "Move from writing to the weekly result.",
    sections: [
      {
        id: "experience-journal-prompt",
        label: "Prompt",
        description: "One moment to notice",
      },
      {
        id: "experience-journal-compose",
        label: "Write",
        description: "Save a Journal Entry",
      },
      {
        id: "experience-weekly",
        label: "Weekly Drift",
        description: "Closed-week result",
      },
    ],
  },
  withEntriesAndResult: {
    eyebrow: "Experience trail",
    title: "Follow your week.",
    description: "Move from writing to the weekly result.",
    sections: [
      {
        id: "experience-journal-prompt",
        label: "Prompt",
        description: "One moment to notice",
      },
      {
        id: "experience-journal-compose",
        label: "Write",
        description: "Save a Journal Entry",
      },
      {
        id: "journal-thread-title",
        label: "Journal Entries",
        description: "Your recorded moments",
      },
      {
        id: "experience-weekly",
        label: "Weekly Drift",
        description: "Closed-week result",
      },
    ],
  },
};

export default function ExperienceSectionMap({
  hasJournalComposer = false,
  hasJournalEntries = false,
  hasWeeklyResult = false,
  view,
}: ExperienceSectionMapProps) {
  let journalMap: keyof typeof JOURNAL_MAPS = "empty";
  if (hasJournalEntries && hasWeeklyResult) {
    journalMap = "withEntriesAndResult";
  } else if (hasJournalEntries) {
    journalMap = "withEntries";
  } else if (hasWeeklyResult) {
    journalMap = "withResult";
  }
  const configuration = view === "journal"
    ? JOURNAL_MAPS[journalMap]
    : view === "persona-picker" ? null : MAPS[view];
  const sections = useMemo(() => {
    const availableSections = configuration?.sections ?? [];
    return view === "journal" && !hasJournalComposer
      ? availableSections.filter((section) => section.id !== "experience-journal-compose")
      : availableSections;
  }, [configuration, hasJournalComposer, view]);

  if (view === "persona-picker") {
    return <section className="persona-demo-guide" aria-labelledby="persona-demo-guide-title">
      <h2 id="persona-demo-guide-title">See how Twinkl works</h2>
      <ul className="persona-demo-guide__copy">
        <li>Read a Persona’s saved Journal Entries.</li>
        <li>See the weekly results.</li>
        <li>Open Inspect to check which entries support those results.</li>
      </ul>
    </section>;
  }
  if (!configuration) return null;

  return (
    <SectionMap
      {...configuration}
      sections={sections}
      navigationLabel="Experience sections"
    />
  );
}
