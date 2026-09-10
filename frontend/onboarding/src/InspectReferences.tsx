// Curated assessment references, not the execution revision of a recorded run.
export const INSPECT_REFERENCE_REVISION = "b277f72df47350cf5f7f5c73a7870dff783076b1";
const REPOSITORY_BASE = `https://github.com/DesmondChoy/twinkl/blob/${INSPECT_REFERENCE_REVISION}/`;

export const INSPECT_REFERENCES = {
  profile: {
    label: "Profile calculation",
    method: "docs/onboarding/onboarding_spec.md#5-scoring-and-product-transformation",
    implementation: "frontend/onboarding/src/domain.ts#L363-L422",
  },
  reviewer: {
    label: "Weekly Drift Reviewer",
    method: "docs/weekly/weekly_drift_detection.md#approved-weekly-drift-detection-runtime",
    implementation: "src/weekly_drift_reviewer.py#L243",
  },
  detector: {
    label: "Drift Detector",
    method: "docs/weekly/weekly_drift_detection.md#current-state-and-historical-drift-records",
    implementation: "src/drift_detector.py#L94",
  },
  coach: {
    label: "Coach Digest",
    method: "docs/weekly/weekly_drift_detection.md#coach-digest-prompt-and-response-contract",
    implementation: "src/coach/weekly_digest.py#L1433",
  },
  coachComparison: {
    label: "Coach Digest comparison",
    method: "docs/north_star/demo_coach_comparison.md",
    implementation: "src/coach/demo_comparison.py#L253",
  },
  northStar: {
    label: "North Star Moment",
    method: "docs/north_star/north_star_moment.md#2-which-writing-qualifies",
    implementation: "src/north_star/runtime.py#L171",
  },
  nudge: {
    label: "Nudge decision",
    method: "docs/pipeline/nudge_design_rationale.md#displayed-nudge-implementation",
    implementation: "src/nudge/runtime.py#L145",
  },
} as const;

export type InspectReferenceKey = keyof typeof INSPECT_REFERENCES;

export default function InspectReferences({ reference }: { reference: InspectReferenceKey }) {
  const { label, method, implementation } = INSPECT_REFERENCES[reference];
  return (
    <aside className="inspect-references" aria-label={`${label} references`}>
      <div className="inspect-references__links">
        <a href={`${REPOSITORY_BASE}${method}`} target="_blank" rel="noopener noreferrer">
          Read the method<span className="sr-only"> for {label} (opens in a new tab)</span>
          <span aria-hidden="true"> ↗</span>
        </a>
        <a href={`${REPOSITORY_BASE}${implementation}`} target="_blank" rel="noopener noreferrer">
          View implementation<span className="sr-only"> for {label} (opens in a new tab)</span>
          <span aria-hidden="true"> ↗</span>
        </a>
      </div>
      <small>Reference revision <code>{INSPECT_REFERENCE_REVISION.slice(0, 7)}</code>.
        {reference === "profile"
          ? " Explains the browser calculation."
          : " Recorded prompts and receipts remain the evidence for this run."}
      </small>
    </aside>
  );
}
