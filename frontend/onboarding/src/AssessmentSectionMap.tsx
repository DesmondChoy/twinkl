import SectionMap from "./SectionMap";

export const ASSESSMENT_SECTIONS = [
  {
    id: "assessment-choices",
    label: "Choices",
    description: "Most and Least choices",
  },
  {
    id: "assessment-counts",
    label: "Counts",
    description: "11 SVBWS objects",
  },
  {
    id: "assessment-universalism",
    label: "Universalism merge",
    description: "Two objects become one value",
  },
  {
    id: "assessment-profile",
    label: "Profile",
    description: "Ten weighted values",
  },
  {
    id: "assessment-checks",
    label: "Checks",
    description: "Calculation invariants",
  },
] as const;

export default function AssessmentSectionMap() {
  return (
    <SectionMap
      description="Each step links to the evidence on this page."
      eyebrow="Assessment trail"
      navigationLabel="Assessment sections"
      sections={ASSESSMENT_SECTIONS}
      title="Follow the calculation."
    />
  );
}
