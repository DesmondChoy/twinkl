import { VALUES, type OnboardingProfile } from "./domain";

export default function CoreValueReminder({
  profile,
}: {
  profile: OnboardingProfile;
}) {
  const label = profile.top_values.length === 1 ? "Your Core Value" : "Your Core Values";
  return (
    <section className="core-value-reminder" aria-label={label}>
      <p>{label}</p>
      <ul>
        {profile.top_values.map((value) => (
          <li key={value}>{VALUES[value].phrase}</li>
        ))}
      </ul>
    </section>
  );
}
