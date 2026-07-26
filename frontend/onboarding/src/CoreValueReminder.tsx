import { VALUES, type OnboardingProfile } from "./domain";

export default function CoreValueReminder({
  profile,
}: {
  profile: OnboardingProfile;
}) {
  return (
    <section className="core-value-reminder" aria-label="Your Core Values">
      <p>Your Core Values</p>
      <ul>
        {profile.top_values.map((value) => (
          <li key={value}>{VALUES[value].phrase}</li>
        ))}
      </ul>
    </section>
  );
}
