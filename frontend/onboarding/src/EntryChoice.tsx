import type { Ref } from "react";

interface EntryChoiceProps {
  headingRef: Ref<HTMLHeadingElement>;
  onDemo: () => void;
  onOnboarding: () => void;
  hasOnboardingProgress: boolean;
}

export default function EntryChoice({
  headingRef,
  onDemo,
  onOnboarding,
  hasOnboardingProgress,
}: EntryChoiceProps) {
  return (
    <main className="entry-choice" id="main">
      <h1 ref={headingRef} tabIndex={-1}>Choose how to explore Twinkl.</h1>
      <p className="lede">
        See how everyday choices connect with what matters to you.
      </p>
      <div className="entry-choice__options">
        <button
          className="entry-choice__card"
          type="button"
          aria-labelledby="entry-demo-title"
          aria-describedby="entry-demo-description"
          onClick={onDemo}
        >
          <span className="entry-choice__title" id="entry-demo-title">Try the Demo</span>
          <span id="entry-demo-description">
            Follow a saved Persona’s Journal Entries, see how Drift unfolds,
            and explore the evidence behind each reflection. No writing required.
          </span>
          <span className="entry-choice__note">Five synthetic Persona stories</span>
        </button>
        <button
          className="entry-choice__card"
          type="button"
          aria-labelledby="entry-onboarding-title"
          aria-describedby="entry-onboarding-description"
          onClick={onOnboarding}
        >
          <span className="entry-choice__title" id="entry-onboarding-title">Try Onboarding</span>
          <span id="entry-onboarding-description">
            Discover your Core Values through a short assessment, confirm your
            Profile, and try writing your own Journal Entries.
          </span>
          <span className="entry-choice__note">
            {hasOnboardingProgress ? "Continue your saved progress" : "Experience Twinkl as a user"}
          </span>
        </button>
      </div>
    </main>
  );
}
