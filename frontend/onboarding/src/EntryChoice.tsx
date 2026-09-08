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
      <section className="entry-intro" aria-labelledby="entry-intro-title">
        <div className="entry-intro__inner">
          <div>
            <h1 id="entry-intro-title" ref={headingRef} tabIndex={-1}>Your inner compass.</h1>
            <p className="entry-intro__description">
              Twinkl helps you compare everyday choices with your confirmed Core
              Values. It reviews your Journal Entries over time and shows the
              evidence when your behavior conflicts with what matters to you.
            </p>
          </div>
          <svg className="entry-intro__compass" viewBox="0 0 240 240" fill="none" aria-hidden="true">
            <circle cx="120" cy="120" r="102" stroke="currentColor" strokeWidth="1" />
            <circle cx="120" cy="120" r="76" stroke="currentColor" strokeDasharray="1 9" />
            <path d="M120 8v20m0 184v20M8 120h20m184 0h20" stroke="currentColor" />
            <g transform="rotate(28 120 120)">
              <path d="m120 40 19 80-19-8-19 8Z" fill="var(--apricot)" />
              <path d="m120 200 19-80-19 8-19-8Z" fill="var(--paper)" />
            </g>
            <circle cx="120" cy="120" r="4" fill="var(--paper)" />
          </svg>
        </div>
      </section>
      <section className="entry-choice__paths" aria-labelledby="entry-paths-title">
        <h2 id="entry-paths-title">Choose how to explore Twinkl.</h2>
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
      </section>
    </main>
  );
}
