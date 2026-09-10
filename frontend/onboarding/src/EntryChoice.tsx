import type { Ref } from "react";
import LandingCompass from "./LandingCompass";

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
        <div className="entry-intro__sky" aria-hidden="true">
          <span className="entry-intro__shooting-star" />
        </div>
        <div className="entry-intro__inner">
          <div>
            <h1 id="entry-intro-title" ref={headingRef} tabIndex={-1}>Can AI Recognize When Behavior Conflicts with Values?</h1>
            <p className="entry-intro__description">
              A capstone project combining behavioral science, longitudinal
              reasoning, and systematic experimentation to investigate how AI can
              detect repeated conflict between people’s stated priorities and
              their reported behavior. Model comparisons and ablation studies on
              synthetic Journal Entries inform its design.
            </p>
            <p className="entry-intro__description">
              Grounded in Schwartz’s established theory of human values, Twinkl
              addresses a risk in AI-assisted reflection: affirming someone’s
              account without examining the contradictions within it.
            </p>
          </div>
          <LandingCompass />
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
