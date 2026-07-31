import {
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import {
  VALUES,
  type OnboardingProfile,
} from "./domain";
import { displayWeekRange } from "./displayFormatters";
import JournalExperience from "./JournalExperience";
import type { ExperienceState } from "./session";
import {
  loadSavedScenario,
  loadScenarioCatalog,
  type LoadedScenario,
  type ScenarioCatalog,
  type ScenarioCatalogItem,
} from "./scenarioReplay";

const PLAYBACK_DELAY_MS = 4_000;

function stateLabel(value: string): string {
  switch (value) {
    case "active":
      return "Active Drift";
    case "recovered":
      return "Recovered Drift";
    case "uncertain":
      return "Uncertain";
    case "mixed":
      return "Mixed";
    default:
      return "No Drift";
  }
}

function personaLesson(item: ScenarioCatalogItem): {
  label: string;
  copy: string;
} {
  switch (item.role) {
    case "active_drift":
      return {
        label: "Emergence",
        copy: "Watch a pattern become Active Drift.",
      };
    case "recovered_drift":
      return {
        label: "Recovery",
        copy: "See what ends an Active Drift run.",
      };
    case "uncertain":
      return {
        label: "Uncertainty",
        copy: "See Twinkl pause when evidence is unclear.",
      };
    case "two_core_values":
      return {
        label: "Two Core Values",
        copy: "See two priorities move independently.",
      };
    default:
      return {
        label: "Steady",
        copy: `See ${item.progression.length} weeks with No Drift.`,
      };
  }
}

function usePrefersReducedMotion(): boolean {
  const query = "(prefers-reduced-motion: reduce)";
  const [reduced, setReduced] = useState(
    () => window.matchMedia?.(query).matches ?? false,
  );

  useEffect(() => {
    const media = window.matchMedia?.(query);
    if (!media) return;
    const update = () => setReduced(media.matches);
    update();
    media.addEventListener?.("change", update);
    return () => media.removeEventListener?.("change", update);
  }, []);

  return reduced;
}

interface PersonaReplayPickerProps {
  currentPersonaId?: string | null;
  onBack: () => void;
  onLoad: (loaded: LoadedScenario) => boolean;
}

export function PersonaReplayPicker({
  currentPersonaId = null,
  onBack,
  onLoad,
}: PersonaReplayPickerProps) {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const [catalog, setCatalog] = useState<ScenarioCatalog | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [catalogAttempt, setCatalogAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    void loadScenarioCatalog()
      .then((loadedCatalog) => {
        if (cancelled) return;
        setCatalog(loadedCatalog);
      })
      .catch(() => {
        if (!cancelled) {
          setError("The saved Persona menu could not be loaded.");
        }
      });
    headingRef.current?.focus({ preventScroll: true });
    return () => {
      cancelled = true;
    };
  }, [catalogAttempt, currentPersonaId]);

  const startReplay = async (selected: ScenarioCatalogItem) => {
    if (loadingId !== null || selected.persona_id === currentPersonaId) return;
    setLoadingId(selected.scenario_id);
    setError(null);
    try {
      if (!onLoad(await loadSavedScenario(selected))) {
        setLoadingId(null);
      }
    } catch {
      setError("This saved Persona replay could not be loaded.");
      setLoadingId(null);
    }
  };

  return (
    <section className="persona-picker" aria-labelledby="persona-picker-title">
      <header className="persona-picker__header">
        <p className="eyebrow">Saved Persona replay</p>
        <h1 id="persona-picker-title" ref={headingRef} tabIndex={-1}>
          Choose what you want to observe.
        </h1>
        <p className="lede">
          Five saved stories show different ways Drift can unfold.
        </p>
      </header>

      {catalog ? (
        <div className="persona-menu" aria-label="Choose a demo Persona">
          {catalog.scenarios.map((item) => {
            const lesson = personaLesson(item);
            const current = item.persona_id === currentPersonaId;
            return (
              <article
                className={`persona-option persona-option--${item.role}${
                  current ? " persona-option--current" : ""
                }`}
                key={item.scenario_id}
              >
                <span className="persona-option__thread" aria-hidden="true" />
                <span className="persona-option__copy">
                  <span className="persona-option__identity">
                    <strong>{item.persona_name}</strong>
                    {current ? (
                      <em>Current</em>
                    ) : item.recommended ? (
                      <em>Recommended</em>
                    ) : null}
                  </span>
                  <span>
                    {item.profession} · {item.culture} · {item.age}
                  </span>
                  <span className="persona-option__lesson">
                    <small>{lesson.label}</small>
                    <span>{lesson.copy}</span>
                  </span>
                </span>
                <button
                  className="button button--primary persona-option__action"
                  type="button"
                  disabled={current || loadingId !== null}
                  onClick={() => void startReplay(item)}
                >
                  {current
                    ? "Current replay"
                    : loadingId === item.scenario_id
                      ? "Loading saved replay…"
                      : "Start at week 1"}
                </button>
              </article>
            );
          })}
        </div>
      ) : null}

      <p className="persona-picker__source">
        Saved replay · AI-reviewed synthetic development evidence · not human
        validation
      </p>
      {error ? <p className="persona-picker__error" role="alert">{error}</p> : null}
      <div className="persona-picker__actions">
        <button className="button button--quiet" type="button" onClick={onBack}>
          Back
        </button>
        {catalog === null && error ? (
          <button
            className="button button--primary"
            type="button"
            onClick={() => setCatalogAttempt((value) => value + 1)}
          >
            Try loading again
          </button>
        ) : null}
      </div>
    </section>
  );
}

interface PersonaReplayExperienceProps {
  loaded: LoadedScenario;
  weekIndex: number;
  profile: OnboardingProfile;
  experience: ExperienceState;
  updateExperience: (patch: Partial<ExperienceState>) => void;
  inspectRun: (eventId: string) => void;
  onChoosePersona: () => void;
  onWeekChange: (weekIndex: number) => void;
  headingRef?: RefObject<HTMLHeadingElement | null>;
}

export function PersonaReplayExperience({
  loaded,
  weekIndex,
  profile,
  experience,
  updateExperience,
  inspectRun,
  onChoosePersona,
  onWeekChange,
  headingRef,
}: PersonaReplayExperienceProps) {
  const reducedMotion = usePrefersReducedMotion();
  const weekRailRef = useRef<HTMLOListElement>(null);
  const [playing, setPlaying] = useState(false);
  const weeks = loaded.fixture.scenario.weeks;
  const safeWeekIndex = Math.min(Math.max(weekIndex, 0), weeks.length - 1);
  const [furthestRevealedWeek, setFurthestRevealedWeek] =
    useState(safeWeekIndex);
  const currentWeek = weeks[safeWeekIndex];
  const isFirst = safeWeekIndex === 0;
  const isLast = safeWeekIndex === weeks.length - 1;

  useEffect(() => {
    setFurthestRevealedWeek(safeWeekIndex);
  }, [loaded.catalogItem.scenario_id]);

  useEffect(() => {
    setFurthestRevealedWeek((current) =>
      Math.max(current, safeWeekIndex),
    );
  }, [safeWeekIndex]);

  useEffect(() => {
    const activeWeek = weekRailRef.current?.querySelector<HTMLButtonElement>(
      '.week-rail__button[aria-current="step"]',
    );
    activeWeek?.scrollIntoView?.({
      block: "nearest",
      inline: "center",
    });
  }, [safeWeekIndex]);

  useEffect(() => {
    if (reducedMotion && playing) setPlaying(false);
  }, [playing, reducedMotion]);

  useEffect(() => {
    if (!playing || reducedMotion) return;
    if (isLast) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(
      () => onWeekChange(safeWeekIndex + 1),
      PLAYBACK_DELAY_MS,
    );
    return () => window.clearTimeout(timer);
  }, [isLast, onWeekChange, playing, reducedMotion, safeWeekIndex]);

  useEffect(() => {
    headingRef?.current?.focus({ preventScroll: true });
  }, [headingRef, loaded.catalogItem.scenario_id]);

  return (
    <div className="persona-replay">
      <header className="persona-replay__header">
        <div className="persona-replay__source-line">
          <span>Synthetic demo · saved replay</span>
          <span>Not human validation</span>
        </div>
        <p className="eyebrow">
          {loaded.catalogItem.profession} · {loaded.catalogItem.culture} ·{" "}
          {loaded.catalogItem.age}
        </p>
        <h1 ref={headingRef} tabIndex={-1}>
          {loaded.catalogItem.persona_name}
        </h1>
        <p className="persona-replay__values">
          <strong>What matters to them:</strong>{" "}
          {profile.top_values.map((value) => VALUES[value].phrase).join(" · ")}
        </p>
      </header>

      <section
        className="replay-controls"
        aria-labelledby="replay-week-title"
      >
        <div
          className="replay-controls__week"
          aria-atomic="true"
          aria-live="polite"
        >
          <div>
            <p className="eyebrow">
              Week {safeWeekIndex + 1} of {weeks.length}
            </p>
            <h2 id="replay-week-title">
              {displayWeekRange(
                currentWeek.week_start,
                currentWeek.week_end,
              )}
            </h2>
          </div>
          <strong
            className={`replay-controls__state replay-controls__state--${
              currentWeek.expected_delivery_state
            }`}
          >
            {stateLabel(currentWeek.expected_delivery_state)}
          </strong>
        </div>

        <ol
          className="week-rail"
          aria-label="Saved replay weeks"
          ref={weekRailRef}
        >
          {weeks.map((week, index) => {
            const revealed = index <= furthestRevealedWeek;
            const label = revealed
              ? `Week ${index + 1}: ${
                stateLabel(week.expected_delivery_state)
              }`
              : `Week ${index + 1}, not yet replayed`;
            return (
              <li
                className={`week-rail__week${
                  revealed
                    ? ` week-rail__week--revealed week-rail__week--${week.expected_delivery_state}`
                    : ""
                }`}
                aria-current={index === safeWeekIndex ? "step" : undefined}
                aria-label={label}
                key={week.week_id}
              >
                <button
                  type="button"
                  className="week-rail__button"
                  aria-current={
                    index === safeWeekIndex ? "step" : undefined
                  }
                  aria-label={
                    revealed
                      ? `Show ${label.toLowerCase()}`
                      : `Show week ${index + 1}, outcome hidden`
                  }
                  onClick={() => {
                    setPlaying(false);
                    onWeekChange(index);
                  }}
                >
                  <span>W{index + 1}</span>
                  {revealed ? (
                    <small>{stateLabel(week.expected_delivery_state)}</small>
                  ) : null}
                </button>
              </li>
            );
          })}
        </ol>

        <div className="replay-controls__buttons">
          <button
            className="button button--quiet"
            type="button"
            disabled={isFirst && furthestRevealedWeek === 0}
            onClick={() => {
              setPlaying(false);
              setFurthestRevealedWeek(0);
              onWeekChange(0);
            }}
          >
            Restart
          </button>
          <button
            className="button button--quiet"
            type="button"
            disabled={isFirst}
            onClick={() => {
              setPlaying(false);
              onWeekChange(safeWeekIndex - 1);
            }}
          >
            Previous
          </button>
          <button
            className="button replay-controls__play"
            type="button"
            disabled={reducedMotion || (isLast && !playing)}
            aria-describedby={reducedMotion ? "reduced-motion-note" : undefined}
            onClick={() => setPlaying((current) => !current)}
          >
            {playing ? "Pause" : "Play"}
          </button>
          <button
            className="button button--primary"
            type="button"
            disabled={isLast}
            onClick={() => {
              setPlaying(false);
              onWeekChange(safeWeekIndex + 1);
            }}
          >
            Next
          </button>
        </div>

        {reducedMotion ? (
          <p className="replay-controls__motion-note" id="reduced-motion-note">
            Automatic replay is off because reduced motion is enabled. Previous
            and Next remain available.
          </p>
        ) : null}

        <div className="replay-controls__secondary">
          <button
            className="inspect-run-link"
            type="button"
            onClick={() => {
              setPlaying(false);
              onChoosePersona();
            }}
          >
            Choose another Persona
          </button>
        </div>
      </section>

      <JournalExperience
        profile={profile}
        experience={experience}
        updateExperience={updateExperience}
        inspectRun={inspectRun}
        mode="saved_replay"
      />
    </div>
  );
}
