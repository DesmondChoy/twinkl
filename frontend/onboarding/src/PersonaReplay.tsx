import {
  useEffect,
  useRef,
  useState,
  type RefObject,
} from "react";
import type { OnboardingProfile } from "./domain";
import {
  displayCoreValue,
  displayWeekRange,
} from "./displayFormatters";
import JournalExperience from "./JournalExperience";
import type { ExperienceState } from "./session";
import {
  loadSavedScenario,
  loadScenarioCatalog,
  type LoadedScenario,
  type ScenarioCatalog,
} from "./scenarioReplay";

const PLAYBACK_DELAY_MS = 2_400;

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
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [catalogAttempt, setCatalogAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    void loadScenarioCatalog()
      .then((loadedCatalog) => {
        if (cancelled) return;
        setCatalog(loadedCatalog);
        setSelectedId(
          loadedCatalog.scenarios.find(
            (item) => item.persona_id === currentPersonaId,
          )?.scenario_id ??
            loadedCatalog.scenarios.find((item) => item.recommended)
              ?.scenario_id ??
            loadedCatalog.scenarios[0].scenario_id,
        );
      })
      .catch(() => {
        if (!cancelled) {
          setError("The saved persona menu could not be loaded.");
        }
      });
    headingRef.current?.focus({ preventScroll: true });
    return () => {
      cancelled = true;
    };
  }, [catalogAttempt, currentPersonaId]);

  const selected = catalog?.scenarios.find(
    (item) => item.scenario_id === selectedId,
  ) ?? null;

  const startReplay = async () => {
    if (!selected || loading) return;
    setLoading(true);
    setError(null);
    try {
      if (!onLoad(await loadSavedScenario(selected))) {
        setLoading(false);
      }
    } catch {
      setError("This saved persona replay could not be loaded.");
      setLoading(false);
    }
  };

  return (
    <section className="persona-picker" aria-labelledby="persona-picker-title">
      <header className="persona-picker__header">
        <p className="eyebrow">Saved persona replay</p>
        <h1 id="persona-picker-title" ref={headingRef} tabIndex={-1}>
          Follow a life week by week.
        </h1>
        <p className="lede">
          Choose one synthetic persona to see how Journal Entries become Drift
          and a Weekly Digest over time.
        </p>
      </header>

      {catalog ? (
        <fieldset className="persona-menu" disabled={loading}>
          <legend>Choose a demo persona</legend>
          {catalog.scenarios.map((item) => (
            <label
              className={`persona-option${
                item.scenario_id === selectedId
                  ? " persona-option--selected"
                  : ""
              }`}
              key={item.scenario_id}
            >
              <input
                type="radio"
                name="persona"
                value={item.scenario_id}
                checked={item.scenario_id === selectedId}
                onChange={() => setSelectedId(item.scenario_id)}
              />
              <span className="persona-option__copy">
                <span className="persona-option__identity">
                  <strong>{item.persona_name}</strong>
                  {item.persona_id === currentPersonaId ? (
                    <em>Current</em>
                  ) : item.recommended ? (
                    <em>Recommended</em>
                  ) : null}
                </span>
                <span>
                  {item.profession} · {item.culture} · {item.age}
                </span>
                <span>{item.summary}</span>
                <small>
                  {item.core_values.map(displayCoreValue).join(" · ")} ·{" "}
                  {item.progression.length} weeks
                </small>
              </span>
            </label>
          ))}
        </fieldset>
      ) : null}

      <p className="persona-picker__source">
        Saved replay · AI-reviewed synthetic development evidence
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
        ) : (
          <button
            className="button button--primary"
            type="button"
            disabled={
              !selected ||
              loading ||
              selected.persona_id === currentPersonaId
            }
            onClick={() => void startReplay()}
          >
            {loading
              ? "Loading saved replay…"
              : selected?.persona_id === currentPersonaId
                ? "Current replay"
                : selected
                  ? `Replay ${selected.persona_name}`
                  : "Load persona"}
          </button>
        )}
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
  const [playing, setPlaying] = useState(false);
  const weeks = loaded.fixture.scenario.weeks;
  const safeWeekIndex = Math.min(Math.max(weekIndex, 0), weeks.length - 1);
  const currentWeek = weeks[safeWeekIndex];
  const isFirst = safeWeekIndex === 0;
  const isLast = safeWeekIndex === weeks.length - 1;

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
          <span>Saved replay</span>
          <span>AI-reviewed synthetic development evidence</span>
        </div>
        <p className="eyebrow">
          {loaded.catalogItem.profession} · {loaded.catalogItem.culture}
        </p>
        <h1 ref={headingRef} tabIndex={-1}>
          {loaded.catalogItem.persona_name}, week by week.
        </h1>
        <p className="lede">{loaded.catalogItem.summary}</p>
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

        <ol className="week-rail" aria-label="Saved replay weeks">
          {weeks.map((week, index) => (
            <li
              className={`week-rail__week week-rail__week--${
                week.expected_delivery_state
              }${
                index <= safeWeekIndex ? " week-rail__week--revealed" : ""
              }`}
              aria-current={index === safeWeekIndex ? "step" : undefined}
              aria-label={
                index <= safeWeekIndex
                  ? `Week ${index + 1}: ${
                    stateLabel(week.expected_delivery_state)
                  }`
                  : `Week ${index + 1}, not yet replayed`
              }
              key={week.week_id}
            >
              <span>{index + 1}</span>
            </li>
          ))}
        </ol>

        <div className="replay-controls__buttons">
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
            disabled={isFirst}
            onClick={() => {
              setPlaying(false);
              onWeekChange(0);
            }}
          >
            Restart scenario
          </button>
          <button
            className="inspect-run-link"
            type="button"
            onClick={() => {
              setPlaying(false);
              onChoosePersona();
            }}
          >
            Choose another persona
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
