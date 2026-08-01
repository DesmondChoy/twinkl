import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import {
  VALUES,
  type OnboardingProfile,
} from "./domain";
import { displayWeekRange } from "./displayFormatters";
import ReplayTimeline from "./ReplayTimeline";
import type { ExperienceState } from "./session";
import {
  loadSavedScenario,
  loadScenarioCatalog,
  type LoadedScenario,
  type ScenarioCatalog,
  type ScenarioCatalogItem,
} from "./scenarioReplay";
import type { ScenarioDeliveryState } from "./demoContracts";

const ENTRY_REVEAL_DELAY_MS = 3_600;
const RESULT_REVEAL_DELAY_MS = 3_200;
const NEXT_WEEK_DELAY_MS = 6_000;

function replayStateLabel(state: string): string {
  switch (state) {
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

function keyMomentState(
  role: ScenarioCatalogItem["role"],
): ScenarioDeliveryState | null {
  switch (role) {
    case "active_drift":
      return "active";
    case "recovered_drift":
      return "recovered";
    case "uncertain":
      return "uncertain";
    case "two_core_values":
      return "mixed";
    default:
      return null;
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

      <div id="experience-persona-options">
        {catalog ? (
          <div
            className="persona-menu"
            aria-label="Choose a demo Persona"
          >
            {[...catalog.scenarios]
              .sort(
                (left, right) =>
                  Number(right.recommended) - Number(left.recommended),
              )
              .map((item) => {
                const lesson = personaLesson(item);
                const current = item.persona_id === currentPersonaId;
                return (
                  <article
                    className={`persona-option persona-option--${item.role}${
                      current ? " persona-option--current" : ""
                    }`}
                    key={item.scenario_id}
                  >
                    <span
                      className="persona-option__thread"
                      aria-hidden="true"
                    />
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
      </div>

      <p className="persona-picker__source" id="experience-persona-source">
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
  const currentWeek = weeks[safeWeekIndex];
  const currentWeekEntries = useMemo(() => {
    const entryIds = new Set(currentWeek.journal_entry_ids);
    return experience.journal_entries.filter((entry) =>
      entryIds.has(entry.journal_entry_id)
    );
  }, [currentWeek.journal_entry_ids, experience.journal_entries]);
  const completedStage = currentWeekEntries.length + 1;
  const [revealStage, setRevealStage] = useState(() =>
    safeWeekIndex === 0 ? 0 : completedStage
  );
  const [furthestCompletedWeek, setFurthestCompletedWeek] = useState(
    safeWeekIndex === 0 ? -1 : safeWeekIndex,
  );
  const resultVisible = revealStage > currentWeekEntries.length;
  const isFirst = safeWeekIndex === 0;
  const isLast = safeWeekIndex === weeks.length - 1;
  const preferredKeyState = keyMomentState(loaded.catalogItem.role);
  const preferredKeyIndex = preferredKeyState === null
    ? weeks.length - 1
    : weeks.findIndex(
        (week) => week.expected_delivery_state === preferredKeyState,
      );
  const keyMomentIndex = preferredKeyIndex >= 0
    ? preferredKeyIndex
    : weeks.length - 1;
  const inspectEventId =
    [...experience.trace_events]
      .reverse()
      .find((event) => event.event_type === "weekly_coach_generated")?.event_id
    ?? [...experience.trace_events]
      .reverse()
      .find((event) => event.event_type === "drift_detected")?.event_id
    ?? [...experience.trace_events]
      .reverse()
      .find((event) => event.event_type === "weekly_digest_built")?.event_id
    ?? null;

  useEffect(() => {
    const restored = safeWeekIndex > 0;
    setPlaying(false);
    setRevealStage(restored ? currentWeek.journal_entry_ids.length + 1 : 0);
    setFurthestCompletedWeek(restored ? safeWeekIndex : -1);
  }, [loaded.catalogItem.scenario_id]);

  useEffect(() => {
    const activeWeek = weekRailRef.current?.querySelector<HTMLButtonElement>(
      '.week-rail__button[aria-current="step"]',
    );
    if (activeWeek && weekRailRef.current) {
      const railBounds = weekRailRef.current.getBoundingClientRect();
      const activeBounds = activeWeek.getBoundingClientRect();
      const activeCenter =
        activeBounds.left
        - railBounds.left
        + weekRailRef.current.scrollLeft
        + activeBounds.width / 2;
      weekRailRef.current.scrollLeft = Math.max(
        0,
        activeCenter - weekRailRef.current.clientWidth / 2,
      );
    }
  }, [safeWeekIndex]);

  useEffect(() => {
    if (reducedMotion && playing) setPlaying(false);
  }, [playing, reducedMotion]);

  useEffect(() => {
    if (!playing || reducedMotion) return;
    if (resultVisible && isLast) {
      setPlaying(false);
      return;
    }
    let delay = NEXT_WEEK_DELAY_MS;
    let advance = () => {
      setRevealStage(0);
      onWeekChange(safeWeekIndex + 1);
    };
    if (revealStage < currentWeekEntries.length) {
      delay = ENTRY_REVEAL_DELAY_MS;
      advance = () => setRevealStage((current) => current + 1);
    } else if (!resultVisible) {
      delay = RESULT_REVEAL_DELAY_MS;
      advance = () => {
        setRevealStage(completedStage);
        setFurthestCompletedWeek((current) =>
          Math.max(current, safeWeekIndex)
        );
      };
    }
    const timer = window.setTimeout(advance, delay);
    return () => window.clearTimeout(timer);
  }, [
    completedStage,
    currentWeekEntries.length,
    isLast,
    onWeekChange,
    playing,
    reducedMotion,
    resultVisible,
    revealStage,
    safeWeekIndex,
  ]);

  useEffect(() => {
    headingRef?.current?.focus({ preventScroll: true });
  }, [headingRef, loaded.catalogItem.scenario_id]);

  const showCompletedWeek = (index: number) => {
    if (index < 0 || index >= weeks.length) return;
    setPlaying(false);
    setRevealStage(weeks[index].journal_entry_ids.length + 1);
    setFurthestCompletedWeek((current) => Math.max(current, index));
    onWeekChange(index);
  };

  const advanceOneStep = () => {
    setPlaying(false);
    if (revealStage < currentWeekEntries.length) {
      setRevealStage((current) => current + 1);
      return;
    }
    if (!resultVisible) {
      setRevealStage(completedStage);
      setFurthestCompletedWeek((current) => Math.max(current, safeWeekIndex));
      return;
    }
    if (!isLast) {
      setRevealStage(0);
      onWeekChange(safeWeekIndex + 1);
    }
  };

  return (
    <div className="persona-replay">
      <h1 className="visually-hidden" ref={headingRef} tabIndex={-1}>
        {loaded.catalogItem.persona_name}
      </h1>
      <details className="replay-persona" id="experience-persona-profile">
        <summary>
          <span>
            <small>Persona · saved replay</small>
            <strong>{loaded.catalogItem.persona_name}</strong>
          </span>
          <span className="replay-persona__value">
            <small>
              Schwartz Core {profile.top_values.length === 1 ? "Value" : "Values"}
            </small>
            <span>
              {profile.top_values.map((value) => VALUES[value].name).join(" · ")}
            </span>
          </span>
          <span className="replay-persona__expand">Profile details</span>
        </summary>
        <div className="replay-persona__details">
          <p className="replay-persona__context">
            {loaded.catalogItem.summary}
          </p>
          <p>
            {loaded.catalogItem.profession} · {loaded.catalogItem.culture} ·{" "}
            age {loaded.catalogItem.age}
          </p>
          <p>
            <strong>Core Values:</strong>{" "}
            {profile.top_values.map((value) => VALUES[value].name).join(" · ")}
          </p>
          <div className="persona-replay__source-line">
            <span>Synthetic demo · saved replay</span>
            <span>AI-reviewed development evidence · not human validation</span>
          </div>
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
      </details>

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
          {resultVisible ? (
            <strong
              className={`replay-controls__state replay-controls__state--${
                currentWeek.expected_delivery_state
              }`}
            >
              {replayStateLabel(currentWeek.expected_delivery_state)}
            </strong>
          ) : (
            <span className="replay-controls__pending">
              {playing ? "Replaying…" : "Ready"}
            </span>
          )}
        </div>

        <ol
          className="week-rail"
          aria-label="Saved replay weeks"
          ref={weekRailRef}
        >
          {weeks.map((week, index) => {
            const revealed =
              index <= furthestCompletedWeek
              || (index === safeWeekIndex && resultVisible);
            const label = revealed
              ? `Week ${index + 1}: ${
                replayStateLabel(week.expected_delivery_state)
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
                  disabled={!revealed || (index === safeWeekIndex && !resultVisible)}
                  aria-current={
                    index === safeWeekIndex ? "step" : undefined
                  }
                  aria-label={
                    revealed
                      ? `Show ${label.toLowerCase()}`
                      : `Show week ${index + 1}, outcome hidden`
                  }
                  onClick={() => {
                    if (!revealed || (index === safeWeekIndex && !resultVisible)) {
                      return;
                    }
                    showCompletedWeek(index);
                  }}
                >
                  <span>W{index + 1}</span>
                  {revealed ? (
                    <small>{replayStateLabel(week.expected_delivery_state)}</small>
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
            disabled={isFirst && furthestCompletedWeek < 0 && revealStage === 0}
            onClick={() => {
              setPlaying(false);
              setFurthestCompletedWeek(-1);
              setRevealStage(0);
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
              showCompletedWeek(safeWeekIndex - 1);
            }}
          >
            Previous
          </button>
          <button
            className="button replay-controls__play"
            type="button"
            disabled={reducedMotion || (isLast && resultVisible && !playing)}
            aria-describedby={reducedMotion ? "reduced-motion-note" : undefined}
            onClick={() => setPlaying((current) => !current)}
          >
            {playing ? "Pause replay" : "Auto replay"}
          </button>
          <button
            className="button button--primary"
            type="button"
            disabled={isLast && resultVisible}
            onClick={advanceOneStep}
          >
            Next step
          </button>
          <button
            className="button button--quiet replay-controls__jump"
            type="button"
            disabled={safeWeekIndex === keyMomentIndex && resultVisible}
            onClick={() => showCompletedWeek(keyMomentIndex)}
          >
            Jump to key moment
          </button>
        </div>

        {reducedMotion ? (
          <p className="replay-controls__motion-note" id="reduced-motion-note">
            Automatic replay is off because reduced motion is enabled. Previous
            and Next step remain available.
          </p>
        ) : null}
      </section>

      <ReplayTimeline
        profile={profile}
        week={currentWeek}
        journalEntries={currentWeekEntries}
        reviewedJournalEntries={experience.journal_entries}
        weeklyReviewerDecisions={experience.weekly_reviewer_decisions}
        reviewTraceEvents={experience.trace_events}
        selectedJournalEntryId={experience.selected_entry_id}
        cumulativeEntryCount={experience.journal_entries.length}
        visibleEntryCount={Math.min(revealStage, currentWeekEntries.length)}
        resultVisible={resultVisible}
        playing={playing}
        driftResult={experience.drift_result}
        weeklyDigest={experience.weekly_digest}
        inspectRun={inspectRun}
        inspectEventId={inspectEventId}
        onSelectJournalEntry={(journalEntryId) =>
          updateExperience({ selected_entry_id: journalEntryId })
        }
      />
    </div>
  );
}
