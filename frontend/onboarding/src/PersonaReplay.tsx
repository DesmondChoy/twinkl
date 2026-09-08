import {
  useCallback,
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
import { isDisplayableNudge } from "./nudgeReveal";

function replayStateLabel(state: string): string {
  switch (state) {
    case "active_drift":
      return "Active Drift";
    case "insufficient_evidence":
      return "Insufficient Evidence";
    default:
      return "No Active Drift";
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
    case "drift_ended":
      return {
        label: "Pattern ended",
        copy: "See what ends an Active Drift run.",
      };
    case "insufficient_evidence":
      return {
        label: "Evidence limit",
        copy: "See Twinkl pause when evidence is unclear.",
      };
    case "two_core_values":
      return {
        label: "Two Core Values",
        copy: "See two priorities move independently.",
      };
    default:
      return {
        label: "No Active Drift",
        copy: `See ${item.progression.length} weeks without Active Drift.`,
      };
  }
}

function keyMomentState(
  role: ScenarioCatalogItem["role"],
): ScenarioDeliveryState | null {
  switch (role) {
    case "active_drift":
      return "active_drift";
    case "drift_ended":
      return "no_active_drift";
    case "insufficient_evidence":
      return "insufficient_evidence";
    case "two_core_values":
      return "insufficient_evidence";
    default:
      return null;
  }
}

function keyMomentIndexFor(
  item: ScenarioCatalogItem,
  weeks: LoadedScenario["fixture"]["scenario"]["weeks"],
): number {
  if (item.key_week_start != null) {
    return weeks.findIndex((week) => week.week_start === item.key_week_start);
  }
  const role = item.role;
  if (role === "drift_ended") {
    return weeks.findIndex(
      (week, index) =>
        week.expected_delivery_state === "no_active_drift"
        && weeks.slice(0, index).some(
          (prior) => prior.expected_delivery_state === "active_drift",
        ),
    );
  }
  const preferredState = keyMomentState(role);
  return preferredState === null
    ? weeks.length - 1
    : weeks.findIndex(
        (week) => week.expected_delivery_state === preferredState,
      );
}

function keyMomentLabel(role: ScenarioCatalogItem["role"], weekIndex: number): string {
  const purpose = role === "two_core_values"
    ? "independent Core Value states"
    : role === "drift_ended"
      ? "Drift ending"
      : replayStateLabel(keyMomentState(role) ?? "no_active_drift");
  return `Show ${purpose} — week ${weekIndex + 1}`;
}

function progressionGuide(item: ScenarioCatalogItem): string {
  const groups: { state: string; first: number; last: number }[] = [];
  item.progression.forEach((state, index) => {
    const previous = groups.at(-1);
    if (previous?.state === state) previous.last = index + 1;
    else groups.push({ state, first: index + 1, last: index + 1 });
  });
  return groups.map(({ state, first, last }) =>
    `${first === last ? `Week ${first}` : `Weeks ${first}–${last}`}: ${replayStateLabel(state)}`
  ).join("; ");
}

interface PersonaReplayPickerProps {
  currentPersonaId?: string | null;
  onBack: () => void;
  onLoad: (loaded: LoadedScenario) => boolean;
  onResume?: () => void;
}

export function PersonaReplayPicker({
  currentPersonaId = null,
  onBack,
  onLoad,
  onResume,
}: PersonaReplayPickerProps) {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const replayRequestGenerationRef = useRef(0);
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
      replayRequestGenerationRef.current += 1;
    };
  }, [catalogAttempt, currentPersonaId]);

  const startReplay = async (selected: ScenarioCatalogItem) => {
    if (loadingId !== null || selected.persona_id === currentPersonaId) return;
    const generation = ++replayRequestGenerationRef.current;
    setLoadingId(selected.scenario_id);
    setError(null);
    try {
      const loaded = await loadSavedScenario(selected);
      if (generation !== replayRequestGenerationRef.current) return;
      if (!onLoad(loaded)) {
        setLoadingId(null);
      }
    } catch {
      if (generation !== replayRequestGenerationRef.current) return;
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
                      disabled={(current && !onResume) || loadingId !== null}
                      onClick={() => current ? onResume?.() : void startReplay(item)}
                    >
                      {current
                        ? onResume ? "Continue replay" : "Current replay"
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

      {catalog ? (
        <details className="inspect-technical">
          <summary>Professor guide: states by week</summary>
          <p>These are saved results. Each Core Value is assessed independently.</p>
          <ul>
            {catalog.scenarios.map((item) => (
              <li key={item.scenario_id}>
                <strong>{item.persona_name}.</strong> {progressionGuide(item)}.
              </li>
            ))}
          </ul>
          <p>
            {catalog.scenarios.map((item) =>
              `${item.persona_name}: ${personaLesson(item).copy}`
            ).join(" ")}
          </p>
        </details>
      ) : null}

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
  const weekRailRef = useRef<HTMLOListElement>(null);
  const [reviewedWeekKey, setReviewedWeekKey] = useState<string | null>(null);
  const weeks = loaded.fixture.scenario.weeks;
  const safeWeekIndex = Math.min(Math.max(weekIndex, 0), weeks.length - 1);
  const currentWeek = weeks[safeWeekIndex];
  const currentWeekEntries = useMemo(() => {
    const entryIds = new Set(currentWeek.journal_entry_ids);
    return experience.journal_entries.filter((entry) =>
      entryIds.has(entry.journal_entry_id)
    );
  }, [currentWeek.journal_entry_ids, experience.journal_entries]);
  // Retain the saved step-count format so existing browser progress remains readable.
  const completedStage = currentWeek.journal_entry_ids.length + experience.nudges.filter(
    (nudge) => currentWeek.journal_entry_ids.includes(nudge.journal_entry_id)
      && isDisplayableNudge(nudge),
  ).length + 1;
  const weekKey = `${loaded.catalogItem.scenario_id}:${currentWeek.week_id}`;
  const resultVisible = reviewedWeekKey === weekKey;
  const progress = experience.replay_progress?.scenario_id === loaded.catalogItem.scenario_id
    ? experience.replay_progress
    : null;
  const furthestCompletedWeek = Math.min(
    progress?.furthest_completed_week ?? (safeWeekIndex === 0 ? -1 : safeWeekIndex),
    weeks.length - 1,
  );
  const recordProgress = useCallback((
    stage: number,
    week = safeWeekIndex,
    furthest = furthestCompletedWeek,
  ) => {
    updateExperience({ replay_progress: {
      scenario_id: loaded.catalogItem.scenario_id,
      week_index: week,
      reveal_stage: stage,
      furthest_completed_week: furthest,
    } });
  }, [furthestCompletedWeek, loaded.catalogItem.scenario_id, safeWeekIndex, updateExperience]);
  const isFirst = safeWeekIndex === 0;
  const isLast = safeWeekIndex === weeks.length - 1;
  const preferredKeyIndex = keyMomentIndexFor(loaded.catalogItem, weeks);
  const keyMomentIndex = preferredKeyIndex >= 0
    ? preferredKeyIndex
    : weeks.length - 1;
  const currentWeekEventIds = new Set(currentWeek.event_ids);
  const currentWeekEvents = [...experience.trace_events].reverse().filter(
    (event) => currentWeekEventIds.has(event.event_id),
  );
  const inspectEventId = [
    "north_star_reviewed", "weekly_coach_generated", "drift_detected", "weekly_digest_built",
  ].map((eventType) => currentWeekEvents.find((event) => event.event_type === eventType))
    .find((event) => event !== undefined)?.event_id ?? null;

  useEffect(() => {
    setReviewedWeekKey(null);
  }, [weekKey]);

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
    headingRef?.current?.focus({ preventScroll: true });
  }, [headingRef, loaded.catalogItem.scenario_id]);

  const showWeek = (index: number) => {
    if (index < 0 || index >= weeks.length) return;
    setReviewedWeekKey(null);
    onWeekChange(index);
    recordProgress(0, index);
  };

  const reviewWeek = () => {
    setReviewedWeekKey(weekKey);
    recordProgress(completedStage, safeWeekIndex, Math.max(furthestCompletedWeek, safeWeekIndex));
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
          <p>
            This Persona Profile is a synthetic projection. It does not
            represent a completed SVBWS assessment.
          </p>
          <button
            className="inspect-run-link"
            type="button"
            onClick={() => {
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
              Ready to review
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
                    showWeek(index);
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

        <div
          className="replay-controls__buttons"
          style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }}
        >
          <button
            className="button button--quiet"
            type="button"
            disabled={isFirst && furthestCompletedWeek < 0 && !resultVisible}
            onClick={() => {
              setReviewedWeekKey(null);
              onWeekChange(0);
              recordProgress(0, 0, -1);
            }}
          >
            Restart
          </button>
          <button
            className="button button--quiet"
            type="button"
            disabled={isFirst}
            onClick={() => {
              showWeek(safeWeekIndex - 1);
            }}
          >
            Previous
          </button>
          <button
            className="button button--primary"
            type="button"
            disabled={isLast || !resultVisible}
            onClick={() => showWeek(safeWeekIndex + 1)}
          >
            Next week
          </button>
          <button
            className="button button--quiet replay-controls__jump"
            style={{ gridColumn: "1 / -1", fontSize: "0.75rem" }}
            type="button"
            disabled={safeWeekIndex === keyMomentIndex && resultVisible}
            onClick={() => showWeek(keyMomentIndex)}
          >
            {keyMomentLabel(loaded.catalogItem.role, keyMomentIndex)}
          </button>
        </div>
      </section>

      <ReplayTimeline
        profile={profile}
        week={currentWeek}
        journalEntries={currentWeekEntries}
        nudges={experience.nudges}
        reviewedJournalEntries={experience.journal_entries}
        weeklyReviewerDecisions={experience.weekly_reviewer_decisions}
        reviewTraceEvents={experience.trace_events}
        selectedJournalEntryId={experience.selected_entry_id}
        cumulativeEntryCount={experience.journal_entries.length}
        resultVisible={resultVisible}
        onRevealResult={reviewWeek}
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
