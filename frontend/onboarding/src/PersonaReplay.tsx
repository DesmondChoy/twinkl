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

function personaLesson(item: ScenarioCatalogItem): { label: string; copy: string; key: string } {
  switch (item.role) {
    case "active_drift":
      return {
        label: "A Drift emerges",
        copy: "Active Drift appears in week 4, after three weeks without it. By week 5, the recorded pattern no longer continues.",
        key: "Week 4 · Active Drift appears",
      };
    case "persistent_drift":
      return {
        label: "Drift continues across weeks",
        copy: "The same Drift stays active through weeks 1–4. Repeated conflicts keep the pattern going; by week 5, it is no longer active.",
        key: "Week 4 · The Drift is still active",
      };
    case "insufficient_evidence":
      return {
        label: "Evidence limits a Drift claim",
        copy: "The first four weeks have No Active Drift. In weeks 5–6, the evidence is insufficient to determine the state; uncertainty does not become a Drift claim.",
        key: "Week 5 · Insufficient Evidence",
      };
    case "two_core_values":
      return {
        label: "Drift affects one Core Value",
        copy: "In week 1, choices under social pressure conflict with Self-Direction, while Tradition has No Active Drift. The two Core Values have different results in the same week.",
        key: "Week 1 · Two different results",
      };
    default:
      return {
        label: `${item.progression.length} weeks without Active Drift`,
        copy: "A baseline for comparison: no Active Drift is detected across the saved history. This does not mean every individual choice aligns with a Core Value.",
        key: `Week ${item.progression.length} · No Active Drift throughout`,
      };
  }
}

function keyMomentState(role: ScenarioCatalogItem["role"]): ScenarioDeliveryState | null {
  switch (role) {
    case "active_drift":
    case "persistent_drift":
    case "two_core_values":
      return "active_drift";
    case "insufficient_evidence":
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
  const preferredState = keyMomentState(item.role);
  return preferredState === null
    ? weeks.length - 1
    : weeks.findIndex((week) => week.expected_delivery_state === preferredState);
}

function keyMomentLabel(role: ScenarioCatalogItem["role"], weekIndex: number): string {
  const purpose = role === "two_core_values"
    ? "Drift affecting one Core Value"
    : role === "persistent_drift"
      ? "continuing Active Drift"
      : replayStateLabel(keyMomentState(role) ?? "no_active_drift");
  return `Show ${purpose} — week ${weekIndex + 1}`;
}

function useCompactPicker(): boolean {
  const [compact, setCompact] = useState(() =>
    window.matchMedia?.("(max-width: 1190px)").matches ?? false);
  useEffect(() => {
    const query = window.matchMedia?.("(max-width: 1190px)");
    if (!query) return;
    const update = () => setCompact(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return compact;
}

interface PersonaReplayPickerProps {
  currentPersonaId?: string | null;
  currentWeekIndex?: number | null;
  onLoad: (loaded: LoadedScenario) => boolean;
  onResume?: () => void;
}

export function PersonaReplayPicker({
  currentPersonaId = null,
  currentWeekIndex = null,
  onLoad,
  onResume,
}: PersonaReplayPickerProps) {
  const headingRef = useRef<HTMLHeadingElement>(null);
  const replayRequestGenerationRef = useRef(0);
  const [catalog, setCatalog] = useState<ScenarioCatalog | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [catalogAttempt, setCatalogAttempt] = useState(0);
  const compact = useCompactPicker();

  useEffect(() => {
    let cancelled = false;
    setError(null);
    void loadScenarioCatalog()
      .then((loadedCatalog) => {
        if (cancelled) return;
        setCatalog(loadedCatalog);
        setSelectedId((previous) => loadedCatalog.scenarios.find((item) =>
          item.persona_id === currentPersonaId || item.scenario_id === previous)?.scenario_id
          ?? loadedCatalog.scenarios.find((item) => item.recommended)!.scenario_id);
      })
      .catch(() => {
        if (!cancelled) setError("The saved Persona menu could not be loaded.");
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
      if (!onLoad(loaded)) setLoadingId(null);
    } catch {
      if (generation !== replayRequestGenerationRef.current) return;
      setError("This saved Persona replay could not be loaded. Try starting it again.");
      setLoadingId(null);
    }
  };

  const selected = catalog?.scenarios.find((item) => item.scenario_id === selectedId);
  const lesson = selected ? personaLesson(selected) : null;
  const current = selected?.persona_id === currentPersonaId;
  const detail = selected && lesson ? (
    <section className="persona-selection" aria-labelledby="selected-persona-title">
      <div className="persona-selection__overview" aria-live="polite" aria-atomic="true">
        <div className="persona-selection__identity">
          <p className="persona-selection__label">Selected Persona</p>
          <h2 id="selected-persona-title">{selected.persona_name}</h2>
          <p className="persona-selection__length">{selected.progression.length}-week replay</p>
        </div>
        <div className="persona-selection__story">
          <p className="persona-selection__description">{lesson.copy}</p>
          {selected.role === "two_core_values" ? (
            <dl className="persona-selection__values">
              <div><dt>Choosing her own path <small>Self-Direction</small></dt><dd>Active Drift</dd></div>
              <div><dt>Honoring traditions <small>Tradition</small></dt><dd>No Active Drift</dd></div>
            </dl>
          ) : null}
          <p className="persona-selection__key"><strong>Key week</strong>{lesson.key}</p>
          {selected.recommended ? (
            <p className="persona-selection__recommendation">A good first walkthrough: the earlier weeks make the change easy to follow.</p>
          ) : null}
        </div>
      </div>
      <div className="persona-selection__start">
        <p>{current ? "Continue from your saved week." : "Begin with the Journal Entries, then reveal the weekly reflection."}</p>
        <button className="button button--primary" type="button"
          disabled={(current && !onResume) || loadingId !== null}
          onClick={() => current ? onResume?.() : void startReplay(selected)}>
          {loadingId !== null ? "Loading saved replay…"
            : current ? `Continue replay · Week ${Math.min(Math.max(currentWeekIndex ?? 0, 0), selected.progression.length - 1) + 1}`
              : "Start at week 1"}
        </button>
      </div>
    </section>
  ) : null;
  const weekCount = Math.max(...(catalog?.scenarios.map((item) => item.progression.length) ?? [0]));

  return (
    <section className="persona-picker" aria-labelledby="persona-picker-title">
      <header className="persona-picker__header">
        <h1 id="persona-picker-title" ref={headingRef} tabIndex={-1}>See how Drift changes over time.</h1>
        <p className="lede">Choose a Persona to explore the Journal Entries and the recorded results each week.</p>
      </header>
      {catalog ? (
        <div className="persona-picker__comparison">
          <div className="persona-picker__matrix">
            <table className="persona-comparison" aria-label="Choose a demo Persona: states by week">
              <colgroup><col className="persona-comparison__identity-column" /><col className="persona-comparison__value-column" />
                {Array.from({ length: weekCount }, (_, index) => <col key={index} />)}</colgroup>
              <thead><tr><th scope="col" colSpan={2}>Persona / demonstration</th>
                {Array.from({ length: weekCount }, (_, index) => <th scope="col" key={index}>Week {index + 1}</th>)}
              </tr></thead>
              {catalog.scenarios.map((item) => {
                const valueKeys = item.role === "two_core_values" ? item.core_values : [null];
                const chosen = item.scenario_id === selectedId;
                const itemLesson = personaLesson(item);
                return (
                  <tbody key={item.scenario_id} className={`persona-comparison__persona${chosen ? " persona-comparison__persona--selected" : ""}`}>
                    {valueKeys.map((value, valueIndex) => (
                      <tr key={value ?? "overview"} onClick={() => {
                        if (loadingId === null) setSelectedId(item.scenario_id);
                      }}>
                        {valueIndex === 0 ? (
                          <th scope={valueKeys.length > 1 ? "rowgroup" : "row"} rowSpan={valueKeys.length}
                            colSpan={value === null ? 2 : 1} className="persona-comparison__identity">
                            <label>
                              <input type="radio" name="demo-persona" value={item.scenario_id}
                                checked={chosen} disabled={loadingId !== null}
                                aria-label={item.persona_name} aria-controls="selected-persona-title"
                                onChange={() => setSelectedId(item.scenario_id)} />
                              <span><strong>{item.persona_name}</strong><span>{itemLesson.label}</span>
                                {item.recommended ? <small>First walkthrough</small> : null}
                                {item.persona_id === currentPersonaId ? <small>Saved progress</small> : null}
                              </span>
                            </label>
                          </th>
                        ) : null}
                        {value !== null ? <th scope="row" className="persona-comparison__value">{VALUES[value as keyof typeof VALUES].name}</th> : null}
                        {Array.from({ length: weekCount }, (_, index) => {
                          const state = (value === null ? item.progression : item.core_value_progression[value])[index];
                          return <td key={index} data-week={`Week ${index + 1}`}
                            className={`persona-comparison__state persona-comparison__state--${state ?? "absent"}`}>
                            {state ? replayStateLabel(state) : <><span aria-hidden="true">—</span><span className="sr-only">Replay ends before this week</span></>}
                          </td>;
                        })}
                      </tr>
                    ))}
                    {compact && chosen ? <tr className="persona-comparison__detail"><td colSpan={weekCount + 2}>{detail}</td></tr> : null}
                  </tbody>
                );
              })}
            </table>
            <div className="persona-picker__legend">
              <p><span aria-hidden="true">—</span> Replay ends before this week.</p>
              <p>States are shown at each week’s close. Each Core Value has its own result in the replay.</p>
            </div>
          </div>
          {!compact ? detail : null}
        </div>
      ) : !error ? <p role="status">Loading saved Personas…</p> : null}
      {error ? <p className="persona-picker__error" role="alert">{error}</p> : null}
      {catalog === null && error ? <button className="button button--primary" type="button"
        onClick={() => setCatalogAttempt((value) => value + 1)}>Try loading again</button> : null}
      <p className="persona-picker__source">Saved replay · AI-reviewed synthetic development evidence · not human validation</p>
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
