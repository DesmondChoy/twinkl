import {
  Component,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ErrorInfo,
  type ReactNode,
} from "react";
import {
  BWS_OBJECTS,
  BWS_SETS,
  VALUES,
  createProfile,
  normalizePreferredName,
  scoreResponses,
  type BwsObjectKey,
  type OnboardingProfile,
} from "./domain";
import AssessmentSectionMap from "./AssessmentSectionMap";
import CoreValueReminder from "./CoreValueReminder";
import ExperienceSectionMap, {
  type ExperienceSectionMapView,
} from "./ExperienceSectionMap";
import {
  createExperienceSession,
  ExperienceApiError,
  readExperienceTrace,
} from "./experienceApi";
import {
  clearSession,
  clearChoice,
  setChoice,
  type OnboardingSession,
} from "./session";
import InspectView from "./InspectView";
import JournalExperience from "./JournalExperience";
import LivingCompass from "./LivingCompass";
import {
  PersonaReplayExperience,
  PersonaReplayPicker,
} from "./PersonaReplay";
import {
  loadSavedScenarioById,
  projectScenarioWeek,
  type LoadedScenario,
} from "./scenarioReplay";
import { SharedSessionProvider, useSharedSession } from "./sharedSession";

const AUTO_ADVANCE_DELAY_MS = 1_000;
const CARD_BACKGROUNDS = [
  "/card-backgrounds/memory-atlas-01.jpg",
  "/card-backgrounds/memory-atlas-02.jpg",
  "/card-backgrounds/memory-atlas-03.jpg",
  "/card-backgrounds/memory-atlas-04.jpg",
  "/card-backgrounds/memory-atlas-05.jpg",
  "/card-backgrounds/memory-atlas-06.jpg",
] as const;

export class AppErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Experience recovery boundary", error, info);
  }

  private restart = () => {
    clearSession();
    window.location.reload();
  };

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="app-shell">
        <header className="topbar">
          <a className="wordmark" href="/">
            twinkl<span>·</span>
          </a>
        </header>
        <main className="app-recovery" id="main">
          <p className="eyebrow">Experience recovery</p>
          <h1>This saved view could not be restored.</h1>
          <p>Your browser data can be cleared without changing project files.</p>
          <button
            className="button button--primary"
            type="button"
            onClick={this.restart}
          >
            Start over
          </button>
        </main>
      </div>
    );
  }
}

function milestoneFor(session: OnboardingSession): number {
  if (session.stage === "name") return 0;
  if (session.stage === "set") {
    return Math.min(session.set_index + 1, BWS_SETS.length);
  }
  return BWS_SETS.length + 1;
}

function Progress({ session }: { session: OnboardingSession }) {
  const label =
    session.stage === "set"
      ? `Values · ${session.set_index + 1} of ${BWS_SETS.length}`
      : "Your compass";
  const completedSets =
    session.stage === "name"
      ? 0
      : session.stage === "set"
        ? session.set_index + 1
        : BWS_SETS.length;
  return (
    <div
      className="progress"
      id="assessment-progress"
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={BWS_SETS.length}
      aria-valuenow={completedSets}
      aria-valuetext={label}
    >
      <div className="progress__label">
        <span>{label}</span>
      </div>
      <div className="progress__track">
        <span style={{ width: `${(completedSets / BWS_SETS.length) * 100}%` }} />
      </div>
    </div>
  );
}

type CardLocation = "pool" | "most" | "least";
type DropTarget = CardLocation | null;

interface DraggableCardProps {
  value: BwsObjectKey;
  location: CardLocation;
  index: number;
  locateTarget: (clientX: number, clientY: number) => DropTarget;
  locked: boolean;
  nextChoice: "most" | "least" | null;
  onDragTarget: (target: DropTarget) => void;
  onMove: (
    value: BwsObjectKey,
    from: CardLocation,
    to: CardLocation,
    focusAfterMove?: boolean,
  ) => void;
  onTap: (value: BwsObjectKey, location: CardLocation) => void;
}

function DraggableCard({
  value,
  location,
  index,
  locateTarget,
  locked,
  nextChoice,
  onDragTarget,
  onMove,
  onTap,
}: DraggableCardProps) {
  const dragRef = useRef<{ pointerId: number; x: number; y: number; moved: boolean } | null>(null);
  const suppressClickRef = useRef(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const dragging = dragRef.current !== null;
  const phrase = BWS_OBJECTS[value].descriptor;

  const finishDrag = () => {
    dragRef.current = null;
    setOffset({ x: 0, y: 0 });
    onDragTarget(null);
  };

  const keyboardMove = (event: React.KeyboardEvent<HTMLElement>) => {
    if (locked) return;
    const key = event.key.toLowerCase();
    if (key === "enter" || key === " ") {
      event.preventDefault();
      onTap(value, location);
      return;
    }
    const target = key === "m" ? "most" : key === "l" ? "least" : null;
    if (target) {
      event.preventDefault();
      onMove(value, location, target, true);
      return;
    }
    if (location !== "pool" && ["backspace", "delete", "arrowdown"].includes(key)) {
      event.preventDefault();
      onMove(value, location, "pool", true);
    }
  };

  const keyboardHint =
    locked
      ? "Selections saved. Moving to the next group."
      : location === "pool"
        ? `Tap to choose this card as ${nextChoice === "least" ? "Least" : "Most"}, or press M for Most and L for Least.`
        : `Selected as ${location}. Tap, press Backspace, or press Arrow Down to return it.`;

  return (
    <article
      className={`value-card${location === "pool" ? "" : ` value-card--${location} value-card--placed`}${locked ? " value-card--locked" : ""}${dragging ? " value-card--dragging" : ""}`}
      data-testid="value-card"
      data-value={value}
      data-location={location}
      data-background-position={index}
      role="button"
      tabIndex={locked ? -1 : 0}
      aria-disabled={locked || undefined}
      aria-label={`${phrase}. ${keyboardHint}`}
      onKeyDown={keyboardMove}
      onClick={(event) => {
        event.stopPropagation();
        if (locked) return;
        if (suppressClickRef.current) {
          suppressClickRef.current = false;
          return;
        }
        onTap(value, location);
      }}
      onPointerDown={(event) => {
        if (locked) return;
        if (event.pointerType === "touch") return;
        if (event.pointerType === "mouse" && event.button !== 0) return;
        dragRef.current = { pointerId: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
        event.currentTarget.setPointerCapture?.(event.pointerId);
        setOffset({ x: 0.01, y: 0 });
      }}
      onPointerMove={(event) => {
        const drag = dragRef.current;
        if (!drag || drag.pointerId !== event.pointerId) return;
        if (Math.abs(event.clientX - drag.x) > 8 || Math.abs(event.clientY - drag.y) > 8) {
          drag.moved = true;
        }
        setOffset({ x: event.clientX - drag.x, y: event.clientY - drag.y });
        onDragTarget(locateTarget(event.clientX, event.clientY));
      }}
      onPointerUp={(event) => {
        const drag = dragRef.current;
        if (!drag || drag.pointerId !== event.pointerId) return;
        const wasDrag = drag.moved;
        const locatedTarget = locateTarget(event.clientX, event.clientY);
        const distanceY = event.clientY - drag.y;
        let target = locatedTarget;
        if (location === "most" && locatedTarget !== "least" && distanceY > 45) {
          target = "pool";
        } else if (location === "least" && locatedTarget !== "most" && distanceY < -45) {
          target = "pool";
        } else if (location === "pool" && locatedTarget !== "pool" && distanceY < -45) {
          target = "most";
        } else if (location === "pool" && locatedTarget !== "pool" && distanceY > 45) {
          target = "least";
        }
        event.currentTarget.releasePointerCapture?.(event.pointerId);
        finishDrag();
        if (wasDrag && target && target !== location) onMove(value, location, target);
        if (wasDrag) {
          suppressClickRef.current = true;
          window.setTimeout(() => {
            suppressClickRef.current = false;
          }, 0);
        }
      }}
      onPointerCancel={finishDrag}
      style={
        {
          "--card-accent": "#5576d9",
          "--card-angle": `${(index - 2.5) * 0.8}deg`,
          "--card-delay": `${index * 70}ms`,
          "--card-background-image": `url("${CARD_BACKGROUNDS[index]}")`,
          "--drag-x": `${offset.x}px`,
          "--drag-y": `${offset.y}px`,
        } as React.CSSProperties
      }
    >
      <span className="value-card__phrase">{phrase}</span>
    </article>
  );
}

interface AppProps {
  onStartJournal?: (profile: NonNullable<OnboardingSession["confirmed_profile"]>) => void;
}

function ExperienceInspectApp({ onStartJournal }: AppProps = {}) {
  const {
    session,
    persistenceError,
    updateSession,
    updateExperience,
    showView,
    inspectRun,
    restart: restartSession,
  } = useSharedSession();
  const [activeDrop, setActiveDrop] = useState<DropTarget>(null);
  const [personaPickerOpen, setPersonaPickerOpen] = useState(false);
  const [loadedScenario, setLoadedScenario] = useState<LoadedScenario | null>(
    null,
  );
  const [scenarioLoadError, setScenarioLoadError] = useState<string | null>(
    null,
  );
  const [scenarioLoadAttempt, setScenarioLoadAttempt] = useState(0);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const mostDropRef = useRef<HTMLElement>(null);
  const leastDropRef = useRef<HTMLElement>(null);
  const selectionRef = useRef<HTMLDivElement>(null);
  const pendingCardFocusRef = useRef<{ value: BwsObjectKey; location: CardLocation } | null>(null);
  const choicesCompletedAtRef = useRef<number | null>(null);
  const profileSyncGenerationRef = useRef(0);
  const profileSyncInFlightRef = useRef<{
    generation: number;
    sessionId: string;
  } | null>(null);
  const milestone = milestoneFor(session);
  const journalStarted = session.experience.journal_started;
  const activeView = session.experience.active_view;
  const selectedPersonaId = session.experience.selected_persona_id;
  const inspectAvailable = session.stage !== "set"
    && session.responses.length === BWS_SETS.length;
  const profileAwaitingConfirmation = session.stage === "summary";
  const profileTraceReady = session.confirmed_profile !== null
    && session.experience.trace_events.some(
      (event) =>
        event.event_type === "profile_confirmed"
        && event.session_id === session.confirmed_profile?.session_id,
    );
  const profileTracePending = !profileTraceReady
    && session.experience.run_state === "running";
  const profileTraceFailed = !profileTraceReady
    && session.experience.run_state === "failed";
  const currentSetIndex = session.set_order[session.set_index];
  const currentSet = BWS_SETS[currentSetIndex];
  const currentOrder = session.displayed_orders[currentSetIndex];
  const availableValues = currentOrder.filter(
    (value) => value !== session.draft_best && value !== session.draft_worst,
  );
  const isReviewing = session.stage === "set" && Boolean(session.draft_best && session.draft_worst);
  const nextChoice: "most" | "least" = session.draft_best ? "least" : "most";
  const personaReplayReady = Boolean(
    selectedPersonaId
    && loadedScenario?.catalogItem.persona_id === selectedPersonaId,
  );
  const experienceSectionView: ExperienceSectionMapView | null =
    personaPickerOpen
      ? "persona-picker"
      : personaReplayReady
        ? "persona-replay"
        : !selectedPersonaId && session.stage === "summary"
          ? "summary"
          : !selectedPersonaId && session.stage === "complete" && !journalStarted
            ? "complete"
            : !selectedPersonaId && journalStarted
              ? "journal"
              : null;
  const showProfileCompass = !selectedPersonaId
    && !journalStarted
    && (session.stage === "summary" || session.stage === "complete");

  const update = (patch: Partial<OnboardingSession>) => {
    updateSession(patch);
  };

  const synchronizeProfileTrace = useCallback(async (
    profile: OnboardingProfile,
  ): Promise<boolean> => {
    if (profileSyncInFlightRef.current?.sessionId === profile.session_id) {
      return false;
    }
    const generation = ++profileSyncGenerationRef.current;
    profileSyncInFlightRef.current = {
      generation,
      sessionId: profile.session_id,
    };
    updateExperience({
      run_state: "running",
      retryable: false,
      error_message: null,
    });
    try {
      const response = await createExperienceSession(profile);
      const trace = await readExperienceTrace(profile.session_id);
      if (
        !trace.events.some(
          (event) =>
            event.event_type === "profile_confirmed"
            && event.session_id === profile.session_id,
        )
      ) {
        throw new ExperienceApiError(
          "The Profile trace did not include Profile confirmation.",
          "missing_profile_trace",
        );
      }
      if (profileSyncGenerationRef.current !== generation) return false;
      updateExperience({
        revision: response.session.revision,
        journal_entries: response.session.journal_entries,
        nudges: response.session.nudges,
        weekly_reviewer_decisions:
          response.session.weekly_reviewer_decisions,
        drift_result: response.session.drift_result,
        weekly_digest: response.session.weekly_digest,
        assessment_clock: response.session.assessment_clock,
        weekly_coach: null,
        run_state: "idle",
        retryable: false,
        error_message: null,
        trace_event_ids: response.session.trace_event_ids,
        trace_events: trace.events,
      });
      return true;
    } catch (error) {
      if (profileSyncGenerationRef.current !== generation) return false;
      updateExperience({
        run_state: "failed",
        retryable:
          error instanceof ExperienceApiError ? error.retryable : true,
        error_message:
          "Your Profile is saved in this browser, but its Inspect trace could not be loaded.",
      });
      return false;
    } finally {
      if (profileSyncInFlightRef.current?.generation === generation) {
        profileSyncInFlightRef.current = null;
      }
    }
  }, [updateExperience]);

  useLayoutEffect(() => {
    document.documentElement.scrollTop = 0;
    document.body.scrollTop = 0;
    if (activeView === "experience") {
      headingRef.current?.focus({ preventScroll: true });
    }
  }, [
    session.stage,
    session.set_index,
    journalStarted,
    activeView,
    selectedPersonaId,
    personaPickerOpen,
  ]);

  useEffect(() => {
    const profile = session.confirmed_profile;
    if (
      !profile
      || selectedPersonaId !== null
      || journalStarted
      || session.stage !== "complete"
      || profileTraceReady
      || session.experience.journal_entries.length > 0
      || session.experience.nudges.length > 0
      || session.experience.run_state !== "idle"
    ) {
      return;
    }
    void synchronizeProfileTrace(profile);
  }, [
    profileTraceReady,
    journalStarted,
    selectedPersonaId,
    session.confirmed_profile,
    session.experience.journal_entries.length,
    session.experience.nudges.length,
    session.experience.run_state,
    session.stage,
    synchronizeProfileTrace,
  ]);

  useLayoutEffect(() => {
    const target = pendingCardFocusRef.current;
    if (!target) return;
    const container = target.location === "most"
      ? mostDropRef.current
      : target.location === "least"
        ? leastDropRef.current
        : selectionRef.current;
    const card = container?.querySelector<HTMLElement>(
      `[data-value="${target.value}"][data-location="${target.location}"]`,
    );
    if (!card) return;
    pendingCardFocusRef.current = null;
    card.focus({ preventScroll: true });
  }, [session.draft_best, session.draft_worst]);

  const scores = useMemo(() => {
    if (session.responses.length === 0) return null;
    return scoreResponses(session.responses);
  }, [session.responses]);

  const restart = () => {
    if (!window.confirm("Start over and clear this progress?")) return;
    profileSyncGenerationRef.current += 1;
    profileSyncInFlightRef.current = null;
    choicesCompletedAtRef.current = null;
    setPersonaPickerOpen(false);
    setLoadedScenario(null);
    setScenarioLoadError(null);
    restartSession();
  };

  const applyScenarioWeek = (
    loaded: LoadedScenario,
    weekIndex: number,
  ) => {
    const projection = projectScenarioWeek(loaded.fixture, weekIndex);
    const profile = projection.session.profile;
    const visibleEntryIds = new Set(
      projection.session.journal_entries.map(
        (entry) => entry.journal_entry_id,
      ),
    );
    const visibleEventIds = new Set(projection.session.trace_event_ids);
    const previousEntryId = session.experience.selected_entry_id;
    const previousEventId = session.experience.selected_event_id;
    const sameWeek = session.experience.selected_week === weekIndex;
    update({
      user_id: profile.user_id,
      preferred_name:
        profile.preferred_name ?? loaded.catalogItem.persona_name,
      session_id: profile.session_id,
      started_at: profile.started_at,
      stage: "complete",
      set_index: BWS_SETS.length - 1,
      set_order: BWS_SETS.map((_, index) => index),
      displayed_orders: BWS_SETS.map((set) => {
        const response = profile.bws_responses.find(
          (item) => item.set_number === set.setNumber,
        );
        return response ? [...response.item_order_shown] : [...set.items];
      }),
      responses: profile.bws_responses,
      draft_best: null,
      draft_worst: null,
      confirmed_profile: profile,
    });
    updateExperience({
      journal_started: true,
      journal_draft: "",
      revision: projection.session.revision,
      journal_entries: projection.session.journal_entries,
      nudges: projection.session.nudges,
      pending_submission: null,
      nudge_response_draft: "",
      error_message: null,
      selected_persona_id: loaded.catalogItem.persona_id,
      selected_week: weekIndex,
      selected_entry_id:
        sameWeek && previousEntryId && visibleEntryIds.has(previousEntryId)
          ? previousEntryId
          : null,
      selected_event_id:
        sameWeek && previousEventId && visibleEventIds.has(previousEventId)
          ? previousEventId
          : null,
      weekly_reviewer_decisions:
        projection.session.weekly_reviewer_decisions,
      drift_result: projection.session.drift_result,
      weekly_digest: projection.session.weekly_digest,
      assessment_clock: null,
      weekly_coach: null,
      run_state: "complete",
      retryable: false,
      trace_event_ids: projection.session.trace_event_ids,
      trace_events: projection.events,
    });
  };

  const activateScenario = (loaded: LoadedScenario) => {
    const hasManualProgress =
      selectedPersonaId === null &&
      (
        session.responses.length > 0 ||
        session.confirmed_profile !== null ||
        session.experience.journal_entries.length > 0 ||
        session.experience.journal_draft.trim().length > 0
      );
    if (
      hasManualProgress &&
      !window.confirm(
        "Load this saved Persona and replace your current progress?",
      )
    ) {
      return false;
    }
    profileSyncGenerationRef.current += 1;
    profileSyncInFlightRef.current = null;
    setLoadedScenario(loaded);
    setScenarioLoadError(null);
    setPersonaPickerOpen(false);
    applyScenarioWeek(loaded, 0);
    return true;
  };

  useEffect(() => {
    if (
      !selectedPersonaId ||
      loadedScenario?.catalogItem.persona_id === selectedPersonaId
    ) {
      return;
    }
    let cancelled = false;
    setScenarioLoadError(null);
    void loadSavedScenarioById(selectedPersonaId)
      .then((loaded) => {
        if (cancelled) return;
        const requestedWeek = session.experience.selected_week ?? 0;
        const safeWeek = Math.min(
          requestedWeek,
          loaded.fixture.scenario.weeks.length - 1,
        );
        setLoadedScenario(loaded);
        applyScenarioWeek(loaded, safeWeek);
      })
      .catch(() => {
        if (!cancelled) {
          setScenarioLoadError(
            "The saved Persona replay could not be restored.",
          );
        }
      });
    return () => {
      cancelled = true;
    };
  }, [
    loadedScenario?.catalogItem.persona_id,
    scenarioLoadAttempt,
    selectedPersonaId,
  ]);

  const locateTarget = (clientX: number, clientY: number): DropTarget => {
    const targets: [CardLocation, HTMLElement | null][] = [
      ["most", mostDropRef.current],
      ["least", leastDropRef.current],
      ["pool", selectionRef.current],
    ];
    for (const [target, element] of targets) {
      if (!element) continue;
      const bounds = element.getBoundingClientRect();
      if (
        clientX >= bounds.left &&
        clientX <= bounds.right &&
        clientY >= bounds.top &&
        clientY <= bounds.bottom
      ) {
        return target;
      }
    }
    return null;
  };

  const moveValue = (
    value: BwsObjectKey,
    from: CardLocation,
    to: CardLocation,
    focusAfterMove = false,
  ) => {
    if (from === to || isReviewing) return;
    const otherChoice = to === "most" ? session.draft_worst : session.draft_best;
    choicesCompletedAtRef.current = to !== "pool" && otherChoice !== null && otherChoice !== value
      ? Date.now()
      : null;
    if (focusAfterMove) {
      pendingCardFocusRef.current = { value, location: to };
    }
    setActiveDrop(null);
    const nextSession = to === "pool"
      ? from === "pool" ? session : clearChoice(session, from)
      : setChoice(session, to, value);
    updateSession(nextSession);
  };

  const submitSet = () => {
    if (!session.draft_best || !session.draft_worst) return;
    const choicesCompletedAt = choicesCompletedAtRef.current ?? Date.now();
    const response = {
      set_number: currentSet.setNumber,
      items: [...currentSet.items],
      item_order_shown: [...currentOrder],
      selected_best: session.draft_best,
      selected_worst: session.draft_worst,
      response_time_ms: Math.max(0, Math.round(choicesCompletedAt - session.stage_started_at_ms)),
    };
    const responses = [...session.responses.filter((item) => item.set_number !== currentSet.setNumber), response].sort(
      (left, right) => left.set_number - right.set_number,
    );
    const isLastSet = session.set_index === BWS_SETS.length - 1;
    choicesCompletedAtRef.current = null;
    update({
      responses,
      stage: isLastSet ? "summary" : "set",
      set_index: isLastSet ? session.set_index : session.set_index + 1,
      draft_best: null,
      draft_worst: null,
      stage_started_at_ms: Date.now(),
    });
  };

  useEffect(() => {
    if (!isReviewing) return;
    const timer = window.setTimeout(submitSet, AUTO_ADVANCE_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [isReviewing, session.set_index, session.draft_best, session.draft_worst]);

  const confirm = () => {
    const completedAt = new Date().toISOString();
    const profile = createProfile({
      userId: session.user_id,
      preferredName: session.preferred_name,
      sessionId: session.session_id,
      startedAt: session.started_at,
      completedAt,
      responses: session.responses,
      userConfirmed: true,
    });
    update({
      stage: "complete",
      confirmed_profile: profile,
    });
  };

  const savePreferredName = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const preferredName = normalizePreferredName(session.preferred_name);
    update({
      preferred_name: preferredName,
      stage: "set",
      stage_started_at_ms: Date.now(),
    });
  };

  const tapCard = (value: BwsObjectKey, location: CardLocation) => {
    if (isReviewing) return;
    if (location === "pool") {
      moveValue(value, "pool", nextChoice, true);
      return;
    }
    moveValue(value, location, "pool", true);
  };

  const startFirstJournal = () => {
    if (!session.confirmed_profile) return;
    if (!profileTraceReady) {
      profileSyncGenerationRef.current += 1;
      profileSyncInFlightRef.current = null;
    }
    onStartJournal?.(session.confirmed_profile);
    window.dispatchEvent(
      new CustomEvent("twinkl:start-first-journal", {
        detail: session.confirmed_profile,
      }),
    );
    updateExperience({
      journal_started: true,
      ...(!profileTraceReady
        ? {
            run_state: "idle" as const,
            retryable: false,
            error_message: null,
          }
        : {}),
    });
  };

  const cardPrompt = isReviewing
    ? "Choices set. Take a moment to review them—we’ll continue automatically."
    : !session.draft_best
      ? session.set_index === 0
        ? "Start with Most. Across 11 groups, tap what matters most, then least, as a guide for your life. Some cards will return."
        : "Start with Most. Tap the principle that matters most to you in this group."
      : "Now choose Least. Tap the principle that matters least to you in this group.";

  return (
    <div className={`app-shell app-shell--${
      activeView === "inspect"
        ? "inspect"
        : personaPickerOpen
          ? "persona"
          : journalStarted
            ? "journal"
            : session.stage
    }${selectedPersonaId ? " app-shell--saved-persona" : ""}`}>
      <header className="topbar">
        <a className="wordmark" href="#main">
          twinkl<span>·</span>
        </a>
        <nav className="view-switcher" aria-label="Demo view">
          <button
            className={activeView === "experience" ? "view-switcher__option view-switcher__option--active" : "view-switcher__option"}
            type="button"
            aria-pressed={activeView === "experience"}
            onClick={() => showView("experience")}
          >
            Experience
          </button>
          <button
            className={activeView === "inspect" ? "view-switcher__option view-switcher__option--active" : "view-switcher__option"}
            type="button"
            aria-pressed={activeView === "inspect"}
            aria-disabled={!inspectAvailable}
            aria-describedby={!inspectAvailable ? "inspect-availability" : undefined}
            title={!inspectAvailable ? "Available after all 11 questions" : undefined}
            onClick={() => showView("inspect")}
          >
            <span>Inspect</span>
            {!inspectAvailable ? <small>After questions</small> : null}
          </button>
          {!inspectAvailable ? (
            <span className="sr-only" id="inspect-availability">
              Available after all 11 questions
            </span>
          ) : null}
        </nav>
        <div className="topbar-actions">
          {!personaPickerOpen ? (
            <button
              className="restart"
              type="button"
              onClick={() => {
                setPersonaPickerOpen(true);
                showView("experience");
              }}
            >
              {selectedPersonaId ? "Change Persona" : "Try demo"}
            </button>
          ) : null}
          <button className="restart" type="button" onClick={restart}>
            Start over
          </button>
        </div>
      </header>

      {persistenceError ? (
        <p className="storage-warning" role="alert">
          Progress could not be saved in this browser. Keep this tab open while
          you continue.
        </p>
      ) : null}

      {activeView === "experience" ? (
        <main
          id="main"
          className={`layout${
            experienceSectionView ? " layout--section-rail" : ""
          }`}
        >
          <aside
            className={`instrument-panel${showProfileCompass ? " instrument-panel--with-profile-compass" : ""}`}
          >
            {experienceSectionView ? (
              <>
                {showProfileCompass ? (
                  <LivingCompass
                    currentQuestionIndex={null}
                    leastSelected={false}
                    milestone={milestone}
                    mostSelected={false}
                  />
                ) : null}
                <ExperienceSectionMap
                  hasJournalEntries={
                    session.experience.journal_entries.length > 0
                  }
                  hasWeeklyResult={
                    session.experience.drift_result !== null
                    && session.experience.weekly_digest !== null
                  }
                  view={experienceSectionView}
                />
              </>
            ) : (
              <>
                <LivingCompass
                  currentQuestionIndex={
                    session.stage === "set" ? session.set_index : null
                  }
                  leastSelected={session.draft_worst !== null}
                  milestone={milestone}
                  mostSelected={session.draft_best !== null}
                />
                <div className="instrument-copy">
                  <p className="eyebrow">Your inner compass</p>
                </div>
              </>
            )}
          </aside>

          <section className="flow-panel">
          {!personaPickerOpen && !selectedPersonaId && !journalStarted ? (
            <Progress session={session} />
          ) : null}

          {personaPickerOpen ? (
            <PersonaReplayPicker
              currentPersonaId={selectedPersonaId}
              onBack={() => setPersonaPickerOpen(false)}
              onLoad={activateScenario}
            />
          ) : null}

          {!personaPickerOpen && selectedPersonaId &&
          loadedScenario?.catalogItem.persona_id === selectedPersonaId &&
          session.confirmed_profile ? (
            <div className="stage stage--journal">
              <PersonaReplayExperience
                loaded={loadedScenario}
                weekIndex={session.experience.selected_week ?? 0}
                profile={session.confirmed_profile}
                experience={session.experience}
                updateExperience={updateExperience}
                inspectRun={inspectRun}
                onChoosePersona={() => setPersonaPickerOpen(true)}
                onWeekChange={(weekIndex) =>
                  applyScenarioWeek(loadedScenario, weekIndex)
                }
                headingRef={headingRef}
              />
            </div>
          ) : null}

          {!personaPickerOpen && selectedPersonaId &&
          loadedScenario?.catalogItem.persona_id !== selectedPersonaId ? (
            <div className="stage stage--journal replay-loading" aria-live="polite">
              <p className="eyebrow">Saved Persona replay</p>
              <h1 ref={headingRef} tabIndex={-1}>
                {scenarioLoadError
                  ? "The replay needs another try."
                  : "Restoring the replay…"}
              </h1>
              {scenarioLoadError ? (
                <>
                  <p className="lede">{scenarioLoadError}</p>
                  <button
                    className="button button--primary"
                    type="button"
                    onClick={() => setScenarioLoadAttempt((value) => value + 1)}
                  >
                    Try loading again
                  </button>
                </>
              ) : null}
            </div>
          ) : null}

          {!personaPickerOpen && !selectedPersonaId &&
          session.stage === "name" ? (
            <form className="stage stage--name" onSubmit={savePreferredName}>
              <p className="eyebrow">Before we begin</p>
              <h1 ref={headingRef} tabIndex={-1}>
                What should Twinkl call you?
              </h1>
              <p className="stage-note">
                We’ll use your name sparingly, when it makes a reflection feel
                more personal.
              </p>
              <label className="name-field">
                <span>Preferred name</span>
                <input
                  autoComplete="name"
                  maxLength={80}
                  name="preferred-name"
                  placeholder="Your name"
                  value={session.preferred_name}
                  onChange={(event) =>
                    update({ preferred_name: event.target.value })
                  }
                />
              </label>
              <div className="actions actions--end">
                <button
                  className="button button--primary"
                  type="submit"
                  disabled={!session.preferred_name.trim()}
                >
                  Continue
                </button>
              </div>
            </form>
          ) : null}

          {!personaPickerOpen && !selectedPersonaId && session.stage === "set" ? (
            <div className="stage stage--cards">
              <h1
                ref={headingRef}
                tabIndex={-1}
                aria-describedby="assessment-progress"
              >
                What matters most as you find your way?
              </h1>
              <p className="card-reassurance">
                There are no right answers here. More than one principle can matter.
              </p>
              <p className="card-prompt" aria-atomic="true" aria-live="polite">
                <span className="card-prompt__label" aria-hidden="true">
                  Next step
                </span>
                <span>{cardPrompt}</span>
              </p>
              <div className={`choice-board${isReviewing ? " choice-board--reviewing" : ""}`}>
                <section
                  ref={mostDropRef}
                  className={`drop-box drop-box--most${activeDrop === "most" ? " drop-box--active" : ""}${!session.draft_best ? " drop-box--guided" : ""}`}
                  data-testid="drop-most"
                  aria-label={`Most${!session.draft_best ? ", choose this first" : ", selected"}`}
                >
                  <div className="drop-box__label">
                    <strong>Most</strong>
                    <span>Matters most</span>
                  </div>
                  {session.draft_best ? (
                    <DraggableCard
                      value={session.draft_best}
                      location="most"
                      index={currentOrder.indexOf(session.draft_best)}
                      locateTarget={locateTarget}
                      locked={isReviewing}
                      nextChoice={null}
                      onDragTarget={setActiveDrop}
                      onMove={moveValue}
                      onTap={tapCard}
                    />
                  ) : (
                    <p>Tap a card first</p>
                  )}
                </section>
                <div
                  ref={selectionRef}
                  className={`selection-area${activeDrop === "pool" ? " selection-area--active" : ""}`}
                  data-testid="selection-area"
                >
                  <div className="selection-area__label">
                    <strong>{isReviewing ? "Choices set" : nextChoice === "most" ? "Choose Most" : "Choose Least"}</strong>
                    <span>{isReviewing ? "Next group in a moment" : "Tap one card"}</span>
                  </div>
                  <div className="card-deck">
                    {availableValues.map((value) => (
                      <DraggableCard
                        value={value}
                        location="pool"
                        index={currentOrder.indexOf(value)}
                        key={value}
                        locateTarget={locateTarget}
                        locked={isReviewing}
                        nextChoice={nextChoice}
                        onDragTarget={setActiveDrop}
                        onMove={moveValue}
                        onTap={tapCard}
                      />
                    ))}
                  </div>
                </div>
                <section
                  ref={leastDropRef}
                  className={`drop-box drop-box--least${activeDrop === "least" ? " drop-box--active" : ""}${session.draft_best && !session.draft_worst ? " drop-box--guided" : ""}`}
                  data-testid="drop-least"
                  aria-label={`Least${session.draft_worst ? ", selected" : session.draft_best ? ", choose this next" : ", choose this second"}`}
                >
                  <div className="drop-box__label">
                    <strong>Least</strong>
                    <span>Matters least</span>
                  </div>
                  {session.draft_worst ? (
                    <DraggableCard
                      value={session.draft_worst}
                      location="least"
                      index={currentOrder.indexOf(session.draft_worst)}
                      locateTarget={locateTarget}
                      locked={isReviewing}
                      nextChoice={null}
                      onDragTarget={setActiveDrop}
                      onMove={moveValue}
                      onTap={tapCard}
                    />
                  ) : (
                    <p>{session.draft_best ? "Tap a card next" : "Then choose here"}</p>
                  )}
                </section>
              </div>
            </div>
          ) : null}

          {!personaPickerOpen && !selectedPersonaId &&
          session.stage === "summary" && scores ? (
            <div className="stage stage--summary" id="experience-profile">
              <h1 ref={headingRef} tabIndex={-1}>
                What sits at the center.
              </h1>
              <div className="core-values">
                {scores.profile.top_values.map((value) => (
                  <article key={value}>
                    <span aria-hidden="true">✦</span>
                    <p>{VALUES[value].phrase}</p>
                  </article>
                ))}
              </div>
              <p className="summary-explainer">
                This result reflects the Most and Least choices you made most
                consistently across all 11 groups.
              </p>
              <div
                className="actions actions--end"
                id="experience-confirm"
              >
                <button className="button button--primary" type="button" onClick={confirm}>
                  Confirm my compass
                </button>
              </div>
            </div>
          ) : null}

          {!personaPickerOpen && !selectedPersonaId &&
          session.stage === "complete" && session.confirmed_profile &&
          !journalStarted ? (
            <div className="stage stage--complete" id="experience-ready">
              <h1 ref={headingRef} tabIndex={-1}>
                Your compass is ready, {session.preferred_name}.
              </h1>
              <p className="lede">Start with one moment from the past week. Twinkl will build from what you notice.</p>
              <CoreValueReminder profile={session.confirmed_profile} />
              <div
                className="journal-handoff"
                id="experience-journal-handoff"
              >
                <small>First Journal Entry</small>
                <p>When did you feel most like yourself?</p>
              </div>
              <div className="actions actions--end">
                <button
                  className="button button--primary"
                  type="button"
                  onClick={startFirstJournal}
                >
                  Start my first Journal Entry
                </button>
              </div>
            </div>
          ) : null}

          {!personaPickerOpen && !selectedPersonaId &&
          session.stage === "complete" && session.confirmed_profile &&
          journalStarted ? (
            <div className="stage stage--journal">
              <JournalExperience
                profile={session.confirmed_profile}
                experience={session.experience}
                updateExperience={updateExperience}
                inspectRun={inspectRun}
                headingRef={headingRef}
              />
            </div>
          ) : null}
          </section>
        </main>
      ) : (
        <main
          id="main"
          className={`layout layout--inspect layout--section-rail${
            !selectedPersonaId && scores ? " layout--assessment-inspect" : ""
          }`}
        >
          <aside className="instrument-panel instrument-panel--inspect">
            {!selectedPersonaId && scores ? (
              <AssessmentSectionMap />
            ) : selectedPersonaId ? (
              <ExperienceSectionMap view="inspect" />
            ) : (
              <>
                <div className="inspect-lens" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </div>
                <div className="instrument-copy">
                  <p className="eyebrow">Same saved replay</p>
                  <h2>How Twinkl reached this moment.</h2>
                  <p>
                    Inspect follows the exact week and Persona selected in
                    Experience.
                  </p>
                </div>
              </>
            )}
          </aside>
          <section className="flow-panel flow-panel--inspect">
            <InspectView
              events={session.experience.trace_events}
              currentWeekEventIds={
                loadedScenario && selectedPersonaId
                  ? loadedScenario.fixture.scenario.weeks[
                      Math.min(
                        session.experience.selected_week ?? 0,
                        loadedScenario.fixture.scenario.weeks.length - 1,
                      )
                    ]?.event_ids
                  : undefined
              }
              currentJournalEntryIds={
                session.experience.trace_events.length > 0
                  ? session.experience.journal_entries.map(
                      (entry) => entry.journal_entry_id,
                    )
                  : undefined
              }
              onboarding={!selectedPersonaId && scores ? {
                confirmed: session.confirmed_profile !== null,
                responses: session.responses,
                scores,
                setOrder: session.set_order,
              } : undefined}
              selectedEventId={session.experience.selected_event_id}
              traceLabel={
                loadedScenario
                  ? `${loadedScenario.catalogItem.persona_name} · saved replay`
                  : "Current Experience session"
              }
              emptyMessage={
                profileAwaitingConfirmation
                  ? "The browser calculation is complete. Confirm the result in Experience to send the Profile to Python validation."
                  : profileTracePending
                  ? "Profile validation is in progress. No later work has happened."
                  : profileTraceFailed
                    ? session.experience.error_message
                      ?? "The Profile trace could not be loaded."
                    : "No backend work has been recorded for this Experience yet."
              }
              emptyActionLabel={
                profileTraceFailed && session.experience.retryable
                  ? "Retry Profile validation"
                  : undefined
              }
              onEmptyAction={
                profileTraceFailed && session.experience.retryable
                  ? () => {
                    if (session.confirmed_profile) {
                      void synchronizeProfileTrace(session.confirmed_profile);
                    }
                  }
                  : undefined
              }
              onReturn={() => showView("experience")}
            />
          </section>
        </main>
      )}
    </div>
  );
}

export default function App(props: AppProps = {}) {
  return (
    <AppErrorBoundary>
      <SharedSessionProvider>
        <ExperienceInspectApp {...props} />
      </SharedSessionProvider>
    </AppErrorBoundary>
  );
}
