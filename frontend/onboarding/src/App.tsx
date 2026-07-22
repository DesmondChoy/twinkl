import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  BWS_OBJECTS,
  BWS_SETS,
  GOALS,
  VALUE_ORDER,
  VALUES,
  createProfile,
  scoreResponses,
  type BwsObjectKey,
  type GoalCategory,
} from "./domain";
import {
  clearChoice,
  setChoice,
  type OnboardingSession,
} from "./session";
import { SharedSessionProvider, useSharedSession } from "./sharedSession";

const MILESTONE_COUNT = BWS_SETS.length + 2;
const AUTO_ADVANCE_DELAY_MS = 1_000;
const CARD_BACKGROUNDS = [
  "/card-backgrounds/memory-atlas-01.jpg",
  "/card-backgrounds/memory-atlas-02.jpg",
  "/card-backgrounds/memory-atlas-03.jpg",
  "/card-backgrounds/memory-atlas-04.jpg",
  "/card-backgrounds/memory-atlas-05.jpg",
  "/card-backgrounds/memory-atlas-06.jpg",
] as const;

function milestoneFor(session: OnboardingSession): number {
  if (session.stage === "set") {
    return Math.min(session.set_index + 1, BWS_SETS.length);
  }
  if (session.stage === "goal") return BWS_SETS.length + 1;
  return BWS_SETS.length + 2;
}

function Compass({ milestone }: { milestone: number }) {
  const progress = milestone / MILESTONE_COUNT;
  return (
    <div
      className="compass"
      style={{ "--compass-progress": `${progress * 360}deg` } as React.CSSProperties}
      aria-hidden="true"
    >
      <div className="compass__orbit">
        {VALUE_ORDER.map((value, index) => (
          <span
            className="compass__star"
            key={value}
            style={{ "--star-index": index } as React.CSSProperties}
          />
        ))}
      </div>
      <div className="compass__needle" />
      <div className="compass__center">
        <span>✦</span>
        <small>compass</small>
      </div>
    </div>
  );
}

function Progress({ session, milestone }: { session: OnboardingSession; milestone: number }) {
  if (milestone === 0) return null;
  const label = session.stage === "set"
    ? `Values · ${session.set_index + 1} of ${BWS_SETS.length}`
    : session.stage === "goal"
      ? "Your focus"
      : "Your compass";
  return (
    <div className="progress" aria-label={label}>
      <div className="progress__label">
        <span>{label}</span>
      </div>
      <div className="progress__track">
        <span style={{ width: `${(milestone / MILESTONE_COUNT) * 100}%` }} />
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
    updateSession,
    updateExperience,
    showView,
    restart: restartSession,
  } = useSharedSession();
  const [activeDrop, setActiveDrop] = useState<DropTarget>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const mostDropRef = useRef<HTMLElement>(null);
  const leastDropRef = useRef<HTMLElement>(null);
  const selectionRef = useRef<HTMLDivElement>(null);
  const pendingCardFocusRef = useRef<{ value: BwsObjectKey; location: CardLocation } | null>(null);
  const choicesCompletedAtRef = useRef<number | null>(null);
  const milestone = milestoneFor(session);
  const journalStarted = session.experience.journal_started;
  const journalDraft = session.experience.journal_draft;
  const activeView = session.experience.active_view;
  const inspectAvailable = session.confirmed_profile !== null;
  const currentSetIndex = session.set_order[session.set_index];
  const currentSet = BWS_SETS[currentSetIndex];
  const currentOrder = session.displayed_orders[currentSetIndex];
  const availableValues = currentOrder.filter(
    (value) => value !== session.draft_best && value !== session.draft_worst,
  );
  const isReviewing = session.stage === "set" && Boolean(session.draft_best && session.draft_worst);
  const nextChoice: "most" | "least" = session.draft_best ? "least" : "most";

  const update = (patch: Partial<OnboardingSession>) => {
    updateSession(patch);
  };

  useEffect(() => {
    headingRef.current?.focus({ preventScroll: true });
  }, [session.stage, session.set_index, journalStarted, activeView]);

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
    if (!window.confirm("Start over and clear these onboarding choices?")) return;
    choicesCompletedAtRef.current = null;
    restartSession();
  };

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
      stage: isLastSet ? "goal" : "set",
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
    if (!session.goal_category) return;
    const completedAt = new Date().toISOString();
    const profile = createProfile({
      userId: session.user_id,
      sessionId: session.session_id,
      startedAt: session.started_at,
      completedAt,
      responses: session.responses,
      goalCategory: session.goal_category,
      userConfirmed: true,
    });
    update({
      stage: "complete",
      confirmed_profile: profile,
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
    onStartJournal?.(session.confirmed_profile);
    window.dispatchEvent(
      new CustomEvent("twinkl:start-first-journal", {
        detail: session.confirmed_profile,
      }),
    );
    updateExperience({ journal_started: true });
  };

  const cardPrompt = isReviewing
    ? "Choices set. Take a moment to review them—we’ll continue automatically."
    : !session.draft_best
      ? session.set_index === 0
        ? "Start with Most. Across 11 groups, tap what matters most, then least, as a guide for your life. Some cards will return."
        : "Start with Most. Tap the principle that matters most to you in this group."
      : "Now choose Least. Tap the principle that matters least to you in this group.";

  return (
    <div className={`app-shell app-shell--${activeView === "inspect" ? "inspect" : journalStarted ? "journal" : session.stage}`}>
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
            title={!inspectAvailable ? "Available after Profile confirmation" : undefined}
            onClick={() => showView("inspect")}
          >
            <span>Inspect</span>
            {!inspectAvailable ? <small>After Profile</small> : null}
          </button>
          {!inspectAvailable ? (
            <span className="sr-only" id="inspect-availability">
              Available after Profile confirmation
            </span>
          ) : null}
        </nav>
        <button className="restart" type="button" onClick={restart}>
          Start over
        </button>
      </header>

      {activeView === "experience" ? <main id="main" className="layout">
        <aside className="instrument-panel">
          <Compass milestone={milestone} />
          <div className="instrument-copy">
            <p className="eyebrow">Your inner compass</p>
          </div>
        </aside>

        <section className="flow-panel">
          {!journalStarted ? <Progress session={session} milestone={milestone} /> : null}

          {session.stage === "set" ? (
            <div className="stage stage--cards">
              <h1 ref={headingRef} tabIndex={-1}>
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

          {session.stage === "goal" ? (
            <div className="stage">
              <h1 ref={headingRef} tabIndex={-1}>
                What brought you here right now?
              </h1>
              <p className="stage-note">Choose the one closest to what brought you here.</p>
              <div className="goal-list">
                {(Object.entries(GOALS) as [GoalCategory, string][]).map(([key, text]) => (
                  <label className={`goal-card${session.goal_category === key ? " goal-card--selected" : ""}`} key={key}>
                    <input
                      type="radio"
                      name="goal"
                      value={key}
                      checked={session.goal_category === key}
                      onChange={() => update({ goal_category: key })}
                    />
                    <span>{text}</span>
                  </label>
                ))}
              </div>
              <div className="actions actions--end">
                <button
                  className="button button--primary"
                  type="button"
                  disabled={!session.goal_category}
                  onClick={() => update({ stage: "summary" })}
                >
                  See my compass
                </button>
              </div>
            </div>
          ) : null}

          {session.stage === "summary" && scores && session.goal_category ? (
            <div className="stage stage--summary">
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
              <div className="focus-line">
                <small>What brought you here</small>
                <p>{GOALS[session.goal_category]}</p>
              </div>
              <div className="actions actions--end">
                <button className="button button--primary" type="button" onClick={confirm}>
                  Set my compass
                </button>
              </div>
            </div>
          ) : null}

          {session.stage === "complete" && session.confirmed_profile && !journalStarted ? (
            <div className="stage stage--complete">
              <h1 ref={headingRef} tabIndex={-1}>
                Your compass is ready.
              </h1>
              <p className="lede">Start with one moment from the past week. Twinkl will build from what you notice.</p>
              <div className="journal-handoff">
                <small>First Journal Entry</small>
                <p>When did you feel most like yourself?</p>
              </div>
              <div className="actions actions--end">
                <button className="button button--primary" type="button" onClick={startFirstJournal}>
                  Start my first Journal Entry
                </button>
              </div>
            </div>
          ) : null}

          {session.stage === "complete" && session.confirmed_profile && journalStarted ? (
            <div className="stage stage--journal">
              <p className="eyebrow">First Journal Entry</p>
              <h1 ref={headingRef} tabIndex={-1}>
                When did you feel most like yourself?
              </h1>
              <p className="lede" id="first-journal-help">
                Think of one moment from the past week. What was happening, and what felt true about it?
              </p>
              <textarea
                aria-label="First Journal Entry"
                aria-describedby="first-journal-help"
                placeholder="Start with the moment…"
                value={journalDraft}
                onChange={(event) => updateExperience({ journal_draft: event.target.value })}
              />
            </div>
          ) : null}
        </section>
      </main> : (
        <main id="main" className="layout layout--inspect">
          <aside className="instrument-panel instrument-panel--inspect">
            <div className="inspect-lens" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <div className="instrument-copy">
              <p className="eyebrow">Same session</p>
              <h2>Behind this moment.</h2>
              <p>Inspect follows the exact work connected to what you see in Experience.</p>
            </div>
          </aside>
          <section className="flow-panel flow-panel--inspect">
            <div className="stage stage--inspect">
              <p className="eyebrow">Inspect</p>
              <h1 ref={headingRef} tabIndex={-1}>The trail starts here.</h1>
              {session.experience.selected_event_id ? (
                <div className="inspect-selection" data-testid="inspect-selection">
                  <small>Selected event</small>
                  <code>{session.experience.selected_event_id}</code>
                  <p>Its event details will appear here when the Inspect timeline is connected.</p>
                </div>
              ) : (
                <p className="lede">
                  Profile validation is ready to inspect. Event details will appear as Journal Entries and saved persona runs are connected.
                </p>
              )}
              <div className="actions">
                <button className="button button--primary" type="button" onClick={() => showView("experience")}>
                  Return to Experience
                </button>
              </div>
            </div>
          </section>
        </main>
      )}
    </div>
  );
}

export default function App(props: AppProps = {}) {
  return (
    <SharedSessionProvider>
      <ExperienceInspectApp {...props} />
    </SharedSessionProvider>
  );
}
