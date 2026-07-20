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
  clearSession,
  createSession,
  loadOrCreateSession,
  persistSession,
  setChoice,
  type OnboardingSession,
} from "./session";

const MILESTONE_COUNT = BWS_SETS.length + 2;
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
  picked: boolean;
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
  picked,
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
    location === "pool"
      ? "Tap to choose this card, or press M for Most and L for Least."
      : `Selected as ${location}. Tap, press Backspace, or press Arrow Down to return it.`;

  return (
    <article
      className={`value-card${location === "pool" ? "" : ` value-card--${location} value-card--placed`}${picked ? " value-card--picked" : ""}${dragging ? " value-card--dragging" : ""}`}
      data-testid="value-card"
      data-value={value}
      data-location={location}
      data-background-position={index}
      role="button"
      tabIndex={0}
      aria-pressed={location === "pool" ? picked : undefined}
      aria-label={`${phrase}. ${keyboardHint}`}
      onKeyDown={keyboardMove}
      onClick={(event) => {
        event.stopPropagation();
        if (suppressClickRef.current) {
          suppressClickRef.current = false;
          return;
        }
        onTap(value, location);
      }}
      onPointerDown={(event) => {
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

export default function App({ onStartJournal }: AppProps = {}) {
  const [session, setSession] = useState<OnboardingSession>(() => loadOrCreateSession());
  const [activeDrop, setActiveDrop] = useState<DropTarget>(null);
  const [pickedValue, setPickedValue] = useState<BwsObjectKey | null>(null);
  const [journalStarted, setJournalStarted] = useState(false);
  const [journalDraft, setJournalDraft] = useState("");
  const headingRef = useRef<HTMLHeadingElement>(null);
  const mostDropRef = useRef<HTMLElement>(null);
  const leastDropRef = useRef<HTMLElement>(null);
  const selectionRef = useRef<HTMLDivElement>(null);
  const pendingCardFocusRef = useRef<{ value: BwsObjectKey; location: CardLocation } | null>(null);
  const milestone = milestoneFor(session);
  const currentSetIndex = session.set_order[session.set_index];
  const currentSet = BWS_SETS[currentSetIndex];
  const currentOrder = session.displayed_orders[currentSetIndex];
  const availableValues = currentOrder.filter(
    (value) => value !== session.draft_best && value !== session.draft_worst,
  );

  const update = (patch: Partial<OnboardingSession>) => {
    setSession((current) => ({ ...current, ...patch }));
  };

  useEffect(() => {
    persistSession(session);
  }, [session]);

  useEffect(() => {
    headingRef.current?.focus({ preventScroll: true });
  }, [session.stage, session.set_index, journalStarted]);

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
    clearSession();
    setJournalStarted(false);
    setJournalDraft("");
    setSession(createSession());
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
    if (from === to) return;
    if (focusAfterMove) {
      pendingCardFocusRef.current = { value, location: to };
    }
    setPickedValue(null);
    setSession((current) => {
      if (to === "pool") {
        return from === "pool" ? current : clearChoice(current, from);
      }
      return setChoice(current, to, value);
    });
  };

  const submitSet = () => {
    if (!session.draft_best || !session.draft_worst) return;
    setPickedValue(null);
    const response = {
      set_number: currentSet.setNumber,
      items: [...currentSet.items],
      item_order_shown: [...currentOrder],
      selected_best: session.draft_best,
      selected_worst: session.draft_worst,
      response_time_ms: Math.max(0, Math.round(Date.now() - session.stage_started_at_ms)),
    };
    const responses = [...session.responses.filter((item) => item.set_number !== currentSet.setNumber), response].sort(
      (left, right) => left.set_number - right.set_number,
    );
    const isLastSet = session.set_index === BWS_SETS.length - 1;
    update({
      responses,
      stage: isLastSet ? "goal" : "set",
      set_index: isLastSet ? session.set_index : session.set_index + 1,
      draft_best: null,
      draft_worst: null,
      stage_started_at_ms: Date.now(),
    });
  };

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
    if (location === "pool") {
      setPickedValue((current) => current === value ? null : value);
      return;
    }
    moveValue(value, location, "pool");
  };

  const placePicked = (target: "most" | "least") => {
    if (!pickedValue) return;
    moveValue(pickedValue, "pool", target);
  };

  const startFirstJournal = () => {
    if (!session.confirmed_profile) return;
    onStartJournal?.(session.confirmed_profile);
    window.dispatchEvent(
      new CustomEvent("twinkl:start-first-journal", {
        detail: session.confirmed_profile,
      }),
    );
    setJournalStarted(true);
  };

  const cardPrompt = pickedValue
    ? "Where does this principle sit for you in this group? Choose Most or Least."
    : !session.draft_best && !session.draft_worst
      ? session.set_index === 0
        ? "Across 11 groups, choose what matters most and least as a guide for your life. Some cards will return."
        : "Take your time. Choose what matters most and least as a guide for your life in this group."
      : !session.draft_best
        ? "Choose the principle that matters most to you in this group."
        : !session.draft_worst
          ? "Now choose the principle that matters least to you in this group."
          : "Both choices are set. You can move either card if you want to reconsider.";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="wordmark" href="#main">
          twinkl<span>·</span>
        </a>
        <button className="restart" type="button" onClick={restart}>
          Start over
        </button>
      </header>

      <main id="main" className="layout">
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
              <div className="choice-board">
                <section
                  ref={mostDropRef}
                  className={`drop-box drop-box--most${activeDrop === "most" ? " drop-box--active" : ""}${pickedValue ? " drop-box--ready" : ""}`}
                  data-testid="drop-most"
                  aria-label="Most"
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
                      picked={false}
                      onDragTarget={setActiveDrop}
                      onMove={moveValue}
                      onTap={tapCard}
                    />
                  ) : (
                    <p>Drop a card here</p>
                  )}
                  {pickedValue ? (
                    <button
                      className="drop-box__tap-target"
                      type="button"
                      aria-label={`Place ${BWS_OBJECTS[pickedValue].descriptor} in Most`}
                      onClick={() => placePicked("most")}
                    >
                      Tap to place
                    </button>
                  ) : null}
                </section>
                <div
                  ref={selectionRef}
                  className={`selection-area${activeDrop === "pool" ? " selection-area--active" : ""}`}
                  data-testid="selection-area"
                >
                  <div className="selection-area__label">
                    <strong>{pickedValue ? "Principle selected" : "Take a look"}</strong>
                    <span>{pickedValue ? "Choose Most or Least" : "Choose one for each place"}</span>
                    <span className="selection-area__touch-hint">
                      {pickedValue ? "Choose Most or Least below" : "Tap to choose · Scroll to explore"}
                    </span>
                  </div>
                  <div className="card-deck">
                    {availableValues.map((value) => (
                      <DraggableCard
                        value={value}
                        location="pool"
                        index={currentOrder.indexOf(value)}
                        key={value}
                        locateTarget={locateTarget}
                        picked={pickedValue === value}
                        onDragTarget={setActiveDrop}
                        onMove={moveValue}
                        onTap={tapCard}
                      />
                    ))}
                  </div>
                </div>
                <section
                  ref={leastDropRef}
                  className={`drop-box drop-box--least${activeDrop === "least" ? " drop-box--active" : ""}${pickedValue ? " drop-box--ready" : ""}`}
                  data-testid="drop-least"
                  aria-label="Least"
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
                      picked={false}
                      onDragTarget={setActiveDrop}
                      onMove={moveValue}
                      onTap={tapCard}
                    />
                  ) : (
                    <p>Drop a card here</p>
                  )}
                  {pickedValue ? (
                    <button
                      className="drop-box__tap-target"
                      type="button"
                      aria-label={`Place ${BWS_OBJECTS[pickedValue].descriptor} in Least`}
                      onClick={() => placePicked("least")}
                    >
                      Tap to place
                    </button>
                  ) : null}
                </section>
              </div>
              <div className="actions actions--end">
                <button
                  className="button button--primary"
                  type="button"
                  disabled={!session.draft_best || !session.draft_worst}
                  onClick={submitSet}
                >
                  Continue
                </button>
              </div>
            </div>
          ) : null}

          {session.stage === "set" && pickedValue ? (
            <div className="touch-choice-bar" role="group" aria-label="Place selected principle">
              <span>Place selected card</span>
              <button
                className="touch-choice-bar__most"
                type="button"
                aria-label="Choose Most for selected principle"
                onClick={() => placePicked("most")}
              >
                Most
              </button>
              <button
                className="touch-choice-bar__least"
                type="button"
                aria-label="Choose Least for selected principle"
                onClick={() => placePicked("least")}
              >
                Least
              </button>
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
                onChange={(event) => setJournalDraft(event.target.value)}
              />
            </div>
          ) : null}
        </section>
      </main>
    </div>
  );
}
