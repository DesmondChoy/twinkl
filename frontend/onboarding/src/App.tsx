import { useEffect, useMemo, useRef, useState } from "react";
import {
  BWS_SETS,
  GOALS,
  VALUE_ORDER,
  VALUES,
  createProfile,
  scoreResponses,
  type GoalCategory,
  type ValueKey,
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

const MILESTONE_COUNT = 8;
const VALUE_TONES: Record<ValueKey, string> = {
  self_direction: "#5576d9",
  stimulation: "#e26844",
  hedonism: "#d79a24",
  achievement: "#398a70",
  power: "#8759a8",
  security: "#397d9b",
  conformity: "#7870bd",
  tradition: "#9b684f",
  benevolence: "#c75d78",
  universalism: "#238879",
};

function milestoneFor(session: OnboardingSession): number {
  if (session.stage === "set") {
    return Math.min(session.set_index + 1, 6);
  }
  if (session.stage === "mirror") return 3;
  if (session.stage === "goal") return 7;
  return 8;
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
    ? `Values · ${session.set_index + 1} of 6`
    : session.stage === "mirror"
      ? "Values · 3 of 6"
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

function PhraseList({ values }: { values: ValueKey[] }) {
  return (
    <ul className="phrase-list">
      {values.map((value) => (
        <li key={value}>{VALUES[value].phrase}</li>
      ))}
    </ul>
  );
}

function CardArtwork({ value }: { value: ValueKey }) {
  return (
    <svg
      className={`value-card__art value-card__art--${value}`}
      viewBox="0 0 180 132"
      aria-hidden="true"
    >
      {value === "self_direction" ? (
        <>
          <path d="M90 116V75M90 75 49 34M90 75l41-41M49 34v24M49 34h24M131 34v24M131 34h-24" />
          <circle className="art-fill" cx="90" cy="112" r="7" />
        </>
      ) : null}
      {value === "stimulation" ? (
        <>
          <path className="art-fill" d="m100 10-42 61h31l-9 51 43-67H91z" />
          <path d="M39 28 25 14M143 34l16-17M35 100l-20 10M145 99l22 10" />
        </>
      ) : null}
      {value === "hedonism" ? (
        <>
          <circle className="art-fill" cx="90" cy="47" r="25" />
          <path d="M90 7v13M90 74v13M50 47H37M143 47h-13M61 18l9 9M119 76l9 9M119 18l-9 9M61 76l-9 9" />
          <path d="M27 108c20-21 39 20 62-1 23-20 43 19 66-2" />
        </>
      ) : null}
      {value === "achievement" ? (
        <>
          <path d="m20 112 49-77 25 39 18-27 48 65Z" />
          <path className="art-fill" d="M69 35v-21h37L94 25l12 10Z" />
          <circle cx="69" cy="35" r="6" />
        </>
      ) : null}
      {value === "power" ? (
        <>
          <circle cx="90" cy="66" r="19" />
          <circle cx="90" cy="66" r="43" />
          <path d="M90 47V14M109 66h37M90 85v33M71 66H34" />
          <circle className="art-fill" cx="90" cy="14" r="7" />
          <circle className="art-fill" cx="146" cy="66" r="7" />
          <circle className="art-fill" cx="90" cy="118" r="7" />
          <circle className="art-fill" cx="34" cy="66" r="7" />
        </>
      ) : null}
      {value === "security" ? (
        <>
          <path className="art-fill" d="M90 10c20 15 39 16 55 18v33c0 35-22 53-55 65-33-12-55-30-55-65V28c16-2 35-3 55-18Z" />
          <path d="M61 72c8-19 20-28 29-28s21 9 29 28c-12 13-22 20-29 23-7-3-17-10-29-23Z" />
        </>
      ) : null}
      {value === "conformity" ? (
        <>
          <path d="M38 43h104M38 89h104" />
          <circle className="art-fill" cx="48" cy="43" r="15" />
          <circle className="art-fill" cx="90" cy="43" r="15" />
          <circle className="art-fill" cx="132" cy="43" r="15" />
          <circle cx="48" cy="89" r="15" />
          <circle cx="90" cy="89" r="15" />
          <circle cx="132" cy="89" r="15" />
        </>
      ) : null}
      {value === "tradition" ? (
        <>
          <path d="M28 114h124M39 105V46M67 105V46M113 105V46M141 105V46M27 46h126L90 11Z" />
          <path className="art-fill" d="M19 114h142v10H19z" />
          <circle cx="90" cy="33" r="7" />
        </>
      ) : null}
      {value === "benevolence" ? (
        <>
          <path className="art-fill" d="M90 112S37 82 37 45c0-20 25-30 39-12l14 18 14-18c14-18 39-8 39 12 0 37-53 67-53 67Z" />
          <path d="M18 80c22-3 36 7 45 25M162 80c-22-3-36 7-45 25" />
        </>
      ) : null}
      {value === "universalism" ? (
        <>
          <circle cx="88" cy="67" r="45" />
          <path d="M43 67h90M88 22c18 14 26 29 26 45s-8 31-26 45M88 22C70 36 62 51 62 67s8 31 26 45" />
          <path d="M27 37c32-23 81-27 126 4M25 97c35 19 88 19 128-5" />
          <path className="art-fill" d="m146 15 4 10 11 4-11 4-4 11-4-11-11-4 11-4Z" />
        </>
      ) : null}
    </svg>
  );
}

type CardLocation = "pool" | "most" | "least";
type DropTarget = CardLocation | null;

interface DraggableCardProps {
  value: ValueKey;
  location: CardLocation;
  index: number;
  locateTarget: (clientX: number, clientY: number) => DropTarget;
  picked: boolean;
  onDragTarget: (target: DropTarget) => void;
  onMove: (value: ValueKey, from: CardLocation, to: CardLocation) => void;
  onTap: (value: ValueKey, location: CardLocation) => void;
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
  const phrase = VALUES[value].phrase;

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
      onMove(value, location, target);
      return;
    }
    if (location !== "pool" && ["backspace", "delete", "arrowdown"].includes(key)) {
      event.preventDefault();
      onMove(value, location, "pool");
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
          "--card-accent": VALUE_TONES[value],
          "--card-angle": `${(index - 1.5) * 1.2}deg`,
          "--card-delay": `${index * 70}ms`,
          "--drag-x": `${offset.x}px`,
          "--drag-y": `${offset.y}px`,
        } as React.CSSProperties
      }
    >
      <CardArtwork value={value} />
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
  const [pickedValue, setPickedValue] = useState<ValueKey | null>(null);
  const [journalStarted, setJournalStarted] = useState(false);
  const [journalDraft, setJournalDraft] = useState("");
  const headingRef = useRef<HTMLHeadingElement>(null);
  const mostDropRef = useRef<HTMLElement>(null);
  const leastDropRef = useRef<HTMLElement>(null);
  const selectionRef = useRef<HTMLDivElement>(null);
  const milestone = milestoneFor(session);
  const currentOrder = session.displayed_orders[session.set_index];
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
    value: ValueKey,
    from: CardLocation,
    to: CardLocation,
  ) => {
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
    const currentSet = BWS_SETS[session.set_index];
    const response = {
      set_number: currentSet.setNumber,
      items: [...currentSet.items],
      item_order_shown: [...session.displayed_orders[session.set_index]],
      selected_best: session.draft_best,
      selected_worst: session.draft_worst,
      response_time_ms: Math.max(0, Math.round(Date.now() - session.stage_started_at_ms)),
    };
    const responses = [...session.responses.filter((item) => item.set_number !== currentSet.setNumber), response].sort(
      (left, right) => left.set_number - right.set_number,
    );
    const nextStage = session.set_index === 2 ? "mirror" : session.set_index === 5 ? "goal" : "set";
    update({
      responses,
      stage: nextStage,
      set_index: session.set_index === 5 ? session.set_index : session.set_index + 1,
      draft_best: null,
      draft_worst: null,
      stage_started_at_ms: Date.now(),
    });
  };

  const continueFromMirror = () => {
    setPickedValue(null);
    update({
      stage: "set",
      set_index: 3,
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

  const tapCard = (value: ValueKey, location: CardLocation) => {
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
    ? "Now tap Most or Least."
    : !session.draft_best && !session.draft_worst
      ? session.set_index === 0
        ? "Move or tap one card into Most and another into Least. Your choices help Twinkl understand what matters to you."
        : session.set_index === 2
          ? "Some cards return. Choose what feels true in this group, then place one in Most and one in Least."
          : "Move or tap one card into Most and another into Least."
    : !session.draft_best
      ? "Choose another card, then place it in Most."
      : !session.draft_worst
        ? "Choose another card, then place it in Least."
        : "Both choices are set. Tap a placed card to reconsider.";

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
                What feels true for you right now?
              </h1>
              <p className="card-prompt" aria-live="polite">
                {cardPrompt}
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
                    <span>Feels like me</span>
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
                      aria-label={`Place ${VALUES[pickedValue].phrase} in Most`}
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
                    <strong>{pickedValue ? "Card selected" : "Choose from these"}</strong>
                    <span>{pickedValue ? "Tap Most or Least" : "Move one to each box"}</span>
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
                    <span>Feels like me</span>
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
                      aria-label={`Place ${VALUES[pickedValue].phrase} in Least`}
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

          {session.stage === "mirror" && scores ? (
            <div className="stage stage--mirror">
              <h1 ref={headingRef} tabIndex={-1}>
                A pattern is beginning to appear.
              </h1>
              <div className="mirror-grid">
                <div className="mirror-card mirror-card--high">
                  <small>Pulling you forward</small>
                  <PhraseList values={scores.top_values} />
                </div>
                <div className="mirror-card mirror-card--low">
                  <small>Quieter for now</small>
                  <PhraseList values={scores.bottom_values} />
                </div>
              </div>
              <div className="actions actions--end">
                <button className="button button--primary" type="button" onClick={continueFromMirror}>
                  Keep going
                </button>
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
                {scores.top_values.map((value) => (
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
