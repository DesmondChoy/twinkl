import { useId, useRef } from "react";
import { createPortal } from "react-dom";
import type { JournalEntryContract } from "./demoContracts";
import useModalFocus from "./useModalFocus";

interface JournalEntryDialogProps {
  entry: JournalEntryContract | null;
  responseVisible?: boolean;
  onClose: () => void;
}

export default function JournalEntryDialog({
  entry,
  responseVisible = true,
  onClose,
}: JournalEntryDialogProps) {
  const id = useId();
  const closeRef = useRef<HTMLButtonElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  useModalFocus(entry !== null, overlayRef, closeRef, onClose);

  if (!entry) return null;
  const date = new Intl.DateTimeFormat(undefined, {
    day: "numeric", month: "short",
  }).format(new Date(`${entry.date}T00:00:00`));

  return createPortal(
    <div
      ref={overlayRef}
      className="replay-entry-drawer"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="replay-entry-drawer__panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${id}-title`}
      >
        <header>
          <div>
            <p className="eyebrow">Journal Entry</p>
            <h2 id={`${id}-title`}>{date}</h2>
          </div>
          <button
            className="replay-entry-drawer__close"
            ref={closeRef}
            type="button"
            aria-label="Close Journal Entry"
            onClick={onClose}
          >
            Close
          </button>
        </header>
        <p className="replay-entry-drawer__content">{entry.content}</p>
        {responseVisible && entry.nudge_response ? (
          <section className="replay-entry-drawer__response" aria-labelledby={`${id}-response-title`}>
            <h3 id={`${id}-response-title`}>Response to the Nudge</h3>
            <p className="replay-entry-drawer__content">{entry.nudge_response}</p>
          </section>
        ) : null}
      </section>
    </div>,
    document.body,
  );
}
