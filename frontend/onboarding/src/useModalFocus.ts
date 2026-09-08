import { useEffect, useRef, type RefObject } from "react";

const focusableSelector = [
  "button", "a[href]", "input", "select", "textarea", "[tabindex]", "summary",
].join(",");

export default function useModalFocus(
  open: boolean,
  overlayRef: RefObject<HTMLDivElement | null>,
  initialFocusRef: RefObject<HTMLButtonElement | null>,
  onClose: () => void,
) {
  const onCloseRef = useRef(onClose);
  useEffect(() => { onCloseRef.current = onClose; }, [onClose]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!open || !overlay) return;

    const background = Array.from(document.body.children)
      .filter((element) => element !== overlay)
      .map((element) => ({ element, inert: element.hasAttribute("inert") }));
    for (const { element } of background) element.setAttribute("inert", "");
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const focusInitial = () => initialFocusRef.current?.focus({ preventScroll: true });
    focusInitial();

    const keepFocusInside = (event: FocusEvent) => {
      if (event.target instanceof Node && !overlay.contains(event.target)) {
        focusInitial();
      }
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        onCloseRef.current();
      }
      if (event.key !== "Tab") return;
      const focusable = Array.from(overlay.querySelectorAll<HTMLElement>(focusableSelector))
        .filter((element) => element.tabIndex >= 0
          && !element.matches(":disabled")
          && !element.closest("[hidden], [inert]"));
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first?.focus();
      }
    };
    document.addEventListener("focusin", keepFocusInside);
    document.addEventListener("keydown", handleKeyDown, true);
    return () => {
      document.removeEventListener("focusin", keepFocusInside);
      document.removeEventListener("keydown", handleKeyDown, true);
      document.body.style.overflow = previousOverflow;
      for (const { element, inert } of background) {
        if (!inert) element.removeAttribute("inert");
      }
    };
  }, [open, overlayRef, initialFocusRef]);
}
