import { useEffect, useRef } from "react";

export default function LandingCompass() {
  const hostRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const needleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const host = hostRef.current;
    const body = bodyRef.current;
    const needle = needleRef.current;
    if (!host || !body || !needle || !window.matchMedia) return;
    const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let frame = 0;
    let previous: number | null = null;
    let elapsed = 0;
    let targetX = 0;
    let targetY = 0;
    let pointerX = 0;
    let pointerY = 0;

    const animate = (now: number) => {
      const dt = previous === null ? 0 : Math.min((now - previous) / 1000, 0.05);
      previous = now;
      elapsed += dt;
      // Exponential damping keeps pointer response consistent across refresh rates.
      const blend = 1 - Math.exp(-4 * dt);
      pointerX += (targetX - pointerX) * blend;
      pointerY += (targetY - pointerY) * blend;
      const t = elapsed;
      const pitch = 32 + 5 * Math.sin(t * 0.37) - pointerY * 9;
      const yaw = -12 + 8 * Math.sin(t * 0.29) + pointerX * 12;
      const roll = -8 + 3 * Math.sin(t * 0.23);
      body.style.transform = `translateY(${4 * Math.sin(t * 0.6)}px) rotateX(${pitch}deg) rotateY(${yaw}deg) rotateZ(${roll}deg)`;
      // Superposed waves search continuously without waypoint pauses or loop resets.
      const heading = 32 + 54 * Math.sin(t * 0.48) + 17 * Math.sin(t * 0.83);
      needle.style.transform = `translateZ(18px) rotateZ(${heading}deg)`;
      body.style.setProperty("--compass-light", `${125 + yaw * 2}deg`);
      frame = requestAnimationFrame(animate);
    };
    const resetPointer = () => { targetX = 0; targetY = 0; };
    const movePointer = (event: PointerEvent) => {
      if (event.pointerType === "touch") return;
      const bounds = host.getBoundingClientRect();
      targetX = Math.max(-1, Math.min(1, (event.clientX - bounds.left) / bounds.width * 2 - 1));
      targetY = Math.max(-1, Math.min(1, (event.clientY - bounds.top) / bounds.height * 2 - 1));
    };
    const syncMotion = () => {
      cancelAnimationFrame(frame);
      previous = null;
      if (motion.matches) {
        body.style.removeProperty("transform");
        body.style.removeProperty("--compass-light");
        needle.style.removeProperty("transform");
        resetPointer();
        pointerX = 0;
        pointerY = 0;
      } else if (!document.hidden) {
        frame = requestAnimationFrame(animate);
      }
    };
    motion.addEventListener("change", syncMotion);
    document.addEventListener("visibilitychange", syncMotion);
    host.addEventListener("pointermove", movePointer);
    host.addEventListener("pointerleave", resetPointer);
    syncMotion();
    return () => {
      cancelAnimationFrame(frame);
      motion.removeEventListener("change", syncMotion);
      document.removeEventListener("visibilitychange", syncMotion);
      host.removeEventListener("pointermove", movePointer);
      host.removeEventListener("pointerleave", resetPointer);
    };
  }, []);

  return (
    <div className="entry-intro__compass" ref={hostRef} aria-hidden="true">
      <div className="entry-compass__orbit" />
      <div className="entry-compass__body" ref={bodyRef}>
        <div className="entry-compass__base" />
        <div className="entry-compass__face">
          <svg viewBox="0 0 240 240" fill="none">
            <circle cx="120" cy="120" r="108" stroke="#9aafc4" />
            <circle cx="120" cy="120" r="93" stroke="#63788f" strokeDasharray="1 8.74" />
            <circle cx="120" cy="120" r="72" stroke="#63788f" strokeOpacity="0.4" />
            <path d="M120 20v17m0 166v17M20 120h17m166 0h17" stroke="#d3dee8" />
            <path d="m53 53 7 7m120 120 7 7M53 187l7-7M180 60l7-7" stroke="#63788f" />
          </svg>
        </div>
        <div className="entry-compass__needle" ref={needleRef}>
          <svg viewBox="0 0 240 240" fill="none">
            <path d="m120 39 19 81-19-9Z" fill="#ffad83" />
            <path d="m120 39-19 81 19-9Z" fill="#e76d43" />
            <path d="m120 201 19-81-19 9Z" fill="#f7faf9" />
            <path d="m120 201-19-81 19 9Z" fill="#91a6bf" />
          </svg>
        </div>
        <div className="entry-compass__pivot" />
        <div className="entry-compass__glass" />
      </div>
    </div>
  );
}
