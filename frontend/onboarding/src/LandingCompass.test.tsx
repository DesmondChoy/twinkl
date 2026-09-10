import { act, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import LandingCompass from "./LandingCompass";

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

function motionHarness(reduced = false) {
  const listeners = new Set<() => void>();
  const media = {
    matches: reduced,
    addEventListener: (_: string, listener: () => void) => listeners.add(listener),
    removeEventListener: (_: string, listener: () => void) => listeners.delete(listener),
  };
  const frames = new Map<number, FrameRequestCallback>();
  let nextId = 0;
  vi.stubGlobal("matchMedia", () => media);
  vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
    frames.set(++nextId, callback);
    return nextId;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => frames.delete(id));
  vi.spyOn(document, "hidden", "get").mockReturnValue(false);
  return {
    frames, listeners,
    tick(time: number) {
      const pending = [...frames.values()];
      frames.clear();
      act(() => pending.forEach(callback => callback(time)));
    },
    reduce() {
      media.matches = true;
      act(() => listeners.forEach(listener => listener()));
    },
  };
}

describe("LandingCompass", () => {
  it("keeps searching over time and releases its animation on unmount", () => {
    const motion = motionHarness();
    const { container, unmount } = render(<LandingCompass />);
    const needle = container.querySelector<HTMLElement>(".entry-compass__needle")!;
    motion.tick(0);
    const initial = needle.style.transform;
    for (let time = 50; time <= 30000; time += 50) motion.tick(time);
    expect(needle.style.transform).not.toBe(initial);
    expect(motion.frames.size).toBe(1);
    unmount();
    expect(motion.frames.size).toBe(0);
    expect(motion.listeners.size).toBe(0);
  });

  it("renders without animation when reduced motion is requested", () => {
    const motion = motionHarness(true);
    const { container, unmount } = render(<LandingCompass />);
    expect(motion.frames.size).toBe(0);
    expect(container.firstElementChild?.getAttribute("aria-hidden")).toBe("true");
    unmount();
  });

  it("stops and restores the static pose when reduced motion changes", () => {
    const motion = motionHarness();
    const { container, unmount } = render(<LandingCompass />);
    motion.tick(0);
    motion.tick(50);
    motion.reduce();
    expect(motion.frames.size).toBe(0);
    expect(container.querySelector<HTMLElement>(".entry-compass__body")!.style.transform).toBe("");
    expect(container.querySelector<HTMLElement>(".entry-compass__needle")!.style.transform).toBe("");
    unmount();
  });

  it("pauses offscreen and resumes without a time jump", () => {
    const motion = motionHarness();
    const { container, unmount } = render(<LandingCompass />);
    motion.tick(0);
    motion.tick(50);
    const needle = container.querySelector<HTMLElement>(".entry-compass__needle")!;
    const before = needle.style.transform;
    vi.spyOn(document, "hidden", "get").mockReturnValue(true);
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    expect(motion.frames.size).toBe(0);
    vi.spyOn(document, "hidden", "get").mockReturnValue(false);
    act(() => document.dispatchEvent(new Event("visibilitychange")));
    motion.tick(60000);
    expect(needle.style.transform).toBe(before);
    expect(motion.frames.size).toBe(1);
    unmount();
  });
});
