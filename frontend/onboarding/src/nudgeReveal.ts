import type { NudgeInteractionContract } from "./demoContracts";

export const NUDGE_REVEAL_DELAY_MS = 800;

export function isDisplayableNudge(
  nudge: NudgeInteractionContract,
): boolean {
  return Boolean(nudge.text)
    && ["displayed", "answered", "skipped"].includes(nudge.outcome);
}
