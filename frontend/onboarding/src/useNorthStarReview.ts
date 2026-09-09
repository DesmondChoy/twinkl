import { useCallback, useEffect, useRef, useState } from "react";
import type { OnboardingProfile } from "./domain";
import type { ExperienceState } from "./session";
import { createExperienceSession, ExperienceApiError, readExperienceTrace, reviewNorthStar } from "./experienceApi";
import { currentNorthStarEvent, useNorthStarProfileRef } from "./northStar";

export default function useNorthStarReview({
  profile, experience, updateExperience, enabled, busy, autoReview = true,
}: {
  profile: OnboardingProfile;
  experience: ExperienceState;
  updateExperience: (patch: Partial<ExperienceState>) => void;
  enabled: boolean;
  busy: boolean;
  autoReview?: boolean;
}) {
  const profileRef = useNorthStarProfileRef(profile);
  const identity = JSON.stringify({
    profile, digest: experience.weekly_digest, drift: experience.drift_result,
    entries: experience.journal_entries, revision: experience.revision,
  });
  const latest = useRef({ identity, revision: experience.revision, enabled, busy });
  latest.current = { identity, revision: experience.revision, enabled, busy };
  const mounted = useRef(false);
  const running = useRef<string | null>(null);
  const attempts = useRef(new Map<string, number>());
  const [pendingIdentity, setPendingIdentity] = useState<string | null>(null);
  const [failure, setFailure] = useState<{ identity: string; retryable: boolean } | null>(null);
  const result = currentNorthStarEvent({
    events: experience.trace_events, profile, profileRef,
    weeklyDigest: experience.weekly_digest, journalEntries: experience.journal_entries,
  });
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const run = useCallback(async (retry: boolean) => {
    const weekStart = experience.weekly_digest?.week_start;
    if (!enabled || busy || !profileRef || typeof weekStart !== "string"
      || running.current !== null) return;
    const count = attempts.current.get(identity) ?? 0;
    if (count >= 2 || (!retry && (count > 0
      || (result !== null && result.record.status !== "pending")))) return;
    const expectedRevision = experience.revision;
    const stillCurrent = () => mounted.current && latest.current.enabled
      && !latest.current.busy && latest.current.identity === identity
      && latest.current.revision === expectedRevision;
    running.current = identity;
    attempts.current.set(identity, count + 1);
    setPendingIdentity(identity);
    setFailure(null);
    let acceptedRevision: number | null = null;
    let acceptedEventIds: string[] | null = null;
    let settled = false;
    try {
      const request = { sessionId: profile.session_id, expectedRevision, weekStart, retry };
      let response;
      try {
        response = await reviewNorthStar(request);
      } catch (error) {
        const traceIds = new Set(experience.trace_events.map((event) => event.event_id));
        const hasCompleteTrace = traceIds.size > 0
          && experience.trace_event_ids.length === traceIds.size
          && experience.trace_event_ids.every((id) => traceIds.has(id));
        if (!(error instanceof ExperienceApiError) || error.code !== "session_not_found"
          || !hasCompleteTrace || !stillCurrent()) throw error;
        const restored = await createExperienceSession(profile, {
          session_id: profile.session_id, revision: expectedRevision,
          journal_entries: experience.journal_entries, nudges: experience.nudges,
          assessment_clock: experience.assessment_clock, trace_events: experience.trace_events,
        });
        if (!stillCurrent()) return;
        if (restored.session.revision !== expectedRevision) {
          throw new ExperienceApiError("The saved review belongs to an older session.", "session_conflict", false);
        }
        response = await reviewNorthStar(request);
      }
      if (!stillCurrent()) return;
      if (response.session.session_id !== profile.session_id
        || response.session.revision !== expectedRevision) {
        throw new ExperienceApiError("The review belongs to a different session state.", "session_conflict", false);
      }
      acceptedRevision = response.session.revision;
      acceptedEventIds = response.session.trace_event_ids;
      const trace = await readExperienceTrace(profile.session_id);
      if (!stillCurrent()) return;
      if (trace.session_id !== profile.session_id) {
        throw new ExperienceApiError("The review trace belongs to a different session.", "session_conflict", false);
      }
      updateExperience({
        revision: acceptedRevision,
        trace_event_ids: acceptedEventIds,
        trace_events: trace.events,
      });
      settled = true;
    } catch (error) {
      if (!stillCurrent()) return;
      if (acceptedRevision !== null && acceptedEventIds !== null) {
        updateExperience({ revision: acceptedRevision, trace_event_ids: acceptedEventIds });
      }
      setFailure({
        identity,
        retryable: error instanceof ExperienceApiError ? error.retryable : true,
      });
      settled = true;
    } finally {
      running.current = null;
      if (!settled) attempts.current.delete(identity);
      if (mounted.current) setPendingIdentity(null);
    }
  }, [busy, enabled, experience, identity, profile, profileRef, result, updateExperience]);

  useEffect(() => {
    if (!autoReview || !enabled || busy || !profileRef || !experience.weekly_digest
      || !experience.drift_result || experience.pending_submission) return;
    void run(false);
  }, [autoReview, busy, enabled, experience.drift_result, experience.pending_submission, experience.weekly_digest, profileRef, run]);

  const pending = pendingIdentity === identity
    || (result?.record.status === "pending" && failure?.identity !== identity
      && (attempts.current.get(identity) ?? 0) === 0);
  const failed = failure?.identity === identity || result?.record.status === "failed";
  const retryable = (attempts.current.get(identity) ?? 0) < 2
    && (failure?.identity === identity ? failure.retryable
      : result?.record.status === "failed" && result.record.retryable);
  return { pending, failed, retryable, retry: () => void run(true) };
}
