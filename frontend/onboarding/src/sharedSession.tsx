import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  clearSession,
  createSession,
  inspectRun as selectInspectRun,
  loadOrCreateSession,
  persistSession,
  showView as selectView,
  type DemoView,
  type ExperienceState,
  type OnboardingSession,
} from "./session";

interface SharedSessionValue {
  session: OnboardingSession;
  updateSession: (patch: Partial<OnboardingSession>) => void;
  updateExperience: (patch: Partial<ExperienceState>) => void;
  showView: (view: DemoView) => void;
  inspectRun: (eventId: string) => void;
  restart: () => void;
}

const SharedSessionContext = createContext<SharedSessionValue | null>(null);

export function SharedSessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<OnboardingSession>(() => loadOrCreateSession());

  useEffect(() => {
    persistSession(session);
  }, [session]);

  const updateSession = useCallback((patch: Partial<OnboardingSession>) => {
    setSession((current) => ({ ...current, ...patch }));
  }, []);

  const updateExperience = useCallback((patch: Partial<ExperienceState>) => {
    setSession((current) => ({
      ...current,
      experience: {
        ...current.experience,
        ...patch,
      },
    }));
  }, []);

  const showView = useCallback((view: DemoView) => {
    setSession((current) => selectView(current, view));
  }, []);

  const inspectRun = useCallback((eventId: string) => {
    setSession((current) => selectInspectRun(current, eventId));
  }, []);

  const restart = useCallback(() => {
    clearSession();
    setSession(createSession());
  }, []);

  const value = useMemo(
    () => ({
      session,
      updateSession,
      updateExperience,
      showView,
      inspectRun,
      restart,
    }),
    [session, updateSession, updateExperience, showView, inspectRun, restart],
  );

  return <SharedSessionContext.Provider value={value}>{children}</SharedSessionContext.Provider>;
}

export function useSharedSession(): SharedSessionValue {
  const value = useContext(SharedSessionContext);
  if (!value) throw new Error("useSharedSession must be used within SharedSessionProvider");
  return value;
}
