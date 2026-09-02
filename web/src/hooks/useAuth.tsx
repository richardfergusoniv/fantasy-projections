import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { api, ApiClientError } from "../api/client";
import { clearRecommendationCache } from "../cache/recommendationsCache";
import type { User } from "../api/types";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  error: string | null;
  /**
   * True when a session that was working has been rejected by the API. Distinct
   * from `user === null`, which is also the state of a first-time visitor.
   */
  sessionExpired: boolean;
  login: (email: string) => Promise<{ devLink?: string }>;
  verify: (token: string) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
  dismissSessionExpired: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

const CSRF_KEY = "fantasy-decisions:csrf";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sessionExpired, setSessionExpired] = useState(false);
  // Read inside the interceptor without re-subscribing on every user change.
  const userRef = useRef<User | null>(null);
  userRef.current = user;

  const restoreCsrf = useCallback(() => {
    const token = sessionStorage.getItem(CSRF_KEY);
    if (token) {
      api.setCsrfToken(token);
    }
  }, []);

  const hasMagicLinkToken = useCallback(() => {
    if (typeof window === "undefined") return false;
    const rawHash = window.location.hash.startsWith("#")
      ? window.location.hash.slice(1)
      : window.location.hash;
    if (new URLSearchParams(rawHash).get("token")) return true;
    return new URLSearchParams(window.location.search).get("token") != null;
  }, []);

  const clearLocalSession = useCallback(() => {
    sessionStorage.removeItem(CSRF_KEY);
    api.setCsrfToken(undefined);
    setUser(null);
  }, []);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const me = await api.getMe();
      setUser(me);
      setSessionExpired(false);
    } catch (err) {
      setUser(null);
      // A 401 means "not signed in", which is expected on a cold load and is
      // handled by the interceptor. Anything else is a real failure the user
      // needs to see rather than being bounced to a blank login screen.
      if (!(err instanceof ApiClientError) || err.status !== 401) {
        setError(
          err instanceof Error
            ? `Could not reach the server: ${err.message}`
            : "Could not reach the server",
        );
      }
    }
  }, []);

  useEffect(() => {
    // Central 401 handling. Any request from any screen that comes back
    // unauthorized ends the session once, here, instead of surfacing as an
    // unexplained per-screen error.
    return api.onUnauthorized(() => {
      if (userRef.current) {
        setSessionExpired(true);
        clearRecommendationCache();
      }
      clearLocalSession();
    });
  }, [clearLocalSession]);

  useEffect(() => {
    restoreCsrf();
    // On the magic-link callback the LoginScreen verifies and then refreshes.
    // Firing a cold getMe() here would 401 (no session yet) and its
    // onUnauthorized handler would clear the CSRF token that verify races to
    // set — leaving a valid session cookie but no CSRF token, so every later
    // mutation 403s. Let verify be the sole auth path when a link token exists.
    if (hasMagicLinkToken()) {
      setLoading(false);
      return;
    }
    void refresh().finally(() => setLoading(false));
  }, [refresh, restoreCsrf, hasMagicLinkToken]);

  const login = useCallback(async (email: string) => {
    setError(null);
    const result = await api.requestMagicLink(email);
    return { devLink: result.link };
  }, []);

  const verify = useCallback(
    async (token: string) => {
      setError(null);
      const result = await api.verifyMagicLink(token);
      sessionStorage.setItem(CSRF_KEY, result.csrf_token);
      api.setCsrfToken(result.csrf_token);
      setSessionExpired(false);
      await refresh();
    },
    [refresh],
  );

  const logout = useCallback(async () => {
    try {
      // Best effort: revoke server-side so the cookie cannot be replayed. A
      // failure here must still clear local state.
      await api.logout();
    } catch {
      // ignore — local teardown below is what the user asked for
    }
    clearRecommendationCache();
    clearLocalSession();
    setSessionExpired(false);
  }, [clearLocalSession]);

  const dismissSessionExpired = useCallback(() => setSessionExpired(false), []);

  const value = useMemo(
    () => ({
      user,
      loading,
      error,
      sessionExpired,
      login,
      verify,
      logout,
      refresh,
      dismissSessionExpired,
    }),
    [user, loading, error, sessionExpired, login, verify, logout, refresh, dismissSessionExpired],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return ctx;
}
