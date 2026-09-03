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
import { readMagicLinkToken, stripMagicLinkTokenFromUrl } from "../pwa/magicLink";

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

/** Persisted across PWA kills on iOS (sessionStorage is not). */
const CSRF_KEY = "fantasy-decisions:csrf";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sessionExpired, setSessionExpired] = useState(false);
  const userRef = useRef<User | null>(null);
  userRef.current = user;

  const persistCsrf = useCallback((token: string) => {
    localStorage.setItem(CSRF_KEY, token);
    api.setCsrfToken(token);
  }, []);

  const restoreCsrf = useCallback(() => {
    const stored = localStorage.getItem(CSRF_KEY);
    if (stored) {
      api.setCsrfToken(stored);
    }
  }, []);

  const clearLocalSession = useCallback(() => {
    localStorage.removeItem(CSRF_KEY);
    api.setCsrfToken(undefined);
    setUser(null);
  }, []);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const me = await api.getMe();
      setUser({ id: me.id, email: me.email });
      setSessionExpired(false);
      if (me.csrf_token) {
        persistCsrf(me.csrf_token);
      }
    } catch (err) {
      setUser(null);
      if (!(err instanceof ApiClientError) || err.status !== 401) {
        setError(
          err instanceof Error
            ? `Could not reach the server: ${err.message}`
            : "Could not reach the server",
        );
      }
    }
  }, [persistCsrf]);

  useEffect(() => {
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
    const linkToken = readMagicLinkToken();
    if (linkToken) {
      // LoginScreen verifies magic links and then refreshes.
      setLoading(false);
      return;
    }
    void refresh().finally(() => setLoading(false));
  }, [refresh, restoreCsrf]);

  const login = useCallback(async (email: string) => {
    setError(null);
    const result = await api.requestMagicLink(email);
    return { devLink: result.link };
  }, []);

  const verify = useCallback(
    async (token: string) => {
      setError(null);
      const result = await api.verifyMagicLink(token);
      persistCsrf(result.csrf_token);
      stripMagicLinkTokenFromUrl();
      setSessionExpired(false);
      await refresh();
    },
    [persistCsrf, refresh],
  );

  const logout = useCallback(async () => {
    try {
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
