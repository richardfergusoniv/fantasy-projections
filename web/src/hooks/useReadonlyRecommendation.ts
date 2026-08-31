import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import {
  assertNoAuthTokens,
  cacheRecommendation,
  getCachedRecommendation,
} from "../cache/recommendationsCache";
import type {
  LineupRecommendation,
  OpponentMode,
  ReadonlyRecommendationKey,
  WaiverRecommendation,
} from "../api/types";

interface UseRecommendationOptions<T> {
  key: ReadonlyRecommendationKey;
  leagueId: string | null;
  week: number | null;
  /** Request parameters that change the payload, e.g. the opponent mode. */
  variant?: string;
  fetcher: (leagueId: string, week: number) => Promise<T>;
}

export interface RecommendationState<T> {
  data: T | null;
  cachedAt: string | null;
  /** True when `data` was read from the offline cache, not this session's fetch. */
  fromCache: boolean;
  loading: boolean;
  offline: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useReadonlyRecommendation<T extends LineupRecommendation | WaiverRecommendation>({
  key,
  leagueId,
  week,
  variant,
  fetcher,
}: UseRecommendationOptions<T>): RecommendationState<T> {
  const [data, setData] = useState<T | null>(null);
  const [cachedAt, setCachedAt] = useState<string | null>(null);
  const [fromCache, setFromCache] = useState(false);
  const [loading, setLoading] = useState(false);
  const [offline, setOffline] = useState(!navigator.onLine);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onOnline = () => setOffline(false);
    const onOffline = () => setOffline(true);
    window.addEventListener("online", onOnline);
    window.addEventListener("offline", onOffline);
    return () => {
      window.removeEventListener("online", onOnline);
      window.removeEventListener("offline", onOffline);
    };
  }, []);

  const hydrateFromCache = useCallback(() => {
    if (!leagueId || week == null) return false;
    const cached = getCachedRecommendation<T>({ key, leagueId, week, variant });
    if (cached) {
      setData(cached.payload);
      setCachedAt(cached.fetchedAt);
      setFromCache(true);
      return true;
    }
    return false;
  }, [key, leagueId, variant, week]);

  const refresh = useCallback(async () => {
    if (!leagueId || week == null) return;
    setLoading(true);
    setError(null);
    try {
      const result = await fetcher(leagueId, week);
      assertNoAuthTokens(result);
      cacheRecommendation({ key, leagueId, week, variant }, result);
      setData(result);
      setCachedAt(new Date().toISOString());
      setFromCache(false);
    } catch (err) {
      const recovered = hydrateFromCache();
      if (!recovered) {
        setData(null);
      }
      setError(err instanceof Error ? err.message : "Failed to load recommendations");
    } finally {
      setLoading(false);
    }
  }, [fetcher, hydrateFromCache, key, leagueId, variant, week]);

  useEffect(() => {
    // Clear first: a stale payload from the previous league/week/variant must
    // never be shown under the new selection while the new request is inflight.
    setData(null);
    setCachedAt(null);
    setFromCache(false);
    setError(null);
    hydrateFromCache();
    if (navigator.onLine) {
      void refresh();
    }
  }, [hydrateFromCache, refresh]);

  return { data, cachedAt, fromCache, loading, offline, error, refresh };
}

export function useLineupRecommendation(
  leagueId: string | null,
  week: number | null,
  opponentMode: OpponentMode = "current",
) {
  const fetcher = useCallback(
    (id: string, wk: number) => api.getLineup(id, wk, opponentMode),
    [opponentMode],
  );
  return useReadonlyRecommendation({
    key: "lineup",
    leagueId,
    week,
    variant: `opponent_mode=${opponentMode}`,
    fetcher,
  });
}

export function useWaiverRecommendation(leagueId: string | null, week: number | null) {
  const fetcher = useCallback((id: string, wk: number) => api.getWaivers(id, wk), []);
  return useReadonlyRecommendation({ key: "waivers", leagueId, week, fetcher });
}
