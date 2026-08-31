import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api } from "../api/client";
import type { LeagueSummary, Roster } from "../api/types";

const LEAGUE_KEY = "fantasy-decisions:selected-league";
const WEEK_KEY = "fantasy-decisions:selected-week";

interface AppStateValue {
  leagues: LeagueSummary[];
  selectedLeague: LeagueSummary | null;
  selectedLeagueId: string | null;
  selectLeague: (leagueId: string) => void;
  leaguesLoading: boolean;
  leaguesError: string | null;
  refreshLeagues: () => Promise<void>;

  /** Season of the selected league, straight from `GET /leagues`. */
  season: number | null;

  /** Roster snapshots for the selected league, from `GET /leagues/{id}/rosters`. */
  rosters: Roster[];
  rostersLoading: boolean;
  rostersError: string | null;

  /**
   * Weeks the API actually has roster snapshots for. This is the source of the
   * default week: there is no "current NFL week" field on any read endpoint, so
   * the app uses the weeks its own data covers and lets the user pick.
   */
  availableWeeks: number[];
  /** Currently selected week, or null while the league's weeks are unknown. */
  week: number | null;
  setWeek: (week: number) => void;
  /** True when the week came from a saved user choice rather than the data. */
  weekIsUserChosen: boolean;
}

const AppStateContext = createContext<AppStateValue | null>(null);

function readStoredWeek(leagueId: string | null): number | null {
  if (!leagueId) return null;
  const raw = localStorage.getItem(`${WEEK_KEY}:${leagueId}`);
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

/**
 * Single source of truth for league selection and the season/week the screens
 * read against.
 *
 * It lives above the router so every screen shares one `GET /leagues` and one
 * roster fetch, instead of each mount issuing its own.
 */
export function AppStateProvider({ children }: { children: ReactNode }) {
  const [leagues, setLeagues] = useState<LeagueSummary[]>([]);
  const [selectedLeagueId, setSelectedLeagueId] = useState<string | null>(() =>
    localStorage.getItem(LEAGUE_KEY),
  );
  const [leaguesLoading, setLeaguesLoading] = useState(true);
  const [leaguesError, setLeaguesError] = useState<string | null>(null);

  const [rosters, setRosters] = useState<Roster[]>([]);
  const [rostersLoading, setRostersLoading] = useState(false);
  const [rostersError, setRostersError] = useState<string | null>(null);

  const [weekOverride, setWeekOverride] = useState<number | null>(() =>
    readStoredWeek(localStorage.getItem(LEAGUE_KEY)),
  );

  const selectLeague = useCallback((leagueId: string) => {
    setSelectedLeagueId(leagueId);
    localStorage.setItem(LEAGUE_KEY, leagueId);
    setWeekOverride(readStoredWeek(leagueId));
  }, []);

  const refreshLeagues = useCallback(async () => {
    setLeaguesLoading(true);
    setLeaguesError(null);
    try {
      const items = await api.getLeagues();
      setLeagues(items);
      setSelectedLeagueId((current) => {
        if (current && items.some((league) => league.id === current)) {
          return current;
        }
        const next = items[0]?.id ?? null;
        if (next) {
          localStorage.setItem(LEAGUE_KEY, next);
        }
        return next;
      });
    } catch (err) {
      setLeaguesError(err instanceof Error ? err.message : "Failed to load leagues");
    } finally {
      setLeaguesLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshLeagues();
  }, [refreshLeagues]);

  useEffect(() => {
    if (!selectedLeagueId) {
      setRosters([]);
      return;
    }
    let cancelled = false;
    setRostersLoading(true);
    setRostersError(null);
    void api
      .getRosters(selectedLeagueId)
      .then((items) => {
        if (!cancelled) setRosters(items);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setRosters([]);
          setRostersError(err.message);
        }
      })
      .finally(() => {
        if (!cancelled) setRostersLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedLeagueId]);

  const availableWeeks = useMemo(() => {
    const weeks = new Set<number>();
    for (const roster of rosters) {
      if (Number.isInteger(roster.week) && roster.week > 0) {
        weeks.add(roster.week);
      }
    }
    return [...weeks].sort((a, b) => a - b);
  }, [rosters]);

  const derivedWeek = availableWeeks.length ? availableWeeks[availableWeeks.length - 1] : null;
  const week = weekOverride ?? derivedWeek;

  const setWeek = useCallback(
    (next: number) => {
      setWeekOverride(next);
      if (selectedLeagueId) {
        localStorage.setItem(`${WEEK_KEY}:${selectedLeagueId}`, String(next));
      }
    },
    [selectedLeagueId],
  );

  const selectedLeague =
    leagues.find((league) => league.id === selectedLeagueId) ?? null;

  const value = useMemo<AppStateValue>(
    () => ({
      leagues,
      selectedLeague,
      selectedLeagueId,
      selectLeague,
      leaguesLoading,
      leaguesError,
      refreshLeagues,
      season: selectedLeague?.season ?? null,
      rosters,
      rostersLoading,
      rostersError,
      availableWeeks,
      week,
      setWeek,
      weekIsUserChosen: weekOverride != null,
    }),
    [
      availableWeeks,
      leagues,
      leaguesError,
      leaguesLoading,
      refreshLeagues,
      rosters,
      rostersError,
      rostersLoading,
      selectLeague,
      selectedLeague,
      selectedLeagueId,
      setWeek,
      week,
      weekOverride,
    ],
  );

  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState(): AppStateValue {
  const ctx = useContext(AppStateContext);
  if (!ctx) {
    throw new Error("useAppState must be used within AppStateProvider");
  }
  return ctx;
}
