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
import { readLocal, writeLocal } from "../storage/safeStorage";

const LEAGUE_KEY = "fantasy-decisions:selected-league";
const WEEK_KEY = "fantasy-decisions:selected-week";
const SHOW_ALL_LEAGUES_KEY = "fantasy-decisions:show-all-leagues";

/** Current fantasy season for default league filtering (draft / checklist). */
export const ACTIVE_SEASON = 2026;

interface AppStateValue {
  leagues: LeagueSummary[];
  visibleLeagues: LeagueSummary[];
  configuredLeagueIds: string[];
  showAllLeagues: boolean;
  setShowAllLeagues: (show: boolean) => void;
  selectedLeague: LeagueSummary | null;
  selectedLeagueId: string | null;
  selectLeague: (leagueId: string) => void;
  leaguesLoading: boolean;
  leaguesError: string | null;
  refreshLeagues: () => Promise<void>;

  /** Season of the selected league, straight from `GET /leagues`. */
  season: number | null;
  /** Active fantasy season used when historical leagues are hidden. */
  activeSeason: number;

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
  const raw = readLocal(`${WEEK_KEY}:${leagueId}`);
  if (!raw) return null;
  const parsed = Number(raw);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

export function isLeagueDecisionReady(league: LeagueSummary): boolean {
  return (
    league.decision_ready === true ||
    (league.owner_roster_id != null && Number.isFinite(league.owner_roster_id))
  );
}

/**
 * Keep the current selection when it is an active-season league and either it is
 * decision-ready, or no decision-ready leagues exist yet (same gate as
 * `visibleLeagues`). Only snap away from historical / non-ready picks when a
 * ready alternative is available.
 */
export function shouldKeepSelectedLeague(
  selected: LeagueSummary | undefined,
  items: LeagueSummary[],
): boolean {
  if (selected?.season !== ACTIVE_SEASON) return false;
  const anyReady = items.some(
    (league) => league.season === ACTIVE_SEASON && isLeagueDecisionReady(league),
  );
  if (!anyReady) return true;
  return isLeagueDecisionReady(selected);
}

/** Prefer active/configured leagues where the owner has a synced membership. */
export function pickPreferredLeagueId(
  items: LeagueSummary[],
  configured: string[],
  current: string | null,
  options: { showAll: boolean },
): string | null {
  const { showAll } = options;
  const configuredSet = new Set(configured);
  const hasConfiguredOverlap =
    configured.length > 0 && items.some((league) => configuredSet.has(league.id));

  let pool = showAll
    ? items
    : items.filter((league) => league.season === ACTIVE_SEASON);
  if (!pool.length && !showAll) {
    pool = items.filter((league) => league.season === ACTIVE_SEASON);
  }
  if (hasConfiguredOverlap) {
    const configuredPool = pool.filter((league) => configuredSet.has(league.id));
    if (configuredPool.length) pool = configuredPool;
  }

  // Prefer leagues where the configured owner has a synced membership. Historical
  // seasons often exist in `league` without `league_member` rows and cannot load
  // lineup/waiver decisions.
  const withOwner = pool.filter(isLeagueDecisionReady);
  if (withOwner.length) {
    pool = withOwner;
  }

  if (current && pool.some((league) => league.id === current)) {
    return current;
  }
  // If the saved league has no owner membership, fall through to a usable one.
  return pool[0]?.id ?? null;
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
  const [configuredLeagueIds, setConfiguredLeagueIds] = useState<string[]>([]);
  // Historical toggle was removed from the UI, but a sticky localStorage flag
  // previously kept empty-membership leagues selectable. Always clear it.
  const [showAllLeagues, setShowAllLeaguesState] = useState<boolean>(() => {
    if (readLocal(SHOW_ALL_LEAGUES_KEY) === "true") {
      writeLocal(SHOW_ALL_LEAGUES_KEY, "false");
    }
    return false;
  });
  const [selectedLeagueId, setSelectedLeagueId] = useState<string | null>(() =>
    readLocal(LEAGUE_KEY),
  );
  const [leaguesLoading, setLeaguesLoading] = useState(true);
  const [leaguesError, setLeaguesError] = useState<string | null>(null);

  const [rosters, setRosters] = useState<Roster[]>([]);
  const [rostersLoading, setRostersLoading] = useState(false);
  const [rostersError, setRostersError] = useState<string | null>(null);

  const [weekOverride, setWeekOverride] = useState<number | null>(() =>
    readStoredWeek(readLocal(LEAGUE_KEY)),
  );

  const selectLeague = useCallback((leagueId: string) => {
    setSelectedLeagueId(leagueId);
    writeLocal(LEAGUE_KEY, leagueId);
    setWeekOverride(readStoredWeek(leagueId));
  }, []);

  const setShowAllLeagues = useCallback((show: boolean) => {
    setShowAllLeaguesState(show);
    writeLocal(SHOW_ALL_LEAGUES_KEY, show ? "true" : "false");
  }, []);

  const refreshLeagues = useCallback(async () => {
    setLeaguesLoading(true);
    setLeaguesError(null);
    try {
      const { leagues: items, configuredLeagueIds: configured } = await api.getLeagues();
      setLeagues(items);
      setConfiguredLeagueIds(configured);
      setSelectedLeagueId((current) => {
        // Ignore obsolete show-all sticky state; decisions need an owner roster.
        const next = pickPreferredLeagueId(items, configured, current, {
          showAll: false,
        });
        if (next) {
          writeLocal(LEAGUE_KEY, next);
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
        writeLocal(`${WEEK_KEY}:${selectedLeagueId}`, String(next));
      }
    },
    [selectedLeagueId],
  );

  const visibleLeagues = useMemo(() => {
    const configuredSet = new Set(configuredLeagueIds);
    const hasConfiguredOverlap =
      configuredLeagueIds.length > 0 &&
      leagues.some((league) => configuredSet.has(league.id));

    let pool = showAllLeagues
      ? leagues
      : leagues.filter((league) => league.season === ACTIVE_SEASON);

    if (!showAllLeagues && hasConfiguredOverlap) {
      const configuredPool = pool.filter((league) => configuredSet.has(league.id));
      if (configuredPool.length) pool = configuredPool;
    }

    // Hide leagues that cannot resolve the owner whenever any decision-ready
    // league exists. This removes the need to manually clear show-all storage.
    const ready = pool.filter(isLeagueDecisionReady);
    if (ready.length) pool = ready;
    return pool;
  }, [configuredLeagueIds, leagues, showAllLeagues]);

  // Snap away from historical / non-decision-ready selections automatically.
  // The historical toggle was removed from the UI; do not honor sticky show-all.
  // When no decision-ready leagues exist (e.g. e2e without SLEEPER_USER_ID), keep
  // any active-season selection — same gate as visibleLeagues.
  useEffect(() => {
    if (leaguesLoading || !leagues.length) return;
    const selected = leagues.find((league) => league.id === selectedLeagueId);
    if (shouldKeepSelectedLeague(selected, leagues)) return;
    const next = pickPreferredLeagueId(leagues, configuredLeagueIds, null, {
      showAll: false,
    });
    if (next && next !== selectedLeagueId) {
      setSelectedLeagueId(next);
      writeLocal(LEAGUE_KEY, next);
      setWeekOverride(readStoredWeek(next));
    }
  }, [
    configuredLeagueIds,
    leagues,
    leaguesLoading,
    selectedLeagueId,
  ]);

  const selectedLeague =
    leagues.find((league) => league.id === selectedLeagueId) ?? null;

  const value = useMemo<AppStateValue>(
    () => ({
      leagues,
      visibleLeagues,
      configuredLeagueIds,
      showAllLeagues,
      setShowAllLeagues,
      selectedLeague,
      selectedLeagueId,
      selectLeague,
      leaguesLoading,
      leaguesError,
      refreshLeagues,
      season: selectedLeague?.season ?? null,
      activeSeason: ACTIVE_SEASON,
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
      configuredLeagueIds,
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
      setShowAllLeagues,
      setWeek,
      showAllLeagues,
      visibleLeagues,
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
