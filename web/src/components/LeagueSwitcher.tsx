import { useAppState } from "../hooks/useAppState";

/**
 * League and week controls for the app shell.
 *
 * Both live here rather than on Home so every screen can switch without
 * navigating away, and both read the one shared `GET /leagues` fetch.
 */
export function LeagueSwitcher() {
  const {
    leagues,
    selectedLeagueId,
    selectLeague,
    leaguesLoading,
    leaguesError,
    availableWeeks,
    week,
    setWeek,
    rostersLoading,
  } = useAppState();

  return (
    <div className="shell-controls">
      <div className="shell-control">
        <label htmlFor="shell-league-select">League</label>
        <select
          id="shell-league-select"
          value={selectedLeagueId ?? ""}
          onChange={(event) => selectLeague(event.target.value)}
          disabled={leaguesLoading || leagues.length === 0}
        >
          {leaguesLoading ? <option value="">Loading leagues…</option> : null}
          {!leaguesLoading && leagues.length === 0 ? (
            <option value="">No leagues synced</option>
          ) : null}
          {leagues.map((league) => (
            <option key={league.id} value={league.id}>
              {league.name} · {league.season}
            </option>
          ))}
        </select>
      </div>

      <div className="shell-control">
        <label htmlFor="shell-week-select">Week</label>
        <select
          id="shell-week-select"
          value={week ?? ""}
          onChange={(event) => setWeek(Number(event.target.value))}
          disabled={rostersLoading || availableWeeks.length === 0}
        >
          {rostersLoading ? <option value="">Loading…</option> : null}
          {!rostersLoading && availableWeeks.length === 0 ? (
            <option value="">No weeks synced</option>
          ) : null}
          {availableWeeks.map((value) => (
            <option key={value} value={value}>
              Week {value}
            </option>
          ))}
        </select>
      </div>

      {leaguesError ? (
        <p className="error-text" role="alert">
          Leagues unavailable: {leaguesError}
        </p>
      ) : null}
    </div>
  );
}
