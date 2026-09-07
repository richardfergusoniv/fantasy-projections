import { useAppState } from "../hooks/useAppState";

/**
 * League control for the app shell.
 *
 * Week / season pickers and the historical-leagues toggle lived here previously;
 * the app now defaults to the active season and the latest synced week.
 * Regression Model is hidden while that board is developed locally.
 */
export function LeagueSwitcher() {
  const {
    visibleLeagues,
    selectedLeagueId,
    selectLeague,
    leaguesLoading,
    leaguesError,
    activeSeason,
  } = useAppState();

  return (
    <div className="shell-controls">
      <div className="shell-control">
        <label htmlFor="shell-league-select">Leagues</label>
        <select
          id="shell-league-select"
          value={selectedLeagueId ?? ""}
          onChange={(event) => selectLeague(event.target.value)}
          disabled={leaguesLoading || visibleLeagues.length === 0}
        >
          {leaguesLoading ? <option value="">Loading leagues…</option> : null}
          {!leaguesLoading && visibleLeagues.length === 0 ? (
            <option value="">No {activeSeason} leagues synced</option>
          ) : null}
          {visibleLeagues.map((league) => (
            <option key={league.id} value={league.id}>
              {league.name}
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
