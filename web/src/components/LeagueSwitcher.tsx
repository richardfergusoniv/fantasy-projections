import { useAppState } from "../hooks/useAppState";

/**
 * League control for the app shell.
 *
 * Week / season pickers and the historical-leagues toggle lived here
 * previously; the app now defaults to the active season and the latest synced
 * week. A "Regression Model" shortcut lived here too, on every screen — it
 * named an implementation rather than a result, and pointed at one of the two
 * Draft boards from places (Lineup, Waivers, Trade) that have nothing to do
 * with drafting. Both draft boards are now tabs on the Draft screen itself.
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
