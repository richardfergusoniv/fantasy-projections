import { Link, useLocation } from "react-router-dom";
import { useAppState } from "../hooks/useAppState";

/**
 * League control for the app shell, with a shortcut to the Regression Model
 * draft board on its left.
 *
 * Week / season pickers and the historical-leagues toggle lived here previously;
 * the app now defaults to the active season and the latest synced week.
 */
export function LeagueSwitcher() {
  const location = useLocation();
  const onDraftScreen = location.pathname.startsWith("/draft");
  const pane = new URLSearchParams(location.search).get("pane");
  const regressionActive =
    onDraftScreen && pane !== "checklist" && pane !== "assistant" && pane !== "draft-assistant";

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
      <Link
        to="/draft"
        className={`shell-pane-link${regressionActive ? " is-active" : ""}`}
        aria-current={regressionActive ? "page" : undefined}
      >
        Regression Model
      </Link>

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
