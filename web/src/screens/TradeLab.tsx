import { useEffect, useMemo, useState } from "react";
import { AsyncStateBanner } from "../components/AsyncState";
import { Panel } from "../components/Panel";
import { MaybeNumber } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { api } from "../api/client";
import type { TradeEvaluation } from "../api/types";

type Horizon = TradeEvaluation["horizon"];

const HORIZONS: Horizon[] = ["weekly", "ros", "dynasty"];

function PlayerPicker({
  legend,
  players,
  selected,
  onToggle,
  emptyMessage,
  playerLabels,
}: {
  legend: string;
  players: string[];
  selected: string[];
  onToggle: (playerId: string) => void;
  emptyMessage: string;
  playerLabels: Map<string, string>;
}) {
  if (players.length === 0) {
    return (
      <fieldset className="player-picker">
        <legend>{legend}</legend>
        <p className="muted">{emptyMessage}</p>
      </fieldset>
    );
  }
  return (
    <fieldset className="player-picker">
      <legend>{legend}</legend>
      <div className="player-options">
        {players.map((playerId) => {
          const id = `${legend.replace(/\W+/g, "-").toLowerCase()}-${playerId}`;
          const label = playerLabels.get(playerId) ?? playerId;
          return (
            <div key={playerId} className="player-option">
              <input
                type="checkbox"
                id={id}
                checked={selected.includes(playerId)}
                onChange={() => onToggle(playerId)}
              />
              <label htmlFor={id}>{label}</label>
            </div>
          );
        })}
      </div>
    </fieldset>
  );
}

export function TradeLabScreen() {
  const { selectedLeagueId, selectedLeague, rosters, rostersLoading, rostersError } = useAppState();

  const rosterIds = useMemo(
    () => [...new Set(rosters.map((roster) => roster.roster_id))].sort((a, b) => a - b),
    [rosters],
  );

  const [sideARosterId, setSideARosterId] = useState<number | null>(null);
  const [sideBRosterId, setSideBRosterId] = useState<number | null>(null);
  const [sideAPlayers, setSideAPlayers] = useState<string[]>([]);
  const [sideBPlayers, setSideBPlayers] = useState<string[]>([]);
  const [horizon, setHorizon] = useState<Horizon>("ros");
  const [result, setResult] = useState<TradeEvaluation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Default the two sides to real roster ids returned by the API rather than
  // the fixture ids 1 and 2.
  useEffect(() => {
    setSideAPlayers([]);
    setSideBPlayers([]);
    setResult(null);
    setSideARosterId(rosterIds[0] ?? null);
    setSideBRosterId(rosterIds[1] ?? rosterIds[0] ?? null);
  }, [rosterIds]);

  const playersFor = (rosterId: number | null) =>
    rosterId == null
      ? []
      : [
          ...new Set(
            rosters.filter((roster) => roster.roster_id === rosterId).flatMap((r) => r.players),
          ),
        ];

  const sideAPool = playersFor(sideARosterId);
  const sideBPool = playersFor(sideBRosterId);

  const playerLabels = useMemo(() => {
    const labels = new Map<string, string>();
    for (const roster of rosters) {
      for (const player of roster.player_details ?? []) {
        const position = player.position ? ` (${player.position})` : "";
        labels.set(player.player_id, `${player.name}${position}`);
      }
    }
    return labels;
  }, [rosters]);

  const managerLabel = (rosterId: number) => {
    const roster = rosters.find((row) => row.roster_id === rosterId);
    if (roster?.manager_name) {
      return `${roster.manager_name} (Roster ${rosterId})`;
    }
    return `Roster ${rosterId}`;
  };

  const canEvaluate =
    Boolean(selectedLeagueId) &&
    sideARosterId != null &&
    sideBRosterId != null &&
    sideAPlayers.length > 0 &&
    sideBPlayers.length > 0;

  async function evaluate() {
    if (!selectedLeagueId || sideARosterId == null || sideBRosterId == null) return;
    setLoading(true);
    setError(null);
    try {
      const evaluation = await api.evaluateTrade(
        selectedLeagueId,
        { roster_id: sideARosterId, player_ids: sideAPlayers },
        { roster_id: sideBRosterId, player_ids: sideBPlayers },
        horizon,
      );
      setResult(evaluation);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Trade evaluation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="screen">
      <Panel title="Trade Lab">
        <p className="muted">
          Values, fairness, and acceptance probability are computed by the API from the active
          release. This screen shows those numbers as published — it does not assign a letter grade.
        </p>

        <AsyncStateBanner
          label="Rosters"
          loading={rostersLoading}
          offline={false}
          error={rostersError}
          fromCache={false}
          hasData={rosters.length > 0}
          isEmpty={rosters.length === 0}
          emptyMessage={`No roster snapshots for ${
            selectedLeague?.name ?? "this league"
          }, so there are no players to trade. Run a sync from Operations.`}
        />

        <div className="stack">
          <div className="field">
            <label htmlFor="trade-side-a-roster">Your roster</label>
            <select
              id="trade-side-a-roster"
              value={sideARosterId ?? ""}
              onChange={(event) => {
                setSideARosterId(Number(event.target.value));
                setSideAPlayers([]);
              }}
              disabled={rosterIds.length === 0}
            >
              {rosterIds.length === 0 ? <option value="">No rosters</option> : null}
              {rosterIds.map((id) => (
                <option key={id} value={id}>
                  {managerLabel(id)}
                </option>
              ))}
            </select>
          </div>

          <PlayerPicker
            legend="Players you send"
            players={sideAPool}
            selected={sideAPlayers}
            onToggle={(playerId) =>
              setSideAPlayers((prev) =>
                prev.includes(playerId)
                  ? prev.filter((id) => id !== playerId)
                  : [...prev, playerId],
              )
            }
            emptyMessage="This roster snapshot has no players."
            playerLabels={playerLabels}
          />

          <div className="field">
            <label htmlFor="trade-side-b-roster">Their roster</label>
            <select
              id="trade-side-b-roster"
              value={sideBRosterId ?? ""}
              onChange={(event) => {
                setSideBRosterId(Number(event.target.value));
                setSideBPlayers([]);
              }}
              disabled={rosterIds.length === 0}
            >
              {rosterIds.length === 0 ? <option value="">No rosters</option> : null}
              {rosterIds.map((id) => (
                <option key={id} value={id}>
                  {managerLabel(id)}
                </option>
              ))}
            </select>
          </div>

          <PlayerPicker
            legend="Players you receive"
            players={sideBPool}
            selected={sideBPlayers}
            onToggle={(playerId) =>
              setSideBPlayers((prev) =>
                prev.includes(playerId)
                  ? prev.filter((id) => id !== playerId)
                  : [...prev, playerId],
              )
            }
            emptyMessage="This roster snapshot has no players."
            playerLabels={playerLabels}
          />

          <div className="field">
            <label htmlFor="trade-horizon">Valuation horizon</label>
            <select
              id="trade-horizon"
              value={horizon}
              onChange={(event) => setHorizon(event.target.value as Horizon)}
            >
              {HORIZONS.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>

          <button
            className="btn btn-primary"
            type="button"
            onClick={() => void evaluate()}
            disabled={loading || !canEvaluate}
          >
            {loading ? "Evaluating…" : "Evaluate trade"}
          </button>
          {!canEvaluate && !loading ? (
            <p className="muted">Select at least one player on each side to evaluate.</p>
          ) : null}
        </div>

        {error ? (
          <p className="state-notice state-error" role="alert">
            <span className="state-glyph" aria-hidden="true">
              ✕
            </span>
            <span>
              <strong className="state-title">Evaluation failed.</strong> {error}
            </span>
          </p>
        ) : null}

        {result ? (
          <>
            <h3 className="section-title">Evaluation ({result.horizon} horizon)</h3>
            <div className="placeholder-grid">
              <div className="card-stat">
                <span className="label">Your side value</span>
                <span className="value">
                  <MaybeNumber value={result.objective.side_a_value} digits={1} />
                </span>
              </div>
              <div className="card-stat">
                <span className="label">Their side value</span>
                <span className="value">
                  <MaybeNumber value={result.objective.side_b_value} digits={1} />
                </span>
              </div>
              <div className="card-stat">
                <span className="label">Your net gain</span>
                <span className="value">
                  <MaybeNumber value={result.objective.side_a_gain} digits={1} />
                </span>
              </div>
              <div className="card-stat">
                <span className="label">Their net gain</span>
                <span className="value">
                  <MaybeNumber value={result.objective.side_b_gain} digits={1} />
                </span>
              </div>
              <div className="card-stat">
                <span className="label">Fairness gap</span>
                <span className="value">
                  <MaybeNumber value={result.fairness.gap} digits={2} />
                </span>
              </div>
              <div className="card-stat">
                <span className="label">Fairness uncertainty</span>
                <span className="value">
                  <MaybeNumber value={result.fairness.uncertainty} digits={2} />
                </span>
              </div>
              <div className="card-stat">
                <span className="label">Accept % (them)</span>
                <span className="value">
                  <MaybeNumber
                    value={result.acceptance.side_b_probability}
                    digits={0}
                    percent
                  />
                </span>
              </div>
              <div className="card-stat">
                <span className="label">Tendency adjustment</span>
                <span className="value">
                  <MaybeNumber value={result.acceptance.tendency_adjustment} digits={2} />
                </span>
              </div>
            </div>
            <p className="muted provenance">
              Verdict:{" "}
              {result.fairness.fair == null
                ? "fairness verdict not available"
                : result.fairness.fair
                  ? "the API considers this within its fairness band"
                  : "the API considers this outside its fairness band"}
              . Release <code>{result.meta.projection_run_id}</code>, data as of{" "}
              {new Date(result.meta.data_as_of).toLocaleString()}.
            </p>
          </>
        ) : null}
      </Panel>
    </div>
  );
}
