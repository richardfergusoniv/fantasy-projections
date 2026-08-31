import { useEffect, useMemo, useState } from "react";
import { AsyncStateBanner } from "../components/AsyncState";
import { Panel } from "../components/Panel";
import { MaybeNumber } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { api } from "../api/client";

export function DynastyScreen() {
  const { selectedLeagueId, selectedLeague, rosters, rostersLoading, rostersError } = useAppState();
  const [state, setState] = useState<Awaited<ReturnType<typeof api.getDynastyState>> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const rosterIds = useMemo(
    () => [...new Set(rosters.map((roster) => roster.roster_id))].sort((a, b) => a - b),
    [rosters],
  );
  // The API exposes no owner-to-roster mapping, so the roster is an explicit
  // user choice seeded from the rosters the API actually returned.
  const [rosterId, setRosterId] = useState<number | null>(null);
  useEffect(() => {
    setRosterId(rosterIds[0] ?? null);
  }, [rosterIds]);

  const isDynasty = Boolean(selectedLeague?.is_dynasty);

  useEffect(() => {
    if (!selectedLeagueId || !isDynasty || rosterId == null) {
      setState(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    void api
      .getDynastyState(selectedLeagueId, rosterId)
      .then((next) => {
        if (!cancelled) setState(next);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setState(null);
          setError(err.message);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [isDynasty, rosterId, selectedLeagueId]);

  const manager = state?.manager_state as Record<string, unknown> | undefined;
  const pick = state?.rookie_pick_projection as Record<string, unknown> | undefined;
  const probabilities = (manager?.probabilities as Record<string, number> | undefined) ?? {};

  return (
    <div className="screen">
      <Panel title="Dynasty">
        {!isDynasty ? (
          <p className="state-notice state-empty">
            <span className="state-glyph" aria-hidden="true">
              ∅
            </span>
            <span>
              <strong className="state-title">Not a dynasty league.</strong>{" "}
              {selectedLeague?.name ?? "The selected league"} is redraft, so there is no manager
              state or rookie pick projection. Switch to a dynasty league in the header.
            </span>
          </p>
        ) : (
          <>
            <div className="field">
              <label htmlFor="dynasty-roster">Roster</label>
              <select
                id="dynasty-roster"
                value={rosterId ?? ""}
                onChange={(event) => setRosterId(Number(event.target.value))}
                disabled={rosterIds.length === 0}
              >
                {rosterIds.length === 0 ? <option value="">No rosters</option> : null}
                {rosterIds.map((id) => (
                  <option key={id} value={id}>
                    Roster {id}
                  </option>
                ))}
              </select>
            </div>

            <AsyncStateBanner
              label="Dynasty state"
              loading={loading || rostersLoading}
              offline={false}
              error={error ?? rostersError}
              fromCache={false}
              hasData={Boolean(state)}
              isEmpty={!state}
              emptyMessage="The API returned no manager state for this roster."
            />

            {state ? (
              <ul className="ops-list">
                <li>
                  <span>Manager state</span>
                  <strong>{String(manager?.label ?? "not available")}</strong>
                </li>
                <li>
                  <span>Rookie pick slot</span>
                  <strong>
                    {String(pick?.projected_slot ?? "not available")} (
                    {String(pick?.rule ?? "rule not published")})
                  </strong>
                </li>
                {Object.entries(probabilities).map(([label, value]) => (
                  <li key={label}>
                    <span>{label} probability</span>
                    <strong>
                      <MaybeNumber value={value} digits={0} percent />
                    </strong>
                  </li>
                ))}
              </ul>
            ) : null}
          </>
        )}
      </Panel>
    </div>
  );
}
