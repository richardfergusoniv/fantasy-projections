import { useEffect, useState } from "react";
import { AsyncStateBanner } from "../components/AsyncState";
import { FreshnessBadge } from "../components/FreshnessBadge";
import { Panel } from "../components/Panel";
import { MaybeNumber } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { api } from "../api/client";
import type { DraftBoard, DraftBoardEntry } from "../api/types";

export function DraftScreen() {
  const { selectedLeagueId, selectedLeague } = useAppState();
  const [entries, setEntries] = useState<DraftBoardEntry[]>([]);
  const [context, setContext] = useState<DraftBoard["context"]>();
  const [dataAsOf, setDataAsOf] = useState<string | undefined>();
  const [runId, setRunId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedLeagueId) {
      setEntries([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    void api
      .getDraftBoard(selectedLeagueId)
      .then((board) => {
        if (cancelled) return;
        setEntries(board.entries);
        setContext(board.context);
        setDataAsOf(board.meta.data_as_of);
        setRunId(board.meta.projection_run_id);
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setEntries([]);
        setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedLeagueId]);

  const top = entries[0];
  const missing: string[] = [];
  if (entries.length && entries.every((entry) => entry.vorp == null)) {
    missing.push("VORP");
  }
  if (entries.length && entries.every((entry) => entry.tier == null)) {
    missing.push("tiers");
  }

  return (
    <div className="screen">
      <Panel
        title="Draft"
        actions={<FreshnessBadge dataAsOf={dataAsOf} runId={runId} />}
      >
        <p className="muted">
          Tiered rankings and VORP from the sealed release bundle. Live Sleeper draft context
          connects when a draft is in progress.
        </p>

        <AsyncStateBanner
          label="Draft board"
          loading={loading}
          offline={false}
          error={error}
          fromCache={false}
          dataAsOf={dataAsOf}
          hasData={entries.length > 0}
          isEmpty={entries.length === 0}
          missing={missing}
          emptyMessage={`No draft board published for ${
            selectedLeague?.name ?? "this league"
          }. Promote a release to populate it.`}
        />

        {entries.length ? (
          <>
            <div className="draft-status">
              <span className="on-clock">Best available: {top?.name ?? "not available"}</span>
              <span className="pick-info">
                VORP <MaybeNumber value={top?.vorp ?? null} digits={1} /> · Tier{" "}
                {top?.tier ?? "not available"}
              </span>
            </div>
            {context ? (
              <p className="muted">
                Draft status: {context.draft_status}
                {context.draft_status === "live" && context.current_pick != null
                  ? ` · Pick ${context.current_pick}`
                  : ""}
                {context.nfl_week != null ? ` · NFL week ${context.nfl_week}` : ""}
                {context.season != null ? ` · season ${context.season}` : ""}
              </p>
            ) : null}
            <ul className="starter-list">
              {entries.slice(0, 15).map((entry) => (
                <li key={entry.player_id} className="roster-slot">
                  <span className="label">#{entry.rank}</span>
                  <span>
                    {entry.name}{" "}
                    <span className={`pos-badge ${entry.position}`}>{entry.position}</span>
                    {entry.vorp != null ? ` · VORP ${entry.vorp.toFixed(1)}` : ""}
                  </span>
                </li>
              ))}
            </ul>
          </>
        ) : null}
      </Panel>
    </div>
  );
}
