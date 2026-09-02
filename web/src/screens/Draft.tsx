import { useEffect, useMemo, useState } from "react";
import { AsyncStateBanner } from "../components/AsyncState";
import { FreshnessBadge } from "../components/FreshnessBadge";
import { Panel } from "../components/Panel";
import { MaybeNumber } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { api } from "../api/client";
import type { DraftBoard, DraftBoardEntry } from "../api/types";

const DRAFTED_STORAGE_PREFIX = "fantasy-decisions:drafted";

function draftedStorageKey(leagueId: string, season?: number): string {
  return `${DRAFTED_STORAGE_PREFIX}:${leagueId}:${season ?? "current"}`;
}

function loadDraftedPlayers(key: string): string[] {
  try {
    const value = JSON.parse(localStorage.getItem(key) ?? "[]") as unknown;
    if (!Array.isArray(value)) return [];
    return [...new Set(value.map(String).filter(Boolean))];
  } catch {
    return [];
  }
}

function saveDraftedPlayers(key: string, playerIds: string[]): void {
  localStorage.setItem(key, JSON.stringify(playerIds));
}

function formatVorp(value: number | undefined): string {
  if (value == null) return "—";
  if (Math.abs(value) < 0.05) return "Replacement";
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
}

export function DraftScreen() {
  const { selectedLeagueId, selectedLeague } = useAppState();
  const [entries, setEntries] = useState<DraftBoardEntry[]>([]);
  const [context, setContext] = useState<DraftBoard["context"]>();
  const [profile, setProfile] = useState<DraftBoard["profile"]>();
  const [positionFilter, setPositionFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const [visibleCount, setVisibleCount] = useState(25);
  const [draftedPlayerIds, setDraftedPlayerIds] = useState<string[]>([]);
  const [hideDrafted, setHideDrafted] = useState(true);
  const [dataAsOf, setDataAsOf] = useState<string | undefined>();
  const [runId, setRunId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const storageKey = selectedLeagueId
    ? draftedStorageKey(selectedLeagueId, selectedLeague?.season)
    : null;

  useEffect(() => {
    setDraftedPlayerIds(storageKey ? loadDraftedPlayers(storageKey) : []);
    setVisibleCount(25);
  }, [storageKey]);

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
        setProfile(board.profile);
        setPositionFilter("ALL");
        setSearch("");
        setVisibleCount(25);
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

  const draftedPlayerSet = useMemo(() => new Set(draftedPlayerIds), [draftedPlayerIds]);
  const draftedCount = entries.filter((entry) => draftedPlayerSet.has(entry.player_id)).length;
  const normalizedSearch = search.trim().toLowerCase();
  const positionEntries = entries.filter(
    (entry) => positionFilter === "ALL" || entry.position === positionFilter,
  );
  const searchedEntries = positionEntries.filter(
    (entry) =>
      !normalizedSearch ||
      entry.name.toLowerCase().includes(normalizedSearch) ||
      (entry.team ?? "").toLowerCase().includes(normalizedSearch),
  );
  const filteredEntries = hideDrafted
    ? searchedEntries.filter((entry) => !draftedPlayerSet.has(entry.player_id))
    : searchedEntries;
  const visibleEntries = filteredEntries.slice(0, visibleCount);
  const top = searchedEntries.find((entry) => !draftedPlayerSet.has(entry.player_id));

  function setDrafted(playerId: string, drafted: boolean): void {
    if (!storageKey) return;
    setDraftedPlayerIds((current) => {
      const next = drafted
        ? current.includes(playerId)
          ? current
          : [...current, playerId]
        : current.filter((id) => id !== playerId);
      saveDraftedPlayers(storageKey, next);
      return next;
    });
  }

  function undoLastDrafted(): void {
    const last = draftedPlayerIds.at(-1);
    if (last) setDrafted(last, false);
  }
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
          League-specific season rankings from the sealed release. Scoring, team count, fixed
          starters, FLEX, and SUPER_FLEX determine replacement value.
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
            {profile ? (
              <div className="stack">
                <p className="muted">
                  {profile.league_specific ? "League-adjusted" : "Default format"}
                  {profile.team_count != null ? ` · ${profile.team_count} teams` : ""}
                  {profile.scoring_fidelity ? ` · ${profile.scoring_fidelity}` : ""}
                </p>
                {profile.roster_positions.length ? (
                  <p className="muted">Starting structure: {profile.roster_positions.filter((slot) => !["BN", "IR", "TAXI"].includes(slot)).join(" · ")}</p>
                ) : null}
                {Object.keys(profile.replacement_ranks).length ? (
                  <p className="muted">
                    Replacement ranks: {Object.entries(profile.replacement_ranks)
                      .map(([position, rank]) => `${position}${rank}`)
                      .join(" · ")}
                  </p>
                ) : null}
              </div>
            ) : null}
            <div className="draft-board-controls">
              <div className="field">
                <label htmlFor="draft-player-search">Search players</label>
                <input
                  id="draft-player-search"
                  type="search"
                  value={search}
                  placeholder="Name or team"
                  onChange={(event) => {
                    setSearch(event.target.value);
                    setVisibleCount(25);
                  }}
                />
              </div>
              <div className="field">
                <label htmlFor="draft-position-filter">Position</label>
                <select
                  id="draft-position-filter"
                  value={positionFilter}
                  onChange={(event) => {
                    setPositionFilter(event.target.value);
                    setVisibleCount(25);
                  }}
                >
                  <option value="ALL">All positions</option>
                  {(["QB", "RB", "WR", "TE"] as const).map((position) => (
                    <option key={position} value={position}>{position}</option>
                  ))}
                </select>
              </div>
              <label className="draft-toggle">
                <input
                  type="checkbox"
                  checked={hideDrafted}
                  onChange={(event) => {
                    setHideDrafted(event.target.checked);
                    setVisibleCount(25);
                  }}
                />
                Hide drafted ({draftedCount})
              </label>
              <button
                className="btn btn-secondary"
                type="button"
                disabled={draftedPlayerIds.length === 0}
                onClick={undoLastDrafted}
              >
                Undo last drafted
              </button>
            </div>
            <div className="draft-player-grid" role="list">
              {visibleEntries.map((entry) => {
                const drafted = draftedPlayerSet.has(entry.player_id);
                return (
                  <article
                    key={entry.player_id}
                    className={`draft-player-card${drafted ? " is-drafted" : ""}`}
                    role="listitem"
                    aria-label={`${entry.name} draft card`}
                  >
                    <div className="draft-player-card-header">
                      <span className="draft-rank">#{entry.rank}</span>
                      <div className="draft-player-identity">
                        <h3>{entry.name}</h3>
                        <p>
                          <span className={`pos-badge ${entry.position}`}>{entry.position}</span>
                          {entry.team ? ` · ${entry.team}` : ""}
                          {entry.tier != null ? ` · Tier ${entry.tier}` : ""}
                        </p>
                      </div>
                      <button
                        className={`btn draft-player-action${drafted ? " is-drafted" : ""}`}
                        type="button"
                        aria-pressed={drafted}
                        aria-label={`${drafted ? "Undo" : "Mark"} ${entry.name} drafted`}
                        onClick={() => setDrafted(entry.player_id, !drafted)}
                      >
                        {drafted ? "Undo" : "Mark drafted"}
                      </button>
                    </div>
                    <div className="draft-card-stats">
                      <div className="draft-card-stat">
                        <span className="label">VORP</span>
                        <strong className={entry.vorp != null && entry.vorp < 0 ? "negative" : ""}>
                          {formatVorp(entry.vorp)}
                        </strong>
                        <span className="hint">vs league replacement</span>
                      </div>
                      <div className="draft-card-stat">
                        <span className="label">Projected points</span>
                        <strong>{entry.points_mean != null ? entry.points_mean.toFixed(1) : "—"}</strong>
                        <span className="hint">league-adjusted season</span>
                      </div>
                    </div>
                    <p className="draft-replacement-note">
                      {entry.replacement_rank != null
                        ? `${entry.position}${entry.replacement_rank} replacement`
                        : "Replacement rank unavailable"}
                      {entry.replacement_points != null
                        ? ` · ${entry.replacement_points.toFixed(1)} pts`
                        : ""}
                    </p>
                  </article>
                );
              })}
            </div>
            <p className="muted">
              Showing {visibleEntries.length} of {filteredEntries.length} matching players · {draftedCount} drafted.
            </p>
            {visibleEntries.length < filteredEntries.length ? (
              <button
                className="btn btn-secondary"
                type="button"
                onClick={() => setVisibleCount((current) => current + 25)}
              >
                Show 25 more
              </button>
            ) : null}
          </>
        ) : null}
      </Panel>
    </div>
  );
}
