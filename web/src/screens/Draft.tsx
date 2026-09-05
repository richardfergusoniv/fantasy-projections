import { Fragment, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { AsyncStateBanner } from "../components/AsyncState";
import { FreshnessBadge } from "../components/FreshnessBadge";
import { Panel } from "../components/Panel";
import { MaybeNumber } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { api } from "../api/client";
import type {
  DraftBoard,
  DraftBoardEntry,
  DraftChecklist,
  DraftChecklistEntry,
} from "../api/types";

const DRAFTED_STORAGE_PREFIX = "fantasy-decisions:drafted";

type DraftPane = "checklist" | "ours";

const DRAFT_PANES: Array<[DraftPane, string]> = [
  ["ours", "Our Rankings"],
  ["checklist", "Draft Checklist"],
];

function paneFromSearch(value: string | null): DraftPane {
  if (value === "checklist" || value === "assistant" || value === "draft-assistant") {
    return "checklist";
  }
  if (value === "ours" || value === "rankings") return "ours";
  return "ours";
}

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

const CHECKLIST_POSITIONS = ["ALL", "QB", "RB", "WR", "TE"] as const;

const CHECK_SHORT_LABELS: Record<string, string> = {
  pass_att_rank: "PASS",
  rush_att_rank: "RUSH",
  offense_pts_rank: "PTS",
  offense_yds_rank: "YDS",
  ol_rank: "OL",
  sos_rank: "SOS",
  tgt_rank: "TGT",
  qb_rank: "QB",
};

const RANK_TIER_ORDER: Record<string, number> = {
  market_avg: 0,
  screenshot: 1,
  adp: 2,
  ecr: 3,
  prior_pts: 4,
  none: 5,
};

/** Overall ADP order for the All tab (cross-position). Position tabs keep board order. */
function checklistAdpSort(a: DraftChecklistEntry, b: DraftChecklistEntry): number {
  const tierA = RANK_TIER_ORDER[a.rank_tier] ?? 9;
  const tierB = RANK_TIER_ORDER[b.rank_tier] ?? 9;
  if (tierA !== tierB) return tierA - tierB;
  if (
    a.rank_tier === "market_avg" ||
    a.rank_tier === "screenshot" ||
    a.rank_tier === "adp"
  ) {
    return (a.adp ?? 9999) - (b.adp ?? 9999) || a.name.localeCompare(b.name);
  }
  if (a.rank_tier === "ecr") return (a.ecr ?? 9999) - (b.ecr ?? 9999);
  if (a.rank_tier === "prior_pts") return (b.prior_pts ?? 0) - (a.prior_pts ?? 0);
  return a.name.localeCompare(b.name);
}

function criteriaForEntry(
  checklist: DraftChecklist | null,
  entry: DraftChecklistEntry,
): string[] {
  return checklist?.criteria_by_position[entry.position] ?? [];
}

function shortCheckLabel(key: string, fullLabels: Record<string, string>): string {
  return CHECK_SHORT_LABELS[key] ?? fullLabels[key] ?? key;
}

function rankTierClass(rank: number | null | undefined): string {
  if (rank == null || Number.isNaN(rank)) return "is-muted";
  if (rank <= 8) return "is-strong";
  if (rank <= 16) return "is-mild";
  return "is-muted";
}

/** Average of available numeric ranks for the player's position criteria. */
function averageAvailableRank(entry: DraftChecklistEntry, keys: string[]): number | null {
  const values = keys
    .map((key) => entry.ranks[key])
    .filter((value): value is number => value != null && !Number.isNaN(value));
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export function DraftScreen() {
  const { selectedLeagueId, selectedLeague } = useAppState();
  const [searchParams, setSearchParams] = useSearchParams();
  const [pane, setPane] = useState<DraftPane>(() => paneFromSearch(searchParams.get("pane")));
  const [entries, setEntries] = useState<DraftBoardEntry[]>([]);
  const [checklist, setChecklist] = useState<DraftChecklist | null>(null);
  const [context, setContext] = useState<DraftBoard["context"]>();
  const [profile, setProfile] = useState<DraftBoard["profile"]>();
  const [positionFilter, setPositionFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const [visibleCount, setVisibleCount] = useState(25);
  const [draftedPlayerIds, setDraftedPlayerIds] = useState<string[]>([]);
  const [hideDrafted, setHideDrafted] = useState(true);
  const [maxAvgRank, setMaxAvgRank] = useState(0);
  const [dataAsOf, setDataAsOf] = useState<string | undefined>();
  const [runId, setRunId] = useState<string | undefined>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const storageKey = selectedLeagueId
    ? draftedStorageKey(selectedLeagueId, selectedLeague?.season)
    : null;

  useEffect(() => {
    setPane(paneFromSearch(searchParams.get("pane")));
  }, [searchParams]);

  function selectPane(next: DraftPane) {
    setPane(next);
    setVisibleCount(25);
    const params = new URLSearchParams(searchParams);
    if (next === "ours") {
      params.delete("pane");
    } else {
      params.set("pane", next);
    }
    setSearchParams(params, { replace: true });
  }

  useEffect(() => {
    setDraftedPlayerIds(storageKey ? loadDraftedPlayers(storageKey) : []);
    setVisibleCount(25);
  }, [storageKey]);

  useEffect(() => {
    if (!selectedLeagueId) {
      setEntries([]);
      setChecklist(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    // allSettled, not all: the two panes fail independently, so a checklist
    // outage must not blank the board that "Ours" renders (and vice versa).
    void Promise.allSettled([
      api.getDraftBoard(selectedLeagueId),
      api.getDraftChecklist(selectedLeagueId),
    ])
      .then(([boardResult, checklistResult]) => {
        if (cancelled) return;
        const board = boardResult.status === "fulfilled" ? boardResult.value : null;
        const checklistPayload =
          checklistResult.status === "fulfilled" ? checklistResult.value : null;

        setEntries(board?.entries ?? []);
        setChecklist(checklistPayload);
        setContext(board?.context);
        setProfile(board?.profile);
        setSearch("");
        setVisibleCount(25);
        setDataAsOf(checklistPayload?.meta.data_as_of ?? board?.meta.data_as_of);
        setRunId(
          checklistPayload?.meta.projection_run_id ?? board?.meta.projection_run_id,
        );

        const failures = [boardResult, checklistResult]
          .filter((result) => result.status === "rejected")
          .map((result) => String((result as PromiseRejectedResult).reason?.message ?? result));
        setError(failures.length ? failures.join(" · ") : null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedLeagueId]);

  const draftedPlayerSet = useMemo(() => new Set(draftedPlayerIds), [draftedPlayerIds]);
  const draftedCount = useMemo(() => {
    const pool =
      pane === "checklist"
        ? (checklist?.entries ?? []).map((entry) => entry.player_id)
        : entries.map((entry) => entry.player_id);
    return pool.filter((id) => draftedPlayerSet.has(id)).length;
  }, [pane, checklist, entries, draftedPlayerSet]);

  const normalizedSearch = search.trim().toLowerCase();

  const criteriaLabels = checklist?.criteria_labels ?? {};

  const checklistFiltered = useMemo(() => {
    const rows = checklist?.entries ?? [];
    const filtered = rows.filter((entry) => {
      if (positionFilter !== "ALL" && entry.position !== positionFilter) return false;
      if (
        normalizedSearch &&
        !entry.name.toLowerCase().includes(normalizedSearch) &&
        !(entry.team ?? "").toLowerCase().includes(normalizedSearch)
      ) {
        return false;
      }
      if (hideDrafted && draftedPlayerSet.has(entry.player_id)) return false;
      if (maxAvgRank > 0) {
        const keys = criteriaForEntry(checklist, entry);
        const avg = averageAvailableRank(entry, keys);
        if (avg == null || avg > maxAvgRank) return false;
      }
      return true;
    });
    // Single-position views keep the prepared positional order; All re-sorts
    // into overall market ADP → ECR → prior points so the board reads like a
    // real draft queue.
    if (positionFilter === "ALL") {
      return [...filtered].sort(checklistAdpSort);
    }
    return filtered;
  }, [checklist, positionFilter, normalizedSearch, hideDrafted, draftedPlayerSet, maxAvgRank]);

  const checklistVisible = checklistFiltered.slice(0, visibleCount);
  // Derive the divider from what is actually on screen. Anchoring it to the
  // entry's own unranked_break flag loses it whenever that one player is
  // filtered out (drafted, max-avg-rank) and strands it at the top when every
  // ranked player is filtered away.
  const unrankedBreakIndex = checklistVisible.findIndex(
    (entry) => entry.rank_tier === "prior_pts" || entry.rank_tier === "none",
  );

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
  const checklistTop = checklistFiltered[0];

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
  if (pane === "ours" && entries.length && entries.every((entry) => entry.vorp == null)) {
    missing.push("VORP");
  }
  if (pane === "ours" && entries.length && entries.every((entry) => entry.tier == null)) {
    missing.push("tiers");
  }

  const market = checklist?.checklist_meta.market_as_of;
  const rankSource = checklist?.checklist_meta.rank_source;
  const marketBadge =
    rankSource === "market_avg"
      ? `All tab by ADP (ESPN/FFC/MFL + FP ECR avg) · position tabs by board · Vegas + Sharp SOS ranks · ${
          market?.scoring ?? "ppr"
        }`
      : rankSource === "screenshot"
        ? `Checklist board · Vegas + Sharp SOS ranks · ${market?.scoring ?? "ppr"}`
        : market?.adp_end || market?.ecr_scrape
          ? `Market as of ADP ${market.adp_end ?? "—"} · ECR ${market.ecr_scrape ?? "—"} · ${
              market.scoring ?? "half-ppr"
            } · ${market.teams ?? 12}-team`
          : null;

  return (
    <div className="screen">
      <Panel
        title="Draft assistant"
        actions={<FreshnessBadge dataAsOf={dataAsOf} runId={runId} />}
      >
        <p className="muted">
          Draft Checklist All tab is ordered by ADP; context pills are Vegas volume/offense and Sharp
          SOS ranks. Our Rankings is the league VORP board. Mark drafted to hide a player across both.
        </p>

        <div className="draft-pane-tabs" role="tablist" aria-label="Draft views">
          {DRAFT_PANES.map(([id, label]) => (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={pane === id}
              className={`draft-pane-tab${pane === id ? " is-active" : ""}`}
              onClick={() => selectPane(id)}
            >
              {label}
            </button>
          ))}
        </div>

        <AsyncStateBanner
          label="Draft board"
          loading={loading}
          offline={false}
          error={error}
          fromCache={false}
          dataAsOf={dataAsOf}
          hasData={
            pane === "checklist"
              ? Boolean(checklist?.available && checklist.entries.length)
              : entries.length > 0
          }
          isEmpty={
            pane === "checklist"
              ? !checklist?.available || checklist.entries.length === 0
              : entries.length === 0
          }
          missing={missing}
          emptyMessage={
            pane === "checklist"
              ? "No draft checklist published. Run checklist_prepare after compare_prepare."
              : `No draft board published for ${
                  selectedLeague?.name ?? "this league"
                }. Promote a release to populate it.`
          }
        />

        <div className={`draft-board-controls${pane === "checklist" ? " is-checklist" : ""}`}>
          <div className="field draft-search-field">
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
          {pane === "checklist" ? (
            <div className="draft-pos-chips" role="group" aria-label="Position">
              {CHECKLIST_POSITIONS.map((position) => (
                <button
                  key={position}
                  type="button"
                  className={`draft-pos-chip${positionFilter === position ? " is-active" : ""}`}
                  aria-pressed={positionFilter === position}
                  onClick={() => {
                    setPositionFilter(position);
                    setVisibleCount(25);
                  }}
                >
                  {position === "ALL" ? "All" : position}
                </button>
              ))}
            </div>
          ) : (
            <div className="field">
              <label htmlFor="draft-position-filter">Position</label>
              <select
                id="draft-position-filter"
                value={positionFilter === "ALL" ? "ALL" : positionFilter}
                onChange={(event) => {
                  setPositionFilter(event.target.value);
                  setVisibleCount(25);
                }}
              >
                <option value="ALL">All positions</option>
                {(["QB", "RB", "WR", "TE"] as const).map((position) => (
                  <option key={position} value={position}>
                    {position}
                  </option>
                ))}
              </select>
            </div>
          )}
          {pane === "checklist" ? (
            <div className="field draft-max-avg-rank-field">
              <label htmlFor="draft-mold-filter">Max avg rank</label>
              <select
                id="draft-mold-filter"
                value={maxAvgRank}
                onChange={(event) => {
                  setMaxAvgRank(Number(event.target.value));
                  setVisibleCount(25);
                }}
              >
                <option value={0}>Any</option>
                <option value={8}>≤8</option>
                <option value={12}>≤12</option>
                <option value={16}>≤16</option>
                <option value={24}>≤24</option>
              </select>
            </div>
          ) : null}
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
            className="btn btn-secondary draft-undo-btn"
            type="button"
            disabled={draftedPlayerIds.length === 0}
            onClick={undoLastDrafted}
          >
            Undo
          </button>
        </div>

        {pane === "checklist" && checklist?.available ? (
          <>
            {marketBadge ? <p className="draft-market-badge">{marketBadge}</p> : null}
            <p className="muted draft-checklist-caveat">{checklist.checklist_meta.volume_caveat}</p>
            <div className="draft-status draft-status-compact">
              <span className="on-clock">
                Best: {checklistTop?.name ?? "—"}
                {checklistTop ? (
                  <span className="muted">
                    {" "}
                    · {checklistTop.position}
                    {checklistTop.adp != null
                      ? ` · ADP ${checklistTop.adp}`
                      : checklistTop.ecr != null
                        ? ` · ECR ${checklistTop.ecr}`
                        : ""}
                  </span>
                ) : null}
              </span>
            </div>
            <div className="draft-checklist-list" role="list">
              {checklistVisible.map((entry: DraftChecklistEntry, index: number) => {
                const drafted = draftedPlayerSet.has(entry.player_id);
                const keys = criteriaForEntry(checklist, entry);
                const overallRank =
                  positionFilter === "ALL" ? index + 1 : entry.pos_market_rank;
                return (
                  <Fragment key={entry.player_id}>
                    {index === unrankedBreakIndex ? (
                      <div className="draft-unranked-break-row" role="separator">
                        Unranked / off market — by 2025 pts
                      </div>
                    ) : null}
                    <article
                      className={`draft-checklist-row${drafted ? " is-drafted" : ""}`}
                      role="listitem"
                    >
                      <label className="draft-checklist-mark">
                        <input
                          type="checkbox"
                          checked={drafted}
                          aria-label={`${drafted ? "Undo" : "Mark"} ${entry.name} drafted`}
                          onChange={(event) =>
                            setDrafted(entry.player_id, event.target.checked)
                          }
                        />
                      </label>
                      <div className="draft-checklist-main">
                        <div className="draft-checklist-identity">
                          <span className="draft-rank">#{overallRank}</span>
                          <strong>{entry.name}</strong>
                          <span className={`pos-badge ${entry.position}`}>{entry.position}</span>
                          <span className="muted">
                            {entry.team ?? ""}
                            {entry.adp != null
                              ? ` · ADP ${entry.adp}`
                              : entry.ecr != null
                                ? ` · ECR ${entry.ecr}`
                                : entry.prior_pts != null
                                  ? ` · ${entry.prior_pts.toFixed(0)} pts`
                                  : ""}
                          </span>
                        </div>
                        <div className="draft-rank-pills" aria-label={`${entry.name} ranks`}>
                          {keys.map((key) => {
                            const rank = entry.ranks[key];
                            const label = shortCheckLabel(key, criteriaLabels);
                            const display =
                              rank == null || Number.isNaN(rank) ? "—" : String(rank);
                            return (
                              <span
                                key={key}
                                className={`draft-rank-pill ${rankTierClass(rank)}`}
                                title={`${criteriaLabels[key] ?? key}: ${display}`}
                              >
                                {label} {display}
                              </span>
                            );
                          })}
                        </div>
                      </div>
                    </article>
                  </Fragment>
                );
              })}
            </div>
            <p className="muted">
              Showing {checklistVisible.length} of {checklistFiltered.length} matching players ·{" "}
              {draftedCount} drafted.
            </p>
            {checklistVisible.length < checklistFiltered.length ? (
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

        {pane === "ours" && entries.length ? (
          <>
            <p className="muted">
              League-specific season rankings from the sealed release. Scoring, team count, fixed
              starters, FLEX, and SUPER_FLEX determine replacement value. Raw season points never
              determine the overall draft order.
            </p>
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
                <p className="muted">
                  Ranked by {profile.ranking_basis === "league_vorp" ? "league VORP" : "sealed VORP"},
                  not raw quarterback points · projected points are season totals
                </p>
                {profile.roster_positions.length ? (
                  <p className="muted">
                    Starting structure:{" "}
                    {profile.roster_positions
                      .filter((slot) => !["BN", "IR", "TAXI"].includes(slot))
                      .join(" · ")}
                  </p>
                ) : null}
                {Object.keys(profile.replacement_ranks).length ? (
                  <p className="muted">
                    Replacement ranks:{" "}
                    {Object.entries(profile.replacement_ranks)
                      .map(([position, rank]) => `${position}${rank}`)
                      .join(" · ")}
                  </p>
                ) : null}
              </div>
            ) : null}
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
                        <strong>
                          {entry.points_mean != null ? entry.points_mean.toFixed(1) : "—"}
                        </strong>
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
              Showing {visibleEntries.length} of {filteredEntries.length} matching players ·{" "}
              {draftedCount} drafted.
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
