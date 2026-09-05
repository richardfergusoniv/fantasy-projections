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
type ChecklistSort = "adp" | "vorp";

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

const CHECKLIST_POSITIONS = ["ALL", "QB", "RB", "WR", "TE", "FLEX"] as const;
/** Standard FLEX pool: skill positions that can fill a FLEX seat (not QB). */
const FLEX_ELIGIBLE = new Set(["RB", "WR", "TE"]);

function matchesPositionFilter(position: string, filter: string): boolean {
  if (filter === "ALL") return true;
  if (filter === "FLEX") return FLEX_ELIGIBLE.has(position);
  return position === filter;
}

/** Cross-position boards (All / FLEX) re-sort by ADP; single-pos tabs keep board order. */
function isCrossPositionFilter(filter: string): boolean {
  return filter === "ALL" || filter === "FLEX";
}

const CHECK_SHORT_LABELS: Record<string, string> = {
  fp_rank: "FP",
  total_yds_rank: "YDS",
  rush_yds_rank: "RUSH",
  pass_td_rank: "PTD",
  total_td_rank: "TD",
  offense_pts_rank: "PTS",
  offense_yds_rank: "OFF",
  ol_rank: "OL",
  sos_rank: "SOS",
  rec_rank: "REC",
  rec_yds_rank: "RYDS",
  rec_td_rank: "RTD",
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

/** Overall ADP order for All / FLEX tabs (cross-position). Single-pos tabs keep board order. */
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

/** Order checklist rows by league VORP board rank (missing VORP last). */
function checklistVorpSort(
  a: DraftChecklistEntry,
  b: DraftChecklistEntry,
  vorpRankByPlayerId: Map<string, number>,
): number {
  const rankA = vorpRankByPlayerId.get(a.player_id);
  const rankB = vorpRankByPlayerId.get(b.player_id);
  const missingA = rankA == null || Number.isNaN(rankA);
  const missingB = rankB == null || Number.isNaN(rankB);
  if (missingA !== missingB) return missingA ? 1 : -1;
  if (!missingA && !missingB && rankA !== rankB) return rankA - rankB;
  return checklistAdpSort(a, b);
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

function sentimentTierClass(label: string | null | undefined): string {
  if (label === "positive") return "is-strong";
  if (label === "neutral") return "is-mild";
  if (label === "negative") return "is-negative";
  return "is-muted";
}

function sentimentDisplay(label: string | null | undefined): string {
  if (label === "positive") return "+";
  if (label === "neutral") return "~";
  if (label === "negative") return "−";
  return "—";
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
  const [checklistSort, setChecklistSort] = useState<ChecklistSort>("adp");
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

  /** League VORP board rank + tier (Our Rankings), keyed for checklist VORP sort. */
  const vorpRankByPlayerId = useMemo(() => {
    const map = new Map<string, number>();
    for (const entry of entries) {
      if (entry.player_id && entry.rank != null) {
        map.set(entry.player_id, entry.rank);
      }
    }
    return map;
  }, [entries]);


  const checklistFiltered = useMemo(() => {
    const rows = checklist?.entries ?? [];
    const filtered = rows.filter((entry) => {
      if (!matchesPositionFilter(entry.position, positionFilter)) return false;
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
    // ADP: All/FLEX re-sort by market ADP; single-pos tabs keep board order.
    // VORP: always order by league VORP board rank (Our Rankings).
    if (checklistSort === "vorp") {
      return [...filtered].sort((a, b) => checklistVorpSort(a, b, vorpRankByPlayerId));
    }
    if (isCrossPositionFilter(positionFilter)) {
      return [...filtered].sort(checklistAdpSort);
    }
    return filtered;
  }, [
    checklist,
    positionFilter,
    checklistSort,
    normalizedSearch,
    hideDrafted,
    draftedPlayerSet,
    maxAvgRank,
    vorpRankByPlayerId,
  ]);

  const checklistVisible = checklistFiltered.slice(0, visibleCount);
  // Derive the divider from what is actually on screen. Anchoring it to the
  // entry's own unranked_break flag loses it whenever that one player is
  // filtered out (drafted, max-avg-rank) and strands it at the top when every
  // ranked player is filtered away.
  const unrankedBreakIndex = checklistVisible.findIndex(
    (entry) => entry.rank_tier === "prior_pts" || entry.rank_tier === "none",
  );

  const positionEntries = entries.filter((entry) =>
    matchesPositionFilter(entry.position, positionFilter),
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
      ? `Sort by ADP or VORP · All/FLEX ADP uses ESPN/FFC/MFL + FP ECR avg · Vegas + Sharp SOS ranks · ${
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
          Draft Checklist sorts by ADP or VORP (toggle above the list). FLEX = RB/WR/TE. Context pills
          are Vegas volume/offense and Sharp SOS ranks. Our Rankings is VORP from Vegas season lines.
          Mark drafted to hide a player across both.
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
              ? checklist == null
                ? "Draft checklist request failed. Check the API error above, then confirm draft_checklist_2026.json is published."
                : "No draft checklist published for this season. Run checklist_prepare and ensure draft_checklist_2026.json is in draft_assistant/data/ (and the active release)."
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
            <>
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
            <div className="draft-sort-chips" role="group" aria-label="Sort checklist">
              {(
                [
                  ["adp", "ADP"],
                  ["vorp", "VORP"],
                ] as const
              ).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  className={`draft-pos-chip${checklistSort === value ? " is-active" : ""}`}
                  aria-pressed={checklistSort === value}
                  onClick={() => {
                    setChecklistSort(value);
                    setVisibleCount(25);
                  }}
                >
                  {label}
                </button>
              ))}
            </div>
            </>
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
                {(["QB", "RB", "WR", "TE", "FLEX"] as const).map((position) => (
                  <option key={position} value={position}>
                    {position === "FLEX" ? "FLEX (RB/WR/TE)" : position}
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
                    {checklistSort === "vorp"
                      ? (() => {
                          const rank = vorpRankByPlayerId.get(checklistTop.player_id);
                          return rank != null ? ` · VORP ${rank}` : " · VORP —";
                        })()
                      : checklistTop.adp != null
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
                const vorpRank = vorpRankByPlayerId.get(entry.player_id);
                const overallRank =
                  checklistSort === "vorp"
                    ? vorpRank
                    : isCrossPositionFilter(positionFilter)
                      ? index + 1
                      : entry.pos_market_rank;
                return (
                  <Fragment key={entry.player_id}>
                    {index === unrankedBreakIndex && checklistSort === "adp" ? (
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
                          <span className="draft-rank">#{overallRank ?? "—"}</span>
                          <strong>{entry.name}</strong>
                          <span className={`pos-badge ${entry.position}`}>{entry.position}</span>
                          <span className="muted">
                            {entry.team ?? ""}
                            {checklistSort === "vorp"
                              ? vorpRank != null && !Number.isNaN(vorpRank)
                                ? ` · VORP ${vorpRank}`
                                : " · VORP —"
                              : entry.adp != null
                                ? ` · ADP ${entry.adp}`
                                : entry.ecr != null
                                  ? ` · ECR ${entry.ecr}`
                                  : entry.prior_pts != null
                                    ? ` · ${entry.prior_pts.toFixed(0)} pts`
                                    : ""}
                            {checklistSort === "vorp"
                              ? entry.adp != null
                                ? ` · ADP ${entry.adp}`
                                : entry.ecr != null
                                  ? ` · ECR ${entry.ecr}`
                                  : ""
                              : vorpRank != null && !Number.isNaN(vorpRank)
                                ? ` · VORP ${vorpRank}`
                                : ""}
                            {entry.vegas_fp != null
                              ? ` · Vegas FP ${entry.vegas_fp.toFixed(1)}`
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
                          <span
                            className={`draft-rank-pill ${sentimentTierClass(
                              entry.sentiment?.label,
                            )}`}
                            title={
                              entry.sentiment?.label
                                ? `Sentiment ${entry.sentiment.label}${
                                    entry.sentiment.as_of ? ` as of ${entry.sentiment.as_of}` : ""
                                  }`
                                : "No reviewed sentiment signal"
                            }
                          >
                            SENT {sentimentDisplay(entry.sentiment?.label)}
                          </span>
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
              Rankings use Vegas season fantasy points (half-PPR / 4-pt pass TD from yards,
              receptions, and TDs). Scoring seats, FLEX, and SUPER_FLEX still set replacement.
              Raw quarterback points never set the overall order by themselves.
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
                  Ranked by{" "}
                  {profile.ranking_basis === "league_vorp"
                    ? "league VORP"
                    : profile.ranking_basis === "vegas_vorp"
                      ? "Vegas VORP"
                      : "sealed VORP"}
                  {profile.points_source === "vegas_fp" ? " (Vegas FP)" : ""},
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
              {visibleEntries.map((entry, index) => {
                const drafted = draftedPlayerSet.has(entry.player_id);
                const prevTier = index > 0 ? visibleEntries[index - 1]?.tier : undefined;
                const showTierBreak =
                  entry.tier != null && (index === 0 || entry.tier !== prevTier);
                return (
                  <Fragment key={entry.player_id}>
                    {showTierBreak ? (
                      <div className="draft-tier-break-row" role="separator">
                        Tier {entry.tier}
                      </div>
                    ) : null}
                  <article
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
                  </Fragment>
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
