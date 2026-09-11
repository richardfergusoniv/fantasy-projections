import { useMemo, useState } from "react";
import { AsyncStateBanner } from "../components/AsyncState";
import { CitationList } from "../components/CitationList";
import { FreshnessBadge } from "../components/FreshnessBadge";
import { isActionableInjuryEvidence } from "../components/injuryEvidence";
import { LineupSourceTabs } from "../components/LineupSourceTabs";
import { OpponentModeToggle, OPPONENT_MODES } from "../components/OpponentModeToggle";
import { Panel } from "../components/Panel";
import { MaybeNumber } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { useInjuryEvidence } from "../hooks/useInjuryEvidence";
import { useLineupRecommendation } from "../hooks/useReadonlyRecommendation";
import type { InjuryEvidence, LineupStarter, OpponentMode, PointsRange } from "../api/types";

function starterPoints(player: LineupStarter): PointsRange {
  return {
    p10: player.points_p10,
    p50: player.points_p50,
    p90: player.points_p90,
    mean: player.expected_points,
  };
}

function formatOpp(opponent: string | null | undefined): string {
  if (!opponent) return "—";
  const trimmed = opponent.trim();
  if (!trimmed) return "—";
  if (trimmed.startsWith("@") || trimmed.startsWith("vs")) return trimmed;
  return `vs ${trimmed}`;
}

function sumMeans(players: LineupStarter[]): number | null {
  let total = 0;
  let any = false;
  for (const player of players) {
    if (player.expected_points != null) {
      total += player.expected_points;
      any = true;
    }
  }
  return any ? total : null;
}

/** Align starters to league seat order so you/opp columns line up by slot. */
function orderByRosterSlots(players: LineupStarter[], seats: string[]): LineupStarter[] {
  if (!seats.length) return players;
  const remaining = [...players];
  const ordered: LineupStarter[] = [];
  for (const seat of seats) {
    const idx = remaining.findIndex((player) => (player.slot ?? player.position) === seat);
    if (idx >= 0) {
      ordered.push(remaining.splice(idx, 1)[0]!);
    }
  }
  return [...ordered, ...remaining];
}

function ProjCell({ player }: { player: LineupStarter | null }) {
  if (!player) {
    return (
      <div className="lineup-proj">
        <span className="lineup-proj-mean muted">—</span>
      </div>
    );
  }
  const pts = starterPoints(player);
  return (
    <div className="lineup-proj" aria-label={`Projected ${player.expected_points ?? "n/a"} points`}>
      <span className="lineup-proj-mean">
        <MaybeNumber value={player.expected_points} digits={1} />
      </span>
      {pts.p10 != null && pts.p90 != null ? (
        <span className="lineup-proj-range">
          {pts.p10.toFixed(1)}–{pts.p90.toFixed(1)}
        </span>
      ) : (
        <span className="lineup-proj-range lineup-proj-range-missing">range —</span>
      )}
    </div>
  );
}

function PlayerCell({
  player,
  evidence,
  side,
  selected,
  interactive,
  onSelect,
}: {
  player: LineupStarter | null;
  evidence?: InjuryEvidence;
  side: "you" | "opp";
  selected?: boolean;
  interactive?: boolean;
  onSelect?: () => void;
}) {
  if (!player) {
    return (
      <div className={`matchup-player matchup-player-empty matchup-player-${side}`}>
        <span className="muted">Empty</span>
      </div>
    );
  }
  const showEvidence = isActionableInjuryEvidence(evidence);
  const teamPos = [player.team, player.position].filter(Boolean).join(" · ");
  const locked = Boolean(player.locked);
  const content = (
    <>
      <span className="lineup-name">{player.name}</span>
      <span className="lineup-meta">
        {teamPos ? <span>{teamPos}</span> : null}
        <span className="lineup-opp">{formatOpp(player.opponent)}</span>
        {locked ? <span className="matchup-locked">Locked</span> : null}
      </span>
      {showEvidence && evidence ? (
        <div className="evidence lineup-evidence">
          <p className="muted">
            <span className={`injury-pill injury-${evidence.status.toLowerCase()}`}>
              {evidence.status}
            </span>{" "}
            {evidence.summary}
          </p>
          {evidence.sources.length ? (
            <CitationList
              citations={evidence.sources}
              label={`Injury sources for ${player.name}`}
            />
          ) : null}
        </div>
      ) : null}
    </>
  );

  if (interactive && onSelect && !locked) {
    return (
      <button
        type="button"
        className={`matchup-player matchup-player-${side} matchup-player-btn${
          selected ? " is-selected" : ""
        }`}
        onClick={onSelect}
        aria-pressed={selected}
      >
        {content}
      </button>
    );
  }

  return <div className={`matchup-player matchup-player-${side}`}>{content}</div>;
}

function BenchRow({
  player,
  evidence,
  selected,
  onSelect,
}: {
  player: LineupStarter;
  evidence?: InjuryEvidence;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <li className={`lineup-row matchup-bench-row${selected ? " is-selected" : ""}`}>
      <span className="lineup-slot" title="Bench">
        BN
      </span>
      <PlayerCell
        player={player}
        evidence={evidence}
        side="you"
        selected={selected}
        interactive
        onSelect={onSelect}
      />
      <ProjCell player={player} />
    </li>
  );
}

export function LineupScreen() {
  const { selectedLeagueId, selectedLeague, week, availableWeeks, rostersLoading } = useAppState();
  const [opponentMode, setOpponentMode] = useState<OpponentMode>("current");
  const lineup = useLineupRecommendation(selectedLeagueId, week, opponentMode);

  const [localStarters, setLocalStarters] = useState<LineupStarter[]>([]);
  const [localBench, setLocalBench] = useState<LineupStarter[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [adjusted, setAdjusted] = useState(false);
  const [snapshotKey, setSnapshotKey] = useState("");

  // Sync editable board from the server recommendation without a post-paint lag
  // (so Apply / bench actions are available on the first render with data).
  const dataKey = lineup.data
    ? [
        lineup.data.week,
        lineup.data.opponent_mode,
        lineup.data.meta.projection_run_id,
        lineup.data.starters.map((player) => player.player_id).join(","),
        lineup.data.bench.map((player) => player.player_id).join(","),
      ].join("|")
    : "";
  if (lineup.data && dataKey !== snapshotKey) {
    setSnapshotKey(dataKey);
    setLocalStarters(lineup.data.starters);
    setLocalBench(lineup.data.bench);
    setSelectedId(null);
    setAdjusted(false);
  } else if (!lineup.data && snapshotKey) {
    setSnapshotKey("");
    setLocalStarters([]);
    setLocalBench([]);
    setSelectedId(null);
    setAdjusted(false);
  }

  const modeLabel =
    OPPONENT_MODES.find((option) => option.value === opponentMode)?.label ?? opponentMode;

  const evidencePlayerIds = useMemo(() => {
    return (lineup.data?.swaps ?? []).flatMap((swap) => [
      swap.in_player_id,
      swap.out_player_id,
    ]);
  }, [lineup.data]);
  const evidence = useInjuryEvidence(evidencePlayerIds);

  const seats = selectedLeague?.roster_positions ?? [];
  const orderedYou = useMemo(
    () => orderByRosterSlots(localStarters, seats),
    [localStarters, seats],
  );
  const orderedOpp = useMemo(
    () => orderByRosterSlots(lineup.data?.opponent_starters ?? [], seats),
    [lineup.data?.opponent_starters, seats],
  );
  const boardRows = useMemo(() => {
    const len = Math.max(orderedYou.length, orderedOpp.length, seats.length);
    const rows: Array<{
      slot: string;
      you: LineupStarter | null;
      opp: LineupStarter | null;
    }> = [];
    for (let i = 0; i < len; i += 1) {
      const you = orderedYou[i] ?? null;
      const opp = orderedOpp[i] ?? null;
      const slot = you?.slot ?? opp?.slot ?? seats[i] ?? "—";
      rows.push({ slot, you, opp });
    }
    return rows;
  }, [orderedYou, orderedOpp, seats]);

  const youProj = sumMeans(localStarters);
  const oppProj = lineup.data?.opponent_expected_points ?? sumMeans(orderedOpp);

  function findZone(playerId: string): "starter" | "bench" | null {
    if (localStarters.some((player) => player.player_id === playerId)) return "starter";
    if (localBench.some((player) => player.player_id === playerId)) return "bench";
    return null;
  }

  function swapPlayers(aId: string, bId: string) {
    const aZone = findZone(aId);
    const bZone = findZone(bId);
    if (!aZone || !bZone) return;

    const aStarter = localStarters.find((player) => player.player_id === aId);
    const bStarter = localStarters.find((player) => player.player_id === bId);
    const aBench = localBench.find((player) => player.player_id === aId);
    const bBench = localBench.find((player) => player.player_id === bId);

    if (aZone === "starter" && bZone === "starter" && aStarter && bStarter) {
      setLocalStarters((prev) =>
        prev.map((player) => {
          if (player.player_id === aId) return { ...bStarter, slot: aStarter.slot };
          if (player.player_id === bId) return { ...aStarter, slot: bStarter.slot };
          return player;
        }),
      );
    } else if (aZone === "bench" && bZone === "bench") {
      // No-op beyond selection clear — bench order is projection-sorted.
    } else {
      // Starter ↔ bench exchange: bench inherits the starter slot.
      const starter = aZone === "starter" ? aStarter : bStarter;
      const bench = aZone === "bench" ? aBench : bBench;
      if (!starter || !bench) return;
      if (starter.locked || bench.locked) return;
      setLocalStarters((prev) =>
        prev.map((player) =>
          player.player_id === starter.player_id
            ? { ...bench, slot: starter.slot ?? starter.position }
            : player,
        ),
      );
      setLocalBench((prev) =>
        prev.map((player) =>
          player.player_id === bench.player_id ? { ...starter, slot: "BN" } : player,
        ),
      );
    }
    setAdjusted(true);
    setSelectedId(null);
  }

  function onSelectPlayer(playerId: string) {
    const player =
      localStarters.find((row) => row.player_id === playerId) ??
      localBench.find((row) => row.player_id === playerId);
    if (player?.locked) return;
    if (!selectedId) {
      setSelectedId(playerId);
      return;
    }
    if (selectedId === playerId) {
      setSelectedId(null);
      return;
    }
    swapPlayers(selectedId, playerId);
  }

  function applyRecommendedSwap(outId: string, inId: string) {
    const outPlayer = localStarters.find((player) => player.player_id === outId);
    const inPlayer = localBench.find((player) => player.player_id === inId);
    if (!outPlayer || !inPlayer || outPlayer.locked || inPlayer.locked) return;
    setLocalStarters((prev) =>
      prev.map((player) =>
        player.player_id === outId
          ? { ...inPlayer, slot: outPlayer.slot ?? outPlayer.position }
          : player,
      ),
    );
    setLocalBench((prev) =>
      prev.map((player) =>
        player.player_id === inId ? { ...outPlayer, slot: "BN" } : player,
      ),
    );
    setAdjusted(true);
    setSelectedId(null);
  }

  function resetLineup() {
    if (!lineup.data) return;
    setLocalStarters(lineup.data.starters);
    setLocalBench(lineup.data.bench);
    setSelectedId(null);
    setAdjusted(false);
  }

  const missing: string[] = [];
  if (lineup.data) {
    if (lineup.data.points.p10 == null || lineup.data.points.p90 == null) {
      missing.push("projected points p10/p90");
    }
    if (!Object.keys(lineup.data.matchup_probabilities).length) {
      missing.push("matchup probability breakdown");
    }
  }

  const noLeague = !selectedLeagueId;
  const noWeek = !rostersLoading && week == null;

  return (
    <div className="screen lineup-screen matchup-screen">
      <Panel
        title="Matchup"
        actions={
          <FreshnessBadge
            dataAsOf={lineup.data?.meta.data_as_of}
            cachedAt={lineup.cachedAt}
            fromCache={lineup.fromCache}
            offline={lineup.offline}
            runId={lineup.data?.meta.projection_run_id}
          />
        }
      >
        <LineupSourceTabs boardSource={lineup.data?.board_source} />

        <div className="lineup-toolbar">
          <OpponentModeToggle
            value={opponentMode}
            onChange={setOpponentMode}
            disabled={noLeague || week == null}
            compact
          />
          <p className="lineup-week-label muted" data-testid="active-opponent-mode">
            Week {lineup.data?.week ?? week ?? "—"} · {modeLabel}
          </p>
        </div>

        {noLeague ? (
          <p className="state-notice state-empty">
            <span className="state-glyph" aria-hidden="true">
              ∅
            </span>
            <span>
              <strong className="state-title">No league selected.</strong> Pick a league in the
              header to load a matchup.
            </span>
          </p>
        ) : noWeek ? (
          <p className="state-notice state-empty">
            <span className="state-glyph" aria-hidden="true">
              ∅
            </span>
            <span>
              <strong className="state-title">No weeks synced.</strong> This league has no roster
              snapshots yet, so there is no week to project. Run a sync from Operations.
            </span>
          </p>
        ) : (
          <AsyncStateBanner
            label="Matchup recommendation"
            loading={lineup.loading}
            offline={lineup.offline}
            error={lineup.error}
            fromCache={lineup.fromCache}
            cachedAt={lineup.cachedAt}
            dataAsOf={lineup.data?.meta.data_as_of}
            hasData={Boolean(lineup.data)}
            isEmpty={Boolean(lineup.data && lineup.data.starters.length === 0)}
            missing={missing}
            emptyMessage={`No matchup published for week ${week ?? "?"} of ${
              selectedLeague?.name ?? "this league"
            }. Available weeks: ${availableWeeks.join(", ") || "none"}.`}
            onRetry={() => void lineup.refresh()}
          />
        )}

        {lineup.data ? (
          <>
            <div className="lineup-summary matchup-summary" aria-label="Matchup summary">
              <div className="lineup-summary-stat">
                <span className="lineup-summary-label">You</span>
                <span className="lineup-summary-value">
                  <MaybeNumber value={youProj} digits={1} />
                </span>
                {adjusted ? <span className="lineup-summary-sub">adjusted</span> : null}
              </div>
              <div className="lineup-summary-stat">
                <span className="lineup-summary-label">Win%</span>
                <span className="lineup-summary-value">
                  <MaybeNumber value={lineup.data.win_probability} digits={0} percent />
                </span>
                {adjusted ? (
                  <span className="lineup-summary-sub">from recommended</span>
                ) : null}
              </div>
              <div className="lineup-summary-stat">
                <span className="lineup-summary-label">Opp</span>
                <span className="lineup-summary-value">
                  <MaybeNumber value={oppProj} digits={1} />
                </span>
              </div>
            </div>

            <div className="matchup-board-head">
              <div className="matchup-side-label">You</div>
              <div className="matchup-side-label matchup-side-label-center">Slot</div>
              <div className="matchup-side-label matchup-side-label-opp">Opponent</div>
            </div>

            <ul className="matchup-board" aria-label="Starter matchup board">
              {boardRows.map((row, index) => (
                <li key={`${row.slot}-${index}`} className="matchup-board-row">
                  <div className="matchup-board-you">
                    <PlayerCell
                      player={row.you}
                      evidence={
                        row.you ? evidence.byPlayerId[row.you.player_id] : undefined
                      }
                      side="you"
                      selected={row.you?.player_id === selectedId}
                      interactive
                      onSelect={
                        row.you ? () => onSelectPlayer(row.you!.player_id) : undefined
                      }
                    />
                    <ProjCell player={row.you} />
                  </div>
                  <span className="matchup-board-slot">{row.slot}</span>
                  <div className="matchup-board-opp">
                    <ProjCell player={row.opp} />
                    <PlayerCell player={row.opp} side="opp" />
                  </div>
                </li>
              ))}
            </ul>

            <div className="lineup-section-head">
              <h3 className="section-title">Bench</h3>
              <span className="lineup-col-hint muted">
                {selectedId ? "Tap a starter to swap" : "Tap to select · PROJ"}
              </span>
            </div>
            {adjusted ? (
              <p className="matchup-adjust-bar">
                <span className="muted">Lineup adjusted locally — not pushed to Sleeper.</span>
                <button type="button" className="btn btn-ghost btn-compact" onClick={resetLineup}>
                  Reset
                </button>
              </p>
            ) : null}
            {localBench.length ? (
              <ul className="lineup-list matchup-bench-list">
                {localBench.map((player) => (
                  <BenchRow
                    key={player.player_id}
                    player={player}
                    evidence={evidence.byPlayerId[player.player_id]}
                    selected={selectedId === player.player_id}
                    onSelect={() => onSelectPlayer(player.player_id)}
                  />
                ))}
              </ul>
            ) : (
              <p className="empty-state">No bench players with projections.</p>
            )}

            <h3 className="section-title">Recommended swaps</h3>
            {lineup.data.swaps.length ? (
              <ul className="swap-list lineup-swaps">
                {lineup.data.swaps.map((swap) => {
                  const inEvidence = evidence.byPlayerId[swap.in_player_id];
                  const outEvidence = evidence.byPlayerId[swap.out_player_id];
                  const actionable = [inEvidence, outEvidence].filter(
                    isActionableInjuryEvidence,
                  );
                  const citations = actionable.flatMap((item) => item!.sources);
                  const canApply =
                    localStarters.some((player) => player.player_id === swap.out_player_id) &&
                    localBench.some((player) => player.player_id === swap.in_player_id);
                  return (
                    <li
                      key={`${swap.out_player_id}-${swap.in_player_id}`}
                      className="swap-item"
                    >
                      <p className="swap-rationale">{swap.reason}</p>
                      <p className="muted">
                        Win probability{" "}
                        <MaybeNumber value={swap.win_probability_delta} digits={1} percent />
                      </p>
                      {canApply ? (
                        <button
                          type="button"
                          className="btn btn-ghost btn-compact matchup-apply-swap"
                          onClick={() =>
                            applyRecommendedSwap(swap.out_player_id, swap.in_player_id)
                          }
                        >
                          Apply on board
                        </button>
                      ) : null}
                      {actionable.length ? (
                        <div className="evidence">
                          {actionable.map((item) => (
                            <p key={item!.player_id} className="muted">
                              <span
                                className={`injury-pill injury-${item!.status.toLowerCase()}`}
                              >
                                {item!.status}
                              </span>{" "}
                              {item!.summary}
                            </p>
                          ))}
                          {citations.length ? (
                            <CitationList
                              citations={citations}
                              label={`Injury sources for ${swap.in_player_id} and ${swap.out_player_id}`}
                            />
                          ) : null}
                        </div>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="empty-state">
                No swaps recommended — the optimizer agrees with your current starters.
              </p>
            )}
            {evidence.failed.length && lineup.data.swaps.length ? (
              <p className="state-notice state-partial">
                <span className="state-glyph" aria-hidden="true">
                  ◑
                </span>
                <span>
                  <strong className="state-title">Partial evidence.</strong> Injury sources could
                  not be loaded for: {evidence.failed.join(", ")}.
                </span>
              </p>
            ) : null}
          </>
        ) : null}
      </Panel>
    </div>
  );
}
