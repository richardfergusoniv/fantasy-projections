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

function StarterRow({
  player,
  evidence,
}: {
  player: LineupStarter;
  evidence?: InjuryEvidence;
}) {
  const pts = starterPoints(player);
  const showEvidence = isActionableInjuryEvidence(evidence);
  const teamPos = [player.team, player.position].filter(Boolean).join(" · ");

  return (
    <li className="lineup-row">
      <span className="lineup-slot" title={player.slot ?? player.position}>
        {player.slot ?? player.position}
      </span>
      <div className="lineup-player">
        <span className="lineup-name">{player.name}</span>
        <span className="lineup-meta">
          {teamPos ? <span>{teamPos}</span> : null}
          <span className="lineup-opp">{formatOpp(player.opponent)}</span>
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
      </div>
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
    </li>
  );
}

export function LineupScreen() {
  const { selectedLeagueId, selectedLeague, week, availableWeeks, rostersLoading } = useAppState();
  const [opponentMode, setOpponentMode] = useState<OpponentMode>("current");
  const lineup = useLineupRecommendation(selectedLeagueId, week, opponentMode);

  const modeLabel =
    OPPONENT_MODES.find((option) => option.value === opponentMode)?.label ?? opponentMode;

  // Evidence only for players a swap actually moves — starter-wide lookups just
  // produce "Status unknown / No evidence" noise on every row.
  const evidencePlayerIds = useMemo(() => {
    return (lineup.data?.swaps ?? []).flatMap((swap) => [
      swap.in_player_id,
      swap.out_player_id,
    ]);
  }, [lineup.data]);
  const evidence = useInjuryEvidence(evidencePlayerIds);

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
  const meanPts = lineup.data?.points.mean ?? null;
  const oppPts = lineup.data?.opponent_expected_points ?? null;

  return (
    <div className="screen lineup-screen">
      <Panel
        title="Lineup"
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
              header to load a lineup.
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
            label="Lineup recommendation"
            loading={lineup.loading}
            offline={lineup.offline}
            error={lineup.error}
            fromCache={lineup.fromCache}
            cachedAt={lineup.cachedAt}
            dataAsOf={lineup.data?.meta.data_as_of}
            hasData={Boolean(lineup.data)}
            isEmpty={Boolean(lineup.data && lineup.data.starters.length === 0)}
            missing={missing}
            emptyMessage={`No lineup published for week ${week ?? "?"} of ${
              selectedLeague?.name ?? "this league"
            }. Available weeks: ${availableWeeks.join(", ") || "none"}.`}
            onRetry={() => void lineup.refresh()}
          />
        )}

        {lineup.data ? (
          <>
            <div className="lineup-summary" aria-label="Matchup summary">
              <div className="lineup-summary-stat">
                <span className="lineup-summary-label">Win%</span>
                <span className="lineup-summary-value">
                  <MaybeNumber value={lineup.data.win_probability} digits={0} percent />
                </span>
              </div>
              <div className="lineup-summary-stat">
                <span className="lineup-summary-label">Proj</span>
                <span className="lineup-summary-value">
                  <MaybeNumber value={meanPts} digits={1} />
                </span>
                {lineup.data.points.p10 != null && lineup.data.points.p90 != null ? (
                  <span className="lineup-summary-sub">
                    {lineup.data.points.p10.toFixed(0)}–{lineup.data.points.p90.toFixed(0)}
                  </span>
                ) : null}
              </div>
              <div className="lineup-summary-stat">
                <span className="lineup-summary-label">Opp</span>
                <span className="lineup-summary-value">
                  <MaybeNumber value={oppPts} digits={1} />
                </span>
              </div>
            </div>

            <div className="lineup-section-head">
              <h3 className="section-title">Starters</h3>
              <span className="lineup-col-hint muted">PROJ</span>
            </div>
            {lineup.data.starters.length ? (
              <ul className="lineup-list">
                {lineup.data.starters.map((player) => (
                  <StarterRow
                    key={`${player.slot ?? player.position}-${player.player_id}`}
                    player={player}
                    evidence={evidence.byPlayerId[player.player_id]}
                  />
                ))}
              </ul>
            ) : (
              <p className="empty-state">
                The optimizer returned no starters for this roster snapshot.
              </p>
            )}

            <h3 className="section-title">Recommended swaps</h3>
            {lineup.data.swaps.length ? (
              <ul className="swap-list lineup-swaps">
                {lineup.data.swaps.map((swap) => {
                  const inEvidence = evidence.byPlayerId[swap.in_player_id];
                  const outEvidence = evidence.byPlayerId[swap.out_player_id];
                  const actionable = [inEvidence, outEvidence].filter(isActionableInjuryEvidence);
                  const citations = actionable.flatMap((item) => item!.sources);
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
