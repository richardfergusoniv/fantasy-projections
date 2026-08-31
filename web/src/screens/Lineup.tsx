import { useMemo, useState } from "react";
import { AsyncStateBanner } from "../components/AsyncState";
import { CitationList } from "../components/CitationList";
import { FreshnessBadge } from "../components/FreshnessBadge";
import { OpponentModeToggle, OPPONENT_MODES } from "../components/OpponentModeToggle";
import { Panel } from "../components/Panel";
import { MaybeNumber, UncertaintyRange } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { useInjuryEvidence } from "../hooks/useInjuryEvidence";
import { useLineupRecommendation } from "../hooks/useReadonlyRecommendation";
import type { OpponentMode } from "../api/types";

export function LineupScreen() {
  const { selectedLeagueId, selectedLeague, week, availableWeeks, rostersLoading } = useAppState();
  const [opponentMode, setOpponentMode] = useState<OpponentMode>("current");
  const lineup = useLineupRecommendation(selectedLeagueId, week, opponentMode);

  const modeLabel =
    OPPONENT_MODES.find((option) => option.value === opponentMode)?.label ?? opponentMode;

  // Evidence for the players a swap moves *and* for the players being started.
  // Swap-only was too narrow: when the optimizer agrees with the submitted
  // lineup there are no swaps, so a questionable starter's cited report — the
  // thing most likely to change the owner's mind — was never shown at all.
  const evidencePlayerIds = useMemo(() => {
    const swapped = (lineup.data?.swaps ?? []).flatMap((swap) => [
      swap.in_player_id,
      swap.out_player_id,
    ]);
    const starters = (lineup.data?.starters ?? []).map((player) => player.player_id);
    return [...swapped, ...starters];
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

  return (
    <div className="screen">
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
        <OpponentModeToggle
          value={opponentMode}
          onChange={setOpponentMode}
          disabled={noLeague || week == null}
        />

        <p className="mode-active" data-testid="active-opponent-mode">
          Showing: <strong>{modeLabel}</strong>
          {lineup.data ? (
            <>
              {" "}
              (server confirmed <code>opponent_mode={lineup.data.opponent_mode}</code> for week{" "}
              {lineup.data.week})
            </>
          ) : null}
        </p>

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
            <div className="decision-summary">
              <p className="win-prob">
                Win probability under {modeLabel.toLowerCase()}:{" "}
                <strong>
                  <MaybeNumber value={lineup.data.win_probability} digits={1} percent />
                </strong>
              </p>
              <UncertaintyRange
                label="Projected lineup points"
                range={lineup.data.points}
              />
              {Object.keys(lineup.data.matchup_probabilities).length ? (
                <ul className="prob-breakdown">
                  {Object.entries(lineup.data.matchup_probabilities).map(([outcome, value]) => (
                    <li key={outcome}>
                      <span className="label">{outcome}</span>
                      <MaybeNumber value={value} digits={1} percent />
                    </li>
                  ))}
                </ul>
              ) : null}
              <p className="muted provenance">
                Release <code>{lineup.data.meta.projection_run_id}</code>
                {lineup.data.contract_hash ? (
                  <>
                    {" "}
                    · scoring contract <code>{lineup.data.contract_hash.slice(0, 12)}</code>
                  </>
                ) : null}
              </p>
            </div>

            <h3 className="section-title">Starters</h3>
            {lineup.data.starters.length ? (
              <ul className="starter-list">
                {lineup.data.starters.map((player) => {
                  const playerEvidence = evidence.byPlayerId[player.player_id];
                  return (
                    <li key={player.player_id} className="roster-slot">
                      <span className="label">{player.slot ?? player.position}</span>
                      <span>
                        {player.name}{" "}
                        <span className={`pos-badge ${player.position}`}>{player.position}</span>
                      </span>
                      {playerEvidence ? (
                        <div className="evidence">
                          <p className="muted">
                            Status <strong>{playerEvidence.status}</strong> —{" "}
                            {playerEvidence.summary}
                          </p>
                          <CitationList
                            citations={playerEvidence.sources}
                            label={`Injury sources for ${player.name}`}
                          />
                        </div>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="empty-state">
                The optimizer returned no starters for this roster snapshot.
              </p>
            )}

            <h3 className="section-title">Recommended swaps</h3>
            {lineup.data.swaps.length ? (
              <ul className="swap-list">
                {lineup.data.swaps.map((swap) => {
                  const inEvidence = evidence.byPlayerId[swap.in_player_id];
                  const outEvidence = evidence.byPlayerId[swap.out_player_id];
                  const citations = [
                    ...(inEvidence?.sources ?? []),
                    ...(outEvidence?.sources ?? []),
                  ];
                  return (
                    <li
                      key={`${swap.out_player_id}-${swap.in_player_id}`}
                      className="swap-item"
                    >
                      <p className="swap-rationale">{swap.reason}</p>
                      <p className="muted">
                        Win probability change:{" "}
                        <MaybeNumber value={swap.win_probability_delta} digits={1} percent />
                      </p>
                      {inEvidence || outEvidence ? (
                        <div className="evidence">
                          {[inEvidence, outEvidence].filter(Boolean).map((item) => (
                            <p key={item!.player_id} className="muted">
                              {item!.player_id}: status <strong>{item!.status}</strong> —{" "}
                              {item!.summary}
                            </p>
                          ))}
                          <CitationList
                            citations={citations}
                            label={`Injury sources for ${swap.in_player_id} and ${swap.out_player_id}`}
                            emptyMessage="No injury sources published for either player in this swap."
                          />
                        </div>
                      ) : evidence.loading ? (
                        <p className="muted">Loading injury evidence…</p>
                      ) : (
                        <p className="muted">
                          Injury evidence not available for these players.
                        </p>
                      )}
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="empty-state">
                No swaps recommended — the optimizer agrees with your current starters.
              </p>
            )}
            {evidence.failed.length ? (
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
