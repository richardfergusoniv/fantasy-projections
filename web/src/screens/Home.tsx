import { useEffect, useState } from "react";
import { AppBuildStamp } from "../components/AppBuildStamp";
import { AsyncStateBanner } from "../components/AsyncState";
import { LineupSourceTabs } from "../components/LineupSourceTabs";
import { Panel } from "../components/Panel";
import { MaybeNumber } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { useOperationsStatus } from "../hooks/useOperationsStatus";
import {
  useLineupRecommendation,
  useWaiverRecommendation,
} from "../hooks/useReadonlyRecommendation";
import { forceRefreshAppShell } from "../pwa/registerUpdates";

interface UrgentItem {
  id: string;
  text: string;
  severity: "action" | "warning";
}

function projectedScore(points: {
  mean: number | null;
  p50: number | null;
}): number | null {
  if (points.mean != null) return points.mean;
  if (points.p50 != null) return points.p50;
  return null;
}

export function HomeScreen() {
  const {
    selectedLeague,
    selectedLeagueId,
    leaguesLoading,
    leaguesError,
    leagues,
    week,
    availableWeeks,
    weekIsUserChosen,
    boardSource,
    setBoardSource,
  } = useAppState();
  const lineup = useLineupRecommendation(selectedLeagueId, week, "current", boardSource);
  const waivers = useWaiverRecommendation(selectedLeagueId, week);
  // Ops status probes artifact storage (~seconds on prod). Defer so Matchup /
  // waivers own the first-paint network slot; Urgent still picks up gates later.
  const [opsEnabled, setOpsEnabled] = useState(false);
  useEffect(() => {
    let cancelled = false;
    const enable = () => {
      if (!cancelled) setOpsEnabled(true);
    };
    const idleId =
      typeof window.requestIdleCallback === "function"
        ? window.requestIdleCallback(enable, { timeout: 1500 })
        : null;
    const timerId = idleId == null ? window.setTimeout(enable, 0) : null;
    return () => {
      cancelled = true;
      if (idleId != null && typeof window.cancelIdleCallback === "function") {
        window.cancelIdleCallback(idleId);
      }
      if (timerId != null) window.clearTimeout(timerId);
    };
  }, []);
  const operations = useOperationsStatus({ enabled: opsEnabled });
  const [refreshingShell, setRefreshingShell] = useState(false);

  // Every entry below is derived from a response the app actually received.
  // Nothing is a static reminder string.
  const urgent: UrgentItem[] = [];
  if (lineup.data?.swaps.length) {
    urgent.push({
      id: "swaps",
      severity: "action",
      text: `${lineup.data.swaps.length} start/sit swap${
        lineup.data.swaps.length === 1 ? "" : "s"
      } would raise your week ${lineup.data.week} win probability.`,
    });
  }
  if (waivers.data?.adds.length) {
    const top = waivers.data.adds[0];
    urgent.push({
      id: "waivers",
      severity: "action",
      text: `${waivers.data.adds.length} waiver target${
        waivers.data.adds.length === 1 ? "" : "s"
      } published — top bid ${top.name} at $${top.faab_min}–$${top.faab_max} FAAB.`,
    });
  }
  if (operations.data?.failed_gates?.length) {
    urgent.push({
      id: "gates",
      severity: "warning",
      text: `${operations.data.failed_gates.length} release gate failure${
        operations.data.failed_gates.length === 1 ? "" : "s"
      } are blocking promotion: ${operations.data.failed_gates.slice(0, 3).join(", ")}.`,
    });
  }
  if (operations.data && !operations.data.last_sync_at) {
    urgent.push({
      id: "never-synced",
      severity: "warning",
      text: "No source snapshot has ever been recorded. Run a sync before trusting these numbers.",
    });
  }
  if (lineup.offline || waivers.offline) {
    urgent.push({
      id: "offline",
      severity: "warning",
      text: "You are offline. Everything below is a saved copy and will not reflect late news.",
    });
  }

  const score = lineup.data ? projectedScore(lineup.data.points) : null;
  const oppScore = lineup.data?.opponent_expected_points ?? null;

  return (
    <div className="screen home-screen">
      <div className="home-build-row">
        <AppBuildStamp />
        <button
          className="btn btn-ghost btn-compact"
          type="button"
          disabled={refreshingShell}
          onClick={() => {
            setRefreshingShell(true);
            void forceRefreshAppShell().finally(() => setRefreshingShell(false));
          }}
        >
          {refreshingShell ? "Refreshing…" : "Refresh"}
        </button>
      </div>
      <Panel title="League">
        <AsyncStateBanner
          label="League list"
          loading={leaguesLoading}
          offline={false}
          error={leaguesError}
          fromCache={false}
          hasData={leagues.length > 0}
          isEmpty={leagues.length === 0}
          emptyMessage="No leagues imported yet. Connect Sleeper and run a sync to import them."
        />
        {selectedLeague ? (
          <div className="home-summary">
            <p>
              <strong>{selectedLeague.name}</strong> · season {selectedLeague.season}
            </p>
            <p className="muted">
              {selectedLeague.is_dynasty ? "Dynasty" : "Redraft"} · {selectedLeague.scoring_type} ·{" "}
              {selectedLeague.roster_positions.length} roster slots
            </p>
            <p className="muted">
              Week {week ?? "not determined"}{" "}
              {weekIsUserChosen ? "(your selection)" : "(latest week with synced rosters)"} · synced
              weeks: {availableWeeks.join(", ") || "none"}
            </p>
          </div>
        ) : null}
      </Panel>

      <Panel title="Projection source">
        <p className="muted">
          Choose how Matchup and this snapshot are scored. Same toggle as the Matchup board —
          Vegas Props or League Value (sealed).
        </p>
        <LineupSourceTabs
          value={boardSource}
          onChange={setBoardSource}
          fallbackSource={lineup.data?.board_source}
          effectiveSource={lineup.data?.board_source}
          fallbackReason={lineup.data?.fallback_reason}
          hint={false}
        />
      </Panel>

      <Panel title="Matchup snapshot">
        <AsyncStateBanner
          label="Matchup snapshot"
          loading={lineup.loading}
          offline={lineup.offline}
          error={lineup.error}
          fromCache={lineup.fromCache}
          cachedAt={lineup.cachedAt}
          dataAsOf={lineup.data?.meta.data_as_of}
          hasData={Boolean(lineup.data)}
          isEmpty={false}
          emptyMessage="No lineup recommendation published for the selected league and week."
          onRetry={() => void lineup.refresh()}
        />
        {lineup.data ? (
          <div className="matchup-card matchup-chip" data-testid="matchup-snapshot-chip">
            <div className="matchup-chip-score">
              <span className="matchup-chip-label">Proj</span>
              <strong className="matchup-chip-you">
                <MaybeNumber value={score} digits={1} />
              </strong>
              {oppScore != null ? (
                <>
                  <span className="matchup-chip-sep" aria-hidden="true">
                    –
                  </span>
                  <span className="matchup-chip-opp">
                    <MaybeNumber value={oppScore} digits={1} />
                  </span>
                </>
              ) : null}
            </div>
            {lineup.data.win_probability != null ? (
              <p className="win-prob matchup-chip-win">
                Win{" "}
                <strong>
                  <MaybeNumber value={lineup.data.win_probability} digits={0} percent />
                </strong>
              </p>
            ) : null}
          </div>
        ) : null}
      </Panel>

      <Panel title="Urgent decisions">
        <AsyncStateBanner
          label="Urgent decisions"
          loading={
            lineup.loading || waivers.loading || (opsEnabled && operations.loading)
          }
          offline={false}
          error={operations.error}
          fromCache={false}
          hasData={urgent.length > 0}
          isEmpty={urgent.length === 0}
          emptyMessage="Nothing urgent from current data: no recommended swaps, no waiver targets, and no failed release gates."
          onRetry={() => void operations.refresh()}
        />
        {urgent.length ? (
          <ul className="urgent-list">
            {urgent.map((item) => (
              <li key={item.id} className={`urgent-${item.severity}`}>
                {item.text}
              </li>
            ))}
          </ul>
        ) : null}
      </Panel>
    </div>
  );
}
