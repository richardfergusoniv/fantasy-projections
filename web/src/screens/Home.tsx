import { AsyncStateBanner } from "../components/AsyncState";
import { FreshnessBadge } from "../components/FreshnessBadge";
import { Panel } from "../components/Panel";
import { MaybeNumber, UncertaintyRange } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { useOperationsStatus } from "../hooks/useOperationsStatus";
import {
  useLineupRecommendation,
  useWaiverRecommendation,
} from "../hooks/useReadonlyRecommendation";

interface UrgentItem {
  id: string;
  text: string;
  severity: "action" | "warning";
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
  } = useAppState();
  const lineup = useLineupRecommendation(selectedLeagueId, week);
  const waivers = useWaiverRecommendation(selectedLeagueId, week);
  const operations = useOperationsStatus();

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

  return (
    <div className="screen home-screen">
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

      <Panel
        title="Matchup snapshot"
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
          <div className="matchup-card">
            <p className="win-prob">
              Win probability:{" "}
              <strong>
                <MaybeNumber value={lineup.data.win_probability} digits={1} percent />
              </strong>
            </p>
            <UncertaintyRange label="Projected lineup points" range={lineup.data.points} />
            <p className="muted">
              {lineup.data.swaps.length} suggested swap
              {lineup.data.swaps.length === 1 ? "" : "s"}
            </p>
            <ul className="swap-list">
              {lineup.data.swaps.slice(0, 3).map((swap) => (
                <li key={`${swap.out_player_id}-${swap.in_player_id}`}>{swap.reason}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </Panel>

      <Panel title="Urgent decisions">
        <AsyncStateBanner
          label="Urgent decisions"
          loading={lineup.loading || waivers.loading || operations.loading}
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
