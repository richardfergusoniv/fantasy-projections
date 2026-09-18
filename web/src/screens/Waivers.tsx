import { useMemo } from "react";
import { AsyncStateBanner } from "../components/AsyncState";
import { AsOfChrome } from "../components/AsOfChrome";
import { CitationList } from "../components/CitationList";
import { FreshnessBadge } from "../components/FreshnessBadge";
import { isActionableInjuryEvidence } from "../components/injuryEvidence";
import { Panel } from "../components/Panel";
import { ProjectionRowSkeleton } from "../components/ProjectionRowSkeleton";
import { MaybeNumber } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { useInjuryEvidence } from "../hooks/useInjuryEvidence";
import { useWaiverRecommendation } from "../hooks/useReadonlyRecommendation";

export function WaiversScreen() {
  const { selectedLeagueId, selectedLeague, week, availableWeeks, rostersLoading } = useAppState();
  const waivers = useWaiverRecommendation(selectedLeagueId, week);

  const addIds = useMemo(
    () => (waivers.data?.adds ?? []).slice(0, 5).map((add) => add.player_id),
    [waivers.data],
  );
  const evidence = useInjuryEvidence(addIds);

  const missing: string[] = [];
  if (waivers.data?.adds.length) {
    if (waivers.data.adds.every((add) => add.confidence == null)) {
      missing.push("recommendation confidence");
    }
    if (waivers.data.adds.every((add) => add.start_probability == null)) {
      missing.push("start probability");
    }
  }

  const noLeague = !selectedLeagueId;
  const noWeek = !rostersLoading && week == null;

  return (
    <div className="screen">
      <Panel
        title="Waivers"
        actions={
          <FreshnessBadge
            dataAsOf={waivers.data?.meta.data_as_of}
            cachedAt={waivers.cachedAt}
            fromCache={waivers.fromCache}
            offline={waivers.offline}
            runId={waivers.data?.meta.projection_run_id}
          />
        }
      >
        {noLeague ? (
          <p className="state-notice state-empty">
            <span className="state-glyph" aria-hidden="true">
              ∅
            </span>
            <span>
              <strong className="state-title">No league selected.</strong> Pick a league in the
              header to load waiver targets.
            </span>
          </p>
        ) : noWeek ? (
          <p className="state-notice state-empty">
            <span className="state-glyph" aria-hidden="true">
              ∅
            </span>
            <span>
              <strong className="state-title">No weeks synced.</strong> This league has no roster
              snapshots, so free agents cannot be determined. Run a sync from Operations.
            </span>
          </p>
        ) : (
          <>
            <AsOfChrome
              dataAsOf={waivers.data?.meta.data_as_of}
              runId={waivers.data?.meta.projection_run_id}
              pending={!waivers.data && !waivers.error}
            />
            <AsyncStateBanner
              label="Waiver recommendations"
              loading={waivers.loading}
              offline={waivers.offline}
              error={waivers.error}
              fromCache={waivers.fromCache}
              cachedAt={waivers.cachedAt}
              dataAsOf={waivers.data?.meta.data_as_of}
              hasData={Boolean(waivers.data)}
              isEmpty={Boolean(waivers.data && waivers.data.adds.length === 0)}
              missing={missing}
              emptyMessage={`No waiver targets published for week ${week ?? "?"} of ${
                selectedLeague?.name ?? "this league"
              }. Available weeks: ${availableWeeks.join(", ") || "none"}.`}
              onRetry={() => void waivers.refresh()}
            />
          </>
        )}

        {waivers.loading && !waivers.data && !noLeague && !noWeek ? (
          <ProjectionRowSkeleton variant="waiver" rows={5} />
        ) : null}

        {waivers.data ? (
          <>
            <p className="muted provenance">
              Release <code>{waivers.data.meta.projection_run_id}</code> · week {waivers.data.week}
            </p>
            {waivers.data.adds.length ? (
              <ul className="waiver-list projection-table">
                {waivers.data.adds.map((add) => {
                  const playerEvidence = evidence.byPlayerId[add.player_id];
                  const injury = isActionableInjuryEvidence(playerEvidence)
                    ? playerEvidence
                    : null;
                  return (
                    <li
                      key={add.player_id}
                      className="waiver-item"
                      title={
                        [
                          add.confidence != null ? `Confidence ${Math.round(add.confidence * 100)}%` : null,
                          add.start_probability != null
                            ? `Start ${Math.round(add.start_probability * 100)}%`
                            : null,
                        ]
                          .filter(Boolean)
                          .join(" · ") || undefined
                      }
                    >
                      <strong className="waiver-name">{add.name}</strong>
                      <span className={`pos-badge ${add.position}`}>{add.position}</span>
                      <span className="waiver-util">
                        <MaybeNumber value={add.incremental_utility} digits={1} suffix=" u" />
                      </span>
                      <span className="faab-range">
                        FAAB ${add.faab_min}–${add.faab_max}
                      </span>
                      {add.rationale.length ? (
                        <ul className="rationale-list">
                          {add.rationale.map((line) => (
                            <li key={line}>{line}</li>
                          ))}
                        </ul>
                      ) : (
                        <p className="muted waiver-rationale-empty">
                          No rationale published for this target.
                        </p>
                      )}
                      {injury ? (
                        <div className="evidence waiver-injury">
                          <p className="muted">
                            <span
                              className={`injury-pill injury-${injury.status.toLowerCase()}`}
                            >
                              {injury.status}
                            </span>{" "}
                            {injury.summary}
                          </p>
                          <CitationList
                            citations={injury.sources}
                            label={`Injury sources for ${add.name}`}
                            emptyMessage="No injury sources published for this player."
                          />
                        </div>
                      ) : null}
                    </li>
                  );
                })}
              </ul>
            ) : null}
          </>
        ) : null}
      </Panel>
    </div>
  );
}
