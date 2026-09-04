import { useCallback, useState } from "react";
import { AsyncStateBanner } from "../components/AsyncState";
import { AppBuildStamp } from "../components/AppBuildStamp";
import { FreshnessBadge } from "../components/FreshnessBadge";
import { Panel } from "../components/Panel";
import { MaybeNumber } from "../components/UncertaintyRange";
import { useAppState } from "../hooks/useAppState";
import { useOperationsStatus } from "../hooks/useOperationsStatus";
import { api } from "../api/client";
import { forceRefreshAppShell } from "../pwa/registerUpdates";

const ROLLBACK_MODES = ["weekly", "ros", "dynasty"] as const;
type RollbackMode = (typeof ROLLBACK_MODES)[number];

export function OperationsScreen() {
  const { data, loading, error, refresh } = useOperationsStatus();
  const { season, week, availableWeeks, selectedLeague } = useAppState();
  const [jobMessage, setJobMessage] = useState<string | null>(null);
  const [jobLoading, setJobLoading] = useState(false);
  const [rollbackMode, setRollbackMode] = useState<RollbackMode>("weekly");
  const [refreshingShell, setRefreshingShell] = useState(false);

  const runJob = useCallback(
    async (label: string, action: () => Promise<{ job_id?: string; status?: string }>) => {
      setJobLoading(true);
      setJobMessage(null);
      try {
        const result = await action();
        const detail = result.job_id ? `Job ${result.job_id}` : result.status ?? "done";
        setJobMessage(`${label}: ${detail}`);
        await refresh();
      } catch (err) {
        setJobMessage(err instanceof Error ? err.message : `${label} failed`);
      } finally {
        setJobLoading(false);
      }
    },
    [refresh],
  );

  // ROS and dynasty pointers are season-scoped with no week; weekly needs both.
  const rollbackWeek = rollbackMode === "weekly" ? week : null;
  const canRollback = season != null && (rollbackMode !== "weekly" || rollbackWeek != null);

  return (
    <div className="screen">
      <Panel
        title="Operations"
        actions={
          <FreshnessBadge
            dataAsOf={data?.data_as_of}
            runId={data?.active_projection_run_id}
          />
        }
      >
        <AsyncStateBanner
          label="Operations status"
          loading={loading}
          offline={false}
          error={error}
          fromCache={false}
          dataAsOf={data?.data_as_of}
          hasData={Boolean(data)}
          isEmpty={false}
          emptyMessage="Operations status is unavailable."
          onRetry={() => void refresh()}
        />

        <AppBuildStamp />
        <button
          className="btn btn-ghost"
          type="button"
          disabled={refreshingShell}
          onClick={() => {
            setRefreshingShell(true);
            void forceRefreshAppShell().finally(() => setRefreshingShell(false));
          }}
        >
          {refreshingShell ? "Refreshing app…" : "Refresh installed app shell"}
        </button>

        {/*
          Production panel: sealed release + status overlay health.
          Weekly R&D NO-GO must not mark this panel unhealthy.
        */}
        <h3 className="section-title">Production (sealed release)</h3>
        <ul className="ops-list ops-modes">
          <li>
            <span>Projection source</span>
            <strong>{data?.modes?.projection_source ?? "sealed_release"}</strong>
          </li>
          <li>
            <span>Production health</span>
            <strong
              className={data?.production?.healthy ? undefined : "state-warning-text"}
            >
              {data?.production?.healthy
                ? "healthy"
                : data?.production?.degraded_capabilities?.length
                  ? `degraded (${data.production.degraded_capabilities.length})`
                  : "degraded"}
            </strong>
          </li>
          <li>
            <span>Sleeper data source</span>
            <strong
              className={data?.modes?.sleeper_source === "live" ? undefined : "state-warning-text"}
            >
              {data?.modes?.sleeper_source === "live"
                ? "live (read-only Sleeper API)"
                : data?.modes?.sleeper_source === "fixture"
                  ? "fixture — recorded payloads, not live league data"
                  : "unknown"}
            </strong>
          </li>
        </ul>
        {data?.production?.degraded_capabilities?.length ? (
          <ul className="urgent-list">
            {data.production.degraded_capabilities.map((capability) => {
              const detail = data.capabilities?.capabilities.find(
                (row) => row.capability === capability,
              );
              return (
                <li key={capability} className="urgent-warning">
                  <strong>{capability}</strong>
                  {detail?.detail ? ` — ${detail.detail}` : null}
                </li>
              );
            })}
          </ul>
        ) : null}

        <h3 className="section-title">Weekly modeling R&amp;D</h3>
        <ul className="ops-list ops-modes">
          <li>
            <span>Weekly-v2 artifacts</span>
            <strong
              className={data?.modes?.weekly_v2_state === "trained" ? undefined : "state-warning-text"}
            >
              {data?.modes?.weekly_v2_state === "trained"
                ? `trained (${data.modes.weekly_v2_model_version ?? "version unknown"})`
                : `${data?.modes?.weekly_v2_state ?? "unknown"} — R&D only, not production`}
            </strong>
          </li>
          <li>
            <span>R&amp;D auto-publish</span>
            <strong
              className={data?.weekly_rnd?.auto_publish_allowed ? undefined : "state-warning-text"}
            >
              {data?.weekly_rnd?.auto_publish_allowed
                ? "allowed"
                : "NO-GO — blocked until weekly gates pass"}
            </strong>
          </li>
          <li>
            <span>Weekly R&amp;D enabled</span>
            <strong>{data?.modes?.weekly_rnd_enabled ? "yes" : "no (default)"}</strong>
          </li>
        </ul>
        {data?.modes?.weekly_v2_reasons?.length ? (
          <ul className="urgent-list">
            {data.modes.weekly_v2_reasons.map((reason) => (
              <li key={reason} className="urgent-warning">
                {reason}
              </li>
            ))}
          </ul>
        ) : null}

        <ul className="ops-list">
          <li>
            <span>Last source snapshot</span>
            <strong>
              {data?.last_sync_at
                ? new Date(data.last_sync_at).toLocaleString()
                : "never — no snapshot recorded"}
            </strong>
          </li>
          <li>
            <span>Active release</span>
            <strong>{data?.active_projection_run_id ?? "not available"}</strong>
          </li>
          <li>
            <span>Failed gates</span>
            <strong>{data?.failed_gates?.length ?? 0}</strong>
          </li>
          <li>
            <span>Est. month cost</span>
            <strong>
              <MaybeNumber value={data?.estimated_month_cost_usd ?? null} digits={2} suffix=" USD" />
            </strong>
          </li>
          <li>
            <span>OpenAI configured</span>
            <strong>{data?.openai_configured ? "yes" : "no"}</strong>
          </li>
          {data?.latest_job ? (
            <li>
              <span>Latest job</span>
              <strong>
                {data.latest_job.name} · {data.latest_job.status}
              </strong>
            </li>
          ) : null}
          {data?.latest_promotion ? (
            <li>
              <span>Latest promotion</span>
              <strong>
                {data.latest_promotion.mode} ·{" "}
                {data.latest_promotion.promoted ? "promoted" : "rejected"}
              </strong>
            </li>
          ) : null}
        </ul>

        {data?.failed_gates?.length ? (
          <ul className="urgent-list">
            {data.failed_gates.map((gate) => (
              <li key={gate} className="urgent-warning">
                {gate}
              </li>
            ))}
          </ul>
        ) : null}

        <h3 className="section-title">Jobs</h3>
        <div className="button-row">
          <button
            className="btn btn-primary"
            type="button"
            disabled={jobLoading}
            onClick={() =>
              void runJob("Daily refresh", () => api.triggerSync(`ops-daily-${Date.now()}`))
            }
          >
            {jobLoading ? "Running…" : "Run daily refresh"}
          </button>
          <button
            className="btn btn-secondary"
            type="button"
            disabled={jobLoading}
            onClick={() =>
              void runJob("Full release", () =>
                api.runOperationsJob("full-release", `ops-full-${Date.now()}`),
              )
            }
          >
            Full release
          </button>
        </div>

        <h3 className="section-title">Projection rollback</h3>
        <div className="stack">
          <div className="field">
            <label htmlFor="rollback-mode">Pointer to roll back</label>
            <select
              id="rollback-mode"
              value={rollbackMode}
              onChange={(event) => setRollbackMode(event.target.value as RollbackMode)}
            >
              {ROLLBACK_MODES.map((mode) => (
                <option key={mode} value={mode}>
                  {mode}
                </option>
              ))}
            </select>
          </div>
          <p className="muted">
            Target: season <strong>{season ?? "unknown"}</strong>
            {rollbackMode === "weekly" ? (
              <>
                {" "}
                week <strong>{rollbackWeek ?? "unknown"}</strong>
              </>
            ) : (
              " (season-wide pointer, no week)"
            )}
            . Derived from {selectedLeague?.name ?? "the selected league"}
            {availableWeeks.length ? ` and its synced weeks (${availableWeeks.join(", ")})` : ""}.
            Change either in the header.
          </p>
          <button
            className="btn btn-secondary"
            type="button"
            disabled={jobLoading || !canRollback}
            onClick={() =>
              void runJob(`Rollback ${rollbackMode}`, () =>
                api.rollbackProjection({
                  mode: rollbackMode,
                  season: season as number,
                  week: rollbackWeek,
                }),
              )
            }
          >
            Roll back {rollbackMode}
          </button>
          {!canRollback ? (
            <p className="muted">
              A season is required before a rollback can be targeted. Select a league in the header.
            </p>
          ) : null}
        </div>

        <p className="status-message" role="status" aria-live="polite">
          {jobMessage ?? ""}
        </p>
        <p className="muted">
          Private operations view for job retries, promotion receipts, rollback rehearsal, and
          release health.
        </p>
      </Panel>
    </div>
  );
}
