import { STALE_AFTER_MS } from "./AsyncState";

/**
 * Source-freshness indicator.
 *
 * Offline, saved-copy, and stale-release are three different conditions and get
 * three different badges; a single "stale" pill would tell the user nothing
 * about whether the problem is their network or the projection pipeline.
 */
export function FreshnessBadge({
  dataAsOf,
  cachedAt,
  fromCache,
  offline,
  runId,
}: {
  dataAsOf?: string;
  cachedAt?: string | null;
  fromCache?: boolean;
  offline?: boolean;
  runId?: string | null;
}) {
  const label = dataAsOf ?? cachedAt;
  if (!label) {
    return (
      <span className="freshness-badge unknown" title="No source timestamp available">
        Freshness unknown
      </span>
    );
  }

  const asOf = new Date(label);
  const age = Date.now() - asOf.getTime();
  const formatted = Number.isNaN(asOf.getTime()) ? label : asOf.toLocaleString();

  let variant = "fresh";
  let prefix = "Source as of";
  if (offline) {
    variant = "offline";
    prefix = "Offline · saved";
  } else if (fromCache) {
    variant = "cached";
    prefix = "Saved copy from";
  } else if (Number.isFinite(age) && age > STALE_AFTER_MS) {
    variant = "stale";
    prefix = "Stale release from";
  }

  return (
    <span
      className={`freshness-badge ${variant}`}
      title={runId ? `Projection release ${runId}` : undefined}
    >
      {prefix} {formatted}
    </span>
  );
}
