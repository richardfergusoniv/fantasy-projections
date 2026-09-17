import { describeAsOf, pickAsOfStamp } from "./asOfVintage";

/**
 * Persistent data-vintage chrome for projection boards.
 *
 * Uses published `data_as_of` / `available_at` only. Missing stamps render as
 * "As-of unknown" instead of inventing a clock time.
 */
export function AsOfChrome({
  dataAsOf,
  availableAt,
  runId,
}: {
  dataAsOf?: string | null;
  availableAt?: string | null;
  runId?: string | null;
}) {
  const stamp = pickAsOfStamp(dataAsOf, availableAt);
  const described = describeAsOf(stamp);
  const formatted = described.iso ? new Date(described.iso).toLocaleString() : null;

  return (
    <p
      className={`as-of-chrome is-${described.kind}`}
      data-testid="as-of-chrome"
      role="status"
      title={runId ? `Projection run ${runId}` : undefined}
    >
      {described.iso && formatted ? (
        <>
          As of <time dateTime={described.iso}>{formatted}</time>
          {described.kind === "stale" ? " · stale" : ""}
        </>
      ) : (
        described.label
      )}
    </p>
  );
}
