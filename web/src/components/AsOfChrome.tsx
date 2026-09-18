import { describeAsOf, pickAsOfStamp } from "./asOfVintage";

/**
 * Persistent data-vintage chrome for projection boards.
 *
 * Uses published `data_as_of` / `available_at` only. Missing stamps render as
 * "As-of unknown" instead of inventing a clock time. While a request is still
 * in flight and no stamp has arrived, show a reserved-height pending strip
 * rather than flashing unknown.
 */
export function AsOfChrome({
  dataAsOf,
  availableAt,
  runId,
  pending = false,
}: {
  dataAsOf?: string | null;
  availableAt?: string | null;
  runId?: string | null;
  pending?: boolean;
}) {
  const stamp = pickAsOfStamp(dataAsOf, availableAt);
  if (!stamp && pending) {
    return (
      <p
        className="as-of-chrome is-pending"
        data-testid="as-of-chrome"
        role="status"
        aria-busy="true"
      >
        <span className="skel skel-as-of" />
      </p>
    );
  }

  const described = describeAsOf(stamp);

  return (
    <p
      className={`as-of-chrome is-${described.kind}`}
      data-testid="as-of-chrome"
      role="status"
      title={runId ? `Projection run ${runId}` : undefined}
    >
      {described.iso && described.display ? (
        <>
          As of <time dateTime={described.iso}>{described.display}</time>
          {described.kind === "stale" ? " · stale" : ""}
        </>
      ) : (
        described.label
      )}
    </p>
  );
}
