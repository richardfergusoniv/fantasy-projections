import type { PointsRange } from "../api/types";

const NOT_AVAILABLE = "not available";

function fmt(value: number | null, digits = 1): string {
  return value == null ? NOT_AVAILABLE : value.toFixed(digits);
}

/**
 * Show a projection as the published p10–p90 interval, never as a bare mean.
 *
 * A single number reads as certainty the simulation does not have. When the API
 * omits the quantiles the component says so instead of falling back to the mean.
 */
export function UncertaintyRange({
  label,
  range,
  unit = "pts",
}: {
  label: string;
  range: PointsRange;
  unit?: string;
}) {
  const hasInterval = range.p10 != null && range.p90 != null;
  return (
    <div className="uncertainty">
      <span className="uncertainty-label">{label}</span>
      {hasInterval ? (
        <>
          <span className="uncertainty-range">
            {fmt(range.p10)}–{fmt(range.p90)} {unit}
          </span>
          <span className="uncertainty-detail">
            80% interval · median {fmt(range.p50)} · mean {fmt(range.mean)}
          </span>
        </>
      ) : (
        <>
          <span className="uncertainty-range uncertainty-missing">
            range {NOT_AVAILABLE}
          </span>
          <span className="uncertainty-detail">
            The API published mean {fmt(range.mean)} {unit} without p10/p90 quantiles, so no
            interval is shown.
          </span>
        </>
      )}
    </div>
  );
}

/** Inline p10–p90 for use inside a list row. */
export function InlineRange({ range, unit = "pts" }: { range: PointsRange; unit?: string }) {
  if (range.p10 == null || range.p90 == null) {
    return <span className="inline-range inline-range-missing">range {NOT_AVAILABLE}</span>;
  }
  return (
    <span className="inline-range">
      {fmt(range.p10)}–{fmt(range.p90)} {unit}
    </span>
  );
}

/** Render a number, or an explicit "not available" when the API omitted it. */
export function MaybeNumber({
  value,
  digits = 2,
  suffix = "",
  percent = false,
}: {
  value: number | null;
  digits?: number;
  suffix?: string;
  percent?: boolean;
}) {
  if (value == null) {
    return <span className="value-missing">{NOT_AVAILABLE}</span>;
  }
  const shown = percent ? `${(value * 100).toFixed(digits)}%` : value.toFixed(digits);
  return (
    <span className="value-present">
      {shown}
      {suffix}
    </span>
  );
}
