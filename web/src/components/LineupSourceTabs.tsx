import type { LineupBoardSource } from "../api/types";
import { labelForBoardSource } from "../projectionSource";

const TABS: Array<{ id: LineupBoardSource; label: string }> = [
  { id: "league_value", label: "League Value" },
  { id: "vegas_props", label: "Vegas Props" },
];

/**
 * League Value ↔ Vegas Props toggle shared by Home and Matchup.
 *
 * Preference is persisted in app state (`fantasy-decisions:board-source`) and
 * sent to lineup as `?projection_source=` (APP_PROJECTION_SOURCE equivalent).
 */
export function LineupSourceTabs({
  value,
  onChange,
  /** When preference is unset, highlight this from the last API response. */
  fallbackSource,
  /** Effective board the API scored (may differ from preference on fallback). */
  effectiveSource,
  /** Server fallback reason, e.g. missing_weekly_props_pointer. */
  fallbackReason,
  hint = true,
}: {
  value: LineupBoardSource | null;
  onChange: (source: LineupBoardSource) => void;
  fallbackSource?: LineupBoardSource;
  effectiveSource?: LineupBoardSource;
  fallbackReason?: string | null;
  hint?: boolean;
}) {
  const active: LineupBoardSource = value ?? fallbackSource ?? "league_value";
  const scoredAs = effectiveSource ?? fallbackSource;
  const diverged =
    Boolean(value) && Boolean(scoredAs) && value !== scoredAs;
  return (
    <div className="lineup-source-strip">
      <div
        className="draft-pane-tabs lineup-source-tabs"
        role="tablist"
        aria-label="Projection board"
      >
        {TABS.map((tab) => {
          const selected = tab.id === active;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              className={`draft-pane-tab${selected ? " is-active" : ""}`}
              aria-selected={selected}
              onClick={() => onChange(tab.id)}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      {hint ? (
        <p className="lineup-source-hint muted">
          {diverged && scoredAs ? (
            <>
              Asked for {labelForBoardSource(active)}; scoring with{" "}
              {labelForBoardSource(scoredAs)}
              {fallbackReason ? ` (${fallbackReason.replace(/_/g, " ")})` : ""}.
            </>
          ) : (
            <>
              Scoring with {labelForBoardSource(active)}
              {value == null ? " (server default until you pick)" : ""}.
            </>
          )}
        </p>
      ) : null}
    </div>
  );
}
