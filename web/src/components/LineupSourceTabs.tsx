import type { LineupBoardSource } from "../api/types";

const TABS: Array<{ id: LineupBoardSource; label: string }> = [
  { id: "league_value", label: "League Value" },
  { id: "vegas_props", label: "Vegas Props" },
];

/**
 * Read-only seam for the future League Value ↔ Vegas Props flip.
 *
 * Draft already toggles boards; in-season Lineup will follow the same pattern
 * once `board_source` is published and source switching is wired. Until then
 * the strip shows which board scored this lineup and stays non-interactive.
 */
export function LineupSourceTabs({
  boardSource,
}: {
  boardSource?: LineupBoardSource;
}) {
  const active: LineupBoardSource = boardSource ?? "league_value";
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
              disabled
              title="Board switch coming soon — currently follows the API projection source"
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      <p className="lineup-source-hint muted">
        {boardSource
          ? `Scored with ${active === "vegas_props" ? "Vegas Props" : "League Value"}.`
          : "Projection board follows the server source (toggle coming soon)."}
      </p>
    </div>
  );
}
