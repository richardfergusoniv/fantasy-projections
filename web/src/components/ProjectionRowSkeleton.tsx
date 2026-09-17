/**
 * Loading placeholders that share the loaded row's grid so the board does not
 * jump when data arrives.
 */
export function ProjectionRowSkeleton({
  variant,
  rows = 8,
}: {
  variant: "matchup" | "draft" | "checklist" | "waiver" | "snapshot";
  rows?: number;
}) {
  if (variant === "matchup") {
    return (
      <div
        className="projection-skeleton"
        data-testid="projection-skeleton"
        aria-hidden="true"
      >
        <div className="matchup-board-head">
          <div className="matchup-side-label">You</div>
          <div className="matchup-side-label matchup-side-label-center">Slot</div>
          <div className="matchup-side-label matchup-side-label-opp">Opponent</div>
        </div>
        <ul className="matchup-board">
          {Array.from({ length: rows }, (_, index) => (
            <li key={index} className="matchup-board-row">
              <div className="matchup-board-you">
                <span className="skel skel-name" />
                <span className="skel skel-proj" />
              </div>
              <span className="matchup-board-slot">
                <span className="skel skel-slot" />
              </span>
              <div className="matchup-board-opp">
                <span className="skel skel-proj" />
                <span className="skel skel-name" />
              </div>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  if (variant === "checklist") {
    return (
      <div
        className="draft-checklist-list projection-skeleton"
        data-testid="projection-skeleton"
        aria-hidden="true"
      >
        {Array.from({ length: rows }, (_, index) => (
          <article key={index} className="draft-checklist-row is-skeleton">
            <span className="skel skel-rank" />
            <div className="draft-checklist-main">
              <span className="skel skel-name" />
              <div className="draft-rank-pills">
                <span className="skel skel-stat" />
                <span className="skel skel-stat" />
                <span className="skel skel-stat" />
                <span className="skel skel-stat" />
              </div>
            </div>
          </article>
        ))}
      </div>
    );
  }

  if (variant === "waiver") {
    return (
      <ul
        className="waiver-list projection-table projection-skeleton"
        data-testid="projection-skeleton"
        aria-hidden="true"
      >
        {Array.from({ length: rows }, (_, index) => (
          <li key={index} className="waiver-item">
            <span className="skel skel-name" />
            <span className="skel skel-slot" />
            <span className="skel skel-stat" />
            <span className="skel skel-stat" />
            <span className="skel skel-rationale" />
          </li>
        ))}
      </ul>
    );
  }

  if (variant === "snapshot") {
    return (
      <div
        className="matchup-card matchup-chip projection-skeleton"
        data-testid="projection-skeleton"
        aria-hidden="true"
      >
        <div className="matchup-chip-score">
          <span className="skel skel-slot" />
          <span className="skel skel-proj" />
          <span className="skel skel-proj" />
        </div>
        <span className="skel skel-stat" />
      </div>
    );
  }

  return (
    <div
      className="draft-player-grid projection-skeleton"
      data-testid="projection-skeleton"
      aria-hidden="true"
    >
      {Array.from({ length: rows }, (_, index) => (
        <article key={index} className="draft-player-card is-skeleton">
          <span className="skel skel-rank" />
          <span className="skel skel-name" />
          <span className="skel skel-stat" />
          <span className="skel skel-stat" />
          <span className="skel skel-action" />
        </article>
      ))}
    </div>
  );
}
