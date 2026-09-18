import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProjectionRowSkeleton } from "./ProjectionRowSkeleton";

describe("ProjectionRowSkeleton", () => {
  it("emits matchup rows with the same grid class as loaded data", () => {
    render(<ProjectionRowSkeleton variant="matchup" rows={8} />);
    const region = screen.getByTestId("projection-skeleton");
    expect(region).toHaveAttribute("aria-hidden", "true");
    expect(region.querySelectorAll(".matchup-board-row")).toHaveLength(8);
    expect(region.querySelector(".matchup-board-head")).not.toBeNull();
    expect(region.querySelector(".matchup-board-you")).not.toBeNull();
    expect(region.querySelector(".matchup-board-opp")).not.toBeNull();
  });

  it("emits compact draft rows instead of a spinner", () => {
    render(<ProjectionRowSkeleton variant="draft" rows={6} />);
    const region = screen.getByTestId("projection-skeleton");
    expect(region.querySelectorAll(".draft-player-card")).toHaveLength(6);
    expect(region.querySelector(".draft-card-stats")).toBeNull();
  });

  it("emits checklist rows with the same grid class as loaded data", () => {
    render(<ProjectionRowSkeleton variant="checklist" rows={6} />);
    const region = screen.getByTestId("projection-skeleton");
    expect(region.querySelectorAll(".draft-checklist-row")).toHaveLength(6);
    expect(region.querySelector(".draft-checklist-main")).not.toBeNull();
    expect(region.querySelectorAll(".draft-rank-pills")).toHaveLength(6);
  });
});
