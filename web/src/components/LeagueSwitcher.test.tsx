import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LeagueSwitcher } from "./LeagueSwitcher";

const selectLeague = vi.fn();

const leagues = [
  {
    id: "2026-a",
    name: "Three Wide",
    season: 2026,
    scoring_type: "half_ppr",
    roster_positions: ["QB"],
    is_dynasty: false,
  },
  {
    id: "2025-a",
    name: "Three Wide",
    season: 2025,
    scoring_type: "half_ppr",
    roster_positions: ["QB"],
    is_dynasty: false,
  },
];

let mockState: Record<string, unknown>;

vi.mock("../hooks/useAppState", () => ({
  useAppState: () => mockState,
}));

describe("LeagueSwitcher", () => {
  beforeEach(() => {
    mockState = {
      visibleLeagues: [leagues[0]],
      selectedLeagueId: "2026-a",
      selectLeague,
      leaguesLoading: false,
      leaguesError: null,
      activeSeason: 2026,
    };
  });

  it("puts Regression Model left of Leagues and omits week/history controls", () => {
    render(
      <MemoryRouter initialEntries={["/draft"]}>
        <LeagueSwitcher />
      </MemoryRouter>,
    );

    const regression = screen.getByRole("link", { name: "Regression Model" });
    const leaguesLabel = screen.getByLabelText("Leagues");
    expect(regression.compareDocumentPosition(leaguesLabel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(regression).toHaveAttribute("aria-current", "page");
    expect(screen.queryByLabelText("Week")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Season")).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/historical/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Show historical leagues/i)).not.toBeInTheDocument();
  });

  it("does not mark Regression Model current on the Vegas Props pane", () => {
    render(
      <MemoryRouter initialEntries={["/draft?pane=checklist"]}>
        <LeagueSwitcher />
      </MemoryRouter>,
    );
    expect(screen.getByRole("link", { name: "Regression Model" })).not.toHaveAttribute(
      "aria-current",
    );
  });

  it("keeps Regression Model available off the draft screen", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <LeagueSwitcher />
      </MemoryRouter>,
    );
    expect(screen.getByRole("link", { name: "Regression Model" })).toHaveAttribute("href", "/draft");
    expect(screen.getByLabelText("Leagues")).toBeInTheDocument();
  });
});
