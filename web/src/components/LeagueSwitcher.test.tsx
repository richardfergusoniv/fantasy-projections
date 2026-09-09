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

  it("keeps the league picker and omits week / history controls", () => {
    render(
      <MemoryRouter initialEntries={["/draft"]}>
        <LeagueSwitcher />
      </MemoryRouter>,
    );

    expect(screen.getByLabelText("Leagues")).toBeInTheDocument();
    expect(screen.queryByLabelText("Week")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Season")).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/historical/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Show historical leagues/i)).not.toBeInTheDocument();
  });

  // The shell used to carry a "Regression Model" link to one of the two draft
  // boards, on every screen. Both boards are tabs on the Draft screen now.
  it("carries no draft-board shortcut", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <LeagueSwitcher />
      </MemoryRouter>,
    );
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Leagues")).toBeInTheDocument();
  });

  it("reports a league load failure in the shell", () => {
    mockState = { ...mockState, leaguesError: "sync failed", visibleLeagues: [] };
    render(
      <MemoryRouter initialEntries={["/"]}>
        <LeagueSwitcher />
      </MemoryRouter>,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Leagues unavailable: sync failed");
  });
});
