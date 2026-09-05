import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LeagueSwitcher } from "./LeagueSwitcher";

const setShowAllLeagues = vi.fn();
const selectLeague = vi.fn();
const setWeek = vi.fn();

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
      configuredLeagueIds: ["2026-a"],
      leagues,
      showAllLeagues: false,
      setShowAllLeagues,
      selectedLeagueId: "2026-a",
      selectLeague,
      leaguesLoading: false,
      leaguesError: null,
      availableWeeks: [1, 2],
      week: 2,
      setWeek,
      rostersLoading: false,
      season: 2026,
      activeSeason: 2026,
    };
  });

  it("shows Season instead of Week on the draft screen", () => {
    render(
      <MemoryRouter initialEntries={["/draft?pane=checklist"]}>
        <LeagueSwitcher />
      </MemoryRouter>,
    );
    expect(screen.getByLabelText("Season")).toBeInTheDocument();
    expect(screen.queryByLabelText("Week")).not.toBeInTheDocument();
    expect(screen.getByRole("option", { name: "Three Wide" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /2025/ })).not.toBeInTheDocument();
  });

  it("keeps the Week control on non-draft screens", () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <LeagueSwitcher />
      </MemoryRouter>,
    );
    expect(screen.getByLabelText("Week")).toBeInTheDocument();
    expect(screen.queryByLabelText("Season")).not.toBeInTheDocument();
  });
});
