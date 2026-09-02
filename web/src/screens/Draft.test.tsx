import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { DraftBoard } from "../api/types";
import { DraftScreen } from "./Draft";

const getDraftBoard = vi.hoisted(() => vi.fn());

vi.mock("../api/client", () => ({
  api: {
    getDraftBoard: (...args: unknown[]) => getDraftBoard(...args),
  },
}));

vi.mock("../hooks/useAppState", () => ({
  useAppState: () => ({
    selectedLeagueId: "league-three-wr",
    selectedLeague: { name: "Three Wide League", season: 2026 },
  }),
}));

function board(): DraftBoard {
  return {
    league_id: "league-three-wr",
    entries: Array.from({ length: 40 }, (_, index) => ({
      player_id: `player-${index + 1}`,
      name: `Draft Player ${index + 1}`,
      position: (["QB", "RB", "WR", "TE"] as const)[index % 4],
      rank: index + 1,
      tier: Math.floor(index / 8) + 1,
      vorp: index === 20 ? 0 : 100 - index * 4,
      points_mean: 300 - index,
      replacement_rank: 25,
    })),
    context: { draft_status: "preseason", draft_id: null },
    profile: {
      league_specific: true,
      team_count: 12,
      roster_positions: ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX", "BN"],
      scoring_fidelity: "exact_component_rescore",
      replacement_ranks: { QB: 13, RB: 31, WR: 43, TE: 13 },
      caveats: [],
    },
    meta: {
      data_as_of: "2026-09-01T17:00:00Z",
      projection_run_id: "preseason-v2_baseline_20260830",
    },
  };
}

describe("DraftScreen", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    getDraftBoard.mockResolvedValue(board());
  });

  it("shows the league format and paginates beyond the first 15 players", async () => {
    render(<DraftScreen />);

    expect(
      await screen.findByRole("listitem", { name: "Draft Player 25 draft card" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("listitem", { name: "Draft Player 26 draft card" }),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/League-adjusted · 12 teams/i)).toBeInTheDocument();
    expect(screen.getByText(/WR · WR · WR · TE · FLEX/i)).toBeInTheDocument();
    expect(screen.getByText(/WR43/i)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Show 25 more" }));

    expect(
      screen.getByRole("listitem", { name: "Draft Player 40 draft card" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Showing 40 of 40 matching players/i)).toBeInTheDocument();
    const replacementCard = screen.getByRole("listitem", {
      name: "Draft Player 21 draft card",
    });
    expect(within(replacementCard).getByText("Replacement")).toBeInTheDocument();
  });

  it("filters the complete board by position", async () => {
    render(<DraftScreen />);
    await screen.findByRole("listitem", { name: "Draft Player 25 draft card" });

    fireEvent.change(screen.getByLabelText("Position"), { target: { value: "WR" } });

    expect(screen.getByText(/Showing 10 of 10 matching players/i)).toBeInTheDocument();
    expect(
      screen.getByRole("listitem", { name: "Draft Player 3 draft card" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("listitem", { name: "Draft Player 2 draft card" }),
    ).not.toBeInTheDocument();
  });

  it("marks drafted players, updates best available, and persists per league", async () => {
    const firstRender = render(<DraftScreen />);
    await screen.findByRole("listitem", { name: "Draft Player 1 draft card" });

    fireEvent.click(screen.getByRole("button", { name: "Mark Draft Player 1 drafted" }));

    expect(
      screen.queryByRole("listitem", { name: "Draft Player 1 draft card" }),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/Best available: Draft Player 2/i)).toBeInTheDocument();
    expect(screen.getByText(/1 drafted/i)).toBeInTheDocument();
    expect(
      JSON.parse(
        localStorage.getItem("fantasy-decisions:drafted:league-three-wr:2026") ?? "[]",
      ),
    ).toEqual(["player-1"]);

    firstRender.unmount();
    render(<DraftScreen />);
    await screen.findByText(/Best available: Draft Player 2/i);
    expect(
      screen.queryByRole("listitem", { name: "Draft Player 1 draft card" }),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Undo last drafted" }));
    expect(
      await screen.findByRole("listitem", { name: "Draft Player 1 draft card" }),
    ).toBeInTheDocument();
  });

  it("can show drafted cards and undo an individual player", async () => {
    render(<DraftScreen />);
    await screen.findByRole("listitem", { name: "Draft Player 1 draft card" });
    fireEvent.click(screen.getByRole("button", { name: "Mark Draft Player 1 drafted" }));

    fireEvent.click(screen.getByLabelText(/Hide drafted/i));

    expect(
      screen.getByRole("listitem", { name: "Draft Player 1 draft card" }),
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Undo Draft Player 1 drafted" }));
    expect(screen.getByRole("button", { name: "Mark Draft Player 1 drafted" })).toBeInTheDocument();
  });
});
