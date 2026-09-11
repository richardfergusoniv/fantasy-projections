import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { InjuryEvidence, LeagueSummary, LineupRecommendation } from "../api/types";
import { AppStateProvider } from "../hooks/useAppState";
import { LineupScreen } from "./Lineup";

const META = {
  data_as_of: "2026-09-11T12:00:00Z",
  projection_run_id: "weekly-2026-w01-hashy-release",
};

const LEAGUES: LeagueSummary[] = [
  {
    id: "lg1",
    name: "Standard Half PPR",
    season: 2026,
    scoring_type: "half_ppr",
    roster_positions: ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "DST", "K"],
    is_dynasty: false,
    decision_ready: true,
    owner_roster_id: 1,
  },
];

const ROSTERS = [
  {
    roster_id: 1,
    week: 1,
    players: ["qb1", "rb1", "rb2"],
    starters: ["qb1", "rb1"],
    reserve: [],
  },
];

function starter(
  overrides: Partial<LineupRecommendation["starters"][number]> & {
    player_id: string;
    name: string;
    position: string;
  },
): LineupRecommendation["starters"][number] {
  return {
    team: "KC",
    slot: overrides.position,
    expected_points: 12.5,
    points_p10: 8,
    points_p50: 12,
    points_p90: 18,
    opponent: null,
    ...overrides,
  };
}

function lineup(overrides: Partial<LineupRecommendation> = {}): LineupRecommendation {
  return {
    week: 1,
    opponent_mode: "current",
    starters: [
      starter({
        player_id: "qb1",
        name: "Patrick Mahomes",
        position: "QB",
        slot: "QB",
        team: "KC",
        expected_points: 22.4,
        points_p10: 14,
        points_p90: 31,
        opponent: "HOU",
      }),
      starter({
        player_id: "rb1",
        name: "Isiah Pacheco",
        position: "RB",
        slot: "RB",
        team: "KC",
        expected_points: 11.2,
        opponent: "@JAX",
      }),
    ],
    bench: [
      starter({
        player_id: "rb2",
        name: "Kareem Hunt",
        position: "RB",
        slot: "BN",
        team: "KC",
        expected_points: 9.4,
        points_p10: 4,
        points_p90: 15,
      }),
    ],
    opponent_starters: [
      starter({
        player_id: "opp-qb",
        name: "Josh Allen",
        position: "QB",
        slot: "QB",
        team: "BUF",
        expected_points: 24.1,
        opponent: "@MIA",
      }),
      starter({
        player_id: "opp-rb",
        name: "James Cook",
        position: "RB",
        slot: "RB",
        team: "BUF",
        expected_points: 13.8,
      }),
    ],
    opponent_bench: [],
    swaps: [],
    win_probability: 0.61,
    matchup_probabilities: { win: 0.61, loss: 0.37, tie: 0.02 },
    points: { p10: 95, p50: 118, p90: 142, mean: 118.4 },
    opponent_expected_points: 109.1,
    contract_hash: "should-not-appear-on-screen",
    board_source: "league_value",
    meta: META,
    ...overrides,
  };
}

const getLeagues = vi.fn();
const getRosters = vi.fn();
const getLineup = vi.fn();
const getInjuryEvidence = vi.fn();

vi.mock("../api/client", () => ({
  api: {
    getLeagues: (...args: unknown[]) => getLeagues(...args),
    getRosters: (...args: unknown[]) => getRosters(...args),
    getLineup: (...args: unknown[]) => getLineup(...args),
    getInjuryEvidence: (...args: unknown[]) => getInjuryEvidence(...args),
    onUnauthorized: () => () => {},
  },
}));

function renderLineup() {
  return render(
    <MemoryRouter>
      <AppStateProvider>
        <LineupScreen />
      </AppStateProvider>
    </MemoryRouter>,
  );
}

describe("Matchup board", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    getLeagues.mockResolvedValue({
      leagues: LEAGUES,
      configuredLeagueIds: LEAGUES.map((league) => league.id),
    });
    getRosters.mockResolvedValue(ROSTERS);
    getLineup.mockResolvedValue(lineup());
    getInjuryEvidence.mockResolvedValue({
      player_id: "qb1",
      status: "unknown",
      summary: "No evidence",
      sources: [],
      meta: META,
    } satisfies InjuryEvidence);
  });

  it("renders you vs opponent starters with projections and bench", async () => {
    renderLineup();

    expect(await screen.findByText("Patrick Mahomes")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Matchup" })).toBeInTheDocument();
    expect(screen.getByText("Josh Allen")).toBeInTheDocument();
    expect(screen.getByText("Isiah Pacheco")).toBeInTheDocument();
    expect(screen.getByText("James Cook")).toBeInTheDocument();
    expect(screen.getByText("Kareem Hunt")).toBeInTheDocument();
    expect(screen.getByLabelText("Starter matchup board")).toBeInTheDocument();
    expect(screen.getByText("22.4")).toBeInTheDocument();
    expect(screen.getByText("24.1")).toBeInTheDocument();

    expect(screen.queryByText(/Release/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/should-not-appear-on-screen/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Status unknown/i)).not.toBeInTheDocument();
  });

  it("swaps a bench player into a starter slot locally", async () => {
    renderLineup();
    await screen.findByText("Kareem Hunt");

    fireEvent.click(screen.getByRole("button", { name: /Kareem Hunt/i }));
    fireEvent.click(screen.getByRole("button", { name: /Isiah Pacheco/i }));

    expect(screen.getByText(/Lineup adjusted locally/i)).toBeInTheDocument();
    // Hunt should now appear in the starter board (still named); Pacheco on bench.
    const board = screen.getByLabelText("Starter matchup board");
    expect(board).toHaveTextContent("Kareem Hunt");
    expect(board).not.toHaveTextContent("Isiah Pacheco");
  });

  it("shows a disabled League Value / Vegas Props seam", async () => {
    renderLineup();
    const leagueValue = await screen.findByRole("tab", { name: "League Value" });
    const vegas = screen.getByRole("tab", { name: "Vegas Props" });
    expect(leagueValue).toHaveAttribute("aria-selected", "true");
    expect(leagueValue).toBeDisabled();
    expect(vegas).toBeDisabled();
  });

  it("loads injury evidence for starters even when there are no swaps", async () => {
    renderLineup();
    await screen.findByText("Patrick Mahomes");
    await vi.waitFor(() => {
      expect(getInjuryEvidence).toHaveBeenCalled();
    });
    // Swap-only players are not required; starters are enough for board citations.
    expect(getInjuryEvidence.mock.calls.some((call) => call[0] === "qb1")).toBe(true);
  });

  it("applies a recommended swap onto the board", async () => {
    getLineup.mockResolvedValue(
      lineup({
        swaps: [
          {
            out_player_id: "rb1",
            in_player_id: "rb2",
            win_probability_delta: 0.03,
            reason: "Start Kareem Hunt over Isiah Pacheco",
          },
        ],
      }),
    );

    renderLineup();
    expect(await screen.findByText(/Start Kareem Hunt/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Apply on board" }));
    expect(screen.getByText(/Lineup adjusted locally/i)).toBeInTheDocument();
    const board = screen.getByLabelText("Starter matchup board");
    expect(board).toHaveTextContent("Kareem Hunt");
  });

  it("switches compact opponent mode", async () => {
    renderLineup();
    await screen.findByText("Patrick Mahomes");
    const best = screen.getByRole("button", { name: "Best possible" });
    fireEvent.click(best);
    await vi.waitFor(() => {
      expect(getLineup).toHaveBeenCalledWith("lg1", 1, "optimized");
    });
  });
});
