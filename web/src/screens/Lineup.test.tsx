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
  { roster_id: 1, week: 1, players: ["qb1", "rb1"], starters: ["qb1", "rb1"], reserve: [] },
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

describe("LineupScreen fantasy layout", () => {
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

  it("renders slot / player / opponent / projected points without release hash noise", async () => {
    renderLineup();

    expect(await screen.findByText("Patrick Mahomes")).toBeInTheDocument();
    expect(screen.getByText("Isiah Pacheco")).toBeInTheDocument();
    expect(screen.getByText("QB")).toBeInTheDocument();
    expect(screen.getByText(/KC · QB/)).toBeInTheDocument();
    expect(screen.getByText("vs HOU")).toBeInTheDocument();
    expect(screen.getByText("@JAX")).toBeInTheDocument();
    expect(screen.getByText("22.4")).toBeInTheDocument();
    expect(screen.getByText("11.2")).toBeInTheDocument();

    expect(screen.queryByText(/Release/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/scoring contract/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/should-not-appear-on-screen/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Status unknown/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/No evidence/i)).not.toBeInTheDocument();
  });

  it("shows a disabled League Value / Vegas Props seam", async () => {
    renderLineup();
    const leagueValue = await screen.findByRole("tab", { name: "League Value" });
    const vegas = screen.getByRole("tab", { name: "Vegas Props" });
    expect(leagueValue).toHaveAttribute("aria-selected", "true");
    expect(leagueValue).toBeDisabled();
    expect(vegas).toBeDisabled();
  });

  it("does not fan out injury lookups for every starter when there are no swaps", async () => {
    renderLineup();
    await screen.findByText("Patrick Mahomes");
    expect(getInjuryEvidence).not.toHaveBeenCalled();
  });

  it("shows actionable swap evidence only", async () => {
    getLineup.mockResolvedValue(
      lineup({
        swaps: [
          {
            out_player_id: "rb1",
            in_player_id: "rb2",
            win_probability_delta: 0.03,
            reason: "Start Bijan Robinson over Isiah Pacheco",
          },
        ],
      }),
    );
    getInjuryEvidence.mockImplementation(async (id: string) => {
      if (id === "rb1") {
        return {
          player_id: "rb1",
          status: "Questionable",
          summary: "Ankle — limited practice",
          sources: [{ title: "Injury report", url: "https://example.com/inj", publisher: "example.com" }],
          meta: META,
        } satisfies InjuryEvidence;
      }
      return {
        player_id: id,
        status: "unknown",
        summary: "No evidence",
        sources: [],
        meta: META,
      } satisfies InjuryEvidence;
    });

    renderLineup();
    expect(await screen.findByText(/Start Bijan Robinson/)).toBeInTheDocument();
    expect(await screen.findAllByText("Questionable")).not.toHaveLength(0);
    expect(screen.getAllByText(/Ankle/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/No evidence/i)).not.toBeInTheDocument();
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
