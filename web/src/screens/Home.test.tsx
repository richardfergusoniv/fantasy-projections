import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppStateProvider } from "../hooks/useAppState";
import { BOARD_SOURCE_KEY } from "../projectionSource";
import { HomeScreen } from "./Home";
import type {
  LineupRecommendation,
  OperationsStatus,
  WaiverRecommendation,
} from "../api/types";

/**
 * Home's job is to derive the "urgent decisions" list from responses the app
 * actually received — never from a static reminder string. These tests drive
 * that derivation from the API boundary and assert what the owner would read,
 * including the case where nothing is urgent (which must say so explicitly
 * rather than render an empty panel).
 */

const META = { data_as_of: "2026-08-30T17:00:00Z", projection_run_id: "weekly-2026-w01" };

const LEAGUES = [
  {
    id: "fixture-standard",
    name: "Standard Half PPR",
    season: 2026,
    scoring_type: "half_ppr",
    roster_positions: ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
    is_dynasty: false,
  },
];

const ROSTERS = [
  { roster_id: 1, week: 1, players: ["00-0034857"], starters: ["00-0034857"], reserve: [] },
];

function lineup(overrides: Partial<LineupRecommendation> = {}): LineupRecommendation {
  return {
    week: 1,
    opponent_mode: "current",
    starters: [],
    bench: [],
    opponent_starters: [],
    opponent_bench: [],
    swaps: [],
    win_probability: 0.58,
    matchup_probabilities: { win: 0.58, loss: 0.41, tie: 0.01 },
    points: { p10: 90, p50: 113, p90: 138, mean: 113 },
    opponent_expected_points: 104.2,
    board_source: "league_value",
    meta: META,
    ...overrides,
  };
}

function waivers(overrides: Partial<WaiverRecommendation> = {}): WaiverRecommendation {
  return { week: 1, adds: [], meta: META, ...overrides };
}

function operations(overrides: Partial<OperationsStatus> = {}): OperationsStatus {
  return {
    last_sync_at: "2026-08-30T17:00:00Z",
    failed_gates: [],
    ...overrides,
  } as OperationsStatus;
}

const getLeagues = vi.fn();
const getRosters = vi.fn();
const getLineup = vi.fn();
const getWaivers = vi.fn();
const getOperationsStatus = vi.fn();

vi.mock("../api/client", () => ({
  api: {
    getLeagues: (...args: unknown[]) => getLeagues(...args),
    getRosters: (...args: unknown[]) => getRosters(...args),
    getLineup: (...args: unknown[]) => getLineup(...args),
    getWaivers: (...args: unknown[]) => getWaivers(...args),
    getOperationsStatus: (...args: unknown[]) => getOperationsStatus(...args),
    onUnauthorized: () => () => {},
  },
}));

function renderHome() {
  return render(
    <MemoryRouter>
      <AppStateProvider>
        <HomeScreen />
      </AppStateProvider>
    </MemoryRouter>,
  );
}

describe("HomeScreen urgent decisions", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    getLeagues.mockResolvedValue({
      leagues: LEAGUES,
      configuredLeagueIds: LEAGUES.map((league) => league.id),
    });
    getRosters.mockResolvedValue(ROSTERS);
    getLineup.mockResolvedValue(lineup());
    getWaivers.mockResolvedValue(waivers());
    getOperationsStatus.mockResolvedValue(operations());
  });

  it("shows a scannable matchup chip and projection source toggle", async () => {
    renderHome();

    expect(await screen.findByText("Standard Half PPR")).toBeInTheDocument();
    expect(await screen.findByTestId("matchup-snapshot-chip")).toBeInTheDocument();
    expect(await screen.findByText(/Win/i)).toBeInTheDocument();
    expect(screen.getByText("113.0")).toBeInTheDocument();
    expect(screen.getByText("104.2")).toBeInTheDocument();
    // No interval prose, swap dump, or Draft Assistant deep-links.
    expect(screen.queryByText(/Projected lineup points/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/80% interval/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Draft assistant" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Open Vegas Props" })).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Projection source" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Vegas Props" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "League Value" })).toBeInTheDocument();
    expect(screen.getByTestId("app-build-stamp")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
  });

  it("persists Vegas / League Value as the app projection preference", async () => {
    renderHome();
    await screen.findByText("Standard Half PPR");

    fireEvent.click(screen.getByRole("tab", { name: "Vegas Props" }));
    await waitFor(() => {
      expect(localStorage.getItem(BOARD_SOURCE_KEY)).toBe("vegas_props");
    });
    await waitFor(() => {
      expect(getLineup).toHaveBeenCalledWith(
        "fixture-standard",
        1,
        "current",
        "weekly_props",
      );
    });

    fireEvent.click(screen.getByRole("tab", { name: "League Value" }));
    await waitFor(() => {
      expect(localStorage.getItem(BOARD_SOURCE_KEY)).toBe("league_value");
    });
    await waitFor(() => {
      expect(getLineup).toHaveBeenCalledWith(
        "fixture-standard",
        1,
        "current",
        "sealed_release",
      );
    });
  });

  it("derives urgent items from swaps, waiver targets, and failed gates", async () => {
    getLineup.mockResolvedValue(
      lineup({
        swaps: [
          {
            out_player_id: "00-0032764",
            in_player_id: "00-0039075",
            win_probability_delta: 0.04,
            reason: "Start Puka Nacua over Derrick Henry",
          },
        ],
      }),
    );
    getWaivers.mockResolvedValue(
      waivers({
        adds: [
          {
            player_id: "00-0036971",
            name: "Trevor Lawrence",
            position: "QB",
            faab_min: 4,
            faab_max: 10,
            reason: "Replacement-level upgrade at QB",
            rationale: ["Replacement level at QB: 20.03 pts/wk."],
            confidence: 0.58,
            start_probability: 0.69,
            incremental_utility: 4.99,
          },
        ],
      }),
    );
    getOperationsStatus.mockResolvedValue(
      operations({ failed_gates: ["scoring_contract:unsupported_key"] }),
    );

    renderHome();

    expect(await screen.findByText(/1 start\/sit swap/i)).toBeInTheDocument();
    expect(await screen.findByText(/Trevor Lawrence at \$4–\$10 FAAB/i)).toBeInTheDocument();
    expect(await screen.findByText(/1 release gate failure/i)).toBeInTheDocument();
  });

  it("says nothing is urgent instead of rendering an empty panel", async () => {
    renderHome();

    expect(
      await screen.findByText(/Nothing urgent from current data/i),
    ).toBeInTheDocument();
  });

  it("warns that a never-synced app must not be trusted", async () => {
    getOperationsStatus.mockResolvedValue(operations({ last_sync_at: null }));

    renderHome();

    expect(
      await screen.findByText(/No source snapshot has ever been recorded/i),
    ).toBeInTheDocument();
  });
});
