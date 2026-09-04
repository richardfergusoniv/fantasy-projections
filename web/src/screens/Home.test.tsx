import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AppStateProvider } from "../hooks/useAppState";
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
    swaps: [],
    win_probability: 0.58,
    matchup_probabilities: { win: 0.58, loss: 0.41, tie: 0.01 },
    points: { p10: 90, p50: 113, p90: 138, mean: 113 },
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

  it("shows the selected league and its matchup snapshot", async () => {
    renderHome();

    expect(await screen.findByText("Standard Half PPR")).toBeInTheDocument();
    expect(await screen.findByText(/Win probability/i)).toBeInTheDocument();
    // Uncertainty is never hidden on a recommendation surface.
    expect(await screen.findByText(/Projected lineup points/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Draft assistant" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open Draft Checklist" })).toHaveAttribute(
      "href",
      "/draft?pane=checklist",
    );
    expect(screen.getByTestId("app-build-stamp")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
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
