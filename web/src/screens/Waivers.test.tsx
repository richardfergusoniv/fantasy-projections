import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { InjuryEvidence, LeagueSummary, WaiverRecommendation } from "../api/types";
import { AppStateProvider } from "../hooks/useAppState";
import { WaiversScreen } from "./Waivers";

const META = { data_as_of: "2026-08-30T17:00:00Z", projection_run_id: "weekly-2026-w01" };

const LEAGUES: LeagueSummary[] = [
  {
    id: "lg1",
    name: "Standard Half PPR",
    season: 2026,
    scoring_type: "half_ppr",
    roster_positions: ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
    is_dynasty: false,
    decision_ready: true,
    owner_roster_id: 1,
  },
];

const ROSTERS = [
  {
    roster_id: 1,
    week: 1,
    players: ["qb1"],
    starters: ["qb1"],
    reserve: [],
  },
];

function waivers(overrides: Partial<WaiverRecommendation> = {}): WaiverRecommendation {
  return {
    week: 1,
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
    meta: META,
    ...overrides,
  };
}

const getLeagues = vi.fn();
const getRosters = vi.fn();
const getWaivers = vi.fn();
const getInjuryEvidence = vi.fn();

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      getLeagues: (...args: unknown[]) => getLeagues(...args),
      getRosters: (...args: unknown[]) => getRosters(...args),
      getWaivers: (...args: unknown[]) => getWaivers(...args),
      getInjuryEvidence: (...args: unknown[]) => getInjuryEvidence(...args),
      onUnauthorized: () => () => {},
    },
  };
});

function renderWaivers() {
  return render(
    <MemoryRouter>
      <AppStateProvider>
        <WaiversScreen />
      </AppStateProvider>
    </MemoryRouter>,
  );
}

describe("Waivers compact board", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    getLeagues.mockResolvedValue({
      leagues: LEAGUES,
      configuredLeagueIds: LEAGUES.map((league) => league.id),
    });
    getRosters.mockResolvedValue(ROSTERS);
    getWaivers.mockResolvedValue(waivers());
    getInjuryEvidence.mockResolvedValue({
      player_id: "00-0036971",
      status: "unknown",
      summary: "No evidence",
      sources: [],
      meta: META,
    } satisfies InjuryEvidence);
  });

  it("keeps a FAAB range label and visible rationale on compact rows", async () => {
    renderWaivers();
    expect(await screen.findByText(/FAAB \$\d+–\$\d+/)).toBeInTheDocument();
    expect(document.querySelector(".rationale-list li")).not.toBeNull();
    expect(document.querySelector(".rationale-list li")).toHaveTextContent(
      /Replacement level at QB/,
    );
  });

  it("shows as-of unknown after an error instead of staying pending", async () => {
    getWaivers.mockRejectedValue(new Error("waivers unavailable"));
    renderWaivers();
    expect(await screen.findByText(/As-of unknown/i)).toBeInTheDocument();
    expect(screen.getByTestId("as-of-chrome").className).toMatch(/is-unknown/);
    expect(screen.getByTestId("as-of-chrome").className).not.toMatch(/is-pending/);
  });
});
