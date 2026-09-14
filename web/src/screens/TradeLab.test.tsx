import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Roster } from "../api/types";
import { TradeLabScreen } from "./TradeLab";

let mockState: Record<string, unknown>;

vi.mock("../hooks/useAppState", () => ({
  useAppState: () => mockState,
}));

vi.mock("../api/client", () => ({
  api: {
    evaluateTrade: vi.fn(),
  },
}));

function roster(overrides: Partial<Roster> = {}): Roster {
  return {
    roster_id: 1,
    week: 1,
    players: [],
    starters: [],
    reserve: [],
    ...overrides,
  };
}

function renderTradeLab() {
  return render(<TradeLabScreen />);
}

describe("Trade Lab player labels", () => {
  beforeEach(() => {
    mockState = {
      selectedLeagueId: "dynasty-1",
      selectedLeague: { name: "Hoe Ass Dynasty League", season: 2026 },
      rosters: [
        roster({
          roster_id: 1,
          manager_name: "rdfergus15",
          players: ["11583", "11604", "k-id"],
          player_details: [
            // Production bug: details were remapped onto GSIS while `players`
            // still carried Sleeper ids, so only ST players whose ids already
            // matched (Will Reichard) rendered as names.
            { player_id: "00-0036322", name: "Justin Jefferson", position: "WR" },
            { player_id: "00-0037740", name: "Puka Nacua", position: "WR" },
            { player_id: "k-id", name: "Will Reichard", position: "K" },
          ],
        }),
        roster({
          roster_id: 2,
          manager_name: "kdicillo",
          players: ["10232"],
          player_details: [
            { player_id: "00-0031288", name: "Tyreek Hill", position: "WR" },
          ],
        }),
      ],
      rostersLoading: false,
      rostersError: null,
    };
  });

  it("shows Name (POS) for send/receive checklists instead of raw Sleeper ids", () => {
    renderTradeLab();

    expect(screen.getByLabelText("Justin Jefferson (WR)")).toBeInTheDocument();
    expect(screen.getByLabelText("Puka Nacua (WR)")).toBeInTheDocument();
    expect(screen.getByLabelText("Will Reichard (K)")).toBeInTheDocument();
    expect(screen.getByLabelText("Tyreek Hill (WR)")).toBeInTheDocument();

    expect(screen.queryByLabelText("11583")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("11604")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("10232")).not.toBeInTheDocument();
  });
});
