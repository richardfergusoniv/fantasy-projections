import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { DraftBoard, DraftChecklist } from "../api/types";
import { DraftScreen } from "./Draft";

const getDraftBoard = vi.hoisted(() => vi.fn());
const getDraftChecklist = vi.hoisted(() => vi.fn());

vi.mock("../api/client", () => ({
  api: {
    getDraftBoard: (...args: unknown[]) => getDraftBoard(...args),
    getDraftChecklist: (...args: unknown[]) => getDraftChecklist(...args),
  },
}));

vi.mock("../hooks/useAppState", () => ({
  useAppState: () => ({
    selectedLeagueId: "league-three-wr",
    selectedLeague: { name: "Three Wide League", season: 2026 },
  }),
}));

function renderDraft(initialEntry = "/draft") {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <DraftScreen />
    </MemoryRouter>,
  );
}

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
      ranking_basis: "league_vorp",
      points_unit: "season_total",
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

function checklist(): DraftChecklist {
  const wr = Array.from({ length: 30 }, (_, index) => ({
    player_id: `wr-${index + 1}`,
    name: `WR Player ${index + 1}`,
    position: "WR" as const,
    team: "SEA",
    adp: index < 10 ? index + 1 : null,
    ecr: index < 20 ? index + 5 : null,
    prior_pts: 200 - index,
    rank_tier: (index < 10 ? "adp" : index < 20 ? "ecr" : "prior_pts") as
      | "adp"
      | "ecr"
      | "prior_pts",
    pos_market_rank: index + 1,
    unranked_break: index === 20,
    checks: {
      target_leader_in_group: index % 2 === 0,
      qb_top16: index % 3 === 0,
      offense_top16: true,
      sos_top16: index < 5,
    },
  }));
  return {
    league_id: "league-three-wr",
    season: 2026,
    available: true,
    entries: wr,
    teams: [
      {
        abbr: "SEA",
        name: "Seattle Seahawks",
        offense_rank: 7,
        sos_unit_rank: 12,
      },
      {
        abbr: "DAL",
        name: "Dallas Cowboys",
        offense_rank: 1,
        sos_unit_rank: 20,
      },
    ],
    criteria_by_position: {
      WR: ["target_leader_in_group", "qb_top16", "offense_top16", "sos_top16"],
      QB: ["pass_att_top16", "rush_vol_top16", "offense_top16", "sos_top16"],
      RB: ["target_leader_in_group", "rush_vol_leader_in_group", "offense_top16", "sos_top16"],
      TE: ["te_top2_targets_in_group", "qb_top16", "offense_top16", "sos_top16"],
    },
    criteria_labels: {
      target_leader_in_group: "2025 TGT LEADER IN GROUP",
      qb_top16: "TOP 16 QB",
      offense_top16: "TOP 16 OFFENSE",
      sos_top16: "TOP 16 SOS",
    },
    checklist_meta: {
      market_as_of: {
        adp_end: "2026-09-03",
        ecr_scrape: "2026-08-28",
        scoring: "half-ppr",
        teams: 12,
      },
      sos_included: true,
      ol_included: false,
      volume_caveat: "2025 volume leader within this team's 2026 skill group",
    },
    meta: {
      data_as_of: "2026-09-03T20:00:00Z",
      projection_run_id: "checklist-2026",
    },
  };
}

async function openOursPane() {
  const ours = await screen.findByRole("tab", { name: /Our Rankings/i });
  if (ours.getAttribute("aria-selected") !== "true") fireEvent.click(ours);
  const position = await screen.findByLabelText("Position");
  fireEvent.change(position, { target: { value: "ALL" } });
  expect(await screen.findByText(/League-adjusted · 12 teams/i)).toBeInTheDocument();
}

describe("DraftScreen", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    getDraftBoard.mockResolvedValue(board());
    getDraftChecklist.mockResolvedValue(checklist());
  });

  it("opens the checklist from the pane query", async () => {
    renderDraft("/draft?pane=checklist");
    expect(await screen.findByText(/Market as of ADP 2026-09-03/i)).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Draft Checklist/i })).toHaveAttribute("aria-selected", "true");
  });

  it("defaults to league-specific VORP rankings rather than the market checklist", async () => {
    renderDraft();
    expect(await screen.findByRole("listitem", { name: "Draft Player 1 draft card" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Our Rankings/i })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/Ranked by league VORP, not raw quarterback points/i)).toBeInTheDocument();
    expect(screen.queryByText(/Market as of ADP 2026-09-03/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Draft Checklist/i }));
    expect(await screen.findByText(/Market as of ADP 2026-09-03/i)).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Mark WR Player 1 drafted" })).toBeInTheDocument();
    expect(screen.getByText(/Unranked \/ off market board/i)).toBeInTheDocument();
  });

  it("hides checklist rows when drafted via checkbox", async () => {
    renderDraft();
    fireEvent.click(await screen.findByRole("tab", { name: /Draft Checklist/i }));
    await screen.findByRole("checkbox", { name: "Mark WR Player 1 drafted" });
    fireEvent.click(screen.getByRole("checkbox", { name: "Mark WR Player 1 drafted" }));
    expect(screen.queryByRole("checkbox", { name: /WR Player 1 drafted/i })).not.toBeInTheDocument();
    expect(
      JSON.parse(
        localStorage.getItem("fantasy-decisions:drafted:league-three-wr:2026") ?? "[]",
      ),
    ).toEqual(["wr-1"]);
  });

  it("keeps the Ours board usable when the checklist request fails", async () => {
    getDraftChecklist.mockRejectedValue(new Error("checklist unavailable"));
    renderDraft();

    expect(await screen.findByText(/checklist unavailable/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: /Our Rankings/i }));
    fireEvent.change(await screen.findByLabelText("Position"), {
      target: { value: "ALL" },
    });
    expect(
      await screen.findByRole("listitem", { name: "Draft Player 1 draft card" }),
    ).toBeInTheDocument();
  });

  it("keeps the checklist usable when the draft board request fails", async () => {
    getDraftBoard.mockRejectedValue(new Error("board unavailable"));
    renderDraft();

    fireEvent.click(await screen.findByRole("tab", { name: /Draft Checklist/i }));
    expect(
      await screen.findByRole("checkbox", { name: "Mark WR Player 1 drafted" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/board unavailable/i)).toBeInTheDocument();
  });

  it("keeps the unranked divider when the flagged player is filtered out", async () => {
    renderDraft();
    fireEvent.click(await screen.findByRole("tab", { name: /Draft Checklist/i }));
    await screen.findByRole("checkbox", { name: "Mark WR Player 1 drafted" });
    // WR Player 21 carries unranked_break; hiding it must not hide the divider.
    fireEvent.click(screen.getByRole("checkbox", { name: "Mark WR Player 21 drafted" }));

    expect(
      screen.queryByRole("checkbox", { name: /WR Player 21 drafted/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/Unranked \/ off market board/i)).toBeInTheDocument();
  });

  it("shows o-line offense ranks when OL is unavailable", async () => {
    renderDraft();
    fireEvent.click(await screen.findByRole("tab", { name: /O-line/i }));
    expect(await screen.findByText(/Offense #1/i)).toBeInTheDocument();
    expect(screen.getAllByText(/OL ranks unavailable/i).length).toBeGreaterThan(0);
  });

  it("shows the league format and paginates beyond the first 15 players", async () => {
    renderDraft();
    await openOursPane();

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
    renderDraft();
    await openOursPane();
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
    const firstRender = renderDraft();
    await openOursPane();
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
    renderDraft();
    await openOursPane();
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
    renderDraft();
    await openOursPane();
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
