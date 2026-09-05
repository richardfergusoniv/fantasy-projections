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
    ranks: {
      rec_rank: index % 2 === 0 ? 3 : 18,
      qb_rank: index % 3 === 0 ? 5 : 22,
      offense_pts_rank: 7,
      sos_rank: index < 5 ? 4 : 20,
    },
  }));
  const qb = [
    {
      player_id: "qb-1",
      name: "QB Player 1",
      position: "QB" as const,
      team: "BUF",
      adp: 0.5,
      ecr: 1,
      prior_pts: 350,
      rank_tier: "adp" as const,
      pos_market_rank: 1,
      unranked_break: false,
      ranks: {
        total_yds_rank: 2,
        rush_yds_rank: 6,
        offense_pts_rank: 4,
        sos_rank: 19,
      },
    },
  ];
  return {
    league_id: "league-three-wr",
    season: 2026,
    available: true,
    entries: [...qb, ...wr],
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
      WR: ["rec_rank", "qb_rank", "offense_pts_rank", "sos_rank"],
      QB: ["total_yds_rank", "rush_yds_rank", "offense_pts_rank", "sos_rank"],
      RB: ["rec_rank", "rush_yds_rank", "offense_pts_rank", "sos_rank"],
      TE: ["rec_rank", "qb_rank", "offense_pts_rank", "sos_rank"],
    },
    criteria_labels: {
      rec_rank: "REC RANK",
      qb_rank: "QB RANK",
      offense_pts_rank: "OFFENSE PTS RANK",
      offense_yds_rank: "OFFENSE YDS RANK",
      sos_rank: "SOS RANK",
      total_yds_rank: "TOTAL YDS RANK",
      rush_yds_rank: "RUSH YDS RANK",
      ol_rank: "OL RANK",
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
      volume_caveat: "Vegas consensus volume/offense ranks + Sharp Fantasy SOS",
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
    expect(screen.getByRole("button", { name: "All", pressed: true })).toBeInTheDocument();
    // All tab sorts by overall ADP — QB at 0.5 beats WR Player 1 at ADP 1.
    const checkboxes = screen.getAllByRole("checkbox", { name: /Mark .+ drafted/i });
    expect(checkboxes[0]).toHaveAccessibleName("Mark QB Player 1 drafted");
    expect(checkboxes[1]).toHaveAccessibleName("Mark WR Player 1 drafted");
    expect(screen.getByText("YDS 2")).toBeInTheDocument();
    expect(screen.getByText("RUSH 6")).toBeInTheDocument();
    expect(screen.getByLabelText("Max avg rank")).toBeInTheDocument();
    expect(screen.getByText(/Unranked \/ off market/i)).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /O-line/i })).not.toBeInTheDocument();
  });

  it("filters checklist rows by max average rank", async () => {
    renderDraft("/draft?pane=checklist");
    await screen.findByRole("checkbox", { name: "Mark QB Player 1 drafted" });
    // QB avg = (2+6+4+19)/4 = 7.75 → kept at ≤8; many WRs average worse.
    fireEvent.change(screen.getByLabelText("Max avg rank"), { target: { value: "8" } });
    expect(screen.getByRole("checkbox", { name: "Mark QB Player 1 drafted" })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: /WR Player 2 drafted/i })).not.toBeInTheDocument();
  });

  it("filters the checklist to a single position from the All tab", async () => {
    renderDraft("/draft?pane=checklist");
    await screen.findByRole("checkbox", { name: "Mark QB Player 1 drafted" });

    fireEvent.click(screen.getByRole("button", { name: "WR" }));

    expect(screen.getByRole("button", { name: "WR", pressed: true })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Mark WR Player 1 drafted" })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: /QB Player 1 drafted/i })).not.toBeInTheDocument();
  });

  it("filters the checklist to FLEX-eligible positions (RB/WR/TE)", async () => {
    renderDraft("/draft?pane=checklist");
    await screen.findByRole("checkbox", { name: "Mark QB Player 1 drafted" });

    fireEvent.click(screen.getByRole("button", { name: "FLEX" }));

    expect(screen.getByRole("button", { name: "FLEX", pressed: true })).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Mark WR Player 1 drafted" })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: /QB Player 1 drafted/i })).not.toBeInTheDocument();
  });

  it("hides checklist rows when drafted via checkbox", async () => {
    renderDraft();
    fireEvent.click(await screen.findByRole("tab", { name: /Draft Checklist/i }));
    fireEvent.click(await screen.findByRole("button", { name: "WR" }));
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
      await screen.findByRole("checkbox", { name: "Mark QB Player 1 drafted" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/board unavailable/i)).toBeInTheDocument();
  });

  it("keeps the unranked divider when the flagged player is filtered out", async () => {
    renderDraft();
    fireEvent.click(await screen.findByRole("tab", { name: /Draft Checklist/i }));
    fireEvent.click(await screen.findByRole("button", { name: "WR" }));
    await screen.findByRole("checkbox", { name: "Mark WR Player 1 drafted" });
    // WR Player 21 carries unranked_break; hiding it must not hide the divider.
    fireEvent.click(screen.getByRole("checkbox", { name: "Mark WR Player 21 drafted" }));

    expect(
      screen.queryByRole("checkbox", { name: /WR Player 21 drafted/i }),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/Unranked \/ off market/i)).toBeInTheDocument();
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

    fireEvent.click(screen.getByRole("button", { name: "Undo" }));
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
