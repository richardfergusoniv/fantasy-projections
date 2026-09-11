import { describe, expect, it } from "vitest";
import { adaptLeagues, adaptLineup } from "./adapters";

describe("adaptLeagues", () => {
  it("preserves owner_roster_id including null for unsynced memberships", () => {
    const leagues = adaptLeagues({
      leagues: [
        {
          id: "owned",
          league_id: "owned",
          name: "Owned",
          season: 2026,
          owner_roster_id: 6,
          decision_ready: true,
        },
        {
          id: "orphan",
          league_id: "orphan",
          name: "Orphan",
          season: 2025,
          owner_roster_id: null,
          decision_ready: false,
        },
      ],
    });
    expect(leagues[0].owner_roster_id).toBe(6);
    expect(leagues[1].owner_roster_id).toBeNull();
    expect(leagues[0].decision_ready).toBe(true);
    expect(leagues[1].decision_ready).toBe(false);
  });
});

describe("adaptLineup", () => {
  it("does not fabricate win probability when the API withholds it", () => {
    const lineup = adaptLineup({
      week: 1,
      starters: [],
      swaps: [],
      win_probability: null,
      matchup_win_probability_available: false,
      matchup_probabilities: { win: null, tie: null, loss: null },
      expected_points: 110,
      quantiles: { "0.1": 90, "0.5": 110, "0.9": 130 },
      data_as_of: "2026-08-30T17:00:00Z",
      projection_run_id: "preseason-v2_baseline_20260830",
    });

    expect(lineup.win_probability).toBeNull();
    expect(lineup.matchup_probabilities.win).toBeNull();
  });

  it("preserves published win probability when available", () => {
    const lineup = adaptLineup({
      week: 1,
      starters: [],
      swaps: [],
      win_probability: 0.62,
      matchup_win_probability_available: true,
      matchup_probabilities: { win: 0.62, tie: 0.01, loss: 0.37 },
      expected_points: 110,
      quantiles: { "0.1": 90, "0.5": 110, "0.9": 130 },
      data_as_of: "2026-08-30T17:00:00Z",
      projection_run_id: "weekly-2026-w01",
    });

    expect(lineup.win_probability).toBe(0.62);
  });

  it("maps per-starter points, team, and board_source without inventing opponent", () => {
    const lineup = adaptLineup({
      week: 2,
      starters: [
        {
          player_id: "00-0034857",
          name: "Patrick Mahomes",
          position: "QB",
          team: "KC",
          slot: "QB",
          expected_points: 22.4,
          points_p10: 14.1,
          points_p50: 21.8,
          points_p90: 31.2,
        },
      ],
      bench: [
        {
          player_id: "00-bench",
          name: "Bench RB",
          position: "RB",
          slot: "BN",
          expected_points: 6.2,
          points_p10: 2,
          points_p90: 11,
        },
      ],
      opponent_starter_details: [
        {
          player_id: "00-opp",
          name: "Josh Allen",
          position: "QB",
          team: "BUF",
          slot: "QB",
          expected_points: 24.0,
        },
      ],
      opponent_bench: [],
      opponent_starters: ["00-opp"],
      swaps: [],
      win_probability: 0.55,
      matchup_probabilities: { win: 0.55, tie: 0.02, loss: 0.43 },
      expected_points: 118.2,
      opponent_expected_points: 111.4,
      quantiles: { "0.1": 95, "0.5": 118, "0.9": 142 },
      board_source: "league_value",
      effective_source: "sealed_release",
      data_as_of: "2026-09-11T12:00:00Z",
      projection_run_id: "weekly-2026-w02",
      contract_hash: "abc123deadbeef",
    });

    expect(lineup.starters).toHaveLength(1);
    expect(lineup.starters[0]).toMatchObject({
      name: "Patrick Mahomes",
      team: "KC",
      slot: "QB",
      expected_points: 22.4,
      points_p10: 14.1,
      points_p90: 31.2,
      opponent: null,
    });
    expect(lineup.bench).toHaveLength(1);
    expect(lineup.opponent_starters[0]).toMatchObject({
      name: "Josh Allen",
      slot: "QB",
      expected_points: 24.0,
    });
    expect(lineup.opponent_expected_points).toBe(111.4);
    expect(lineup.board_source).toBe("league_value");
    expect(lineup.contract_hash).toBe("abc123deadbeef");
  });

  it("infers vegas_props board_source from weekly_props effective_source", () => {
    const lineup = adaptLineup({
      week: 1,
      starters: [],
      swaps: [],
      effective_source: "weekly_props",
      expected_points: 100,
      data_as_of: "2026-09-11T12:00:00Z",
      projection_run_id: "weekly-props-w01",
    });
    expect(lineup.board_source).toBe("vegas_props");
  });
});
