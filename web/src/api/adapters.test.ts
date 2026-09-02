import { describe, expect, it } from "vitest";
import { adaptLineup } from "./adapters";

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
});
