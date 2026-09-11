import { describe, expect, it } from "vitest";
import type { LeagueSummary } from "../api/types";
import { ACTIVE_SEASON, pickPreferredLeagueId } from "./useAppState";

function league(
  partial: Partial<LeagueSummary> & Pick<LeagueSummary, "id" | "name" | "season">,
): LeagueSummary {
  return {
    scoring_type: "ppr",
    roster_positions: [],
    is_dynasty: false,
    is_configured: true,
    owner_roster_id: null,
    decision_ready: false,
    ...partial,
  };
}

describe("pickPreferredLeagueId", () => {
  it("prefers an active league with owner membership over a saved orphan", () => {
    const items = [
      league({ id: "hist", name: "2025", season: 2025, owner_roster_id: null }),
      league({ id: "owned", name: "2026", season: ACTIVE_SEASON, owner_roster_id: 6 }),
      league({
        id: "orphan-2026",
        name: "2026 orphan",
        season: ACTIVE_SEASON,
        owner_roster_id: null,
      }),
    ];
    expect(
      pickPreferredLeagueId(items, ["owned", "orphan-2026"], "hist", { showAll: false }),
    ).toBe("owned");
  });

  it("keeps the current league when it already has owner membership", () => {
    const items = [
      league({ id: "a", name: "A", season: ACTIVE_SEASON, owner_roster_id: 2 }),
      league({ id: "b", name: "B", season: ACTIVE_SEASON, owner_roster_id: 7 }),
    ];
    expect(pickPreferredLeagueId(items, ["a", "b"], "b", { showAll: false })).toBe("b");
  });
});

  it("still prefers owner-ready leagues when showAll is true", () => {
    const items = [
      league({ id: "hist", name: "2025", season: 2025, owner_roster_id: null, decision_ready: false }),
      league({
        id: "owned",
        name: "2026",
        season: ACTIVE_SEASON,
        owner_roster_id: 6,
        decision_ready: true,
      }),
    ];
    expect(
      pickPreferredLeagueId(items, ["owned"], "hist", { showAll: true }),
    ).toBe("owned");
  });
