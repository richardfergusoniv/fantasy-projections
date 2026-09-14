import { describe, expect, it } from "vitest";
import type { Roster } from "../api/types";
import {
  buildRosterPlayerLabels,
  formatRosterPlayerLabel,
  isUnresolvedPlayerName,
} from "./labels";

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

describe("formatRosterPlayerLabel", () => {
  it("renders Name (POS) when the detail has a real name", () => {
    expect(
      formatRosterPlayerLabel("11583", {
        player_id: "11583",
        name: "Justin Jefferson",
        position: "WR",
      }),
    ).toBe("Justin Jefferson (WR)");
  });

  it("still uses the display name when details were remapped onto GSIS", () => {
    expect(
      formatRosterPlayerLabel("11583", {
        player_id: "00-0036322",
        name: "Justin Jefferson",
        position: "WR",
      }),
    ).toBe("Justin Jefferson (WR)");
  });

  it("falls back to the roster id when name is just the raw id", () => {
    expect(
      formatRosterPlayerLabel("11583", {
        player_id: "11583",
        name: "11583",
        position: "FLEX",
      }),
    ).toBe("11583");
  });
});

describe("isUnresolvedPlayerName", () => {
  it("treats empty and id-echo names as unresolved", () => {
    expect(isUnresolvedPlayerName("", "11583")).toBe(true);
    expect(isUnresolvedPlayerName("11583", "11583")).toBe(true);
    expect(isUnresolvedPlayerName("Justin Jefferson", "11583")).toBe(false);
  });
});

describe("buildRosterPlayerLabels", () => {
  it("joins remapped player_details onto Sleeper roster ids by index", () => {
    const labels = buildRosterPlayerLabels([
      roster({
        players: ["11583", "k-id"],
        player_details: [
          { player_id: "00-0036322", name: "Justin Jefferson", position: "WR" },
          { player_id: "k-id", name: "Will Reichard", position: "K" },
        ],
      }),
    ]);
    expect(labels.get("11583")).toBe("Justin Jefferson (WR)");
    expect(labels.get("k-id")).toBe("Will Reichard (K)");
  });

  it("joins by sleeper_id when details keep a canonical player_id", () => {
    const labels = buildRosterPlayerLabels([
      roster({
        players: ["11583"],
        player_details: [
          {
            player_id: "00-0036322",
            name: "Justin Jefferson",
            position: "WR",
            sleeper_id: "11583",
            gsis_id: "00-0036322",
          },
        ],
      }),
    ]);
    expect(labels.get("11583")).toBe("Justin Jefferson (WR)");
  });

  it("does not invent names when lengths match but details are reversed", () => {
    const labels = buildRosterPlayerLabels([
      roster({
        players: ["11583", "11604"],
        player_details: [
          { player_id: "11604", name: "Puka Nacua", position: "WR" },
          { player_id: "11583", name: "Justin Jefferson", position: "WR" },
        ],
      }),
    ]);
    expect(labels.get("11583")).toBe("Justin Jefferson (WR)");
    expect(labels.get("11604")).toBe("Puka Nacua (WR)");
  });

  it("falls back to raw ids when a new payload is reversed without matching aliases", () => {
    const labels = buildRosterPlayerLabels([
      roster({
        players: ["11583", "11604"],
        player_details: [
          {
            player_id: "00-0037740",
            name: "Puka Nacua",
            position: "WR",
            gsis_id: "00-0037740",
          },
          {
            player_id: "00-0036322",
            name: "Justin Jefferson",
            position: "WR",
            gsis_id: "00-0036322",
          },
        ],
      }),
    ]);
    expect(labels.get("11583")).toBe("11583");
    expect(labels.get("11604")).toBe("11604");
  });
});
