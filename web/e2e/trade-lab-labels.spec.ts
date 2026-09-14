import { expect, test } from "@playwright/test";
import { signIn } from "./helpers";

/**
 * Production Trade Lab listed Sleeper ids because `player_details.player_id`
 * was remapped to GSIS while `players` kept the Sleeper id. This intercepts
 * that payload so the checklist must join names by index/alias, not id equality.
 */
const REMAPPED_ROSTERS = {
  rosters: [
    {
      roster_id: 1,
      week: 1,
      manager_name: "rdfergus15",
      players: ["11583", "11604", "k-id"],
      starters: ["11583"],
      reserve: [],
      player_details: [
        { player_id: "00-0036322", name: "Justin Jefferson", position: "WR" },
        { player_id: "00-0037740", name: "Puka Nacua", position: "WR" },
        { player_id: "k-id", name: "Will Reichard", position: "K" },
      ],
    },
    {
      roster_id: 2,
      week: 1,
      manager_name: "kdicillo",
      players: ["10232", "11632"],
      starters: ["10232"],
      reserve: [],
      player_details: [
        { player_id: "00-0031288", name: "Tyreek Hill", position: "WR" },
        { player_id: "00-0037746", name: "Garrett Wilson", position: "WR" },
      ],
    },
  ],
};

test.describe("Trade Lab player labels", () => {
  test("renders Name (POS) when roster ids and player_details ids differ", async ({
    page,
  }) => {
    await page.route("**/api/v1/leagues/**/rosters", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(REMAPPED_ROSTERS),
      });
    });

    await signIn(page);
    await page.getByRole("link", { name: "Trade", exact: true }).click();
    await expect(page.getByRole("heading", { name: /Trade Lab/i })).toBeVisible();

    await expect(page.getByLabel("Justin Jefferson (WR)")).toBeVisible();
    await expect(page.getByLabel("Puka Nacua (WR)")).toBeVisible();
    await expect(page.getByLabel("Will Reichard (K)")).toBeVisible();
    await expect(page.getByLabel("Tyreek Hill (WR)")).toBeVisible();
    await expect(page.getByLabel("Garrett Wilson (WR)")).toBeVisible();

    await expect(page.getByLabel("11583")).toHaveCount(0);
    await expect(page.getByLabel("10232")).toHaveCount(0);
  });
});
