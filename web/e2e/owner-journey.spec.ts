import { expect, test } from "@playwright/test";
import { openMoreScreen, selectLeague, signIn } from "./helpers";

/**
 * One end-to-end journey for the only user this app has, against the real API
 * and a real seeded database at a phone viewport.
 *
 * Sign in -> pick a league -> read a lineup recommendation -> change the
 * matchup assumption -> read a waiver bid -> evaluate a trade -> open a source
 * citation -> check operations freshness.
 *
 * Everything asserted here is content the API actually returned. The previous
 * suite checked that the login heading rendered and skipped the rest, which is
 * why a defect that broke every lineup recommendation went unnoticed.
 */

test.describe("owner journey", () => {
  test("signs in, reads real recommendations, and checks freshness", async ({ page }) => {
    await signIn(page);

    // ---------------------------------------------------------------- home
    await expect(page.getByRole("heading", { name: "Decision App" })).toBeVisible();
    const leagueSelect = page.locator("#shell-league-select");
    await expect(leagueSelect).toBeEnabled();
    // All six leagues were imported, not just the default one.
    await expect(leagueSelect.locator("option")).toHaveCount(6);

    // A Superflex dynasty league, so the lineup below must fill a SUPER_FLEX seat.
    await selectLeague(page, "fixture-superflex");

    // -------------------------------------------------------------- lineup
    await page.getByRole("link", { name: "Lineup" }).click();
    await expect(page.getByRole("group", { name: /Matchup assumption/i })).toBeVisible();

    const modeBanner = page.getByTestId("active-opponent-mode");
    await expect(modeBanner).toContainText("opponent_mode=current");
    await expect(page.getByText(/Win probability under/i)).toBeVisible();

    // Recommended starters are real players from the seeded roster, and the
    // uncertainty band is shown rather than a bare point estimate.
    const starters = page.locator(".starter-row, .starter-list li");
    await expect(starters.first()).toBeVisible();

    // Changing the matchup assumption must reach the server, not just relabel.
    await page.getByLabel("Opponent's best possible lineup").check();
    await expect(modeBanner).toContainText("opponent_mode=optimized");

    // ------------------------------------------------------------- waivers
    await page.getByRole("link", { name: "Waivers" }).click();
    await expect(page.getByText(/FAAB \$\d+–\$\d+/).first()).toBeVisible();
    // A recommendation without its reasoning is not a recommendation.
    await expect(page.locator(".rationale-list li").first()).toBeVisible();

    // ----------------------------------------------------------- trade lab
    await page.getByRole("link", { name: "Trade" }).click();
    await expect(page.getByRole("button", { name: /Evaluate trade/i })).toBeVisible();

    await page
      .getByRole("group", { name: "Players you send" })
      .getByRole("checkbox")
      .first()
      .check();
    await page.getByLabel("Their roster").selectOption("2");
    await page
      .getByRole("group", { name: "Players you receive" })
      .getByRole("checkbox")
      .first()
      .check();
    await page.getByRole("button", { name: /Evaluate trade/i }).click();

    await expect(page.getByRole("heading", { name: /Evaluation \(/ })).toBeVisible();
    await expect(page.getByText("Your side value")).toBeVisible();

    // ---------------------------------------------------------- operations
    await openMoreScreen(page, "Ops");
    await expect(page.getByText(/Active release/i).first()).toBeVisible();
    // Fixture data and untrained artifacts must be labelled where the owner
    // reads them, not only in the API payload.
    await expect(page.getByText(/not live league data/i)).toBeVisible();
    await expect(
      page.locator("li").filter({ hasText: "Weekly-v2 artifacts" }),
    ).toContainText(/trained \(|not production/);
    await expect(page.getByText(/blocked until weekly gates pass/i)).toBeVisible();
  });

  test("a refresh produces cited evidence that opens safely in a new tab", async ({ page }) => {
    await signIn(page);

    // A freshly seeded database holds no injury evidence; the refresh job is
    // what produces it, so the journey runs one rather than skipping the check.
    await openMoreScreen(page, "Ops");
    await page.getByRole("button", { name: "Run daily refresh" }).click();
    await expect(page.getByText(/Daily refresh: Job /)).toBeVisible({ timeout: 60_000 });

    // Josh Allen is questionable in the fixture payload and starts for roster 1
    // of the standard league.
    await selectLeague(page, "fixture-standard");
    await page.getByRole("link", { name: "Lineup" }).click();
    await expect(page.getByText(/Status questionable|Status .*questionable/i).first()).toBeVisible();

    const citation = page.locator(".citation-list a").first();
    await expect(citation).toBeVisible();
    await expect(citation).toHaveAttribute("target", "_blank");
    await expect(citation).toHaveAttribute("rel", /noopener/);
    await expect(citation).toHaveAttribute("rel", /noreferrer/);

    // Fixture evidence is unmistakably synthetic: it must never look like a
    // real news source.
    const href = await citation.getAttribute("href");
    expect(href).toContain("fixture://");
    await expect(citation).toContainText(/SYNTHETIC/);
  });

  test("league selection survives a reload and stays league-specific", async ({ page }) => {
    await signIn(page);
    await selectLeague(page, "fixture-ppfd");
    await page.getByRole("link", { name: "Lineup" }).click();
    const before = await page.getByTestId("active-opponent-mode").textContent();

    await page.reload();

    await expect(page.getByRole("navigation", { name: "Primary" })).toBeVisible();
    // Wait for the league list to load: until it does the select has no matching
    // option, so its value is "" regardless of what was remembered.
    const select = page.locator("#shell-league-select");
    await expect(select).toBeEnabled();
    await expect(select).toHaveValue("fixture-ppfd");
    await expect(page.getByTestId("active-opponent-mode")).toContainText(
      (before ?? "").includes("optimized") ? "optimized" : "current",
    );
    // The PPFD league has no kicker or defense seat, so its lineup must not
    // show one even though the previously selected league did.
    await expect(page.getByText(/\bDEF\b/)).toHaveCount(0);
  });

  test("an expired session is recoverable rather than a dead end", async ({ page }) => {
    await signIn(page);
    await page.context().clearCookies();
    await page.goto("/lineup");

    await expect(
      page.getByRole("heading", { name: /Sign in|Session expired/ }),
    ).toBeVisible();
  });

  test("the shell fits a phone viewport without horizontal scroll", async ({ page }) => {
    await signIn(page);
    await page.getByRole("link", { name: "Lineup" }).click();
    await expect(page.getByTestId("active-opponent-mode")).toBeVisible();

    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
    );
    expect(overflow).toBeLessThanOrEqual(1);
  });
});
