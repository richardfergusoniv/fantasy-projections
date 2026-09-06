import { expect, test } from "@playwright/test";
import { openMoreScreen, signIn } from "./helpers";

test.describe("screen smoke (production bundle)", () => {
  test("every primary and More screen renders without error", async ({ page }) => {
    await signIn(page);

    const primary: Array<{
      link: string;
      heading: string | RegExp;
      exact?: boolean;
      exactHeading?: boolean;
    }> = [
      { link: "Home", heading: "Decisions", exactHeading: true },
      { link: "Lineup", heading: /Lineup/i },
      { link: "Waivers", heading: /Waivers/i },
      { link: "Trade", heading: /Trade Lab/i, exact: false },
      { link: "Draft", heading: /Draft assistant/i },
    ];

    for (const screen of primary) {
      await page.getByRole("link", { name: screen.link, exact: screen.exact ?? true }).click();
      await expect(
        page.getByRole("heading", { name: screen.heading, exact: screen.exactHeading }),
      ).toBeVisible();
    }

    const more: Array<{ label: string; heading: string | RegExp }> = [
      { label: "Dynasty", heading: /Dynasty/i },
      { label: "Assist", heading: /Assistant/i },
      { label: "Ops", heading: /Operations/i },
    ];

    for (const screen of more) {
      await openMoreScreen(page, screen.label);
      await expect(page.getByRole("heading", { name: screen.heading })).toBeVisible();
    }

    // Draft is primary nav; Vegas Props is the market assistant pane.
    await page.getByRole("link", { name: "Draft", exact: true }).click();
    await expect(page.getByRole("link", { name: "Regression Model" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Vegas Props" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "O-line" })).toHaveCount(0);

    await page.getByRole("link", { name: "Home", exact: true }).click();
    await page.getByRole("link", { name: "Open Vegas Props" }).click();
    await expect(page).toHaveURL(/\/draft\?pane=checklist/);
    await expect(page.getByRole("tab", { name: "Vegas Props" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });
});
