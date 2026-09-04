import { expect, test } from "@playwright/test";
import { openMoreScreen, signIn } from "./helpers";

test.describe("screen smoke (production bundle)", () => {
  test("every primary and More screen renders without error", async ({ page }) => {
    await signIn(page);

    const primary: Array<{ link: string; heading: string | RegExp; exact?: boolean }> = [
      { link: "Home", heading: "Decision App" },
      { link: "Lineup", heading: /Lineup/i },
      { link: "Waivers", heading: /Waivers/i },
      { link: "Trade", heading: /Trade Lab/i, exact: false },
      { link: "Draft", heading: /Draft assistant/i },
    ];

    for (const screen of primary) {
      await page.getByRole("link", { name: screen.link, exact: screen.exact ?? true }).click();
      await expect(page.getByRole("heading", { name: screen.heading })).toBeVisible();
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

    // Draft is primary nav; checklist is the market assistant pane.
    await page.getByRole("link", { name: "Draft", exact: true }).click();
    await expect(page.getByRole("tab", { name: "Our Rankings" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Draft Checklist" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "O-line" })).toBeVisible();

    await page.getByRole("link", { name: "Home", exact: true }).click();
    await page.getByRole("link", { name: "Open Draft Checklist" }).click();
    await expect(page).toHaveURL(/\/draft\?pane=checklist/);
    await expect(page.getByRole("tab", { name: "Draft Checklist" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });
});
