import { expect, type Page } from "@playwright/test";

export const OWNER = "owner@example.com";

export async function signIn(page: Page) {
  await page.goto("/login");
  await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();

  await page.getByLabel("Email").fill(OWNER);
  await page.getByRole("button", { name: "Send magic link" }).click();

  const devLink = page.locator(".dev-link a");
  await expect(devLink).toBeVisible();
  const href = await devLink.getAttribute("href");
  expect(href).toContain("token=");

  // Production magic links put the token in the hash fragment; the login screen
  // auto-verifies when navigated there directly.
  await page.goto(href!);
  await expect(page.getByRole("navigation", { name: "Primary" })).toBeVisible();
}

export async function openMoreScreen(page: Page, label: string) {
  await page.getByRole("button", { name: "More" }).click();
  await page
    .getByRole("dialog", { name: "More screens" })
    .getByRole("button", { name: label })
    .click();
}

export async function selectLeague(page: Page, leagueId: string) {
  const select = page.locator("#shell-league-select");
  await expect(select).toBeEnabled();
  await select.selectOption(leagueId);
  await expect(select).toHaveValue(leagueId);
}
