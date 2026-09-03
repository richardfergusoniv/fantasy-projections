import { expect, test } from "@playwright/test";
import { signIn } from "./helpers";

const CANONICAL_ORIGIN = "https://fantasy-projections-xi.vercel.app";

test.describe("PWA auth and manifest (production build)", () => {
  test("manifest uses standalone mode and canonical production URL", async ({ page }) => {
    const manifestHref = await page.goto("/");
    expect(manifestHref?.ok()).toBeTruthy();

    const link = page.locator('link[rel="manifest"]');
    await expect(link).toHaveAttribute("href", /manifest\.webmanifest$/);
    const href = await link.getAttribute("href");
    const response = await page.request.get(href!);
    expect(response.ok()).toBeTruthy();
    const manifest = (await response.json()) as Record<string, unknown>;

    expect(manifest.display).toBe("standalone");
    expect(manifest.name).toBe("Fantasy Decisions");
    expect(manifest.theme_color).toBe("#0f1419");
    expect(manifest.background_color).toBe("#0f1419");
    expect(manifest.start_url).toBe(`${CANONICAL_ORIGIN}/`);
    expect(manifest.scope).toBe(`${CANONICAL_ORIGIN}/`);

    const icons = manifest.icons as Array<{ sizes: string; purpose?: string }>;
    expect(icons.some((icon) => icon.sizes === "192x192")).toBe(true);
    expect(icons.some((icon) => icon.sizes === "512x512" && icon.purpose === "maskable")).toBe(true);
  });

  test("index.html advertises the apple touch icon", async ({ page }) => {
    await page.goto("/login");
    await expect(page.locator('link[rel="apple-touch-icon"]')).toHaveAttribute(
      "href",
      "/apple-touch-icon.png",
    );
  });

  test("session survives a cold reload after sign-in", async ({ page, context }) => {
    await signIn(page);
    await expect(page.getByRole("navigation", { name: "Primary" })).toBeVisible();

    await page.reload();
    await expect(page.getByRole("navigation", { name: "Primary" })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Sign in|Session expired/ })).toHaveCount(0);

    await context.clearCookies();
    await page.goto("/lineup");
    await expect(page.getByRole("heading", { name: /Sign in|Session expired/ })).toBeVisible();
  });

  test("magic-link callback lands in the installed app shell", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel("Email").fill("owner@example.com");
    await page.getByRole("button", { name: "Send magic link" }).click();
    const devLink = page.locator(".dev-link a");
    await expect(devLink).toBeVisible();
    const href = await devLink.getAttribute("href");
    expect(href).toContain("/auth/callback");

    await page.goto(href!);
    await expect(page).toHaveURL(/\/(auth\/callback)?$/);
    await expect(page.getByRole("navigation", { name: "Primary" })).toBeVisible();
    expect(page.url()).not.toContain("token=");
  });

  test("service worker precache excludes authenticated API paths", async ({ page }) => {
    await page.goto("/login");
    const swUrl = await page.evaluate(async () => {
      const registration = await navigator.serviceWorker.getRegistration();
      if (!registration?.active?.scriptURL) {
        await navigator.serviceWorker.ready;
      }
      return navigator.serviceWorker.getRegistration().then((reg) => reg?.active?.scriptURL);
    });
    expect(swUrl).toBeTruthy();
    const response = await page.request.get(swUrl!);
    const body = await response.text();
    expect(body).not.toMatch(/\/api\/v1\/(?!auth)/);
    expect(body).toContain("NetworkOnly");
  });
});
