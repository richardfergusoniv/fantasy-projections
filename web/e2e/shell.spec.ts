import { expect, test } from "@playwright/test";

test.describe("PWA shell", () => {
  test("login screen renders", async ({ page }) => {
    await page.goto("/login");
    await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();
    await expect(page.getByLabel("Email")).toBeVisible();
    await expect(page.getByRole("button", { name: "Send magic link" })).toBeVisible();
  });

  test("cold start does not fetch third-party webfonts", async ({ page }) => {
    const fontRequests: string[] = [];
    page.on("request", (request) => {
      const url = request.url();
      if (url.includes("fonts.googleapis.com") || url.includes("fonts.gstatic.com")) {
        fontRequests.push(url);
      }
    });
    await page.goto("/login");
    await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();
    expect(fontRequests).toEqual([]);
  });

  test("login fits a 320px viewport without horizontal overflow", async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 568 });
    await page.goto("/login");
    await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();
    const overflowed = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    );
    expect(overflowed).toBe(false);
  });
});
