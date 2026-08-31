import { expect, test } from "@playwright/test";

test.describe("PWA shell", () => {
  test("login screen renders", async ({ page }) => {
    await page.goto("/login");
    await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();
    await expect(page.getByLabel("Email")).toBeVisible();
    await expect(page.getByRole("button", { name: "Send magic link" })).toBeVisible();
  });
});
