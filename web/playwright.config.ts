import { defineConfig, devices } from "@playwright/test";

/**
 * The browser suite runs against the real API and a real seeded database, not
 * mocks. Two servers are started: a disposable API (fresh SQLite, migrated and
 * seeded per run) and the Vite dev server, which proxies `/api` to it.
 *
 * `dev` is used in CI as well as locally, because `preview` serves the built
 * bundle without the `/api` proxy — under preview the app cannot reach the API
 * at all, which is how the suite came to test only that the shell renders.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: "list",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "mobile-chromium",
      use: { ...devices["Pixel 7"] },
    },
  ],
  webServer: [
    {
      command: "uv run python scripts/e2e_api.py --port 8000",
      cwd: "..",
      url: "http://127.0.0.1:8000/health/ready",
      reuseExistingServer: !process.env.CI,
      timeout: 180_000,
      stdout: "pipe",
      stderr: "pipe",
    },
    {
      command: "npm run dev -- --port 5173 --host 127.0.0.1",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
    },
  ],
});
