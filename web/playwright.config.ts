import { defineConfig, devices } from "@playwright/test";

/**
 * The browser suite runs against the real API and a real seeded database, not
 * mocks. Two servers are started: a disposable API (fresh SQLite, migrated and
 * seeded per run) and vite preview serving the production bundle from `dist/`,
 * which proxies `/api` to the disposable API via `preview.proxy` in vite.config.
 */
/**
 * E2E ports are isolated from the phone-access stack (5173/8002) so tests can
 * run while production preview is up.
 */
const E2E_API_PORT = process.env.E2E_API_PORT ?? "8765";
const E2E_WEB_PORT = process.env.E2E_WEB_PORT ?? "5174";

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
    baseURL: `http://127.0.0.1:${E2E_WEB_PORT}`,
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
      command: `uv run python scripts/e2e_api.py --port ${E2E_API_PORT}`,
      cwd: "..",
      url: `http://127.0.0.1:${E2E_API_PORT}/health/ready`,
      env: { E2E_WEB_PORT },
      reuseExistingServer: false,
      timeout: 180_000,
      stdout: "pipe",
      stderr: "pipe",
    },
    {
      command: `npm run build && npm run preview -- --host 127.0.0.1 --port ${E2E_WEB_PORT}`,
      env: { API_PROXY_PORT: E2E_API_PORT },
      url: `http://127.0.0.1:${E2E_WEB_PORT}`,
      reuseExistingServer: false,
      timeout: 180_000,
    },
  ],
});
