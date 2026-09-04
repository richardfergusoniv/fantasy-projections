/**
 * URLs that must never be served from a service worker (or browser) cache.
 * Authenticated JSON and health probes are always fetched from the network.
 *
 * Keep this logic in sync with the *inline* `urlPattern` in `web/vite.config.ts`.
 * vite-plugin-pwa stringifies that matcher into `sw.js` and will not bundle this
 * helper — importing it from the Workbox config caused a production
 * ReferenceError that broke API fetches under the installed PWA.
 */
export function isUncacheableAppUrl(url: URL | string): boolean {
  const parsed = typeof url === "string" ? new URL(url, "http://localhost") : url;
  const path = parsed.pathname;
  return path.startsWith("/api/") || path.startsWith("/health/");
}
