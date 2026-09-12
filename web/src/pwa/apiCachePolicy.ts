/**
 * URLs that must never be claimed or cached by the service worker.
 * Authenticated JSON and health probes must hit the network directly.
 *
 * Keep this logic in sync with the *inline* navigate `urlPattern` exclusion in
 * `web/vite.config.ts`. vite-plugin-pwa stringifies that matcher into `sw.js`
 * and will not bundle this helper — importing it from the Workbox config
 * caused a production ReferenceError that broke API fetches under the PWA.
 *
 * Important: do **not** register a Workbox `NetworkOnly` route for these
 * paths. NetworkOnly still wraps `fetch` and converts timeouts / dropped
 * connections into opaque `no-response` Workbox errors. Leaving them
 * unmatched lets the browser surface native network failures instead.
 */
export function isUncacheableAppUrl(url: URL | string): boolean {
  const parsed = typeof url === "string" ? new URL(url, "http://localhost") : url;
  const path = parsed.pathname;
  return path.startsWith("/api/") || path.startsWith("/health/");
}
