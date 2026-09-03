/**
 * URLs that must never be served from a service worker (or browser) cache.
 * Authenticated JSON and health probes are always fetched from the network.
 */
export function isUncacheableAppUrl(url: URL | string): boolean {
  const parsed = typeof url === "string" ? new URL(url, "http://localhost") : url;
  const path = parsed.pathname;
  return path.startsWith("/api/") || path.startsWith("/health/");
}
