/**
 * Hashed Vite assets under /assets/ are immutable. Workbox should CacheFirst
 * them. HTML, /api, /health, and sw.js stay off this path.
 *
 * Keep the *inline* Workbox urlPattern in `web/vite.config.ts` in sync — the
 * plugin stringifies that matcher into sw.js and will not bundle this helper.
 */
export const HASHED_ASSET_CACHE_NAME = "hashed-static-assets";

export const HASHED_ASSET_DESTINATIONS = [
  "script",
  "style",
  "font",
  "image",
  "worker",
] as const;

export function isHashedStaticAssetPath(pathname: string): boolean {
  return pathname.startsWith("/assets/");
}
