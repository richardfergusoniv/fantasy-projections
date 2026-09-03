/** Read the one-time magic-link token from hash (preferred) or query. */
export function readMagicLinkToken(location: { search: string; hash: string } = window.location): string | null {
  const hashRaw = location.hash.startsWith("#") ? location.hash.slice(1) : location.hash;
  const hashToken = new URLSearchParams(hashRaw).get("token");
  if (hashToken) return hashToken;
  return new URLSearchParams(location.search).get("token");
}

/**
 * Remove the consumed token from the address bar so a reload or PWA reopen
 * cannot replay it (or skip session restore because a token is still present).
 */
export function stripMagicLinkTokenFromUrl(): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  url.searchParams.delete("token");
  const hashParams = new URLSearchParams(url.hash.startsWith("#") ? url.hash.slice(1) : url.hash);
  hashParams.delete("token");
  const remainder = hashParams.toString();
  const next = `${url.pathname}${url.search}${remainder ? `#${remainder}` : ""}`;
  window.history.replaceState(window.history.state, "", next || "/");
}
