/**
 * localStorage that cannot take the app down.
 *
 * Every browser path into `window.localStorage` can throw, and the app used it
 * bare in places that run on success paths:
 *
 * - reading `window.localStorage` at all throws `SecurityError` when the
 *   browser blocks site data (Safari "Prevent cross-site tracking" in an
 *   embedded context, Chrome's third-party-cookie blocking for some origins).
 *   That happened inside a `useState` initializer, so the whole app failed to
 *   mount rather than losing one remembered preference.
 * - `setItem` throws `QuotaExceededError` once the origin's ~5 MB budget is
 *   full. The recommendation cache writes on every successful fetch, so a full
 *   quota turned a good API response into an unhandled exception — and marking
 *   a player drafted mid-draft did the same.
 *
 * Persistence here is a convenience (a remembered league, an offline copy, the
 * drafted list). Losing it should degrade the session, never end it.
 */

function backing(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function readLocal(key: string): string | null {
  try {
    return backing()?.getItem(key) ?? null;
  } catch {
    return null;
  }
}

/** Returns false when the value could not be persisted. */
export function writeLocal(key: string, value: string): boolean {
  const store = backing();
  if (!store) return false;
  try {
    store.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

export function removeLocal(key: string): boolean {
  const store = backing();
  if (!store) return false;
  try {
    store.removeItem(key);
    return true;
  } catch {
    return false;
  }
}
