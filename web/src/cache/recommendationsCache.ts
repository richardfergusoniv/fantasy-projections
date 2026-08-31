import type {
  CachedRecommendation,
  LineupRecommendation,
  ReadonlyRecommendationKey,
  WaiverRecommendation,
} from "../api/types";

const STORAGE_KEY = "fantasy-decisions:recommendations";

/**
 * Cache schema version. Bumping it retires every previously stored entry on the
 * next read, which is how a shape change (or a key-collision fix like adding
 * `opponent_mode`) invalidates stale offline data instead of misreading it.
 */
export const CACHE_SCHEMA_VERSION = 2;

/** Most entries we keep. Two screens x two opponent modes x two leagues. */
const MAX_ENTRIES = 8;

type RecommendationPayload = LineupRecommendation | WaiverRecommendation;

interface CacheEnvelope {
  version: number;
  entries: CachedRecommendation<RecommendationPayload>[];
}

export interface RecommendationCacheSlot {
  key: ReadonlyRecommendationKey;
  leagueId: string;
  week: number;
  /** Request parameters that change the payload, e.g. `opponent_mode=current`. */
  variant?: string;
}

/**
 * Identity of one cache slot.
 *
 * `variant` and `week` are part of the identity: without them, switching
 * opponent mode (or week) reads back the other request's lineup and presents it
 * as the answer to the question the user just asked.
 */
export function recommendationCacheKey(slot: RecommendationCacheSlot): string {
  return [slot.key, slot.leagueId, String(slot.week), slot.variant ?? "default"].join("|");
}

function readStore(): CachedRecommendation<RecommendationPayload>[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw) as CacheEnvelope | unknown;
    if (
      !parsed ||
      typeof parsed !== "object" ||
      (parsed as CacheEnvelope).version !== CACHE_SCHEMA_VERSION
    ) {
      // Pre-versioned or superseded payload: drop it rather than misread it.
      localStorage.removeItem(STORAGE_KEY);
      return [];
    }
    const entries = (parsed as CacheEnvelope).entries;
    return Array.isArray(entries) ? entries : [];
  } catch {
    return [];
  }
}

function writeStore(entries: CachedRecommendation<RecommendationPayload>[]): void {
  const envelope: CacheEnvelope = { version: CACHE_SCHEMA_VERSION, entries };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(envelope));
}

/** Persist the last successful read-only recommendation for offline display. */
export function cacheRecommendation<T extends RecommendationPayload>(
  slot: RecommendationCacheSlot,
  payload: T,
): void {
  const entry: CachedRecommendation<T> = {
    key: slot.key,
    leagueId: slot.leagueId,
    week: slot.week,
    variant: slot.variant ?? "default",
    fetchedAt: new Date().toISOString(),
    payload,
  };

  const slotKey = recommendationCacheKey(slot);
  const others = readStore().filter(
    (item) => recommendationCacheKey(item) !== slotKey,
  );
  writeStore([entry, ...others].slice(0, MAX_ENTRIES));
}

export function getCachedRecommendation<T extends RecommendationPayload>(
  slot: RecommendationCacheSlot,
): CachedRecommendation<T> | null {
  const slotKey = recommendationCacheKey(slot);
  const match = readStore().find((item) => recommendationCacheKey(item) === slotKey);
  return (match as CachedRecommendation<T> | undefined) ?? null;
}

export function clearRecommendationCache(): void {
  localStorage.removeItem(STORAGE_KEY);
}

/** Guardrail: never persist auth/session material in the recommendations cache. */
export function assertNoAuthTokens(value: unknown): void {
  const serialized = JSON.stringify(value).toLowerCase();
  const forbidden = ["session", "csrf_token", "bearer", "authorization", "magic"];
  if (forbidden.some((token) => serialized.includes(token))) {
    throw new Error("Refusing to cache auth-related payload");
  }
}
