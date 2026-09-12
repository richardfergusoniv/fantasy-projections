/**
 * In-season League Value ↔ Vegas Props preference.
 *
 * Mirrors APP_PROJECTION_SOURCE for decision APIs. Home and Matchup share this
 * seam so one toggle drives lineup / matchup fetches.
 */
import type { LineupBoardSource } from "./api/types";
import { readLocal, writeLocal } from "./storage/safeStorage";

export const BOARD_SOURCE_KEY = "fantasy-decisions:board-source";

/** API values accepted by `?projection_source=` (same as APP_PROJECTION_SOURCE). */
export type ProjectionSourceParam =
  | "sealed_release"
  | "status_adjusted_release"
  | "weekly_props";

export function isBoardSource(value: string | null | undefined): value is LineupBoardSource {
  return value === "league_value" || value === "vegas_props";
}

export function readBoardSourcePreference(): LineupBoardSource | null {
  const raw = readLocal(BOARD_SOURCE_KEY);
  return isBoardSource(raw) ? raw : null;
}

export function writeBoardSourcePreference(source: LineupBoardSource): void {
  writeLocal(BOARD_SOURCE_KEY, source);
}

/** Map UI board → API projection_source query value. */
export function boardSourceToProjectionSource(
  board: LineupBoardSource,
): ProjectionSourceParam {
  return board === "vegas_props" ? "weekly_props" : "sealed_release";
}

export function labelForBoardSource(board: LineupBoardSource): string {
  return board === "vegas_props" ? "Vegas Props" : "League Value";
}
