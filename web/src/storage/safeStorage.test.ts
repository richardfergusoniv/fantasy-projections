import { afterEach, describe, expect, it, vi } from "vitest";
import { readLocal, removeLocal, writeLocal } from "./safeStorage";
import { cacheRecommendation, getCachedRecommendation } from "../cache/recommendationsCache";
import type { LineupRecommendation } from "../api/types";

function breakStorage(error: Error) {
  return vi.spyOn(window, "localStorage", "get").mockImplementation(() => {
    throw error;
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
});

describe("safeStorage", () => {
  it("round-trips values when storage works", () => {
    expect(writeLocal("k", "v")).toBe(true);
    expect(readLocal("k")).toBe("v");
    expect(removeLocal("k")).toBe(true);
    expect(readLocal("k")).toBeNull();
  });

  it("survives a browser that blocks site data entirely", () => {
    // Safari/Chrome throw SecurityError from the localStorage getter itself.
    breakStorage(new DOMException("denied", "SecurityError"));
    expect(readLocal("k")).toBeNull();
    expect(writeLocal("k", "v")).toBe(false);
    expect(removeLocal("k")).toBe(false);
  });

  it("reports a failed write instead of throwing when the quota is full", () => {
    const setItem = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(() => {
        throw new DOMException("full", "QuotaExceededError");
      });
    expect(writeLocal("k", "v")).toBe(false);
    expect(setItem).toHaveBeenCalled();
  });
});

describe("recommendation cache under storage failure", () => {
  const payload = {
    league_id: "l1",
    week: 1,
    win_probability: 0.5,
    points: { mean: 100, p10: 90, p90: 110 },
    starters: [],
    bench: [],
    swaps: [],
    meta: { data_as_of: "2026-09-01", projection_run_id: "run-1" },
  } as unknown as LineupRecommendation;

  it("does not turn a full quota into a failed fetch", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new DOMException("full", "QuotaExceededError");
    });
    // Caching runs on the success path of every read-only recommendation, so a
    // throw here surfaced as a crash right after a good API response.
    expect(() =>
      cacheRecommendation({ key: "lineup", leagueId: "l1", week: 1 }, payload),
    ).not.toThrow();
    expect(getCachedRecommendation({ key: "lineup", leagueId: "l1", week: 1 })).toBeNull();
  });
});
