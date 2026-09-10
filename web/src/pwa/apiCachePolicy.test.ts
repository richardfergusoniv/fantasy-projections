import { describe, expect, it } from "vitest";
import { isUncacheableAppUrl } from "./apiCachePolicy";

describe("isUncacheableAppUrl", () => {
  it("excludes API and health paths from service-worker caching", () => {
    expect(isUncacheableAppUrl("/api/v1/leagues/1/lineup/1")).toBe(true);
    expect(isUncacheableAppUrl("/api/v1/leagues/1/waivers/1")).toBe(true);
    expect(isUncacheableAppUrl("/health/ready")).toBe(true);
    expect(isUncacheableAppUrl("/assets/index.js")).toBe(false);
    expect(isUncacheableAppUrl("/")).toBe(false);
  });
});
