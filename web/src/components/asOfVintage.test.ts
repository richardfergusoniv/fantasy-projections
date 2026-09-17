import { afterEach, describe, expect, it, vi } from "vitest";
import { describeAsOf, pickAsOfStamp, readPublishedAsOf } from "./asOfVintage";

describe("readPublishedAsOf", () => {
  it("returns a published ISO string and refuses to invent one", () => {
    expect(readPublishedAsOf("2026-08-30T17:00:00Z")).toBe("2026-08-30T17:00:00Z");
    expect(readPublishedAsOf("  ")).toBeNull();
    expect(readPublishedAsOf(null)).toBeNull();
    expect(readPublishedAsOf(undefined)).toBeNull();
    expect(readPublishedAsOf("not-a-date")).toBeNull();
  });
});

describe("pickAsOfStamp", () => {
  it("skips empty candidates instead of falling back to now()", () => {
    expect(pickAsOfStamp("", null, "2026-09-01T17:00:00Z")).toBe("2026-09-01T17:00:00Z");
    expect(pickAsOfStamp("", undefined, "   ")).toBeNull();
  });
});

describe("describeAsOf", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("labels a missing stamp as unknown instead of fabricating now()", () => {
    const described = describeAsOf(null, Date.parse("2026-09-17T12:00:00Z"));
    expect(described.kind).toBe("unknown");
    expect(described.label).toMatch(/as-of unknown/i);
    expect(described.iso).toBeNull();
  });

  it("marks a stamp older than 12 hours as stale", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-17T12:00:00Z"));
    const fresh = describeAsOf("2026-09-17T08:00:00Z");
    const stale = describeAsOf("2026-09-16T12:00:00Z");
    expect(fresh.kind).toBe("ok");
    expect(fresh.label).toMatch(/^As of /);
    expect(stale.kind).toBe("stale");
    expect(stale.label).toMatch(/stale/i);
  });
});
