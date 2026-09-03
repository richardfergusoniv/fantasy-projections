import { describe, expect, it, vi } from "vitest";
import { isUncacheableAppUrl } from "./apiCachePolicy";
import {
  CANONICAL_PRODUCTION_ORIGIN,
  PWA_APP_NAME,
  PWA_BACKGROUND_COLOR,
  PWA_THEME_COLOR,
} from "./canonical";
import { isIosDevice, isIosSafari, isStandaloneDisplay } from "./displayMode";
import { readMagicLinkToken, stripMagicLinkTokenFromUrl } from "./magicLink";

describe("PWA canonical config", () => {
  it("pins the adopted production origin", () => {
    expect(CANONICAL_PRODUCTION_ORIGIN).toBe("https://fantasy-projections-xi.vercel.app");
    expect(PWA_APP_NAME).toBe("Fantasy Decisions");
    expect(PWA_THEME_COLOR).toBe("#0f1419");
    expect(PWA_BACKGROUND_COLOR).toBe("#0f1419");
  });
});

describe("service worker cache policy", () => {
  it("never caches API or health responses", () => {
    expect(isUncacheableAppUrl("http://127.0.0.1:5174/api/v1/me")).toBe(true);
    expect(isUncacheableAppUrl("http://127.0.0.1:5174/api/v1/leagues/fixture/lineup/1")).toBe(true);
    expect(isUncacheableAppUrl("http://127.0.0.1:5174/health/ready")).toBe(true);
    expect(isUncacheableAppUrl("http://127.0.0.1:5174/assets/index.js")).toBe(false);
    expect(isUncacheableAppUrl("http://127.0.0.1:5174/login")).toBe(false);
  });
});

describe("magic link callback helpers", () => {
  it("prefers hash tokens over query tokens", () => {
    const token = readMagicLinkToken({
      search: "?token=query-token",
      hash: "#token=hash-token",
    });
    expect(token).toBe("hash-token");
  });

  it("strips consumed tokens from the URL bar", () => {
    const replaceState = vi.fn();
    vi.stubGlobal("history", { replaceState, state: null });
    vi.stubGlobal("location", {
      href: "https://fantasy-projections-xi.vercel.app/auth/callback#token=abc123",
      pathname: "/auth/callback",
      search: "",
      hash: "#token=abc123",
    });

    stripMagicLinkTokenFromUrl();

    expect(replaceState).toHaveBeenCalledWith(null, "", "/auth/callback");
    vi.unstubAllGlobals();
  });
});

describe("iPhone install detection", () => {
  it("detects standalone display mode", () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query.includes("display-mode: standalone"),
      media: query,
    }));
    vi.stubGlobal("navigator", { standalone: false });

    expect(isStandaloneDisplay()).toBe(true);
    vi.unstubAllGlobals();
  });

  it("detects iOS Safari vs other iOS browsers", () => {
    vi.stubGlobal("navigator", {
      userAgent:
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
      platform: "iPhone",
      maxTouchPoints: 5,
    });
    expect(isIosDevice()).toBe(true);
    expect(isIosSafari()).toBe(true);

    vi.stubGlobal("navigator", {
      userAgent:
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/120.0.6099.119 Mobile/15E148 Safari/604.1",
      platform: "iPhone",
      maxTouchPoints: 5,
    });
    expect(isIosSafari()).toBe(false);
    vi.unstubAllGlobals();
  });
});
