import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { PostAuthInstallPrompt } from "./PostAuthInstallPrompt";
import {
  POST_AUTH_INSTALL_KEY,
  clearPostAuthInstallNeeded,
  markPostAuthInstallNeeded,
  shouldShowPostAuthInstall,
} from "../pwa/postAuthInstall";

function stubIphoneSafari(standalone = false) {
  vi.stubGlobal("matchMedia", (query: string) => ({
    matches: standalone && query.includes("display-mode: standalone"),
    media: query,
  }));
  vi.stubGlobal("navigator", {
    userAgent:
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    platform: "iPhone",
    maxTouchPoints: 5,
    standalone,
  });
}

describe("postAuthInstall helpers", () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("marks install coach only on iPhone Safari browser tabs", () => {
    stubIphoneSafari(false);
    markPostAuthInstallNeeded();
    expect(sessionStorage.getItem(POST_AUTH_INSTALL_KEY)).toBe("1");
    expect(shouldShowPostAuthInstall()).toBe(true);
  });

  it("skips the coach when already standalone", () => {
    stubIphoneSafari(true);
    markPostAuthInstallNeeded();
    expect(sessionStorage.getItem(POST_AUTH_INSTALL_KEY)).toBeNull();
    expect(shouldShowPostAuthInstall()).toBe(false);
  });
});

describe("PostAuthInstallPrompt", () => {
  afterEach(() => {
    sessionStorage.clear();
  });

  it("explains magic-link Safari install and clears the coach flag", () => {
    sessionStorage.setItem(POST_AUTH_INSTALL_KEY, "1");
    const onContinue = vi.fn();
    render(<PostAuthInstallPrompt onContinueInSafari={onContinue} />);

    expect(screen.getByRole("heading", { name: /Signed in — install the app/i })).toBeInTheDocument();
    expect(screen.getByText(/magic link opens/i)).toBeInTheDocument();
    expect(screen.getAllByText(/fantasy-projections-xi\.vercel\.app/i).length).toBeGreaterThan(0);

    screen.getByRole("button", { name: /Continue in Safari for now/i }).click();
    expect(onContinue).toHaveBeenCalledOnce();
    expect(sessionStorage.getItem(POST_AUTH_INSTALL_KEY)).toBeNull();
    clearPostAuthInstallNeeded();
  });
});
