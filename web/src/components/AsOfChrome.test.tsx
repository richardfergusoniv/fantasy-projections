import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { AsOfChrome } from "./AsOfChrome";

describe("AsOfChrome", () => {
  it("renders a persistent as-of status from the published vintage", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-17T12:00:00Z"));
    render(<AsOfChrome dataAsOf="2026-09-17T08:00:00Z" runId="weekly-2026-w01" />);
    const chrome = screen.getByTestId("as-of-chrome");
    expect(chrome).toHaveAttribute("role", "status");
    expect(chrome).toHaveTextContent(/As of /);
    expect(chrome.querySelector("time")).toHaveAttribute("datetime", "2026-09-17T08:00:00.000Z");
    vi.useRealTimers();
  });

  it("shows a stale-looking unknown state when the API omitted as-of", () => {
    render(<AsOfChrome dataAsOf="" />);
    const chrome = screen.getByTestId("as-of-chrome");
    expect(chrome).toHaveTextContent(/As-of unknown/i);
    expect(chrome.className).toMatch(/is-unknown/);
    expect(chrome.querySelector("time")).toBeNull();
  });

  it("reserves height while pending instead of flashing unknown", () => {
    render(<AsOfChrome pending dataAsOf="" />);
    const chrome = screen.getByTestId("as-of-chrome");
    expect(chrome).toHaveAttribute("aria-busy", "true");
    expect(chrome.className).toMatch(/is-pending/);
    expect(chrome).not.toHaveTextContent(/As-of unknown/i);
    expect(chrome.querySelector("time")).toBeNull();
  });

  it("marks an old published stamp as stale", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-17T12:00:00Z"));
    render(<AsOfChrome dataAsOf="2026-09-16T12:00:00Z" />);
    const chrome = screen.getByTestId("as-of-chrome");
    expect(chrome).toHaveTextContent(/stale/i);
    expect(chrome.className).toMatch(/is-stale/);
    vi.useRealTimers();
  });
});
