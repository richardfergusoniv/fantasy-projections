import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useHideChromeOnScroll } from "./useHideChromeOnScroll";

function setScrollY(y: number) {
  Object.defineProperty(window, "scrollY", { configurable: true, value: y });
}

describe("useHideChromeOnScroll", () => {
  beforeEach(() => {
    setScrollY(0);
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
      cb(0);
      return 1;
    });
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("stays expanded near the top of the page", () => {
    const { result } = renderHook(() => useHideChromeOnScroll());
    expect(result.current).toBe(false);

    act(() => {
      setScrollY(20);
      window.dispatchEvent(new Event("scroll"));
    });
    expect(result.current).toBe(false);
  });

  it("collapses on scroll down and expands on scroll up", () => {
    const { result } = renderHook(() => useHideChromeOnScroll());

    act(() => {
      setScrollY(120);
      window.dispatchEvent(new Event("scroll"));
    });
    expect(result.current).toBe(true);

    act(() => {
      setScrollY(90);
      window.dispatchEvent(new Event("scroll"));
    });
    expect(result.current).toBe(false);
  });

  it("ignores tiny scroll jitter below the delta threshold", () => {
    const { result } = renderHook(() =>
      useHideChromeOnScroll({ deltaThreshold: 8 }),
    );

    act(() => {
      setScrollY(100);
      window.dispatchEvent(new Event("scroll"));
    });
    expect(result.current).toBe(true);

    act(() => {
      setScrollY(96);
      window.dispatchEvent(new Event("scroll"));
    });
    expect(result.current).toBe(true);
  });
});
