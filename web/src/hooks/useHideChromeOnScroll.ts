import { useEffect, useRef, useState } from "react";

export type HideChromeOnScrollOptions = {
  /** Always show chrome when scrollY is below this (px). */
  topThreshold?: number;
  /** Minimum scroll delta (px) before toggling direction. */
  deltaThreshold?: number;
};

/**
 * Hide sticky/fixed shell chrome while the user scrolls down a long list, and
 * reveal it again on scroll up (or when near the top of the page).
 *
 * Listens to window/document scroll because Draft and other screens use page
 * scroll rather than an inner overflow container.
 */
export function useHideChromeOnScroll(
  options: HideChromeOnScrollOptions = {},
): boolean {
  const topThreshold = options.topThreshold ?? 48;
  const deltaThreshold = options.deltaThreshold ?? 8;
  const [collapsed, setCollapsed] = useState(false);
  const lastYRef = useRef(0);
  const collapsedRef = useRef(false);
  const scheduledRef = useRef(false);
  const frameRef = useRef<number | null>(null);

  useEffect(() => {
    lastYRef.current = window.scrollY;
    collapsedRef.current = false;
    scheduledRef.current = false;
    setCollapsed(false);

    const apply = (next: boolean) => {
      if (collapsedRef.current === next) return;
      collapsedRef.current = next;
      setCollapsed(next);
    };

    const update = () => {
      scheduledRef.current = false;
      frameRef.current = null;
      const y = Math.max(0, window.scrollY);
      const delta = y - lastYRef.current;
      lastYRef.current = y;

      if (y <= topThreshold) {
        apply(false);
        return;
      }
      if (delta >= deltaThreshold) {
        apply(true);
      } else if (delta <= -deltaThreshold) {
        apply(false);
      }
    };

    const onScroll = () => {
      // Use a boolean gate so sync requestAnimationFrame stubs (tests) cannot
      // leave a stale frame id that blocks later scrolls.
      if (scheduledRef.current) return;
      scheduledRef.current = true;
      frameRef.current = window.requestAnimationFrame(update);
    };

    window.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      window.removeEventListener("scroll", onScroll);
      if (frameRef.current != null) {
        window.cancelAnimationFrame(frameRef.current);
      }
    };
  }, [topThreshold, deltaThreshold]);

  return collapsed;
}
