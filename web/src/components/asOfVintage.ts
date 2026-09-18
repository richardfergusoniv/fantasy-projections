import { STALE_AFTER_MS } from "./AsyncState";

/**
 * Board vintage from a published API field. Empty / unparseable values stay
 * unknown — never substitute `new Date()`.
 */
export function readPublishedAsOf(value: unknown): string | null {
  if (value == null) return null;
  const text = String(value).trim();
  if (!text) return null;
  const parsed = Date.parse(text);
  if (Number.isNaN(parsed)) return null;
  return text;
}

export function pickAsOfStamp(
  ...candidates: Array<string | null | undefined>
): string | null {
  for (const candidate of candidates) {
    const published = readPublishedAsOf(candidate);
    if (published) return published;
  }
  return null;
}

export type AsOfKind = "ok" | "stale" | "unknown" | "pending";

export interface AsOfDescription {
  kind: AsOfKind;
  label: string;
  iso: string | null;
  display: string | null;
}

export function describeAsOf(
  value: string | null | undefined,
  nowMs: number = Date.now(),
): AsOfDescription {
  const published = readPublishedAsOf(value);
  if (!published) {
    return { kind: "unknown", label: "As-of unknown", iso: null, display: null };
  }
  const date = new Date(published);
  const iso = date.toISOString();
  const display = date.toLocaleString();
  const age = nowMs - date.getTime();
  if (Number.isFinite(age) && age > STALE_AFTER_MS) {
    return { kind: "stale", label: `As of ${display} · stale`, iso, display };
  }
  return { kind: "ok", label: `As of ${display}`, iso, display };
}
