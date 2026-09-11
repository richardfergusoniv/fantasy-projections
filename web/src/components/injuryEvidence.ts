import type { InjuryEvidence } from "../api/types";

const INERT_STATUSES = new Set(["", "unknown", "healthy", "active", "none"]);
const INERT_SUMMARIES = new Set(["", "no evidence", "no evidence published"]);

/**
 * True when injury evidence is worth showing on a fantasy decision surface.
 *
 * The injury endpoint often returns a placeholder row (`status=unknown`,
 * `summary=No evidence`, empty sources) for every looked-up player. Spamming
 * that on every starter makes the lineup look like an ops console.
 */
export function isActionableInjuryEvidence(evidence: InjuryEvidence | null | undefined): boolean {
  if (!evidence) return false;
  if (evidence.sources.length > 0) return true;
  const status = evidence.status.trim().toLowerCase();
  if (INERT_STATUSES.has(status)) return false;
  const summary = evidence.summary.trim().toLowerCase();
  if (INERT_SUMMARIES.has(summary)) return false;
  return true;
}
