import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { InjuryEvidence } from "../api/types";

/** Upper bound on parallel evidence lookups so a deep roster can't fan out. */
const MAX_LOOKUPS = 8;

export interface InjuryEvidenceState {
  byPlayerId: Record<string, InjuryEvidence>;
  loading: boolean;
  /** Players whose evidence lookup failed, so the UI can say so per player. */
  failed: string[];
}

/**
 * Fetch injury evidence for the players a decision actually hinges on.
 *
 * Failures are per player and non-fatal: one unavailable citation must not blank
 * out the recommendation it annotates.
 */
export function useInjuryEvidence(playerIds: string[]): InjuryEvidenceState {
  const key = useMemo(
    () => [...new Set(playerIds.filter(Boolean))].slice(0, MAX_LOOKUPS).sort().join(","),
    [playerIds],
  );
  const [state, setState] = useState<InjuryEvidenceState>({
    byPlayerId: {},
    loading: false,
    failed: [],
  });

  useEffect(() => {
    const ids = key ? key.split(",") : [];
    if (ids.length === 0) {
      setState({ byPlayerId: {}, loading: false, failed: [] });
      return;
    }
    let cancelled = false;
    setState((prev) => ({ ...prev, loading: true }));
    void Promise.all(
      ids.map((id) =>
        api
          .getInjuryEvidence(id)
          .then((evidence) => ({ id, evidence }))
          .catch(() => ({ id, evidence: null as InjuryEvidence | null })),
      ),
    ).then((results) => {
      if (cancelled) return;
      const byPlayerId: Record<string, InjuryEvidence> = {};
      const failed: string[] = [];
      for (const result of results) {
        if (result.evidence) {
          byPlayerId[result.id] = result.evidence;
        } else {
          failed.push(result.id);
        }
      }
      setState({ byPlayerId, loading: false, failed });
    });
    return () => {
      cancelled = true;
    };
  }, [key]);

  return state;
}
