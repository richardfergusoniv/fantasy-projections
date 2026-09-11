import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { OperationsStatus } from "../api/types";

export interface UseOperationsStatusOptions {
  /**
   * When false, skip the network call. Home defers ops so lineup/waivers
   * claim the first-paint network slot before the storage-probing status route.
   */
  enabled?: boolean;
}

export function useOperationsStatus(options: UseOperationsStatusOptions = {}) {
  const { enabled = true } = options;
  const [data, setData] = useState<OperationsStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const status = await api.getOperationsStatus();
      setData(status);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load operations status");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    void refresh();
  }, [enabled, refresh]);

  return { data, loading, error, refresh, enabled };
}
