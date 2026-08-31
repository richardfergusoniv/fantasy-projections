import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { OperationsStatus } from "../api/types";

export function useOperationsStatus() {
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
    void refresh();
  }, [refresh]);

  return { data, loading, error, refresh };
}
