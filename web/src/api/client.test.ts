import { describe, expect, it, vi } from "vitest";
import { ApiClient, ApiClientError } from "./client";

describe("ApiClient error parsing", () => {
  it("reads structured detail objects without calling map", async () => {
    const client = new ApiClient({ baseUrl: "http://example.test" });
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      statusText: "Bad Request",
      json: async () => ({
        detail: {
          code: "lineup_unavailable",
          message: "Lineup recommendation is unavailable for this league and week.",
        },
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    await expect(client.getLeagues()).rejects.toMatchObject({
      message: "Lineup recommendation is unavailable for this league and week.",
      status: 400,
    } satisfies Partial<ApiClientError>);

    vi.unstubAllGlobals();
  });
});
