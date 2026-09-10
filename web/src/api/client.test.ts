import { describe, expect, it, vi } from "vitest";
import { ApiClient, ApiClientError, recoveryActionForError } from "./client";

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
      message:
        "Lineup recommendation is unavailable for this league and week. (lineup_unavailable)",
      status: 400,
      code: "lineup_unavailable",
    } satisfies Partial<ApiClientError>);

    vi.unstubAllGlobals();
  });

  it("maps decision codes to recovery actions", () => {
    const err = new ApiClientError(
      "Roster players could not be linked (identity_resolution_incomplete)",
      400,
      {
        detail: {
          code: "identity_resolution_incomplete",
          message: "Roster players could not be linked",
        },
      },
    );
    expect(recoveryActionForError(err)).toContain("Run sync");
  });
});
