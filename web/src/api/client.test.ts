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

  it("steers weekly_props gaps toward League Value", () => {
    const err = new ApiClientError(
      "Weekly Vegas props are not promoted (weekly_props_unavailable)",
      422,
      {
        detail: {
          code: "weekly_props_unavailable",
          message: "Weekly Vegas props are not promoted for this week.",
        },
      },
    );
    expect(recoveryActionForError(err)).toContain("League Value");
  });

  it("rewrites Workbox no-response fetch failures", async () => {
    const { formatFetchFailure } = await import("./client");
    expect(
      formatFetchFailure(
        new Error(
          'FetchEvent.respondWith received an error: no-response: no-response :: [{"url":"https://example.test/api/v1/leagues/1/lineup/1"}]',
        ),
        "/leagues/1/lineup/1",
      ),
    ).toMatch(/service worker got no response/i);

    const client = new ApiClient({ baseUrl: "http://example.test" });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(
        new Error("no-response :: [{\"url\":\"http://example.test/api/v1/leagues\"}]"),
      ),
    );
    await expect(client.getLeagues()).rejects.toMatchObject({
      name: "ApiClientError",
      status: 0,
      message: expect.stringMatching(/service worker got no response/i),
    });
    vi.unstubAllGlobals();
  });

  it("steers historical membership gaps away from SLEEPER_USER_ID blame", () => {
    const err = new ApiClientError(
      "This league has no synced memberships (league_membership_unavailable)",
      400,
      {
        detail: {
          code: "league_membership_unavailable",
          message: "This league has no synced memberships",
        },
      },
    );
    expect(recoveryActionForError(err)).toContain("2026");
    expect(recoveryActionForError(err)).not.toContain("SLEEPER_USER_ID");
  });
});
