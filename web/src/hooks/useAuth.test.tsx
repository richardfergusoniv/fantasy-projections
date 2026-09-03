import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { AuthProvider } from "../hooks/useAuth";
import { ApiClientError } from "../api/client";

const getMe = vi.fn();
const verifyMagicLink = vi.fn();

let unauthorizedHandler: (() => void) | null = null;

vi.mock("../api/client", () => ({
  ApiClientError: class MockApiClientError extends Error {
    constructor(
      message: string,
      public readonly status: number,
    ) {
      super(message);
      this.name = "ApiClientError";
    }
  },
  api: {
    getMe: (...args: unknown[]) => getMe(...args),
    requestMagicLink: vi.fn(),
    verifyMagicLink: (...args: unknown[]) => verifyMagicLink(...args),
    logout: vi.fn(),
    onUnauthorized: (listener: () => void) => {
      unauthorizedHandler = listener;
      return () => {
        unauthorizedHandler = null;
      };
    },
    setCsrfToken: vi.fn(),
  },
}));

function renderApp(initialPath = "/") {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <AuthProvider>
        <App />
      </AuthProvider>
    </MemoryRouter>,
  );
}

describe("auth session lifecycle", () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
    getMe.mockReset();
    verifyMagicLink.mockReset();
    unauthorizedHandler = null;
  });

  it("restores a session from the HTTP-only cookie via /me", async () => {
    getMe.mockResolvedValue({
      id: "user-1",
      email: "owner@example.com",
      csrf_token: "csrf-from-me",
      session_expires_at: "2026-10-03T00:00:00+00:00",
    });

    renderApp("/");

    await waitFor(() => {
      expect(getMe).toHaveBeenCalled();
    });
    expect(localStorage.getItem("fantasy-decisions:csrf")).toBe("csrf-from-me");
  });

  it("verifies magic-link callbacks and clears the token from the URL", async () => {
    const replaceState = vi.fn();
    vi.stubGlobal("history", { replaceState, state: null });
    vi.stubGlobal("location", {
      href: "http://127.0.0.1:5174/auth/callback#token=link-token",
      pathname: "/auth/callback",
      search: "",
      hash: "#token=link-token",
    });

    verifyMagicLink.mockResolvedValue({ status: "ok", csrf_token: "csrf-from-verify" });
    getMe.mockResolvedValue({
      id: "user-1",
      email: "owner@example.com",
      csrf_token: "csrf-from-verify",
    });

    renderApp("/auth/callback#token=link-token");

    await waitFor(() => {
      expect(verifyMagicLink).toHaveBeenCalledWith("link-token");
    });
    expect(replaceState).toHaveBeenCalledWith(null, "", "/auth/callback");
    expect(localStorage.getItem("fantasy-decisions:csrf")).toBe("csrf-from-verify");
    vi.unstubAllGlobals();
  });

  it("shows an explicit expired-session recovery screen", async () => {
    localStorage.setItem("fantasy-decisions:csrf", "stale-csrf");
    getMe.mockResolvedValue({
      id: "user-1",
      email: "owner@example.com",
      csrf_token: "stale-csrf",
    });

    renderApp("/lineup");
    await waitFor(() => {
      expect(screen.getByRole("navigation", { name: "Primary" })).toBeInTheDocument();
    });

    getMe.mockRejectedValue(new ApiClientError("Unauthorized", 401));
    unauthorizedHandler?.();

    await waitFor(() => {
      expect(screen.getByRole("heading", { name: "Session expired" })).toBeInTheDocument();
    });
    expect(localStorage.getItem("fantasy-decisions:csrf")).toBeNull();
  });
});
