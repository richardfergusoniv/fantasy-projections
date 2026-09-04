import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { IphoneInstallGuide } from "./IphoneInstallGuide";

describe("IphoneInstallGuide", () => {
  it("is hidden when already running standalone", () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query.includes("display-mode: standalone"),
      media: query,
    }));
    vi.stubGlobal("navigator", {
      userAgent:
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
      platform: "iPhone",
      maxTouchPoints: 5,
      standalone: true,
    });
    localStorage.clear();

    const { container } = render(<IphoneInstallGuide />);
    expect(container).toBeEmptyDOMElement();
    vi.unstubAllGlobals();
  });

  it("can be dismissed on iOS Safari", async () => {
    vi.stubGlobal("matchMedia", () => ({ matches: false, media: "" }));
    vi.stubGlobal("navigator", {
      userAgent:
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
      platform: "iPhone",
      maxTouchPoints: 5,
    });
    localStorage.clear();

    render(<IphoneInstallGuide />);
    expect(screen.getByText(/Install on iPhone/i)).toBeInTheDocument();
    expect(screen.getByText(/fantasy-projections-xi\.vercel\.app/i)).toBeInTheDocument();
    expect(screen.getByText(/fantasy-projections\.vercel\.app/i)).toBeInTheDocument();
    screen.getByRole("button", { name: "Dismiss" }).click();
    await waitFor(() => {
      expect(screen.queryByText(/Install on iPhone/i)).not.toBeInTheDocument();
    });
    expect(localStorage.getItem("fantasy-decisions:iphone-install-dismissed")).toBe("1");
    vi.unstubAllGlobals();
  });
});
