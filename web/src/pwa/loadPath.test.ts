import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  HASHED_ASSET_CACHE_NAME,
  HASHED_ASSET_DESTINATIONS,
  isHashedStaticAssetPath,
} from "./staticAssetCache";

const webRoot = process.cwd();
const repoRoot = path.resolve(webRoot, "..");

function readWeb(relative: string): string {
  return readFileSync(path.join(webRoot, relative), "utf8");
}

describe("hashed static asset cache policy", () => {
  it("treats versioned /assets paths as CacheFirst candidates", () => {
    expect(isHashedStaticAssetPath("/assets/index-abc123.js")).toBe(true);
    expect(isHashedStaticAssetPath("/assets/index-abc123.css")).toBe(true);
    expect(HASHED_ASSET_DESTINATIONS).toEqual(
      expect.arrayContaining(["script", "style", "font", "image"]),
    );
    expect(HASHED_ASSET_CACHE_NAME).toMatch(/hashed/);
  });

  it("does not treat HTML, API, or the service worker as hashed assets", () => {
    expect(isHashedStaticAssetPath("/")).toBe(false);
    expect(isHashedStaticAssetPath("/login")).toBe(false);
    expect(isHashedStaticAssetPath("/index.html")).toBe(false);
    expect(isHashedStaticAssetPath("/sw.js")).toBe(false);
    expect(isHashedStaticAssetPath("/api/v1/me")).toBe(false);
    expect(isHashedStaticAssetPath("/health/ready")).toBe(false);
  });
});

describe("cold-start document and font loading", () => {
  it("does not pull third-party Google Fonts on first paint", () => {
    const html = readWeb("index.html");
    expect(html).not.toMatch(/fonts\.googleapis\.com/);
    expect(html).not.toMatch(/fonts\.gstatic\.com/);
    expect(html).not.toMatch(/onload="this\.media='all'"/);
  });

  it("inlines a tiny critical paint so the shell is not white before CSS", () => {
    const html = readWeb("index.html");
    expect(html).toMatch(/<style>[\s\S]*background:\s*#0b1219/i);
    expect(html).toMatch(/color-scheme:\s*dark/);
  });

  it("uses the platform font stack instead of a webfont family", () => {
    const css = readWeb("src/index.css");
    expect(css).not.toMatch(/Manrope/);
    expect(css).not.toMatch(/IBM Plex Mono/);
    expect(css).toMatch(/--font:\s*system-ui/);
    expect(css).toMatch(/--font-sans:\s*var\(--font\)/);
    expect(css).toMatch(/--font-mono:\s*ui-monospace/);
  });
});

describe("HTTP cache headers at the edge", () => {
  it("immutably caches hashed /assets and revalidates the HTML/SW shell", () => {
    const vercel = JSON.parse(readFileSync(path.join(repoRoot, "vercel.json"), "utf8")) as {
      headers?: Array<{ source: string; headers: Array<{ key: string; value: string }> }>;
    };
    expect(vercel.headers, "vercel.json needs Cache-Control rules").toBeDefined();

    const bySource = new Map(
      (vercel.headers ?? []).map((rule) => [
        rule.source,
        rule.headers.find((header) => header.key === "Cache-Control")?.value ?? "",
      ]),
    );

    expect(bySource.get("/assets/(.*)")).toBe("public, max-age=31536000, immutable");
    expect(bySource.get("/sw.js")).toMatch(/max-age=0/);
    expect(bySource.get("/sw.js")).not.toMatch(/no-store/);
    expect(bySource.get("/manifest.webmanifest")).toMatch(/max-age=0/);
    expect(bySource.get("/((?!assets/).*)")).toMatch(/max-age=0/);
    expect(bySource.get("/((?!assets/).*)")).not.toMatch(/no-store/);
    expect(bySource.get("/(.*)")).toBeUndefined();
  });

  it("applies later matching header rules last, without stripping /assets immutable", () => {
    const vercel = JSON.parse(readFileSync(path.join(repoRoot, "vercel.json"), "utf8")) as {
      headers?: Array<{ source: string; headers: Array<{ key: string; value: string }> }>;
    };
    const rules = vercel.headers ?? [];

    function sourceMatches(source: string, pathname: string): boolean {
      return new RegExp(`^${source}$`).test(pathname);
    }

    function effectiveCacheControl(pathname: string): string | undefined {
      let value: string | undefined;
      for (const rule of rules) {
        if (!sourceMatches(rule.source, pathname)) continue;
        const cacheControl = rule.headers.find((header) => header.key === "Cache-Control");
        if (cacheControl) value = cacheControl.value;
      }
      return value;
    }

    expect(effectiveCacheControl("/assets/index-abc123.js")).toBe(
      "public, max-age=31536000, immutable",
    );
    expect(effectiveCacheControl("/assets/index-abc123.css")).toMatch(/immutable/);
    expect(effectiveCacheControl("/")).toMatch(/max-age=0/);
    expect(effectiveCacheControl("/")).not.toMatch(/no-store/);
    expect(effectiveCacheControl("/login")).toMatch(/max-age=0/);
    expect(effectiveCacheControl("/sw.js")).toMatch(/max-age=0/);
    expect(effectiveCacheControl("/manifest.webmanifest")).toMatch(/max-age=0/);
  });
});

describe("Workbox runtime routes stay aligned with the load-path helpers", () => {
  it("CacheFirsts hashed assets and keeps HTML NetworkOnly", () => {
    const viteConfig = readWeb("vite.config.ts");
    expect(viteConfig).toMatch(/handler:\s*"CacheFirst"/);
    expect(viteConfig).toMatch(/pathname\.startsWith\("\/assets\/"\)/);
    expect(viteConfig).toMatch(/cacheName:\s*"hashed-static-assets"/);
    expect(viteConfig).toMatch(/handler:\s*"NetworkOnly"/);
    expect(viteConfig).not.toMatch(/handler:\s*"NetworkFirst"/);
  });
});

describe("responsive shell uses container queries, not only viewport breakpoints", () => {
  it("declares an app-shell container and fluid type", () => {
    const css = readWeb("src/index.css");
    expect(css).toMatch(/container-type:\s*inline-size/);
    expect(css).toMatch(/container-name:\s*app-shell/);
    expect(css).toMatch(/@container\s+app-shell/);
    expect(css).toMatch(/@media\s*\(min-width:\s*640px\)[\s\S]{0,180}\.stack \.btn-primary/);
    expect(css).toMatch(/--text-md:\s*clamp\(/);
    expect(css).not.toMatch(/will-change:\s*transform/);
  });
});
