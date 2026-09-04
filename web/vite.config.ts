/// <reference types="vitest/config" />
import { execSync } from "node:child_process";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import {
  CANONICAL_PRODUCTION_ORIGIN,
  PWA_APP_NAME,
  PWA_BACKGROUND_COLOR,
  PWA_THEME_COLOR,
} from "./src/pwa/canonical";

function resolveAppBuildId(): string {
  try {
    return execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim();
  } catch {
    return "dev";
  }
}

const APP_BUILD_ID = resolveAppBuildId();

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.svg", "apple-touch-icon.png"],
      manifest: {
        id: `${CANONICAL_PRODUCTION_ORIGIN}/`,
        name: PWA_APP_NAME,
        short_name: "Fantasy",
        description: "Private fantasy football decision assistant",
        theme_color: PWA_THEME_COLOR,
        background_color: PWA_BACKGROUND_COLOR,
        display: "standalone",
        orientation: "portrait-primary",
        start_url: `${CANONICAL_PRODUCTION_ORIGIN}/`,
        scope: `${CANONICAL_PRODUCTION_ORIGIN}/`,
        lang: "en-US",
        icons: [
          {
            src: "/pwa-192x192.png",
            sizes: "192x192",
            type: "image/png",
          },
          {
            src: "/pwa-512x512.png",
            sizes: "512x512",
            type: "image/png",
          },
          {
            src: "/pwa-192x192-maskable.png",
            sizes: "192x192",
            type: "image/png",
            purpose: "maskable",
          },
          {
            src: "/pwa-512x512-maskable.png",
            sizes: "512x512",
            type: "image/png",
            purpose: "maskable",
          },
        ],
      },
      workbox: {
        // Precache hashed assets only. Binding navigations to a precached
        // index.html freezes installed PWAs on an old shell until the SW file
        // itself changes — NetworkFirst HTML picks up new script hashes on open.
        globPatterns: ["**/*.{js,css,ico,png,svg,woff2}"],
        // Override vite-plugin-pwa's default "index.html" NavigationRoute.
        // Empty string disables createHandlerBoundToURL so NetworkFirst can win.
        navigateFallback: "",
        // Keep matchers self-contained: vite-plugin-pwa stringifies urlPattern
        // into sw.js and will not bundle imported helpers.
        runtimeCaching: [
          {
            urlPattern: ({ request, url }: { request: Request; url: URL }) =>
              request.mode === "navigate" &&
              !url.pathname.startsWith("/api/") &&
              !url.pathname.startsWith("/health/"),
            handler: "NetworkFirst",
            options: {
              cacheName: "html-navigations",
              networkTimeoutSeconds: 3,
              expiration: {
                maxEntries: 8,
                maxAgeSeconds: 60 * 60 * 24,
              },
            },
          },
          {
            urlPattern: ({ url }: { url: URL }) => {
              const path = url.pathname;
              return path.startsWith("/api/") || path.startsWith("/health/");
            },
            handler: "NetworkOnly",
          },
        ],
      },
    }),
  ],
  define: {
    __CANONICAL_PRODUCTION_ORIGIN__: JSON.stringify(CANONICAL_PRODUCTION_ORIGIN),
    __APP_BUILD_ID__: JSON.stringify(APP_BUILD_ID),
  },
  server: {
    port: 5173,
    allowedHosts: [".trycloudflare.com"],
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${process.env.API_PROXY_PORT ?? "8000"}`,
        changeOrigin: true,
      },
      "/health": {
        target: `http://127.0.0.1:${process.env.API_PROXY_PORT ?? "8000"}`,
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 5173,
    allowedHosts: [".trycloudflare.com"],
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${process.env.API_PROXY_PORT ?? "8000"}`,
        changeOrigin: true,
      },
      "/health": {
        target: `http://127.0.0.1:${process.env.API_PROXY_PORT ?? "8000"}`,
        changeOrigin: true,
      },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
    exclude: ["**/node_modules/**", "**/dist/**", "e2e/**"],
  },
});
