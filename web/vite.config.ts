/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import {
  CANONICAL_PRODUCTION_ORIGIN,
  PWA_APP_NAME,
  PWA_BACKGROUND_COLOR,
  PWA_THEME_COLOR,
} from "./src/pwa/canonical";

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
        globPatterns: ["**/*.{js,css,html,ico,png,svg,woff2}"],
        navigateFallback: "/index.html",
        navigateFallbackDenylist: [/^\/api\//, /^\/health\//],
        // Keep this matcher self-contained. vite-plugin-pwa stringifies
        // urlPattern into sw.js and will not bundle imported helpers — a
        // closed-over `isUncacheableAppUrl` becomes a ReferenceError on every
        // API fetch once the service worker controls the page.
        runtimeCaching: [
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
