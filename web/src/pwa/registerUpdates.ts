/**
 * Keep installed PWAs (especially iOS standalone) from sticking on a stale shell.
 *
 * vite-plugin-pwa autoUpdate reloads when a new worker activates, but iOS often
 * skips update checks while the home-screen app stays warm. Re-check on focus
 * and periodically so a new sw.js is noticed promptly.
 */
const LEGACY_HTML_CACHE = "html-navigations";

async function deleteLegacyHtmlCache(): Promise<void> {
  if (!("caches" in window)) return;
  try {
    await caches.delete(LEGACY_HTML_CACHE);
  } catch {
    /* ignore */
  }
}

export function registerServiceWorkerUpdates(
  register: typeof import("virtual:pwa-register").registerSW,
): void {
  void deleteLegacyHtmlCache();

  register({
    immediate: true,
    onRegisteredSW(_swUrl, registration) {
      if (!registration) return;

      const checkForUpdates = () => {
        void registration.update().catch(() => {
          /* ignore transient update-check failures */
        });
      };

      checkForUpdates();
      setInterval(checkForUpdates, 60_000);

      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "visible") {
          void deleteLegacyHtmlCache();
          checkForUpdates();
        }
      });

      window.addEventListener("focus", () => {
        void deleteLegacyHtmlCache();
        checkForUpdates();
      });
    },
  });
}

/** Clear Workbox caches and unregister workers, then reload the shell. */
export async function forceRefreshAppShell(): Promise<void> {
  if ("serviceWorker" in navigator) {
    const registrations = await navigator.serviceWorker.getRegistrations();
    await Promise.all(registrations.map((registration) => registration.unregister()));
  }
  if ("caches" in window) {
    const keys = await caches.keys();
    await Promise.all(keys.map((key) => caches.delete(key)));
  }
  window.location.reload();
}
