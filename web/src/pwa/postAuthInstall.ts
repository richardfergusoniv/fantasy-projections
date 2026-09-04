import { isIosDevice, isStandaloneDisplay } from "./displayMode";

/** Session flag: magic-link verify just completed in browser Safari. */
export const POST_AUTH_INSTALL_KEY = "fantasy-decisions:show-post-auth-install";

export function markPostAuthInstallNeeded(): void {
  if (typeof window === "undefined") return;
  if (isStandaloneDisplay()) return;
  if (!isIosDevice()) return;
  window.sessionStorage.setItem(POST_AUTH_INSTALL_KEY, "1");
}

export function clearPostAuthInstallNeeded(): void {
  if (typeof window === "undefined") return;
  window.sessionStorage.removeItem(POST_AUTH_INSTALL_KEY);
}

export function shouldShowPostAuthInstall(): boolean {
  if (typeof window === "undefined") return false;
  if (isStandaloneDisplay()) return false;
  if (!isIosDevice()) return false;
  return window.sessionStorage.getItem(POST_AUTH_INSTALL_KEY) === "1";
}
