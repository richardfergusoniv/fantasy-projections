export function isStandaloneDisplay(): boolean {
  if (typeof window === "undefined") return false;
  const media = window.matchMedia?.("(display-mode: standalone)")?.matches === true;
  const iosStandalone = "standalone" in window.navigator && Boolean((window.navigator as Navigator & { standalone?: boolean }).standalone);
  return media || iosStandalone;
}

export function isIosDevice(): boolean {
  if (typeof navigator === "undefined") return false;
  const ua = navigator.userAgent || "";
  if (/iPad|iPhone|iPod/i.test(ua)) return true;
  return navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1;
}

export function isIosSafari(): boolean {
  if (!isIosDevice()) return false;
  const ua = navigator.userAgent || "";
  const isWebKit = /WebKit/i.test(ua);
  const isOtherBrowser = /CriOS|FxiOS|EdgiOS|OPiOS|OPT\//i.test(ua);
  return isWebKit && !isOtherBrowser;
}
