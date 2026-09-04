import { useEffect, useState } from "react";
import { CANONICAL_PRODUCTION_ORIGIN } from "../pwa/canonical";
import { isIosDevice, isStandaloneDisplay } from "../pwa/displayMode";

const DISMISS_KEY = "fantasy-decisions:iphone-install-dismissed";

export function IphoneInstallGuide() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (isStandaloneDisplay()) return;
    if (!isIosDevice()) return;
    if (window.localStorage.getItem(DISMISS_KEY) === "1") return;
    setVisible(true);
  }, []);

  if (!visible) {
    return null;
  }

  const host = CANONICAL_PRODUCTION_ORIGIN.replace(/^https:\/\//, "");

  return (
    <aside className="iphone-install-guide" aria-label="Install on iPhone">
      <p>
        <strong>Install on iPhone.</strong> Open {host} in Safari, tap Share, then{" "}
        <strong>Add to Home Screen</strong>. Open that icon after sign-in so the app keeps
        your session. Do not install from fantasy-projections.vercel.app — that is a different
        legacy site.
      </p>
      <button
        className="btn btn-ghost btn-compact"
        type="button"
        onClick={() => {
          window.localStorage.setItem(DISMISS_KEY, "1");
          setVisible(false);
        }}
      >
        Dismiss
      </button>
    </aside>
  );
}
