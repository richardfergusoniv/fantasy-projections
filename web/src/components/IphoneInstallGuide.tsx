import { useEffect, useState } from "react";
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

  return (
    <aside className="iphone-install-guide" aria-label="Install on iPhone">
      <p>
        <strong>Install on iPhone.</strong> Tap Share, then{" "}
        <strong>Add to Home Screen</strong>. Open that icon after sign-in so the
        app keeps your session.
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
