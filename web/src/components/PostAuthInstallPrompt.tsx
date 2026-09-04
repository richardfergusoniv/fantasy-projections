import { CANONICAL_PRODUCTION_ORIGIN, PWA_APP_NAME } from "../pwa/canonical";
import { clearPostAuthInstallNeeded } from "../pwa/postAuthInstall";

type Props = {
  onContinueInSafari: () => void;
};

/**
 * Shown after magic-link sign-in on iPhone Safari. Magic links always open the
 * website; Add to Home Screen from that Safari tab is the correct install path.
 */
export function PostAuthInstallPrompt({ onContinueInSafari }: Props) {
  const host = CANONICAL_PRODUCTION_ORIGIN.replace(/^https:\/\//, "");

  return (
    <div className="screen login-screen">
      <section className="panel login-panel post-auth-install-panel" aria-labelledby="post-auth-install-title">
        <header className="panel-header">
          <h2 id="post-auth-install-title">Signed in — install the app</h2>
        </header>
        <div className="panel-body stack">
          <p>
            The magic link opens <strong>{host}</strong> in Safari. That is expected — it is how
            you get a signed-in session before installing.
          </p>
          <ol className="post-auth-install-steps">
            <li>
              Stay on this Safari page (confirm the address is {host}, not
              fantasy-projections.vercel.app).
            </li>
            <li>
              Tap <strong>Share</strong>, then <strong>Add to Home Screen</strong>.
            </li>
            <li>
              Open the new <strong>{PWA_APP_NAME}</strong> icon from your Home Screen — use that
              icon going forward, not this Safari tab.
            </li>
          </ol>
          <p className="muted">
            The Home Screen icon keeps your sign-in and runs as the app. Bookmarking or leaving
            this Safari tab open is the &ldquo;website&rdquo; experience.
          </p>
          <button
            className="btn btn-primary"
            type="button"
            onClick={() => {
              clearPostAuthInstallNeeded();
              onContinueInSafari();
            }}
          >
            Continue in Safari for now
          </button>
        </div>
      </section>
    </div>
  );
}
