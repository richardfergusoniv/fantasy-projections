import { useEffect, useRef, useState, type FormEvent } from "react";
import { Navigate, useSearchParams } from "react-router-dom";
import { AppBuildStamp } from "../components/AppBuildStamp";
import { useAuth } from "../hooks/useAuth";
import { CANONICAL_PRODUCTION_ORIGIN } from "../pwa/canonical";
import { readMagicLinkToken } from "../pwa/magicLink";
import { forceRefreshAppShell } from "../pwa/registerUpdates";

export function LoginScreen() {
  const { user, loading, login, verify, sessionExpired, error: authError } = useAuth();
  const [searchParams] = useSearchParams();
  const [email, setEmail] = useState("");
  const [token, setToken] = useState("");
  const [devLink, setDevLink] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [refreshingShell, setRefreshingShell] = useState(false);
  const autoVerified = useRef(false);

  const queryToken = searchParams.get("token");
  const linkToken = readMagicLinkToken() ?? queryToken;

  useEffect(() => {
    // Arriving from the emailed magic link: verify without making the user
    // copy the token out of the URL bar. Production links use a hash fragment
    // so mail scanners cannot burn the one-time token on prefetch.
    if (!linkToken || autoVerified.current) return;
    autoVerified.current = true;
    setSubmitting(true);
    setMessage("Verifying your sign-in link…");
    void verify(linkToken)
      .catch((err: unknown) => {
        setMessage(err instanceof Error ? err.message : "Invalid or expired token");
      })
      .finally(() => setSubmitting(false));
  }, [linkToken, verify]);

  if (!loading && user) {
    return <Navigate to="/" replace />;
  }

  async function onRequestLink(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setMessage(null);
    try {
      const result = await login(email);
      setDevLink(result.devLink ?? null);
      setMessage(
        result.devLink
          ? "Development magic link generated below."
          : "Check your email for a sign-in link.",
      );
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Unable to send magic link");
    } finally {
      setSubmitting(false);
    }
  }

  async function onVerify(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setMessage(null);
    try {
      await verify(token);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Invalid or expired token");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="screen login-screen">
      <section className="panel login-panel">
        <header className="panel-header">
          <h2>Sign in</h2>
        </header>
        <div className="panel-body stack">
          {sessionExpired ? (
            <p className="state-notice state-error" role="alert">
              <span className="state-glyph" aria-hidden="true">
                ✕
              </span>
              <span>
                <strong className="state-title">Session expired.</strong> Your previous sign-in is
                no longer valid. Request a new link and sign in again.
              </span>
            </p>
          ) : null}
          <p className="muted">
            Email magic-link authentication. Use{" "}
            {CANONICAL_PRODUCTION_ORIGIN.replace(/^https:\/\//, "")} — the bare
            fantasy-projections.vercel.app host is a different legacy app.
          </p>
          <AppBuildStamp />
          <form className="stack" onSubmit={onRequestLink}>
            <div className="field">
              <label htmlFor="login-email">Email</label>
              <input
                id="login-email"
                name="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
              />
            </div>
            <button className="btn btn-primary" type="submit" disabled={submitting}>
              Send magic link
            </button>
          </form>
          {devLink ? (
            <p className="dev-link">
              Dev link:{" "}
              <a href={devLink} target="_blank" rel="noopener noreferrer">
                {devLink}
              </a>
            </p>
          ) : null}
          <form className="stack" onSubmit={onVerify}>
            <div className="field">
              <label htmlFor="login-token">Token</label>
              <input
                id="login-token"
                name="token"
                type="text"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Paste token from email"
              />
            </div>
            <button className="btn btn-ghost" type="submit" disabled={submitting || !token}>
              Verify token
            </button>
          </form>
          <p className="status-message" role="status" aria-live="polite">
            {message ?? authError ?? ""}
          </p>
          <button
            className="btn btn-ghost"
            type="button"
            disabled={refreshingShell}
            onClick={() => {
              setRefreshingShell(true);
              void forceRefreshAppShell().finally(() => setRefreshingShell(false));
            }}
          >
            {refreshingShell ? "Refreshing…" : "Stuck on an old version? Refresh"}
          </button>
        </div>
      </section>
    </div>
  );
}
