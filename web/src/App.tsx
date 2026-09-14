import { lazy, Suspense } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { useAuth } from "./hooks/useAuth";
import { AppStateProvider } from "./hooks/useAppState";
import { HomeScreen } from "./screens/Home";
import { LoginScreen } from "./screens/Login";

const LineupScreen = lazy(() =>
  import("./screens/Lineup").then((module) => ({ default: module.LineupScreen })),
);
const WaiversScreen = lazy(() =>
  import("./screens/Waivers").then((module) => ({ default: module.WaiversScreen })),
);
const TradeLabScreen = lazy(() =>
  import("./screens/TradeLab").then((module) => ({ default: module.TradeLabScreen })),
);
const DynastyScreen = lazy(() =>
  import("./screens/Dynasty").then((module) => ({ default: module.DynastyScreen })),
);
const DraftScreen = lazy(() =>
  import("./screens/Draft").then((module) => ({ default: module.DraftScreen })),
);
const AssistantScreen = lazy(() =>
  import("./screens/Assistant").then((module) => ({ default: module.AssistantScreen })),
);
const OperationsScreen = lazy(() =>
  import("./screens/Operations").then((module) => ({ default: module.OperationsScreen })),
);

function ScreenFallback() {
  return <div className="screen empty-state">Loading screen…</div>;
}

/**
 * Explicit recovery path for an expired session.
 *
 * Session loss mid-use is a distinct state from "never signed in", so it gets
 * its own screen. Silently bouncing to the login form leaves the user guessing
 * whether they were signed out or the app broke.
 */
function SessionExpiredScreen() {
  const navigate = useNavigate();
  const { dismissSessionExpired } = useAuth();

  return (
    <div className="screen login-screen">
      <section className="panel login-panel" aria-label="Session expired">
        <header className="panel-header">
          <h2>Session expired</h2>
        </header>
        <div className="panel-body stack">
          <p className="state-notice state-error" role="alert">
            <span className="state-glyph" aria-hidden="true">
              ✕
            </span>
            <span>
              <strong className="state-title">Your session expired.</strong> The server no longer
              accepts this sign-in, so nothing on screen is current. Sign in again to continue.
            </span>
          </p>
          <p className="muted">
            Offline copies of recommendations were cleared, because they belonged to the previous
            session.
          </p>
          <button
            className="btn btn-primary"
            type="button"
            onClick={() => {
              dismissSessionExpired();
              void navigate("/login", { replace: true });
            }}
          >
            Sign in again
          </button>
        </div>
      </section>
    </div>
  );
}

function ProtectedRoutes() {
  const { user, loading, sessionExpired } = useAuth();

  if (loading) {
    return <div className="screen empty-state">Loading session…</div>;
  }

  if (!user && sessionExpired) {
    return <SessionExpiredScreen />;
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return (
    <AppStateProvider>
      <Suspense fallback={<ScreenFallback />}>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<HomeScreen />} />
            <Route path="lineup" element={<LineupScreen />} />
            <Route path="waivers" element={<WaiversScreen />} />
            <Route path="trade-lab" element={<TradeLabScreen />} />
            <Route path="dynasty" element={<DynastyScreen />} />
            <Route path="draft" element={<DraftScreen />} />
            <Route path="assistant" element={<AssistantScreen />} />
            <Route path="operations" element={<OperationsScreen />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </AppStateProvider>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginScreen />} />
      {/* Target of the emailed magic link: APP_PUBLIC_URL/auth/callback?token=... */}
      <Route path="/auth/callback" element={<LoginScreen />} />
      <Route path="/*" element={<ProtectedRoutes />} />
    </Routes>
  );
}
