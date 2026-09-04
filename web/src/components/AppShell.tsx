import { Outlet } from "react-router-dom";
import { BottomNav } from "./BottomNav";
import { LeagueSwitcher } from "./LeagueSwitcher";
import { useAuth } from "../hooks/useAuth";

export function AppShell() {
  const { user, logout, error } = useAuth();

  return (
    <div className="app-shell">
      <a className="skip-link" href="#app-main">
        Skip to main content
      </a>
      <header className="app-topbar">
        <div className="topbar-row">
          <div className="brand-block">
            <p className="brand-label">Fantasy Decisions</p>
            <h1 className="app-title">Decision App</h1>
          </div>
          <div className="account-block">
            {user ? (
              <span className="account-email" title={user.email}>
                {user.email}
              </span>
            ) : null}
            <button className="btn btn-ghost btn-compact" type="button" onClick={() => void logout()}>
              Sign out
            </button>
          </div>
        </div>
        <LeagueSwitcher />
      </header>
      {error ? (
        <p className="error-text shell-error" role="alert">
          {error}
        </p>
      ) : null}
      <main className="app-main" id="app-main" tabIndex={-1}>
        <Outlet />
      </main>
      <BottomNav />
    </div>
  );
}
