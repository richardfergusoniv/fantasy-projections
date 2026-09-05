import { useEffect, useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";

const PRIMARY_NAV = [
  { to: "/", label: "Home", icon: "⌂" },
  { to: "/lineup", label: "Lineup", icon: "▣" },
  { to: "/waivers", label: "Waivers", icon: "＋" },
  { to: "/trade-lab", label: "Trade", icon: "⇄" },
  { to: "/draft", label: "Draft", icon: "◎" },
] as const;

const MORE_NAV = [
  { to: "/dynasty", label: "Dynasty", icon: "◈" },
  { to: "/assistant", label: "Assist", icon: "✦" },
  { to: "/operations", label: "Ops", icon: "⚙" },
] as const;

const MORE_PATHS = new Set<string>(MORE_NAV.map((item) => item.to));

type BottomNavProps = {
  /** When true, shell chrome is sliding away — close More and mark nav inert. */
  chromeCollapsed?: boolean;
};

export function BottomNav({ chromeCollapsed = false }: BottomNavProps) {
  const [moreOpen, setMoreOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const moreActive = MORE_PATHS.has(location.pathname);

  useEffect(() => {
    setMoreOpen(false);
  }, [location.pathname]);

  useEffect(() => {
    if (chromeCollapsed) setMoreOpen(false);
  }, [chromeCollapsed]);

  return (
    <>
      {moreOpen ? (
        <div className="bottom-nav-more" role="dialog" aria-label="More screens">
          <ul>
            {MORE_NAV.map((item) => (
              <li key={item.to}>
                <button
                  type="button"
                  className={location.pathname === item.to ? "active" : undefined}
                  onClick={() => {
                    setMoreOpen(false);
                    void navigate(item.to);
                  }}
                >
                  <span className="nav-icon" aria-hidden>
                    {item.icon}
                  </span>
                  <span className="nav-label">{item.label}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <nav className="bottom-nav" aria-label="Primary" inert={chromeCollapsed || undefined}>
        <ul>
          {PRIMARY_NAV.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.to === "/"}
                className={({ isActive }) => (isActive ? "active" : undefined)}
              >
                <span className="nav-icon" aria-hidden>
                  {item.icon}
                </span>
                <span className="nav-label">{item.label}</span>
              </NavLink>
            </li>
          ))}
          <li>
            <button
              type="button"
              className={moreActive || moreOpen ? "active" : undefined}
              aria-expanded={moreOpen}
              aria-haspopup="dialog"
              onClick={() => setMoreOpen((open) => !open)}
            >
              <span className="nav-icon" aria-hidden>
                ⋯
              </span>
              <span className="nav-label">More</span>
            </button>
          </li>
        </ul>
      </nav>
    </>
  );
}
