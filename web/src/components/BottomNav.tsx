import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Home", icon: "⌂" },
  { to: "/lineup", label: "Lineup", icon: "▣" },
  { to: "/waivers", label: "Waivers", icon: "＋" },
  { to: "/trade-lab", label: "Trade", icon: "⇄" },
  { to: "/dynasty", label: "Dynasty", icon: "◈" },
  { to: "/draft", label: "Draft", icon: "◎" },
  { to: "/assistant", label: "Assist", icon: "✦" },
  { to: "/operations", label: "Ops", icon: "⚙" },
] as const;

export function BottomNav() {
  return (
    <nav className="bottom-nav" aria-label="Primary">
      <ul>
        {NAV_ITEMS.map((item) => (
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
      </ul>
    </nav>
  );
}
