"""Injury event detection and policy mapping for depth-chart refresh."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from src.comparison.sleeper_compare import _normalize_name

# Positions that participate in the curated fantasy depth chart.
SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

# Cap applied to PUP / long uncertain availability (season projections).
PUP_GAMES_CAP = 8.0

# Sleeper injury_status values → policy bucket.
IR_STATUSES = {"IR", "Injured Reserve"}
PUP_STATUSES = {"PUP"}
SHORT_TERM_STATUSES = {"Out", "Doubtful"}
# Questionable / NA / Sus / COV / DNR: flag-only or ignored for season chart.


def policy_for_status(injury_status: str | None) -> dict | None:
    """Return override/depth policy for a Sleeper (or mapped) injury_status."""
    if injury_status is None or (isinstance(injury_status, float) and pd.isna(injury_status)):
        return None
    status = str(injury_status).strip()
    if status in IR_STATUSES:
        return {
            "bucket": "season_ending",
            "override_mode": "zero",
            "override_games": None,
            "remove_from_chart": True,
            "promote_next": True,
            "auto_safe": True,
        }
    if status in PUP_STATUSES:
        return {
            "bucket": "uncertain",
            "override_mode": "cap",
            "override_games": PUP_GAMES_CAP,
            "remove_from_chart": False,
            "promote_next": False,
            "auto_safe": True,
        }
    if status in SHORT_TERM_STATUSES:
        return {
            "bucket": "short_term",
            "override_mode": None,
            "override_games": None,
            "remove_from_chart": False,
            "promote_next": False,
            "auto_safe": False,
        }
    return None


def _resolve_gsis(
    status_row: pd.Series,
    chart: pd.DataFrame,
    id_lookup: pd.DataFrame | None = None,
) -> str | None:
    gsis = status_row.get("gsis_id")
    if pd.notna(gsis) and str(gsis).strip():
        return str(gsis).strip()
    name_key = status_row.get("name_key") or _normalize_name(status_row.get("display_name"))
    pos = status_row.get("position")
    if not name_key:
        return None

    def _from_frame(frame: pd.DataFrame) -> str | None:
        if frame is None or frame.empty or "name_key" not in frame.columns:
            return None
        hit = frame[frame["name_key"] == name_key]
        if pos and "position" in frame.columns:
            pos_hit = hit[hit["position"] == pos]
            if not pos_hit.empty:
                hit = pos_hit
        if hit.empty:
            return None
        if len(hit) > 1 and pd.notna(status_row.get("team")) and "team" in hit.columns:
            team_hit = hit[hit["team"] == status_row["team"]]
            if len(team_hit) == 1:
                return str(team_hit.iloc[0]["gsis_id"])
        if len(hit) == 1:
            return str(hit.iloc[0]["gsis_id"])
        return None

    resolved = _from_frame(chart)
    if resolved:
        return resolved
    return _from_frame(id_lookup)


def detect_injury_events(
    sleeper_status: pd.DataFrame,
    depth_chart: pd.DataFrame,
    as_of: str | date | datetime | None = None,
    source: str = "sleeper",
    id_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build proposal rows for charted (and IR off-chart) injury events.

    Columns include action, override fields, auto_safe, and confirmed (False).
    ``id_lookup`` is an optional frame with at least ``gsis_id`` + ``name_key``
    (and ideally ``position`` / ``team``) for resolving Sleeper rows whose
    ``gsis_id`` is null (common) but who still need a season-ending override.
    """
    cols = [
        "event_id", "as_of_date", "source", "gsis_id", "player_name", "team",
        "position", "injury_status", "bucket", "action", "override_mode",
        "override_games", "remove_from_chart", "promote_next", "auto_safe",
        "confirmed", "on_curated_chart", "reason",
    ]
    if sleeper_status is None or sleeper_status.empty:
        return pd.DataFrame(columns=cols)

    if as_of is None:
        as_of_date = datetime.now(timezone.utc).date().isoformat()
    elif isinstance(as_of, datetime):
        as_of_date = as_of.date().isoformat()
    elif isinstance(as_of, date):
        as_of_date = as_of.isoformat()
    else:
        as_of_date = str(as_of)[:10]

    chart = depth_chart.copy() if depth_chart is not None else pd.DataFrame()
    if not chart.empty and "name_key" not in chart.columns:
        chart["name_key"] = chart["player_name"].map(_normalize_name)
    lookup = id_lookup.copy() if id_lookup is not None else pd.DataFrame()
    if not lookup.empty and "name_key" not in lookup.columns and "display_name" in lookup.columns:
        lookup["name_key"] = lookup["display_name"].map(_normalize_name)
    curated_ids = set(chart["gsis_id"].dropna().astype(str)) if not chart.empty else set()

    events = []
    for _, row in sleeper_status.iterrows():
        policy = policy_for_status(row.get("injury_status"))
        if policy is None:
            continue
        gsis = _resolve_gsis(row, chart, lookup if not lookup.empty else None)
        on_chart = bool(gsis and gsis in curated_ids)
        pos = row.get("position")
        # Season-ending IR: keep even if off curated (Pearsall-shaped overrides),
        # but only for fantasy skill positions — defensive IR would flood overrides.
        # PUP/short-term: only if currently on the curated/live chart.
        if not on_chart and policy["bucket"] != "season_ending":
            continue
        if not on_chart and policy["bucket"] == "season_ending" and pos not in SKILL_POSITIONS:
            continue
        if not gsis:
            continue
        name = row.get("display_name") or row.get("player_name") or gsis
        if policy["bucket"] == "season_ending":
            action = "remove_and_zero" if on_chart else "override_zero"
        elif policy["bucket"] == "uncertain":
            action = "cap_availability"
        else:
            action = "flag_only"

        event_id = f"{as_of_date}:{gsis}:{policy['bucket']}"
        events.append({
            "event_id": event_id,
            "as_of_date": as_of_date,
            "source": source,
            "gsis_id": gsis,
            "player_name": name,
            "team": row.get("team"),
            "position": row.get("position"),
            "injury_status": row.get("injury_status"),
            "bucket": policy["bucket"],
            "action": action,
            "override_mode": policy["override_mode"],
            "override_games": policy["override_games"],
            "remove_from_chart": bool(policy["remove_from_chart"] and on_chart),
            "promote_next": bool(policy["promote_next"] and on_chart),
            "auto_safe": bool(policy["auto_safe"]),
            "confirmed": False,
            "on_curated_chart": on_chart,
            "reason": (
                f"{source} injury_status={row.get('injury_status')}"
                + (f"; {row.get('injury_body_part')}" if pd.notna(row.get("injury_body_part")) else "")
            ),
        })

    if not events:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(events).reset_index(drop=True)
