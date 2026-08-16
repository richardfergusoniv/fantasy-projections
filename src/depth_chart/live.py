"""Build a derived live depth chart from curated base + injury events."""

from __future__ import annotations

import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEPTH_DIR = os.path.join(REPO_ROOT, "src", "depth_chart")

# Preseason WR formation columns (Ourlads LWR/RWR/SWR). Chart UX / research
# priors only — they do not allocate projection volume.
WR_FORMATION_ROLES = ("LWR", "RWR", "SWR")
WR_FORMATION_ROLE_PRIORS = {"LWR": 0.1554, "RWR": 0.0667, "SWR": 0.0386}
DEPTH_RANK_TO_WR_FORMATION_ROLE = {1: "LWR", 2: "RWR", 3: "SWR"}

# Rank-order defaults when formation_role is missing (legacy path).
WR_USAGE_SLOTS = [WR_FORMATION_ROLE_PRIORS[r] for r in WR_FORMATION_ROLES]


def curated_path(season: int) -> str:
    return os.path.join(DEPTH_DIR, f"starters_{season}.csv")


def live_path(season: int) -> str:
    return os.path.join(DEPTH_DIR, f"live_depth_{season}.csv")


def load_curated_depth_chart(season: int) -> pd.DataFrame:
    path = curated_path(season)
    if not os.path.exists(path):
        return pd.DataFrame()
    dc = pd.read_csv(path)
    return dc[dc["season"] == season].copy()


def _renumber_room(room: pd.DataFrame) -> pd.DataFrame:
    room = room.sort_values("depth_rank").copy()
    room["depth_rank"] = range(1, len(room) + 1)
    return room


def _recompute_wr_priors(room: pd.DataFrame) -> pd.DataFrame:
    """Refresh unreviewed WR slot defaults after a removal.

    Prefer stable ``formation_role`` (LWR/RWR/SWR) over renumbered depth_rank
    so removing an LWR does not promote the RWR into the LWR prior. Rank-order
    WR_USAGE_SLOTS remain the fallback when formation_role is absent.

    Slot means are display/research starting points only. They do not move
    projections; keep usage_share_reviewed False so chart renumber stays
    research metadata.
    """
    room = room.sort_values("depth_rank").copy()
    has_role = "formation_role" in room.columns
    for i, idx in enumerate(room.index):
        prior = None
        if has_role:
            role = room.at[idx, "formation_role"]
            if isinstance(role, str) and role in WR_FORMATION_ROLE_PRIORS:
                prior = WR_FORMATION_ROLE_PRIORS[role]
        if prior is None and i < len(WR_USAGE_SLOTS):
            prior = WR_USAGE_SLOTS[i]
        if prior is not None:
            room.at[idx, "usage_share_prior"] = prior
        if "usage_share_reviewed" in room.columns:
            room.at[idx, "usage_share_reviewed"] = False
    return room


def build_live_depth_chart(
    curated: pd.DataFrame,
    events: pd.DataFrame,
    as_of_date: str,
    apply_auto_safe: bool = True,
    apply_confirmed: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (live_chart, applied_events).

    Removes charted players with remove_from_chart on auto_safe/confirmed
    events, renumbers depth_rank within (team, position), and refreshes
    unreviewed WR slot defaults for rooms that lost a WR.
    """
    empty_events = pd.DataFrame()
    live = curated.copy()
    if live.empty:
        return live, empty_events

    for col in ("as_of_date", "derived_from", "injury_event_id"):
        if col not in live.columns:
            live[col] = pd.NA

    live["derived_from"] = "curated"
    live["as_of_date"] = as_of_date

    if events is None or events.empty:
        return live, empty_events

    selectable = pd.Series(False, index=events.index)
    if apply_auto_safe:
        selectable = selectable | events["auto_safe"].fillna(False).astype(bool)
    if apply_confirmed:
        selectable = selectable | events["confirmed"].fillna(False).astype(bool)
    selectable = selectable & events["action"].ne("flag_only")
    applicable = events.loc[selectable].copy()
    if applicable.empty:
        return live, applicable

    remove = applicable[applicable["remove_from_chart"].fillna(False).astype(bool)]
    remove_ids = set(remove["gsis_id"].dropna().astype(str))

    if remove_ids:
        removed_rows = live[live["gsis_id"].astype(str).isin(remove_ids)]
        touched = set(zip(removed_rows["team"], removed_rows["position"]))
        # Map (team, position) -> event_id that removed someone there.
        room_event = {}
        for _, er in remove.iterrows():
            key = (er.get("team"), er.get("position"))
            if key in touched and pd.notna(er.get("event_id")):
                room_event[key] = er["event_id"]

        live = live[~live["gsis_id"].astype(str).isin(remove_ids)].copy()
        pieces = []
        for (team, position), room in live.groupby(["team", "position"], sort=False):
            room = _renumber_room(room)
            if (team, position) in touched and position == "WR":
                room = _recompute_wr_priors(room)
            if (team, position) in room_event:
                room["injury_event_id"] = room_event[(team, position)]
            pieces.append(room)
        live = pd.concat(pieces, ignore_index=True) if pieces else live.iloc[0:0]

    applied = applicable[
        applicable["remove_from_chart"].fillna(False).astype(bool)
        | applicable["override_mode"].notna()
    ].copy()
    return live, applied


def write_live_depth_chart(live: pd.DataFrame, season: int) -> str:
    path = live_path(season)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    live.to_csv(path, index=False)
    return path
