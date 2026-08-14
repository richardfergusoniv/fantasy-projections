"""Build a derived live depth chart from curated base + injury events."""

from __future__ import annotations

import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEPTH_DIR = os.path.join(REPO_ROOT, "src", "depth_chart")

WR_USAGE_SLOTS = [0.1554, 0.0667, 0.0386]


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
    """Assign slot-mean usage priors by current depth_rank order after a removal."""
    room = room.sort_values("depth_rank").copy()
    for i, idx in enumerate(room.index):
        if i < len(WR_USAGE_SLOTS):
            room.at[idx, "usage_share_prior"] = WR_USAGE_SLOTS[i]
        if "usage_share_reviewed" in room.columns:
            room.at[idx, "usage_share_reviewed"] = True
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
    events, renumbers depth_rank within (team, position), and recomputes WR
    usage priors for rooms that lost a WR.
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
