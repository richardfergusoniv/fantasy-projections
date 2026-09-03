"""Role-aware expected-start allocation (infrastructure; frozen avail coeffs).

Separates:
- productivity while active (not used to infer starter status)
- probability of starting / expected starts
- backup / package role

Availability shrink coefficients in ``expected_availability`` are unchanged.
This layer applies preseason depth-chart role after that frozen estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.qb_active_archetype.active_rates import expected_availability
from src.projection.qb_active_archetype.thresholds import AVAIL_FULL_SEASON_GAMES
from src.projection.transitions import SEASON_GAMES

# Residual-split weights — NOT availability shrink coefficients.
BACKUP_RESIDUAL_WEIGHT = 1.0
PACKAGE_RESIDUAL_WEIGHT = 0.25
ROOKIE_BACKUP_RESIDUAL_WEIGHT = 0.35
ROOKIE_PACKAGE_RESIDUAL_WEIGHT = 0.20


def role_from_preseason(*, depth_tier: float | None, is_rookie: bool) -> str:
    try:
        depth = float(depth_tier) if depth_tier is not None and pd.notna(depth_tier) else np.nan
    except (TypeError, ValueError):
        depth = np.nan
    if pd.notna(depth) and depth == 1.0:
        return "rookie_starter" if is_rookie else "starter"
    if pd.notna(depth) and depth == 2.0:
        return "rookie_backup" if is_rookie else "backup"
    if is_rookie:
        return "rookie_package"
    if pd.notna(depth) and depth >= 3.0:
        return "package"
    return "package"


def residual_weight(role: str) -> float:
    return {
        "backup": BACKUP_RESIDUAL_WEIGHT,
        "rookie_backup": ROOKIE_BACKUP_RESIDUAL_WEIGHT,
        "package": PACKAGE_RESIDUAL_WEIGHT,
        "rookie_package": ROOKIE_PACKAGE_RESIDUAL_WEIGHT,
    }.get(role, PACKAGE_RESIDUAL_WEIGHT)


def _prior_starts(history: pd.DataFrame, player_id: str, target_season: int) -> float:
    hist = history[
        (history.player_id.astype(str) == str(player_id)) & (history.season < int(target_season))
    ]
    if hist.empty or "active_starts" not in hist.columns:
        return 0.0
    return float(pd.to_numeric(hist["active_starts"], errors="coerce").fillna(0).sum())


def allocate_team_expected_starts(
    *,
    history: pd.DataFrame,
    target_season: int,
    room: pd.DataFrame,
) -> pd.DataFrame:
    """Allocate expected starts for one team's QB room.

    ``room`` must include player_id, preseason_depth_tier, is_rookie_at_cutoff.
    Active-game rates are ignored for role / start probability.
    """
    players = room.copy()
    players["player_id"] = players["player_id"].astype(str)
    players["is_rookie_at_cutoff"] = players.get(
        "is_rookie_at_cutoff", pd.Series(False, index=players.index)
    ).fillna(False).astype(bool)
    players["preseason_role"] = [
        role_from_preseason(depth_tier=r.get("preseason_depth_tier"), is_rookie=bool(r["is_rookie_at_cutoff"]))
        for _, r in players.iterrows()
    ]
    players["prior_active_starts_sum"] = [
        _prior_starts(history, pid, target_season) for pid in players["player_id"]
    ]
    # Frozen availability (coefficients unchanged) — used for the starter only.
    avail_rows = []
    for pid in players["player_id"]:
        avail_rows.append(expected_availability(history, player_id=pid, target_season=target_season))
    players["frozen_expected_starts"] = [a["expected_active_starts"] for a in avail_rows]
    players["frozen_partial_exit_rate"] = [a.get("partial_exit_rate") or 0.0 for a in avail_rows]
    players["frozen_avail_method"] = [a.get("method") for a in avail_rows]

    starter_mask = players["preseason_role"].isin(("starter", "rookie_starter"))
    if not starter_mask.any():
        # No charted QB1: promote most prior active starts (not highest rate).
        promote = int(players["prior_active_starts_sum"].idxmax())
        players.loc[promote, "preseason_role"] = (
            "rookie_starter" if bool(players.loc[promote, "is_rookie_at_cutoff"]) else "starter"
        )
        starter_mask = players["preseason_role"].isin(("starter", "rookie_starter"))
    if starter_mask.sum() > 1:
        # Tie: keep the QB1 with the most prior active starts.
        keep = players.loc[starter_mask].sort_values(
            "prior_active_starts_sum", ascending=False
        ).index[0]
        demote = starter_mask & (players.index != keep)
        players.loc[demote, "preseason_role"] = players.loc[demote].apply(
            lambda r: "rookie_backup" if r["is_rookie_at_cutoff"] else "backup", axis=1
        )
        starter_mask = players.index == keep

    starter = players.loc[starter_mask].iloc[0]
    starter_starts = float(np.clip(starter["frozen_expected_starts"], 0.0, AVAIL_FULL_SEASON_GAMES))
    starter_partial = float(np.clip(starter["frozen_partial_exit_rate"], 0.0, 0.5))
    residual_games = max(0.0, SEASON_GAMES - starter_starts)
    partial_exposure = starter_starts * 0.5 * starter_partial
    backup_budget = residual_games + partial_exposure

    players["allocated_expected_starts"] = 0.0
    players.loc[starter_mask, "allocated_expected_starts"] = starter_starts
    players["is_qb1"] = starter_mask
    backups = players.loc[~starter_mask].copy()
    if not backups.empty and backup_budget > 0:
        weights = backups["preseason_role"].map(residual_weight).astype(float)
        # Explicitly ignore productivity: no attempts/carries in the weight.
        wsum = float(weights.sum())
        if wsum <= 0:
            weights = pd.Series(1.0, index=backups.index)
            wsum = float(len(backups))
        share = backup_budget * (weights / wsum)
        players.loc[backups.index, "allocated_expected_starts"] = share.to_numpy()

    # Conserve scheduled team games onto the room (starter unchanged; leftover
    # after rounding noise stays on backups).
    total = float(players["allocated_expected_starts"].sum())
    gap = SEASON_GAMES - total
    if abs(gap) > 1e-9 and (~starter_mask).any():
        bench_idx = players.index[~starter_mask]
        bench_sum = float(players.loc[bench_idx, "allocated_expected_starts"].sum())
        if bench_sum > 0 and gap != 0:
            players.loc[bench_idx, "allocated_expected_starts"] *= (bench_sum + gap) / bench_sum
        elif gap > 0:
            # No bench volume yet: split leftover by residual weights.
            weights = players.loc[bench_idx, "preseason_role"].map(residual_weight).astype(float)
            wsum = float(weights.sum()) or float(len(bench_idx))
            players.loc[bench_idx, "allocated_expected_starts"] += gap * (weights / wsum)
    players["allocated_expected_starts"] = players["allocated_expected_starts"].clip(lower=0.0)
    players["team_starts_conserved"] = float(players["allocated_expected_starts"].sum())
    players["backup_budget"] = backup_budget
    players["starter_partial_exposure"] = partial_exposure
    return players


def allocate_league_expected_starts(
    *,
    history: pd.DataFrame,
    target_season: int,
    rooms: pd.DataFrame,
    team_col: str = "team",
) -> pd.DataFrame:
    """Run :func:`allocate_team_expected_starts` for every team in ``rooms``."""
    parts = []
    for team, room in rooms.groupby(team_col, dropna=False):
        if pd.isna(team):
            continue
        allocated = allocate_team_expected_starts(
            history=history, target_season=target_season, room=room
        )
        allocated[team_col] = team
        parts.append(allocated)
    if not parts:
        return rooms.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def assert_backups_do_not_inherit_starter_volume(
    allocated: pd.DataFrame, *, starter_floor: float = 8.0, backup_ceiling: float = 7.0
) -> list[dict]:
    """Return violations where a non-QB1 received starter-like expected starts."""
    violations = []
    for _, r in allocated.iterrows():
        if r.get("is_qb1"):
            continue
        starts = float(r.get("allocated_expected_starts") or 0.0)
        if starts >= starter_floor:
            violations.append(
                {
                    "player_id": r["player_id"],
                    "allocated_expected_starts": starts,
                    "role": r.get("preseason_role"),
                    "reason": "backup_inherited_starter_starts",
                }
            )
        if starts > backup_ceiling and r.get("preseason_role") in (
            "backup",
            "rookie_backup",
            "package",
            "rookie_package",
        ):
            violations.append(
                {
                    "player_id": r["player_id"],
                    "allocated_expected_starts": starts,
                    "role": r.get("preseason_role"),
                    "reason": "backup_starts_above_ceiling",
                }
            )
    return violations
