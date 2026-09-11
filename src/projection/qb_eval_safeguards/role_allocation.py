"""Role-aware expected-start allocation (evaluation infrastructure).

Separates:
- productivity while active (not used to infer starter status)
- probability of starting / expected starts
- backup / package role

Uses only seasons before the target-season cutoff. Destination-team preseason
role at the cutoff is the applicable role. Target-season actual starts must
never enter this path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.transitions import SEASON_GAMES

# Self-contained availability shrink (not H3 model coefficients / package).
AVAIL_LOOKBACK_SEASONS = 4
AVAIL_FULL_SEASON_GAMES = float(SEASON_GAMES)
AVAIL_PRIOR_STRENGTH_GAMES = 8.0
LEAGUE_STARTER_EXPECTED_GAMES = 15.0
LEAGUE_PARTIAL_EXIT_RATE = 0.08

# Residual-split weights — NOT availability shrink coefficients.
BACKUP_RESIDUAL_WEIGHT = 1.0
PACKAGE_RESIDUAL_WEIGHT = 0.25
ROOKIE_BACKUP_RESIDUAL_WEIGHT = 0.35
ROOKIE_PACKAGE_RESIDUAL_WEIGHT = 0.20


class IncompleteQbRoom(RuntimeError):
    """Raised when a QB room cannot be allocated without silent normalization."""


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


def expected_availability(
    history: pd.DataFrame,
    *,
    player_id: str,
    target_season: int,
) -> dict:
    """Expected active starts using only seasons < target (leakage-safe)."""
    if history is None or history.empty:
        return {
            "expected_active_starts": LEAGUE_STARTER_EXPECTED_GAMES,
            "partial_exit_rate": LEAGUE_PARTIAL_EXIT_RATE,
            "sample_active_starts": 0.0,
            "input_seasons": [],
            "method": "league_prior",
        }
    hist = history[
        (history["player_id"].astype(str) == str(player_id))
        & (history["season"] < int(target_season))
        & (history["season"] >= int(target_season) - AVAIL_LOOKBACK_SEASONS)
    ].copy()
    if hist.empty:
        return {
            "expected_active_starts": LEAGUE_STARTER_EXPECTED_GAMES,
            "partial_exit_rate": LEAGUE_PARTIAL_EXIT_RATE,
            "sample_active_starts": 0.0,
            "input_seasons": [],
            "method": "league_prior",
        }
    if "active_starts" in hist.columns:
        w = pd.to_numeric(hist["active_starts"], errors="coerce").fillna(0.0).clip(lower=0.0)
    elif "games" in hist.columns:
        w = pd.to_numeric(hist["games"], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        raise IncompleteQbRoom(
            "history must include active_starts or games for expected availability"
        )
    # Reject target-season leakage if caller forgot to filter.
    if (hist["season"] >= int(target_season)).any():
        raise AssertionError("future-season rows entered expected_availability")
    season_games = w.clip(upper=AVAIL_FULL_SEASON_GAMES)
    emp = float(season_games.mean()) if len(season_games) else LEAGUE_STARTER_EXPECTED_GAMES
    n = float(w.sum())
    shrink = n / (n + AVAIL_PRIOR_STRENGTH_GAMES)
    expected = shrink * emp + (1.0 - shrink) * LEAGUE_STARTER_EXPECTED_GAMES
    expected = float(np.clip(expected, 1.0, AVAIL_FULL_SEASON_GAMES))
    if "partial_exit_rate" in hist.columns:
        partial = pd.to_numeric(hist["partial_exit_rate"], errors="coerce")
        pw = w.where(partial.notna(), 0.0)
        if float(pw.sum()) > 0:
            partial_rate = float(np.average(partial[pw.gt(0)], weights=pw[pw.gt(0)]))
        else:
            partial_rate = LEAGUE_PARTIAL_EXIT_RATE
    else:
        partial_rate = LEAGUE_PARTIAL_EXIT_RATE
    return {
        "expected_active_starts": expected,
        "partial_exit_rate": float(np.clip(partial_rate, 0.0, 0.5)),
        "sample_active_starts": n,
        "input_seasons": [int(s) for s in sorted(hist["season"].unique())],
        "method": "empirical_shrink",
    }


def _prior_starts(history: pd.DataFrame, player_id: str, target_season: int) -> float:
    hist = history[
        (history.player_id.astype(str) == str(player_id)) & (history.season < int(target_season))
    ]
    if hist.empty:
        return 0.0
    col = "active_starts" if "active_starts" in hist.columns else "games"
    if col not in hist.columns:
        return 0.0
    return float(pd.to_numeric(hist[col], errors="coerce").fillna(0).sum())


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
    if room is None or room.empty:
        raise IncompleteQbRoom("QB room is empty; refusing silent normalization")
    if "player_id" not in room.columns:
        raise IncompleteQbRoom("QB room missing player_id")

    players = room.copy()
    players["player_id"] = players["player_id"].astype(str)
    if players["player_id"].isna().any() or (players["player_id"] == "").any():
        raise IncompleteQbRoom("QB room has missing player_id values")
    players["is_rookie_at_cutoff"] = (
        players.get("is_rookie_at_cutoff", pd.Series(False, index=players.index))
        .fillna(False)
        .astype(bool)
    )
    players["preseason_role"] = [
        role_from_preseason(
            depth_tier=r.get("preseason_depth_tier"),
            is_rookie=bool(r["is_rookie_at_cutoff"]),
        )
        for _, r in players.iterrows()
    ]
    players["prior_active_starts_sum"] = [
        _prior_starts(history, pid, target_season) for pid in players["player_id"]
    ]
    avail_rows = [
        expected_availability(history, player_id=pid, target_season=target_season)
        for pid in players["player_id"]
    ]
    players["frozen_expected_starts"] = [a["expected_active_starts"] for a in avail_rows]
    players["frozen_partial_exit_rate"] = [a.get("partial_exit_rate") or 0.0 for a in avail_rows]
    players["frozen_avail_method"] = [a.get("method") for a in avail_rows]
    # Provenance: never use target-season outcomes.
    for a in avail_rows:
        if any(s >= int(target_season) for s in a.get("input_seasons") or []):
            raise AssertionError("future-season availability inputs detected")

    starter_mask = players["preseason_role"].isin(("starter", "rookie_starter"))
    if not starter_mask.any():
        # No charted QB1: promote most prior active starts (not highest rate).
        promote = int(players["prior_active_starts_sum"].idxmax())
        players.loc[promote, "preseason_role"] = (
            "rookie_starter"
            if bool(players.loc[promote, "is_rookie_at_cutoff"])
            else "starter"
        )
        starter_mask = players["preseason_role"].isin(("starter", "rookie_starter"))
    if starter_mask.sum() > 1:
        keep = (
            players.loc[starter_mask]
            .sort_values("prior_active_starts_sum", ascending=False)
            .index[0]
        )
        demote = starter_mask & (players.index != keep)
        players.loc[demote, "preseason_role"] = players.loc[demote].apply(
            lambda r: "rookie_backup" if r["is_rookie_at_cutoff"] else "backup",
            axis=1,
        )
        starter_mask = players.index == keep

    starter = players.loc[starter_mask].iloc[0]
    starter_starts = float(np.clip(starter["frozen_expected_starts"], 0.0, AVAIL_FULL_SEASON_GAMES))
    starter_partial = float(np.clip(starter["frozen_partial_exit_rate"], 0.0, 0.5))
    residual_games = max(0.0, float(SEASON_GAMES) - starter_starts)
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

    total = float(players["allocated_expected_starts"].sum())
    gap = float(SEASON_GAMES) - total
    if abs(gap) > 1e-9 and (~starter_mask).any():
        bench_idx = players.index[~starter_mask]
        bench_sum = float(players.loc[bench_idx, "allocated_expected_starts"].sum())
        if bench_sum > 0 and gap != 0:
            players.loc[bench_idx, "allocated_expected_starts"] *= (bench_sum + gap) / bench_sum
        elif gap > 0:
            weights = players.loc[bench_idx, "preseason_role"].map(residual_weight).astype(float)
            wsum = float(weights.sum()) or float(len(bench_idx))
            players.loc[bench_idx, "allocated_expected_starts"] += gap * (weights / wsum)
    elif abs(gap) > 1e-6 and not (~starter_mask).any():
        # Solo QB1 room: pin exactly to scheduled games rather than invent backups.
        players.loc[starter_mask, "allocated_expected_starts"] = float(SEASON_GAMES)

    players["allocated_expected_starts"] = players["allocated_expected_starts"].clip(lower=0.0)
    players["team_starts_conserved"] = float(players["allocated_expected_starts"].sum())
    players["backup_budget"] = backup_budget
    players["starter_partial_exposure"] = partial_exposure
    if abs(float(players["allocated_expected_starts"].sum()) - float(SEASON_GAMES)) > 1e-6:
        raise IncompleteQbRoom(
            f"QB-room expected starts do not conserve {SEASON_GAMES}: "
            f"{float(players['allocated_expected_starts'].sum())}"
        )
    return players


def allocate_league_expected_starts(
    *,
    history: pd.DataFrame,
    target_season: int,
    rooms: pd.DataFrame,
    team_col: str = "team",
) -> pd.DataFrame:
    """Run :func:`allocate_team_expected_starts` for every team in ``rooms``."""
    if rooms is None or rooms.empty:
        raise IncompleteQbRoom("league rooms frame is empty")
    parts = []
    for team, room in rooms.groupby(team_col, dropna=False):
        if pd.isna(team):
            raise IncompleteQbRoom("QB room has null team; refusing silent drop")
        allocated = allocate_team_expected_starts(
            history=history, target_season=target_season, room=room
        )
        allocated[team_col] = team
        parts.append(allocated)
    if not parts:
        raise IncompleteQbRoom("no teams available for expected-start allocation")
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
