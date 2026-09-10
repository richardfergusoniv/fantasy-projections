"""Joint QB-room allocation: starter share first, backups residual.

Enabled only when ``CompositionContext.qb_joint_room_allocation`` is True.
Shipped default remains the existing ``reconcile_team_volume`` path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.projection.artifacts import load_reconcile_calibration, reconcile_alpha_for
from src.projection.contracts import TEAM_VOLUME_SHARES, TEAM_VOLUME_SIBLINGS
from src.projection.qb_repair.allocation import (
    AllocationParams,
    estimate_starter_backup_shares,
)
from src.projection.team_reconcile import _row_exposure
from src.projection.transitions import SEASON_GAMES


@dataclass(frozen=True)
class JointAllocationReport:
    teams: int
    starters_assigned: int
    conservation_violations: list
    allocation: dict


def _is_tier1(rows: pd.DataFrame) -> pd.Series:
    if "depth_tier" in rows.columns:
        return pd.to_numeric(rows["depth_tier"], errors="coerce").eq(1.0)
    if "depth_rank" in rows.columns:
        return pd.to_numeric(rows["depth_rank"], errors="coerce").eq(1.0)
    return rows["pred_pg"] == rows.groupby("team")["pred_pg"].transform("max")


def _expected_missed_games(rows: pd.DataFrame, starter_ids: set[str]) -> float:
    """Expected starter missed games from Gate-A raw exposure when present."""
    starter = rows[rows["player_id"].astype(str).isin(starter_ids)]
    if starter.empty:
        return 0.0
    raw = pd.to_numeric(starter.get("projected_games_raw"), errors="coerce")
    games = pd.to_numeric(starter.get("projected_games"), errors="coerce")
    # After full-season baseline, projected_games is 17; raw holds Gate A.
    if raw is not None and raw.notna().any():
        raw_g = float(raw.dropna().iloc[0])
        return float(max(0.0, SEASON_GAMES - raw_g))
    if games is not None and games.notna().any():
        return float(max(0.0, SEASON_GAMES - float(games.dropna().iloc[0])))
    return 0.0


def reconcile_qb_joint_room(
    df: pd.DataFrame,
    *,
    allocation: AllocationParams | None = None,
    target_season: int | None = None,
    alpha: float | None = None,
    volume_shares: dict | None = None,
    volume_siblings: dict | None = None,
    conserve_tol: float = 1e-6,
) -> tuple[pd.DataFrame, dict]:
    """Allocate QB passing volume with starter-first residual backups.

    Procedure per team (attempts and passing yards separately):

    1. Establish team passing claim = team anchor × season games × room share.
    2. Assign the healthy projected QB1 the historical starter share of that claim
       (soft-blended with alpha toward the raw starter prediction).
    3. Size backup residual from expected starter missed games, not from the
       backup's unrestricted raw rate × full season.
    4. Distribute residual across backups proportional to their raw rates.
    5. Conserve exactly onto the team claim.
    """
    if allocation is None:
        season = int(target_season or df.get("season", pd.Series([2026])).iloc[0])
        allocation = estimate_starter_backup_shares(target_season=season)

    calibration = load_reconcile_calibration()
    use_per_cell = alpha is None
    shares = TEAM_VOLUME_SHARES if volume_shares is None else volume_shares
    siblings = TEAM_VOLUME_SIBLINGS if volume_siblings is None else volume_siblings
    out = df.copy()
    if "team_volume_scale" not in out.columns:
        out["team_volume_scale"] = 1.0

    report = {
        "mode": "joint_qb_room",
        "allocation": allocation.__dict__,
        "teams": 0,
        "starters_assigned": 0,
        "conservation_violations": [],
    }
    if out.empty:
        return out, report

    exposure = _row_exposure(out)
    qb_cells = [
        (("QB", "attempts"), "starter_attempt_share"),
        (("QB", "passing_yards"), "starter_yard_share"),
    ]

    for (position, stat), share_attr in qb_cells:
        if (position, stat) not in shares:
            continue
        anchor_col, room_share = shares[(position, stat)]
        cell_alpha = (
            reconcile_alpha_for(position, stat, calibration)
            if use_per_cell
            else float(alpha)
        )
        if not cell_alpha or anchor_col not in out.columns:
            continue
        group = [stat] + list(siblings.get((position, stat), ()))
        anchored = out["position"].eq(position) & out["stat"].eq(stat)
        if not anchored.any():
            continue
        rows = out.loc[anchored].copy()
        season_pred = pd.to_numeric(rows["pred_pg"], errors="coerce") * exposure.loc[rows.index]
        target = (
            pd.to_numeric(rows.drop_duplicates("team").set_index("team")[anchor_col], errors="coerce")
            * SEASON_GAMES
            * room_share
        )
        starter_share = float(getattr(allocation, share_attr))
        is_starter = _is_tier1(rows)
        starter_ids = set(rows.loc[is_starter, "player_id"].astype(str))

        for team, team_rows in rows.groupby("team"):
            tgt = float(target.get(team, np.nan))
            if not np.isfinite(tgt) or tgt <= 0:
                continue
            idx = team_rows.index
            pred = season_pred.loc[idx]
            starter_idx = team_rows.index[team_rows["player_id"].astype(str).isin(starter_ids)]
            bench_idx = team_rows.index.difference(starter_idx)
            starter_pred = float(pred.loc[starter_idx].sum()) if len(starter_idx) else 0.0
            bench_pred = float(pred.loc[bench_idx].sum()) if len(bench_idx) else 0.0

            # Expected backup residual from starter missed games.
            missed = _expected_missed_games(team_rows, starter_ids)
            # Backup conditional rate: use bench raw rate but scale exposure to
            # expected missed games rather than a full season.
            if bench_pred > 0 and missed > 0:
                bench_rate_per_game = bench_pred / max(
                    float(exposure.loc[bench_idx].mean()) if len(bench_idx) else SEASON_GAMES,
                    1e-6,
                )
                # Cap residual by historical complement of starter share.
                residual_cap = tgt * (1.0 - starter_share)
                residual_from_misses = bench_rate_per_game * missed
                desired_bench = float(min(residual_cap, max(0.0, residual_from_misses)))
            elif bench_pred > 0:
                # Healthy full-season starter: backups get only the historical
                # complement share, never their inflated raw full-season claim.
                desired_bench = tgt * (1.0 - starter_share)
            else:
                desired_bench = 0.0

            desired_starter = tgt - desired_bench
            # Soft alpha toward desired; never let backups squeeze starter below share.
            min_starter = tgt * starter_share
            desired_starter = max(desired_starter, min_starter)
            desired_bench = max(0.0, tgt - desired_starter)

            new_starter = starter_pred + cell_alpha * (desired_starter - starter_pred) if starter_pred > 0 or desired_starter > 0 else 0.0
            new_bench = bench_pred + cell_alpha * (desired_bench - bench_pred) if bench_pred > 0 or desired_bench > 0 else 0.0
            total_new = new_starter + new_bench
            if total_new > 0:
                scale = tgt / total_new
                new_starter *= scale
                new_bench *= scale

            starter_factor = (new_starter / starter_pred) if starter_pred > 0 else 1.0
            bench_factor = (new_bench / bench_pred) if bench_pred > 0 else 1.0

            touched = out["position"].eq(position) & out["stat"].isin(group) & out["team"].eq(team)
            is_touch_starter = out.loc[touched, "player_id"].astype(str).isin(starter_ids)
            factor = pd.Series(
                np.where(is_touch_starter, starter_factor, bench_factor),
                index=out.loc[touched].index,
            )
            for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                if col in out.columns:
                    out.loc[touched, col] = (
                        pd.to_numeric(out.loc[touched, col], errors="coerce") * factor
                    )
            out.loc[touched, "team_volume_scale"] = factor.astype(float)
            report["teams"] += 1
            if len(starter_idx):
                report["starters_assigned"] += 1

            post = out.loc[anchored & out["team"].eq(team)]
            post_exposure = _row_exposure(out).loc[post.index]
            post_sum = float(
                (pd.to_numeric(post["pred_pg"], errors="coerce") * post_exposure).sum()
            )
            if abs(post_sum - tgt) > max(conserve_tol, 0.05 * abs(tgt)):
                report["conservation_violations"].append(
                    {
                        "team": team,
                        "stat": stat,
                        "target": tgt,
                        "realized": post_sum,
                        "rel_error": (post_sum - tgt) / tgt if tgt else None,
                        "missed_games": missed,
                    }
                )

    return out, report
