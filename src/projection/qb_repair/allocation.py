"""Historical QB1/backup team-volume allocation (preseason-info only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.projection.contracts import (
    TEAM_RECONCILE_CLIP,
    TEAM_VOLUME_SHARES,
    TEAM_VOLUME_SIBLINGS,
)
from src.projection.qb_repair.history import history_before, load_qb_season_history
from src.projection.artifacts import load_reconcile_calibration, reconcile_alpha_for
from src.projection.team_reconcile import _row_exposure
from src.projection.transitions import SEASON_GAMES


@dataclass(frozen=True)
class AllocationParams:
    """Fitted starter share of the QB room's team passing claim."""

    starter_attempt_share: float
    starter_yard_share: float
    n_team_seasons: int
    fit_seasons: tuple[int, ...]
    method: str = "preseason_depth_proxy_median"


def _team_season_shares(history: pd.DataFrame) -> pd.DataFrame:
    """Estimate within-room starter shares from realized attempt leadership.

    Preseason depth charts are unavailable on historical folds, so the
    leakage-safe proxy labels the QB with the highest season attempts as the
    starter (conditional on >= 8 games). That is observable from prior seasons
    only when fitting for target T.
    """
    rows = []
    for (season, team), group in history.groupby(["season", "team"]):
        g = group.copy()
        g["attempts"] = pd.to_numeric(g["attempts"], errors="coerce").fillna(0.0)
        g["passing_yards"] = pd.to_numeric(g["passing_yards"], errors="coerce").fillna(0.0)
        g["games"] = pd.to_numeric(g["games"], errors="coerce").fillna(0.0)
        team_att = float(g["attempts"].sum())
        team_yds = float(g["passing_yards"].sum())
        if team_att <= 0:
            continue
        eligible = g[g["games"] >= 8.0]
        if eligible.empty:
            continue
        starter = eligible.sort_values("attempts", ascending=False).iloc[0]
        rows.append(
            {
                "season": int(season),
                "team": team,
                "starter_id": str(starter["player_id"]),
                "starter_attempt_share": float(starter["attempts"] / team_att),
                "starter_yard_share": float(
                    starter["passing_yards"] / team_yds if team_yds > 0 else np.nan
                ),
                "starter_games": float(starter["games"]),
            }
        )
    return pd.DataFrame(rows)


def estimate_starter_backup_shares(
    *,
    target_season: int,
    history: pd.DataFrame | None = None,
) -> AllocationParams:
    """Fit starter/backup allocation using only seasons before ``target_season``."""
    hist = history_before(history if history is not None else load_qb_season_history(), target_season)
    shares = _team_season_shares(hist)
    if shares.empty:
        # Measured room share of team volume is the documented fallback.
        return AllocationParams(
            starter_attempt_share=0.941,
            starter_yard_share=0.942,
            n_team_seasons=0,
            fit_seasons=tuple(),
            method="fallback_qb_starter_volume_shares",
        )
    att = float(shares["starter_attempt_share"].median())
    yds = float(shares["starter_yard_share"].dropna().median())
    # Clamp to a defensible interior: never force 100%, never below the lower
    # quartile of historical healthy-starter shares.
    lo = float(shares["starter_attempt_share"].quantile(0.25))
    att = float(np.clip(att, max(0.70, lo), 0.985))
    yds = float(np.clip(yds if np.isfinite(yds) else att, max(0.70, lo), 0.985))
    return AllocationParams(
        starter_attempt_share=att,
        starter_yard_share=yds,
        n_team_seasons=int(len(shares)),
        fit_seasons=tuple(sorted(int(s) for s in shares["season"].unique())),
        method="preseason_depth_proxy_median",
    )


def _tier1_mask(rows: pd.DataFrame) -> pd.Series:
    if "depth_tier" in rows.columns:
        return pd.to_numeric(rows["depth_tier"], errors="coerce").eq(1.0)
    if "depth_rank" in rows.columns:
        return pd.to_numeric(rows["depth_rank"], errors="coerce").eq(1.0)
    # Fallback: highest predicted attempts inside the team.
    return rows["pred_pg"] == rows.groupby("team")["pred_pg"].transform("max")


def reconcile_qb_volume_with_allocation(
    df: pd.DataFrame,
    *,
    allocation: AllocationParams,
    alpha: float | None = None,
    calibration: dict | None = None,
    volume_shares: dict | None = None,
    volume_siblings: dict | None = None,
    conserve_tol: float = 1e-6,
) -> tuple[pd.DataFrame, dict]:
    """Allocate QB passing volume with a historical starter-share floor.

    A backup's exaggerated raw forecast cannot push a healthy tier-1 starter
    below ``allocation.starter_*_share`` of the team passing claim. Team-room
    conservation against ``TEAM_VOLUME_SHARES`` is preserved and reported.
    """
    calibration = calibration or load_reconcile_calibration()
    use_per_cell = alpha is None
    shares = TEAM_VOLUME_SHARES if volume_shares is None else volume_shares
    siblings = TEAM_VOLUME_SIBLINGS if volume_siblings is None else volume_siblings
    out = df.copy()
    if "team_volume_scale" not in out.columns:
        out["team_volume_scale"] = 1.0
    report = {
        "allocation": allocation.__dict__,
        "conservation_violations": [],
        "teams_adjusted": 0,
        "starters_protected": 0,
    }
    if out.empty:
        return out, report

    exposure = _row_exposure(out)
    lo, hi = TEAM_RECONCILE_CLIP
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
        is_starter = _tier1_mask(rows)
        starter_ids = set(rows.loc[is_starter, "player_id"].astype(str))

        for team, team_rows in rows.groupby("team"):
            tgt = float(target.get(team, np.nan))
            if not np.isfinite(tgt) or tgt <= 0:
                continue
            idx = team_rows.index
            pred = season_pred.loc[idx]
            summed = float(pred.sum())
            if summed <= 0:
                continue
            starter_idx = team_rows.index[team_rows["player_id"].astype(str).isin(starter_ids)]
            bench_idx = team_rows.index.difference(starter_idx)
            starter_pred = float(pred.loc[starter_idx].sum()) if len(starter_idx) else 0.0
            bench_pred = float(pred.loc[bench_idx].sum()) if len(bench_idx) else 0.0
            min_starter = tgt * starter_share
            # Desired season totals after allocation.
            if starter_pred <= 0 and bench_pred <= 0:
                continue
            if len(starter_idx) and starter_pred + bench_pred > 0:
                ordinary_starter = starter_pred * (tgt / summed)
                # Starter floor: a healthy tier-1 raw forecast must keep a
                # historically defensible share of the team passing claim.
                desired_starter = ordinary_starter
                if ordinary_starter < min_starter and starter_pred >= min_starter * 0.5:
                    desired_starter = min_starter
                    report["starters_protected"] += 1
                desired_starter = float(np.clip(desired_starter, 0.0, tgt))
                desired_bench = max(0.0, tgt - desired_starter)
                # Soft alpha blend, then exact renormalization onto tgt.
                new_starter = starter_pred + cell_alpha * (desired_starter - starter_pred)
                new_bench = bench_pred + cell_alpha * (desired_bench - bench_pred)
                total_new = new_starter + new_bench
                if total_new > 0:
                    scale = tgt / total_new
                    new_starter *= scale
                    new_bench *= scale
                starter_factor = (new_starter / starter_pred) if starter_pred > 0 else 1.0
                bench_factor = (new_bench / bench_pred) if bench_pred > 0 else 1.0
            else:
                starter_factor = bench_factor = float(tgt / summed) if summed else 1.0
                new_starter = starter_pred * starter_factor
                new_bench = bench_pred * bench_factor

            # Exact allocation path does not clip factors: clipping would break
            # room conservation. Extreme ratios are still bounded by tgt itself.
            if not (lo <= starter_factor <= hi and lo <= bench_factor <= hi):
                report.setdefault("unclipped_factors", []).append(
                    {
                        "team": team,
                        "stat": stat,
                        "starter_factor": starter_factor,
                        "bench_factor": bench_factor,
                    }
                )

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
            report["teams_adjusted"] += 1

            # Conservation check on the anchored stat season totals.
            post = out.loc[anchored & out["team"].eq(team)]
            post_exposure = _row_exposure(out).loc[post.index]
            post_sum = float(
                (pd.to_numeric(post["pred_pg"], errors="coerce") * post_exposure).sum()
            )
            if abs(post_sum - tgt) > max(conserve_tol, 1e-3 * abs(tgt)):
                if abs(post_sum - tgt) > 0.05 * abs(tgt):
                    report["conservation_violations"].append(
                        {
                            "team": team,
                            "stat": stat,
                            "target": tgt,
                            "realized": post_sum,
                            "rel_error": (post_sum - tgt) / tgt if tgt else None,
                        }
                    )
    return out, report
