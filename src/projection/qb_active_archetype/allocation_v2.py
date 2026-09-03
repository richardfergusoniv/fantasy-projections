"""Simplify starter season mapping in joint v2 allocation."""
from __future__ import annotations

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


def _is_tier1(rows: pd.DataFrame) -> pd.Series:
    if "depth_tier" in rows.columns:
        return pd.to_numeric(rows["depth_tier"], errors="coerce").eq(1.0)
    if "depth_rank" in rows.columns:
        return pd.to_numeric(rows["depth_rank"], errors="coerce").eq(1.0)
    return rows["pred_pg"] == rows.groupby("team")["pred_pg"].transform("max")


def reconcile_qb_joint_room_v2(
    df: pd.DataFrame,
    *,
    allocation: AllocationParams | None = None,
    target_season: int | None = None,
    alpha: float | None = None,
    volume_shares: dict | None = None,
    volume_siblings: dict | None = None,
    expected_active_starts: dict[str, float] | None = None,
    conserve_tol: float = 1e-6,
) -> tuple[pd.DataFrame, dict]:
    """Active-conditional starter allocation with backup residual only.

    - Team claim established from anchors.
    - Starter season volume = max(active_rate × expected_starts, historical share).
    - Board stores rates under full-season exposure; starter pred_pg becomes
      season_volume / exposure so backups cannot compress the active rate.
    - Backup volume is residual; package usage allowed only in the residual.
    """
    if allocation is None:
        allocation = estimate_starter_backup_shares(target_season=int(target_season or 2026))

    calibration = load_reconcile_calibration()
    use_per_cell = alpha is None
    shares = TEAM_VOLUME_SHARES if volume_shares is None else volume_shares
    siblings = TEAM_VOLUME_SIBLINGS if volume_siblings is None else volume_siblings
    out = df.copy()
    if "team_volume_scale" not in out.columns:
        out["team_volume_scale"] = 1.0
    expected_active_starts = expected_active_starts or {}

    report = {
        "mode": "joint_qb_room_v2_active_conditional",
        "allocation": allocation.__dict__,
        "teams": 0,
        "starters_assigned": 0,
        "conservation_violations": [],
        "starter_active_rates": {},
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
            starter_mask = team_rows["player_id"].astype(str).isin(starter_ids)
            starter_rows = team_rows[starter_mask]
            bench_rows = team_rows[~starter_mask]
            if starter_rows.empty:
                continue

            starter_pid = str(starter_rows["player_id"].iloc[0])
            starter_idx = starter_rows.index
            active_rate = float(pd.to_numeric(starter_rows["pred_pg"], errors="coerce").iloc[0])
            exp_starts = float(
                expected_active_starts.get(starter_pid, SEASON_GAMES * starter_share)
            )
            exp_starts = float(np.clip(exp_starts, 1.0, SEASON_GAMES))
            starter_season = max(active_rate * exp_starts, tgt * starter_share)
            starter_season = min(starter_season, tgt)  # cannot exceed team claim
            bench_season = max(0.0, tgt - starter_season)

            board_exp = float(exposure.loc[starter_idx].iloc[0])
            desired_starter_pg = starter_season / max(board_exp, 1e-6)
            raw_starter_pg = active_rate
            new_starter_pg = raw_starter_pg + cell_alpha * (desired_starter_pg - raw_starter_pg)
            if cell_alpha >= 0.99:
                new_starter_pg = desired_starter_pg
            # Hard rule: never below the season claim implied by active×starts.
            new_starter_pg = max(new_starter_pg, desired_starter_pg)

            # Apply starter
            factor_s = (new_starter_pg / raw_starter_pg) if raw_starter_pg > 0 else 1.0
            touched_s = (
                out["position"].eq(position)
                & out["stat"].isin(group)
                & out["team"].eq(team)
                & out["player_id"].astype(str).eq(starter_pid)
            )
            for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                if col in out.columns:
                    out.loc[touched_s, col] = (
                        pd.to_numeric(out.loc[touched_s, col], errors="coerce") * factor_s
                    )
            out.loc[touched_s, "team_volume_scale"] = float(factor_s)

            # Bench residual
            if not bench_rows.empty:
                bench_raw = pd.to_numeric(bench_rows["pred_pg"], errors="coerce").fillna(0.0)
                bench_exp = exposure.loc[bench_rows.index].replace(0, np.nan).fillna(SEASON_GAMES)
                bench_claim = bench_raw * bench_exp
                total_claim = float(bench_claim.sum())
                weights = (
                    bench_claim / total_claim
                    if total_claim > 0
                    else pd.Series(1.0 / len(bench_rows), index=bench_rows.index)
                )
                for idx in bench_rows.index:
                    share = float(weights.loc[idx])
                    exp_i = float(exposure.loc[idx])
                    desired_pg = (bench_season * share) / max(exp_i, 1e-6)
                    raw_pg = float(pd.to_numeric(out.loc[idx, "pred_pg"], errors="coerce") or 0.0)
                    new_pg = raw_pg + cell_alpha * (desired_pg - raw_pg)
                    if cell_alpha >= 0.99:
                        new_pg = desired_pg
                    factor = (new_pg / raw_pg) if raw_pg > 0 else (0.0 if new_pg == 0 else 1.0)
                    touched = (
                        out["position"].eq(position)
                        & out["stat"].isin(group)
                        & out["team"].eq(team)
                        & out["player_id"].astype(str).eq(str(out.loc[idx, "player_id"]))
                    )
                    for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                        if col in out.columns:
                            out.loc[touched, col] = (
                                pd.to_numeric(out.loc[touched, col], errors="coerce") * factor
                            )
                    out.loc[touched, "team_volume_scale"] = float(factor)

            # Exact conserve: shrink backups only — never reduce starter below
            # the active×starts season claim.
            post = out.loc[anchored & out["team"].eq(team)]
            post_exp = exposure.loc[post.index]
            post_sum = float((pd.to_numeric(post["pred_pg"], errors="coerce") * post_exp).sum())
            starter_post = out.loc[
                anchored
                & out["team"].eq(team)
                & out["player_id"].astype(str).eq(starter_pid)
            ]
            starter_sum = float(
                (
                    pd.to_numeric(starter_post["pred_pg"], errors="coerce")
                    * exposure.loc[starter_post.index]
                ).sum()
            )
            bench_idx = post.index.difference(starter_post.index)
            if abs(post_sum - tgt) > max(conserve_tol, 1e-6):
                if post_sum > tgt and len(bench_idx):
                    overflow = post_sum - tgt
                    bench_sum = float(
                        (
                            pd.to_numeric(out.loc[bench_idx, "pred_pg"], errors="coerce")
                            * exposure.loc[bench_idx]
                        ).sum()
                    )
                    if bench_sum > 0:
                        # Prefer cutting backups; if still over, cut into package only.
                        cut = min(overflow, bench_sum)
                        fix_b = (bench_sum - cut) / bench_sum
                        for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                            if col in out.columns:
                                out.loc[bench_idx, col] = (
                                    pd.to_numeric(out.loc[bench_idx, col], errors="coerce") * fix_b
                                )
                        out.loc[bench_idx, "team_volume_scale"] = (
                            pd.to_numeric(out.loc[bench_idx, "team_volume_scale"], errors="coerce")
                            * fix_b
                        )
                    # Recompute; if still over target, scale starter only down to
                    # min(starter_sum, tgt) — backups already ~0.
                    post = out.loc[anchored & out["team"].eq(team)]
                    post_sum = float(
                        (pd.to_numeric(post["pred_pg"], errors="coerce") * exposure.loc[post.index]).sum()
                    )
                    if post_sum > tgt + 1e-6:
                        starter_idx2 = starter_post.index
                        s_sum = float(
                            (
                                pd.to_numeric(out.loc[starter_idx2, "pred_pg"], errors="coerce")
                                * exposure.loc[starter_idx2]
                            ).sum()
                        )
                        if s_sum > 0:
                            fix_s = min(1.0, tgt / s_sum)
                            for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                                if col in out.columns:
                                    out.loc[starter_idx2, col] = (
                                        pd.to_numeric(out.loc[starter_idx2, col], errors="coerce")
                                        * fix_s
                                    )
                elif post_sum < tgt and len(bench_idx) == 0:
                    # Give residual back to starter.
                    fix_s = tgt / max(starter_sum, 1e-6)
                    for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                        if col in out.columns:
                            out.loc[starter_post.index, col] = (
                                pd.to_numeric(out.loc[starter_post.index, col], errors="coerce")
                                * fix_s
                            )
                elif post_sum < tgt and len(bench_idx):
                    # Add residual to backups proportionally.
                    gap = tgt - post_sum
                    bench_sum = float(
                        (
                            pd.to_numeric(out.loc[bench_idx, "pred_pg"], errors="coerce")
                            * exposure.loc[bench_idx]
                        ).sum()
                    )
                    if bench_sum > 0:
                        fix_b = (bench_sum + gap) / bench_sum
                    else:
                        # assign gap to first backup evenly via rate
                        fix_b = None
                    if fix_b is not None:
                        for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
                            if col in out.columns:
                                out.loc[bench_idx, col] = (
                                    pd.to_numeric(out.loc[bench_idx, col], errors="coerce") * fix_b
                                )
                    else:
                        first = bench_idx[0]
                        exp_i = float(exposure.loc[first])
                        out.loc[first, "pred_pg"] = float(out.loc[first, "pred_pg"] or 0) + gap / max(exp_i, 1e-6)

            report["teams"] += 1
            report["starters_assigned"] += 1
            final_pg = float(pd.to_numeric(out.loc[starter_idx, "pred_pg"], errors="coerce").iloc[0])
            report["starter_active_rates"][f"{team}:{stat}"] = {
                "player_id": starter_pid,
                "active_rate_in": active_rate,
                "expected_active_starts": exp_starts,
                "starter_season": float(final_pg * board_exp),
                "bench_season": bench_season,
                "board_pg_out": final_pg,
            }

    return out, report
