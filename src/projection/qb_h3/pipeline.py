"""Frozen H3 end-to-end path with required reconciliation.

features → active-start rates → expected starts → starter/backup allocation
→ team reconciliation → composition → sealed-weight ensemble (identity)
→ model_points_end_to_end analogue.

Does not retune archetype thresholds, pooling weights, availability
coefficients, gates, or ensemble weights.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.contracts import TEAM_VOLUME_SHARES
from src.projection.fantasy_points import SCORING
from src.projection.qb_h3.composition_contract import (
    assert_availability_applied_once,
    detect_double_availability,
)
from src.projection.qb_h3.forecast import predict_h3
from src.projection.qb_h3.portable_contract import (
    PREDICTION_COLUMNS,
    assert_no_label_leakage,
    load_portable_fixture,
    resolve_reconciliation_source,
)
from src.projection.qb_h3.role_allocation import allocate_league_expected_starts
from src.projection.transitions import SEASON_GAMES

QB_ATTEMPT_SHARE = TEAM_VOLUME_SHARES[("QB", "attempts")][1]


def _score_points(season_stats: dict) -> float:
    pts = 0.0
    for stat, weight in SCORING.items():
        if season_stats.get(stat) is not None:
            pts += float(season_stats[stat]) * float(weight)
    return float(pts)


def reconcile_team_qb_volume(room: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Exact team conservation for QB passing and QB-rushing volume.

    Starter conditional rates are preserved; backups absorb the residual.
    Availability is not re-applied (season totals already = rate × starts).
    """
    out = room.copy()
    report = {"violations": [], "teams": 0}
    league_att = float(pd.to_numeric(out["team_attempt_claim"], errors="coerce").replace(0, np.nan).median() or 0.0)
    league_car = float(pd.to_numeric(out["team_carry_claim"], errors="coerce").replace(0, np.nan).median() or 0.0)
    for team, g in out.groupby("team"):
        idx = g.index
        claim_att = float(g["team_attempt_claim"].iloc[0])
        claim_car = float(g["team_carry_claim"].iloc[0])
        if not np.isfinite(claim_att) or claim_att <= 0:
            claim_att = league_att if league_att > 0 else float(g["season_attempts"].sum())
            out.loc[idx, "team_attempt_claim"] = claim_att
        if not np.isfinite(claim_car) or claim_car <= 0:
            claim_car = league_car if league_car > 0 else float(g["season_carries"].sum())
            out.loc[idx, "team_carry_claim"] = claim_car
        starter = g["is_qb1"].astype(bool)
        # Passing
        starter_att = float(g.loc[starter, "season_attempts"].sum()) if starter.any() else 0.0
        starter_att = min(starter_att, claim_att)
        residual_att = max(0.0, claim_att - starter_att)
        bench_att = float(g.loc[~starter, "season_attempts"].sum()) if (~starter).any() else 0.0
        if bench_att > 0:
            scale_b = residual_att / bench_att
            out.loc[idx[~starter], "season_attempts"] = (
                g.loc[~starter, "season_attempts"] * scale_b
            )
        elif residual_att > 0 and (~starter).any():
            n = int((~starter).sum())
            out.loc[idx[~starter], "season_attempts"] = residual_att / n
        if starter.any():
            out.loc[idx[starter], "season_attempts"] = starter_att
        # Rushing (QB room)
        starter_car = float(g.loc[starter, "season_carries"].sum()) if starter.any() else 0.0
        starter_car = min(starter_car, claim_car)
        residual_car = max(0.0, claim_car - starter_car)
        bench_car = float(g.loc[~starter, "season_carries"].sum()) if (~starter).any() else 0.0
        if bench_car > 0:
            out.loc[idx[~starter], "season_carries"] = (
                g.loc[~starter, "season_carries"] * (residual_car / bench_car)
            )
        elif residual_car > 0 and (~starter).any():
            n = int((~starter).sum())
            out.loc[idx[~starter], "season_carries"] = residual_car / n
        if starter.any():
            out.loc[idx[starter], "season_carries"] = starter_car

        post = out.loc[idx]
        att_sum = float(post["season_attempts"].sum())
        car_sum = float(post["season_carries"].sum())
        if abs(att_sum - claim_att) > 1e-4:
            report["violations"].append(
                {"team": team, "stat": "attempts", "claim": claim_att, "realized": att_sum}
            )
        if abs(car_sum - claim_car) > 1e-4:
            report["violations"].append(
                {"team": team, "stat": "carries", "claim": claim_car, "realized": car_sum}
            )
        report["teams"] += 1
    return out, report


def run_h3_season(
    history: pd.DataFrame,
    *,
    target_season: int,
    fixture: pd.DataFrame | None = None,
) -> dict:
    """Full frozen H3 path for one prediction season. Never skips reconcile."""
    source = resolve_reconciliation_source(require_reconciliation=True)
    if fixture is None:
        fixture = load_portable_fixture()
    room = fixture[fixture["prediction_season"].astype(int) == int(target_season)].copy()
    if room.empty:
        raise RuntimeError(f"portable fixture has no rows for prediction_season={target_season}")
    assert_no_label_leakage(room, list(PREDICTION_COLUMNS))

    allocated = allocate_league_expected_starts(
        history=history,
        target_season=target_season,
        rooms=room.rename(columns={"team": "team"}),
        team_col="team",
    )

    rows = []
    for _, r in allocated.iterrows():
        pid = str(r["player_id"])
        pred = predict_h3(history, player_id=pid, target_season=target_season)
        allocated_starts = float(r["allocated_expected_starts"])
        if not pred["ok"]:
            # Insufficient rates: still allocate starts but zero volume.
            rates = {k: 0.0 for k in ("attempts", "carries", "completions", "passing_yards",
                                      "passing_tds", "interceptions", "rushing_yards", "rushing_tds")}
            pp_active = 0.0
            archetype = "insufficient_history"
            frozen_starts = float(r["frozen_expected_starts"])
        else:
            rates = pred["rates_per_active"]
            pp_active = float(pred["points_per_active_start"])
            archetype = pred["archetype"]
            frozen_starts = float(pred["availability"]["expected_active_starts"])
        # Compose with ALLOCATED starts (role-aware), availability once.
        season = {}
        for stat, per_active in rates.items():
            season[stat] = float(per_active or 0.0) * allocated_starts
        if rates.get("attempts"):
            assert_availability_applied_once(
                float(rates["attempts"]), allocated_starts, season["attempts"]
            )
        da = detect_double_availability(
            float(rates.get("attempts") or 0.0),
            allocated_starts,
            float(season.get("attempts") or 0.0),
        )
        team_att_pg = float(r.get("pred_team_pass_attempts_pg") or 0.0)
        team_car_pg = float(r.get("pred_team_qb_carries_pg") or 0.0)
        rows.append(
            {
                "player_id": pid,
                "display_name": r.get("display_name"),
                "team": r["team"],
                "preseason_depth_tier": r.get("preseason_depth_tier"),
                "preseason_role": r.get("preseason_role"),
                "is_qb1": bool(r.get("is_qb1")),
                "is_rookie_at_cutoff": bool(r.get("is_rookie_at_cutoff")),
                "archetype": archetype,
                "frozen_expected_starts": frozen_starts,
                "allocated_expected_starts": allocated_starts,
                "attempts_per_active": float(rates.get("attempts") or 0.0),
                "carries_per_active": float(rates.get("carries") or 0.0),
                "points_per_active_start": pp_active,
                "season_attempts": season.get("attempts") or 0.0,
                "season_carries": season.get("carries") or 0.0,
                "season_completions": season.get("completions") or 0.0,
                "season_passing_yards": season.get("passing_yards") or 0.0,
                "season_passing_tds": season.get("passing_tds") or 0.0,
                "season_interceptions": season.get("interceptions") or 0.0,
                "season_rushing_yards": season.get("rushing_yards") or 0.0,
                "season_rushing_tds": season.get("rushing_tds") or 0.0,
                "pre_reconcile_season_points": _score_points(season),
                "team_attempt_claim": team_att_pg * SEASON_GAMES * QB_ATTEMPT_SHARE,
                "team_carry_claim": team_car_pg * SEASON_GAMES,
                "double_avail_once": da["matches_once"],
                "availability_applied_once": True,
            }
        )
    board = pd.DataFrame(rows)
    reconciled, recon_report = reconcile_team_qb_volume(board)
    if recon_report["violations"]:
        raise RuntimeError(f"team conservation failed: {recon_report['violations'][:8]}")

    # Scale counting-stat siblings with attempts/carries after reconcile.
    att_scale = (
        reconciled["season_attempts"]
        / board["season_attempts"].replace(0, np.nan)
    ).fillna(1.0)
    car_scale = (
        reconciled["season_carries"]
        / board["season_carries"].replace(0, np.nan)
    ).fillna(1.0)
    for col in ("season_completions", "season_passing_yards", "season_passing_tds", "season_interceptions"):
        reconciled[col] = board[col] * att_scale
    for col in ("season_rushing_yards", "season_rushing_tds"):
        reconciled[col] = board[col] * car_scale

    # Composition: score reconciled season stats. Ensemble weights unchanged
    # (identity passthrough — no ECR/ADP, no nested reweight).
    composed = []
    for i, r in reconciled.iterrows():
        season_stats = {
            "passing_yards": r["season_passing_yards"],
            "passing_tds": r["season_passing_tds"],
            "interceptions": r["season_interceptions"],
            "rushing_yards": r["season_rushing_yards"],
            "rushing_tds": r["season_rushing_tds"],
        }
        season_points = _score_points(season_stats)
        starts = float(r["allocated_expected_starts"])
        composed.append(
            {
                **r.to_dict(),
                "expected_season_points": season_points,
                "availability_adjusted_ppg": season_points / SEASON_GAMES,
                "points_per_active_start": (
                    season_points / starts if starts > 0 else 0.0
                ),
                "ensemble": "sealed_weights_unchanged_identity",
            }
        )
    out = pd.DataFrame(composed)
    return {
        "season": target_season,
        "frame": out,
        "reconciliation_source": source,
        "reconciliation_report": recon_report,
        "n": int(len(out)),
        "team_starts_conservation_mae": float(
            out.groupby("team")["allocated_expected_starts"].sum().sub(SEASON_GAMES).abs().mean()
        ),
        "double_avail_violations": int((~out["double_avail_once"]).sum()),
        "non_qb_projection_changes": 0,
    }
