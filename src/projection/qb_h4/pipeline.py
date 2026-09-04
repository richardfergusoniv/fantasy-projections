"""H4 forecast + end-to-end path (isolated from H3 defaults)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.contracts import TEAM_VOLUME_SHARES
from src.projection.fantasy_points import SCORING
from src.projection.qb_active_archetype.active_rates import expected_availability
from src.projection.qb_h3.composition_contract import (
    assert_availability_applied_once,
    detect_double_availability,
)
from src.projection.qb_h3.pipeline import reconcile_team_qb_volume
from src.projection.qb_h3.portable_contract import (
    PREDICTION_COLUMNS,
    assert_no_label_leakage,
    load_portable_fixture,
    resolve_reconciliation_source,
)
from src.projection.qb_h3.role_allocation import allocate_league_expected_starts
from src.projection.qb_h4.decision_policy import MODEL_ID
from src.projection.qb_h4.experience import classify_experience
from src.projection.qb_h4.priors import h4_active_rates
from src.projection.transitions import SEASON_GAMES

QB_ATTEMPT_SHARE = TEAM_VOLUME_SHARES[("QB", "attempts")][1]


def _score_points(season_stats: dict) -> float:
    pts = 0.0
    for stat, weight in SCORING.items():
        if season_stats.get(stat) is not None:
            pts += float(season_stats[stat]) * float(weight)
    return float(pts)


def predict_h4(
    history: pd.DataFrame,
    *,
    player_id: str,
    target_season: int,
    is_rookie_at_cutoff: bool = False,
    preseason_role: str | None = None,
) -> dict:
    """H4 component forecast. Availability coefficients unchanged from H3."""
    exp = classify_experience(
        player_id=player_id,
        target_season=target_season,
        history=history,
        is_rookie_at_cutoff=is_rookie_at_cutoff,
    )
    avail = expected_availability(history, player_id=player_id, target_season=target_season)
    rates_pack = h4_active_rates(
        history,
        player_id=player_id,
        target_season=target_season,
        experience_class=exp["experience_class"],
        preseason_role=preseason_role,
    )
    rates = rates_pack["rates"]
    if rates.get("attempts") is None or rates.get("carries") is None:
        return {
            "ok": False,
            "reason": "missing_active_rates",
            "experience_class": exp["experience_class"],
            "model_id": MODEL_ID,
        }
    # Season totals use allocated starts downstream; here compose with frozen
    # availability for the per-player component audit only.
    starts = float(avail["expected_active_starts"])
    partial = float(avail.get("partial_exit_rate") or 0.0)
    effective = starts * (1.0 - 0.5 * partial)
    season = {k: float(v) * effective if v is not None else None for k, v in rates.items()}
    assert_availability_applied_once(float(rates["attempts"]), effective, float(season["attempts"]))
    pp_active = 0.0
    for stat, pts in SCORING.items():
        if rates.get(stat) is not None:
            pp_active += float(rates[stat]) * float(pts)
    return {
        "ok": True,
        "model_id": MODEL_ID,
        "experience_class": exp["experience_class"],
        "experience": exp,
        "archetype": rates_pack["archetype"],
        "availability": avail,
        "rates_per_active": rates,
        "rates_method": rates_pack["method"],
        "shrink": rates_pack.get("shrink"),
        "peer_n": rates_pack.get("peer_n"),
        "season_stats": season,
        "points_per_active_start": pp_active,
        "expected_season_points": pp_active * effective,
        "availability_adjusted_ppg": (pp_active * effective) / SEASON_GAMES,
        "effective_starts": effective,
        "availability_applied_once": True,
    }


def run_h4_season(
    history: pd.DataFrame,
    *,
    target_season: int,
    fixture: pd.DataFrame | None = None,
) -> dict:
    """Full H4 path with role allocation + portable team reconciliation."""
    source = resolve_reconciliation_source(require_reconciliation=True)
    if fixture is None:
        fixture = load_portable_fixture()
    room = fixture[fixture["prediction_season"].astype(int) == int(target_season)].copy()
    if room.empty:
        raise RuntimeError(f"portable fixture has no rows for prediction_season={target_season}")
    assert_no_label_leakage(room, list(PREDICTION_COLUMNS))

    allocated = allocate_league_expected_starts(
        history=history, target_season=target_season, rooms=room, team_col="team"
    )

    rows = []
    for _, r in allocated.iterrows():
        pid = str(r["player_id"])
        role = r.get("preseason_role")
        is_rookie = bool(r.get("is_rookie_at_cutoff"))
        pred = predict_h4(
            history,
            player_id=pid,
            target_season=target_season,
            is_rookie_at_cutoff=is_rookie,
            preseason_role=role,
        )
        allocated_starts = float(r["allocated_expected_starts"])
        if not pred["ok"]:
            rates = {k: 0.0 for k in (
                "attempts", "carries", "completions", "passing_yards",
                "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
            )}
            archetype = "insufficient_history"
            exp_class = pred.get("experience_class", "insufficient_history")
            frozen_starts = float(r["frozen_expected_starts"])
            method = "failed"
        else:
            rates = pred["rates_per_active"]
            archetype = pred["archetype"]
            exp_class = pred["experience_class"]
            frozen_starts = float(pred["availability"]["expected_active_starts"])
            method = pred["rates_method"]
        season = {stat: float(rates.get(stat) or 0.0) * allocated_starts for stat in rates}
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
                "preseason_role": role,
                "is_qb1": bool(r.get("is_qb1")),
                "is_rookie_at_cutoff": is_rookie,
                "experience_class": exp_class,
                "archetype": archetype,
                "rates_method": method,
                "model_id": MODEL_ID,
                "frozen_expected_starts": frozen_starts,
                "allocated_expected_starts": allocated_starts,
                "attempts_per_active": float(rates.get("attempts") or 0.0),
                "carries_per_active": float(rates.get("carries") or 0.0),
                "season_attempts": season.get("attempts") or 0.0,
                "season_carries": season.get("carries") or 0.0,
                "season_completions": season.get("completions") or 0.0,
                "season_passing_yards": season.get("passing_yards") or 0.0,
                "season_passing_tds": season.get("passing_tds") or 0.0,
                "season_interceptions": season.get("interceptions") or 0.0,
                "season_rushing_yards": season.get("rushing_yards") or 0.0,
                "season_rushing_tds": season.get("rushing_tds") or 0.0,
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
    att_scale = (
        reconciled["season_attempts"] / board["season_attempts"].replace(0, np.nan)
    ).fillna(1.0)
    car_scale = (
        reconciled["season_carries"] / board["season_carries"].replace(0, np.nan)
    ).fillna(1.0)
    for col in ("season_completions", "season_passing_yards", "season_passing_tds", "season_interceptions"):
        reconciled[col] = board[col] * att_scale
    for col in ("season_rushing_yards", "season_rushing_tds"):
        reconciled[col] = board[col] * car_scale

    composed = []
    for _, r in reconciled.iterrows():
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
                "points_per_active_start": season_points / starts if starts > 0 else 0.0,
                "ensemble": "sealed_weights_unchanged_identity",
            }
        )
    out = pd.DataFrame(composed)
    return {
        "season": target_season,
        "model_id": MODEL_ID,
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
