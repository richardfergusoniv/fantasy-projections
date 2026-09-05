"""Apply active-start + archetype candidate to a long projection board."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.composition import compose_board, shipped_context
from src.projection.qb_active_archetype.active_rates import (
    expected_availability,
    pooled_active_rate,
)
from src.projection.qb_active_archetype.allocation_v2 import reconcile_qb_joint_room_v2
from src.projection.qb_active_archetype.archetypes import hierarchical_rush_priors
from src.projection.qb_repair.apply_board import non_qb_invariance_check, score_long_to_fantasy
from src.projection.qb_repair.arms import ARM_BASELINE, run_arm
from src.projection.transitions import SEASON_GAMES

PASS_STATS = ("attempts", "completions", "passing_yards", "passing_tds", "interceptions")
RUSH_STATS = ("carries", "rushing_yards", "rushing_tds")


def _set_stat_rate(board: pd.DataFrame, player_id: str, stat: str, rate: float) -> None:
    mask = (board["player_id"].astype(str) == str(player_id)) & (board["stat"] == stat)
    if not mask.any() or not np.isfinite(rate):
        return
    for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
        if col not in board.columns:
            continue
        cur = pd.to_numeric(board.loc[mask, col], errors="coerce")
        base = pd.to_numeric(board.loc[mask, "pred_pg"], errors="coerce").replace(0, np.nan)
        if col == "pred_pg":
            board.loc[mask, col] = rate
        else:
            # Preserve band width ratio when possible.
            ratio = (cur / base).fillna(1.0)
            board.loc[mask, col] = rate * ratio


def rewrite_qb_rates_from_active_priors(
    board: pd.DataFrame,
    history: pd.DataFrame,
    *,
    target_season: int,
) -> tuple[pd.DataFrame, dict]:
    """Replace QB pred_pg with active-start rates; store expected starts for alloc."""
    out = board.copy()
    audit = {"players": {}, "expected_active_starts": {}}
    qb_ids = out.loc[out["position"].astype(str).eq("QB"), "player_id"].astype(str).unique()
    for pid in qb_ids:
        avail = expected_availability(history, player_id=pid, target_season=target_season)
        rush = hierarchical_rush_priors(history, player_id=pid, target_season=target_season)
        rates = {}
        for stat in PASS_STATS:
            pooled = pooled_active_rate(
                history, player_id=pid, target_season=target_season, rate_col=f"{stat}_per_active"
            )
            if pooled["value"] is not None:
                rates[stat] = pooled["value"]
                _set_stat_rate(out, pid, stat, pooled["value"])
        # Rush from archetype-conditional hierarchical priors when available.
        mapping = {
            "carries": "carries_per_active",
            "rushing_yards": "rushing_yards_per_active",
            "rushing_tds": "rushing_tds_per_active",
        }
        for stat, prior_key in mapping.items():
            val = rush["priors"].get(prior_key)
            if val is None:
                pooled = pooled_active_rate(
                    history, player_id=pid, target_season=target_season, rate_col=f"{stat}_per_active"
                )
                val = pooled["value"]
            if val is not None:
                rates[stat] = val
                _set_stat_rate(out, pid, stat, float(val))
        # Store expected starts on all rows for this player for downstream alloc.
        exp = float(avail["expected_active_starts"])
        audit["expected_active_starts"][pid] = exp
        audit["players"][pid] = {
            "availability": avail,
            "archetype": rush["archetype"],
            "rates": rates,
            "designed_carries_per_active": rush["priors"].get("designed_carries_per_active"),
            "scramble_per_dropback": rush["priors"].get("scramble_per_dropback"),
        }
        if "projected_games_raw" in out.columns:
            mask = out["player_id"].astype(str).eq(pid)
            # Gate-A style raw exposure tracks expected active starts.
            out.loc[mask, "projected_games_raw"] = exp
    return out, audit


def compose_candidate(
    raw: pd.DataFrame,
    history: pd.DataFrame,
    *,
    target_season: int = 2026,
) -> dict:
    """Baseline shipped compose vs candidate active+archetype+joint-v2."""
    raw_clean = raw.copy()
    for col in ("pred_season", "pred_season_low", "pred_season_high", "team_volume_scale", "td_rate_clip_applied"):
        if col in raw_clean.columns:
            raw_clean = raw_clean.drop(columns=[col])

    ctx_base = shipped_context(conn=None, target_season=target_season)
    baseline = run_arm(raw_clean, ctx_base, ARM_BASELINE, target_season=target_season)

    rewritten, audit = rewrite_qb_rates_from_active_priors(
        raw_clean, history, target_season=target_season
    )
    # Compose with shipped hygiene but replace QB reconcile with v2.
    ctx = shipped_context(conn=None, target_season=target_season)
    # Use internal compose then overlay joint v2 on QB cells.
    # Force joint off so shipped reconcile runs for non-QB; then re-do QB.
    ctx.qb_joint_room_allocation = False
    board = compose_board(rewritten.copy(), ctx)
    # Undo QB team-volume scales and apply v2.
    qb = board["position"].astype(str).eq("QB")
    if qb.any() and "team_volume_scale" in board.columns:
        scale = pd.to_numeric(board.loc[qb, "team_volume_scale"], errors="coerce").replace(0, pd.NA).fillna(1.0)
        for col in ("pred_pg", "pred_pg_low", "pred_pg_high"):
            if col in board.columns:
                board.loc[qb, col] = pd.to_numeric(board.loc[qb, col], errors="coerce") / scale
        board.loc[qb, "team_volume_scale"] = 1.0
    board, alloc_report = reconcile_qb_joint_room_v2(
        board,
        target_season=target_season,
        expected_active_starts=audit["expected_active_starts"],
        alpha=1.0,  # experimental path: full active-conditional allocation
    )
    inv = non_qb_invariance_check(baseline_long=baseline.board, candidate_long=board)
    return {
        "baseline_board": baseline.board,
        "candidate_board": board,
        "baseline_fantasy": score_long_to_fantasy(baseline.board),
        "candidate_fantasy": score_long_to_fantasy(board),
        "rewrite_audit": audit,
        "allocation_report": alloc_report,
        "non_qb_invariance": inv,
    }
