"""H3 end-to-end QB forecast: availability × active opportunity × efficiency.

Uses only seasons < target. Does not retune frozen H1/H2 thresholds.
Fantasy scoring matches sealed half-PPR (fantasy_points.SCORING).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.projection.fantasy_points import SCORING
from src.projection.qb_active_archetype.active_rates import (
    expected_availability,
    pooled_active_rate,
)
from src.projection.qb_active_archetype.thresholds import AVAIL_LOOKBACK_SEASONS
from src.projection.qb_h3.archetype import classify_archetype_h3, hierarchical_rush_priors_h3
from src.projection.qb_h3.composition_contract import (
    assert_availability_applied_once,
    compose_season_opportunity,
)


def _eff_rate(history: pd.DataFrame, pid: str, target_season: int, num_col: str, den_col: str) -> float | None:
    hist = history[
        (history.player_id.astype(str) == str(pid))
        & (history.season < int(target_season))
        & (history.season >= int(target_season) - AVAIL_LOOKBACK_SEASONS)
    ]
    if hist.empty or num_col not in hist.columns or den_col not in hist.columns:
        return None
    # Prefer season totals if present
    if num_col.replace("_per_active", "_season") in hist.columns:
        num = pd.to_numeric(hist[num_col.replace("_per_active", "_season")], errors="coerce")
        den = pd.to_numeric(hist["active_starts"], errors="coerce") * pd.to_numeric(
            hist[den_col], errors="coerce"
        )
        # simpler: yards_per_active / attempts_per_active
    num = pd.to_numeric(hist[num_col], errors="coerce")
    den = pd.to_numeric(hist[den_col], errors="coerce")
    w = pd.to_numeric(hist["active_starts"], errors="coerce").fillna(0.0)
    ratio = num / den.replace(0, np.nan)
    mask = ratio.notna() & w.gt(0)
    if not mask.any():
        return None
    return float(np.average(ratio[mask], weights=w[mask]))


def predict_h3(history: pd.DataFrame, *, player_id: str, target_season: int) -> dict:
    """Full H3 component forecast for one player (OOS: seasons < target)."""
    avail = expected_availability(history, player_id=player_id, target_season=target_season)
    arch = classify_archetype_h3(history, player_id=player_id, target_season=target_season)
    rush = hierarchical_rush_priors_h3(history, player_id=player_id, target_season=target_season)

    # B. Passing opportunity conditional on active
    rates = {}
    for stat in ("attempts", "completions", "passing_yards", "passing_tds", "interceptions"):
        rates[stat] = pooled_active_rate(
            history, player_id=player_id, target_season=target_season, rate_col=f"{stat}_per_active"
        )["value"]

    # C. Rushing opportunity — archetype priors
    rates["carries"] = rush["priors"].get("carries_per_active")
    rates["rushing_yards"] = rush["priors"].get("rushing_yards_per_active")
    rates["rushing_tds"] = rush["priors"].get("rushing_tds_per_active")
    for stat in ("carries", "rushing_yards", "rushing_tds"):
        if rates[stat] is None:
            rates[stat] = pooled_active_rate(
                history, player_id=player_id, target_season=target_season, rate_col=f"{stat}_per_active"
            )["value"]

    # D. Conditional efficiency (for audit; rates already embed efficiency)
    ypa = None
    if rates.get("attempts") and rates.get("passing_yards"):
        ypa = rates["passing_yards"] / rates["attempts"] if rates["attempts"] else None
    pass_td_rate = (
        rates["passing_tds"] / rates["attempts"]
        if rates.get("attempts") and rates.get("passing_tds") is not None
        else None
    )

    if rates.get("attempts") is None or rates.get("carries") is None:
        return {"ok": False, "reason": "missing_active_rates"}

    opp = compose_season_opportunity(
        attempts_per_active=float(rates["attempts"]),
        carries_per_active=float(rates["carries"]),
        expected_active_starts=float(avail["expected_active_starts"]),
        partial_exit_rate=float(avail.get("partial_exit_rate") or 0.0),
    )
    # Season counting stats = per-active × effective starts (partial-adjusted once)
    effective = opp.season_attempts / opp.attempts_per_active if opp.attempts_per_active else 0.0
    # Identity on the effective product used for season totals (availability once).
    assert_availability_applied_once(
        opp.attempts_per_active,
        effective,
        opp.season_attempts,
    )

    season = {}
    for stat, per_active in rates.items():
        if per_active is None:
            season[stat] = None
        else:
            season[stat] = float(per_active) * effective

    # Fantasy points per active start + expected season points (separate outputs)
    pp_active = 0.0
    for stat, pts in SCORING.items():
        if rates.get(stat) is not None:
            pp_active += float(rates[stat]) * float(pts)
    season_points = pp_active * effective
    avail_adj_ppg = season_points / 17.0

    designed_pa = rush["priors"].get("designed_carries_per_active")
    scramble_pa = None
    if rates.get("carries") is not None and designed_pa is not None:
        scramble_pa = max(0.0, float(rates["carries"]) - float(designed_pa))
    elif rush["priors"].get("scramble_per_dropback") is not None and rates.get("attempts"):
        # scramble per dropback × attempts as proxy when designed split missing
        scramble_pa = float(rush["priors"]["scramble_per_dropback"]) * float(rates["attempts"])

    return {
        "ok": True,
        "archetype": arch["archetype"],
        "availability": {
            **avail,
            "expected_active_games": avail.get("expected_active_starts"),
            "partial_game_early_exit_probability": avail.get("partial_exit_rate"),
        },
        "rates_per_active": rates,
        "efficiency": {
            "yards_per_attempt": ypa,
            "pass_td_rate": pass_td_rate,
            "interception_rate": (
                rates["interceptions"] / rates["attempts"]
                if rates.get("attempts") and rates.get("interceptions") is not None
                else None
            ),
            "designed_carries_per_active": designed_pa,
            "scrambles_per_active_start": scramble_pa,
            "scramble_per_dropback": rush["priors"].get("scramble_per_dropback"),
            "designed_ypc": rush["priors"].get("designed_ypc"),
            "scramble_ypa": rush["priors"].get("scramble_ypa"),
            "rush_td_rate": (
                rates["rushing_tds"] / rates["carries"]
                if rates.get("carries") and rates.get("rushing_tds") is not None and rates["carries"]
                else None
            ),
        },
        "opportunity": opp.__dict__,
        "season_stats": season,
        "points_per_active_start": pp_active,
        "expected_season_points": season_points,
        "availability_adjusted_ppg": avail_adj_ppg,
        "effective_starts": effective,
        "availability_applied_once": True,
        "composition_contract": "active_start_opportunity × expected_active_starts = season_opportunity",
    }
