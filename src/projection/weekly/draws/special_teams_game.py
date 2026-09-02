"""Game-linked kicker and DST draws (shared game environment)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class GameOffenseState:
    """Shared offense outcomes from the joint game draw."""

    team_touchdowns: float
    scoring_drives: float
    points_scored: float
    pass_attempts: float
    rush_attempts: float
    turnovers: float
    home: bool = True
    weather_factor: float = 1.0
    venue_factor: float = 1.0


@dataclass(frozen=True)
class DstGameContext:
    opponent_points_scored: float
    opponent_yards: float
    pressure_rate: float = 0.08
    turnover_prior: float = 0.10
    pace: float = 1.0
    home: bool = True
    weather_factor: float = 1.0


def simulate_kicker_from_game(
    rng: np.random.Generator,
    offense: GameOffenseState,
    *,
    fg_accuracy: float = 0.85,
    redzone_td_rate: float = 0.55,
) -> dict[str, float]:
    """Kicker attempts linked to team TDs and scoring-drive settlement."""
    scoring_ops = max(0.0, float(offense.scoring_drives))
    # Field-goal opportunities ≈ scoring drives not settled as TDs.
    implied_fg_ops = max(0.0, scoring_ops * (1.0 - redzone_td_rate))
    # Also allow residual FG ops from points/TD mismatch.
    td = max(0, int(round(offense.team_touchdowns)))
    fg_attempts = int(rng.poisson(max(0.1, implied_fg_ops * offense.weather_factor)))
    makes = 0
    buckets = {"fgm_0_39": 0, "fgm_40_49": 0, "fgm_50p": 0}
    for _ in range(fg_attempts):
        if rng.random() < fg_accuracy * offense.weather_factor * offense.venue_factor:
            makes += 1
            u = rng.random()
            if u < 0.45:
                buckets["fgm_0_39"] += 1
            elif u < 0.80:
                buckets["fgm_40_49"] += 1
            else:
                buckets["fgm_50p"] += 1
    xp_attempts = td
    xp_makes = sum(1 for _ in range(xp_attempts) if rng.random() < 0.94)
    return {
        "fgm_0_39": float(buckets["fgm_0_39"]),
        "fgm_30_39": float(buckets["fgm_0_39"]),  # alias for older contracts
        "fgm_40_49": float(buckets["fgm_40_49"]),
        "fgm_50p": float(buckets["fgm_50p"]),
        "xpm": float(xp_makes),
        "fg_attempts": float(fg_attempts),
        "xp_attempts": float(xp_attempts),
        "_linked_team_tds": float(td),
    }


def simulate_dst_from_game(
    rng: np.random.Generator,
    ctx: DstGameContext,
) -> dict[str, float]:
    """DST components coherent with opponent offense points/yards."""
    plays = max(40.0, (ctx.pace * 65.0))
    sacks = float(rng.poisson(max(0.1, ctx.pressure_rate * plays * 0.35)))
    interceptions = float(rng.binomial(max(1, int(round(plays * 0.35))), min(0.08, ctx.turnover_prior)))
    fumble_recoveries = float(
        rng.binomial(max(1, int(round(plays * 0.1))), min(0.08, ctx.turnover_prior * 0.6))
    )
    points_allowed = max(0.0, float(ctx.opponent_points_scored))
    yards_allowed = max(0.0, float(ctx.opponent_yards))
    def_tds = 1.0 if rng.random() < 0.06 else 0.0
    safeties = 1.0 if rng.random() < 0.02 else 0.0
    blocked_kicks = 1.0 if rng.random() < 0.015 else 0.0
    return {
        "sacks": sacks,
        "interceptions": interceptions,
        "fumble_recoveries": fumble_recoveries,
        "points_allowed": points_allowed,
        "yards_allowed": yards_allowed,
        "def_tds": def_tds,
        "safeties": safeties,
        "blk_kick": blocked_kicks,
        "_linked_opponent_points": points_allowed,
    }


def assert_k_dst_game_coherence(
    kicker: Mapping[str, float],
    dst: Mapping[str, float],
    offense: GameOffenseState,
    opponent_offense: GameOffenseState,
) -> list[str]:
    """Return incoherence reasons (empty if coherent)."""
    bad: list[str] = []
    if abs(_f(kicker, "_linked_team_tds") - round(offense.team_touchdowns)) > 1e-6:
        bad.append("kicker_td_link")
    if abs(_f(dst, "_linked_opponent_points") - opponent_offense.points_scored) > 1e-6:
        bad.append("dst_points_link")
    if _f(kicker, "xp_attempts") > offense.team_touchdowns + 1e-6:
        bad.append("xp_gt_tds")
    return bad


def _f(m: Mapping[str, Any], key: str) -> float:
    try:
        return float(m.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
