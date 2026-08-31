"""Simple DST and kicker draw models."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class TeamContext:
    opponent_implied_points: float = 22.0
    pressure_rate: float = 0.08
    turnover_prior: float = 0.10
    venue_factor: float = 1.0
    weather_factor: float = 1.0


@dataclass
class KickerContext:
    drives_per_game: float = 10.5
    redzone_td_rate: float = 0.55
    fg_accuracy: float = 0.85
    venue_factor: float = 1.0
    weather_factor: float = 1.0


def simulate_dst_draw(ctx: TeamContext, *, seed: int = 0) -> dict[str, float]:
    rng = random.Random(seed)
    sacks = max(0.0, 2.0 + rng.gauss(0.0, 1.0))
    interceptions = 1 if rng.random() < ctx.turnover_prior else 0
    fumble_recoveries = 1 if rng.random() < ctx.turnover_prior * 0.6 else 0
    points_allowed = max(0.0, rng.gauss(ctx.opponent_implied_points, 7.0))
    def_tds = 1 if rng.random() < 0.08 else 0
    return {
        "sacks": float(sacks),
        "interceptions": float(interceptions),
        "fumble_recoveries": float(fumble_recoveries),
        "points_allowed": points_allowed,
        "def_tds": float(def_tds),
    }


def simulate_kicker_draw(ctx: KickerContext, *, seed: int = 0) -> dict[str, float]:
    rng = random.Random(seed)
    scoring_ops = max(0.0, rng.gauss(ctx.drives_per_game * 0.45, 1.5))
    tds = scoring_ops * ctx.redzone_td_rate
    fgs = max(0.0, scoring_ops - tds)
    fg_makes = sum(1 for _ in range(int(round(fgs))) if rng.random() < ctx.fg_accuracy * ctx.weather_factor)
    xp_attempts = int(round(tds))
    xp_makes = sum(1 for _ in range(xp_attempts) if rng.random() < 0.95)
    return {
        "fgm_30_39": float(fg_makes * 0.4),
        "fgm_40_49": float(fg_makes * 0.35),
        "fgm_50p": float(fg_makes * 0.25),
        "xpm": float(xp_makes),
    }
