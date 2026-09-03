"""H3 composition contract: availability applied exactly once.

active_start_opportunity × expected_active_starts = expected_season_opportunity
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.projection.transitions import SEASON_GAMES


@dataclass(frozen=True)
class SeasonOpportunity:
    attempts_per_active: float
    carries_per_active: float
    expected_active_starts: float
    partial_exit_rate: float
    season_attempts: float
    season_carries: float
    avail_adj_attempts_per_sched_game: float
    avail_adj_carries_per_sched_game: float


def compose_season_opportunity(
    *,
    attempts_per_active: float,
    carries_per_active: float,
    expected_active_starts: float,
    partial_exit_rate: float = 0.0,
) -> SeasonOpportunity:
    """Single application of availability to active-start rates."""
    exp = float(np.clip(expected_active_starts, 0.0, SEASON_GAMES))
    # Partial exits slightly reduce effective full-start equivalents.
    effective = exp * (1.0 - 0.5 * float(np.clip(partial_exit_rate, 0.0, 0.5)))
    season_att = float(attempts_per_active) * effective
    season_car = float(carries_per_active) * effective
    return SeasonOpportunity(
        attempts_per_active=float(attempts_per_active),
        carries_per_active=float(carries_per_active),
        expected_active_starts=exp,
        partial_exit_rate=float(partial_exit_rate),
        season_attempts=season_att,
        season_carries=season_car,
        avail_adj_attempts_per_sched_game=season_att / SEASON_GAMES,
        avail_adj_carries_per_sched_game=season_car / SEASON_GAMES,
    )


def assert_availability_applied_once(
    attempts_per_active: float,
    expected_active_starts: float,
    season_attempts: float,
    *,
    tol: float = 1e-6,
) -> None:
    """Identity: season_attempts == attempts_per_active × expected_active_starts.

    (Using effective starts without partial for the strict identity test.)
    """
    expected = float(attempts_per_active) * float(expected_active_starts)
    if abs(float(season_attempts) - expected) > tol:
        raise AssertionError(
            f"availability applied incorrectly: {season_attempts} != "
            f"{attempts_per_active} × {expected_active_starts} (= {expected})"
        )


def detect_double_availability(
    attempts_per_active: float,
    expected_active_starts: float,
    reported_season_attempts: float,
    *,
    tol: float = 1e-4,
) -> dict:
    """Flag if season total looks like rate×starts×starts/17 (double avail)."""
    once = attempts_per_active * expected_active_starts
    twice_via_sched = attempts_per_active * expected_active_starts * (expected_active_starts / SEASON_GAMES)
    twice_via_17 = (attempts_per_active * expected_active_starts / SEASON_GAMES) * expected_active_starts
    return {
        "matches_once": abs(reported_season_attempts - once) <= tol * max(1.0, abs(once)),
        "matches_double_sched": abs(reported_season_attempts - twice_via_sched)
        <= 0.05 * max(1.0, abs(twice_via_sched)),
        "matches_double_17": abs(reported_season_attempts - twice_via_17)
        <= 0.05 * max(1.0, abs(twice_via_17)),
        "once": once,
        "reported": reported_season_attempts,
    }


def allocate_starter_backup_season(
    *,
    team_season_attempts: float,
    starter_attempts_per_active: float,
    starter_expected_starts: float,
    backup_attempts_per_active: float | None = None,
) -> dict:
    """Starter conditional volume first; backups fill residual missed starts.

    Backups cannot reduce starter's conditional active-start rate.
    """
    starter_season = starter_attempts_per_active * starter_expected_starts
    starter_season = min(starter_season, team_season_attempts)
    residual = max(0.0, team_season_attempts - starter_season)
    missed = max(0.0, SEASON_GAMES - starter_expected_starts)
    if backup_attempts_per_active and missed > 0:
        backup_natural = backup_attempts_per_active * missed
        backup_season = min(residual, backup_natural)
    else:
        backup_season = residual
    # Conserve exactly onto team claim
    total = starter_season + backup_season
    if total > 0 and abs(total - team_season_attempts) > 1e-9:
        # Only shrink backup
        if total > team_season_attempts:
            backup_season = max(0.0, team_season_attempts - starter_season)
        else:
            backup_season = team_season_attempts - starter_season
    return {
        "starter_season_attempts": starter_season,
        "backup_season_attempts": backup_season,
        "starter_attempts_per_active_preserved": starter_attempts_per_active,
        "conserved_total": starter_season + backup_season,
        "team_claim": team_season_attempts,
    }
