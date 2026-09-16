"""Leakage-safe rolling-origin harness for Milestone 2 (synthetic).

This is the M2 backtest, not a renamed M3. It never trains a hierarchical
model. Week t features may use opponent EPA from weeks < t (or prior-season)
only. Injecting same-week ``team_attempts`` (or siblings) must fail closed.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.projection.weekly_latent.allocate import allocate_team_weeks, allocate_team_weeks_m2
from src.projection.weekly_latent.constants import (
    FORBIDDEN_SAME_WEEK_TRAINING_FEATURES,
    M1_AVAILABLE_AT,
    PRIOR_SEASON_EPA_AVAILABLE_AT,
)
from src.projection.weekly_latent.environment import refuse_forbidden_m2_columns
from src.projection.weekly_latent.priors import assert_priors_are_as_of, build_as_of_priors


def synthetic_rolling_history() -> dict[str, pd.DataFrame]:
    """Two teams, four weeks. Realized volume is an *outcome*, not a feature.

    BBB is a tough pass defense (low EPA allowed). AAA's realized pass
    attempts fall in weeks they face BBB. Prior-season EPA is the week-1
    fallback; weeks 2–4 may use lagged realized defense EPA.
    """
    schedule = pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "week": [1, 2, 3, 4],
            "team": ["AAA", "AAA", "AAA", "AAA"],
            "opponent": ["BBB", "CCC", "BBB", "CCC"],
            "is_home": [1, 0, 1, 0],
            "is_bye": [0, 0, 0, 0],
            "is_neutral": [0, 0, 0, 0],
            "is_international": [0, 0, 0, 0],
            "A_team_w": [1.0, 1.0, 1.0, 1.0],
            "rest_days": [7, 7, 4, 10],
            "shrinkage_lambda": [1.0, 1.0, 1.0, 1.0],
            "gameday": [
                "2026-09-10",
                "2026-09-17",
                "2026-09-24",
                "2026-10-01",
            ],
        }
    )
    # Realized same-week volume: outcome table, never joined onto features.
    outcomes = pd.DataFrame(
        {
            "team": ["AAA"] * 4,
            "week": [1, 2, 3, 4],
            "team_pass_attempts": [31.0, 36.0, 30.0, 37.0],
        }
    )
    # Defense EPA allowed by opponent, observed *after* that week.
    weekly_epa = pd.DataFrame(
        {
            "team": ["BBB", "CCC", "BBB", "CCC", "BBB", "CCC", "BBB", "CCC"],
            "week": [1, 1, 2, 2, 3, 3, 4, 4],
            "pass_epa": [-4.0, 5.0, -3.8, 4.8, -3.5, 4.5, -3.6, 4.6],
            "rush_epa": [-1.0, 1.0, -0.9, 0.95, -0.8, 0.9, -0.85, 0.92],
            "gameday": [
                "2026-09-10",
                "2026-09-10",
                "2026-09-17",
                "2026-09-17",
                "2026-09-24",
                "2026-09-24",
                "2026-10-01",
                "2026-10-01",
            ],
        }
    )
    prior_season = pd.DataFrame(
        {
            "team": ["BBB", "CCC"],
            "def_pass_epa_allowed": [-4.0, 5.0],
            "def_rush_epa_allowed": [-1.0, 1.0],
            "available_at": [PRIOR_SEASON_EPA_AVAILABLE_AT, PRIOR_SEASON_EPA_AVAILABLE_AT],
        }
    )
    volume = pd.DataFrame(
        {
            "team": ["AAA"],
            "team_pass_attempts": [136.0],
            "team_rush_attempts": [100.0],
            "team_passing_yards": [1360.0],
            "team_rushing_yards": [400.0],
        }
    )
    return {
        "schedule": schedule,
        "outcomes": outcomes,
        "weekly_epa": weekly_epa,
        "prior_season": prior_season,
        "volume": volume,
    }


def features_for_week(
    history: dict[str, pd.DataFrame],
    *,
    as_of_week: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prediction frame for week t: schedule row + lagged opponent priors only."""
    schedule = history["schedule"]
    row = schedule[schedule["week"].eq(as_of_week)].copy()
    refuse_forbidden_m2_columns(row, where=f"as-of week {as_of_week} schedule")
    priors = build_as_of_priors(
        history["weekly_epa"],
        as_of_week=as_of_week,
        prior_season=history["prior_season"],
    )
    assert_priors_are_as_of(priors, as_of_week=as_of_week)
    refuse_forbidden_m2_columns(priors, where=f"as-of week {as_of_week} priors")
    return row, priors


def poison_same_week_volume(features: pd.DataFrame) -> pd.DataFrame:
    poisoned = features.copy()
    poisoned["team_attempts"] = 999.0
    return poisoned


def run_synthetic_rolling_origin() -> dict[str, Any]:
    """Walk weeks 1–4; predict from lagged features; score vs outcomes."""
    history = synthetic_rolling_history()
    rows = []
    poison_raised = False
    for week in (1, 2, 3, 4):
        team_week, priors = features_for_week(history, as_of_week=week)
        try:
            refuse_forbidden_m2_columns(
                poison_same_week_volume(team_week),
                where=f"poison week {week}",
            )
        except ValueError:
            poison_raised = True
        else:
            raise AssertionError("same-week team_attempts must fail closed")
        m2 = allocate_team_weeks_m2(
            team_week,
            history["volume"],
            opponent_priors=priors,
            board_available_at=M1_AVAILABLE_AT,
            n_active_override=4.0,
        )
        m1 = allocate_team_weeks(
            team_week,
            history["volume"],
            opponent_priors=priors,
        )
        realized = float(
            history["outcomes"].loc[history["outcomes"]["week"].eq(week), "team_pass_attempts"].iloc[0]
        )
        pred_m2 = float(m2["team_pass_attempts"].iloc[0])
        pred_m1 = float(m1["team_pass_attempts"].iloc[0])
        naive = float(history["volume"]["team_pass_attempts"].iloc[0]) / 4.0
        # Single-week M1 renormalize puts ALL season mass in this one row.
        # Compare M2 against the even-split naive, which is the fair baseline
        # on a 1-row as-of frame. M1 comparison is reported but not the score.
        rows.append(
            {
                "week": week,
                "pred_m2": pred_m2,
                "pred_m1_one_row": pred_m1,
                "naive": naive,
                "realized": realized,
                "abs_err_m2": abs(pred_m2 - realized),
                "abs_err_naive": abs(naive - realized),
                "available_at": str(m2["available_at"].iloc[0]),
                "prior_source": (
                    str(priors["prior_source"].iloc[0]) if "prior_source" in priors.columns else None
                ),
            }
        )
    frame = pd.DataFrame(rows)
    mae_m2 = float(frame["abs_err_m2"].mean())
    mae_naive = float(frame["abs_err_naive"].mean())
    leaked = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(history["schedule"].columns)
    later_cutoffs = [
        parse_ok(a) > parse_ok(PRIOR_SEASON_EPA_AVAILABLE_AT)
        for a in frame.loc[frame["week"] > 1, "available_at"]
    ]
    return {
        "schema_version": "weekly_latent_m2_backtest_synthetic_v1",
        "passes": poison_raised and leaked == set() and mae_m2 < mae_naive,
        "poison_same_week_raised": poison_raised,
        "forbidden_columns_on_schedule": sorted(leaked),
        "mae_m2": mae_m2,
        "mae_naive_even_split": mae_naive,
        "beats_naive": mae_m2 < mae_naive,
        "later_weeks_overwrite_available_at": all(later_cutoffs) if later_cutoffs else False,
        "weeks": rows,
        "holdout": "synthetic rolling-origin; no random split; no 2026 live outcomes",
        "not_a_promotion": True,
    }


def parse_ok(stamp: str):
    from src.projection.weekly_latent.constants import parse_available_at

    return parse_available_at(stamp)
