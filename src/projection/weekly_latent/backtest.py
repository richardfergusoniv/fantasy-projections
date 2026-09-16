"""Leakage-safe rolling-origin harnesses for Milestone 2.

Two evaluations of the object M2 actually ships (deterministic team-week
latent that may move season mass):

1. Synthetic 2-team walk (weeks 1–4) — harness demo.
2. Historical 2025-shaped schedule (5 teams, byes, rest, home/away) —
   prior-only opponent features, outcomes held out.

Neither trains a hierarchical model. Week t features may use opponent EPA
from weeks < t (or prior-season) only. Injecting same-week
``team_attempts`` / ``team_carries`` / ``team_targets`` / ``team_air_yards``
must fail closed. Not 6–8 live 2026 weeks. Not a promotion.
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

HISTORICAL_BOARD_AVAILABLE_AT = "2025-08-28T00:00:00+00:00"
HISTORICAL_PRIOR_SEASON_AVAILABLE_AT = "2025-01-06T00:00:00+00:00"
HISTORICAL_N_ACTIVE = 4.0
HISTORICAL_WEEKS = (1, 2, 3, 4, 5)
HISTORICAL_TEAMS = ("KC", "BUF", "SF", "PHI", "BAL")


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


def poison_same_week_volume(
    features: pd.DataFrame, column: str = "team_attempts"
) -> pd.DataFrame:
    if column not in FORBIDDEN_SAME_WEEK_TRAINING_FEATURES:
        raise ValueError(f"{column} is not a forbidden same-week feature")
    poisoned = features.copy()
    poisoned[column] = 999.0
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
        "later_weeks_advance_available_at": all(later_cutoffs) if later_cutoffs else False,
        "weeks": rows,
        "holdout": "synthetic rolling-origin; no random split; no 2026 live outcomes",
        "not_a_promotion": True,
    }


def historical_schedule_history() -> dict[str, Any]:
    """2025-shaped multi-team slate. Outcomes never live on the feature frame.

    Five teams, five weeks, one bye each. Season volume is a *prior-season*
    analogue (not the sum of 2025 realized weeks — that would leak future
    weeks into week 1). Opponent EPA for week t uses weeks < t only.
    """
    games = (
        # week, gameday, home, away, home_rest, away_rest
        (1, "2025-09-07", "KC", "BUF", 7, 7),
        (1, "2025-09-07", "SF", "PHI", 7, 7),
        (2, "2025-09-14", "BUF", "SF", 7, 7),
        (2, "2025-09-14", "PHI", "BAL", 7, 7),
        (3, "2025-09-21", "BAL", "KC", 7, 7),
        (3, "2025-09-21", "PHI", "BUF", 7, 4),
        (4, "2025-09-28", "PHI", "KC", 7, 7),
        (4, "2025-09-28", "SF", "BAL", 10, 7),
        (5, "2025-10-05", "BAL", "BUF", 7, 7),
        (5, "2025-10-05", "SF", "KC", 7, 7),
    )
    byes = {1: "BAL", 2: "KC", 3: "SF", 4: "BUF", 5: "PHI"}
    gameday_by_week = {row[0]: row[1] for row in games}
    rows: list[dict[str, Any]] = []
    for week, gameday, home, away, home_rest, away_rest in games:
        rows.append(
            _historical_team_week(
                team=home,
                opponent=away,
                week=week,
                gameday=gameday,
                is_home=1,
                rest_days=home_rest,
                is_bye=0,
            )
        )
        rows.append(
            _historical_team_week(
                team=away,
                opponent=home,
                week=week,
                gameday=gameday,
                is_home=0,
                rest_days=away_rest,
                is_bye=0,
            )
        )
    for week, team in byes.items():
        rows.append(
            _historical_team_week(
                team=team,
                opponent=None,
                week=week,
                gameday=gameday_by_week[week],
                is_home=0,
                rest_days=7,
                is_bye=1,
            )
        )
    schedule = pd.DataFrame(rows).sort_values(["week", "team"]).reset_index(drop=True)
    refuse_forbidden_m2_columns(schedule, where="historical schedule")

    def_pass = {"KC": 0.2, "BUF": -4.0, "SF": 5.0, "PHI": 4.5, "BAL": -3.5}
    def_rush = {"KC": 0.0, "BUF": -1.0, "SF": 1.0, "PHI": 0.9, "BAL": -0.8}
    epa_rows = []
    for week, gameday, home, away, _hr, _ar in games:
        for team in (home, away):
            epa_rows.append(
                {
                    "team": team,
                    "week": week,
                    "pass_epa": def_pass[team] + 0.05 * (week - 1),
                    "rush_epa": def_rush[team] + 0.02 * (week - 1),
                    "gameday": gameday,
                }
            )
    weekly_epa = pd.DataFrame(epa_rows)
    refuse_forbidden_m2_columns(weekly_epa, where="historical weekly EPA")

    prior_season = pd.DataFrame(
        {
            "team": list(HISTORICAL_TEAMS),
            "def_pass_epa_allowed": [def_pass[t] for t in HISTORICAL_TEAMS],
            "def_rush_epa_allowed": [def_rush[t] for t in HISTORICAL_TEAMS],
            "available_at": [HISTORICAL_PRIOR_SEASON_AVAILABLE_AT] * len(HISTORICAL_TEAMS),
        }
    )
    volume = pd.DataFrame(
        {
            "team": list(HISTORICAL_TEAMS),
            "team_pass_attempts": [140.0] * len(HISTORICAL_TEAMS),
            "team_rush_attempts": [100.0] * len(HISTORICAL_TEAMS),
            "team_passing_yards": [1400.0] * len(HISTORICAL_TEAMS),
            "team_rushing_yards": [400.0] * len(HISTORICAL_TEAMS),
        }
    )
    opp_pass_adj = {"BUF": -5.0, "BAL": -4.0, "KC": 0.0, "SF": 5.0, "PHI": 4.0}
    out_rows = []
    for rec in rows:
        if rec["is_bye"]:
            realized_pass = 0.0
            realized_rush = 0.0
        else:
            home_adj = 1.0 if rec["is_home"] else -1.0
            realized_pass = 35.0 + opp_pass_adj[rec["opponent"]] + home_adj
            realized_rush = 25.0 - 0.5 * opp_pass_adj[rec["opponent"]]
        out_rows.append(
            {
                "team": rec["team"],
                "week": rec["week"],
                "realized_pass_attempts": realized_pass,
                "realized_rush_attempts": realized_rush,
            }
        )
    outcomes = pd.DataFrame(out_rows)
    refuse_forbidden_m2_columns(outcomes, where="historical outcomes")
    return {
        "schedule": schedule,
        "outcomes": outcomes,
        "weekly_epa": weekly_epa,
        "prior_season": prior_season,
        "volume": volume,
        "n_active": HISTORICAL_N_ACTIVE,
        "board_available_at": HISTORICAL_BOARD_AVAILABLE_AT,
        "schedule_env_available_at": HISTORICAL_BOARD_AVAILABLE_AT,
    }


def _historical_team_week(
    *,
    team: str,
    opponent: str | None,
    week: int,
    gameday: str,
    is_home: int,
    rest_days: int,
    is_bye: int,
) -> dict[str, Any]:
    return {
        "season": 2025,
        "week": week,
        "team": team,
        "opponent": opponent,
        "is_home": is_home,
        "is_bye": is_bye,
        "is_neutral": 0,
        "is_international": 0,
        "A_team_w": 0.0 if is_bye else 1.0,
        "rest_days": rest_days,
        "shrinkage_lambda": 1.0,
        "gameday": gameday,
        "is_home_for_mult": int(is_home == 1 and is_bye == 0),
    }


def historical_features_for_week(
    history: dict[str, pd.DataFrame],
    *,
    as_of_week: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prediction frame for week t on the historical slate: lagged priors only."""
    return features_for_week(history, as_of_week=as_of_week)


def run_historical_schedule_backtest() -> dict[str, Any]:
    """Walk a 2025-shaped schedule; score M2 volume vs held-out outcomes."""
    history = historical_schedule_history()
    n_active = float(history["n_active"])
    board_at = str(history["board_available_at"])
    env_at = str(history["schedule_env_available_at"])
    volume = history["volume"]
    outcomes = history["outcomes"]
    rows: list[dict[str, Any]] = []
    allocated: list[pd.DataFrame] = []
    poisoned_columns: list[str] = []
    for week in HISTORICAL_WEEKS:
        team_week, priors = historical_features_for_week(history, as_of_week=week)
        leaked = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(team_week.columns)
        if leaked:
            raise AssertionError(
                f"historical week {week} feature frame has forbidden columns: {sorted(leaked)}"
            )
        for col in sorted(FORBIDDEN_SAME_WEEK_TRAINING_FEATURES):
            try:
                refuse_forbidden_m2_columns(
                    poison_same_week_volume(team_week, column=col),
                    where=f"historical poison {col} week {week}",
                )
            except ValueError:
                if col not in poisoned_columns:
                    poisoned_columns.append(col)
            else:
                raise AssertionError(f"same-week {col} must fail closed")
        m2 = allocate_team_weeks_m2(
            team_week,
            volume,
            opponent_priors=priors,
            board_available_at=board_at,
            n_active_override=n_active,
            schedule_env_available_at=env_at,
        )
        allocated.append(m2)
        scored = m2.merge(outcomes, on=["team", "week"], how="left")
        refuse_forbidden_m2_columns(m2, where=f"historical M2 allocation week {week}")
        for _, rec in scored.iterrows():
            pred = float(rec["team_pass_attempts"])
            realized = float(rec["realized_pass_attempts"])
            naive = 0.0 if int(rec["is_bye"]) == 1 else float(
                volume.loc[volume["team"].eq(rec["team"]), "team_pass_attempts"].iloc[0]
            ) / n_active
            rows.append(
                {
                    "week": int(rec["week"]),
                    "team": rec["team"],
                    "is_bye": int(rec["is_bye"]),
                    "pred_m2": pred,
                    "naive": naive,
                    "realized": realized,
                    "abs_err_m2": abs(pred - realized),
                    "abs_err_naive": abs(naive - realized),
                    "available_at": str(rec["available_at"]),
                    "prior_source": (
                        str(priors["prior_source"].iloc[0])
                        if "prior_source" in priors.columns
                        else None
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    all_m2 = pd.concat(allocated, ignore_index=True)
    m2_season = all_m2.groupby("team")["team_pass_attempts"].sum()
    sealed = volume.set_index("team")["team_pass_attempts"]
    season_mass_moved = bool((m2_season - sealed).abs().max() > 1e-6)
    mae_m2 = float(frame["abs_err_m2"].mean())
    mae_naive = float(frame["abs_err_naive"].mean())
    week1 = frame[frame["week"].eq(1) & frame["is_bye"].eq(0)]
    later_active = frame[frame["week"].gt(1) & frame["is_bye"].eq(0)]
    later_cutoffs = [
        parse_ok(a) > parse_ok(board_at) for a in later_active["available_at"]
    ]
    week1_at = str(week1["available_at"].iloc[0]) if len(week1) else None
    week1_src = str(week1["prior_source"].iloc[0]) if len(week1) else None
    later_src = (
        str(later_active["prior_source"].iloc[0]) if len(later_active) else None
    )
    leaked_features = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(
        history["schedule"].columns
    )
    poison_ok = set(poisoned_columns) == set(FORBIDDEN_SAME_WEEK_TRAINING_FEATURES)
    later_ok = all(later_cutoffs) if later_cutoffs else False
    passes = (
        poison_ok
        and leaked_features == set()
        and later_ok
        and week1_at == HISTORICAL_BOARD_AVAILABLE_AT
        and week1_src == "prior_season_fallback"
        and later_src == "lagged_weeks_before_as_of"
        and season_mass_moved
    )
    return {
        "schema_version": "weekly_latent_m2_backtest_historical_v1",
        "object_evaluated": "allocate_team_weeks_m2",
        "passes": passes,
        "poison_same_week_raised": poison_ok,
        "poisoned_columns": sorted(poisoned_columns),
        "forbidden_columns_on_features": sorted(leaked_features),
        "mae_m2": mae_m2,
        "mae_naive_even_split": mae_naive,
        "beats_naive": mae_m2 < mae_naive,
        "later_weeks_advance_available_at": later_ok,
        "season_mass_moved_vs_sealed": season_mass_moved,
        "week1_available_at": week1_at,
        "week1_prior_source": week1_src,
        "later_prior_source": later_src,
        "n_team_weeks_scored": int(len(frame)),
        "weeks": rows,
        "holdout": (
            "historical 2025-shaped schedule rolling-origin; "
            "prior-only opponent features; not 6-8 live 2026 weeks; "
            "not a promotion"
        ),
        "not_a_promotion": True,
    }


def run_m2_backtests() -> dict[str, Any]:
    """Synthetic harness plus historical schedule; both required to pass."""
    synthetic = run_synthetic_rolling_origin()
    historical = run_historical_schedule_backtest()
    return {
        "schema_version": "weekly_latent_m2_backtest_v2",
        "passes": bool(synthetic.get("passes")) and bool(historical.get("passes")),
        "synthetic": synthetic,
        "historical": historical,
        "not_a_promotion": True,
        "holdout": (
            "synthetic + historical-schedule rolling-origin; "
            "not 6-8 live 2026 weeks; not a promotion"
        ),
    }


def parse_ok(stamp: str):
    from src.projection.weekly_latent.constants import parse_available_at

    return parse_available_at(stamp)
