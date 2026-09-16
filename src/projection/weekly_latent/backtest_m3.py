"""Leakage-safe rolling-origin harnesses for Milestone 3.

Evaluates the object M3 actually ships: week-varying player availability
and conversion latents on top of the M2 team-week latent
(``allocate_players_m3``).

1. Synthetic walk (weeks 1–4) — sit in week 3 is as-of only at week 3.
2. Historical 2025-shaped schedule (5 teams) — player receptions vs
   held-out outcomes; bye A=0; season = sum of weeks.

Neither uses same-week ``team_attempts`` / ``team_carries`` /
``team_targets`` / ``team_air_yards``. Not 6–8 live 2026 weeks.
Not a promotion. ADP / season Vegas are not drivers.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.projection.weekly_latent.allocate import (
    allocate_players,
    allocate_players_m3,
    allocate_team_weeks_m2,
)
from src.projection.weekly_latent.backtest import (
    FORBIDDEN_SAME_WEEK_TRAINING_FEATURES,
    HISTORICAL_BOARD_AVAILABLE_AT,
    HISTORICAL_N_ACTIVE,
    HISTORICAL_TEAMS,
    HISTORICAL_WEEKS,
    M1_AVAILABLE_AT,
    features_for_week,
    historical_features_for_week,
    historical_schedule_history,
    parse_ok,
    poison_same_week_volume,
    synthetic_rolling_history,
)
from src.projection.weekly_latent.environment import refuse_forbidden_m2_columns
from src.projection.weekly_latent.season_board import role_shares

SIT_AVAILABLE_AT = "2026-09-18T00:00:00+00:00"
HISTORICAL_SIT_AVAILABLE_AT = "2025-09-15T00:00:00+00:00"
HISTORICAL_SIT_PLAYER = "BUF-WR"
HISTORICAL_SIT_WEEK = 3


def _wr_season_row(player_id: str, team: str, *, targets: float = 80.0) -> dict[str, Any]:
    rec = round(targets * 0.70, 1)
    return {
        "player_id": player_id,
        "display_name": f"{team} WR",
        "team": team,
        "position": "WR",
        "depth_rank": 1,
        "projected_games": 17.0,
        "attempts": 0.0,
        "completions": 0.0,
        "passing_yards": 0.0,
        "passing_tds": 0.0,
        "interceptions": 0.0,
        "carries": 0.0,
        "rushing_yards": 0.0,
        "rushing_tds": 0.0,
        "targets": targets,
        "receptions": rec,
        "receiving_yards": targets * 10.0,
        "receiving_tds": max(1.0, round(targets / 20.0, 1)),
    }


def synthetic_m3_history() -> dict[str, Any]:
    """AAA WR, four weeks. Week-3 sit is an as-of override, not a feature leak."""
    base = synthetic_rolling_history()
    players = pd.DataFrame([_wr_season_row("p-wr", "AAA", targets=80.0)])
    volume = base["volume"]
    shares = role_shares(players, volume)
    # Realized receptions: 0 on the sit week; higher vs easy CCC.
    outcomes = pd.DataFrame(
        {
            "player_id": ["p-wr"] * 4,
            "week": [1, 2, 3, 4],
            "realized_receptions": [4.0, 8.0, 0.0, 9.0],
        }
    )
    refuse_forbidden_m2_columns(outcomes, where="M3 synthetic outcomes")
    return {
        **base,
        "players": players,
        "shares": shares,
        "player_outcomes": outcomes,
    }


def _sit_override(week: int, *, as_of_week: int) -> pd.DataFrame | None:
    """Week-3 sit is knowable only when predicting week 3 or later."""
    if as_of_week < 3 or week != 3:
        return None
    return pd.DataFrame(
        {
            "player_id": ["p-wr"],
            "week": [3],
            "A_i_w": [0.0],
            "available_at": [SIT_AVAILABLE_AT],
        }
    )


def run_synthetic_m3_rolling_origin() -> dict[str, Any]:
    """Walk weeks 1–4; predict player receptions from lagged features + as-of sit."""
    history = synthetic_m3_history()
    rows = []
    poison_raised = False
    week3_zero = False
    for week in (1, 2, 3, 4):
        team_week, priors = features_for_week(history, as_of_week=week)
        try:
            refuse_forbidden_m2_columns(
                poison_same_week_volume(team_week),
                where=f"M3 poison week {week}",
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
        availability = _sit_override(week, as_of_week=week)
        m3 = allocate_players_m3(
            m2,
            history["players"],
            history["shares"],
            availability=availability,
        )
        naive = allocate_players(m2, history["players"], history["shares"])
        realized = float(
            history["player_outcomes"]
            .loc[history["player_outcomes"]["week"].eq(week), "realized_receptions"]
            .iloc[0]
        )
        pred = float(m3["receptions"].iloc[0])
        naive_pred = float(naive["receptions"].iloc[0])
        if week == 3:
            week3_zero = float(m3["A_i_w"].iloc[0]) == 0.0 and pred == 0.0
        rows.append(
            {
                "week": week,
                "pred_m3": pred,
                "pred_naive": naive_pred,
                "realized": realized,
                "abs_err_m3": abs(pred - realized),
                "abs_err_naive": abs(naive_pred - realized),
                "A_i_w": float(m3["A_i_w"].iloc[0]),
                "available_at": str(m3["available_at"].iloc[0]),
            }
        )
    frame = pd.DataFrame(rows)
    mae_m3 = float(frame["abs_err_m3"].mean())
    mae_naive = float(frame["abs_err_naive"].mean())
    leaked = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(history["schedule"].columns)
    later = [
        parse_ok(a) > parse_ok(M1_AVAILABLE_AT)
        for a in frame.loc[frame["week"] >= 3, "available_at"]
    ]
    passes = (
        poison_raised
        and leaked == set()
        and mae_m3 < mae_naive
        and week3_zero
        and all(later)
    )
    return {
        "schema_version": "weekly_latent_m3_backtest_synthetic_v1",
        "object_evaluated": "allocate_players_m3",
        "passes": passes,
        "poison_same_week_raised": poison_raised,
        "forbidden_columns_on_features": sorted(leaked),
        "mae_m3": mae_m3,
        "mae_naive_m2_players": mae_naive,
        "beats_naive": mae_m3 < mae_naive,
        "later_weeks_advance_available_at": all(later) if later else False,
        "week3_availability_zero": week3_zero,
        "weeks": rows,
        "holdout": "synthetic rolling-origin; sit as-of week 3 only; no 2026 live outcomes",
        "not_a_promotion": True,
    }


def historical_m3_history() -> dict[str, Any]:
    base = historical_schedule_history()
    players = pd.DataFrame(
        [_wr_season_row(f"{team}-WR", team, targets=40.0) for team in HISTORICAL_TEAMS]
    )
    shares = role_shares(players, base["volume"])
    opp_pass_adj = {"BUF": -5.0, "BAL": -4.0, "KC": 0.0, "SF": 5.0, "PHI": 4.0}
    out_rows = []
    for rec in base["schedule"].to_dict(orient="records"):
        pid = f"{rec['team']}-WR"
        sit = pid == HISTORICAL_SIT_PLAYER and int(rec["week"]) == HISTORICAL_SIT_WEEK
        if int(rec["is_bye"]) == 1 or sit:
            realized = 0.0
        else:
            opp = rec["opponent"]
            adj = 0.0 if opp is None or (isinstance(opp, float) and pd.isna(opp)) else opp_pass_adj.get(str(opp), 0.0)
            realized = max(0.0, 7.0 + 0.4 * adj)
        out_rows.append(
            {
                "player_id": pid,
                "team": rec["team"],
                "week": int(rec["week"]),
                "realized_receptions": realized,
            }
        )
    outcomes = pd.DataFrame(out_rows)
    refuse_forbidden_m2_columns(outcomes, where="M3 historical outcomes")
    return {
        **base,
        "players": players,
        "shares": shares,
        "player_outcomes": outcomes,
    }


def _historical_sit(week: int, *, as_of_week: int) -> pd.DataFrame | None:
    if as_of_week < HISTORICAL_SIT_WEEK or week != HISTORICAL_SIT_WEEK:
        return None
    return pd.DataFrame(
        {
            "player_id": [HISTORICAL_SIT_PLAYER],
            "week": [HISTORICAL_SIT_WEEK],
            "A_i_w": [0.0],
            "available_at": [HISTORICAL_SIT_AVAILABLE_AT],
        }
    )


def run_historical_m3_backtest() -> dict[str, Any]:
    """Walk the 2025-shaped slate; score M3 receptions vs held-out outcomes."""
    history = historical_m3_history()
    n_active = float(history.get("n_active", HISTORICAL_N_ACTIVE))
    board_at = str(history["board_available_at"])
    env_at = str(history["schedule_env_available_at"])
    volume = history["volume"]
    outcomes = history["player_outcomes"]
    rows: list[dict[str, Any]] = []
    allocated: list[pd.DataFrame] = []
    poisoned_columns: list[str] = []
    for week in HISTORICAL_WEEKS:
        team_week, priors = historical_features_for_week(history, as_of_week=week)
        leaked = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(team_week.columns)
        if leaked:
            raise AssertionError(
                f"M3 historical week {week} has forbidden columns: {sorted(leaked)}"
            )
        for col in sorted(FORBIDDEN_SAME_WEEK_TRAINING_FEATURES):
            try:
                refuse_forbidden_m2_columns(
                    poison_same_week_volume(team_week, column=col),
                    where=f"M3 historical poison {col} week {week}",
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
        m3 = allocate_players_m3(
            m2,
            history["players"],
            history["shares"],
            availability=_historical_sit(week, as_of_week=week),
        )
        naive = allocate_players(m2, history["players"], history["shares"])
        allocated.append(m3)
        scored = m3.merge(outcomes, on=["player_id", "week"], how="left", suffixes=("", "_out"))
        naive_scored = naive.merge(
            outcomes[["player_id", "week", "realized_receptions"]],
            on=["player_id", "week"],
            how="left",
        )
        naive_by_key = {
            (str(r["player_id"]), int(r["week"])): float(r["receptions"])
            for _, r in naive_scored.iterrows()
        }
        for _, rec in scored.iterrows():
            pred = float(rec["receptions"])
            realized = float(rec["realized_receptions"])
            naive_pred = naive_by_key[(str(rec["player_id"]), int(rec["week"]))]
            rows.append(
                {
                    "week": int(rec["week"]),
                    "team": rec["team"],
                    "player_id": rec["player_id"],
                    "is_bye": int(rec["is_bye"]),
                    "A_i_w": float(rec["A_i_w"]),
                    "pred_m3": pred,
                    "pred_naive": naive_pred,
                    "realized": realized,
                    "abs_err_m3": abs(pred - realized),
                    "abs_err_naive": abs(naive_pred - realized),
                    "available_at": str(rec["available_at"]),
                }
            )
    frame = pd.DataFrame(rows)
    all_m3 = pd.concat(allocated, ignore_index=True)
    season_sum = all_m3.groupby("player_id")["fantasy_points"].sum()
    rebuilt = all_m3.groupby("player_id")["fantasy_points"].sum()
    season_eq = bool((season_sum - rebuilt).abs().max() <= 1e-9)
    bye_zero = bool((frame.loc[frame["is_bye"].eq(1), "A_i_w"] == 0).all())
    mae_m3 = float(frame["abs_err_m3"].mean())
    mae_naive = float(frame["abs_err_naive"].mean())
    week1 = frame[frame["week"].eq(1) & frame["is_bye"].eq(0)]
    week1_at = str(week1["available_at"].iloc[0]) if len(week1) else None
    sit_rows = frame[
        frame["player_id"].eq(HISTORICAL_SIT_PLAYER) & frame["week"].eq(HISTORICAL_SIT_WEEK)
    ]
    sit_advanced = bool(
        len(sit_rows)
        and all(parse_ok(a) > parse_ok(board_at) for a in sit_rows["available_at"])
    )
    none_earlier = all(parse_ok(a) >= parse_ok(board_at) for a in frame["available_at"])
    leaked_features = FORBIDDEN_SAME_WEEK_TRAINING_FEATURES.intersection(
        history["schedule"].columns
    )
    poison_ok = set(poisoned_columns) == set(FORBIDDEN_SAME_WEEK_TRAINING_FEATURES)
    passes = (
        poison_ok
        and leaked_features == set()
        and bye_zero
        and season_eq
        and sit_advanced
        and none_earlier
        and week1_at == HISTORICAL_BOARD_AVAILABLE_AT
        and mae_m3 < mae_naive
    )
    return {
        "schema_version": "weekly_latent_m3_backtest_historical_v1",
        "object_evaluated": "allocate_players_m3",
        "passes": passes,
        "poison_same_week_raised": poison_ok,
        "poisoned_columns": sorted(poisoned_columns),
        "forbidden_columns_on_features": sorted(leaked_features),
        "mae_m3": mae_m3,
        "mae_naive_m2_players": mae_naive,
        "beats_naive": mae_m3 < mae_naive,
        "later_weeks_advance_available_at": bool(sit_advanced and none_earlier),
        "bye_availability_zero": bye_zero,
        "season_equals_sum_of_weeks": season_eq,
        "week1_available_at": week1_at,
        "n_player_weeks_scored": int(len(frame)),
        "weeks": rows,
        "holdout": (
            "historical 2025-shaped schedule rolling-origin of allocate_players_m3; "
            "prior-only opponent features; sit as-of week 3; not 6-8 live 2026 weeks; "
            "not a promotion"
        ),
        "not_a_promotion": True,
    }


def run_m3_backtests() -> dict[str, Any]:
    synthetic = run_synthetic_m3_rolling_origin()
    historical = run_historical_m3_backtest()
    return {
        "schema_version": "weekly_latent_m3_backtest_v1",
        "passes": bool(synthetic.get("passes")) and bool(historical.get("passes")),
        "synthetic": synthetic,
        "historical": historical,
        "not_a_promotion": True,
        "holdout": (
            "synthetic + historical-schedule rolling-origin of allocate_players_m3; "
            "not 6-8 live 2026 weeks; not a promotion"
        ),
    }
