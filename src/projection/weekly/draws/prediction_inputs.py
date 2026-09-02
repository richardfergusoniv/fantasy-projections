"""Leakage-safe builders for joint draw prediction inputs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.projection.weekly.config.paths import SKILL_POSITIONS
from src.projection.weekly.draws.feature_outcome_split import (
    SAME_WEEK_OUTCOME_DENYLIST,
    assert_no_outcome_columns,
    is_allowed_prediction_column,
)
from src.projection.weekly.draws.game_engine import (
    PlayerGameInput,
    ScheduledGameInput,
    TeamGameInput,
)

FORBIDDEN_INPUT_KEYS = SAME_WEEK_OUTCOME_DENYLIST | frozenset(
    {
        "fantasy_points",
        "offense_snaps",
        "targets",
        "carries",
        "attempts",
        "target_share",
        "carry_share",
        "snap_share",
    }
)

REQUIRED_PREDICTION_KEYS = frozenset(
    {
        "p_active",
        "p_participates",
        "p_positive_usage",
        "pred_target_share",
        "pred_carry_share",
        "pred_mean_pass_attempts",
        "pred_mean_rush_attempts",
    }
)


def _reject_actual_columns(row: Mapping[str, Any]) -> None:
    blocked = [k for k in row if k in FORBIDDEN_INPUT_KEYS]
    if blocked:
        raise ValueError(f"row contains forbidden same-week actual columns: {sorted(blocked)}")


def projection_rows_only(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Strip same-week outcome columns before building draw inputs."""
    cleaned: list[dict[str, Any]] = []
    for r in rows:
        cleaned.append({k: v for k, v in r.items() if is_allowed_prediction_column(k) or k.startswith("p_") or k.startswith("pred_")})
    return cleaned


def build_team_game_input_from_predictions(
    team_rows: Sequence[Mapping[str, Any]],
    *,
    team: str,
    opponent: str,
    home: bool = True,
) -> TeamGameInput:
    """Construct ``TeamGameInput`` from fold-specific predictions only."""
    for r in team_rows:
        _reject_actual_columns(r)
    team_rows = projection_rows_only(team_rows)
    if not team_rows:
        return TeamGameInput(team=team, opponent=opponent, home=home, players=[])
    assert_no_outcome_columns(list(team_rows[0].keys()))

    players: list[PlayerGameInput] = []
    pass_atts: list[float] = []
    rush_atts = 0.0
    for r in team_rows:
        pos = str(r.get("position") or "")
        if pos not in SKILL_POSITIONS:
            continue
        pid = str(r.get("gsis_id") or r.get("player_id") or "")
        if not pid:
            continue
        for key in ("p_participates", "p_positive_usage"):
            if key not in r or r[key] is None:
                raise ValueError(f"missing required event prediction {key!r} for {pid}")
        p_active = float(r.get("p_active") if r.get("p_active") is not None else r.get("play_prob") or 1.0)
        players.append(
            PlayerGameInput(
                player_id=pid,
                position=pos,
                team=team,
                p_active=p_active,
                p_participates=float(r["p_participates"]),
                p_positive_usage=float(r["p_positive_usage"]),
                target_share=float(r.get("pred_target_share") or 0.05),
                carry_share=float(r.get("pred_carry_share") or 0.05),
                dropback_share=float(r.get("pred_dropback_share") or (0.97 if pos == "QB" else 0.0)),
                catch_rate=float(r.get("pred_catch_rate") or 0.65),
                ypa=float(r.get("pred_ypa") or 7.2),
                ypc=float(r.get("pred_ypc") or 4.2),
                ypr=float(r.get("pred_ypr") or 11.0),
            )
        )
        if pos == "QB" and r.get("pred_mean_pass_attempts") is not None:
            pass_atts.append(float(r["pred_mean_pass_attempts"]))
        rush_atts += float(r.get("pred_carry_share") or 0.0)

    team_pass = float(np.mean(pass_atts)) if pass_atts else float(team_rows[0].get("pred_mean_pass_attempts") or 34.0)
    team_rush = float(team_rows[0].get("pred_mean_rush_attempts") or max(15.0, rush_atts * 30.0))
    return TeamGameInput(
        team=team,
        opponent=opponent,
        home=home,
        mean_pass_attempts=max(20.0, team_pass),
        mean_rush_attempts=max(15.0, team_rush),
        players=players,
    )


def build_scheduled_game_from_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    game_id: str,
    season: int,
    week: int,
    home_team: str,
    away_team: str,
) -> ScheduledGameInput:
    home_rows = [r for r in rows if str(r.get("team")) == home_team]
    away_rows = [r for r in rows if str(r.get("team")) == away_team]
    return ScheduledGameInput(
        game_id=game_id,
        season=season,
        week=week,
        home=build_team_game_input_from_predictions(
            home_rows, team=home_team, opponent=away_team, home=True
        ),
        away=build_team_game_input_from_predictions(
            away_rows, team=away_team, opponent=home_team, home=False
        ),
    )
