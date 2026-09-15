"""Deterministic weekly allocation with exact season-total conservation.

Matchup multipliers (home/away + shrunk opponent) only reshape the weekly
path. After renormalization they cannot change team season totals. That is
the M1 contract: conservation first, before any later milestone is allowed
to let matchups move season mass.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.projection.weekly_latent.constants import (
    AWAY_MULT,
    CONVERSION_RATES,
    FORBIDDEN_SAME_WEEK_TRAINING_FEATURES,
    HOME_MULT,
    NEUTRAL_MULT,
    PLAYER_SHARE_POOLS,
    TEAM_VOLUME_PG_COLUMNS,
)
from src.projection.weekly_latent.schedule import attach_opponent_priors

# Half-PPR, 4-pt passing TD. Same numbers as
# src.projection.weekly.config.scoring.half_ppr() — imported here as literals so
# this shadow module does not load src.projection.weekly (polars pipeline).
HALF_PPR = {
    "pass_yard_points": 0.04,
    "pass_td_points": 4.0,
    "interception_points": -2.0,
    "rush_rec_yard_points": 0.1,
    "rush_rec_td_points": 6.0,
    "reception_points": 0.5,
}


def _home_away_mult(row: pd.Series) -> float:
    if int(row.get("is_bye") or 0) == 1:
        return 0.0
    if int(row.get("is_neutral") or 0) == 1:
        return NEUTRAL_MULT
    if int(row.get("is_home_for_mult") or row.get("is_home") or 0) == 1:
        return HOME_MULT
    return AWAY_MULT


def shrunk_opponent_mult(raw_factor: float, lam: float) -> float:
    """m = 1 + λ_w (s_opp − 1). λ_w → 0 late in the preseason slate."""
    return 1.0 + float(lam) * (float(raw_factor) - 1.0)


def _renormalize(raw: pd.Series, team: pd.Series) -> pd.Series:
    """Non-negative weights that sum to 1 within each team (or 0 if all mass is gone).

    Clip before summing. Zeroing negatives *after* dividing by a total that
    still includes them lets survivors sum to more than 1 (e.g. [2, -1, 1]
    → [1, 0, 0.5]). Opponent factor ≤ 0 with early-season λ=1 can reach that.
    """
    clipped = raw.clip(lower=0.0)
    totals = clipped.groupby(team).transform("sum")
    out = clipped.copy()
    positive = totals > 1e-12
    out.loc[positive] = clipped.loc[positive] / totals.loc[positive]
    out.loc[~positive] = 0.0
    return out


def allocate_team_weeks(
    team_weeks: pd.DataFrame,
    team_volume: pd.DataFrame,
    *,
    opponent_priors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Distribute each team's season volume across 18 weeks, bye = 0."""
    frame = attach_opponent_priors(team_weeks, opponent_priors)
    if set(FORBIDDEN_SAME_WEEK_TRAINING_FEATURES) & set(frame.columns):
        raise ValueError(
            "same-week realized volume columns leaked onto team-weeks: "
            + sorted(set(FORBIDDEN_SAME_WEEK_TRAINING_FEATURES) & set(frame.columns)).__repr__()
        )
    frame["home_away_mult"] = frame.apply(_home_away_mult, axis=1)
    frame["opp_pass_mult"] = [
        shrunk_opponent_mult(f, lam)
        for f, lam in zip(frame["opp_pass_factor"], frame["shrinkage_lambda"])
    ]
    frame["opp_rush_mult"] = [
        shrunk_opponent_mult(f, lam)
        for f, lam in zip(frame["opp_rush_factor"], frame["shrinkage_lambda"])
    ]
    active = frame["A_team_w"].astype(float)
    frame["raw_pass_weight"] = active * frame["home_away_mult"] * frame["opp_pass_mult"]
    frame["raw_rush_weight"] = active * frame["home_away_mult"] * frame["opp_rush_mult"]
    frame["pass_week_weight"] = _renormalize(frame["raw_pass_weight"], frame["team"])
    frame["rush_week_weight"] = _renormalize(frame["raw_rush_weight"], frame["team"])
    vol = team_volume.rename(
        columns={name: f"season_{name}" for name in TEAM_VOLUME_PG_COLUMNS}
    )
    frame = frame.merge(vol, on="team", how="left")
    pass_stats = ("team_pass_attempts", "team_passing_yards")
    rush_stats = ("team_rush_attempts", "team_rushing_yards")
    for name in pass_stats:
        frame[name] = frame["pass_week_weight"] * frame[f"season_{name}"].fillna(0.0)
    for name in rush_stats:
        frame[name] = frame["rush_week_weight"] * frame[f"season_{name}"].fillna(0.0)
    # Bye rows must be exact zeros even if a weight bug slips through.
    bye = frame["is_bye"].eq(1)
    for name in TEAM_VOLUME_PG_COLUMNS:
        frame.loc[bye, name] = 0.0
    return frame


def _conversion_rate(numer: pd.Series, denom: pd.Series) -> pd.Series:
    n = numer.astype(float)
    d = denom.astype(float)
    rate = pd.Series(0.0, index=n.index)
    ok = d.abs() > 1e-12
    rate.loc[ok] = n.loc[ok] / d.loc[ok]
    return rate


def _fantasy_points(stats: pd.DataFrame, scoring: dict[str, float] | None = None) -> pd.Series:
    scoring = scoring or HALF_PPR
    def col(name: str) -> pd.Series:
        if name not in stats.columns:
            return pd.Series(0.0, index=stats.index)
        return pd.to_numeric(stats[name], errors="coerce").fillna(0.0)

    return (
        col("passing_yards") * scoring["pass_yard_points"]
        + col("passing_tds") * scoring["pass_td_points"]
        + col("interceptions") * scoring["interception_points"]
        + col("rushing_yards") * scoring["rush_rec_yard_points"]
        + col("rushing_tds") * scoring["rush_rec_td_points"]
        + col("receptions") * scoring["reception_points"]
        + col("receiving_yards") * scoring["rush_rec_yard_points"]
        + col("receiving_tds") * scoring["rush_rec_td_points"]
    )


def allocate_players(
    team_weeks: pd.DataFrame,
    players: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    scoring: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Weekly player means = (role share) × (weekly team pool), bye inactive."""
    scoring = scoring or HALF_PPR
    share_wide = shares.pivot_table(
        index=["player_id", "team"],
        columns="stat",
        values="share",
        aggfunc="first",
    ).reset_index()
    share_wide.columns = [
        f"{c}_role_share" if c not in {"player_id", "team"} else c for c in share_wide.columns
    ]
    other = (
        shares.groupby(["team", "stat"], as_index=False)["other_share"]
        .first()
        .pivot(index="team", columns="stat", values="other_share")
        .reset_index()
    )
    other.columns = [
        f"{c}_other_share" if c != "team" else c for c in other.columns
    ]
    ident_cols = [
        c
        for c in (
            "player_id",
            "display_name",
            "team",
            "position",
            "depth_rank",
            "role",
            "projected_games",
        )
        if c in players.columns
    ]
    rate_src = players[ident_cols].copy()
    for stat, (numer, denom) in CONVERSION_RATES.items():
        rate_src[f"{stat}_rate"] = _conversion_rate(players[numer], players[denom])
    for player_stat in PLAYER_SHARE_POOLS:
        if player_stat in players.columns:
            rate_src[f"season_{player_stat}"] = players[player_stat]
    for numer, _denom in CONVERSION_RATES.items():
        if numer in players.columns:
            rate_src[f"season_{numer}"] = players[numer]

    team_cols = [
        "season",
        "week",
        "team",
        "opponent",
        "is_home",
        "is_neutral",
        "is_international",
        "is_division",
        "is_bye",
        "A_team_w",
        "rest_days",
        "roof",
        "surface",
        "stadium",
        "gameday",
        "shrinkage_lambda",
        "home_away_mult",
        "pass_week_weight",
        "rush_week_weight",
        *TEAM_VOLUME_PG_COLUMNS.keys(),
    ]
    team_cols = [c for c in team_cols if c in team_weeks.columns]
    base = rate_src.merge(share_wide, on=["player_id", "team"], how="left")
    weekly = team_weeks[team_cols].merge(other, on="team", how="left").merge(
        base, on="team", how="left"
    )
    weekly["A_i_w"] = weekly["A_team_w"].astype(float)
    for player_stat, pool in PLAYER_SHARE_POOLS.items():
        share_col = f"{player_stat}_role_share"
        if share_col not in weekly.columns:
            weekly[share_col] = 0.0
        weekly[share_col] = weekly[share_col].fillna(0.0)
        weekly[player_stat] = weekly[share_col] * weekly[pool].astype(float) * weekly["A_i_w"]
    for stat, (_numer, denom) in CONVERSION_RATES.items():
        weekly[stat] = weekly[f"{stat}_rate"].fillna(0.0) * weekly[denom].astype(float)
    bye = weekly["is_bye"].eq(1)
    zero_stats = list(PLAYER_SHARE_POOLS) + list(CONVERSION_RATES)
    for col in zero_stats:
        weekly.loc[bye, col] = 0.0
    weekly["fantasy_points"] = _fantasy_points(weekly, scoring)
    weekly.loc[bye, "fantasy_points"] = 0.0
    return weekly


def season_box_from_players(
    players: pd.DataFrame, scoring: dict[str, float] | None = None
) -> pd.DataFrame:
    scoring = scoring or HALF_PPR
    out = players.copy()
    out["fantasy_points_board"] = _fantasy_points(out, scoring)
    return out


@dataclass
class AllocationTables:
    team_weeks: pd.DataFrame
    player_weeks: pd.DataFrame
    shares: pd.DataFrame
    players: pd.DataFrame
    team_volume: pd.DataFrame
