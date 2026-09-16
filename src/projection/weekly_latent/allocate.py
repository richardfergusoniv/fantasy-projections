"""Deterministic weekly allocation with exact season-total conservation.

Matchup multipliers (home/away + shrunk opponent) only reshape the weekly
path. After renormalization they cannot change team season totals. That is
the M1 contract: conservation first, before any later milestone is allowed
to let matchups move season mass.

Every team-week and player-week row carries ``available_at``. For M1 this is
the sealed preseason snapshot (``M1_AVAILABLE_AT`` / ``v2_baseline_20260830``),
not kickoff: the only features on the row are schedule scaffolding and
allocated season volume. M2 must overwrite that cutoff when opponent priors
or in-season updates arrive.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.projection.weekly_latent.constants import (
    AWAY_MULT,
    CONVERSION_RATES,
    FORBIDDEN_SAME_WEEK_TRAINING_FEATURES,
    HOME_MULT,
    M1_AVAILABLE_AT,
    M2_SCHEDULE_ENV_AVAILABLE_AT,
    NEUTRAL_MULT,
    PLAYER_SHARE_POOLS,
    TEAM_VOLUME_PG_COLUMNS,
    YARDAGE_PER_OPP,
)
from src.projection.weekly_latent.environment import (
    attach_environment,
    refuse_forbidden_m2_columns,
    stamp_available_at,
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
    # One writer for the M1 cutoff. Preseason snapshot, not gameday/kickoff.
    frame["available_at"] = M1_AVAILABLE_AT
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


def allocate_team_weeks_m2(
    team_weeks: pd.DataFrame,
    team_volume: pd.DataFrame,
    *,
    opponent_priors: pd.DataFrame | None = None,
    board_available_at: str = M1_AVAILABLE_AT,
    n_active_override: float | None = None,
    schedule_env_available_at: str | None = None,
) -> pd.DataFrame:
    """Team-week latent that may move season mass vs the sealed prior.

    Identity (M2, not M1)::

        V_{t,k,w} = A_{t,w} * (V_{t,k}^{sealed} / sum_{w'} A_{t,w'})
                    * m^{HA}_{t,w} * m^{opp}_{t,k,w} * m^{env}_{t,w}

    Multipliers are **not** renormalized, so
    ``sum_w V_{t,k,w} = V^{sealed} * mean_{active}(m_HA * m_opp * m_env)``
    and may differ from the sealed season total. Bye weeks stay 0.

    ``available_at`` is the latest attached vintage (board, schedule-env,
    opponent prior), not blindly the M1 preseason stamp.
    """
    frame = attach_opponent_priors(team_weeks, opponent_priors)
    refuse_forbidden_m2_columns(frame, where="M2 allocate_team_weeks")
    if opponent_priors is not None and not opponent_priors.empty:
        refuse_forbidden_m2_columns(opponent_priors, where="M2 opponent_priors")
    env_stamp = schedule_env_available_at or M2_SCHEDULE_ENV_AVAILABLE_AT
    frame = attach_environment(frame, env_available_at=env_stamp)
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
    frame["pass_matchup_mult"] = (
        frame["home_away_mult"].astype(float)
        * frame["opp_pass_mult"].astype(float)
        * frame["env_mult"].astype(float)
    )
    frame["rush_matchup_mult"] = (
        frame["home_away_mult"].astype(float)
        * frame["opp_rush_mult"].astype(float)
        * frame["env_mult"].astype(float)
    )
    # Un-normalized path: raw weight is A * matchup. M1 would divide by
    # the team sum here; M2 does not.
    frame["raw_pass_weight"] = active * frame["pass_matchup_mult"]
    frame["raw_rush_weight"] = active * frame["rush_matchup_mult"]
    if n_active_override is not None:
        n_active = pd.Series(float(n_active_override), index=frame.index)
    else:
        n_active = active.groupby(frame["team"]).transform("sum")
    frame["n_active_weeks"] = n_active
    vol = team_volume.rename(
        columns={name: f"season_{name}" for name in TEAM_VOLUME_PG_COLUMNS}
    )
    frame = frame.merge(vol, on="team", how="left")
    pass_stats = ("team_pass_attempts", "team_passing_yards")
    rush_stats = ("team_rush_attempts", "team_rushing_yards")
    for name in pass_stats:
        sealed = frame[f"season_{name}"].fillna(0.0)
        per_active = pd.Series(0.0, index=frame.index)
        ok = n_active > 1e-12
        per_active.loc[ok] = sealed.loc[ok] / n_active.loc[ok]
        frame[f"baseline_{name}"] = per_active
        frame[name] = active * per_active * frame["pass_matchup_mult"]
    for name in rush_stats:
        sealed = frame[f"season_{name}"].fillna(0.0)
        per_active = pd.Series(0.0, index=frame.index)
        ok = n_active > 1e-12
        per_active.loc[ok] = sealed.loc[ok] / n_active.loc[ok]
        frame[f"baseline_{name}"] = per_active
        frame[name] = active * per_active * frame["rush_matchup_mult"]
    bye = frame["is_bye"].eq(1)
    for name in TEAM_VOLUME_PG_COLUMNS:
        frame.loc[bye, name] = 0.0
    if opponent_priors is not None and not opponent_priors.empty and "available_at" in opponent_priors.columns:
        prior_stamp = opponent_priors.rename(
            columns={"team": "opponent"} if "team" in opponent_priors.columns else {}
        )
        slim = prior_stamp[["opponent", "available_at"]].drop_duplicates("opponent")
        slim = slim.rename(columns={"available_at": "prior_available_at"})
        if "prior_available_at" in frame.columns:
            frame = frame.drop(columns=["prior_available_at"])
        frame = frame.merge(slim, on="opponent", how="left")
    else:
        frame["prior_available_at"] = pd.NA
    frame = stamp_available_at(frame, board_available_at)
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


def _player_week_skeleton(
    team_weeks: pd.DataFrame,
    players: pd.DataFrame,
    shares: pd.DataFrame,
) -> pd.DataFrame:
    """Join team-weeks × role shares × season rates. No availability yet."""
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
    for yards_stat, opp_stat in YARDAGE_PER_OPP.items():
        numer = players[yards_stat] if yards_stat in players.columns else 0.0
        denom = players[opp_stat] if opp_stat in players.columns else 0.0
        rate_src[f"{yards_stat}_per_opp"] = _conversion_rate(
            numer if isinstance(numer, pd.Series) else pd.Series(0.0, index=players.index),
            denom if isinstance(denom, pd.Series) else pd.Series(0.0, index=players.index),
        )
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
        "gametime",
        "kickoff_at",
        "available_at",
        "shrinkage_lambda",
        "home_away_mult",
        "pass_week_weight",
        "rush_week_weight",
        "env_mult",
        "rest_mult",
        "travel_mult",
        "pass_matchup_mult",
        "rush_matchup_mult",
        "opp_pass_mult",
        "opp_rush_mult",
        "n_active_weeks",
        "prior_available_at",
        "available_at_board",
        "env_available_at",
        *TEAM_VOLUME_PG_COLUMNS.keys(),
    ]
    team_cols = [c for c in team_cols if c in team_weeks.columns]
    base = rate_src.merge(share_wide, on=["player_id", "team"], how="left")
    weekly = team_weeks[team_cols].merge(other, on="team", how="left").merge(
        base, on="team", how="left"
    )
    return weekly


def _apply_opportunity(weekly: pd.DataFrame) -> pd.DataFrame:
    """Player opportunity = A_i,w × role share × weekly team pool."""
    out = weekly.copy()
    for player_stat, pool in PLAYER_SHARE_POOLS.items():
        share_col = f"{player_stat}_role_share"
        if share_col not in out.columns:
            out[share_col] = 0.0
        out[share_col] = out[share_col].fillna(0.0)
        pool_series = (
            out[pool].astype(float) if pool in out.columns else pd.Series(0.0, index=out.index)
        )
        out[player_stat] = out[share_col] * pool_series * out["A_i_w"].astype(float)
    return out


def allocate_players(
    team_weeks: pd.DataFrame,
    players: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    scoring: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Weekly player means = (role share) × (weekly team pool), bye inactive."""
    scoring = scoring or HALF_PPR
    weekly = _player_week_skeleton(team_weeks, players, shares)
    weekly["A_i_w"] = weekly["A_team_w"].astype(float)
    weekly = _apply_opportunity(weekly)
    for stat, (_numer, denom) in CONVERSION_RATES.items():
        weekly[stat] = weekly[f"{stat}_rate"].fillna(0.0) * weekly[denom].astype(float)
    bye = weekly["is_bye"].eq(1)
    zero_stats = list(PLAYER_SHARE_POOLS) + list(CONVERSION_RATES)
    for col in zero_stats:
        weekly.loc[bye, col] = 0.0
    weekly["fantasy_points"] = _fantasy_points(weekly, scoring)
    weekly.loc[bye, "fantasy_points"] = 0.0
    return weekly


def allocate_players_m3(
    team_weeks: pd.DataFrame,
    players: pd.DataFrame,
    shares: pd.DataFrame,
    *,
    availability: pd.DataFrame | None = None,
    scoring: dict[str, float] | None = None,
) -> pd.DataFrame:
    """M3 player-weeks: week-varying A_i,w and conversion latents.

    Opportunity still uses M2 team pools × role shares × A_i,w. Yards and
    count stats use season rates × opponent-only conversion multipliers.
    Bye forces A=0. Season fantasy is the sum of weeks.
    """
    from src.projection.weekly_latent.availability import (
        attach_player_availability,
        stamp_player_available_at,
    )
    from src.projection.weekly_latent.conversions import apply_conversion_latents
    from src.projection.weekly_latent.environment import refuse_forbidden_m2_columns

    scoring = scoring or HALF_PPR
    refuse_forbidden_m2_columns(team_weeks, where="M3 allocate_players team-weeks")
    refuse_forbidden_m2_columns(players, where="M3 allocate_players players")
    weekly = _player_week_skeleton(team_weeks, players, shares)
    weekly = attach_player_availability(weekly, overrides=availability)
    weekly = _apply_opportunity(weekly)
    # M3 yards come from conversion latents, not the team yardage-pool share.
    weekly = apply_conversion_latents(weekly)
    bye = weekly["is_bye"].eq(1) if "is_bye" in weekly.columns else pd.Series(False, index=weekly.index)
    zero_stats = list(PLAYER_SHARE_POOLS) + list(CONVERSION_RATES)
    for col in zero_stats:
        if col in weekly.columns:
            weekly.loc[bye, col] = 0.0
    weekly["fantasy_points"] = _fantasy_points(weekly, scoring)
    weekly.loc[bye, "fantasy_points"] = 0.0
    weekly = stamp_player_available_at(weekly)
    return weekly


def season_box_from_players(
    players: pd.DataFrame, scoring: dict[str, float] | None = None
) -> pd.DataFrame:
    scoring = scoring or HALF_PPR
    out = players.copy()
    out["fantasy_points_board"] = _fantasy_points(out, scoring)
    return out


def season_box_from_player_weeks(
    player_weeks: pd.DataFrame, scoring: dict[str, float] | None = None
) -> pd.DataFrame:
    """Independent season box: sum weekly box stats, then score.

    This is not a second groupby of ``fantasy_points``. Scoring the
    aggregated box can disagree with ``sum_w fantasy_points`` if the weekly
    points column was set independently of the box.
    """
    keys = ["player_id"]
    if "season" in player_weeks.columns:
        keys = ["player_id", "season"]
    stat_cols = [
        c
        for c in list(PLAYER_SHARE_POOLS) + list(CONVERSION_RATES)
        if c in player_weeks.columns
    ]
    if "player_id" not in player_weeks.columns or not stat_cols:
        return season_box_from_players(pd.DataFrame(columns=["player_id"]), scoring)
    grouped = player_weeks.groupby(keys, as_index=False)[stat_cols].sum()
    return season_box_from_players(grouped, scoring)


@dataclass
class AllocationTables:
    team_weeks: pd.DataFrame
    player_weeks: pd.DataFrame
    shares: pd.DataFrame
    players: pd.DataFrame
    team_volume: pd.DataFrame
