"""Player-week/season usage tables shared by feature building, rookie
handling, and OL-quality aggregation.

Season window for the base usage table is 2016-2025, REG season only
(playoffs excluded - different opponent pool / small week count, would
distort per-game rates and team-share denominators). 2025's `weekly` rows
are the pbp-fallback aggregation (Phase 1 caveat) and have NULL
position/recent_team/season_type - both are backfilled here from `players`
and `weekly_rosters` respectively, and REG-ness is inferred from week<=18
(2025's schedule confirmed REG = weeks 1-18, POST = 19-22) since
`weekly.season_type` is null for the fallback rows.
"""
import os
import sqlite3

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(REPO_ROOT, "data", "projections.db")

POSITIONS = ["QB", "RB", "WR", "TE"]
SEASONS = list(range(2016, 2026))

STAT_COLS = [
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
]


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_weekly_usage(conn):
    """One row per (player_id, season, week) for QB/RB/WR/TE, REG season
    only, with `team` and `position` backfilled for the 2025 pbp-fallback
    rows (which ship with both null)."""
    w = pd.read_sql(
        f"select player_id, season, week, season_type, recent_team, position, "
        f"{', '.join(STAT_COLS)} from weekly", conn,
    )
    is_2025 = w["season"] == 2025
    w.loc[is_2025, "season_type"] = w.loc[is_2025, "week"].apply(lambda wk: "REG" if wk <= 18 else "POST")
    w = w[w["season_type"] == "REG"].copy()

    players_pos = pd.read_sql("select gsis_id, position from players", conn).rename(
        columns={"gsis_id": "player_id", "position": "pos_master"}
    )
    w = w.merge(players_pos, on="player_id", how="left")
    w["position"] = w["position"].fillna(w["pos_master"])
    w = w.drop(columns=["pos_master"])

    needs_team = w["recent_team"].isna()
    if needs_team.any():
        rosters = pd.read_sql(
            "select season, week, player_id, team from weekly_rosters where season = 2025", conn
        )
        w = w.merge(rosters, on=["season", "week", "player_id"], how="left", suffixes=("", "_roster"))
        w["recent_team"] = w["recent_team"].fillna(w["team"])
        w = w.drop(columns=["team"])

    w = w[w["position"].isin(POSITIONS)].reset_index(drop=True)
    w = w.rename(columns={"recent_team": "team"})
    return w


def season_aggregate(weekly_usage):
    """Player-season totals + games_played + resolved season team.

    games_played is counted on the position-relevant usage stat (QB:
    attempts>0; RB/WR/TE: carries>0 or targets>0) rather than "any row
    exists", since a player can appear in `weekly` for a week they were
    inactive/injured with all-zero stats.
    """
    df = weekly_usage.copy()
    df["_active"] = (
        ((df["position"] == "QB") & (df["attempts"] > 0))
        | ((df["position"] != "QB") & ((df["carries"] > 0) | (df["targets"] > 0)))
    )

    totals = df.groupby(["player_id", "season", "position"])[STAT_COLS].sum().reset_index()
    games = df[df["_active"]].groupby(["player_id", "season"]).size().rename("games_played").reset_index()
    totals = totals.merge(games, on=["player_id", "season"], how="left")
    totals["games_played"] = totals["games_played"].fillna(0).astype(int)

    # season team: the team with the most active weeks; ties broken by the
    # most recent week played for that team (approximates "who they
    # finished the season with" for in-season trades).
    active = df[df["_active"]].copy()
    team_counts = (
        active.groupby(["player_id", "season", "team"])
        .agg(n_weeks=("week", "size"), last_week=("week", "max"))
        .reset_index()
    )
    team_counts = team_counts.sort_values(["player_id", "season", "n_weeks", "last_week"])
    season_team = team_counts.groupby(["player_id", "season"]).tail(1)[["player_id", "season", "team"]]
    totals = totals.merge(season_team, on=["player_id", "season"], how="left")

    return totals


def team_season_pbp_totals(conn, seasons=SEASONS):
    """Team-season pass/rush/red-zone attempt totals from pbp, used as the
    denominator for player opportunity shares. REG season only (pbp has a
    real `season_type` column, unlike weekly's 2025 fallback rows)."""
    q = f"""
        select season, posteam as team, play_type, pass_attempt, rush_attempt,
               receiver_player_id, yardline_100
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and posteam is not null and (pass_attempt = 1 or rush_attempt = 1)
    """
    pbp = pd.read_sql(q, conn)
    pbp["is_rz"] = pbp["yardline_100"] <= 20

    pass_mask = pbp["pass_attempt"] == 1
    rush_mask = pbp["rush_attempt"] == 1

    g = pbp.groupby(["season", "team"])
    out = pd.DataFrame({
        "team_pass_attempts": pbp[pass_mask].groupby(["season", "team"]).size(),
        "team_rush_attempts": pbp[rush_mask].groupby(["season", "team"]).size(),
        "team_rz_pass_attempts": pbp[pass_mask & pbp["is_rz"]].groupby(["season", "team"]).size(),
        "team_rz_rush_attempts": pbp[rush_mask & pbp["is_rz"]].groupby(["season", "team"]).size(),
    }).reset_index()
    return out


def player_rz_usage(conn, seasons=SEASONS):
    """Player-season red-zone carries/targets from pbp (yardline_100<=20)."""
    q = f"""
        select season, posteam as team, rush_attempt, pass_attempt,
               rusher_player_id, receiver_player_id, yardline_100
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and yardline_100 <= 20 and (pass_attempt = 1 or rush_attempt = 1)
    """
    pbp = pd.read_sql(q, conn)
    rz_carries = (
        pbp[pbp["rush_attempt"] == 1]
        .groupby(["season", "rusher_player_id"]).size().rename("rz_carries").reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )
    rz_targets = (
        pbp[pbp["pass_attempt"] == 1]
        .groupby(["season", "receiver_player_id"]).size().rename("rz_targets").reset_index()
        .rename(columns={"receiver_player_id": "player_id"})
    )
    out = rz_carries.merge(rz_targets, on=["season", "player_id"], how="outer")
    out[["rz_carries", "rz_targets"]] = out[["rz_carries", "rz_targets"]].fillna(0)
    return out


def player_season_snap_pct(conn, seasons=SEASONS):
    """Average offensive snap % across a player's season, joined via
    players.pfr_id (snap_counts keys off pfr_player_id, not gsis_id)."""
    sc = pd.read_sql(
        f"select season, week, pfr_player_id, offense_pct from snap_counts "
        f"where season in ({','.join(map(str, seasons))}) and game_type = 'REG'", conn,
    )
    crosswalk = pd.read_sql("select gsis_id, pfr_id from players where pfr_id is not null", conn)
    sc = sc.merge(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    out = sc.groupby(["season", "gsis_id"])["offense_pct"].mean().reset_index()
    return out.rename(columns={"gsis_id": "player_id", "offense_pct": "snap_pct"})
