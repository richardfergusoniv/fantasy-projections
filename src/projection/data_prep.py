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


# NFL expanded to a 17-game regular season starting 2021 - used below as the
# denominator for "how many of the team's scheduled games did this player
# miss," a well-known structural fact (not derived from a query) in the same
# spirit as this project's other hardcoded structural constants
# (PARTICIPATION_MIN_SEASON, FTN_MIN_SEASON, etc. in src/ingest/sources.py).
def _team_season_game_count(season):
    return 17 if season >= 2021 else 16


# Weight applied to a week the player was flagged Out/Doubtful/Questionable
# on the injury report but STILL PLAYED (per games_played's own active-week
# definition), relative to a full weight of 1.0 for a week they missed
# outright. Stated judgment call: a genuinely missed game is a stronger
# durability signal for NEXT season than "played through a questionable
# tag" - equal weighting would score a player who was Questionable every
# single week but never missed a game identically to a player who missed
# half the season outright, which is exactly the distinction this feature
# is supposed to capture (see build_player_season_injury_durability's
# docstring). 0.4 is a considered-but-not-tuned choice, not fit to any
# target - same spirit as this project's other stated-not-tuned constants
# (train.py's LGBM_PARAMS, rookies.py's VACATED_CLIP).
INJURY_PLAYED_WEIGHT = 0.4


def build_player_season_injury_durability(conn, seasons=SEASONS):
    """Player-season trailing injury-durability feature (Addendum 4).

    Measures: (missed games this season, weighted 1.0 each) + (games this
    season the player carried an Out/Doubtful/Questionable report status
    but STILL PLAYED, weighted INJURY_PLAYED_WEIGHT each), as a fraction of
    the team's scheduled games that season - clipped to [0, 1].

    This is a TRAILING feature exactly like every other column in
    FEATURE_COLS: it is computed from season N's own observed injury
    history and fed into the season-N feature row, which transitions.py's
    existing season-N -> season-(N+1) pairing turns into a genuine trailing
    predictor for next season automatically - there is no separate shift
    logic needed here, matching how carry_share/snap_pct/etc. already work.

    Why not just "fraction of weeks flagged on the report," full stop
    (the first option floated for this feature): spot-checked a real,
    extreme case (Christian McCaffrey, 2024 Achilles injury, played only 4
    of 17 games) and found the injuries table STOPS filing weekly reports
    for a player once they are on long-term IR - his 2024 injuries rows
    exist only for weeks 1-2 (Questionable/Out, before the initial
    IR placement) and week 10 (Questionable, on his way back); weeks 3-9 have
    NO injuries row at all, not because he wasn't hurt but because there is
    nothing left to report once a player is already out long-term. A
    "fraction of weeks flagged" metric using only weeks with an injuries
    row - or even using only weeks with a `weekly` row as the denominator -
    would score McCaffrey's 2024 as barely notable, exactly backwards for a
    season that should be a maximal durability red flag. Anchoring the
    denominator to the team's actual scheduled game count and crediting
    every genuinely MISSED game (regardless of whether a report row exists
    for it) fixes this: McCaffrey 2024 comes out to (13 missed + 0.4*1
    flagged-but-played week) / 17 = 0.79, correctly one of the highest
    durability-risk scores in the dataset (verified below in the report).

    Collapsing multiple injuries rows per player-week: the table has no
    day-of-week field (checked directly against the actual schema - only a
    raw `date_modified` timestamp), so a Wednesday-practice-report vs.
    Friday-final-status distinction isn't recoverable from what nflverse
    ships. Spot-checked the real row structure: duplicate (season, week,
    gsis_id) rows are rare (2 of ~5-6k player-weeks checked for 2024) and
    represent a status UPDATE during the week (e.g. Questionable -> Out)
    rather than genuinely distinct practice-day snapshots - the judgment
    call made here is to take the row with the latest `date_modified` per
    player-week as that week's status, which is equivalent to "most
    recently known status" and, given how rare duplicates are, close enough
    to "final status of the week" in practice.

    Players who never appear on an injury report AND never miss a game get
    a real 0.0 (not NaN) - by construction of the arithmetic above, not a
    silently-filled default."""
    inj = pd.read_sql(
        f"select season, week, gsis_id as player_id, report_status, date_modified from injuries "
        f"where season in ({','.join(map(str, seasons))}) and gsis_id is not null", conn,
    )
    inj = inj.sort_values("date_modified").drop_duplicates(subset=["season", "week", "player_id"], keep="last")
    inj["flagged"] = inj["report_status"].isin(["Out", "Doubtful", "Questionable"])
    flagged = inj[inj["flagged"]][["season", "week", "player_id"]].drop_duplicates()

    wu = load_weekly_usage(conn)
    wu = wu[wu["season"].isin(seasons)].copy()
    wu["_active"] = (
        ((wu["position"] == "QB") & (wu["attempts"] > 0))
        | ((wu["position"] != "QB") & ((wu["carries"] > 0) | (wu["targets"] > 0)))
    )

    played = wu[wu["_active"]][["season", "week", "player_id"]].drop_duplicates()
    games_played = played.groupby(["season", "player_id"]).size().rename("games_played").reset_index()

    played_flagged = played.merge(flagged, on=["season", "week", "player_id"], how="inner")
    flagged_played_weeks = (
        played_flagged.groupby(["season", "player_id"]).size().rename("flagged_played_weeks").reset_index()
    )

    out = games_played.merge(flagged_played_weeks, on=["season", "player_id"], how="left")
    out["flagged_played_weeks"] = out["flagged_played_weeks"].fillna(0)
    out["team_games"] = out["season"].apply(_team_season_game_count)
    out["missed_games"] = (out["team_games"] - out["games_played"]).clip(lower=0)
    out["injury_durability_rate"] = (
        (out["missed_games"] + INJURY_PLAYED_WEIGHT * out["flagged_played_weeks"]) / out["team_games"]
    ).clip(upper=1.0)
    return out[["season", "player_id", "injury_durability_rate"]]
