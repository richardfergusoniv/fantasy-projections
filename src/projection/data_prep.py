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


_SNAP_APPEARANCE_COLUMNS = {
    "season", "week", "pfr_player_id", "team", "offense_pct", "game_type",
}


def _opportunity_mask(df):
    """Weeks with a position-relevant box-score opportunity."""
    return (
        ((df["position"] == "QB") & (df["attempts"] > 0))
        | ((df["position"] != "QB") & ((df["carries"] > 0) | (df["targets"] > 0)))
    )


def _canonicalize_player_weeks(df):
    """Collapse source aliases to one row per player-season-week.

    The 2025 PBP fallback can emit more than one display-name alias for the
    same GSIS id in a week.  Stats belong to the canonical id and therefore
    add; appearances do not.  Team/position conflicts are resolved
    deterministically from the row carrying the most box-score volume.
    """
    if df.empty:
        return df.copy()
    w = df.copy()
    w[STAT_COLS] = w[STAT_COLS].fillna(0.0)
    w["_row_volume"] = w[STAT_COLS].abs().sum(axis=1)
    identity = (
        w.sort_values(["player_id", "season", "week", "_row_volume"], ascending=[True, True, True, False])
        .drop_duplicates(["player_id", "season", "week"])
        [["player_id", "season", "week", "season_type", "team", "position"]]
    )
    totals = w.groupby(["player_id", "season", "week"], as_index=False)[STAT_COLS].sum()
    appeared = None
    if "_appeared" in w.columns:
        appeared = (
            w.groupby(["player_id", "season", "week"], as_index=False)["_appeared"]
            .max()
        )
    out = identity.merge(totals, on=["player_id", "season", "week"], how="inner")
    if appeared is not None:
        out = out.merge(appeared, on=["player_id", "season", "week"], how="left")
    return out


def _validate_snap_appearance_schema(conn):
    """Fail early, via sqlite itself, if the optional snap schema is old.

    Keeping this separate from the pandas/merge work is intentional: only a
    known database-compatibility failure may trigger the opportunity fallback.
    Programming, crosswalk, and merge failures must remain visible.
    """
    cols = ", ".join(sorted(_SNAP_APPEARANCE_COLUMNS))
    conn.execute(f"select {cols} from snap_counts limit 0")


def _supported_snap_schema_error(exc):
    """Whether an sqlite OperationalError is the optional snap compatibility case."""
    msg = str(exc).lower()
    if "no such table: snap_counts" in msg:
        return True
    if "no such column:" not in msg:
        return False
    missing = msg.split("no such column:", 1)[1].strip().strip('"`[]')
    missing = missing.rsplit(".", 1)[-1]
    return missing in _SNAP_APPEARANCE_COLUMNS


def _augment_snap_appearances(base, conn):
    """Return a fully snap-augmented copy; never mutate ``base`` in place."""
    w = base.copy()
    snaps = pd.read_sql(
        f"select season, week, pfr_player_id, team, offense_pct from snap_counts "
        f"where season in ({','.join(map(str, SEASONS))}) and game_type = 'REG' "
        f"and offense_pct > 0", conn,
    )
    xwalk = pd.read_sql(
        "select gsis_id as player_id, pfr_id, position as master_position from players "
        "where gsis_id is not null and pfr_id is not null", conn,
    )
    snaps = snaps.merge(xwalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    snaps = snaps[snaps["master_position"].isin(POSITIONS)]
    appeared = snaps[["player_id", "season", "week"]].drop_duplicates()
    appeared["_appeared"] = True
    w = w.merge(appeared, on=["player_id", "season", "week"], how="left")

    existing = w[["player_id", "season", "week"]].drop_duplicates()
    missing = snaps.merge(existing, on=["player_id", "season", "week"], how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].copy()
    if not missing.empty:
        # The players master is career/latest-position data. Reusing it for
        # a historical zero-opportunity week can split one player-season
        # across positions. Prefer the position observed that season.
        season_pos = (
            w.sort_values("week")
            .dropna(subset=["position"])
            .drop_duplicates(["player_id", "season"], keep="last")
            [["player_id", "season", "position"]]
            .rename(columns={"position": "season_position"})
        )
        missing = missing.merge(season_pos, on=["player_id", "season"], how="left")
        missing["position"] = missing["season_position"].fillna(missing["master_position"])
        add = missing[["player_id", "season", "week", "team", "position"]].drop_duplicates()
        add["season_type"] = "REG"
        for c in STAT_COLS:
            add[c] = 0.0
        add["_appeared"] = True
        w = pd.concat([w, add[w.columns]], ignore_index=True)

    opportunity = _opportunity_mask(w)
    w["_appeared"] = w["_appeared"].fillna(opportunity).astype(bool)
    return w


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

    w = w.rename(columns={"recent_team": "team"})

    # `weekly` is a box-score table, not an appearance table. In particular,
    # the 2025 pbp fallback contains only players credited with an opportunity,
    # so a receiver who ran routes but drew no target disappears. Augment it
    # with offensive snap rows and keep appearance separate from opportunity.
    # This makes games_played mean "appeared on offense" while retaining the
    # old opportunity-based definition as `opportunity_games` downstream.
    try:
        _validate_snap_appearance_schema(conn)
    except sqlite3.OperationalError as exc:
        if not _supported_snap_schema_error(exc):
            raise
        # Compatibility fallback for an older/minimal DB: appearance cannot
        # be distinguished from opportunity, so assign the booleans directly.
        # Do not assign False and then call fillna(False): booleans have no
        # missing values for fillna to replace.
        w = w.copy()
        w["_appeared"] = _opportunity_mask(w).astype(bool)
        augmented = w
    else:
        # No exception handling here by design. If SQL, crosswalking, merging,
        # or augmentation is broken despite a compatible snap schema, fail loudly.
        augmented = _augment_snap_appearances(w, conn)

    # Filter only after snap augmentation.  This lets augmentation inherit a
    # player's actual season position before the career/latest master position
    # is considered, preventing historical FB/TE/QB rows from being re-added
    # as zero-stat rows under a different modern position.
    augmented = augmented[augmented["position"].isin(POSITIONS)].reset_index(drop=True)
    return _canonicalize_player_weeks(augmented)


def season_aggregate(weekly_usage):
    """Player-season totals + games_played + resolved season team.

    games_played counts offensive appearances derived from snap counts.
    opportunity_games preserves the narrower usage definition (QB:
    attempts>0; RB/WR/TE: carries>0 or targets>0) for conditional-rate and
    rookie-survival analysis.
    """
    df = weekly_usage.copy()
    df["_active"] = _opportunity_mask(df)

    totals = df.groupby(["player_id", "season", "position"])[STAT_COLS].sum().reset_index()
    appeared_col = "_appeared" if "_appeared" in df.columns else "_active"
    games = (
        df[df[appeared_col]][["player_id", "season", "week"]].drop_duplicates()
        .groupby(["player_id", "season"]).size().rename("games_played").reset_index()
    )
    opp_games = (
        df[df["_active"]][["player_id", "season", "week"]].drop_duplicates()
        .groupby(["player_id", "season"]).size().rename("opportunity_games").reset_index()
    )
    totals = totals.merge(games, on=["player_id", "season"], how="left").merge(
        opp_games, on=["player_id", "season"], how="left")
    totals[["games_played", "opportunity_games"]] = totals[["games_played", "opportunity_games"]].fillna(0).astype(int)

    # season team: the team with the most offensive-appearance weeks; ties broken by the
    # most recent week played for that team (approximates "who they
    # finished the season with" for in-season trades).
    active = df[df[appeared_col]].drop_duplicates(
        ["player_id", "season", "week", "team"]
    ).copy()
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


def team_week_pbp_totals(conn, seasons=SEASONS):
    """Same as team_season_pbp_totals but grouped by week too - the
    building block for player_active_team_opportunity below, which needs
    to sum only the weeks a given player was actually active rather than
    every week the team played."""
    q = f"""
        select season, week, posteam as team, pass_attempt, rush_attempt, yardline_100
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and posteam is not null and (pass_attempt = 1 or rush_attempt = 1)
    """
    pbp = pd.read_sql(q, conn)
    pbp["is_rz"] = pbp["yardline_100"] <= 20

    pass_mask = pbp["pass_attempt"] == 1
    rush_mask = pbp["rush_attempt"] == 1

    out = pd.DataFrame({
        "team_pass_attempts": pbp[pass_mask].groupby(["season", "week", "team"]).size(),
        "team_rush_attempts": pbp[rush_mask].groupby(["season", "week", "team"]).size(),
        "team_rz_pass_attempts": pbp[pass_mask & pbp["is_rz"]].groupby(["season", "week", "team"]).size(),
        "team_rz_rush_attempts": pbp[rush_mask & pbp["is_rz"]].groupby(["season", "week", "team"]).size(),
    }).reset_index()
    return out


def team_week_air_yards(conn, seasons=SEASONS):
    """Team-week total receiving air yards from pbp - the weekly building
    block player_active_team_opportunity needs to make team_air_yards
    games-played-aware the same way it already is for pass/rush attempts.
    Same air_yards-not-null exclusion as player_season_air_yards (sacks
    have no target thrown, not a real 0)."""
    q = f"""
        select season, week, posteam as team, air_yards
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and posteam is not null and pass_attempt = 1 and air_yards is not null
    """
    pbp = pd.read_sql(q, conn)
    return pbp.groupby(["season", "week", "team"])["air_yards"].sum().rename("team_air_yards").reset_index()


def player_active_team_opportunity(conn, seasons=SEASONS):
    """Player-season team pass/rush/red-zone attempt totals (+ team air
    yards), restricted to the WEEKS THIS PLAYER APPEARED ON OFFENSE (the
    snap-backed definition used by season_aggregate.games_played) - the
    correct denominator for an appearance-conditional share feature. Only
    an older database without the optional snap table falls back to the
    narrower box-score opportunity definition.

    Bug this fixes: carry_share/target_share/rz_*_share/air_yards_share
    previously divided a player's season totals by the team's FULL 17-game
    totals regardless of how many games the player actually played. For an
    injury-shortened season this mechanically dilutes the share by roughly
    the fraction of the season missed, even though the player commanded a
    genuinely elite share in the games they DID play - e.g. Malik Nabers'
    2025 (4 games, torn ACL): target_share against the full-season
    denominator comes out ~0.061 (looks like a WR3), but restricted to the
    4 weeks he actually played it's ~0.243 (an unambiguous alpha share).
    Sam LaPorta and Jayden Reed (both injury-shortened 2025 seasons) show
    the same pattern. `team_air_yards` (added when air_yards_share was
    built) had the identical bug, found afterward while validating this
    fix - still dividing by the team's full-season air yards total, which
    kept understating air_yards_share for exactly these injury-shortened-
    season players even after the attempt-share fix above landed. Fixed
    here in the same function/pass, not as a separate feature. A
    full-season player's active weeks ARE essentially their full season,
    so this is a no-op for the large majority of rows - it only changes
    anything for players who missed games.

    Also correctly handles in-season team changes: each week's team comes
    from that week's own weekly-usage row (via load_weekly_usage, already
    trade-aware), not the player's single season-resolved team - a
    player's denominator during weeks at their OLD team uses that team's
    attempts for those weeks, not their new team's."""
    wu = load_weekly_usage(conn)
    wu = wu[wu["season"].isin(seasons)].copy()
    appeared_col = "_appeared" if "_appeared" in wu.columns else "_active"
    if appeared_col == "_active":
        wu["_active"] = _opportunity_mask(wu)
    active = wu[wu[appeared_col]][["player_id", "season", "week", "team"]].drop_duplicates()

    team_wk = team_week_pbp_totals(conn, seasons)
    team_wk_ay = team_week_air_yards(conn, seasons)
    joined = active.merge(team_wk, on=["season", "week", "team"], how="left")
    joined = joined.merge(team_wk_ay, on=["season", "week", "team"], how="left")
    # A week where the player was active but the team has zero rows in
    # team_wk/team_wk_ay (bye-week/data artifact) contributes 0, not NaN -
    # fillna(0) here is a real "no plays that week" value, not a cover for
    # a failed join, since both are built from the same pbp table
    # load_weekly_usage ultimately derives from.
    cols = [
        "team_pass_attempts", "team_rush_attempts", "team_rz_pass_attempts", "team_rz_rush_attempts",
        "team_air_yards",
    ]
    joined[cols] = joined[cols].fillna(0)

    out = joined.groupby(["player_id", "season"])[cols].sum().reset_index()
    return out.rename(columns={c: f"{c}_active" for c in cols})


def team_season_yardage_totals(conn, seasons=SEASONS):
    """Team-season total passing yards from pbp (`passing_yards`, 0 on
    incompletions/non-pass plays, not null - no exclusion needed unlike
    air_yards/adot which are null on sacks). This single number is BOTH the
    team's total passing-yards output (the QB-side anchor) AND the team's
    total receiving-yards output (the WR/TE/RB-side anchor) - every yard
    gained on a completed pass is credited identically to `passing_yards`
    and `receiving_yards` in pbp for that same play, so there is no need to
    separately sum `receiving_yards`; they are the same quantity by
    construction. Part of the joint/multi-output team-total x player-share
    decomposition (see the plan this was built from) - `team_passing_yards`
    is the shared anchor `receiving_yards_share` (below) and any future
    QB-side reframing would both draw from."""
    q = f"""
        select season, posteam as team, passing_yards
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and posteam is not null and pass_attempt = 1
    """
    pbp = pd.read_sql(q, conn)
    return pbp.groupby(["season", "team"])["passing_yards"].sum().rename("team_passing_yards").reset_index()


def team_week_yardage_totals(conn, seasons=SEASONS):
    """Same as team_season_yardage_totals but grouped by week - the
    building block for the active-week-aware team_passing_yards_active
    denominator in receiving_yards_share (mirrors team_week_pbp_totals/
    team_week_air_yards's pattern)."""
    q = f"""
        select season, week, posteam as team, passing_yards
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and posteam is not null and pass_attempt = 1
    """
    pbp = pd.read_sql(q, conn)
    return pbp.groupby(["season", "week", "team"])["passing_yards"].sum().rename("team_passing_yards").reset_index()


def player_season_receiving_yards_share(conn, seasons=SEASONS):
    """Player-season receiving_yards_share = player's own receiving_yards /
    the team's total passing yards during every week the player appeared on
    offense.  The numerator therefore includes explicit zero-yard appearance
    weeks, matching the appearance-based games_played availability target and
    the projected_games exposure used by the live share cap. This is a LABEL
    for the team-total x player-share
    decomposition (Phase A of the joint/multi-output plan), not an input
    feature - WR_receiving_yards/TE_receiving_yards/RB_receiving_yards are
    trained to predict this share, then multiplied by a separately-trained
    team_passing_yards_pg forecast at prediction time, rather than
    predicting receiving_yards_pg directly."""
    wu = load_weekly_usage(conn)
    wu = wu[wu["season"].isin(seasons)].copy()
    appeared_col = "_appeared" if "_appeared" in wu.columns else "_active"
    if appeared_col == "_active":
        wu["_active"] = _opportunity_mask(wu)
    active = (
        wu[wu[appeared_col]][["player_id", "season", "week", "team", "receiving_yards"]]
        .groupby(["player_id", "season", "week", "team"], as_index=False, dropna=False)["receiving_yards"]
        .sum()
    )

    team_wk_yds = team_week_yardage_totals(conn, seasons)
    joined = active.merge(team_wk_yds, on=["season", "week", "team"], how="left")
    joined["team_passing_yards"] = joined["team_passing_yards"].fillna(0)

    out = joined.groupby(["player_id", "season"]).agg(
        player_receiving_yards=("receiving_yards", "sum"),
        team_passing_yards_active=("team_passing_yards", "sum"),
    ).reset_index()
    # 0/0 (a player active only in weeks the team had 0 real passing_yards -
    # vanishingly rare, but possible for e.g. a single garbage-time week) ->
    # NaN, left as NaN rather than filled: "no real team passing volume
    # existed to have a share of" is not the same as "confirmed 0 share."
    out["receiving_yards_share"] = (
        out["player_receiving_yards"]
        / out["team_passing_yards_active"].where(out["team_passing_yards_active"].ne(0))
    )
    return out[["player_id", "season", "receiving_yards_share"]]


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


def player_season_age(conn, seasons=SEASONS):
    """Player-season age, from `seasonal_rosters.age` - added to
    investigate whether injury_durability_rate's effect on next-season
    output should vary by age (the user's specific hypothesis: a young
    player like Malik Nabers should be expected to bounce back from a
    severe injury more than an older player like Christian McCaffrey).

    Investigated on real historical transitions before adding this as a
    feature (not assumed): for severe-injury player-seasons
    (injury_durability_rate > 0.35), under-27 players recovered to a median
    96% of their pre-injury per-game rate the following season vs ~80% for
    27+ - a real, meaningful gap. A formal OLS regression found age has a
    significant main effect on next-season decline on its own (p<0.001),
    but the specific age x injury INTERACTION term was NOT significant
    (p=0.35, n=1638, though the severe-injury/older-player cell is thin at
    n=79 so this could be underpowered rather than a true null). Given
    that, `age` is added here as a plain input feature rather than a
    hand-built age x injury interaction term - LightGBM captures feature
    interactions natively, so it can discover whatever real age/injury
    interaction exists in the data instead of this code assuming a
    specific (and, per the regression, not clearly supported) linear form.

    `seasonal_rosters` has multiple rows per player-season (roster
    snapshots through the season, not one row) - `max(age)` per
    (player_id, season) is used deliberately, since age only increases
    within a season (birthdays), so the max is the player's age by
    season's end, not an average across snapshots that would be
    meaningless to average. ~2-15%/season of rows have a null age
    (see Phase 0-era roster data caveats) - left as genuine NaN, not
    imputed; LightGBM handles it natively like every other FEATURE_COLS
    column with real gaps (e.g. OL quality pre-2021)."""
    q = f"""
        select player_id, season, max(age) as age from seasonal_rosters
        where season in ({','.join(map(str, seasons))})
        group by player_id, season
    """
    return pd.read_sql(q, conn)


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
    # Availability is an appearance concept, not an opportunity concept.
    # load_weekly_usage derives `_appeared` from offensive snaps (falling
    # back to opportunity only when snap data is unavailable).
    played = wu[wu["_appeared"]][["season", "week", "player_id"]].drop_duplicates()
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


def player_season_positions(conn, seasons=SEASONS):
    """Resolve the position a player actually carried in each season.

    `players.position` is a career/latest value and is wrong for historical
    hybrids and conversions (notably Devin Funchess and Taysom Hill).  Weekly
    box-score position is preferred, then that season's roster position, and
    only then the master value.  One deterministic row is returned per
    player-season so PBP numerator and denominator grouping use the same
    positional definition.
    """
    season_sql = ",".join(map(str, seasons))
    weekly = pd.read_sql(
        f"select player_id, season, position from weekly "
        f"where season in ({season_sql}) and player_id is not null", conn,
    )
    roster = pd.read_sql(
        f"select player_id, season, position from seasonal_rosters "
        f"where season in ({season_sql}) and player_id is not null", conn,
    )
    candidates = pd.concat(
        [weekly.assign(_source=0), roster.assign(_source=1)],
        ignore_index=True,
    ).dropna(subset=["position"])
    if candidates.empty:
        return pd.DataFrame(columns=["season", "player_id", "position"])
    counts = (
        candidates.groupby(["season", "player_id", "position", "_source"], as_index=False)
        .size()
        .sort_values(
            ["season", "player_id", "_source", "size", "position"],
            ascending=[True, True, True, False, True],
        )
    )
    resolved = counts.drop_duplicates(["season", "player_id"])[
        ["season", "player_id", "position"]
    ]
    return resolved


def team_season_rz_position_totals(conn, seasons=SEASONS):
    """Team-season red-zone carries/targets, totaled by the CARRIER/RECEIVER's
    own position group (not the whole offense), used as the denominator for a
    red-zone "monopoly" feature (see `add_rz_monopoly_features` in
    features.py).

    Why this is a different number than the existing `team_rz_pass_attempts`/
    `team_rz_rush_attempts` (`team_season_pbp_totals`): those totals are
    denominators for "share of ALL the team's red-zone plays," which mixes a
    WR's red-zone targets in with RB red-zone targets, etc. A player who gets
    every single red-zone target their team throws to a wide receiver (the
    "clear #1 red-zone option at the position") and a player who gets a
    similar target COUNT but on a team that spreads red-zone work heavily to
    the backfield can have very similar `rz_target_share` while being in
    completely different competitive situations. Dividing by only the looks
    that went to the SAME position group isolates "concentration among your
    positional peers," which is what actually distinguishes a true red-zone
    monopoly from a crowded red-zone offense.

    Position is looked up directly from `players.position` (not restricted to
    this project's QB/RB/WR/TE `POSITIONS` list) so the totals reflect the
    real full pool of red-zone touches at each position (e.g. FB carries
    still count toward the RB-adjacent pool correctly get their own bucket,
    rather than being silently dropped from the RB total or double-counted
    into it)."""
    q = f"""
        select season, posteam as team, rush_attempt, pass_attempt,
               rusher_player_id, receiver_player_id
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and posteam is not null and yardline_100 <= 20
          and (pass_attempt = 1 or rush_attempt = 1)
    """
    pbp = pd.read_sql(q, conn)
    players_pos = player_season_positions(conn, seasons)

    rushes = pbp[pbp["rush_attempt"] == 1][["season", "team", "rusher_player_id"]].rename(
        columns={"rusher_player_id": "player_id"}
    )
    rushes = rushes.merge(players_pos, on=["season", "player_id"], how="left")
    rz_carries_pos = (
        rushes.groupby(["season", "team", "position"]).size().rename("team_rz_carries_pos").reset_index()
    )

    targets = pbp[pbp["pass_attempt"] == 1][["season", "team", "receiver_player_id"]].rename(
        columns={"receiver_player_id": "player_id"}
    )
    targets = targets.merge(players_pos, on=["season", "player_id"], how="left")
    rz_targets_pos = (
        targets.groupby(["season", "team", "position"]).size().rename("team_rz_targets_pos").reset_index()
    )

    out = rz_carries_pos.merge(rz_targets_pos, on=["season", "team", "position"], how="outer")
    out[["team_rz_carries_pos", "team_rz_targets_pos"]] = out[["team_rz_carries_pos", "team_rz_targets_pos"]].fillna(0)
    return out


def team_week_rz_position_totals(conn, seasons=SEASONS):
    """Week-grain sibling of team_season_rz_position_totals: red-zone
    carries/targets per (season, week, team, carrier/receiver position
    group). Exists solely as the building block for
    player_active_rz_position_opportunity below - the active-weeks
    denominator fix for the monopoly features (Phase 3 of the
    consensus-gap work)."""
    q = f"""
        select season, week, posteam as team, rush_attempt, pass_attempt,
               rusher_player_id, receiver_player_id
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and posteam is not null and yardline_100 <= 20
          and (pass_attempt = 1 or rush_attempt = 1)
    """
    pbp = pd.read_sql(q, conn)
    players_pos = player_season_positions(conn, seasons)

    rushes = pbp[pbp["rush_attempt"] == 1][["season", "week", "team", "rusher_player_id"]].rename(
        columns={"rusher_player_id": "player_id"}
    )
    rushes = rushes.merge(players_pos, on=["season", "player_id"], how="left")
    rz_carries_pos = (
        rushes.groupby(["season", "week", "team", "position"]).size().rename("team_rz_carries_pos").reset_index()
    )

    targets = pbp[pbp["pass_attempt"] == 1][["season", "week", "team", "receiver_player_id"]].rename(
        columns={"receiver_player_id": "player_id"}
    )
    targets = targets.merge(players_pos, on=["season", "player_id"], how="left")
    rz_targets_pos = (
        targets.groupby(["season", "week", "team", "position"]).size().rename("team_rz_targets_pos").reset_index()
    )

    out = rz_carries_pos.merge(rz_targets_pos, on=["season", "week", "team", "position"], how="outer")
    out[["team_rz_carries_pos", "team_rz_targets_pos"]] = out[["team_rz_carries_pos", "team_rz_targets_pos"]].fillna(0)
    return out


def player_active_rz_position_opportunity(conn, seasons=SEASONS):
    """Player-season position-group red-zone carry/target totals restricted
    to the WEEKS THIS PLAYER APPEARED ON OFFENSE - the position-keyed
    sibling of player_active_team_opportunity, and the Phase-3 fix for the
    last denominator the original active-weeks pass missed.

    Bug this fixes: rz_carry_monopoly/rz_target_monopoly divided a
    player's season red-zone touch counts by the team position group's
    FULL-SEASON totals regardless of games missed. For an injury-shortened
    season this mechanically dilutes the monopoly toward zero even when
    the player owned the position group's red-zone looks every week they
    played - Malik Nabers' 2025 (4 games): 4 of NYG's 36 full-season WR
    red-zone targets reads as monopoly 0.111 (bench-level) when the
    active-weeks truth is 4 of the 9 thrown while he was on the field
    (0.444, an unambiguous alpha number). Sensitivity analysis found this
    single diluted feature was the LARGEST driver of his under-projection
    (+45% predicted share if corrected) - larger than the injury features
    themselves. A separate function rather than more columns on
    player_active_team_opportunity because the join is position-keyed
    (each player's denominator is their OWN position group's weekly
    totals), not team-wide.

    Same trade-aware weekly-team handling as player_active_team_opportunity
    (each active week's team comes from that week's own usage row). The
    accepted trade, stated: a 4-active-week denominator is noisier than a
    17-week one (4/9 vs 4/36) - correct-but-noisy beats wrong-but-stable,
    and the games-weighted feature blending planned as Phase 4 is the
    stabilizer for exactly this."""
    wu = load_weekly_usage(conn)
    wu = wu[wu["season"].isin(seasons)].copy()
    appeared_col = "_appeared" if "_appeared" in wu.columns else "_active"
    if appeared_col == "_active":
        wu["_active"] = _opportunity_mask(wu)
    active = wu[wu[appeared_col]][
        ["player_id", "season", "week", "team", "position"]
    ].drop_duplicates()

    team_wk_rz_pos = team_week_rz_position_totals(conn, seasons)
    joined = active.merge(team_wk_rz_pos, on=["season", "week", "team", "position"], how="left")
    # An active week with no red-zone plays for this position group is a
    # real 0 (nothing to be a monopoly over that week), not a failed join -
    # both sides derive from the same pbp table.
    cols = ["team_rz_carries_pos", "team_rz_targets_pos"]
    joined[cols] = joined[cols].fillna(0)

    out = joined.groupby(["player_id", "season"])[cols].sum().reset_index()
    return out.rename(columns={c: f"{c}_active" for c in cols})


def player_season_air_yards(conn, seasons=SEASONS):
    """Player-season receiving air-yards totals + aDOT (average depth of
    target), and the matching team-season air-yards total, from `pbp.
    air_yards`.

    Schema check done first (not assumed): `air_yards` is populated on every
    pass-attempt row except sacks (~7% of pass attempts per season, ball
    never thrown - a real, expected gap, not a data-quality problem) and is
    always null on rush-attempt rows (no target exists) - both are excluded
    here by construction (`pass_attempt = 1 and air_yards is not null`)
    rather than silently coerced to 0, since "sacked, no target thrown" and
    "targeted at the goal line, air_yards=0" are not the same thing and
    conflating them would understate aDOT for high-sack QBs/teams (not that
    aDOT is computed for QBs here, but the same team-total denominator IS
    shared, so the exclusion still matters for `team_air_yards`).

    `player_adot`: mean air_yards across the player's own targets with a
    real (non-null) air_yards value - a checkdown/possession receiver has a
    low aDOT even with a healthy target COUNT, which is exactly the "true #1
    receiving option vs. possession/checkdown role" distinction the spec
    asked for; `target_share`/`rz_target_share` alone can't separate those
    two roles."""
    q = f"""
        select season, posteam as team, receiver_player_id, air_yards
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and posteam is not null and pass_attempt = 1 and air_yards is not null
    """
    pbp = pd.read_sql(q, conn)

    player = (
        pbp.groupby(["season", "receiver_player_id"])["air_yards"]
        .agg(player_air_yards="sum", player_adot="mean")
        .reset_index()
        .rename(columns={"receiver_player_id": "player_id"})
    )
    team = (
        pbp.groupby(["season", "team"])["air_yards"].sum().rename("team_air_yards").reset_index()
    )
    return player, team


def player_season_designed_rushes(conn, seasons=SEASONS):
    """Player-season DESIGNED rush attempts (excludes QB scrambles, per
    `pbp.qb_scramble`) - a distinct signal from total `rushing_yards`/
    `carry_share`, which don't separate "the offense calls this player's
    number to run the ball" from "the QB took off after the pass play broke
    down." For a QB this is the closer proxy for actual designed run-game
    usage (RPO keepers, called QB runs, read-option) as opposed to scramble
    production, which is much less a stable, schemed-for signal.

    Schema check done first: `qb_scramble` is populated (0/1, never null) on
    every rush-attempt row and is 0 for effectively all non-QB rushers
    (verified directly: 2024 REG season has exactly 2 non-QB rows flagged
    `qb_scramble=1`, both WR, presumably trick-play/broken-play
    mislabeling upstream in nflverse's own charting - not corrected here,
    left as nflverse ships it). Because of this, `designed_rush_attempts`
    for a non-QB is effectively just their normal carry count, and this
    feature is highly collinear with `carry_share` for RB/WR/TE rows - it is
    included generically (not gated to QB rows only) because
    `build_player_season_features` doesn't filter `FEATURE_COLS` by
    position elsewhere either (see `carry_share`/TARGET_STATS handling), but
    it is really a QB-focused signal and should be read that way."""
    q = f"""
        select season, rusher_player_id as player_id, count(*) as designed_rush_attempts
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and rush_attempt = 1 and qb_scramble = 0 and rusher_player_id is not null
        group by season, rusher_player_id
    """
    return pd.read_sql(q, conn)


def team_season_defense_epa(conn, seasons=SEASONS):
    """Team-season defensive efficiency: mean EPA/play ALLOWED (as `defteam`),
    split pass vs. rush, from `pbp.epa`. Lower (more negative) = stronger
    defense (bad plays for the opposing offense). This is the raw ingredient
    for the opponent/schedule-strength feature below - kept as its own
    function since it's indexed by the team's OWN season (as a defense), not
    by who they played."""
    q = f"""
        select season, defteam as team, pass_attempt, rush_attempt, epa
        from pbp
        where season in ({','.join(map(str, seasons))}) and season_type = 'REG'
          and defteam is not null and epa is not null
          and (pass_attempt = 1 or rush_attempt = 1)
    """
    pbp = pd.read_sql(q, conn)
    pass_epa = (
        pbp[pbp["pass_attempt"] == 1].groupby(["season", "team"])["epa"].mean().rename("def_pass_epa_allowed").reset_index()
    )
    rush_epa = (
        pbp[pbp["rush_attempt"] == 1].groupby(["season", "team"])["epa"].mean().rename("def_rush_epa_allowed").reset_index()
    )
    return pass_epa.merge(rush_epa, on=["season", "team"], how="outer")


def team_season_opponent_strength(conn, seasons=SEASONS):
    """Team-season opponent/schedule-strength proxy: for team T in season S,
    the average of T's actual opponents' PASS/RUSH defensive EPA/play
    allowed in season S-1 (the most recent observed defensive performance
    for each opponent at the time season S kicked off - S-1's own defense is
    the only "known" quality signal a real preseason schedule-strength
    read could have used, since S's defensive stats don't exist yet at the
    start of S).

    Built from `schedules` (REG season only) for the actual game-by-game
    opponent list - NOT derived from `pbp.defteam` directly, since the
    `schedules` table is the authoritative, one-row-per-game source and
    avoids any risk of miscounting a team's own bye-week/multi-posteam-per-
    game pbp artifacts.

    This targets the QB volume over-projection (all QBs run ~18% hot on
    attempts/passing_yards per the Sleeper comparison, per this task's
    brief) - a team whose PRIOR-year opponents happened to have weak pass
    defenses will show inflated observed attempts/completion-driven yardage
    in the season being used as the feature row, and the model has had no
    way to discount that until now.

    NaN handling, stated plainly: season S=2016 (this project's earliest
    season) has no season S-1=2015 in this DB, so `opp_def_pass_epa_prior`/
    `opp_def_rush_epa_prior` are genuinely NaN for every 2016 team-season -
    not imputed, not backfilled with a league-average. This does not affect
    training, since `train.py`/`transitions.py` only use `season_from` in
    [2021, 2022, 2023, 2024], all of which have a real season-1 (2020-2023)
    present in this DB's 2016-2025 window."""
    sched = pd.read_sql(
        f"select season, home_team, away_team from schedules "
        f"where season in ({','.join(map(str, seasons))}) and game_type = 'REG'", conn,
    )
    # `schedules` uses each franchise's HISTORICAL abbreviation for the
    # season it was actually played under (OAK through 2019, SD through
    # 2016), while `pbp.posteam`/`defteam` (which is what `base`'s `team`
    # column and `team_season_defense_epa` are both built from) ship
    # relocated franchises' CURRENT code retroactively for every season
    # (LV, LAC). Checked directly: these are the only two mismatches in this
    # DB's 2016-2025 window (the Rams' STL->LA move predates 2016, already
    # consistent as "LA" in both tables). Left unnormalized, every 2016-2019
    # Raiders/2016 Chargers team-season would silently fail this merge and
    # come back NaN - normalizing here instead of silently dropping/NaN-ing
    # those rows, per this project's "never silently fill or drop on a
    # failed join - report it" rule (reported in this function's own
    # docstring history / the task report, not just fixed invisibly).
    _FRANCHISE_CODE_FIX = {"OAK": "LV", "SD": "LAC"}
    sched["home_team"] = sched["home_team"].replace(_FRANCHISE_CODE_FIX)
    sched["away_team"] = sched["away_team"].replace(_FRANCHISE_CODE_FIX)
    home = sched.rename(columns={"home_team": "team", "away_team": "opponent"})[["season", "team", "opponent"]]
    away = sched.rename(columns={"away_team": "team", "home_team": "opponent"})[["season", "team", "opponent"]]
    schedule_long = pd.concat([home, away], ignore_index=True)

    def_epa = team_season_defense_epa(conn, seasons)
    # opponent's PRIOR-season defense: shift the join season forward by 1,
    # i.e. team T's season-S opponent list joins to def_epa's season S-1 row
    # for each opponent.
    def_epa_prior = def_epa.rename(columns={"team": "opponent"})
    def_epa_prior = def_epa_prior.assign(season=def_epa_prior["season"] + 1)

    merged = schedule_long.merge(def_epa_prior, on=["season", "opponent"], how="left")
    out = (
        merged.groupby(["season", "team"])[["def_pass_epa_allowed", "def_rush_epa_allowed"]]
        .mean()
        .reset_index()
        .rename(columns={
            "def_pass_epa_allowed": "opp_def_pass_epa_prior",
            "def_rush_epa_allowed": "opp_def_rush_epa_prior",
        })
    )
    return out
