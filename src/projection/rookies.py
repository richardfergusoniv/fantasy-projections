"""Rookie-season projection path - deliberately separate from the veteran
LightGBM pipeline in train.py, per the hard project rule: a rookie has no
prior-NFL-season trailing features (the entire premise of the veteran
model), and must not use same-season NFL stats or college production as
inputs (none of which exist in this DB anyway - college production is
simply not modeled here).

Allowed inputs only:
- draft capital (round/pick from `draft_picks`)
- "vacated team opportunity": the target/carry share the rookie's new team
  lost from players who were on that team in season N-1 but are NOT active
  (no games with usage) for that same team in season N. This uses only
  season N-1 data plus who's on/off the roster in season N - never the
  rookie's or anyone's season-N production - so it doesn't leak forward
  information a real preseason projection wouldn't have.

Model: rule-based, not LightGBM. Rookie sample sizes per position x
draft-round-bucket are too small (see PHASE4_REPORT.md) for a tree model
to learn anything but noise, and a rookie's feature vector is structurally
incompatible with the veteran model's inputs anyway - this is exactly the
"distinct path" the spec calls for, not a shrunken version of the same
pipeline. Historical per-game rates for rookies in the same
position/draft-round bucket are averaged, then scaled by the ratio of this
player's team's vacated opportunity to the bucket's historical average
vacated opportunity (clipped to avoid small-sample blowups).
"""
import numpy as np
import pandas as pd

from src.projection.data_prep import SEASONS, load_weekly_usage, season_aggregate
from src.projection.features import TARGET_STATS
from src.comparison.sleeper_compare import fetch_sleeper_play_probability, _normalize_name

# Sleeper doesn't even bother filing a season projection (no `gp` field at
# all) for a large share of deep-roster players - itself a strong signal
# of near-zero relevance, not a "no data" case to shrug off as prob=1.0.
NO_SLEEPER_MATCH_PLAY_PROB = 0.05

ROUND_BUCKETS = {1: "round_1", 2: "round_2_3", 3: "round_2_3", 4: "round_4_7", 5: "round_4_7",
                  6: "round_4_7", 7: "round_4_7"}
VACATED_CLIP = (0.3, 2.5)
ROOKIE_INTERVAL_QUANTILES = (0.10, 0.90)  # same width as the veteran empirical interval, for comparability
ROOKIE_INTERVAL_MIN_N = 20  # bucket sample sizes below this get interval_low_n_flag=True

# draft_picks.team ships PFR-style abbreviations, not nflverse's standard
# ones used everywhere else in this project (weekly/pbp/seasonal_rosters/
# oc_tendency_profiles) - found while validating identify_target_season_rookie_class's
# output: every 2026 draft pick's team fell back to draft_picks.team (their
# placeholder id isn't in seasonal_rosters at all - see that function's
# docstring), and 100% of those came through as GNB/KAN/LAR/LVR/NOR/NWE/SFO/TAM
# instead of GB/KC/LA/LV/NO/NE/SF/TB. Left unfixed, this both mislabels
# rookie rows in the final output AND corrupts team_vacated_opportunity's
# roster-fallback join (a Rams rookie tagged "LAR" would never match the
# team's actual vacated-share row, keyed "LA").
TEAM_ABBR_FIX = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "LVR": "LV",
    "NOR": "NO", "NWE": "NE", "SFO": "SF", "TAM": "TB",
}


def _round_bucket(rnd):
    if pd.isna(rnd):
        return "undrafted"
    return ROUND_BUCKETS.get(int(rnd), "round_4_7")


def load_draft_capital(conn):
    dp = pd.read_sql(
        "select season as draft_season, gsis_id as player_id, round, pick, position "
        "from draft_picks where gsis_id is not null and position in ('QB','RB','WR','TE')", conn,
    )
    dp["round_bucket"] = dp["round"].apply(_round_bucket)
    return dp


def identify_rookie_seasons(conn, seasons=SEASONS):
    """(player_id, season) pairs where `season` is the player's first
    season with any active week in `weekly` (2016-2025 window) AND matches
    their draft season - i.e. no prior-NFL-season row exists at all."""
    wu = load_weekly_usage(conn)
    agg = season_aggregate(wu)
    active = agg[agg["games_played"] > 0]
    first_season = active.groupby("player_id")["season"].min().rename("first_active_season").reset_index()

    draft = load_draft_capital(conn)
    rookies = draft.merge(first_season, on="player_id", how="inner")
    rookies = rookies[rookies["draft_season"] == rookies["first_active_season"]]
    rookies = rookies[rookies["draft_season"].isin(seasons)]
    rookies = rookies[["player_id", "draft_season", "round", "pick", "round_bucket", "position"]].rename(
        columns={"draft_season": "season"}
    )
    rookies["rookie_tier"] = "drafted"
    return rookies


def identify_udfa_rookie_seasons(conn, seasons=SEASONS):
    """UDFA counterpart to identify_rookie_seasons: (player_id, season) pairs
    for players with NO draft round (`players.draft_round` is null) whose
    first active `weekly` season equals `players.rookie_season`.

    Deliberately NOT implemented as "absent from draft_picks in this
    season" - draft_picks only covers 2016+, so a veteran actually drafted
    before 2016 (data left-censored) would look falsely "undrafted" the
    moment their first active week in OUR window happens to land in 2016
    (spot-checked: 484 players' first-active-season lands in 2016 under
    that naive definition, nearly all of them established veterans per
    players.rookie_season/draft_year predating 2016, e.g. Tom Brady,
    Drew Brees - not real UDFA rookies). `players.rookie_season` and
    `draft_round` are nflverse master-roster fields, not windowed by this
    project's ingestion range, so they don't have this problem."""
    wu = load_weekly_usage(conn)
    agg = season_aggregate(wu)
    active = agg[agg["games_played"] > 0]
    first_season = active.groupby("player_id")["season"].min().rename("first_active_season").reset_index()

    players = pd.read_sql(
        "select gsis_id as player_id, rookie_season, draft_round from players where draft_round is null", conn,
    )
    udfa = players.merge(first_season, on="player_id", how="inner")
    udfa = udfa[udfa["rookie_season"] == udfa["first_active_season"]]
    udfa = udfa[udfa["first_active_season"].isin(seasons)]

    pos_lookup = active[["player_id", "season", "position"]].drop_duplicates(subset=["player_id", "season"])
    udfa = udfa.merge(
        pos_lookup, left_on=["player_id", "first_active_season"], right_on=["player_id", "season"], how="inner",
    )
    udfa = udfa[udfa["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    udfa["round"] = np.nan
    udfa["pick"] = np.nan
    udfa["round_bucket"] = "undrafted"
    udfa["rookie_tier"] = "udfa"
    return udfa[["player_id", "season", "round", "pick", "round_bucket", "position", "rookie_tier"]]


def team_vacated_opportunity(conn, seasons=SEASONS):
    """(season, team) -> vacated_carry_share, vacated_target_share: the
    fraction of the team's season-(N-1) carries/targets that belonged to
    players who did NOT have an active week for that same team in season N
    (retired, cut, signed elsewhere, or simply lost their role - this
    doesn't distinguish why, only that the volume is no longer theirs).

    Roster-fallback for a genuinely future `season` with zero played games
    (e.g. projecting a season that hasn't started): "active in season N" is
    structurally undeterminable from `weekly` there, so this falls back to
    `seasonal_rosters[season]` (available pre-season, unlike game logs) to
    determine which season-(N-1) players are still on the same roster. This
    is a weaker signal than confirmed game participation (a rostered player
    could still be cut/inactive all year), but it's the best pre-season
    proxy available - the alternative would be reporting zero vacated
    opportunity for every rookie in the target season, which is worse."""
    wu = load_weekly_usage(conn)
    agg = season_aggregate(wu)
    agg = agg[agg["games_played"] > 0]
    seasons_with_games = set(agg["season"].unique())

    roster_fallback_seasons = [s for s in seasons if s not in seasons_with_games]
    roster_team = pd.DataFrame(columns=["player_id", "season", "team"])
    if roster_fallback_seasons:
        q = (
            "select player_id, season, team from seasonal_rosters where season in "
            f"({','.join(map(str, roster_fallback_seasons))})"
        )
        roster_team = pd.read_sql(q, conn).drop_duplicates(subset=["player_id", "season"])

    rows = []
    for season in seasons:
        prev = agg[agg["season"] == season - 1]
        if prev.empty:
            continue
        if season in seasons_with_games:
            curr_team_of_player = agg[agg["season"] == season].set_index("player_id")["team"]
        else:
            curr_team_of_player = roster_team[roster_team["season"] == season].set_index("player_id")["team"]
        prev = prev.copy()
        prev["returning_same_team"] = prev["player_id"].map(curr_team_of_player) == prev["team"]

        g = prev.groupby("team")
        team_totals = g[["carries", "targets", "attempts"]].sum().rename(
            columns={"carries": "prev_team_carries", "targets": "prev_team_targets", "attempts": "prev_team_attempts"}
        )
        returning = prev[prev["returning_same_team"]].groupby("team")[["carries", "targets", "attempts"]].sum().rename(
            columns={"carries": "returning_carries", "targets": "returning_targets", "attempts": "returning_attempts"}
        )
        merged = team_totals.join(returning, how="left").fillna(0)
        merged["vacated_carry_share"] = 1 - merged["returning_carries"] / merged["prev_team_carries"].replace(0, np.nan)
        merged["vacated_target_share"] = 1 - merged["returning_targets"] / merged["prev_team_targets"].replace(0, np.nan)
        # QB-specific proxy - see predict_rookies' docstring for the bug this
        # fixes: vacated_target_share reflects WR/TE/RB receiving-corps
        # turnover, which has nothing to do with whether a backup/rookie QB
        # gets snaps. vacated_attempts_share instead measures how much of
        # the team's PASSING volume belonged to QBs no longer on the roster
        # (attempts is ~exclusively a QB stat, so this isolates QB-room
        # turnover specifically) - the right signal for "is the starting job
        # actually open," not receiver churn.
        merged["vacated_attempts_share"] = 1 - merged["returning_attempts"] / merged["prev_team_attempts"].replace(0, np.nan)
        merged["season"] = season
        rows.append(merged.reset_index()[
            ["season", "team", "vacated_carry_share", "vacated_target_share", "vacated_attempts_share"]
        ])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["season", "team", "vacated_carry_share", "vacated_target_share", "vacated_attempts_share"]
    )


def build_rookie_dataset(conn, feature_table, seasons=SEASONS):
    """Rookie player-seasons with draft capital + vacated opportunity +
    actual per-game rates (for fitting the historical bucket averages /
    for backtest evaluation) - NOT the veteran feature columns.

    Includes both drafted rookies and UDFA rookies (identify_udfa_rookie_seasons)
    in one combined table so fit_rookie_baselines' groupby('round_bucket')
    naturally produces an 'undrafted' bucket populated by real UDFA per-game
    rates, not the empty bucket PHASE4_REPORT.md flagged (drafted rows with
    round_bucket='undrafted' essentially never occur - that requires a
    draft_picks row with a null round, which doesn't happen in practice)."""
    rookies = identify_rookie_seasons(conn, seasons)
    udfa = identify_udfa_rookie_seasons(conn, seasons)
    rookies = pd.concat([rookies, udfa], ignore_index=True, sort=False)
    vacated = team_vacated_opportunity(conn, seasons)

    stat_cols = sorted({s for stats in TARGET_STATS.values() for s in stats})
    pg_cols = [f"{s}_pg" for s in stat_cols]
    actuals = feature_table[["player_id", "season", "team", "games_played"] + pg_cols]

    df = rookies.merge(actuals, on=["player_id", "season"], how="inner")
    df = df.merge(vacated, on=["season", "team"], how="left")
    return df


def identify_target_season_rookie_class(conn, target_season):
    """Drafted + UDFA rookie class for `target_season`, built WITHOUT
    requiring any played game in that season - unlike
    identify_rookie_seasons/identify_udfa_rookie_seasons (both require
    confirmed active-week production and are only appropriate for fitting
    HISTORICAL baselines). A genuinely future target_season has no `weekly`
    rows at all, so "first active season" can't be computed - the target
    season's rookie class has to be read directly off draft_picks (drafted)
    and seasonal_rosters (UDFA: years_exp==0, draft_number null - both
    fields already populated pre-season) instead.

    Team is resolved from `seasonal_rosters[target_season]` (the most
    current post-draft/free-agency snapshot) with a fallback to
    draft_picks' own `team` column for a drafted player not found there
    (e.g. a crosswalk/ids gap)."""
    drafted = pd.read_sql(
        f"select gsis_id as player_id, round, pick, position, team as draft_team, pfr_player_name as name "
        f"from draft_picks where season = {target_season} and gsis_id is not null "
        f"and position in ('QB','RB','WR','TE')", conn,
    )
    drafted["draft_team"] = drafted["draft_team"].replace(TEAM_ABBR_FIX)
    drafted["round_bucket"] = drafted["round"].apply(_round_bucket)
    drafted["rookie_tier"] = "drafted"

    roster = pd.read_sql(
        f"select player_id, team, position, years_exp, draft_number, player_name as name from seasonal_rosters "
        f"where season = {target_season}", conn,
    ).drop_duplicates(subset=["player_id"])

    udfa = roster[
        (roster["years_exp"] == 0) & (roster["draft_number"].isna())
        & (roster["position"].isin(["QB", "RB", "WR", "TE"]))
        & (~roster["player_id"].isin(set(drafted["player_id"])))
    ].copy()
    udfa["round"] = np.nan
    udfa["pick"] = np.nan
    udfa["round_bucket"] = "undrafted"
    udfa["rookie_tier"] = "udfa"

    team_map = roster.set_index("player_id")["team"]
    drafted["team"] = drafted["player_id"].map(team_map).fillna(drafted["draft_team"])
    # name is needed for predict_rookies' Sleeper play-probability lookup
    # (QB only) - a drafted player's own gsis_id, if it's one of the
    # current-draft-class placeholder ids Phase 5 found (not a real gsis_id
    # yet), won't match anything in Sleeper's data by id, so the name is
    # the only usable join key for this year's draft class.
    name_map = roster.set_index("player_id")["name"]
    drafted["name"] = drafted["player_id"].map(name_map).fillna(drafted["name"])

    cols = ["player_id", "team", "round", "pick", "position", "round_bucket", "rookie_tier", "name"]
    combined = pd.concat([drafted[cols], udfa[cols]], ignore_index=True)
    combined["season"] = target_season
    return combined


def fit_rookie_baselines(rookie_df, train_seasons):
    """Historical (position, round_bucket) -> mean per-game rate + mean
    vacated_carry/target_share, fit ONLY on train_seasons (so the backtest
    holdout season's own rookies never inform their own baseline)."""
    train = rookie_df[rookie_df["season"].isin(train_seasons) & (rookie_df["games_played"] > 0)]
    pg_cols = [c for c in rookie_df.columns if c.endswith("_pg")]
    vacated_cols = ["vacated_carry_share", "vacated_target_share", "vacated_attempts_share"]
    baselines = train.groupby(["position", "round_bucket"])[pg_cols + vacated_cols].mean()
    counts = train.groupby(["position", "round_bucket"]).size().rename("n_train_rookies")
    return baselines.join(counts)


def predict_rookies(rookie_df, baselines, target_seasons, depth_chart=None):
    """Rule-based projection: bucket mean per-game rate, scaled by this
    player's team's vacated opportunity vs. the bucket's historical average
    vacated opportunity (RB: vacated carry share; WR/TE: vacated target
    share; QB: vacated ATTEMPTS share - see team_vacated_opportunity's
    docstring for why this must be QB-specific and not the receiving-corps
    target share used for WR/TE. Bug found and fixed here: QB rookies used
    to be scaled by vacated_target_share too, which measures WR/TE/RB
    turnover, not QB-room turnover - a team that lost a lot of receivers
    (nothing to do with the QB depth chart) would inflate a buried
    7th-round/UDFA QB's projection toward starter volume purely because of
    that unrelated churn. Verified: Athan Kaliakmanis (WAS, round_4_7,
    clearly a long-shot) was projected ~28 attempts/game before this fix -
    driven by WAS's 0.52 vacated_target_share (real WR/TE turnover) versus
    the round_4_7 QB bucket's historical average, with nothing capping a QB
    scale factor derived from a receiving-corps signal.
    Falls back to the unscaled bucket mean if the bucket has no historical
    rows or the vacated feature is null (e.g. an expansion-style edge case).

    pg_cols is read off `baselines.columns`, not `rookie_df.columns` - the
    target rows (a genuinely future season) have no actual per-game rates
    to speak of, so rookie_df itself may not carry any `*_pg` columns; the
    baselines (fit on historical data) always do.

    `depth_chart` (optional, Phase 6's src/depth_chart/starters_2026.csv -
    empty/None for any other season): caps the UPWARD half of the vacancy
    scale (>1.0) to a rookie the curated table actually lists for their
    (team, position) - same principle predict.py's reassign_team_changers
    already applies to team-changing veterans, applied here to fix its
    rookie-path sibling. Bug this fixes: a rookie/UDFA with zero real
    chance of playing could still get boosted toward starter volume purely
    because their team's ACTUAL new starter opened up a big vacancy (e.g. a
    random UDFA QB scaled toward 45 attempts/game because the team's
    veteran QB left for another team) - the vacancy is real, but nothing
    checked whether THIS specific long-shot player is who's stepping into
    it. Downward scaling (a below-average opportunity) still always
    applies - a below-average situation should reduce even an unconfirmed
    player's already-modest bucket-mean projection.
    """
    pg_cols = [c for c in baselines.columns if c.endswith("_pg")]
    target = rookie_df[rookie_df["season"].isin(target_seasons)].copy()
    curated_players = set()
    if depth_chart is not None and not depth_chart.empty:
        curated_players = set(zip(depth_chart["gsis_id"], depth_chart["position"]))

    # QB-only survivorship-bias correction (see module docstring / caller):
    # the historical QB bucket mean is itself computed only over rookie
    # QB-seasons with real snaps (games_played > 0), which for a position
    # where most backups NEVER play is a biased sample of "the ones who
    # got lucky," not "the typical camp arm." Rather than build our own
    # probability-of-playing estimator from scratch, this borrows Sleeper's
    # own projected games-played (fetch_sleeper_play_probability) as that
    # signal - it already reflects real depth-chart/beat-reporter judgment
    # this project has no other free source for. Only applied if `rookie_df`
    # carries a `name` column (identify_target_season_rookie_class adds one;
    # the historical identify_rookie_seasons path used for backtest.py does
    # not, so backtest correctness is unaffected - this is strictly a
    # target-season prediction-quality fix, not a training-time change).
    play_prob_by_id, play_prob_by_name = {}, {}
    has_names = "name" in target.columns and target["position"].eq("QB").any()
    if has_names:
        try:
            for season in target["season"].unique():
                pp = fetch_sleeper_play_probability(int(season))
                pp_qb = pp[pp["position"] == "QB"]
                play_prob_by_id.update(pp_qb.dropna(subset=["player_id"]).set_index("player_id")["play_prob"].to_dict())
                play_prob_by_name.update(
                    pp_qb.dropna(subset=["name_key"]).set_index("name_key")["play_prob"].to_dict()
                )
        except Exception as e:
            print(f"WARNING: could not fetch Sleeper play-probability data ({e}) - "
                  f"QB rookie projections will NOT get the survivorship-bias correction this run.")

    def _qb_play_prob(row):
        if row["player_id"] in play_prob_by_id:
            return play_prob_by_id[row["player_id"]]
        name_key = _normalize_name(row.get("name"))
        if name_key in play_prob_by_name:
            return play_prob_by_name[name_key]
        return NO_SLEEPER_MATCH_PLAY_PROB

    def project_row(row):
        key = (row["position"], row["round_bucket"])
        if key not in baselines.index:
            return pd.Series({c: np.nan for c in pg_cols} | {"low_confidence": True, "baseline_n": 0})
        b = baselines.loc[key]
        if row["position"] == "RB":
            vac_col = "vacated_carry_share"
        elif row["position"] == "QB":
            vac_col = "vacated_attempts_share"
        else:
            vac_col = "vacated_target_share"
        player_vac, hist_vac = row.get(vac_col), b[vac_col]
        if pd.isna(player_vac) or pd.isna(hist_vac) or hist_vac == 0:
            scale = 1.0
        else:
            scale = np.clip(player_vac / hist_vac, *VACATED_CLIP)
            if (row["player_id"], row["position"]) not in curated_players:
                scale = min(scale, 1.0)
        preds = {c: b[c] * scale for c in pg_cols}

        play_prob = np.nan
        if row["position"] == "QB" and has_names:
            play_prob = _qb_play_prob(row)
            preds = {c: v * play_prob for c, v in preds.items()}
        preds["qb_sleeper_play_prob"] = play_prob

        preds["low_confidence"] = True
        preds["baseline_n"] = b["n_train_rookies"]
        return pd.Series(preds)

    preds = target.apply(project_row, axis=1)
    out = pd.concat(
        [target[["player_id", "season", "team", "position", "round_bucket", "pick", "rookie_tier"]], preds], axis=1
    )
    return out


def rookie_interval_ratios(rookie_df, baselines, train_seasons, quantiles=ROOKIE_INTERVAL_QUANTILES):
    """Rookie prediction intervals: no naive-baseline backtest comparison
    exists for rookies (no prior season to carry forward as a baseline -
    see PHASE4_REPORT.md), so veteran-style additive residuals from a
    held-out backtest aren't available either. Fallback used instead: for
    each (position, round_bucket, stat), the empirical p10/p90 of
    actual_pg / bucket_mean_pg across all historical rookies in that bucket
    (train_seasons only, same rows fit_rookie_baselines used). Applied
    MULTIPLICATIVELY to a given player's point prediction
    (pred_pg_low = pred_pg * ratio_low) rather than additively, because the
    point prediction itself is bucket_mean * a vacated-opportunity scale
    factor - a multiplicative ratio stays consistent with that scaling
    logic instead of bolting on a flat additive band that ignores how much
    a specific player's prediction was scaled up/down.

    Sample sizes here are the bucket sizes already reported in
    PHASE4_REPORT.md (11-132) - smaller than the veteran backtest's 61-170,
    so buckets below ROOKIE_INTERVAL_MIN_N are flagged via
    interval_low_n_flag rather than silently presented at equal
    confidence to a well-sampled bucket."""
    train = rookie_df[rookie_df["season"].isin(train_seasons) & (rookie_df["games_played"] > 0)]
    pg_cols = [c for c in rookie_df.columns if c.endswith("_pg")]
    stat_names = [c[:-3] for c in pg_cols]

    rows = []
    for (position, bucket), grp in train.groupby(["position", "round_bucket"]):
        n = len(grp)
        if (position, bucket) not in baselines.index:
            continue
        b = baselines.loc[(position, bucket)]
        for stat, pg_col in zip(stat_names, pg_cols):
            mean = b[pg_col]
            vals = grp[pg_col].dropna()
            if pd.isna(mean) or mean == 0 or len(vals) < 3:
                continue
            ratio = vals / mean
            lo, hi = np.quantile(ratio, quantiles)
            rows.append({
                "position": position, "round_bucket": bucket, "stat": stat, "n": n,
                "ratio_low": float(lo), "ratio_high": float(hi),
                "interval_low_n_flag": n < ROOKIE_INTERVAL_MIN_N,
            })
    return pd.DataFrame(rows)
