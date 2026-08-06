"""Phase 5 entry point: generate per-game rate projections for a target
season, combining the veteran LightGBM models (trained in train.py, saved
under models/) and the rookie rule-based path (rookies.py). Does not
re-derive anything from Phase 2/3 - just loads saved artifacts and this
phase's feature-building code.

Usage: `python -m src.projection.predict --season 2026`, or import
`project_season(conn, season)` directly.

Output: one DataFrame, one row per (player_id, position, stat), with a
`source` column ('veteran_model' | 'rookie_rule') and `low_confidence`
(True for every rookie row, per the hard project rule that rookie
projections must be flagged separately from veteran ones - never silently
mixed in as equally-confident numbers).

Prediction intervals (`pred_pg_low`/`pred_pg_high`, 10th/90th empirical
percentile, i.e. an 80% interval - see PHASE5_REPORT.md for why this width
and why empirical over a second quantile-regression model): veteran rows
use `models/interval_residuals.csv` (built by `backtest.py` from the SAME
2025 held-out backtest as the MAE table - genuine out-of-sample residuals,
added to pred_pg). Rookie rows have no naive-baseline backtest to draw
residuals from, so they use a different, multiplicative fallback (within-
bucket historical ratio of actual/bucket-mean per-game rate - see
rookies.py's rookie_interval_ratios) instead of reusing the veteran
residuals, since rookie-year variance is a distinct, larger regime.
`models/interval_residuals.csv` must exist before calling this (run
`python -m src.projection.backtest` once, after `train.py`).

Framing caveat carried from train.py/transitions.py: veteran projections
for season N+1 use season N's own observed opportunity/scheme features as
the best available proxy for season N+1 conditions (season N+1 hasn't been
played yet, so its own oc_tendency_profiles/OL-quality rows don't exist).
If a team's OC situation is KNOWN to change entering the target season
(e.g. a newly hired play-caller), the caller should override the relevant
oc_tendency_profiles columns with that OC's `inherited_*` row before
calling this function - this module does not do that substitution
automatically, since it doesn't know how new the coaching sitution is,
only what's already computed in the DB for completed seasons.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

from src.projection.data_prep import get_conn
from src.projection.features import build_player_season_features, TARGET_STATS, OC_METRICS
from src.projection.ol_quality import team_season_ol_quality
from src.projection.transitions import ALL_FEATURES
from src.projection.rookies import (
    build_rookie_dataset, fit_rookie_baselines, predict_rookies,
    identify_target_season_rookie_class,
    team_vacated_opportunity, rookie_interval_ratios,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
INTERVAL_RESIDUALS_PATH = os.path.join(MODELS_DIR, "interval_residuals.csv")
DEPTH_CHART_PATH = os.path.join(REPO_ROOT, "src", "depth_chart", "starters_2026.csv")
# Rookie ratio fallback if a bucket/stat combo has too few historical rows
# for its own empirical ratio (rookie_interval_ratios drops any with <3
# values) - deliberately wide, and always flagged via interval_low_n_flag.
ROOKIE_RATIO_FALLBACK = (0.2, 3.0)

# Team-changer share-transfer scale clip - deliberately reused from
# rookies.VACATED_CLIP (same 0.3-2.5 band) rather than inventing a new
# number, since it's the same kind of small-sample-safety clamp on a
# vacated-opportunity ratio (see reassign_team_changers docstring below).
TEAM_CHANGE_SHARE_CLIP = (0.3, 2.5)
# Heavy discount applied to a veteran's point prediction/interval when they
# are NOT in the curated src/depth_chart/starters_2026.csv relevant-depth
# table for their (new team, position) - see apply_depth_chart_gating.
DEEP_BENCH_DISCOUNT = 0.15


def load_models():
    models = {}
    for position, stats in TARGET_STATS.items():
        for stat in stats:
            path = os.path.join(MODELS_DIR, f"{position}_{stat}.joblib")
            models[(position, stat)] = joblib.load(path)
    return models


def load_interval_residuals():
    if not os.path.exists(INTERVAL_RESIDUALS_PATH):
        raise FileNotFoundError(
            f"{INTERVAL_RESIDUALS_PATH} not found - run `python -m src.projection.backtest` "
            "once (after train.py) to build it before calling project_season."
        )
    return pd.read_csv(INTERVAL_RESIDUALS_PATH)


def load_target_roster_map(conn, target_season):
    """player_id -> (team, status) from seasonal_rosters[target_season], the
    source of truth for "what team is this player actually on for the
    season being projected" (fixes the Cousins-on-ATL/Murray-on-MIN bug -
    the old code used the player's season_from RESOLVED team for
    everything). A player can have >1 roster row in a season (practice
    squad stint, in-season cut/re-sign); status='ACT' is preferred when
    present, per the spec's explicit guidance. Non-ACT statuses (RES/PUP,
    CUT, RET, E14) are NOT dropped or treated as "no longer relevant" -
    IR/PUP doesn't mean out for a season that hasn't started, and even a
    late cut is still worth surfacing rather than silently vanishing - they
    are kept with their roster team and flagged via the output's
    `roster_status` column so a reader can judge for themselves (e.g. a
    RET status probably means don't trust this row at all)."""
    df = pd.read_sql(f"select player_id, team, status from seasonal_rosters where season={target_season}", conn)
    df["is_act"] = (df["status"] == "ACT").astype(int)
    df = df.sort_values(["player_id", "is_act"], ascending=[True, False])
    df = df.drop_duplicates(subset=["player_id"], keep="first")
    return df.set_index("player_id")[["team", "status"]]


TEAM_CONTEXT_COLS = ["ol_pass_protection_score", "ol_run_blocking_score", "ol_confidence_low_churn"] + OC_METRICS


# Roles the curated depth chart (Task 2) confirms as genuinely eligible to
# absorb a NEW team's windfall opportunity (scale > 1.0). A player whose
# curated role is 'backup', or who isn't in the curated table at all for
# their new team, still gets the DOWNWARD half of the scale (a worse
# situation than average should still reduce their projected share) but not
# the upward half - see the bug this fixes in reassign_team_changers'
# docstring.
BOOST_ELIGIBLE_ROLES = {"starter", "committee"}


def reassign_team_changers(conn, df, target_season, depth_chart):
    """Task 1 fix. For every player-row (source_season features), resolve
    the player's ACTUAL target_season team from seasonal_rosters and, for
    players who changed teams, re-point every team-dependent feature at the
    NEW team instead of silently keeping the old (source_season) team's
    numbers - the exact bug the project owner found by eyeballing Cousins
    (shown on ATL, his 2025 team, instead of his real 2026 team LV) and
    Murray (shown on ARI instead of MIN).

    Three things get re-pointed for a team-changer, all justified in
    PHASE6_REPORT.md:

    1. Output team label -> target_season roster team (source of truth).
    2. Team-context features (oc_tendency_profiles + OL quality) -> the
       NEW team's most recently OBSERVED season (source_season, i.e.
       2025 for a 2026 target) instead of the old team's. This is the same
       judgment call train.py/transitions.py already make for the
       no-team-change case ("last observed season stands in for the
       unplayed next one") - just correctly re-pointed at the team the
       player is actually walking into, not the one they left.
    3. Player SHARE features (carry/target/rz share, snap_pct) -> the old
       team's share number does NOT carry over to a new team with a
       different depth chart and different available volume (that would be
       just as wrong as the original bug, in a subtler way). Estimated
       instead via a team-changer adaptation of rookies.py's
       "vacated opportunity" concept: this player's own established share
       at their OLD team is used as a "quality tier" signal (how much
       volume this player is capable of commanding when given a role), then
       scaled by how much MORE or LESS opportunity is open at the new team
       compared to a league-average team this season:
           scale = clip(new_team_vacated_share / league_avg_vacated_share, 0.3, 2.5)
           new_share = old_team_share * scale
       `team_vacated_opportunity` (rookies.py) already computes, per team,
       what fraction of last season's carries/targets belonged to players
       who are no longer on that team for target_season (roster-fallback
       already built in for a season with zero played games) - exactly the
       "how much room is actually open here" signal a real preseason
       projection needs. The league average (not the player's OLD team's
       own vacated share) is used as the baseline, mirroring how
       rookies.predict_rookies scales against the historical BUCKET
       average rather than the specific player's own prior situation.

    LIMITATION, stated plainly: this does not, and cannot, capture scheme
    fit ("the new team's offense throws far more to the slot than the old
    team's did") - it only reflects how much raw opportunity is open, not
    how a specific scheme will actually distribute it. That residual is a
    real, unaddressed source of error for every team-changer in this
    output, on top of whatever normal projection error already exists.

    BUG FOUND AND FIXED (post-Phase-6, spot-checked while building fantasy
    points): the scale above was originally applied uncapped to every
    team-changer independently, with no check on whether another player
    (the team's actual new starter) was already the one absorbing that
    vacated opportunity. Concretely: Kenny Gainwell's own PIT carry_share
    (0.28, a real committee share) times TB's vacated-carry scale (1.48x,
    since Bucky Irving's team lost a lot of 2025 carries) produced an
    implied 0.41 carry_share for a player the curated depth chart correctly
    lists as TB's RB2 BEHIND Irving - bell-cow volume for a backup, because
    the team's whole vacancy was being credited to every team-changer at
    once instead of primarily to whoever the depth chart says is actually
    stepping into it. Fixed: the UPWARD half of the scale (>1.0, i.e. "this
    team has more room than average") is only applied to players the
    curated depth chart (`depth_chart` param, Task 2's
    src/depth_chart/starters_2026.csv) confirms as `role in {'starter',
    'committee'}` for their new team+position - see BOOST_ELIGIBLE_ROLES.
    Everyone else (confirmed 'backup', or not in the curated table for
    their new team at all) still gets the DOWNWARD half of the scale (a
    worse-than-average opportunity should still reduce their share) but is
    capped at 1.0 on the upside, so a windfall at the team level can no
    longer inflate a confirmed backup past their own established volume.
    This only applies for target_season=2026 (the only season with a
    curated table); for any other season, depth_chart is empty and every
    team-changer gets the ORIGINAL uncapped scale (unchanged pre-fix
    behavior) - there's no curated role signal to gate on outside 2026.
    snap_pct is scaled by the same carry/target scale as the position's
    primary opportunity type (RB->carry scale, WR/TE->target scale, QB->
    left unchanged, since starter-QB snap_pct is ~100% regardless of new
    team and the depth-chart gating in Task 3, not this share model,
    is what actually distinguishes a new team's QB1 from its QB3)."""
    roster_map = load_target_roster_map(conn, target_season)
    # reset_index: every merge below (pd.merge always returns a fresh
    # RangeIndex, regardless of the input frames' index) needs to line up
    # positionally with boolean masks computed on this same row order -
    # without this, `changed`'s original (filtered, non-contiguous) index
    # silently fails to align against the post-merge frame's index.
    df = df.copy().reset_index(drop=True)
    df["team_target"] = df["player_id"].map(roster_map["team"])
    df["roster_status"] = df["player_id"].map(roster_map["status"])
    no_info = df["team_target"].isna()
    if no_info.any():
        # Not found in target_season's roster at all (retired, out of
        # league, or a crosswalk gap) - keep the old team rather than
        # inventing one; this is a real, if rare, residual gap.
        df.loc[no_info, "team_target"] = df.loc[no_info, "team"]
    changed = (df["team_target"] != df["team"]) & ~no_info
    df["team_changed"] = changed

    if changed.any():
        source_season = target_season - 1

        oc = pd.read_sql(
            f"select season, team, {', '.join(OC_METRICS)} from oc_tendency_profiles where season={source_season}", conn,
        )
        olq = team_season_ol_quality(conn, [source_season])
        team_ctx = oc.merge(olq, on=["season", "team"], how="left").drop(columns=["season"])
        team_ctx = team_ctx.rename(columns={"team": "team_target"})

        df = df.merge(team_ctx, on="team_target", how="left", suffixes=("", "_new"))
        for c in TEAM_CONTEXT_COLS:
            new_c = f"{c}_new"
            if new_c in df.columns:
                df.loc[changed, c] = df.loc[changed, new_c]
        df = df.drop(columns=[c for c in df.columns if c.endswith("_new")])

        vacated = team_vacated_opportunity(conn, [target_season])
        vacated = vacated[vacated["season"] == target_season]
        league_avg_carry_vac = vacated["vacated_carry_share"].mean()
        league_avg_target_vac = vacated["vacated_target_share"].mean()
        vacated = vacated.rename(columns={"team": "team_target"})[
            ["team_target", "vacated_carry_share", "vacated_target_share"]
        ]
        df = df.merge(vacated, on="team_target", how="left")

        carry_scale = (df["vacated_carry_share"] / league_avg_carry_vac).clip(*TEAM_CHANGE_SHARE_CLIP).fillna(1.0)
        target_scale = (df["vacated_target_share"] / league_avg_target_vac).clip(*TEAM_CHANGE_SHARE_CLIP).fillna(1.0)

        if not depth_chart.empty:
            role_lookup = depth_chart[["position", "gsis_id", "role"]].rename(columns={"gsis_id": "player_id"})
            role_lookup = role_lookup.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
            df = df.merge(role_lookup, on=["player_id", "position"], how="left")
        else:
            df["role"] = None
        boost_eligible = df["role"].isin(BOOST_ELIGIBLE_ROLES)
        carry_scale = carry_scale.where(boost_eligible, carry_scale.clip(upper=1.0))
        target_scale = target_scale.where(boost_eligible, target_scale.clip(upper=1.0))
        df = df.drop(columns=["role"])

        for c in ["carry_share", "rz_carry_share"]:
            df.loc[changed, c] = (df.loc[changed, c] * carry_scale[changed]).clip(upper=1.0)
        for c in ["target_share", "rz_target_share"]:
            df.loc[changed, c] = (df.loc[changed, c] * target_scale[changed]).clip(upper=1.0)

        snap_scale = pd.Series(1.0, index=df.index)
        snap_scale[df["position"] == "RB"] = carry_scale[df["position"] == "RB"]
        snap_scale[df["position"].isin(["WR", "TE"])] = target_scale[df["position"].isin(["WR", "TE"])]
        df.loc[changed, "snap_pct"] = (df.loc[changed, "snap_pct"] * snap_scale[changed]).clip(upper=1.0)

        df = df.drop(columns=["vacated_carry_share", "vacated_target_share"])

    df["team"] = df["team_target"]
    df = df.drop(columns=["team_target"])
    return df


def load_depth_chart(target_season):
    """Manually curated relevant-depth table (Task 2) - only built for
    2026 so far (src/depth_chart/starters_2026.csv). Returns an empty
    DataFrame (not an error) for any other target_season, so gating simply
    becomes a no-op rather than breaking historical backtesting/other
    seasons - the file is a 2026-specific research deliverable, not a
    general mechanism yet."""
    if target_season != 2026 or not os.path.exists(DEPTH_CHART_PATH):
        return pd.DataFrame(columns=["team", "position", "depth_rank", "gsis_id", "role", "confidence"])
    dc = pd.read_csv(DEPTH_CHART_PATH)
    return dc[dc["season"] == target_season]


def apply_depth_chart_gating(df, depth_chart):
    """Task 3. Gate veteran-model output using the curated depth chart:
    a player who is NOT within the curated relevant-depth table for their
    (new team, position) is discounted, not presented as an equally-weighted
    full projection alongside the team's actual starters/committee members.
    This is the direct fix for the Arizona-3-QB problem (Brissett/Minshew
    are BOTH in the curated table as a real competitive QB1/QB2 situation;
    Slovis is not, and gets discounted instead of standing shoulder to
    shoulder with the other two at full confidence).

    Mechanism chosen: keep the row (do not silently drop it - the project's
    hard rule is never silently disappear an entity), but multiply
    pred_pg/pred_pg_low/pred_pg_high by DEEP_BENCH_DISCOUNT and flag
    low_confidence=True + depth_chart_status='deep_bench_discounted'. Full
    exclusion was considered and rejected: this project has already been
    burned once (PHASE5_REPORT.md's rookie-filter bug) by code that made a
    real player silently vanish from the output; a heavily-discounted,
    clearly-flagged row is auditable in a way a missing row is not, and a
    reader who wants a "starters only" view can trivially filter on
    depth_chart_status themselves.

    Coverage note: src/depth_chart/starters_2026.csv covers QB/RB/WR/TE at
    every one of the 32 teams down to the depths specified in
    PHASE6_REPORT.md (QB1[+2 if competitive], RB1-2, WR1-3, TE1-2) - so for
    2026 there is no "we ran out of time to research this team/position"
    gap; every non-curated veteran at a covered position is a deliberate
    scope decision (below the researched depth), not a data gap. The
    `depth_chart_status` values distinguish this explicitly:
      - 'curated': matched a row in the table (gets depth_rank/role).
      - 'deep_bench_discounted': target_season IS covered by the table
         (i.e. we researched this team's position group) but this specific
         player wasn't in the curated top-N - confirmed outside the
         relevant depth, not merely unresearched.
      - 'not_curated_no_table': target_season has no curated table at all
         (any season other than 2026) - gating is a no-op, not a claim
         about this player's role."""
    df = df.copy()
    if depth_chart.empty:
        df["depth_rank"] = np.nan
        df["role"] = None
        df["depth_chart_status"] = "not_curated_no_table"
        return df

    dc = depth_chart[["position", "gsis_id", "depth_rank", "role"]].rename(columns={"gsis_id": "player_id"})
    dc = dc.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
    df = df.merge(dc, on=["player_id", "position"], how="left")

    matched = df["depth_rank"].notna()
    df["depth_chart_status"] = np.where(matched, "curated", "deep_bench_discounted")

    to_discount = ~matched
    for col in ["pred_pg", "pred_pg_low", "pred_pg_high"]:
        df.loc[to_discount, col] = df.loc[to_discount, col] * DEEP_BENCH_DISCOUNT
    df.loc[to_discount, "low_confidence"] = True
    df.loc[to_discount, "role"] = "deep_bench"
    return df


def project_veterans(conn, feat, source_season, models, resid, target_season):
    """source_season's feature rows -> next-season per-game rate
    predictions, for every player with real source_season production.
    target_season's OWN incoming rookie class (drafted in target_season, or
    UDFA per identify_target_season_rookie_class) is projected separately
    below via the rule-based path instead - but that class is naturally
    ABSENT from source_season's features already (they have no NFL history
    before target_season), so no extra exclusion is needed here to avoid
    double-counting them.

    target_season is now required (previously predict.py effectively
    assumed the player's source_season team carried forward unchanged -
    see reassign_team_changers' docstring for the bug this fixes) and is
    also used to gate output against the curated depth chart (Task 3).

    Bug found and fixed here (Phase 6, while spot-checking a random 2025
    rookie per Task 4): this function used to additionally exclude every
    player whose OWN rookie season equaled source_season (via
    identify_rookie_seasons/identify_udfa_rookie_seasons([source_season])),
    on the stated reasoning "source_season IS their only season, so they
    have no real trailing features." That reasoning doesn't hold - a
    player's rookie season IS a complete, real season of production, and is
    exactly the trailing data the veteran model needs for their SOPHOMORE
    projection. The effect was silently dropping every player's entire
    second season from the output (verified: Ashton Jeanty, a real 2025
    rookie with a full 17-game/267-carry season, was completely absent from
    2026's output before this fix, despite having a valid 2025 feature row -
    the same class of silent-drop bug PHASE5_REPORT.md already found and
    fixed once for long-tenured veterans; this is its sibling for players
    exactly one year removed from being a rookie, not caught by that
    review's Allen/Chase/McCaffrey spot-checks since none of them were in
    their second season at the time). Removing the exclusion here is safe
    (see previous paragraph: target_season's real rookies can't appear in
    source_season's features regardless), and was re-verified end-to-end
    after the fix (Jeanty now appears; see PHASE6_REPORT.md's spot-checks)."""
    depth_chart = load_depth_chart(target_season)

    base = feat[(feat["season"] == source_season) & (feat["games_played"] > 0)]
    base = reassign_team_changers(conn, base, target_season, depth_chart)

    rows = []
    for position, stats in TARGET_STATS.items():
        pos_df = base[base["position"] == position]
        if pos_df.empty:
            continue
        X = pos_df[ALL_FEATURES]
        for stat in stats:
            m = models[(position, stat)]
            preds = m["model"].predict(X)
            out = pos_df[["player_id", "team", "position", "team_changed", "roster_status"]].copy()
            out["stat"] = stat
            out["pred_pg"] = np.clip(preds, 0, None)  # a per-game rate can't be negative; LightGBM isn't constrained
            out["source"] = "veteran_model"
            out["low_confidence"] = False

            r = resid[(resid["position"] == position) & (resid["stat"] == stat)]
            if r.empty:
                out["pred_pg_low"], out["pred_pg_high"], out["interval_low_n_flag"] = np.nan, np.nan, True
            else:
                r = r.iloc[0]
                out["pred_pg_low"] = (out["pred_pg"] + r["resid_low"]).clip(lower=0)
                out["pred_pg_high"] = out["pred_pg"] + r["resid_high"]
                out["interval_low_n_flag"] = bool(r["low_n_flag"])
            rows.append(out)
    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if combined.empty:
        return combined

    depth_chart = load_depth_chart(target_season)
    combined = apply_depth_chart_gating(combined, depth_chart)
    return combined


def project_season(conn, target_season):
    """Project `target_season` per-game rates using `target_season - 1`
    features for veterans, and the rookie rule-based path for
    `target_season`'s actual draft-year rookies."""
    source_season = target_season - 1
    models = load_models()
    resid = load_interval_residuals()

    feat = build_player_season_features(conn, seasons=list(range(2016, target_season)))
    vet = project_veterans(conn, feat, source_season, models, resid, target_season)
    vet["season"] = target_season

    # rookie path: bucket baselines fit on all completed rookie seasons
    # strictly before target_season (drafted + UDFA, both requiring confirmed
    # active-week production - see rookies.py), applied to target_season's
    # rookie class. target_season's own rookie class is identified separately
    # (identify_target_season_rookie_class) since target_season has no played
    # games yet to confirm anyone's "first active season" against - it reads
    # the class directly off draft_picks + seasonal_rosters instead.
    hist_seasons = list(range(2016, target_season))
    hist_feat = build_player_season_features(conn, seasons=hist_seasons)
    rdf = build_rookie_dataset(conn, hist_feat, seasons=hist_seasons)
    baselines = fit_rookie_baselines(rdf, hist_seasons)
    ratios = rookie_interval_ratios(rdf, baselines, hist_seasons)

    target_class = identify_target_season_rookie_class(conn, target_season)
    vacated = team_vacated_opportunity(conn, [target_season])
    target_class = target_class.merge(vacated, on=["season", "team"], how="left")
    depth_chart = load_depth_chart(target_season)
    rookie_preds = predict_rookies(target_class, baselines, [target_season], depth_chart=depth_chart)

    pg_cols = [c for c in rookie_preds.columns if c.endswith("_pg")]
    rookie_long = rookie_preds.melt(
        id_vars=["player_id", "team", "position", "season", "rookie_tier", "round_bucket", "qb_sleeper_play_prob"],
        value_vars=pg_cols, var_name="stat", value_name="pred_pg",
    )
    rookie_long["stat"] = rookie_long["stat"].str.replace("_pg", "", regex=False)
    rookie_long["source"] = "rookie_rule"
    rookie_long["low_confidence"] = True
    rookie_long = rookie_long.dropna(subset=["pred_pg"])
    rookie_long = rookie_long[
        rookie_long.apply(lambda r: r["stat"] in TARGET_STATS.get(r["position"], []), axis=1)
    ]

    rookie_long = rookie_long.merge(ratios, on=["position", "round_bucket", "stat"], how="left")
    no_ratio = rookie_long["ratio_low"].isna()
    rookie_long.loc[no_ratio, "ratio_low"] = ROOKIE_RATIO_FALLBACK[0]
    rookie_long.loc[no_ratio, "ratio_high"] = ROOKIE_RATIO_FALLBACK[1]
    rookie_long.loc[no_ratio, "interval_low_n_flag"] = True
    rookie_long["interval_low_n_flag"] = rookie_long["interval_low_n_flag"].fillna(False)
    rookie_long["pred_pg_low"] = (rookie_long["pred_pg"] * rookie_long["ratio_low"]).clip(lower=0)
    rookie_long["pred_pg_high"] = rookie_long["pred_pg"] * rookie_long["ratio_high"]
    rookie_long = rookie_long.drop(columns=["ratio_low", "ratio_high", "round_bucket", "n"], errors="ignore")

    # Rookies already use the correct target_season team (identify_target_season_rookie_class
    # reads it off draft_picks/seasonal_rosters directly - see PHASE5_REPORT.md), so no
    # team-change adjustment or discount-gating applies here. Still attach
    # depth_rank/role from the curated table when the rookie happens to be
    # on it (e.g. a rookie who's already his team's curated RB1/WR1), purely
    # informational - not used to discount rookie_rule predictions, which
    # are already flagged low_confidence by construction.
    depth_chart = load_depth_chart(target_season)
    rookie_long["team_changed"] = False
    rookie_long["roster_status"] = np.nan
    if not depth_chart.empty:
        dc = depth_chart[["position", "gsis_id", "depth_rank", "role"]].rename(columns={"gsis_id": "player_id"})
        dc = dc.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
        rookie_long = rookie_long.merge(dc, on=["player_id", "position"], how="left")
    else:
        rookie_long["depth_rank"], rookie_long["role"] = np.nan, None
    rookie_long["depth_chart_status"] = "rookie_path"

    combined = pd.concat([vet, rookie_long], ignore_index=True, sort=False)
    return combined


OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

OUTPUT_COLUMNS = [
    "player_id", "display_name", "team", "position", "stat",
    "pred_pg", "pred_pg_low", "pred_pg_high",
    "source", "low_confidence", "rookie_tier", "interval_low_n_flag", "season",
    # Phase 6 additions:
    "team_changed", "roster_status",            # Task 1 - team reassignment transparency
    "depth_rank", "role", "depth_chart_status",  # Task 2/3 - curated depth chart + gating
    "qb_sleeper_play_prob",  # rookie QB survivorship-bias correction - NaN for non-QB/veteran rows
]


def with_display_names(conn, out, target_season):
    """Join players.display_name onto the combined projection output - a
    human reading this CSV needs a name, not just a raw gsis_id.

    Data-quality gap found and worked around: nfl_data_py's 2026
    draft_picks.gsis_id column does NOT contain real gsis_ids for this
    draft class (nflverse hasn't back-filled them yet - spot-checked: 0/230
    2026 rows match the `00-0######` gsis_id format, vs 256/256 for 2025)
    - it's some other placeholder id, so these players are structurally
    absent from `players.gsis_id` too. Falls back to draft_picks'
    `pfr_player_name` (drafted rookies) and seasonal_rosters' `player_name`
    (UDFA) for exactly the rows players.display_name can't resolve, rather
    than shipping a CSV with blank names for the entire incoming rookie
    class."""
    players = pd.read_sql("select gsis_id as player_id, display_name from players", conn)
    out = out.merge(players, on="player_id", how="left")

    draft_names = pd.read_sql(
        f"select gsis_id as player_id, pfr_player_name as name from draft_picks where season = {target_season}", conn,
    ).drop_duplicates(subset=["player_id"]).set_index("player_id")["name"]
    roster_names = pd.read_sql(
        f"select player_id, player_name as name from seasonal_rosters where season = {target_season}", conn,
    ).drop_duplicates(subset=["player_id"]).set_index("player_id")["name"]

    missing = out["display_name"].isna()
    out.loc[missing, "display_name"] = out.loc[missing, "player_id"].map(draft_names)
    still_missing = out["display_name"].isna()
    out.loc[still_missing, "display_name"] = out.loc[still_missing, "player_id"].map(roster_names)

    unresolved = out["display_name"].isna().sum()
    if unresolved:
        print(f"WARNING: {unresolved} projection rows have no resolvable display name at all "
              f"(not in players, draft_picks, or seasonal_rosters for {target_season}) - left null, not faked.")
    return out


def export_projections(conn, target_season, path):
    out = project_season(conn, target_season)
    out = with_display_names(conn, out, target_season)
    out = out[OUTPUT_COLUMNS].sort_values(["position", "team", "player_id", "stat"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_csv(path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--out", default=None, help="CSV output path (default: output/projections_<season>.csv)")
    args = ap.parse_args()
    out_path = args.out or os.path.join(OUTPUT_DIR, f"projections_{args.season}.csv")

    conn = get_conn()
    out = export_projections(conn, args.season, out_path)
    conn.close()

    pd.set_option("display.width", 200)
    print(out.head(30).to_string(index=False))
    print(f"\n{len(out)} projection rows for season {args.season} "
          f"({(out.source=='rookie_rule').sum()} rookie rows flagged low_confidence, "
          f"{out['interval_low_n_flag'].sum()} rows with a low-n interval flag)")
    print(f"Written -> {out_path}")


if __name__ == "__main__":
    main()
