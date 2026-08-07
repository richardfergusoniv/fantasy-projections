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
import sys

import joblib
import numpy as np
import pandas as pd

from src.projection.data_prep import get_conn, team_season_opponent_strength
from src.projection.features import build_player_season_features, TARGET_STATS, OC_METRICS
from src.projection.ol_quality import team_season_ol_quality
from src.projection.transitions import (
    ALL_FEATURES, TEAM_FEATURES, REFRAMED_SHARE_STATS,
    RECEIVING_SHARE_SUM_CAP, receiving_share_scale,
)
from src.projection.rookies import (
    build_rookie_dataset, fit_rookie_baselines, predict_rookies,
    identify_target_season_rookie_class,
    team_vacated_opportunity, rookie_interval_ratios, combine_athletic_scores_by_pfr_id,
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

# Bug found via output/sleeper_comparison_2026.csv (comparing our per-game
# volume predictions against Sleeper's public projections): a player who IS
# in the curated depth chart (so DEEP_BENCH_DISCOUNT above never fires) but
# whose curated `role` is 'committee' or 'backup' was still getting the raw
# LightGBM veteran model's per-game rate untouched - the curated table was
# only ever consulted to GATE the upward vacancy boost in
# reassign_team_changers (BOOST_ELIGIBLE_ROLES), never to shrink the
# model's own raw rate prediction. The veteran model's features (career
# rate stats, share features, etc.) reflect the player's OWN historical
# per-game usage, which for a committee back or a clipboard-holding backup
# QB can be inflated by a prior season spent as a starter elsewhere, an
# injury-driven spike week, or simply not knowing this specific team's 2026
# depth chart at all - the curated table is exactly the correction signal
# the model itself has no way to see. Concrete evidence (attempts/carries
# per game, our model vs. Sleeper):
#   Alvin Kamara (NO, committee):    12.7 vs 2.5
#   Rachaad White (WAS, committee):  14.7 vs 6.3
#   Emari Demercado (KC, committee):  8.9 vs 1.6
#   Tyler Allgeier (ARI, committee): 12.7 vs 5.3
#   Isiah Pacheco (DET, backup):      9.0 vs 3.7
#   J.J. McCarthy / Rudolph / Richardson (QB, backup): 28.2 / 16.0 / 11.1
#     attempts/g vs Sleeper's 1.8 / 1.8 / 1.6 - a real backup NFL QB behind
#     a healthy starter gets close to zero live-game volume, not
#     near-starter attempts.
# ROLE_VOLUME_DISCOUNT below is a stated, un-tuned judgment call (same
# spirit as DEEP_BENCH_DISCOUNT and TEAM_CHANGE_SHARE_CLIP - considered,
# not backtested against a held-out season, since there's no historical
# curated depth chart to validate discount magnitudes against). 'committee'
# lands well short of 'backup' - a committee player is a confirmed
# meaningful contributor sharing a real role (Sleeper's numbers above still
# show it as a fraction, not a rounding error, of a lead back's volume),
# while a curated 'backup' is functionally the same "gets on the field only
# if the starter is hurt/blown out" situation DEEP_BENCH_DISCOUNT already
# describes for a player who is not curated at all - hence the two
# multipliers land close together (0.15 vs 0.4).
ROLE_VOLUME_DISCOUNT = {"committee": 0.4, "backup": 0.15}


def load_models():
    models = {}
    for position, stats in TARGET_STATS.items():
        for stat in stats:
            path = os.path.join(MODELS_DIR, f"{position}_{stat}.joblib")
            models[(position, stat)] = joblib.load(path)
    # Joint/multi-output Phase A team-total model (train.py) - same
    # ("TEAM", "passing_yards") key backtest.py's own rows use.
    models[("TEAM", "passing_yards")] = joblib.load(os.path.join(MODELS_DIR, "team_passing_yards.joblib"))
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


# opp_def_pass_epa_prior/opp_def_rush_epa_prior (added alongside the
# ceiling/concentration features, see features.py's FEATURE_COLS comment)
# are team-season schedule-strength context exactly like the OL/OC columns
# below - a team-changer's opponent slate for the target season depends on
# their NEW team's schedule, not their old one, so they belong in the same
# re-pointing list.
TEAM_CONTEXT_COLS = (
    ["ol_pass_protection_score", "ol_run_blocking_score", "ol_confidence_low_churn"]
    + OC_METRICS
    + ["opp_def_pass_epa_prior", "opp_def_rush_epa_prior"]
)


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
        # opp_def_pass_epa_prior/opp_def_rush_epa_prior for the NEW team,
        # same "most recently observed season stands in for the unplayed
        # target season" logic already used for OC/OL above - the new
        # team's source_season schedule-strength value is the best proxy
        # available for their target_season schedule.
        # Bug found while integrating (Sleeper-comparison investigation):
        # team_season_opponent_strength internally shifts its defense-EPA
        # lookup to season-1 to align each opponent's PRIOR-season defense
        # against the schedule season - calling it with ONLY [source_season]
        # means that prior season's defense-EPA rows were never fetched at
        # all, so the shifted join always missed and came back NaN for
        # every team. Passing [source_season - 1, source_season] gives it
        # both halves of its own join.
        opp_strength = team_season_opponent_strength(conn, [source_season - 1, source_season])
        opp_strength = opp_strength[opp_strength["season"] == source_season]
        team_ctx = oc.merge(olq, on=["season", "team"], how="left")
        team_ctx = team_ctx.merge(opp_strength, on=["season", "team"], how="left").drop(columns=["season"])
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

        # Bug found via Sleeper comparison (Waddle MIA->DEN, DJ Moore
        # CHI->BUF, both curated 'starter' at their new team but crushed to
        # ~0.06 target_share by the DOWNWARD half of the vacancy scale):
        # team_vacated_opportunity measures how much volume LEFT a team
        # (players no longer on the roster) - a trade acquisition into an
        # already-full room legitimately shows near-zero "vacated" share by
        # that definition even though the team specifically traded for and
        # is starting this player. The curated table confirming role=
        # 'starter' is a stronger, player-specific signal than the
        # team-level vacancy heuristic, so a confirmed starter's own
        # established share (their "quality tier" per this function's
        # class docstring) is never scaled BELOW what they already had -
        # only the upside (extra room beyond a normal team) still applies.
        # 'committee'/'backup' deliberately keep the original full
        # 0.3-2.5 range: a genuine committee/backup landing spot CAN mean
        # real volume loss, and that case is handled separately (and
        # correctly) by ROLE_VOLUME_DISCOUNT in apply_depth_chart_gating
        # rather than needing a floor here too.
        confirmed_starter = changed & (df["role"] == "starter")
        carry_scale = carry_scale.where(~confirmed_starter, carry_scale.clip(lower=1.0))
        target_scale = target_scale.where(~confirmed_starter, target_scale.clip(lower=1.0))

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
         about this player's role.

    Role-based volume discount (bug found post-Phase-6 via
    output/sleeper_comparison_2026.csv - see ROLE_VOLUME_DISCOUNT's
    module-level comment for the evidence): being IN the curated table only
    ever meant the row kept its full, undiscounted pred_pg - a curated
    'committee'/'backup' player's raw per-game rate came straight out of
    the LightGBM model's own read of their historical usage, with nothing
    correcting it toward what this specific 2026 depth-chart role actually
    implies (a real backup gets almost no live-game volume regardless of
    what their feature row says). Fixed here: AFTER the curated/
    deep_bench_discounted split above, curated rows whose `role` is in
    ROLE_VOLUME_DISCOUNT get pred_pg/pred_pg_low/pred_pg_high multiplied by
    that role's factor too, `low_confidence` forced True (same reasoning as
    deep_bench_discounted - a heavily rescaled number is exactly the kind
    of row a reader should be able to tell apart from an un-rescaled one),
    and a NEW `role_discount_applied` boolean column set True so this is
    auditable without having to cross-reference `role` against a constant
    buried in this module - per the project's hard rule, a discount must
    never silently disappear into the number. `depth_chart_status` is left
    as 'curated' (unchanged) for these rows since they genuinely ARE in the
    table - `role_discount_applied` is the correct place to see the
    rescaling, not a new depth_chart_status value that would blur "matched
    the table" with "how the match was treated."

    Interaction with reassign_team_changers' BOOST_ELIGIBLE_ROLES, checked
    explicitly: a team-changing 'committee' player can get BOTH the upward
    vacancy-scale boost there AND this downward role discount here. That is
    not double-counting - they answer different questions. The boost in
    reassign_team_changers scales this player's OWN established share
    (their prior team's carry_share/target_share) by how much MORE
    opportunity the new team has open overall (a team-level signal, applied
    before the model even runs). This discount is applied AFTER the model
    has already produced a prediction from those (possibly boosted)
    features, and corrects for the fact that a 'committee' role means this
    player is only expected to capture a FRACTION of whatever volume - big
    or small - actually materializes at their position on the new team; the
    model has no feature that tells it "a specific other player is sharing
    this backfield with you." A big team-level opportunity split three ways
    should still end up smaller per-player than the same opportunity given
    to a lone bell-cow, which is exactly what applying both produces."""
    df = df.copy()
    if depth_chart.empty:
        df["depth_rank"] = np.nan
        df["role"] = None
        df["depth_chart_status"] = "not_curated_no_table"
        df["role_discount_applied"] = False
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

    role_factor = df["role"].map(ROLE_VOLUME_DISCOUNT)
    to_role_discount = matched & role_factor.notna()
    df["role_discount_applied"] = to_role_discount
    for col in ["pred_pg", "pred_pg_low", "pred_pg_high"]:
        df.loc[to_role_discount, col] = df.loc[to_role_discount, col] * role_factor[to_role_discount]
    df.loc[to_role_discount, "low_confidence"] = True

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
            out["source"] = "veteran_model"
            out["low_confidence"] = False

            if (position, stat) in REFRAMED_SHARE_STATS:
                # preds here is a SHARE (see train.py's fit_one), not a
                # rate - composed into a real pred_pg by
                # _compose_reframed_receiving_predictions AFTER this whole
                # loop finishes, since normalizing a team's shares needs
                # ALL THREE reframed positions' predictions at once, and
                # this loop produces them at different iterations. pred_pg
                # holds the raw share as a placeholder; low/high are
                # deferred entirely (computed in rate units, post-compose).
                out["pred_pg"] = np.clip(preds, 0, None)
                out["pred_pg_low"], out["pred_pg_high"], out["interval_low_n_flag"] = np.nan, np.nan, False
            else:
                out["pred_pg"] = np.clip(preds, 0, None)  # a per-game rate can't be negative; LightGBM isn't constrained
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

    # Phase 2 of the consensus-gap work: the reframed receiving rows leave
    # this function with pred_pg still in (now role-discounted) SHARE
    # units plus a team_total_pred helper column - project_season composes
    # them into real rates only after the rookie path has produced the
    # incoming rookies' implied shares for the same denominator. Gating
    # multiplying a share by its role discount here is exactly what lets
    # the later renormalization see discounted shares.
    combined = _attach_team_total_pred(combined, base, models[("TEAM", "passing_yards")])

    depth_chart = load_depth_chart(target_season)
    combined = apply_depth_chart_gating(combined, depth_chart)
    _warn_discounted_high_usage(conn, combined, base, source_season)
    return combined


# Curation-tripwire thresholds: season-N usage above ANY of these makes a
# discounted projection suspicious enough to warrant a human second look at
# the curated depth chart. Calibrated on the two known misses these would
# have caught (Parker Washington: 97 targets/56.5 rec ypg; Wan'Dale
# Robinson: 141 targets/63.4 rec ypg, both absent from starters_2026.csv
# and silently auto-discounted 0.15x) while sitting above genuine
# deep-bench usage levels.
TRIPWIRE_SEASON_TARGETS = 70
TRIPWIRE_TARGETS_PG = 5.0
TRIPWIRE_REC_YPG = 40.0


def _warn_discounted_high_usage(conn, combined, base, source_season):
    """Warn (stderr only - curation stays a human decision, this NEVER
    changes a number) when a player whose projection was discounted by
    depth-chart gating had clearly fantasy-relevant usage in the source
    season. A high-usage player falling through to a 0.15x/0.4x discount is
    far more likely a curation gap in the depth-chart table than a real
    role collapse."""
    discounted = combined[
        (combined["depth_chart_status"] == "deep_bench_discounted")
        | combined["role_discount_applied"]
    ]
    if discounted.empty:
        return
    usage = base[["player_id", "targets", "targets_pg", "receiving_yards_pg", "games_played"]]
    flagged = (
        discounted[["player_id", "team", "position", "role"]]
        .drop_duplicates("player_id")
        .merge(usage, on="player_id")
    )
    # Per-game thresholds need a minimum sample (>= 4 games) so a 1-3 game
    # blip (e.g. 5 targets in a single spot appearance) doesn't fire; the
    # season-total threshold needs no such guard by construction. 4 keeps
    # genuinely relevant injury-shortened seasons (Tyreek Hill's 4-game
    # 66-ypg 2025) inside the net.
    enough_games = flagged["games_played"] >= 4
    flagged = flagged[
        (flagged["targets"] >= TRIPWIRE_SEASON_TARGETS)
        | (enough_games & (flagged["targets_pg"] >= TRIPWIRE_TARGETS_PG))
        | (enough_games & (flagged["receiving_yards_pg"] >= TRIPWIRE_REC_YPG))
    ]
    if flagged.empty:
        return
    names = pd.read_sql("SELECT gsis_id AS player_id, display_name FROM players", conn)
    flagged = flagged.merge(names, on="player_id", how="left")
    print(
        f"CURATION TRIPWIRE: {len(flagged)} discounted player(s) had fantasy-relevant "
        f"season-{source_season} usage - verify their rows in the curated depth chart:",
        file=sys.stderr,
    )
    for _, r in flagged.sort_values("targets", ascending=False).iterrows():
        print(
            f"  {r['display_name']} ({r['team']} {r['position']}, role={r['role']}): "
            f"{r['targets']:.0f} targets, {r['targets_pg']:.1f} tgt/g, "
            f"{r['receiving_yards_pg']:.1f} rec ypg in {r['games_played']:.0f} games",
            file=sys.stderr,
        )


def _attach_team_total_pred(combined, base, team_model):
    """Attach the team_passing_yards_pg forecast (joint/multi-output Phase
    A's shared anchor) as a helper column, drawn from `base`'s own
    TEAM_FEATURES (already re-pointed to a team-changer's NEW team by
    reassign_team_changers). Attached BEFORE depth-chart gating so the
    composition itself can run AFTER gating - Phase 2 of the consensus-gap
    work moved it there, so the share-sum renormalization sees discounted
    shares (a 0.15x bench player consumes 0.15x of the budget) and, at the
    project_season level, the rookie path's implied shares.
    _compose_reframed_receiving_predictions drops the column when done."""
    team_feat = base.dropna(subset=TEAM_FEATURES).drop_duplicates(subset=["team"])[["team"] + TEAM_FEATURES]
    team_feat["team_total_pred"] = np.clip(team_model["model"].predict(team_feat[TEAM_FEATURES]), 0, None)
    combined = combined.merge(team_feat[["team", "team_total_pred"]], on="team", how="left")
    # A team with no resolvable TEAM_FEATURES row (the same rare team=NaN
    # gap backtest.py's reframed path already documents) has no
    # team_total_pred to compose with - falls back to 0 rather than
    # NaN-propagating a whole player's row into an unusable prediction;
    # genuinely rare (verified 0 occurrences in the 2026 live run).
    combined["team_total_pred"] = combined["team_total_pred"].fillna(0)
    return combined


def _compose_reframed_receiving_predictions(combined, resid, rookie_receiving=None):
    """Joint/multi-output Phase A: turns the SHARE predictions the main
    project_veterans loop produced for REFRAMED_SHARE_STATS rows
    (WR/TE/RB receiving_yards) into real pred_pg values, by composing them
    with the team_passing_yards_pg forecast _attach_team_total_pred left on
    the frame - see transitions.py's REFRAMED_SHARE_STATS/
    RECEIVING_SHARE_LABEL for the shared source of truth on which
    (position, stat) combos are reframed.

    Phase 2 of the consensus-gap work: this now runs AFTER depth-chart
    gating (project_season calls it once the rookie path has run too), so
    the shares arriving here are already role-discounted, and the share-sum
    renormalization (transitions.receiving_share_scale, shared with
    backtest.py) therefore charges each player their SHIPPED share of the
    team budget, not their raw one. Previously the cap ran on raw shares:
    63.6% of NYG's 1.86 "over-budget" share sum came from players about to
    be multiplied 0.15x/0.4x, and the renormalization squeezed Malik Nabers
    19% for it. `rookie_receiving` (per-team rookie-path receiving_yards
    pred_pg rows) enters the denominator as implied shares - the
    user-diagnosed Robinson/Tate case: an incoming 1st-round WR consumes
    real target share the veteran share models can't see.

    Interval note: pred_pg_low/high = composed pred +/- empirical residual,
    with the residual in absolute rate units NOT scaled by any role
    discount (pre-Phase-2, gating ran after compose and scaled the whole
    interval; the residuals were calibrated on undiscounted backtest
    predictions, so keeping them absolute is the more faithful reading).

    Non-reframed rows pass through unchanged (minus the helper column)."""
    reframed_index = pd.MultiIndex.from_tuples(REFRAMED_SHARE_STATS, names=["position", "stat"])
    mask = combined.set_index(["position", "stat"]).index.isin(reframed_index)
    if not mask.any():
        return combined.drop(columns=["team_total_pred"], errors="ignore")
    reframed = combined[mask].copy()
    other = combined[~mask].drop(columns=["team_total_pred"], errors="ignore").copy()
    other["receiving_share_capped"] = np.nan

    extra_team_share = None
    if rookie_receiving is not None and not rookie_receiving.empty:
        team_totals = reframed.drop_duplicates("team").set_index("team")["team_total_pred"]
        rr = rookie_receiving.copy()
        rr["team_total_pred"] = rr["team"].map(team_totals)
        # A rookie on a team with no composable veteran total (or a 0
        # fallback total) contributes nothing rather than dividing by 0.
        rr = rr[rr["team_total_pred"] > 0]
        extra_team_share = (rr["pred_pg"] / rr["team_total_pred"]).groupby(rr["team"]).sum()

    share_df = reframed[["team"]].copy()
    share_df["share"] = reframed["pred_pg"]
    scale, over_cap = receiving_share_scale(share_df, extra_team_share=extra_team_share)
    reframed["receiving_share_capped"] = over_cap
    reframed["pred_pg"] = reframed["pred_pg"] * scale * reframed["team_total_pred"]
    reframed = reframed.drop(columns=["team_total_pred"])

    r = resid[(resid["stat"] == "receiving_yards") & (resid["position"].isin(["WR", "TE", "RB"]))]
    reframed = reframed.merge(
        r[["position", "stat", "resid_low", "resid_high", "low_n_flag"]], on=["position", "stat"], how="left",
    )
    has_resid = reframed["resid_low"].notna()
    reframed.loc[has_resid, "pred_pg_low"] = (
        reframed.loc[has_resid, "pred_pg"] + reframed.loc[has_resid, "resid_low"]
    ).clip(lower=0)
    reframed.loc[has_resid, "pred_pg_high"] = reframed.loc[has_resid, "pred_pg"] + reframed.loc[has_resid, "resid_high"]
    reframed["interval_low_n_flag"] = reframed["low_n_flag"].fillna(True)
    reframed = reframed.drop(columns=["resid_low", "resid_high", "low_n_flag"])

    return pd.concat([other, reframed], ignore_index=True, sort=False)


# Band for team_pass_catch_coherence_flag (see add_team_pass_catch_coherence_flag
# below): a stated, un-tuned judgment call, not fit to any target - real
# historical team-seasons show this ratio can legitimately range far wider
# than this band (up to ~10x, driven by in-season QB injuries/changes) per
# the research that motivated this flag, so this is deliberately loose
# (flag clearly incoherent cases like CHI/NYG/MIA's 2026 projections, not
# every team that's merely off-center) rather than tight.
PASS_CATCH_COHERENCE_BAND = (0.8, 1.35)


def add_team_pass_catch_coherence_flag(df, depth_chart):
    """Diagnostic-only column, added post-Phase-6 after two prototyped fixes
    (a team-volume input feature for the WR/TE/RB models, and a hard
    post-hoc rescale of receiver predictions to match the team's QB
    prediction) both backtested as a wash-to-negative - see the research
    behind this addition. Neither fix is safe to apply automatically: the
    rescale in particular was found to propagate the QB model's OWN
    prediction error onto every receiver on a team (a low-confidence new
    starter like Malik Willis dragging Tyreek Hill's prediction down with
    him), which is exactly the kind of silent-error-propagation this
    project's other gating mechanisms (ROLE_VOLUME_DISCOUNT,
    apply_depth_chart_gating) are built to avoid, not reproduce.

    Instead: this computes, per team, the ratio of (sum of ALL WR/TE/RB
    predicted receiving_yards_pg, every row regardless of source/
    low_confidence - a bench player's garbage-time catches still count
    against the team's real passing total) to (the curated depth chart's
    confirmed role='starter' QB's predicted passing_yards_pg for that
    team), and flags team-season rows where that ratio falls outside
    PASS_CATCH_COHERENCE_BAND. This is purely informational - no pred_pg
    value is touched - surfacing exactly the kind of internal-inconsistency
    signal that first raised the Nabers/Reed/Burden question (their teams'
    receiving corps totals didn't add up against their own QB's projected
    volume) so a reader can judge for themselves which teams' predictions
    to treat with extra skepticism, the same "never silently hide, always
    flag" principle as low_confidence/deep_bench_discounted elsewhere in
    this module.

    Only meaningful for a target_season with a curated depth chart (2026
    currently - see load_depth_chart) since it needs a confirmed, singular
    starter to anchor against; every row gets
    team_pass_catch_coherence_flag=NaN (not False - "not computable", not
    "confirmed coherent") for any other season, or for a team where the
    curated table doesn't resolve to exactly one starter QB."""
    df = df.copy()
    if depth_chart.empty:
        df["team_pass_catch_ratio"] = np.nan
        df["team_pass_catch_coherence_flag"] = np.nan
        return df

    qb_rows = df[(df["position"] == "QB") & (df["stat"] == "passing_yards")]
    starters = depth_chart[(depth_chart["position"] == "QB") & (depth_chart["role"] == "starter")]
    starters = starters[["team", "gsis_id"]].rename(columns={"gsis_id": "player_id"})
    anchor = qb_rows.merge(starters, on=["team", "player_id"], how="inner")
    # Exactly one curated starter per team is the expected case (verified
    # true for every 2026 team) - a team with 0 or >1 resolved starter rows
    # (e.g. a future season's curated table drawn up differently) falls
    # back to the mean of whatever resolves rather than erroring, but is
    # real, rare, and would show up as an odd-looking anchor if it ever
    # happens - not silently special-cased away.
    anchor = anchor.groupby("team")["pred_pg"].mean().rename("qb_anchor_pg")

    recv = df[(df["position"].isin(["WR", "TE", "RB"])) & (df["stat"] == "receiving_yards")]
    recv_sum = recv.groupby("team")["pred_pg"].sum().rename("team_receiving_sum_pg")

    team_ratio = pd.concat([anchor, recv_sum], axis=1)
    team_ratio["team_pass_catch_ratio"] = team_ratio["team_receiving_sum_pg"] / team_ratio["qb_anchor_pg"]
    low, high = PASS_CATCH_COHERENCE_BAND
    # object dtype (not bool) so the NaN assigned below for an unresolved
    # ratio is representable - a plain bool column can't hold NaN and
    # .between() on a NaN input silently evaluates to False, which would
    # otherwise misrepresent "not computable" as "confirmed coherent."
    team_ratio["team_pass_catch_coherence_flag"] = (
        ~team_ratio["team_pass_catch_ratio"].between(low, high)
    ).astype(object)
    # NaN (not False) when either side of the ratio isn't resolvable (no
    # curated starter found for this team, or no WR/TE/RB rows at all) -
    # "not computable" is a distinct, honestly-reported state from
    # "confirmed coherent."
    team_ratio.loc[team_ratio["team_pass_catch_ratio"].isna(), "team_pass_catch_coherence_flag"] = np.nan
    team_ratio = team_ratio.reset_index().rename(columns={"index": "team"})

    df = df.merge(
        team_ratio[["team", "team_pass_catch_ratio", "team_pass_catch_coherence_flag"]],
        on="team", how="left",
    )
    return df


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
    # Combine-athleticism tier (Addendum 4, Part 3) - joined via pfr_id
    # (identify_target_season_rookie_class now carries draft_picks'
    # pfr_player_id / seasonal_rosters' pfr_id directly), NOT via player_id,
    # because target_season's drafted rookies have a placeholder gsis_id
    # (see that function's docstring) that would silently fail to match
    # combine_athletic_scores' player_id-keyed form for nearly the entire
    # drafted class. 'no_data' (not NaN) for any rookie with no combine_data
    # match at all (didn't test, or genuinely absent from the pull), so
    # predict_rookies' scale lookup always resolves.
    athletic = combine_athletic_scores_by_pfr_id(conn)
    target_class = target_class.merge(athletic, on="pfr_id", how="left")
    target_class["athletic_tier"] = target_class["athletic_tier"].fillna("no_data")
    depth_chart = load_depth_chart(target_season)
    rookie_preds = predict_rookies(target_class, baselines, [target_season], depth_chart=depth_chart)

    pg_cols = [c for c in rookie_preds.columns if c.endswith("_pg")]
    rookie_long = rookie_preds.melt(
        id_vars=["player_id", "team", "position", "season", "rookie_tier", "round_bucket",
                 "qb_sleeper_play_prob", "athletic_tier"],
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
    # role_discount_applied is a veteran_model-only concept (see
    # apply_depth_chart_gating) - rookie_rule rows are already
    # low_confidence=True by construction and never pass through that
    # function, so this is always False here, not a claim about the
    # rookie's actual role.
    rookie_long["role_discount_applied"] = False

    # Compose the veteran reframed receiving shares into real per-game
    # rates now that the rookie path exists: rookie receiving predictions
    # enter the share-sum denominator as implied shares (Phase 2 of the
    # consensus-gap work - the user-diagnosed Robinson/Tate case, where a
    # 1st-round rookie's incoming target share must squeeze the veterans).
    rookie_receiving = rookie_long[rookie_long["stat"] == "receiving_yards"][["team", "pred_pg"]]
    vet = _compose_reframed_receiving_predictions(vet, resid, rookie_receiving=rookie_receiving)

    combined = pd.concat([vet, rookie_long], ignore_index=True, sort=False)
    combined = add_team_pass_catch_coherence_flag(combined, depth_chart)
    return combined


OUTPUT_DIR = os.path.join(REPO_ROOT, "output")

OUTPUT_COLUMNS = [
    "player_id", "display_name", "team", "position", "stat",
    "pred_pg", "pred_pg_low", "pred_pg_high",
    "source", "low_confidence", "rookie_tier", "interval_low_n_flag", "season",
    # Phase 6 additions:
    "team_changed", "roster_status",            # Task 1 - team reassignment transparency
    "depth_rank", "role", "depth_chart_status",  # Task 2/3 - curated depth chart + gating
    "role_discount_applied",  # curated committee/backup volume discount - see ROLE_VOLUME_DISCOUNT
    "qb_sleeper_play_prob",  # rookie QB survivorship-bias correction - NaN for non-QB/veteran rows
    "athletic_tier",  # Addendum 4 - combine-athleticism scale tier; NaN for veteran_model rows
    "team_pass_catch_ratio", "team_pass_catch_coherence_flag",  # diagnostic-only, see add_team_pass_catch_coherence_flag
    "receiving_share_capped",  # joint/multi-output Phase A - see _compose_reframed_receiving_predictions
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
