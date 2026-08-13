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
from src.projection.depth_history import attach_availability_depth_rank, attach_depth_rank
from src.projection.features import build_player_season_features, TARGET_STATS, OC_METRICS
from src.projection.ol_quality import team_season_ol_quality
from src.projection.transitions import (
    ALL_FEATURES, TEAM_FEATURES, REFRAMED_SHARE_STATS,
    RECEIVING_SHARE_SUM_CAP, receiving_share_scale, SEASON_GAMES,
)
from src.projection.corrections import elite_shrinkage_adjustment
from src.projection.rookies import (
    build_rookie_dataset, fit_rookie_baselines, predict_rookies,
    identify_target_season_rookie_class,
    team_vacated_opportunity, rookie_interval_ratios, combine_athletic_scores_by_pfr_id,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
INTERVAL_RESIDUALS_PATH = os.path.join(MODELS_DIR, "interval_residuals.csv")
CORRECTIONS_PATH = os.path.join(MODELS_DIR, "corrections.joblib")
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
# ---------------------------------------------------------------------
# Volume discount (Gate B). Replaces DEEP_BENCH_DISCOUNT = 0.15 and
# ROLE_VOLUME_DISCOUNT = {'committee': 0.4, 'backup': 0.15}, both of which
# were stated, un-tuned judgment calls made when no historical depth chart
# existed to validate against. depth_history.py now supplies one for every
# season, so these are FIT against outcomes rather than asserted.
#
# What was actually wrong with 0.15/0.40: they were season-scale numbers
# multiplying a per-game RATE. The old constants absorbed the availability
# error that Gate A has since moved into projected_games, and the two
# errors cancelled on season totals while both were wrong individually.
#
# The estimand, stated precisely because getting it wrong is what produced
# 0.15: pred_pg is a rate CONDITIONAL ON PLAYING (the rate models train on
# games_played_to > 0), so the multiplier that belongs here is
# sum(actual_pg) / sum(pred_pg) among players at that depth rank WHO
# PLAYED. Availability is a separate question, already answered separately.
#
# Fit leave-one-transition-out, 2021-2025 (4 folds; QB/RB confirmed on the
# wide 2017-2025 window, 8 folds, which the reframed receiving path cannot
# use because its RidgeCV team model has no OL features before 2021):
#
#   QB   rank 1  1.10   rank 2  0.84/0.82   rank 3+ 0.80/0.67   off 1.08
#   RB   rank 1  1.09   rank 2  1.04/1.04   rank 3  0.85/0.84   rank 4-5 0.66/0.76   off 0.83
#   WR   rank 1-3  1.10/1.09/1.01   rank 4  1.11   rank 5  0.89   rank 6+ 0.85   off 0.78
#   TE   rank 1  1.06   rank 2  0.94   rank 3  0.94   rank 4-5 0.91   off 0.84
#
# The headline: conditional on playing, the rate needs almost no discount.
# Nothing fits below 0.66, against shipped values of 0.15. A WR4 (Troy
# Franklin's bucket) fits at 1.11 - the model was UNDER-predicting him
# while he was being multiplied by 0.15. An off-chart QB fits at ~1.08,
# which is not a paradox: an off-chart QB who plays is playing precisely
# because he became the starter, so his rate is a starter's rate. All of
# the suppression those players need is availability, and projected_games
# now carries it.
#
# Two deliberate departures from the raw fit:
#   1. CAPPED AT 1.0. Every position fits rank 1 at ~1.06-1.12, i.e. the
#      model under-predicts starters by ~10%. That is a real finding, but
#      it is a model-calibration issue, not a depth-chart one, and
#      silently inflating every starter by 10% inside a function called
#      "depth chart gating" is exactly the kind of hidden change this
#      project's conventions forbid. A discount discounts. Reported as a
#      follow-up instead - corrections.py is where a bias term belongs.
#   2. Shaded toward MORE discount where the per-fold spread is wide or
#      n < 30 (QB off: point estimate ~1.05 but per-fold 0.80-1.68 on
#      n=29, so 0.85; QB 3+: 0.67-0.80 on n=14-28, so 0.70).
#
# Keyed on the nflverse preseason rank rather than the curated `role`
# because that is what was fit, and because rank resolves the gradient the
# role tiers cannot: 'deep_bench' pooled a real WR4 with a camp-body WR9
# and gave both 0.15. The curated chart remains authoritative for
# membership, team assignment, and the displayed `role`.
DEPTH_RATE_LADDER = {
    "QB": {1: 1.00, 2: 0.85},
    "RB": {1: 1.00, 2: 1.00, 3: 0.85},
    "WR": {1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00, 5: 0.90},
    "TE": {1: 1.00, 2: 0.95, 3: 0.95},
}
# Applied to a player on the chart but deeper than the ladder's last rung.
DEPTH_RATE_DEEP = {"QB": 0.70, "RB": 0.70, "WR": 0.85, "TE": 0.90}
# Applied to a player absent from the preseason chart entirely.
DEPTH_RATE_OFF_CHART = {"QB": 0.85, "RB": 0.80, "WR": 0.80, "TE": 0.85}


def depth_rate_factor(position, rank):
    """The Gate B volume multiplier for one (position, preseason rank).
    NaN rank = off the chart. Unknown position falls back to 1.0 (no
    discount) rather than to a guess: this ladder was fit per position and
    has nothing to say about one it never saw."""
    if position not in DEPTH_RATE_LADDER:
        return 1.0
    if rank is None or (isinstance(rank, float) and np.isnan(rank)):
        return DEPTH_RATE_OFF_CHART[position]
    rung = DEPTH_RATE_LADDER[position]
    return rung.get(int(rank), DEPTH_RATE_DEEP[position])


# Retained only so an external reader of this module still finds the old
# names next to an explanation of what replaced them. Nothing reads these.
DEEP_BENCH_DISCOUNT = 0.15  # superseded by DEPTH_RATE_OFF_CHART

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
# The reasoning above was sound about the DIRECTION and wrong about the
# magnitude, and Gate B's fit shows exactly why. The Sleeper comparisons
# quoted here are per-game on our side but season-total/18 on Sleeper's
# (sleeper_gp is 18.0 for all 712 matched rows - a bookkeeping default, see
# sleeper_compare.py), so every one of those gaps is inflated by the ratio
# of the player's real games to 18. Reading "28.2 vs 1.8 attempts/g" off
# that mismatch and picking 0.15 put a season-scale correction on a
# per-game rate. Fit against outcomes, a curated 'committee' RB needs 1.04
# (no discount at all) and a 'backup' TE 0.94, not 0.40 and 0.15.
#
# The observation that a clipboard QB gets near-zero live volume is still
# true - it is just a statement about GAMES, not about his rate in a game
# he plays, and projected_games is where it now lives (Gate A: off-chart
# players actually play 1.2-2.5 games; the old model said 3.9-5.2).
ROLE_VOLUME_DISCOUNT = {"committee": 0.4, "backup": 0.15}  # superseded by DEPTH_RATE_LADDER


def load_availability_models():
    """Per-position games-played models (Phase 11). Returns {} - a real
    "no availability estimate is produced" state, not an error - when the
    files predate this feature, so an older models/ directory still
    predicts rather than failing."""
    out = {}
    for position in TARGET_STATS:
        path = os.path.join(MODELS_DIR, f"{position}_games.joblib")
        if os.path.exists(path):
            out[position] = joblib.load(path)
    return out


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


def load_corrections():
    """Elite-shrinkage correction parameters fit by train.py (see
    corrections.py). Returns {} - a real "no correction is applied" state,
    not an error - when the file predates this feature, so an older
    models/ directory still predicts rather than failing."""
    if not os.path.exists(CORRECTIONS_PATH):
        return {}
    return joblib.load(CORRECTIONS_PATH)


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

# Damping on the incumbent vacancy boost (see
# apply_incumbent_vacancy_boost), per opportunity type: the fraction of a
# team's NET vacated opportunity that its returning players actually
# absorb. Both values are MEASURED, not assumed - fit by grid search over
# every 2017->2025 transition against what returning players' shares
# actually did the following season:
#   targets alpha=0.5  (MAE -1.85% vs carry-forward, consistency 2.06,
#                       6/9 seasons positive, all 5 most recent positive)
#   carries alpha=1.0  (MAE -13.6%, consistency 6.33, 9/9 positive)
# Full proportional redistribution (alpha=1.0) is what the carry data
# wants and clearly OVERSHOOTS for targets - a departing back's carries
# have almost nowhere else to go, while vacated targets are far more
# readily absorbed by rookies, practice-squad callups, and scheme change.
# Calibration at the target value is close to exact in the buckets that
# matter: modelled 1.057/1.122/1.242x growth vs actual 1.042/1.131/1.247x
# across rising net-vacancy buckets.
#
# CARRIES ARE NEVERTHELESS SHIPPED AT 0.0 - disabled, not deleted. The
# historical evidence above is the strongest in this whole module, but
# applying it live made two of three RB metrics WORSE (vs-consensus
# correlation 0.848 -> 0.833, mean abs delta 2.367 -> 2.444; bias did
# improve, -1.01 -> -0.75) and split the boosted backs 11-closer/9-further
# essentially at random. The reason is visible in who moved: Jacobs, Hall,
# Judkins and McCaffrey were ALREADY projected well above consensus, so a
# correctly-measured share boost stacks on top of a separate, pre-existing
# RB level bias and amplifies it. Both things are true at once - the
# vacancy signal is real AND our lead-back level is too high - and the
# second has to be fixed before the first can be turned on without doing
# net harm. Flip this to 1.0 once that RB level bias is addressed; the
# code path is otherwise identical and already exercised by the target
# side.
INCUMBENT_VACANCY_ALPHA = {"target": 0.5, "carry": 0.0}

# Damping on the TEAM-CHANGER vacancy scale, the arrival-side sibling of
# INCUMBENT_VACANCY_ALPHA. Also measured over every 2017->2025 transition,
# scored on what arrivals' shares actually did at their new team:
#   targets alpha=0.35 (MAE 0.05460 vs 0.05868 carrying the old share
#                       forward, -6.75%, consistency 6.13, 9/9 positive)
#   carries alpha=0.25 (0.19082 vs 0.19864, -4.19%, consistency 2.33, 6/9)
# The un-damped, un-netted scale this replaces was measurably WORSE than
# doing nothing at all - 0.07758 vs 0.05868 for targets (+32%) and
# 0.32697 vs 0.19864 for carries (+65%). Both halves of the fix are load
# bearing and neither works alone: netting at full strength is still
# worse than naive (0.06476), and damping the un-netted scale never beats
# naive at any alpha. Net first, then shrink toward carrying the share
# forward.
TEAM_CHANGE_VACANCY_ALPHA = {"target": 0.35, "carry": 0.25}
# Net vacancy is clipped before use, and the resulting scale capped, so a
# freak roster teardown can't produce an unbounded multiplier out of a
# region with no supporting data (observed net vacancy essentially never
# exceeds ~0.5).
INCUMBENT_VACANCY_NET_CLIP = 0.75
INCUMBENT_VACANCY_SCALE_CAP = 2.0


def _incoming_volume_share(df, changed):
    """Per destination team, the source-season carry/target volume walking
    IN, expressed on team_vacated_opportunity's own raw-count basis (the
    arrivals' prior totals over the destination team's prior total) so the
    two are directly subtractable.

    Shared by both vacancy adjustments so they cannot drift apart: the
    incumbent boost subtracts this whole quantity (arrivals absorb the
    room, so returners shouldn't be credited with it), while the
    team-changer scale subtracts it MINUS the player's own contribution
    (each arrival should see the room net of its COMPETITORS, never net
    of itself). Assumes df["team"] is still the source-season team, i.e.
    it must be called before the team reassignment at the end of
    reassign_team_changers."""
    prev_team = df.groupby("team")[["carries", "targets"]].sum()
    incoming = df[changed].groupby("team_target")[["carries", "targets"]].sum()
    out = {}
    for col in ["carries", "targets"]:
        share = (incoming[col] / prev_team[col]).replace([np.inf, -np.inf], np.nan)
        own = df[col] / df["team_target"].map(prev_team[col]).replace(0, np.nan)
        out[col] = (share.fillna(0.0), own.fillna(0.0))
    return out


def drop_players_absent_from_target_season(conn, df, depth_chart, target_season):
    """Drop players who have NO target_season roster row at all AND are
    not vouched for by the curated depth chart - players who have left the
    league, not players having a quiet season.

    The case that surfaced this: Philip Rivers, five years retired, came
    out of retirement for Indianapolis in weeks 15-17 of 2025 (28/37/32
    attempts, 544 yards, 4 TDs) during a QB emergency. That is a real
    event and the data recording it is correct - an earlier pass through
    this project wrongly wrote it off as an upstream nflverse ID
    mislabeling, which it is not. But a genuine 3-game emergency stint at
    age 44 is one of the great outliers in league history, not the basis
    for a 2026 projection, and he duly showed up in the deliverable with
    a 37.3 passing-yards-per-game line on a team he is not on.

    The leak is structural, not Rivers-specific, and this fixes the
    class: reassign_team_changers' `no_info` branch keeps a player's OLD
    team when they cannot be found in target_season's roster, which is
    the right call for a crosswalk gap but silently converts "out of the
    league" into "still on last year's team." 65 players reached the 2026
    output that way.

    Two guards keep this from becoming its own silent-deletion bug - the
    failure mode this project has already been burned by once (see
    project_veterans' docstring on the rookie-filter bug):

    1. The curated depth chart WINS over a missing roster row. A player
       our own hand research affirmatively places on a 2026 roster is
       kept even with no roster row, because the human signal is stronger
       than the absence of a machine one - the same precedence already
       used when a curated starter overrides the vacancy heuristic. This
       is load-bearing: Deebo Samuel and Stefon Diggs are both curated
       starters with no 2026 roster row, and a blanket rule would have
       deleted two legitimate starter projections.
    2. Every dropped player is PRINTED, with the count and the most
       significant names. A drop that is announced is auditable; a drop
       that is silent is the bug.

    Runs before the models, not as an output filter, so a departed player
    also stops consuming his old team's receiving-share budget."""
    if "roster_status" not in df.columns:
        return df
    absent = df["roster_status"].isna()
    if not absent.any():
        return df

    curated_ids = set()
    if not depth_chart.empty:
        curated_ids = set(depth_chart["gsis_id"].dropna())
    to_drop = absent & ~df["player_id"].isin(curated_ids)
    if not to_drop.any():
        return df

    names = pd.read_sql("select gsis_id as player_id, display_name from players", conn)
    names = names.drop_duplicates("player_id").set_index("player_id")["display_name"]

    dropped = df[to_drop].copy()
    dropped["display_name"] = dropped["player_id"].map(names).fillna(dropped["player_id"])
    kept_anyway = df[absent & df["player_id"].isin(curated_ids)].copy()
    kept_anyway["display_name"] = kept_anyway["player_id"].map(names).fillna(kept_anyway["player_id"])

    print(
        f"Dropped {len(dropped)} player(s) with no {target_season} roster row and no curated "
        f"depth-chart entry (out of the league, not merely low-usage):"
    )
    # Ranked by prior-season receiving/rushing volume, not games - the
    # ones worth a human second look are the ones who were PRODUCTIVE.
    dropped["_vol"] = dropped[["receiving_yards", "rushing_yards", "passing_yards"]].fillna(0).max(axis=1)
    for _, r in dropped.nlargest(min(10, len(dropped)), "_vol").iterrows():
        print(f"    {r['display_name']} ({r['position']}, last seen {r['team']}, "
              f"{r['games_played']:.0f} games in {target_season - 1})")
    if len(dropped) > 10:
        print(f"    ... and {len(dropped) - 10} more")
    if not kept_anyway.empty:
        print(f"  KEPT {len(kept_anyway)} player(s) with no roster row but a curated depth-chart entry "
              f"(human research outranks a missing roster row): {', '.join(sorted(kept_anyway['display_name']))}")
    return df[~to_drop]


def apply_incumbent_vacancy_boost(conn, df, target_season, depth_chart, changed):
    """Credit a team's RETURNING players with the opportunity its departed
    players left behind.

    The gap this closes (found by the project owner asking who absorbs
    Green Bay's work after Romeo Doubs and Dontayvion Wicks left):
    reassign_team_changers' vacated-opportunity scaling only ever fires
    for players who CHANGED TEAMS. A player who stays put has his share
    features read straight off the source season - a season in which the
    now-departed teammates were still there taking their cut - so the
    model gives their vacated volume to nobody at all.

    League-wide this hides, because most teams replace departures from
    outside and those arrivals DO get boosted: across the 2026 slate, a
    team's vacated target share correlates POSITIVELY (+0.48) with how
    much of its passing offense we allocate. It only surfaces on a team
    that replaces from within. Green Bay lost 132 of 462 targets (28.6%)
    and re-signed nobody of note, and ended up allocating 93.1% of its
    projected team passing yards - second-lowest in the league against a
    111.5% mean - with three of the five worst remaining WR consensus
    gaps (Reed, Golden, Watson) all on that one roster.

    Mechanism, with each piece measured rather than assumed:

    1. NET vacancy, not gross. `team_vacated_opportunity` measures volume
       that LEFT; from it we subtract the volume walking IN (the
       source-season carries/targets of players joining this team, over
       the team's own source-season total - the same raw-count basis
       vacated_* uses, so the two are subtractable). Without this a team
       that lost three starters and signed three would boost its
       incumbents AND its arrivals for the same opening, double-counting
       the room.
    2. Proportional redistribution, damped: scale = 1 + alpha * v_net /
       (1 - v_net), with alpha per opportunity type - see
       INCUMBENT_VACANCY_ALPHA for the fitted values and the evidence.
    3. Depth-chart gated, upward only. Only curated `starter`/`committee`
       incumbents receive it (BOOST_ELIGIBLE_ROLES - the same guard that
       stopped a confirmed backup from out-projecting his own starter in
       the Phase-6 Gainwell bug); everyone else keeps their share
       untouched. The boost never reduces a share, so a low-vacancy team
       is a strict no-op rather than a penalty - incumbent shares DO decay
       on stable teams (observed 0.907x median), but that is ordinary
       regression the models already learn, and re-applying it here would
       double-count it.

    LIMITATIONS, stated: (a) incoming ROOKIES are not subtracted in step
    1 - they have no prior NFL volume to measure and their predictions
    don't exist yet at this point in the pipeline; the share-sum cap at
    composition time is the backstop, and the team-allocation diagnostic
    is how it gets checked. (b) Requires a curated depth chart, so this
    is a no-op for any season other than 2026 - deliberately, since
    without role gating it would inflate every bench player on a
    high-turnover team. (c) Like every other adjustment in this function,
    it is applied at PREDICT time only and so is not exercised by
    backtest.py; its evidence is the historical validation above, not the
    2025 holdout."""
    if depth_chart.empty:
        return df

    incumbent = ~changed
    if not incumbent.any():
        return df

    vacated = team_vacated_opportunity(conn, [target_season])
    vacated = vacated[vacated["season"] == target_season].set_index("team")
    if vacated.empty:
        return df

    # Volume arriving from elsewhere, on team_vacated_opportunity's own
    # raw-count basis. Shared with the team-changer scale via
    # _incoming_volume_share so the two vacancy adjustments can never
    # disagree about how much room the arrivals are taking; the incumbent
    # side subtracts the WHOLE incoming share (arrivals absorb the room),
    # the arrival side subtracts it net of the player's own contribution.
    incoming = _incoming_volume_share(df, changed)
    incoming_carry = incoming["carries"][0]
    incoming_target = incoming["targets"][0]

    role_lookup = depth_chart[["position", "gsis_id", "role"]].rename(columns={"gsis_id": "player_id"})
    role_lookup = role_lookup.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
    roles = df[["player_id", "position"]].merge(role_lookup, on=["player_id", "position"], how="left")["role"]
    # QB rows are excluded, for the same reason predict_rookies needs
    # vacated_attempts_share rather than vacated_target_share: receiving-
    # corps and backfield turnover say nothing about a QB's own workload,
    # and scaling a quarterback's carry_share by how many RB carries left
    # the building is a category error, not a small one. It is also
    # simply unvalidated - the historical fit behind
    # INCUMBENT_VACANCY_ALPHA covers RB/WR/TE only.
    boostable_position = df["position"].isin(["RB", "WR", "TE"]).to_numpy()
    eligible = incumbent & roles.isin(BOOST_ELIGIBLE_ROLES).to_numpy() & boostable_position
    if not eligible.any():
        return df

    def _scale(vac_col, incoming_share, kind):
        gross = df["team_target"].map(vacated[vac_col])
        absorbed = df["team_target"].map(incoming_share).fillna(0.0)
        v_net = (gross - absorbed).fillna(0.0).clip(lower=0.0, upper=INCUMBENT_VACANCY_NET_CLIP)
        lever = v_net / (1.0 - v_net)
        s = 1.0 + INCUMBENT_VACANCY_ALPHA[kind] * lever
        return s.clip(upper=INCUMBENT_VACANCY_SCALE_CAP)

    carry_scale = _scale("vacated_carry_share", incoming_carry, "carry")
    target_scale = _scale("vacated_target_share", incoming_target, "target")

    for c in ["carry_share", "rz_carry_share"]:
        df.loc[eligible, c] = (df.loc[eligible, c] * carry_scale[eligible]).clip(upper=1.0)
    for c in ["target_share", "rz_target_share"]:
        df.loc[eligible, c] = (df.loc[eligible, c] * target_scale[eligible]).clip(upper=1.0)

    # snap_pct follows the position's primary opportunity type, matching
    # reassign_team_changers' own treatment.
    snap_scale = pd.Series(1.0, index=df.index)
    is_rb = df["position"] == "RB"
    is_recv = df["position"].isin(["WR", "TE"])
    snap_scale[is_rb] = carry_scale[is_rb]
    snap_scale[is_recv] = target_scale[is_recv]
    df.loc[eligible, "snap_pct"] = (df.loc[eligible, "snap_pct"] * snap_scale[eligible]).clip(upper=1.0)

    return df


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

    # The curated depth chart OVERRIDES seasonal_rosters for team
    # assignment, and every disagreement is printed.
    #
    # Why curated wins: seasonal_rosters is a cached upstream snapshot
    # that lags real transactions by days-to-weeks in the preseason, and
    # nothing in this pipeline can tell "the roster is right and the CSV
    # is stale" from "the CSV is right and the roster hasn't caught up."
    # The curated table is the surface a human actually edits, so if it
    # were silently outranked by a stale snapshot, correcting a player's
    # team by hand would appear to do nothing - the worst possible
    # failure mode for a manually-maintained override. Deebo Samuel
    # (re-signed SF) and Stefon Diggs (signed WAS) were both being
    # projected on their 2025 teams for exactly this reason.
    #
    # The safety valve is the printed reconciliation below: a stale
    # CURATED row is now the thing that can go wrong, so every case where
    # the two sources disagree is surfaced by name for review rather than
    # resolved silently in either direction.
    if not depth_chart.empty:
        curated_team = (
            depth_chart.dropna(subset=["gsis_id"])
            .drop_duplicates(subset=["gsis_id"])
            .set_index("gsis_id")["team"]
        )
        curated = df["player_id"].map(curated_team)
        disagree = curated.notna() & df["team_target"].notna() & (curated != df["team_target"])
        filled = curated.notna() & df["team_target"].isna()
        if disagree.any() or filled.any():
            names = pd.read_sql("select gsis_id as player_id, display_name from players", conn)
            names = names.drop_duplicates("player_id").set_index("player_id")["display_name"]
            if disagree.any():
                print(f"Curated depth chart OVERRODE seasonal_rosters on team for "
                      f"{int(disagree.sum())} player(s) - verify the CSV is not the stale side:")
                for _, r in df[disagree].iterrows():
                    print(f"    {names.get(r['player_id'], r['player_id'])}: curated "
                          f"{curated[r.name]} vs roster {r['team_target']}")
            if filled.any():
                print(f"Curated depth chart SUPPLIED a team for {int(filled.sum())} player(s) absent "
                      f"from the {target_season} roster snapshot: "
                      f"{', '.join(sorted(names.get(p, p) for p in df.loc[filled, 'player_id']))}")
        df.loc[curated.notna(), "team_target"] = curated[curated.notna()]

    no_info = df["team_target"].isna()
    if no_info.any():
        # Not found in target_season's roster at all and not curated
        # (retired, out of league, or a crosswalk gap) - keep the old team
        # rather than inventing one. drop_players_absent_from_target_season
        # is what decides whether such a player stays in the output.
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

        # Net the vacancy across COMPETING ARRIVALS, then damp toward
        # carrying the player's own share forward. Before this, every
        # arrival was handed the team's ENTIRE vacancy independently -
        # Washington vacated 52.3% of its targets and so awarded a 2.18x
        # share boost to Stefon Diggs, Chig Okonkwo AND Rachaad White at
        # once, inflating Diggs (0.185 -> 0.403) past the incumbent
        # McLaurin and inverting the team's pecking order. It is the same
        # bug this function's own docstring already describes for the
        # Gainwell case - "no check on whether another player was already
        # absorbing that vacated opportunity" - which the role gate below
        # only partly contained, since a curated starter passes it.
        #
        # `others_incoming` deliberately excludes the player's own volume:
        # an arrival should see the room net of its competitors, never net
        # of itself. See TEAM_CHANGE_VACANCY_ALPHA for the measured
        # damping and for how badly the un-netted version scored.
        incoming = _incoming_volume_share(df, changed)
        scales = {}
        for col, vac_col, kind, league_avg in [
            ("carries", "vacated_carry_share", "carry", league_avg_carry_vac),
            ("targets", "vacated_target_share", "target", league_avg_target_vac),
        ]:
            share_by_team, own = incoming[col]
            others = (df["team_target"].map(share_by_team).fillna(0.0) - own).clip(lower=0.0)
            v_net = (df[vac_col] - others).clip(lower=0.0)
            raw = (v_net / league_avg).clip(*TEAM_CHANGE_SHARE_CLIP).fillna(1.0)
            scales[kind] = 1.0 + TEAM_CHANGE_VACANCY_ALPHA[kind] * (raw - 1.0)
        carry_scale, target_scale = scales["carry"], scales["target"]

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

    # Incumbents (everyone the block above did NOT touch) get the other
    # half of the same idea: credit for the opportunity their departing
    # teammates left behind. Runs here, while df["team"] is still the
    # source-season team and `changed` is available to net out arrivals.
    df = apply_incumbent_vacancy_boost(conn, df, target_season, depth_chart, changed)

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
        df["role_discount_factor"] = 1.0
        return df
    if "nfl_depth_rank" not in df.columns:
        raise ValueError(
            "apply_depth_chart_gating needs nfl_depth_rank (Gate B) - call "
            "depth_history.attach_depth_rank(df, target_season) first. Defaulting "
            "it to NaN would silently apply the off-chart factor to every player.")

    dc = depth_chart[["position", "gsis_id", "depth_rank", "role"]].rename(columns={"gsis_id": "player_id"})
    dc = dc.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id", "position"])
    df = df.merge(dc, on=["player_id", "position"], how="left")

    matched = df["depth_rank"].notna()
    df["depth_chart_status"] = np.where(matched, "curated", "deep_bench_discounted")

    # The net multiplier this row received, recorded as it is applied.
    # Consumed by _compose_reframed_receiving_predictions so the Phase-7
    # elite-shrinkage correction (an ADDITIVE yards/game term, fit on
    # undiscounted out-of-sample residuals) can be scaled by the same
    # factor the rest of the row was: without it, a discounted player with
    # an elite season-N rate would have a full-size bonus added on top of
    # a 0.15x-ed prediction, quietly undoing the discount.
    #
    # Gate B: the factor comes from the nflverse preseason rank via
    # DEPTH_RATE_LADDER, not from the curated `role`. One lookup now covers
    # what used to be two separate mechanisms (DEEP_BENCH_DISCOUNT for rows
    # off the curated table, ROLE_VOLUME_DISCOUNT for curated
    # committee/backup rows), which is why `role_discount_applied` could
    # previously be False on a row that had just been multiplied by 0.15.
    df["role_discount_factor"] = [
        depth_rate_factor(p, r) for p, r in zip(df["position"], df["nfl_depth_rank"])
    ]

    discounted = df["role_discount_factor"] < 1.0
    for col in ["pred_pg", "pred_pg_low", "pred_pg_high"]:
        df[col] = df[col] * df["role_discount_factor"]
    # low_confidence tracks the CURATED table, not the new factor: a WR4 now
    # keeps a full-size rate (fit 1.11), but "the hand-verified table does
    # not carry him" is still the honest confidence statement about him, and
    # weakening it would quietly drop ~240 players out of the flag that
    # tells a reader to check them.
    df.loc[~matched, "low_confidence"] = True
    df.loc[~matched, "role"] = "deep_bench"
    df.loc[discounted, "low_confidence"] = True
    # Now means exactly "this row's numbers were scaled down", for every
    # path. role_discount_factor beside it says by how much.
    df["role_discount_applied"] = discounted

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
    base = drop_players_absent_from_target_season(conn, base, depth_chart, target_season)

    # Availability (Phase 11): a per-game rate alone can't express season
    # value - two players at the same rate are worth very different
    # amounts if one plays 8 games and the other 16, which is exactly the
    # Mike Evans / Deebo Samuel case. Measured directly on the 2024->2025
    # holdout (backtest_season_totals, re-measured after Gate A): rate x a
    # fixed 17 games is WORSE than naive carry-forward at predicting season
    # totals (WR 252.3 vs 170.5 yards MAE, TE 152.8 vs 110.9, RB 295.1 vs
    # 191.3, QB 1420.4 vs 780.0); rate x predicted games beats both (WR
    # 151.8, TE 90.7, RB 172.7, QB 615.3). Attached per player here,
    # carried through to the output so the split stays visible rather than
    # being folded into the rate.
    #
    # The model reads target_season's preseason depth chart (Gate A) - see
    # depth_history.py and train.fit_availability. Attached here, on the
    # post-reassign_team_changers frame, so a player who changed teams is
    # looked up on his NEW team's chart, which is the whole point: Deebo at
    # SF WR-something and Deebo off WAS's chart are different availability
    # predictions.
    #
    # Note this is the nflverse chart, NOT the curated
    # src/depth_chart/starters_2026.csv one used for gating below. The two
    # are deliberately separate: the curated file is hand-verified and
    # authoritative for membership/team/role, but exists only for 2026, so
    # it can never be a trained-on feature. The nflverse chart exists for
    # every season, which is the only reason the availability model can be
    # honestly held out on it.
    avail_models = load_availability_models()
    base = base.copy()
    base = attach_availability_depth_rank(base, target_season, conn=conn)
    # The untruncated rank, for the Gate B volume ladder. Attached here so
    # it rides along into `combined` and is available to gating and to the
    # share renormalization without a second lookup.
    base = attach_depth_rank(base, target_season, conn=conn)
    base["projected_games"] = np.nan
    for position, am in avail_models.items():
        mask = base["position"] == position
        if mask.any():
            # am["features"], not ALL_FEATURES: the availability models
            # carry a wider schema than the rate models (AVAILABILITY_
            # FEATURES), and an older models/ directory predating Gate A
            # still carries the narrower one and must keep working.
            base.loc[mask, "projected_games"] = np.clip(
                am["model"].predict(base.loc[mask, am["features"]]), 0, SEASON_GAMES)
    _warn_availability_chart_disagreement(base, depth_chart)

    rows = []
    for position, stats in TARGET_STATS.items():
        pos_df = base[base["position"] == position]
        if pos_df.empty:
            continue
        X = pos_df[ALL_FEATURES]
        for stat in stats:
            m = models[(position, stat)]
            preds = m["model"].predict(X)
            out = pos_df[["player_id", "team", "position", "team_changed",
                          "roster_status", "projected_games", "nfl_depth_rank"]].copy()
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

# Passing and rushing equivalents. Their absence was a real hole, not a
# judgment that QB/RB curation matters less: the criteria above key on
# targets and receiving yards only, so a discounted QB or RB could not fire
# this tripwire at ALL regardless of volume. 2026 had 26 discounted rows
# with >=150 attempts or >=100 carries in 2025 - Flacco (436 att), Tua
# (417), Dowdle (237 car), Henderson (180) - none of which could be
# surfaced.
#
# These apply only where the discount asserts IRRELEVANCE (net factor
# <= TRIPWIRE_SEVERE_FACTOR, i.e. the 0.15x deep-bench/backup path), not to
# the 0.4x 'committee' path. That exclusion is the whole reason these
# thresholds are usable: at carries >= 100, 14 of 17 discounted RBs are
# curated 'committee', and a committee back carrying 100+ times is not a
# curation error - it is precisely what 'committee' claims. Firing on them
# would bury the real misses under correct curations and train a reader to
# ignore the warning. A 'deep_bench' RB with 155 carries (Kimani Vidal)
# contradicts his own label, and that is the case worth surfacing.
TRIPWIRE_SEASON_ATTEMPTS = 200
TRIPWIRE_ATTEMPTS_PG = 25.0
TRIPWIRE_SEASON_CARRIES = 100
TRIPWIRE_CARRIES_PG = 8.0
TRIPWIRE_SEVERE_FACTOR = 0.2

# The >= 4 game guard the receiving criteria use is too loose for a rushing
# RATE, and the reason is specific to the stat rather than general caution:
# carries per game over a 4-5 game sample is dominated by whether the
# player happened to start those weeks, so it reads as a starter's rate by
# construction and says nothing about his next-season role (Audric Estime,
# 46 carries in 5 games, rates 9.2/g). Half a season is the point where the
# rate reflects a role. Passing keeps the 4-game guard: a QB who threw 189
# times in 5 games was unambiguously starting them, and that IS the signal.
TRIPWIRE_RUSH_MIN_GAMES = 8


def _warn_availability_chart_disagreement(base, depth_chart):
    """Warn (stderr only, never changes a number - same contract as
    _warn_discounted_high_usage below) when the hand-curated depth chart
    lists a player that the nflverse preseason chart does not carry at
    availability depth.

    The two charts are independent and normally agree closely (2026:
    182/183 curated starters, 27/27 committee, 50/54 backups). Where they
    disagree, the nflverse absence is what the availability model sees, and
    it is a genuine signal - a player on nobody's published chart in early
    August really does play fewer games. But a curator who deliberately
    placed a player on a team (Diggs -> WAS, Deebo -> SF; see
    reassign_team_changers) deserves to be told that the games model
    disagrees, rather than discovering a halved projected_games by
    accident. Resolving it is a human call about which chart is right.

    Covers EVERY curated role, not just the ones where a games error costs
    the most fantasy points. A curated backup absent from the nflverse
    chart is the cheaper case in isolation, but it is the same underlying
    event - the two sources disagree about whether this player has a job -
    and suppressing the cheap half would make the warning's silence mean
    "no disagreement above a threshold I chose" instead of "no
    disagreement." Listed worst-role-first so triage is still ordered."""
    if depth_chart.empty:
        return
    curated = depth_chart[depth_chart["role"].notna()]
    off = base[base["player_id"].isin(curated["gsis_id"])
               & base["target_depth_rank"].isna()]
    if off.empty:
        return
    # `base` is the feature frame and carries no display_name; the curated
    # chart does, and is the table a reader would go fix.
    by_id = curated.drop_duplicates("gsis_id").set_index("gsis_id")
    name_of, role_of = by_id["player_name"], by_id["role"]
    order = {"starter": 0, "committee": 1, "backup": 2}
    listed = sorted(
        ((order.get(role_of.get(r.player_id), 9), name_of.get(r.player_id, r.player_id),
          r.team, r.position, role_of.get(r.player_id), r.projected_games)
         for r in off.itertuples()),
        key=lambda x: (x[0], -x[5]))
    names = ", ".join(f"{n} ({t} {p}, curated {role}, {g:.1f} g)"
                      for _, n, t, p, role, g in listed)
    counts = ", ".join(f"{c} {r}" for r, c in
                       sorted(((r, sum(1 for x in listed if x[4] == r))
                               for r in set(x[4] for x in listed)),
                              key=lambda kv: order.get(kv[0], 9)))
    print(f"AVAILABILITY CHART DISAGREEMENT: {len(off)} curated player(s) ({counts}) "
          f"are absent from the nflverse preseason chart, so the games model treated "
          f"them as off-chart: {names}", file=sys.stderr)


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
    usage = base[["player_id", "targets", "targets_pg", "receiving_yards_pg",
                  "attempts", "attempts_pg", "carries", "carries_pg", "games_played"]]
    flagged = (
        discounted[["player_id", "team", "position", "role", "role_discount_factor"]]
        .drop_duplicates("player_id")
        .merge(usage, on="player_id")
    )
    # Per-game thresholds need a minimum sample (>= 4 games) so a 1-3 game
    # blip (e.g. 5 targets in a single spot appearance) doesn't fire; the
    # season-total threshold needs no such guard by construction. 4 keeps
    # genuinely relevant injury-shortened seasons (Tyreek Hill's 4-game
    # 66-ypg 2025) inside the net.
    enough_games = flagged["games_played"] >= 4
    receiving = (
        (flagged["targets"] >= TRIPWIRE_SEASON_TARGETS)
        | (enough_games & (flagged["targets_pg"] >= TRIPWIRE_TARGETS_PG))
        | (enough_games & (flagged["receiving_yards_pg"] >= TRIPWIRE_REC_YPG))
    )
    # Keyed on the net multiplier actually applied, not on role names, so a
    # future role tier is covered by its severity rather than by having
    # been added to a list here.
    severe = flagged["role_discount_factor"] <= TRIPWIRE_SEVERE_FACTOR
    enough_rush_games = flagged["games_played"] >= TRIPWIRE_RUSH_MIN_GAMES
    passing = severe & (
        (flagged["attempts"] >= TRIPWIRE_SEASON_ATTEMPTS)
        | (enough_games & (flagged["attempts_pg"] >= TRIPWIRE_ATTEMPTS_PG))
    )
    rushing = severe & (
        (flagged["carries"] >= TRIPWIRE_SEASON_CARRIES)
        | (enough_rush_games & (flagged["carries_pg"] >= TRIPWIRE_CARRIES_PG))
    )
    flagged = flagged[receiving | passing | rushing].copy()
    if flagged.empty:
        return
    # Which criterion fired, so a reader can tell a 400-attempt QB2 from a
    # 70-target WR without reading the thresholds out of this file.
    flagged["_why"] = np.select(
        [passing.reindex(flagged.index, fill_value=False),
         rushing.reindex(flagged.index, fill_value=False)],
        ["passing", "rushing"], default="receiving")
    flagged["_sort"] = np.where(
        flagged["_why"] == "passing", flagged["attempts"],
        np.where(flagged["_why"] == "rushing", flagged["carries"], flagged["targets"]))
    names = pd.read_sql("SELECT gsis_id AS player_id, display_name FROM players", conn)
    flagged = flagged.merge(names, on="player_id", how="left")
    print(
        f"CURATION TRIPWIRE: {len(flagged)} discounted player(s) had fantasy-relevant "
        f"season-{source_season} usage - verify their rows in the curated depth chart:",
        file=sys.stderr,
    )
    for _, r in flagged.sort_values(["_why", "_sort"], ascending=[True, False]).iterrows():
        if r["_why"] == "passing":
            detail = f"{r['attempts']:.0f} attempts, {r['attempts_pg']:.1f} att/g"
        elif r["_why"] == "rushing":
            detail = f"{r['carries']:.0f} carries, {r['carries_pg']:.1f} car/g"
        else:
            detail = (f"{r['targets']:.0f} targets, {r['targets_pg']:.1f} tgt/g, "
                      f"{r['receiving_yards_pg']:.1f} rec ypg")
        print(
            f"  {r['display_name']} ({r['team']} {r['position']}, role={r['role']}, "
            f"{r['role_discount_factor']:.2f}x): {detail} in {r['games_played']:.0f} games",
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
    # The player's OWN observed season-N receiving rate, carried alongside
    # so the Phase-7 elite-shrinkage correction can key on it (see
    # corrections.py for why the observed rate and not the predicted one).
    observed = base[["player_id", "receiving_yards_pg"]].rename(
        columns={"receiving_yards_pg": "_observed_recv_pg"})
    combined = combined.merge(observed.drop_duplicates("player_id"), on="player_id", how="left")
    # A team with no resolvable TEAM_FEATURES row (the same rare team=NaN
    # gap backtest.py's reframed path already documents) has no
    # team_total_pred to compose with - falls back to 0 rather than
    # NaN-propagating a whole player's row into an unusable prediction;
    # genuinely rare (verified 0 occurrences in the 2026 live run).
    combined["team_total_pred"] = combined["team_total_pred"].fillna(0)
    return combined


# Intermediate columns dropped once composition is done. `role_discount_
# factor` is deliberately NOT among them any more: it is the actual
# multiplier applied to the row, and this project's rule is that a discount
# must be visible in the output table rather than buried in a module. The
# two flags that used to be the only evidence are each partial - `role_
# discount_applied` is False for deep-bench rows despite their 0.15x, and
# `depth_chart_status` says which path fired but not how hard - so a reader
# had to consult two columns and still could not see the magnitude.
_HELPER_COLS = ["team_total_pred", "_observed_recv_pg"]


# The receiving stats that ride along with a receiving_yards correction.
# receiving_tds is included on the same measured evidence as the other two
# (ratio 1.07/1.08) even though it is the noisiest of the three.
ELITE_COMPANION_STATS = ("receptions", "targets", "receiving_tds")


def _propagate_elite_correction(other, reframed, before, adj):
    """Scale a corrected player's companion receiving stats by the same
    proportion his receiving_yards moved - see the call site for why
    proportional is the measured answer.

    Guarded on `before > 0`: a player whose composed yards prediction is
    zero has no defined proportion, and scaling from zero would either
    divide by zero or invent volume from nothing. Those rows keep their
    uncorrected companion stats, which is the honest fallback."""
    moved = adj > 0
    if not moved.any():
        return other
    ratio = pd.Series(
        np.where(before > 0, (before + adj) / np.where(before > 0, before, 1.0), 1.0),
        index=reframed.index,
    )
    key = pd.MultiIndex.from_arrays(
        [reframed.loc[moved, "player_id"], reframed.loc[moved, "position"]])
    per_player = pd.Series(ratio[moved].to_numpy(), index=key)
    per_player = per_player[~per_player.index.duplicated()]

    hit = other["stat"].isin(ELITE_COMPANION_STATS)
    if not hit.any():
        return other
    other = other.copy()
    idx = pd.MultiIndex.from_arrays([other.loc[hit, "player_id"], other.loc[hit, "position"]])
    scale = pd.Series(per_player.reindex(idx).to_numpy(), index=other.index[hit]).fillna(1.0)
    for col in ["pred_pg", "pred_pg_low", "pred_pg_high"]:
        other.loc[hit, col] = other.loc[hit, col] * scale
    return other


def _compose_reframed_receiving_predictions(combined, resid, rookie_receiving=None, corrections=None):
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
        return combined.drop(columns=_HELPER_COLS, errors="ignore")
    reframed = combined[mask].copy()
    other = combined[~mask].drop(columns=_HELPER_COLS, errors="ignore").copy()
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
    # Participation weight for the cap denominator (Gate B). A player
    # projected for 2 games consumes 2/17 of the team's season share
    # budget, not all of it and not an arbitrary 0.15 of it. Missing
    # projected_games (an older models/ directory with no availability
    # model) falls back to 1.0 - the pre-Gate-B behaviour - rather than to
    # 0, which would drop that team's denominator to nothing and disable
    # the cap silently.
    share_df["weight"] = (
        (reframed["projected_games"] / SEASON_GAMES).clip(0, 1).fillna(1.0)
        if "projected_games" in reframed.columns else 1.0
    )
    scale, over_cap = receiving_share_scale(share_df, extra_team_share=extra_team_share)
    reframed["receiving_share_capped"] = over_cap
    reframed["pred_pg"] = reframed["pred_pg"] * scale * reframed["team_total_pred"]

    # Phase 7: additive elite-shrinkage correction, applied AFTER
    # composition (it is fit in rate units on composed out-of-sample
    # residuals) and scaled by the row's own role discount so a
    # discounted player can't be handed an undiscounted bonus. Rows with
    # no observed season-N rate, or a position with no fitted parameters,
    # get exactly 0.0 - see corrections.elite_shrinkage_adjustment.
    reframed["elite_correction_pg"] = 0.0
    if corrections:
        adj = elite_shrinkage_adjustment(
            reframed["position"], reframed["_observed_recv_pg"], corrections)
        adj = adj * reframed["role_discount_factor"].fillna(1.0).to_numpy()
        reframed["elite_correction_pg"] = adj
        before = reframed["pred_pg"].to_numpy()
        reframed["pred_pg"] = before + adj
        # Carry the same proportional bump to the player's OTHER receiving
        # stats. Without this the correction adds yards to a tight end while
        # leaving his receptions and targets untouched, so the shipped row
        # says he gains 8 yards a game on exactly the same catches - an
        # internally inconsistent player in fantasy_points_<season>.csv,
        # where receptions are scored separately (0.5 each in this league).
        #
        # Proportional, and specifically NOT a separately-fit correction,
        # because the elite under-prediction is a uniform VOLUME effect
        # rather than a yards-per-catch one. Measured LOO over 2021-2025 on
        # the same above-knot cohort, actual/predicted per stat:
        #   WR (n=111): yards 1.04  receptions 1.06  targets 1.05  rec TDs 1.07
        #   TE (n=23):  yards 1.05  receptions 1.05  targets 1.05  rec TDs 1.08
        # The ratios are the same stat to stat, so holding the player's own
        # yards-per-reception fixed and scaling volume is what the data
        # says - and it needs no new fitted parameters, which matters given
        # the TE fit already rests on 23 rows.
        other = _propagate_elite_correction(other, reframed, before, adj)
    reframed = reframed.drop(columns=_HELPER_COLS, errors="ignore")

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
    # Rookie availability (Phase 11): the veteran games model needs a
    # season-N feature row, which an incoming rookie by definition does
    # not have. Uses the historical mean games played by first-year
    # players at the same position instead (career_year == 0 rows, added
    # in Phase 5) - the same "bucket average" spirit as the rest of the
    # rookie path, and a real estimate rather than a blank or a silently
    # assumed full season. Deliberately NOT differentiated by draft round.
    #
    # Left unchanged by Gate A, and this is a known gap rather than a
    # judgment that it doesn't matter: the reason given for the flat prior
    # used to be that the veteran availability signal was itself weak
    # (+5-6% over carry-forward), and that is no longer true (+25-28%, see
    # train.fit_availability). Rookies DO appear on the target season's
    # preseason chart, so the same feature is available to them - but they
    # have no season-N feature row, so it cannot simply be routed through
    # the veteran model, and a rookie-specific games model is its own
    # modeling step with its own validation. Flagged, not silently folded
    # into this change.
    rookie_games = (
        feat[feat["career_year"] == 0].groupby("position")["games_played"].mean()
        if "career_year" in feat.columns else pd.Series(dtype=float)
    )
    rookie_long["projected_games"] = rookie_long["position"].map(rookie_games)
    # role_discount_applied is a veteran_model-only concept (see
    # apply_depth_chart_gating) - rookie_rule rows are already
    # low_confidence=True by construction and never pass through that
    # function, so this is always False here, not a claim about the
    # rookie's actual role.
    rookie_long["role_discount_applied"] = False
    # Likewise 1.0 = "no discount was applied to this row," which is true,
    # rather than a claim that the rookie was evaluated and found
    # undiscountable. Set explicitly so the column is never NaN for a whole
    # source, which would read as "unknown" and invite a fillna somewhere
    # downstream.
    rookie_long["role_discount_factor"] = 1.0

    # Compose the veteran reframed receiving shares into real per-game
    # rates now that the rookie path exists: rookie receiving predictions
    # enter the share-sum denominator as implied shares (Phase 2 of the
    # consensus-gap work - the user-diagnosed Robinson/Tate case, where a
    # 1st-round rookie's incoming target share must squeeze the veterans).
    rookie_receiving = rookie_long[rookie_long["stat"] == "receiving_yards"][["team", "pred_pg"]]
    vet = _compose_reframed_receiving_predictions(
        vet, resid, rookie_receiving=rookie_receiving, corrections=load_corrections())

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
    "nfl_depth_rank",         # Gate B - untruncated nflverse preseason rank, the ladder's input
    "role_discount_factor",   # the multiplier actually applied (1.0 = none); covers BOTH the
                              # deep-bench and committee/backup paths, which the two flags above
                              # only cover between them - see _HELPER_COLS
    "qb_sleeper_play_prob",  # rookie QB survivorship-bias correction - NaN for non-QB/veteran rows
    "athletic_tier",  # Addendum 4 - combine-athleticism scale tier; NaN for veteran_model rows
    "team_pass_catch_ratio", "team_pass_catch_coherence_flag",  # diagnostic-only, see add_team_pass_catch_coherence_flag
    "receiving_share_capped",  # joint/multi-output Phase A - see _compose_reframed_receiving_predictions
    "elite_correction_pg",  # Phase 7 additive elite-shrinkage correction, in yards/game - see corrections.py
    "projected_games",  # Phase 11 availability estimate; season value = pred_pg x this. See train.fit_availability.
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
