"""Season N -> season N+1 transition-pair construction, shared by
train.py and backtest.py.

Train/predict framing (explicit, per the spec's ask to think carefully
about this): each training row is (player's season N feature vector) ->
(player's season N+1 per-game rate). This is a genuine next-season
projection task, not same-season leakage - season N's features (shares,
snap %, OL quality, OC tendencies) are all fully known by the end of
season N, and the label is strictly season N+1.

One real limitation, documented rather than hidden: season N's OWN
opportunity/scheme features (oc_tendency_profiles, OL quality) are used as
the proxy for season N+1 conditions, because season N+1's actual observed
tendencies obviously don't exist yet at prediction time for a genuinely
future season. This mirrors what a real preseason projection has to do
(assume the most recently observed team context persists), and is exactly
why `oc_tendency_profiles.inherited_*` exists for a genuinely new OC - but
wiring that in for a not-yet-played season is Phase 5's job (choosing which
season's row to feed the trained model here), not this module's.
"""
import numpy as np
import pandas as pd

from src.projection.depth_history import (
    AVAILABILITY_DEPTH_FEATURE, attach_availability_depth_rank,
)
from src.projection.features import FEATURE_COLS, TARGET_STATS, OC_METRICS

EXTRA_FEATURES = ["games_played"]
ALL_FEATURES = FEATURE_COLS + EXTRA_FEATURES

# The availability model gets ONE feature the rate models deliberately do
# not: where the player sits on his team's depth chart entering the season
# being predicted (see depth_history.py). It is kept as a separate list,
# rather than appended to ALL_FEATURES, because the two model families need
# different things from it:
#
#   - Availability is exactly the question a depth chart answers. A
#     preseason chart is public before week 1, so using the TARGET season's
#     chart is a real preseason input, not leakage - it is the same
#     information the live 2026 run already reads off
#     src/depth_chart/starters_2026.csv, just supplied for every historical
#     season too so the model can be trained and held out on it.
#   - The per-game RATE models are trained only on players who actually
#     played (build_transition_pairs filters games_played_to > 0), which is
#     the population where the chart has least to say, and their input
#     schema is frozen across every saved model in models/. Widening
#     ALL_FEATURES would silently invalidate all of them.
#
# Every saved model carries its own `features` list in its joblib, so
# predict.py and backtest.py read the schema off the model rather than
# assuming ALL_FEATURES for both families.
AVAILABILITY_FEATURES = ALL_FEATURES + [AVAILABILITY_DEPTH_FEATURE]

# Team-season-grain columns within FEATURE_COLS (identical for every player
# on the same team-season, since none of these are player-specific - see
# features.py's FEATURE_COLS docstring) - the input side of the
# joint/multi-output Phase A team_passing_yards model
# (build_team_transition_pairs below, train.py's team-grain model).
# `play_action_rate` (from OC_METRICS) is deliberately EXCLUDED here, not
# forgotten: it's NaN for every 2021 team-season (all 32 teams - a real
# upstream charting-coverage gap, consistent with FTN charting data
# starting in 2022 per this project's Phase 0 findings), and unlike
# LightGBM (used everywhere else in this pipeline, handles NaN natively),
# the RidgeCV model this feeds errors on any NaN input. Dropping one
# column rather than imputing a fill value keeps the team-total model
# simple and avoids inventing a new "how to fill NaN for a linear model"
# mechanism this project has never needed before.
TEAM_FEATURES = [
    "ol_pass_protection_score", "ol_run_blocking_score", "ol_confidence_low_churn",
    "opp_def_pass_epa_prior", "opp_def_rush_epa_prior",
] + [c for c in OC_METRICS if c != "play_action_rate"]

# (position, stat) pairs reframed under the joint/multi-output Phase A
# decomposition to predict a share of `TEAM_TOTAL_LABEL` instead of an
# absolute per-game rate directly (see build_team_transition_pairs above
# and train.py's team_passing_yards model) - shared by train.py,
# backtest.py, and predict.py so all three agree on which (position, stat)
# combos are reframed, rather than re-deriving/duplicating this set.
REFRAMED_SHARE_STATS = {("WR", "receiving_yards"), ("TE", "receiving_yards"), ("RB", "receiving_yards")}
RECEIVING_SHARE_LABEL = "receiving_yards_share"
TEAM_TOTAL_LABEL = "team_passing_yards_pg"

# Cap on a team's summed receiving-share predictions across WR+TE+RB before
# composing with team_passing_yards_pg (joint/multi-output Phase A). Shares
# are NOT forced to sum to exactly 1 - practice-squad/emergency production
# isn't fully in the modeled universe, and the real 2024-2025 held-out
# receiving/passing ratio (see backtest.py's coherence_ratio_backtest)
# itself ranges well above 1 for some teams even on ACTUAL outcomes - only
# scaled down if the predicted sum exceeds this ceiling. Lives here (not
# predict.py) since Phase 2 of the consensus-gap work: backtest.py applies
# the identical cap via receiving_share_scale below, so the MAE/interval
# calibration and the shipped composition cannot drift apart.
#
# Value history: 1.5 originally (a stated, un-tuned judgment call, and -
# found in Phase 2 - effectively a bug amplifier: computed on raw
# pre-discount shares it squeezed real starters for bench players' phantom
# volume, and once fixed it never bound at all). Tightened 1.5 -> 1.2 at
# the Phase-2 gate (user decision, 2026-08-07) on held-out evidence:
# capping the correctly-measured share sum improves 2024->2025 MAE
# monotonically down to ~1.1 (reframed overall 8.70 -> 8.47; 1.2 captures
# most of it at 8.53) and reverses at 1.0. 1.2 rather than the 1.1
# optimum deliberately: the backtest denominator population (test-pair
# players, no discounts/rookies) is not identical to the live one (full
# rosters, discounted, rookie-implied shares included), so a margin is
# left for that asymmetry rather than tuning to the holdout's edge.
RECEIVING_SHARE_SUM_CAP = 1.2


def receiving_share_scale(share_df, extra_team_share=None, cap=RECEIVING_SHARE_SUM_CAP):
    """Per-team renormalization scale for reframed receiving-share
    predictions, shared by predict.py (live composition) and backtest.py
    (MAE + interval calibration) so the two cannot diverge.

    share_df: columns ['team', 'share'], one row per (player, reframed
    stat) prediction, plus an OPTIONAL 'weight' column (see below).

    extra_team_share: optional Series indexed by team, added to that
    team's denominator only. predict.py passes rookie-path implied shares
    here (an incoming 1st-round WR like TEN's Carnell Tate consumes real
    target share, but is predicted outside the share models - without
    this, a veteran room's shares never feel rookie competition at all).

    THE DENOMINATOR IS PARTICIPATION-WEIGHTED (Gate B). 'weight' is the
    fraction of the season the player is expected to be active
    (projected_games / SEASON_GAMES); absent, it defaults to 1.0 and this
    function behaves exactly as before, which is what keeps backtest.py's
    calibration comparable.

    Why weighting is the right denominator, and not a tuning knob: a share
    here is a per-game share CONDITIONAL ON PLAYING, so summing raw shares
    across a 40-man roster counts players who will never dress. Weighting
    each by participation measures the quantity that is physically bounded
    - and it really is bounded. Summed over a team's actual receivers and
    weighted by games played, the historical share sum is 0.99 in every
    season 2021-2025 (0.99/0.99/0.99/0.99/0.99), because in any single game
    the active receivers' shares of team passing yards sum to 1. The
    unweighted sum over the same players is 1.31-1.41 and drifts with how
    many bodies a roster carries, which is not a property of football.
    RECEIVING_SHARE_SUM_CAP = 1.2 therefore now sits ~20% above a known
    invariant rather than above a number whose meaning moved.

    This replaces the previous contract, where 'share' was expected to
    arrive pre-multiplied by the caller's role/depth-chart discount. That
    worked only because the old 0.15/0.4 discounts happened to be small
    enough to stand in for participation. Gate B fit those multipliers
    against outcomes and they are near 1.0, so the proxy is gone: with
    fitted rates and an unweighted denominator, 17 of 32 teams breach the
    cap and starters get scaled to 0.716. With participation weighting,
    none do. The two jobs the old discount was doing - scale the rate,
    weight the denominator - are now done by the two quantities that
    actually answer them.

    Returns (scale, over_cap), both aligned to share_df.index."""
    weight = share_df["weight"] if "weight" in share_df.columns else 1.0
    weighted = share_df["share"] * weight
    denom = weighted.groupby(share_df["team"]).transform("sum")
    if extra_team_share is not None:
        denom = denom + share_df["team"].map(extra_team_share).fillna(0.0)
    over_cap = denom > cap
    scale = pd.Series(1.0, index=share_df.index)
    scale.loc[over_cap] = cap / denom[over_cap]
    return scale, over_cap


def build_transition_pairs(feat, position, stat, season_pairs, label_col=None):
    """Stack (X, y) rows across the given list of (season_from, season_to)
    pairs for one position/stat. Requires games_played > 0 in season_to
    (a real per-game rate to learn from) but NOT in season_from - a
    veteran coming off an injury-limited season is still a real veteran
    with real trailing features, just noisier ones (games_played is itself
    a feature, so the model can learn to discount low-sample seasons).

    `label_col` overrides the default `{stat}_pg` label - used by the
    joint/multi-output Phase A reframing (train.py), where
    WR_receiving_yards/TE_receiving_yards/RB_receiving_yards are trained on
    `receiving_yards_share` instead of `receiving_yards_pg`. The actual
    `{stat}_pg` rate is always still included in the output (as its own
    column, not renamed away) even when `label_col` is set, so a caller
    reconstructing `team_total_pred x share_pred` has the real rate to
    score against - see backtest.py's reframed-stat handling."""
    pos_df = feat[feat["position"] == position]
    rate_col = f"{stat}_pg"
    y_col = label_col or rate_col

    rows = []
    for season_from, season_to in season_pairs:
        a = pos_df[pos_df["season"] == season_from][["player_id", "team"] + ALL_FEATURES + [rate_col]]
        a = a.rename(columns={rate_col: "naive_pred"})  # season_from's own rate = naive carry-forward baseline
        cols_to = ["player_id", "games_played", rate_col] + ([y_col] if y_col != rate_col else [])
        b = pos_df[pos_df["season"] == season_to][cols_to].rename(columns={"games_played": "games_played_to"})
        merged = a.merge(b, on="player_id", how="inner")
        merged = merged[merged["games_played_to"] > 0]
        merged["season_from"] = season_from
        merged["season_to"] = season_to
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out


# Games in a modern NFL regular season - the ceiling on any availability
# prediction. (Pre-2021 seasons were 16; predicting a 17th game for a
# 2019 row is a rounding-level concern next to the +/-4 game MAE of the
# availability model itself, so a single constant is used rather than a
# per-season lookup.)
SEASON_GAMES = 17

AVAILABILITY_LABEL = "games_played_to"


def build_availability_pairs(feat, position, season_pairs):
    """Season N features -> season N+1 GAMES PLAYED, for the availability
    model (Phase 11).

    Differs from build_transition_pairs in the one way that matters: it
    LEFT joins and keeps `games_played_to = 0`, rather than inner-joining
    and filtering to `games_played_to > 0`. That filter is correct for a
    per-game RATE label - a player who never played has no rate to learn -
    but it makes the single worst outcome in the data structurally
    invisible: 21% of the elite injury cohort studied in Phase 6 missed
    their entire following season, and neither training nor the backtest
    could see or be charged for it. Availability is precisely the target
    those rows belong to.

    A player absent from season N+1's feature frame genuinely played 0
    games: build_player_season_features aggregates from weekly usage, so
    only players with real active weeks appear at all. The fillna(0) here
    is a real value, not a cover for a failed join.

    Each row also carries `target_depth_rank` - the player's position on
    his team's depth chart entering season_to, the one feature that
    separates "returning veteran who will start" from "returning veteran
    nobody has a role for." See AVAILABILITY_FEATURES above and
    depth_history.py. Joined per season_to, since the whole point is that
    it is the TARGET season's chart."""
    pos_df = feat[feat["position"] == position]
    rows = []
    for season_from, season_to in season_pairs:
        a = pos_df[pos_df["season"] == season_from][["player_id", "team"] + ALL_FEATURES]
        b = pos_df[pos_df["season"] == season_to][["player_id", "games_played"]].rename(
            columns={"games_played": AVAILABILITY_LABEL})
        merged = a.merge(b, on="player_id", how="left")
        merged["position"] = position
        merged = attach_availability_depth_rank(merged, season_to)
        merged["played_again"] = merged[AVAILABILITY_LABEL].notna()
        merged[AVAILABILITY_LABEL] = merged[AVAILABILITY_LABEL].fillna(0.0)
        # season_from's own games count = the carry-forward baseline this
        # model has to beat, kept under the same `naive_pred` name every
        # other pair-builder here uses.
        merged["naive_pred"] = merged["games_played"]
        merged["season_from"] = season_from
        merged["season_to"] = season_to
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_team_transition_pairs(feat, season_pairs):
    """Team-grain sibling of build_transition_pairs, for the joint/
    multi-output Phase A `team_passing_yards` model (train.py). One row per
    (season, team) - taken via drop_duplicates rather than aggregation,
    since every player on a team-season already carries identical
    TEAM_FEATURES values (they're team-season grain to begin with, not
    player-specific)."""
    # dropna on team FIRST, not after dedup - some rows (e.g. a QB who
    # played 0 games and has no resolved season team) carry team=NaN, which
    # drop_duplicates would otherwise keep as its own spurious "33rd team"
    # group per season.
    team_df = feat.dropna(subset=["team"]).drop_duplicates(subset=["season", "team"])[
        ["season", "team", "team_passing_yards_pg"] + TEAM_FEATURES
    ]

    rows = []
    for season_from, season_to in season_pairs:
        a = team_df[team_df["season"] == season_from][["team"] + TEAM_FEATURES + ["team_passing_yards_pg"]]
        a = a.rename(columns={"team_passing_yards_pg": "naive_pred"})
        b = team_df[team_df["season"] == season_to][["team", "team_passing_yards_pg"]]
        merged = a.merge(b, on="team", how="inner")
        merged["season_from"] = season_from
        merged["season_to"] = season_to
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out
