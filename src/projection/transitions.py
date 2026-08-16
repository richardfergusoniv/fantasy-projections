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
    AVAILABILITY_DEPTH_FEATURE, DEPTH_TIER_COLUMN, attach_availability_depth_rank,
    attach_depth_tier, load_preseason_depth_chart,
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
# The team model's own lag feature (the team's prior-season passing volume)
# carries a team-grain-only name deliberately - see
# build_team_transition_pairs. Anything that tries to score this model on a
# player-grain frame now fails loudly instead of reading the player's own
# prior rate out of a same-named column. Use `team_model_inputs` below to
# score it for player rows.
TEAM_MODEL_FEATURES = TEAM_FEATURES + ["team_naive_pred"]

# (position, stat) pairs reframed under the joint/multi-output Phase A
# decomposition to predict a share of `TEAM_TOTAL_LABEL` instead of an
# absolute per-game rate directly (see build_team_transition_pairs above
# and train.py's team_passing_yards model) - shared by train.py,
# backtest.py, and predict.py so all three agree on which (position, stat)
# combos are reframed, rather than re-deriving/duplicating this set.
REFRAMED_SHARE_STATS = {("WR", "receiving_yards"), ("TE", "receiving_yards"), ("RB", "receiving_yards")}
RECEIVING_SHARE_LABEL = "receiving_yards_share"
# The role-rate counterpart of RECEIVING_SHARE_LABEL: the same share with the
# appearance-week denominator replaced by full-season team passing yards
# scaled to the player's eligibility. See features.py.
RECEIVING_SHARE_ELIG_LABEL = "receiving_yards_share_elig"
TEAM_TOTAL_LABEL = "team_passing_yards_pg"
TEAM_ATTEMPTS_LABEL = "team_pass_attempts_pg"
TEAM_CARRIES_LABEL = "team_carries_pg"
TEAM_RUSH_YARDS_LABEL = "team_rushing_yards_pg"

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
    """Per-team normalization scale for reframed receiving-share
    predictions, shared by predict.py (live composition) and backtest.py
    (MAE + interval calibration) so the two cannot diverge.

    share_df: columns ['team', 'share'], one row per (player, reframed
    stat) prediction, plus an OPTIONAL 'weight' column (see below).

    extra_team_share: optional Series indexed by team, added to that
    team's denominator only. predict.py passes rookie-path implied shares
    here (an incoming 1st-round WR like TEN's Carnell Tate consumes real
    target share, but is predicted outside the share models - without
    this, a veteran room's shares never feel rookie competition at all).

    THE DENOMINATOR IS EXPOSURE-WEIGHTED (Gate B). ``share`` is defined over
    offensive-appearance weeks, so live callers pass ``weight`` as
    projected_games / SEASON_GAMES. Historical validation must use the same
    games_played / scheduled_games weight. Absent, weight defaults to 1.0
    for compatibility with older callers, but production should always pass
    the explicit exposure.

    Why weighting is the right denominator, and not a tuning knob: a share
    here is a per-game share CONDITIONAL ON PLAYING, so summing raw shares
    across a 40-man roster counts players who will never dress. Weighting
    each by participation measures the quantity that is physically bounded.
    After redefining the label over offensive-appearance weeks, the historical
    team-season mean is 0.989/0.985/0.992/0.992/0.991 for 2021-2025, with zero
    team-seasons above 1.2 (maximum 1.140). Thus the cap once again sits above
    an invariant measured with the same denominator production uses.

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

    This model-stage guard is downward-only. Composition no longer fills
    named receiving toward a team yardage coverage floor; season totals are
    ``pred_pg × projected_games``. Returns (scale, over_cap), both aligned
    to share_df.index."""
    weight = share_df["weight"] if "weight" in share_df.columns else 1.0
    weighted = share_df["share"] * weight
    denom = weighted.groupby(share_df["team"]).transform("sum")
    if extra_team_share is not None:
        denom = denom + share_df["team"].map(extra_team_share).fillna(0.0)
    over_cap = denom > cap
    scale = pd.Series(1.0, index=share_df.index)
    scale.loc[over_cap] = cap / denom[over_cap]
    return scale, over_cap


# Age-effect shrinkage (predict-time only - no retrain, the saved models in
# models/ are unchanged). Added after a user-driven investigation found the
# trained models lean on `age` ~1.6-3.2x harder for WR than RB (gain-share
# and partial-dependence magnitude), which the user found excessive.
# Follow-up work ruled out a bug and ruled out the "stale historical
# pattern" explanation (the age-related WR decline is real, current, and
# specifically concentrated in the model's own 2021-2025 training window -
# see project memory / DRAFT_CAPITAL_REMOVAL-adjacent investigation notes
# for the full writeup). But it also surfaced a genuine asymmetry: RB's age
# signal is thin (sparse older-RB sample) and does NOT earn its keep -
# grid-searching a shrink factor from 1.0 (unshrunk) to 0.0 (fully
# neutralized) against both the single 2024->2025 holdout and the 3-fold
# rolling-origin backtest showed RB accuracy improves MONOTONICALLY as age
# is shrunk toward 0, across all 7 RB stats (holdout: -1.2% mean MAE at
# shrink=0; rolling: -0.4%) - a clean win, not a cosmetic one. WR shows the
# opposite: monotonic MAE REGRESSION as age is shrunk (holdout: +1.3% at
# shrink=0; rolling: +1.0%) - dampening it costs real accuracy. QB/TE are
# within noise either way (<0.5%). User's explicit choice (2026-08-14):
# ship the shrink only where evidence supports it - RB fully neutralized,
# everyone else untouched.
REFERENCE_AGE = {"QB": 27.0, "RB": 25.0, "WR": 25.0, "TE": 26.0}  # median age, that position's 2021-2025 population
AGE_EFFECT_SHRINK = {"QB": 1.0, "RB": 0.0, "WR": 1.0, "TE": 1.0}  # 1.0 = unshrunk, 0.0 = fully neutralized


def age_shrunk_predict(model, X, position, features=None):
    """Predict with `model` on `X[features]` (`features` defaults to
    ALL_FEATURES), dampening `age`'s marginal contribution for `position`
    per AGE_EFFECT_SHRINK - see that constant's comment for the evidence.
    The dampened prediction is `pred_neutral + shrink * (pred - pred_neutral)`,
    where `pred_neutral` is the SAME already-trained model's prediction with
    `age` replaced by REFERENCE_AGE[position] and every other feature left
    at its real observed value - an individual-conditional-expectation swap,
    not a retrain. Skips the second predict() call entirely (a no-op) for
    any position at shrink=1.0, so the 3 unaffected positions pay no extra
    inference cost. Shared by predict.py (live composition) and backtest.py
    (MAE + interval calibration), so the two cannot drift apart - same
    reason receiving_share_scale lives here rather than in predict.py."""
    if features is None:
        features = ALL_FEATURES
    Xf = X[features]
    pred = model.predict(Xf)
    shrink = AGE_EFFECT_SHRINK.get(position, 1.0)
    if shrink >= 1.0:
        return pred
    X_neutral = Xf.copy()
    X_neutral["age"] = REFERENCE_AGE[position]
    pred_neutral = model.predict(X_neutral)
    return pred_neutral + shrink * (pred - pred_neutral)


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

# A rate over fewer than this many eligible weeks is division by a number too
# small to mean anything (a player signed for the last two weeks of a season
# is not evidence about a full-season role). Such rows leave the population
# rather than being clipped, so nothing silently smuggles a 2-week sample in
# at full weight.
MIN_ELIGIBLE_WEEKS = 4

ROLE_ZERO_FLAG = "is_role_zero"


def role_rate_label(stat):
    """The per-eligible-week label name for `stat` - see features.py."""
    return f"{stat}_per_elig"


def build_role_transition_pairs(feat, position, stat, season_pairs, conn=None,
                                label_col=None):
    """Season N features -> season N+1 ROLE RATE, per eligible week.

    Two deliberate differences from build_transition_pairs, and they are the
    whole point:

    1. The label is `{stat}_per_elig`, not `{stat}_pg`. See features.py for
       why the appearance-week denominator was survivorship-selected.

    2. The population KEEPS zero-production seasons whose cause is role, and
       drops the ones whose cause is not. A charted player who was rostered
       and off reserve all year and never took an offensive snap has a true
       role rate of zero, and excluding him is exactly what taught the models
       that a third-stringer produces like a starter. But a player on IR
       belongs to the status-override gate, and a player who was cut is out
       of the population, so neither may enter as a zero - including them
       would bake injury attrition and roster churn back into a rate that is
       supposed to describe a role. `is_role_zero` marks the rows that were
       added this way.

    Requires `conn` to resolve roster status and the preseason chart. Passing
    `conn=None` yields the played-only population and is only appropriate for
    unit tests that supply their own frame.

    `label_col` overrides the label the same way build_transition_pairs does,
    for the reframed receiving-share models.
    """
    pos_df = feat[feat["position"] == position]
    y_col = label_col or role_rate_label(stat)
    rate_col = f"{stat}_pg"

    rows = []
    for season_from, season_to in season_pairs:
        a = pos_df[pos_df["season"] == season_from][
            ["player_id", "team"] + ALL_FEATURES + [rate_col]]
        a = a.rename(columns={rate_col: "naive_pred"})
        cols_to = ["player_id", "games_played", "eligible_weeks", y_col]
        b = pos_df[pos_df["season"] == season_to][cols_to].rename(columns={
            "games_played": "games_played_to", "eligible_weeks": "eligible_weeks_to"})
        merged = a.merge(b, on="player_id", how="left")
        merged[ROLE_ZERO_FLAG] = False

        if conn is not None:
            merged = _admit_role_zeros(merged, position, season_to, conn)
        merged = merged[merged[y_col].notna()]
        merged = merged[merged["eligible_weeks_to"] >= MIN_ELIGIBLE_WEEKS]

        # The chart is keyed by (player, position); ALL_FEATURES carries no
        # position column, so name it before the lookup. Models are strictly
        # per-position, so this is the model's position by construction.
        merged["position"] = position
        merged = attach_depth_tier(merged, season_to, conn=conn)
        merged["season_from"] = season_from
        merged["season_to"] = season_to
        rows.append(merged)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _admit_role_zeros(merged, position, season_to, conn):
    """Fill a genuine role zero, drop every other kind of missing outcome."""
    from src.projection.data_prep import (
        ELIGIBLE_ROSTER_STATUSES, player_dominant_roster_status, player_eligible_weeks)

    y_cols = [c for c in merged.columns if c.endswith("_per_elig")
              or c == RECEIVING_SHARE_ELIG_LABEL]
    chart = load_preseason_depth_chart(season_to, conn=conn)
    charted = set(chart[chart["position"] == position]["player_id"]) if not chart.empty else set()
    status = player_dominant_roster_status(conn, [season_to]).set_index("player_id")["status"]
    elig = player_eligible_weeks(conn, [season_to]).set_index("player_id")["eligible_weeks"]

    # A missing outcome row, or a row with no offensive snap, is a candidate.
    no_output = merged["games_played_to"].isna() | merged["games_played_to"].eq(0)
    is_zero = (
        no_output
        & merged["player_id"].isin(charted)
        & merged["player_id"].map(status).isin(ELIGIBLE_ROSTER_STATUSES)
    )
    merged.loc[is_zero, y_cols] = 0.0
    merged.loc[is_zero, "games_played_to"] = 0.0
    merged.loc[is_zero, "eligible_weeks_to"] = merged.loc[is_zero, "player_id"].map(elig)
    merged.loc[is_zero, ROLE_ZERO_FLAG] = True
    # Everything else that produced nothing leaves the population: rows still
    # carrying a NaN label are dropped by the caller.
    return merged


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
        # Availability follows the player, not the position label. A player
        # moving from WR to TE (or RB to WR) still appeared in season_to and
        # must not be silently relabeled as a zero-game attrition outcome for
        # his source-position model. Collapse any rare multi-position target
        # rows to the player's maximum appearance count, retaining the
        # position attached to that target-season row.  The target chart is
        # keyed by (player, position), so looking the player up under the
        # source/model position would give position changers a false
        # off-chart band even though their appearance outcome is now joined
        # correctly.
        b = (
            feat[feat["season"] == season_to][
                ["player_id", "position", "games_played"]
            ]
            .sort_values(["player_id", "games_played"], ascending=[True, False])
            .drop_duplicates("player_id", keep="first")
            .rename(columns={
                "position": "target_position",
                "games_played": AVAILABILITY_LABEL,
            })
        )
        merged = a.merge(b, on="player_id", how="left")
        merged["position"] = merged["target_position"].fillna(position)
        merged = attach_availability_depth_rank(merged, season_to)
        # The row still trains the source-position availability model.  Only
        # the held-out chart lookup above follows the target position.
        merged["position"] = position
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


def team_model_inputs(feat, season_pairs, season_from, teams):
    """TEAM_MODEL_FEATURES rows aligned to player-grain (`season_from`,
    `teams`) columns, so a team-grain model can be scored for player rows.

    The team model is fit at team-season grain. Scoring it directly on a
    player frame used to "work" because the frames shared a `naive_pred`
    column with incompatible meanings; that produced team-total forecasts
    ~40% low and, through the multiplicative share composition, biased
    every reframed receiving residual upward. Going through this function
    (or through an explicitly-built team frame) is the only supported way.

    Returns a frame positionally aligned to the inputs, with NaN for a
    (season_from, team) the team frame has no row for - the caller decides
    whether that is droppable, rather than this silently substituting.
    """
    pairs = build_team_transition_pairs(feat, season_pairs)
    if pairs.empty:
        return pd.DataFrame(np.nan, index=pd.RangeIndex(len(teams)), columns=TEAM_MODEL_FEATURES)
    lookup = pairs.drop_duplicates(["season_from", "team"]).set_index(
        ["season_from", "team"])[TEAM_MODEL_FEATURES]
    key = pd.MultiIndex.from_arrays([
        pd.Series(season_from).to_numpy(), pd.Series(teams).to_numpy()])
    return lookup.reindex(key).reset_index(drop=True)


def build_team_transition_pairs(feat, season_pairs, label_col=TEAM_TOTAL_LABEL):
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
        ["season", "team", label_col] + TEAM_FEATURES
    ]

    rows = []
    for season_from, season_to in season_pairs:
        a = team_df[team_df["season"] == season_from][["team"] + TEAM_FEATURES + [label_col]]
        a = a.rename(columns={label_col: "naive_pred"})
        # Same value under a team-grain-only name. `naive_pred` is kept
        # because every pair-builder in this module uses that name for its
        # carry-forward baseline and backtest_team_total scores against it.
        # But it ALSO used to be the team model's lag FEATURE, and that is
        # a name the player-grain frames already use for something else
        # entirely (the player's own prior rate). Predicting the team model
        # on a player frame therefore silently fed it ~30 yd/g where it
        # expected ~230 - no error, no NaN, just wrong numbers, which is
        # exactly how it survived two review rounds. TEAM_MODEL_FEATURES
        # now names this column, so the same mistake raises KeyError.
        a["team_naive_pred"] = a["naive_pred"]
        b = team_df[team_df["season"] == season_to][["team", label_col]]
        merged = a.merge(b, on="team", how="inner")
        merged["season_from"] = season_from
        merged["season_to"] = season_to
        rows.append(merged)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out
