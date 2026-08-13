"""Post-hoc corrections fit on the pipeline's OWN out-of-sample residuals.

Phase 6/7 of the consensus-gap work. Everything here is deliberately
small-parameter and fit by leave-one-transition-out, because a correction
fit on in-sample residuals would just relearn the model's training fit
and a correction with many parameters would overfit the ~4 transitions
this project has.

Two corrections were planned. Only one shipped:

**Phase 6, injury-cohort retention: MEASURED NO-OP, not built.** The
premise (established before the feature work) was that the models
applied a 0.45-0.67 next-season retention factor to injury-shortened
elite WRs where history said 0.72-0.77, and under-predicted all three
such players in the 2025 holdout by +8/+28/+30 ypg. After Phases 3-4
(active-weeks red-zone denominators + games-weighted feature blending)
that residual is GONE: re-measured on the same cohort definition (WR,
season-N active games <= 8, season-N >= 50 rec ypg, no suspension-coded
roster weeks - see injury_cohort_gate below), mean signed residual is
-0.60 ypg with 50% positive and model-implied retention 0.58 vs actual
0.56. The gate to build anything was > +3 ypg with >= 65% positive, so
nothing was built. This is the intended outcome: the penalty was a
feature-construction artifact, and fixing it at the feature level is
strictly better than layering a post-hoc multiplier on top of it.
`injury_cohort_gate` is kept as a runnable diagnostic so the claim can
be re-checked whenever the features change again.

**Phase 7, elite-producer shrinkage: SHIPPED.** Receivers with a high
OBSERVED season-N per-game rate are under-predicted at every games
level - the models shrink hard toward the positional mean, which is
correct on average and wrong at the top. Conditioning on the observed
rate is the load-bearing detail: an earlier attempt at this bucketed
residuals by PREDICTED value and found no monotone signature (because
the compression being corrected is exactly what makes the predicted
value unreliable as a key). Keyed to season-N observed rate, the
signature is clean and monotone - WR mean residual by observed ypg
bucket: 0-20 -0.77, 20-35 -0.21, 35-50 +0.52, 50-60 +0.93, 60-70
+2.10, 70+ +8.46.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.projection.depth_history import attach_availability_depth_rank
from src.projection.train import LGBM_PARAMS, fit_team_total, fit_availability
from src.projection.transitions import (
    build_transition_pairs, ALL_FEATURES, TEAM_FEATURES, TEAM_MODEL_FEATURES, team_model_inputs,
    AVAILABILITY_FEATURES,
    REFRAMED_SHARE_STATS, RECEIVING_SHARE_LABEL, receiving_share_scale,
    SEASON_GAMES,
)

# Per-position "elite" thresholds for the shrinkage correction, in
# observed season-N receiving yards per game. Deliberately different by
# position rather than one shared number: a 50-ypg TE is a top-5 player
# at the position while a 50-ypg WR is a WR3, so a single threshold would
# either miss every elite TE or sweep in half the league's WRs. Values
# picked from where each position's residual-by-observed-rate curve
# actually turns positive (see this module's docstring), then checked
# against held-out MAE at 50/60/70 before adopting.
ELITE_KNOTS = {"WR": 60.0, "TE": 50.0}

# RB is deliberately absent from ELITE_KNOTS: only 9 RB rows in the whole
# LOO set clear even a 35-ypg receiving bar, far too few to fit anything
# on. Stated rather than silently omitted.

# Hard ceiling on the additive correction, in yards per game. The fit is
# linear and unbounded by construction, so without this a hypothetical
# 110-ypg season-N receiver would receive a ~+18 ypg bump extrapolated
# from a region with almost no training support. 8 costs a little
# held-out MAE versus an uncapped fit (WR 11.934 vs 11.906) and is kept
# anyway - the cap is protection against the tail, not a fitted value.
ELITE_CORRECTION_CAP = 8.0

# Minimum rows above the knot before a position's correction is fit at
# all. Below this the position ships uncorrected rather than carrying a
# coefficient fit on a handful of players.
MIN_N_ABOVE_KNOT = 15

# Across-season consistency requirement: mean(per-transition mean
# residual) divided by the standard error of those per-transition means
# must clear this before a position's correction is applied. A pooled
# positive mean is NOT sufficient evidence on its own, and this project
# has a concrete case proving it - WR elite residuals by transition run
# +7.3, +16.6, -1.4, +2.6 (ratio 1.6): pooled that is a confident-looking
# +5.8 ypg, but it is one anomalous 2022->2023 season carrying two
# neutral ones and one that reverses outright, and applying it made the
# 2025 holdout's elite WRs slightly WORSE (15.31 -> 15.40 MAE). TE over
# the same transitions runs +7.1, +8.1, +5.8, +3.6 (ratio 6.3) - the same
# size effect, present every single year - and improves the holdout
# markedly (16.80 -> 12.10 elite MAE). The guard is what separates them,
# and it re-evaluates automatically as seasons are added.
MIN_SEASON_CONSISTENCY = 2.0

# Transitions needed before consistency can be judged at all.
MIN_TRANSITIONS_FOR_CONSISTENCY = 3

# Roster status codes counted as suspension (NOT injury) when deciding
# whether a short season belongs in the injury cohort. This is what keeps
# Rashee Rice's 2024 (3 games, suspension) out of an injury-retention
# cohort that would otherwise inflate him - verified against
# weekly_rosters: Malik Nabers' missed 2025 weeks are all R01
# (Reserve/Injured), Rice's are R40.
SUSPENSION_CODES = {"R40"}


def _participation_weight(feat, test, position, train_pairs, held):
    """Out-of-sample projected_games / SEASON_GAMES for the held-out fold's
    players, so the share-sum cap here uses the same participation-weighted
    denominator predict.py now ships (transitions.receiving_share_scale).

    Without it this function composed differently from production: on the
    2024->2025 fold the unweighted denominator tripped the cap on 10 of 32
    teams and scaled 128 of 360 rows down to as low as 0.700, while the
    weighted production denominator trips it on none. The residuals beta is
    fit on were therefore inflated by suppression that the shipped
    composition no longer applies - a silent drift between what is fit and
    what is shipped, worth ~3% on TE's beta (0.4903 -> 0.4747).

    The availability model is refit on `train_pairs` only, never on `held`,
    for the same reason every other model in this function is: a residual
    used to fit a correction has to be genuinely out-of-sample. Using the
    held-out season's ACTUAL games would be leakage - it is the outcome
    being predicted.

    Falls back to 1.0 (the pre-Gate-A behaviour) if the availability model
    cannot be fit for this position, rather than to 0, which would empty
    the team's denominator and disable the cap without saying so."""
    model, n = fit_availability(feat, position, train_pairs)
    if model is None or n == 0:
        return 1.0
    x = attach_availability_depth_rank(
        test[["player_id"]].assign(position=position), held[1])
    x = test.join(x[[AVAILABILITY_FEATURES[-1]]])
    games = np.clip(model.predict(x[AVAILABILITY_FEATURES]), 0, SEASON_GAMES)
    return np.clip(games / SEASON_GAMES, 0, 1)


def compute_loo_receiving_residuals(feat, pairs):
    """Leave-one-transition-out out-of-sample residuals for the reframed
    receiving models, composed exactly the way predict.py ships them
    (share x capped renormalization x team total - see
    transitions.receiving_share_scale).

    For each pair in `pairs`, models are fit on every OTHER pair and used
    to predict the held-out one, so every residual is genuinely
    out-of-sample. Returns one row per (player, position) test row with
    `naive_pred` (the player's own observed season-N rate - the key the
    shrinkage correction conditions on), `pred`, `actual`, `resid`.

    Note this deliberately mirrors backtest.py's composition rather than
    reusing its function: backtest.py holds out one FIXED pair to score
    the pipeline, while this rotates through all of them to gather
    residuals for fitting. Both call the same shared scale helper, which
    is the part that must not drift."""
    if len(pairs) < 2:
        return pd.DataFrame()

    rows = []
    for held in pairs:
        train_pairs = [p for p in pairs if p != held]
        team_model, _ = fit_team_total(feat, train_pairs)
        frames, tests = [], {}
        for position, stat in sorted(REFRAMED_SHARE_STATS):
            train = build_transition_pairs(feat, position, stat, train_pairs, label_col=RECEIVING_SHARE_LABEL)
            test = build_transition_pairs(feat, position, stat, [held], label_col=RECEIVING_SHARE_LABEL)
            test = test.dropna(subset=TEAM_FEATURES).reset_index(drop=True)
            if train.empty or test.empty:
                continue
            model = LGBMRegressor(**LGBM_PARAMS)
            model.fit(train[ALL_FEATURES], train[RECEIVING_SHARE_LABEL])
            f = test[["team"]].copy()
            f["share"] = np.clip(model.predict(test[ALL_FEATURES]), 0, None)
            # Team-grain inputs - see transitions.team_model_inputs. Fitting
            # the elite-shrinkage correction on a composition built from
            # ~40%-low team totals inflated its residuals and therefore beta.
            f["total"] = np.clip(team_model.predict(
                team_model_inputs(feat, [held], test["season_from"], test["team"])), 0, None)
            f["weight"] = _participation_weight(feat, test, position, train_pairs, held)
            f["position"], f["row"] = position, test.index
            frames.append(f)
            tests[position] = test
        if not frames:
            continue
        allf = pd.concat(frames, ignore_index=True)
        scale, _ = receiving_share_scale(allf[["team", "share", "weight"]])
        allf["pred"] = allf["share"] * scale * allf["total"]
        for position, test in tests.items():
            sub = allf[allf["position"] == position].sort_values("row")
            out = test[["player_id", "games_played", "naive_pred"]].copy()
            out["pred"] = sub["pred"].to_numpy()
            out["actual"] = test["receiving_yards_pg"].to_numpy()
            out["position"] = position
            out["season_from"] = held[0]
            rows.append(out)

    if not rows:
        return pd.DataFrame()
    res = pd.concat(rows, ignore_index=True)
    res["resid"] = res["actual"] - res["pred"]
    return res


def fit_elite_shrinkage(residuals):
    """Fit, per position, `resid ~ beta * max(0, observed_ypg - knot)`
    through the origin (single free parameter, ordinary least squares).

    Through-origin is the point, not a simplification: a player whose
    season-N rate is at or below the knot must receive exactly zero
    correction, so the whole non-elite population - the large majority of
    rows, and the part the models already predict without bias - is
    untouched by construction rather than by a fitted intercept that
    happens to land near zero.

    A position is omitted from the returned params entirely (i.e. ships
    uncorrected) unless it clears all three gates: MIN_N_ABOVE_KNOT rows
    above the knot, a positive fitted beta, and MIN_SEASON_CONSISTENCY on
    the across-transition stability of the effect. A negative beta would
    mean elite producers are OVER-predicted - a real finding, but the
    opposite phenomenon from the one this correction exists for, and not
    something to auto-apply. The consistency gate is the one that
    actually bites in practice; see its constant for the WR-vs-TE case
    that motivated it."""
    params = {}
    # A one-transition expanding-window fold has no leave-one-transition-out
    # residuals. Treat that as "no evidence for a correction" instead of
    # indexing a columnless empty frame and aborting the entire backtest.
    required = {"position", "naive_pred", "season_from", "resid"}
    if residuals is None or residuals.empty or not required.issubset(residuals.columns):
        return params
    for position, knot in sorted(ELITE_KNOTS.items()):
        sub = residuals[residuals["position"] == position]
        if sub.empty:
            continue
        above = sub[sub["naive_pred"] > knot]
        n_above = len(above)
        if n_above < MIN_N_ABOVE_KNOT:
            continue

        season_means = above.groupby("season_from")["resid"].mean()
        if len(season_means) < MIN_TRANSITIONS_FOR_CONSISTENCY:
            continue
        # ddof=1 std over transitions / sqrt(k) - the spread BETWEEN
        # seasons, which is what "does this happen every year" asks. A
        # zero-spread edge case is treated as maximally consistent.
        spread = float(season_means.std(ddof=1))
        se = spread / np.sqrt(len(season_means))
        consistency = np.inf if se == 0 else float(season_means.mean() / se)
        if consistency < MIN_SEASON_CONSISTENCY:
            continue

        x = np.maximum(0.0, sub["naive_pred"].to_numpy() - knot)
        y = sub["resid"].to_numpy()
        denom = float((x * x).sum())
        if denom <= 0:
            continue
        beta = float((x * y).sum() / denom)
        if beta <= 0:
            continue
        params[position] = {
            "knot": float(knot), "beta": beta,
            "cap": float(ELITE_CORRECTION_CAP), "n_above": n_above,
            "season_consistency": consistency,
        }
    return params


def elite_shrinkage_adjustment(position, observed_pg, params):
    """Per-row additive correction in yards/game, aligned to the inputs.

    `observed_pg` is the player's OWN season-N receiving yards per game
    (see this module's docstring for why the observed rate, not the
    predicted one, is the correct key). Rows whose position has no fitted
    params, or with a missing observed rate, get 0.0 - never NaN, which
    would silently destroy an otherwise-valid prediction downstream."""
    position = pd.Series(position).reset_index(drop=True)
    observed = pd.to_numeric(pd.Series(observed_pg).reset_index(drop=True), errors="coerce")
    adj = pd.Series(0.0, index=observed.index)
    for pos, p in params.items():
        mask = (position == pos) & observed.notna()
        if not mask.any():
            continue
        excess = np.maximum(0.0, observed[mask] - p["knot"])
        adj.loc[mask] = np.minimum(p["beta"] * excess, p["cap"])
    return adj.to_numpy()


def injury_cohort_gate(residuals, suspension_weeks=None, min_ypg=50.0, max_games=8):
    """Phase-6 diagnostic (see this module's docstring for why nothing was
    built from it): does an injury-shortened elite-WR residual still
    exist? Returns a dict with the cohort's size, mean/median signed
    residual, positive fraction, and model-vs-actual implied retention.

    `suspension_weeks`: optional (player_id, season) -> count of
    suspension-coded roster weeks, used to exclude suspension cases from
    an injury cohort. Without it the cohort is games-based only, which
    WILL sweep in suspensions - pass it whenever the roster data is
    available."""
    wr = residuals[residuals["position"] == "WR"].copy()
    if suspension_weeks is not None and not wr.empty:
        wr = wr.merge(
            suspension_weeks, left_on=["player_id", "season_from"],
            right_on=["player_id", "season"], how="left",
        )
        wr["susp_weeks"] = wr["susp_weeks"].fillna(0)
        wr = wr[wr["susp_weeks"] == 0]
    cohort = wr[(wr["games_played"] <= max_games) & (wr["naive_pred"] >= min_ypg)]
    if cohort.empty:
        return {"n": 0}
    return {
        "n": len(cohort),
        "mean_resid": float(cohort["resid"].mean()),
        "median_resid": float(cohort["resid"].median()),
        "frac_positive": float((cohort["resid"] > 0).mean()),
        "model_retention": float((cohort["pred"] / cohort["naive_pred"]).mean()),
        "actual_retention": float((cohort["actual"] / cohort["naive_pred"]).mean()),
    }


def load_suspension_weeks(conn):
    """(player_id, season) -> suspension-coded roster week count, for
    injury_cohort_gate. See SUSPENSION_CODES."""
    rost = pd.read_sql(
        "select player_id, season, status_description_abbr as code from weekly_rosters", conn)
    rost["is_susp"] = rost["code"].isin(SUSPENSION_CODES)
    return rost.groupby(["player_id", "season"])["is_susp"].sum().rename("susp_weeks").reset_index()
