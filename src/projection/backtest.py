"""Backtest: hold out the 2024 -> 2025 transition, train on 2021->22,
22->23, 23->24 only, predict 2025 per-game rates, compare MAE against the
naive baseline (2024's own per-game rate, carried forward unchanged as the
"prediction" for 2025).

2025 ground-truth caveat (surfaced again here, not just in PHASE1_REPORT.md):
`weekly`'s 2025 rows are the pbp-fallback aggregation, not nflverse's
official player_stats release (see src/ingest/pbp_stats_fallback.py). This
doesn't change the backtest's structure - the model and the naive baseline
are both scored against the same 2025 numbers - but the "ground truth" row
itself carries the fallback methodology's caveats (no fumbles/2pt logic,
built to match the same named fields but not a byte-for-byte replica of
nflverse's own aggregation).

Rookie evaluation is reported separately (fit_rookie_baselines on
2016-2024, predict 2025) since rookies have no naive-baseline equivalent
(no prior season to carry forward) - MAE is reported but there's no
apples-to-apples baseline comparison to make for them.
"""
import os

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import RidgeCV

from src.projection.data_prep import get_conn
from src.projection.features import build_player_season_features, TARGET_STATS
from src.projection.depth_history import attach_depth_tier
from src.projection.transitions import (
    build_transition_pairs, build_team_transition_pairs, build_availability_pairs,
    build_role_transition_pairs, role_label_for, role_rate_label,
    ALL_FEATURES, ROLE_FEATURES, ROLE_PRIOR_FEATURE, RECEIVING_SHARE_ELIG_LABEL,
    AVAILABILITY_FEATURES, TEAM_FEATURES, TEAM_MODEL_FEATURES, team_model_inputs,
    REFRAMED_SHARE_STATS, age_shrunk_predict,
    RECEIVING_SHARE_LABEL, TEAM_TOTAL_LABEL, AVAILABILITY_LABEL, SEASON_GAMES,
    TEAM_ATTEMPTS_LABEL, receiving_share_scale,
)
from src.projection.rookies import build_rookie_dataset, fit_rookie_baselines, predict_rookies
from src.projection.train import LGBM_PARAMS
from src.projection.corrections import (
    compute_loo_receiving_residuals, fit_elite_shrinkage, elite_shrinkage_adjustment,
    injury_cohort_gate, load_suspension_weeks, projected_participation_weight,
)

TRAIN_PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024)]
TEST_PAIR = (2024, 2025)
ROLLING_TEST_PAIRS = [(2022, 2023), (2023, 2024), (2024, 2025)]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
INTERVAL_QUANTILES = (0.10, 0.90)  # 80% empirical interval width - see PHASE5_REPORT.md for why
INTERVAL_MIN_N = 30  # veteran (position, stat) test-set n below this would need a parametric fallback (none do - min is 61)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


# depth_ladder_factors and depth_rate_calibration lived here. Both are gone
# with the Gate B multiplier they served: depth now reaches a prediction as a
# model input (the tier in ROLE_FEATURES), so there is no factor to apply and
# no ladder to calibrate. models/depth_rate_calibration.csv is no longer
# written. See contracts.TEAM_RECONCILE_ALPHA for what replaced the
# team-level half of what the ladder was informally doing.


def _fit_team_total_model(feat, train_pairs):
    team_train = build_team_transition_pairs(feat, train_pairs)
    if team_train.empty:
        return None
    model = RidgeCV(alphas=np.logspace(-2, 3, 20))
    model.fit(team_train[TEAM_MODEL_FEATURES], team_train[TEAM_TOTAL_LABEL])
    return model


_REFRAMED_CACHE = {}
_ROLLING_RESIDUAL_CACHE = {}


def _predict_all_reframed_receiving(feat, train_pairs, test_pairs):
    """Joint/multi-output Phase A, Phase-2-of-the-consensus-gap-work form:
    fit share models for EVERY REFRAMED_SHARE_STATS combo plus one
    team_passing_yards model on train_pairs, predict all of them on
    test_pairs, and compose reconstructed rates through the SAME capped
    per-team share-sum renormalization predict.py ships
    (transitions.receiving_share_scale). The cap is a cross-position
    quantity - a team's WR+TE+RB share sum - so it structurally cannot be
    computed inside a single-(position, stat) fit; that is exactly why the
    pre-Phase-2 backtest never applied it, leaving the MAE table and the
    interval residuals calibrated on uncapped reconstructions while 6 of
    32 live 2026 teams shipped capped ones.

    Parity limits vs predict.py's live composition, stated plainly rather
    than papered over: (1) shares here carry no hand-curated role metadata;
    (2) no rookie-path implied shares enter the denominator - the
    veteran backtest frame has no rookie predictions for the held-out
    season. Achieved parity is "identical composition code path, empty
    discount/rookie inputs," not a byte-for-byte replica of the live run.

    Limit (1) is narrower than it used to be. Gate A added
    depth_history.py, which reconstructs a preseason depth chart for EVERY
    season from nflverse; what is still missing is the curated file's
    hand-verified role tier and a complete historical as-of roster path.

    Team-total predictions are drawn from each test row's OWN
    TEAM_FEATURES (already the player's season_from team context), the
    same "season_from's observed team context stands in for season_to"
    framing every other feature in this pipeline uses. Rows with team=NaN
    (rare, documented elsewhere) are dropped: the RidgeCV team model
    errors on NaN input, unlike LightGBM.

    Returns {(position, stat): (test_df, capped_pred, uncapped_pred)} with
    predictions in RATE units, aligned to test_df row order. Both variants
    are returned so the MAE table can report capped (what ships) alongside
    uncapped (the pre-Phase-2 basis) - whether capping helps on the
    held-out year is a real question the table should answer, not bury.
    Memoized on (train_pairs, test_pairs) since three separate consumers
    (MAE table, interval residuals, coherence backtest) need the same
    fits."""
    if len(test_pairs) != 1:
        raise ValueError(
            "receiving composition requires one held-out transition so its "
            "availability weights can be fit strictly on prior folds"
        )
    held = tuple(test_pairs[0])
    key = (id(feat), tuple(map(tuple, train_pairs)), tuple(map(tuple, test_pairs)))
    if key in _REFRAMED_CACHE:
        return _REFRAMED_CACHE[key]

    team_model = _fit_team_total_model(feat, train_pairs)
    per_combo, frames = {}, []
    for position, stat in sorted(REFRAMED_SHARE_STATS):
        train = build_role_transition_pairs(feat, position, stat, train_pairs, label_col=RECEIVING_SHARE_ELIG_LABEL)
        test = build_role_transition_pairs(feat, position, stat, test_pairs, label_col=RECEIVING_SHARE_ELIG_LABEL)
        test = test.dropna(subset=TEAM_FEATURES).reset_index(drop=True)
        if train.empty or test.empty or team_model is None:
            continue
        share_model = LGBMRegressor(**LGBM_PARAMS)
        share_model.fit(train[ROLE_FEATURES], train[RECEIVING_SHARE_ELIG_LABEL])
        f = test[["team"]].copy()
        f["share"] = np.clip(
            age_shrunk_predict(share_model, test, position, features=ROLE_FEATURES), 0, None)
        # No depth multiplier on the share. Depth is inside the share model's
        # own ROLE_FEATURES now, so the team-level cap below sees exactly the
        # shares production composes.
        # Production uses projected_games/17 in the share guard.  Using the
        # held-out season's actual games_played_to here leaks the outcome and
        # makes the historical composition easier than the live one.  Fit the
        # availability model on prior folds only, exactly as corrections.py's
        # cross-fitted residual path does.
        f["weight"] = projected_participation_weight(
            feat, test, position, train_pairs, held)
        # Team-grain inputs, looked up per (season_from, team). Scoring the
        # team model on `test` directly is what produced ~40%-low team
        # totals and dragged every reframed receiving residual with them.
        f["team_total_pred"] = np.clip(team_model.predict(
            team_model_inputs(feat, test_pairs, test["season_from"], test["team"])), 0, None)
        f["position"], f["stat"], f["row"] = position, stat, test.index
        per_combo[(position, stat)] = test
        frames.append(f)

    if not frames:
        _REFRAMED_CACHE[key] = {}
        return {}

    allf = pd.concat(frames, ignore_index=True)
    scale, _ = receiving_share_scale(allf[["team", "share", "weight"]])
    allf["uncapped"] = allf["share"] * allf["team_total_pred"]
    allf["capped"] = allf["share"] * scale * allf["team_total_pred"]

    # Phase-7 elite-shrinkage correction, refit HERE on an inner
    # leave-one-out over train_pairs ONLY - never train.py's production
    # parameters, which saw the held-out pair. Without this the MAE table
    # and the interval residuals below would be scored against a
    # correction that had already seen their answers.
    corr_params = fit_elite_shrinkage(compute_loo_receiving_residuals(feat, train_pairs))
    if corr_params:
        for (position, stat), test in per_combo.items():
            m = (allf["position"] == position) & (allf["stat"] == stat)
            rows_idx = allf.loc[m, "row"].to_numpy()
            adj = elite_shrinkage_adjustment(
                pd.Series([position] * int(m.sum())),
                test.loc[rows_idx, "naive_pred"].to_numpy(),
                corr_params,
            )
            allf.loc[m, "capped"] = allf.loc[m, "capped"].to_numpy() + adj

    out = {}
    for (position, stat), test in per_combo.items():
        sub = allf[(allf["position"] == position) & (allf["stat"] == stat)].sort_values("row")
        out[(position, stat)] = (test, sub["capped"].to_numpy(), sub["uncapped"].to_numpy())
    _REFRAMED_CACHE[key] = out
    return out


def _predict_reframed_receiving(feat, position, stat, train_pairs, test_pairs):
    """Single-(position, stat) accessor over _predict_all_reframed_receiving
    (which holds the real logic + parity notes). Returns
    (test_df, capped_pred, uncapped_pred) or None."""
    return _predict_all_reframed_receiving(feat, train_pairs, test_pairs).get((position, stat))


def backtest_position_stat(feat, position, stat, train_pairs=TRAIN_PAIRS, test_pair=TEST_PAIR):
    y_col = role_label_for(position, stat)

    if (position, stat) in REFRAMED_SHARE_STATS:
        result = _predict_reframed_receiving(feat, position, stat, train_pairs, [test_pair])
        if result is None:
            return None
        test, pred, pred_uncapped = result
    else:
        train = build_role_transition_pairs(feat, position, stat, train_pairs)
        test = build_role_transition_pairs(feat, position, stat, [test_pair])
        if train.empty or test.empty:
            return None
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(train[ROLE_FEATURES], train[y_col])
        # No multiplier. Depth reaches the prediction through the tier feature
        # inside ROLE_FEATURES, exactly as the shipped path does since the
        # Gate B ladder was retired.
        pred = age_shrunk_predict(model, test, position, features=ROLE_FEATURES)
        pred_uncapped = None

    naive = test["naive_pred"]  # season_from's own pg rate, carried forward unchanged
    actual = test[y_col]

    row = {
        "position": position, "stat": stat, "n_test": len(test),
        "model_mae": mae(pred, actual), "naive_mae": mae(naive, actual),
        "model_wins": mae(pred, actual) < mae(naive, actual),
    }
    if pred_uncapped is not None:
        # Reframed stats: headline model_mae is the CAPPED composition
        # (what predict.py ships, Phase 2 parity); the uncapped MAE is
        # reported alongside so "does capping even help on held-out data"
        # stays a visible, answerable question.
        row["model_mae_uncapped"] = mae(pred_uncapped, actual)
    return row


def backtest_team_total(feat, train_pairs=TRAIN_PAIRS, test_pair=TEST_PAIR,
                        label_col=TEAM_TOTAL_LABEL, stat="passing_yards"):
    """Team-level MAE-vs-naive for the new team_passing_yards model (joint/
    multi-output Phase A) - same held-out discipline as
    backtest_position_stat, at team-season grain instead of player-season."""
    train = build_team_transition_pairs(feat, train_pairs, label_col=label_col)
    test = build_team_transition_pairs(feat, [test_pair], label_col=label_col)
    if train.empty or test.empty:
        return None
    model = RidgeCV(alphas=np.logspace(-2, 3, 20))
    model.fit(train[TEAM_MODEL_FEATURES], train[label_col])
    pred = model.predict(test[TEAM_MODEL_FEATURES])
    naive = test["naive_pred"]
    actual = test[label_col]
    return {
        "position": "TEAM", "stat": stat, "n_test": len(test),
        "model_mae": mae(pred, actual), "naive_mae": mae(naive, actual),
        "model_wins": mae(pred, actual) < mae(naive, actual),
    }


def coherence_ratio_backtest(feat):
    """Returning-veteran receiving coverage diagnostic on the 2025 fold.

    This is *not* the physical team passing/receiving identity: transition
    pairs contain only players with a season-N rate and a season-N+1 rate, so
    rookies, arrivals without a prior rate, and attrition are absent.  The
    numerator is therefore the returning-veteran portion of the receiving
    room divided by the full team passing-yards anchor.  The explicit column
    names below prevent this partial-roster diagnostic from being presented
    as whole-team coherence again.

    Predicted numerators use fold-trained projected availability.  Only the
    observed comparison numerator uses held-out actual games, as ground truth.
    """
    team_model = _fit_team_total_model(feat, TRAIN_PAIRS)
    if team_model is None:
        return pd.DataFrame()

    old_pred, new_pred, actual_recv = [], [], []
    team_total_pred_by_row = {}
    for position, stat in REFRAMED_SHARE_STATS:
        # OLD: today's independent model, trained directly on {stat}_pg.
        old_train = build_transition_pairs(feat, position, stat, TRAIN_PAIRS)
        old_test = build_transition_pairs(feat, position, stat, [TEST_PAIR])
        if old_train.empty or old_test.empty:
            continue
        old_model = LGBMRegressor(**LGBM_PARAMS)
        old_model.fit(old_train[ALL_FEATURES], old_train[f"{stat}_pg"])
        old_test = old_test.copy()
        # Deliberately still the per-appearance arm: this function exists to
        # compare the independent-rate FRAMING against the composed-share one,
        # and the old arm is the pre-reframing baseline. Neither arm applies a
        # depth multiplier any more.
        old_test["old_pred"] = age_shrunk_predict(old_model, old_test, position)
        old_test["actual"] = old_test[f"{stat}_pg"]
        old_test["weight"] = projected_participation_weight(
            feat, old_test, position, TRAIN_PAIRS, TEST_PAIR)
        old_test["actual_weight"] = (
            pd.to_numeric(old_test["games_played_to"], errors="coerce")
            / SEASON_GAMES
        ).clip(0, 1)
        old_pred.append(old_test[
            ["team", "old_pred", "actual", "weight", "actual_weight"]])

        # NEW: reframed share model x team-total model.
        result = _predict_reframed_receiving(feat, position, stat, TRAIN_PAIRS, [TEST_PAIR])
        if result is None:
            continue
        new_test, new_reconstructed, _ = result
        new_test = new_test.copy()
        new_test["new_pred"] = new_reconstructed
        new_test["weight"] = projected_participation_weight(
            feat, new_test, position, TRAIN_PAIRS, TEST_PAIR)
        new_pred.append(new_test[["team", "new_pred", "weight"]])

    if not old_pred or not new_pred:
        return pd.DataFrame()

    old_df = pd.concat(old_pred, ignore_index=True)
    new_df = pd.concat(new_pred, ignore_index=True)

    team_test = build_team_transition_pairs(feat, [TEST_PAIR])
    team_test = team_test.copy()
    team_test["team_total_pred"] = team_model.predict(team_test[TEAM_MODEL_FEATURES])

    old_df["old_expected"] = old_df["old_pred"] * old_df["weight"]
    new_df["new_expected"] = new_df["new_pred"] * new_df["weight"]
    # Actual coverage is the observed returning-veteran contribution, so its
    # held-out games are legitimately part of the target, not a predictor.
    actual_df = old_df[["team", "actual", "actual_weight"]].copy()
    old_sum = old_df.groupby("team")["old_expected"].sum().rename("old_receiving_sum")
    new_sum = new_df.groupby("team")["new_expected"].sum().rename("new_receiving_sum")
    actual_df["actual_expected"] = actual_df["actual"] * actual_df["actual_weight"]
    actual_sum = actual_df.groupby("team")["actual_expected"].sum().rename("actual_receiving_sum")

    out = team_test.set_index("team")[["team_total_pred", TEAM_TOTAL_LABEL]].join(
        [old_sum, new_sum, actual_sum], how="inner"
    ).reset_index()
    out["old_returning_veteran_ratio"] = out["old_receiving_sum"] / out["team_total_pred"]
    out["new_returning_veteran_ratio"] = out["new_receiving_sum"] / out["team_total_pred"]
    out["actual_returning_veteran_ratio"] = out["actual_receiving_sum"] / out[TEAM_TOTAL_LABEL]
    return out


def rolling_residual_rows(feat, test_pairs=ROLLING_TEST_PAIRS):
    """One row per strictly forward, out-of-sample veteran rate residual."""
    cache_key = (id(feat), tuple(map(tuple, test_pairs)))
    if cache_key in _ROLLING_RESIDUAL_CACHE:
        return _ROLLING_RESIDUAL_CACHE[cache_key].copy()
    available = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
    rows = []
    for test_pair in test_pairs:
        train_pairs = [pair for pair in available if pair[1] <= test_pair[0]]
        if not train_pairs:
            continue
        for position, stats in TARGET_STATS.items():
            for stat in stats:
                y_col = role_label_for(position, stat)
                if (position, stat) in REFRAMED_SHARE_STATS:
                    result = _predict_reframed_receiving(
                        feat, position, stat, train_pairs, [test_pair])
                    if result is None:
                        continue
                    test, pred, _ = result
                else:
                    train = build_role_transition_pairs(
                        feat, position, stat, train_pairs)
                    test = build_role_transition_pairs(
                        feat, position, stat, [test_pair])
                    if train.empty or test.empty:
                        continue
                    model = LGBMRegressor(**LGBM_PARAMS)
                    model.fit(train[ROLE_FEATURES], train[y_col])
                    # These residuals ARE models/interval_residuals.csv, so
                    # they must be actual - pred on the basis that ships: a
                    # role rate with no post-hoc multiplier.
                    pred = age_shrunk_predict(
                        model, test, position, features=ROLE_FEATURES)
                actual = pd.to_numeric(test[y_col], errors="coerce").to_numpy()
                frame = pd.DataFrame({
                    "position": position,
                    "stat": stat,
                    "test_season": test_pair[1],
                    "n_train_transitions": len(train_pairs),
                    "player_id": test["player_id"].to_numpy(),
                    "pred": pred,
                    "actual": actual,
                })
                frame["resid"] = frame["actual"] - frame["pred"]
                rows.append(frame)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _ROLLING_RESIDUAL_CACHE[cache_key] = out
    return out.copy()


def residual_quantiles(feat, quantiles=INTERVAL_QUANTILES):
    """Cross-fitted residual quantiles pooled across rolling future folds.

    Each underlying prediction is made using only earlier transitions.  The
    pooled quantiles are the production calibration artifact; forward coverage
    is reported separately by :func:`forward_interval_coverage`, where a fold's
    interval is calibrated only on residuals from earlier test seasons.
    """
    residuals = rolling_residual_rows(feat)
    if residuals.empty:
        return pd.DataFrame()
    rows = []
    for (position, stat), grp in residuals.groupby(["position", "stat"]):
        lo, hi = np.quantile(grp["resid"], quantiles)
        rows.append({
            "position": position,
            "stat": stat,
            "n_test": len(grp),
            "n_crossfit_folds": grp["test_season"].nunique(),
            "first_test_season": int(grp["test_season"].min()),
            "last_test_season": int(grp["test_season"].max()),
            "resid_low": float(lo),
            "resid_high": float(hi),
            "resid_std": float(np.std(grp["resid"])),
            "low_n_flag": len(grp) < INTERVAL_MIN_N,
            "calibration_basis": "pooled strictly-forward rolling residuals",
        })
    return pd.DataFrame(rows)


def forward_interval_coverage(feat, quantiles=INTERVAL_QUANTILES):
    """Coverage on untouched folds using only earlier-fold calibration."""
    residuals = rolling_residual_rows(feat)
    if residuals.empty:
        return pd.DataFrame()
    target = quantiles[1] - quantiles[0]
    rows = []
    for (position, stat), grp in residuals.groupby(["position", "stat"]):
        seasons = sorted(grp["test_season"].unique())
        for season in seasons[1:]:
            calibration = grp[grp["test_season"] < season]["resid"]
            test = grp[grp["test_season"] == season]
            if calibration.empty or test.empty:
                continue
            lo, hi = np.quantile(calibration, quantiles)
            covered = test["actual"].between(test["pred"] + lo, test["pred"] + hi)
            rows.append({
                "position": position,
                "stat": stat,
                "test_season": int(season),
                "n_calibration": len(calibration),
                "n_test": len(test),
                "resid_low": float(lo),
                "resid_high": float(hi),
                "coverage": float(covered.mean()),
                "target_coverage": float(target),
                "coverage_gap": float(covered.mean() - target),
                "calibration_seasons": ",".join(
                    map(str, sorted(grp.loc[grp["test_season"] < season, "test_season"].unique()))),
            })
    return pd.DataFrame(rows)


def backtest_availability(feat, conn=None, test_pairs=ROLLING_TEST_PAIRS):
    """Rolling, causally trained games-played evaluation by eligibility.

    Every fold is trained only on transitions ending no later than the source
    season.  Results explicitly separate players found on the target season's
    roster snapshot from source-season players absent from that snapshot
    (attrition).  The all-player row remains useful for end-to-end season-value
    accounting, but it must not be presented as the model's error among players
    actually eligible for a preseason projection.

    Fits on AVAILABILITY_FEATURES (Gate A), which includes the TEST year's
    preseason depth chart. That is not leakage: a week-1/early-August chart
    is public before the season it describes is played, so it is available
    to a genuine preseason projection - the same standing the live 2026 run
    gives src/depth_chart/starters_2026.csv. What the chart cannot see is
    the outcome being scored (games actually played), which is what would
    make it leakage."""
    available = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
    rows = []
    for test_pair in test_pairs:
        train_pairs = [pair for pair in available if pair[1] <= test_pair[0]]
        if not train_pairs:
            continue
        roster_ids = None
        if conn is not None:
            roster = pd.read_sql(
                f"select distinct player_id from seasonal_rosters "
                f"where season = {int(test_pair[1])} and player_id is not null",
                conn,
            )
            roster_ids = set(roster["player_id"])
        for position in TARGET_STATS:
            train = build_availability_pairs(feat, position, train_pairs)
            test = build_availability_pairs(feat, position, [test_pair])
            if train.empty or test.empty:
                continue
            model = LGBMRegressor(**LGBM_PARAMS)
            model.fit(train[AVAILABILITY_FEATURES], train[AVAILABILITY_LABEL])
            scored = test[["player_id", AVAILABILITY_LABEL, "naive_pred", "played_again"]].copy()
            scored["pred"] = np.clip(
                model.predict(test[AVAILABILITY_FEATURES]), 0, SEASON_GAMES)
            if roster_ids is None:
                scopes = {
                    "all_source_players": pd.Series(True, index=scored.index),
                    "returning_players_outcome_stratum": scored["played_again"],
                    "attrition_outcome_stratum": ~scored["played_again"],
                }
            else:
                eligible = scored["player_id"].isin(roster_ids)
                scopes = {
                    "all_source_players": pd.Series(True, index=scored.index),
                    "target_roster_eligible": eligible,
                    "not_on_target_roster_attrition": ~eligible,
                }
            for scope, mask in scopes.items():
                sub = scored[mask]
                if sub.empty:
                    continue
                model_mae = mae(sub["pred"], sub[AVAILABILITY_LABEL])
                naive_mae = mae(sub["naive_pred"], sub[AVAILABILITY_LABEL])
                rows.append({
                    "test_season": test_pair[1],
                    "n_train_transitions": len(train_pairs),
                    "scope": scope,
                    "position": position,
                    "stat": "games",
                    "n_test": len(sub),
                    "n_never_played_again": int((~sub["played_again"]).sum()),
                    "model_mae": model_mae,
                    "naive_mae": naive_mae,
                    "model_bias": float((sub["pred"] - sub[AVAILABILITY_LABEL]).mean()),
                    "naive_bias": float((sub["naive_pred"] - sub[AVAILABILITY_LABEL]).mean()),
                    "model_wins": model_mae < naive_mae,
                })
    return pd.DataFrame(rows)


def backtest_season_totals(feat, conn=None, train_pairs=TRAIN_PAIRS,
                           test_pair=TEST_PAIR):
    """The question Phase 11 exists to answer: which framing best predicts
    SEASON value? Compares, on the held-out year and scored against actual
    season-N+1 totals for every season-N player (0 for those who never
    played again):
      rate x17          - what a per-game-only deliverable forces a reader
                          to do, and the pre-Phase-11 status quo
      rate x pred games - the shipped decomposition
      naive             - carry season-N's actual total forward
    """
    roster_ids = None
    if conn is not None:
        roster = pd.read_sql(
            f"select distinct player_id from seasonal_rosters "
            f"where season = {int(test_pair[1])} and player_id is not null",
            conn,
        )
        roster_ids = set(roster["player_id"])

    rows = []
    for position, stat in [("WR", "receiving_yards"), ("RB", "rushing_yards"),
                           ("TE", "receiving_yards"), ("QB", "passing_yards")]:
        av_train = build_availability_pairs(feat, position, train_pairs)
        av_test = build_availability_pairs(feat, position, [test_pair])
        rate_train = build_role_transition_pairs(feat, position, stat, train_pairs)
        if av_train.empty or av_test.empty or rate_train.empty:
            continue
        gm = LGBMRegressor(**LGBM_PARAMS).fit(av_train[AVAILABILITY_FEATURES], av_train[AVAILABILITY_LABEL])
        # role_rate_label, not role_label_for: this is the INDEPENDENT-rate
        # arm of the comparison, so it fits the rate directly even for the
        # reframed stats whose shipped path composes a share instead.
        rm = LGBMRegressor(**LGBM_PARAMS).fit(
            rate_train[ROLE_FEATURES], rate_train[role_rate_label(stat)])
        games_hat = np.clip(gm.predict(av_test[AVAILABILITY_FEATURES]), 0, SEASON_GAMES)
        # The availability frame carries AVAILABILITY_FEATURES, not the role
        # ones. Attach the tier from the target season's chart and the prior in
        # the label's own units, so the rate model is scored on the same inputs
        # it was fit on rather than on a frame that happens to share a name.
        av_scoring = attach_depth_tier(av_test, int(test_pair[1]), conn=conn)
        prior = feat[feat["season"] == test_pair[0]].drop_duplicates("player_id").set_index(
            "player_id")[role_rate_label(stat)]
        av_scoring[ROLE_PRIOR_FEATURE] = av_scoring["player_id"].map(prior).to_numpy(dtype=float)
        rate_hat = np.clip(
            age_shrunk_predict(rm, av_scoring, position, features=ROLE_FEATURES), 0, None)
        composed = pd.Series(np.nan, index=av_test.index, dtype=float)
        parity_limit = "independent rate path; production room normalization unavailable"
        if (position, stat) in REFRAMED_SHARE_STATS:
            result = _predict_reframed_receiving(
                feat, position, stat, train_pairs, [test_pair])
            if result is not None:
                composed_test, composed_pred, _ = result
                by_player = pd.Series(
                    composed_pred, index=composed_test["player_id"]
                ).groupby(level=0).first()
                composed = av_test["player_id"].map(by_player)
                rate_hat = composed.fillna(pd.Series(rate_hat, index=av_test.index)).to_numpy()
                parity_limit = (
                    "production team-total×share composition for returning-rate rows; "
                    "independent-rate fallback where the transition interface has no "
                    "season-N+1 conditional-rate row; no historical curated role or rookies"
                )

        # Actual season-N+1 total, and season-N's own total as the naive
        # carry-forward. Both looked up off the feature frame directly:
        # av_test keeps players who vanished, whose total is a real 0.
        st = test_pair[1]
        actual_tot = feat[(feat.position == position) & (feat.season == st)].set_index("player_id")[stat]
        prior_tot = feat[(feat.position == position) & (feat.season == test_pair[0])].set_index("player_id")[stat]
        scored = av_test[["player_id", "played_again"]].copy()
        scored["actual"] = av_test["player_id"].map(actual_tot).fillna(0.0).to_numpy()
        scored["naive"] = av_test["player_id"].map(prior_tot).fillna(0.0).to_numpy()
        scored["rate_hat"] = rate_hat
        scored["games_hat"] = games_hat
        scored["composed"] = composed.notna().to_numpy()
        if roster_ids is None:
            scopes = {"all_source_players": pd.Series(True, index=scored.index)}
        else:
            eligible = scored["player_id"].isin(roster_ids)
            scopes = {
                "all_source_players": pd.Series(True, index=scored.index),
                "target_roster_eligible": eligible,
                "not_on_target_roster_attrition": ~eligible,
            }
        for scope, mask in scopes.items():
            sub = scored[mask]
            if sub.empty:
                continue
            rows.append({
                "test_season": test_pair[1],
                "scope": scope,
                "position": position,
                "stat": stat,
                "n_test": len(sub),
                "n_composed_rate_rows": int(sub["composed"].sum()),
                "composition_coverage": float(sub["composed"].mean()),
                "rate_x17_mae": mae(
                    sub["rate_hat"] * SEASON_GAMES, sub["actual"]),
                "rate_x_games_mae": mae(
                    sub["rate_hat"] * sub["games_hat"], sub["actual"]),
                "naive_mae": mae(sub["naive"], sub["actual"]),
                "parity_limit": parity_limit,
            })
    return pd.DataFrame(rows)


def run_veteran_backtest(feat):
    rows = []
    for position, stats in TARGET_STATS.items():
        for stat in stats:
            r = backtest_position_stat(feat, position, stat)
            if r:
                rows.append(r)
    team_row = backtest_team_total(feat)
    if team_row:
        rows.append(team_row)
    attempts_row = backtest_team_total(
        feat, label_col=TEAM_ATTEMPTS_LABEL, stat="pass_attempts")
    if attempts_row:
        rows.append(attempts_row)
    return pd.DataFrame(rows)


def run_rolling_origin_backtest(feat, test_pairs=ROLLING_TEST_PAIRS):
    """Expanding-window evaluation across multiple genuinely future folds.

    Each test transition is predicted only from earlier transitions.  This
    prevents the repeatedly inspected 2024->2025 result from serving as the
    project's sole evidence.  The first evaluable fold is 2022->2023 because
    2021->2022 is required as its training history.
    """
    available = [(2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
    rows = []
    for test_pair in test_pairs:
        train_pairs = [pair for pair in available if pair[1] <= test_pair[0]]
        if not train_pairs:
            continue
        for position, stats in TARGET_STATS.items():
            for stat in stats:
                result = backtest_position_stat(
                    feat, position, stat, train_pairs=train_pairs, test_pair=test_pair
                )
                if result:
                    result["test_season"] = test_pair[1]
                    result["n_train_transitions"] = len(train_pairs)
                    rows.append(result)
        team = backtest_team_total(feat, train_pairs=train_pairs, test_pair=test_pair)
        if team:
            team["test_season"] = test_pair[1]
            team["n_train_transitions"] = len(train_pairs)
            rows.append(team)
        attempts = backtest_team_total(
            feat, train_pairs=train_pairs, test_pair=test_pair,
            label_col=TEAM_ATTEMPTS_LABEL, stat="pass_attempts")
        if attempts:
            attempts["test_season"] = test_pair[1]
            attempts["n_train_transitions"] = len(train_pairs)
            rows.append(attempts)
    return pd.DataFrame(rows)


def run_rookie_backtest(conn, feat):
    rdf = build_rookie_dataset(conn, feat)
    train_seasons = list(range(2016, 2025))
    baselines = fit_rookie_baselines(rdf, train_seasons)
    preds = predict_rookies(rdf, baselines, [2025])

    actual_2025 = rdf[rdf["season"] == 2025][["player_id"] + [c for c in rdf.columns if c.endswith("_pg")]]
    merged = preds.merge(actual_2025, on="player_id", suffixes=("_pred", "_actual"))

    rows = []
    for position, stats in TARGET_STATS.items():
        for stat in stats:
            p_col, a_col = f"{stat}_pg_pred", f"{stat}_pg_actual"
            sub = merged[(merged["position"] == position) & merged[p_col].notna() & merged[a_col].notna()]
            if sub.empty:
                continue
            rows.append({
                "position": position, "stat": stat, "n_test": len(sub),
                "model_mae": mae(sub[p_col], sub[a_col]),
            })
    return pd.DataFrame(rows)


def main():
    conn = get_conn()
    feat = build_player_season_features(conn)

    print("=== Veteran backtest: train 2021-22/22-23/23-24, test 2024-25 ===")
    vet = run_veteran_backtest(feat)
    pd.set_option("display.width", 160)
    print(vet.to_string(index=False))

    print("\n=== Rolling-origin veteran evaluation (expanding window) ===")
    rolling = run_rolling_origin_backtest(feat)
    if rolling.empty:
        print("(no rolling-origin rows)")
    else:
        print(rolling.to_string(index=False))
        summary = rolling.groupby(["position", "stat"]).agg(
            folds=("test_season", "nunique"),
            n_test=("n_test", "sum"),
            model_mae=("model_mae", "mean"),
            naive_mae=("naive_mae", "mean"),
            fold_win_rate=("model_wins", "mean"),
        ).reset_index()
        print("\nRolling-origin fold-mean summary:")
        print(summary.to_string(index=False))

    print("\n=== Rookie backtest: baselines from 2016-2024 rookies, test 2025 rookies ===")
    rook = run_rookie_backtest(conn, feat)
    print(rook.to_string(index=False))

    print("\n=== Cross-fitted rolling residual quantiles (production veteran intervals) ===")
    resid = residual_quantiles(feat)
    print(resid.to_string(index=False))
    os.makedirs(MODELS_DIR, exist_ok=True)
    resid_path = os.path.join(MODELS_DIR, "interval_residuals.csv")
    resid.to_csv(resid_path, index=False)
    print(f"Saved -> {resid_path}")

    print("\n=== Untouched forward interval coverage (calibrated on earlier folds only) ===")
    coverage = forward_interval_coverage(feat)
    print(coverage.to_string(index=False) if not coverage.empty else "(no forward coverage rows)")
    if not coverage.empty:
        coverage_path = os.path.join(MODELS_DIR, "interval_forward_coverage.csv")
        coverage.to_csv(coverage_path, index=False)
        print(f"Saved -> {coverage_path}")

    print("\n=== Returning-veteran receiving coverage on 2024-2025 holdout ===")
    print("(partial-roster diagnostic, NOT the whole-team passing/receiving identity;")
    print(" predicted numerators use fold-trained availability, actual uses observed availability)")
    coh = coherence_ratio_backtest(feat)
    if coh.empty:
        print("(coherence backtest produced no rows - check TRAIN_PAIRS/TEST_PAIR data availability)")
    else:
        cols = ["team", "old_returning_veteran_ratio",
                "new_returning_veteran_ratio", "actual_returning_veteran_ratio"]
        print(coh[cols].to_string(index=False))
        actual_col = "actual_returning_veteran_ratio"
        print(f"\nMean |returning-veteran ratio - observed| across teams: "
              f"old={coh['old_returning_veteran_ratio'].sub(coh[actual_col]).abs().mean():.3f}, "
              f"new={coh['new_returning_veteran_ratio'].sub(coh[actual_col]).abs().mean():.3f}")

    print("\n=== Rolling availability: all source players vs target-roster eligibility/attrition ===")
    av = backtest_availability(feat, conn=conn)
    print(av.to_string(index=False) if not av.empty else "(no availability rows)")

    print("\n=== Conditional rate by preseason depth (Gate B calibration) ===")

    print("\n=== Season-TOTAL framing (Phase 11): which produces the best season value? ===")
    print("(scored on actual season totals incl. 0 for players who never played again)")
    st = backtest_season_totals(feat, conn=conn)
    print(st.to_string(index=False) if not st.empty else "(no season-total rows)")

    print("\n=== Injury-cohort gate (Phase 6 diagnostic - see corrections.py) ===")
    print("(WR, season-N active games <= 8, season-N >= 50 rec ypg, no suspension-coded weeks;")
    print(" a correction gets built only at mean resid > +3 ypg AND >= 65% positive)")
    gate = injury_cohort_gate(
        compute_loo_receiving_residuals(feat, TRAIN_PAIRS + [TEST_PAIR]),
        suspension_weeks=load_suspension_weeks(conn),
    )
    if gate.get("n"):
        print(f"  n={gate['n']}  mean resid {gate['mean_resid']:+.2f}  median {gate['median_resid']:+.2f}  "
              f"positive {gate['frac_positive']:.0%}")
        print(f"  implied retention: model {gate['model_retention']:.2f} vs actual {gate['actual_retention']:.2f}")
        tripped = gate["mean_resid"] > 3.0 and gate["frac_positive"] >= 0.65
        print(f"  GATE {'TRIPPED - a retention correction is warranted' if tripped else 'NOT tripped - no correction needed'}")
    else:
        print("  (no cohort rows found)")

    conn.close()
    return vet, rook, resid, coh


if __name__ == "__main__":
    main()
