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
from src.projection.transitions import (
    build_transition_pairs, build_team_transition_pairs, build_availability_pairs,
    ALL_FEATURES, AVAILABILITY_FEATURES, TEAM_FEATURES, TEAM_MODEL_FEATURES, team_model_inputs,
    REFRAMED_SHARE_STATS,
    RECEIVING_SHARE_LABEL, TEAM_TOTAL_LABEL, AVAILABILITY_LABEL, SEASON_GAMES,
    TEAM_ATTEMPTS_LABEL, receiving_share_scale,
)
from src.projection.rookies import build_rookie_dataset, fit_rookie_baselines, predict_rookies
from src.projection.train import LGBM_PARAMS
from src.projection.corrections import (
    compute_loo_receiving_residuals, fit_elite_shrinkage, elite_shrinkage_adjustment,
    injury_cohort_gate, load_suspension_weeks,
)
from src.projection.depth_history import attach_depth_rank

TRAIN_PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024)]
TEST_PAIR = (2024, 2025)
ROLLING_TEST_PAIRS = [(2022, 2023), (2023, 2024), (2024, 2025)]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
INTERVAL_QUANTILES = (0.10, 0.90)  # 80% empirical interval width - see PHASE5_REPORT.md for why
INTERVAL_MIN_N = 30  # veteran (position, stat) test-set n below this would need a parametric fallback (none do - min is 61)
DEPTH_VOLUME_STATS = {"QB": "attempts", "RB": "carries", "WR": "targets", "TE": "targets"}


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def depth_rate_calibration(feat, conn, pairs=TRAIN_PAIRS + [TEST_PAIR]):
    """LOTO calibration for predict.DEPTH_RATE_LADDER.

    Uses one opportunity-volume rate per position and the current ``*_pg``
    labels, so a games-played redefinition cannot leave the hard-coded ladder
    silently calibrated to an obsolete denominator. Ratios are reported raw;
    production may cap ratios above 1 because this gate is a discount, not a
    general model-bias correction.
    """
    rows = []
    for position, stat in DEPTH_VOLUME_STATS.items():
        label = f"{stat}_pg"
        for held_out in pairs:
            train_pairs = [pair for pair in pairs if pair != held_out]
            train = build_transition_pairs(feat, position, stat, train_pairs)
            test = build_transition_pairs(feat, position, stat, [held_out])
            if train.empty or test.empty:
                continue
            model = LGBMRegressor(**LGBM_PARAMS)
            model.fit(train[ALL_FEATURES], train[label])
            test = test.copy()
            test["pred"] = np.clip(model.predict(test[ALL_FEATURES]), 0, None)
            test["position"] = position
            test = attach_depth_rank(test, held_out[1], conn=conn)
            test["actual"] = test[label]
            rows.append(test[["position", "nfl_depth_rank", "actual", "pred"]])
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["depth_band"] = out["nfl_depth_rank"].apply(
        lambda rank: "off_chart" if pd.isna(rank)
        else (f"rank_{int(rank)}" if int(rank) <= 5 else "deep"))
    summary = out.groupby(["position", "depth_band"]).agg(
        n=("actual", "size"), actual_sum=("actual", "sum"), pred_sum=("pred", "sum")
    ).reset_index()
    summary["actual_over_pred"] = summary["actual_sum"] / summary["pred_sum"].replace(0, np.nan)
    return summary


def _fit_team_total_model(feat, train_pairs):
    team_train = build_team_transition_pairs(feat, train_pairs)
    if team_train.empty:
        return None
    model = RidgeCV(alphas=np.logspace(-2, 3, 20))
    model.fit(team_train[TEAM_MODEL_FEATURES], team_train[TEAM_TOTAL_LABEL])
    return model


_REFRAMED_CACHE = {}


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
    key = (id(feat), tuple(map(tuple, train_pairs)), tuple(map(tuple, test_pairs)))
    if key in _REFRAMED_CACHE:
        return _REFRAMED_CACHE[key]

    team_model = _fit_team_total_model(feat, train_pairs)
    per_combo, frames = {}, []
    for position, stat in sorted(REFRAMED_SHARE_STATS):
        train = build_transition_pairs(feat, position, stat, train_pairs, label_col=RECEIVING_SHARE_LABEL)
        test = build_transition_pairs(feat, position, stat, test_pairs, label_col=RECEIVING_SHARE_LABEL)
        test = test.dropna(subset=TEAM_FEATURES).reset_index(drop=True)
        if train.empty or test.empty or team_model is None:
            continue
        share_model = LGBMRegressor(**LGBM_PARAMS)
        share_model.fit(train[ALL_FEATURES], train[RECEIVING_SHARE_LABEL])
        f = test[["team"]].copy()
        f["share"] = np.clip(share_model.predict(test[ALL_FEATURES]), 0, None)
        f["weight"] = (
            pd.to_numeric(test["games_played_to"], errors="coerce") / SEASON_GAMES
        ).clip(0, 1)
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
    y_col = f"{stat}_pg"

    if (position, stat) in REFRAMED_SHARE_STATS:
        result = _predict_reframed_receiving(feat, position, stat, train_pairs, [test_pair])
        if result is None:
            return None
        test, pred, pred_uncapped = result
    else:
        train = build_transition_pairs(feat, position, stat, train_pairs)
        test = build_transition_pairs(feat, position, stat, [test_pair])
        if train.empty or test.empty:
            return None
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(train[ALL_FEATURES], train[y_col])
        pred = model.predict(test[ALL_FEATURES])
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
    """The actual go/no-go signal for the joint/multi-output Phase A
    reframing (see the plan this was built from) - computed on the SAME
    2024-2025 held-out season used for every other backtest metric here,
    not just the live 2026 output. For each team in the held-out set:
    sum(predicted receiving_yards_pg across all WR/TE/RB test rows for that
    team) / (predicted team_passing_yards_pg for that team), compared for
    the CURRENT (independent, unreframed) models vs. the NEW (shared-anchor,
    reframed) models, against the REAL 2025 ratio (actual receiving sum /
    actual team passing) as ground truth for what "coherent" should look
    like on real data."""
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
        old_test["old_pred"] = old_model.predict(old_test[ALL_FEATURES])
        old_test["actual"] = old_test[f"{stat}_pg"]
        old_test["weight"] = old_test["games_played_to"] / SEASON_GAMES
        old_pred.append(old_test[["team", "old_pred", "actual", "weight"]])

        # NEW: reframed share model x team-total model.
        result = _predict_reframed_receiving(feat, position, stat, TRAIN_PAIRS, [TEST_PAIR])
        if result is None:
            continue
        new_test, new_reconstructed, _ = result
        new_test = new_test.copy()
        new_test["new_pred"] = new_reconstructed
        new_test["weight"] = new_test["games_played_to"] / SEASON_GAMES
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
    old_df["actual_expected"] = old_df["actual"] * old_df["weight"]
    old_sum = old_df.groupby("team")["old_expected"].sum().rename("old_receiving_sum")
    new_sum = new_df.groupby("team")["new_expected"].sum().rename("new_receiving_sum")
    actual_sum = old_df.groupby("team")["actual_expected"].sum().rename("actual_receiving_sum")

    out = team_test.set_index("team")[["team_total_pred", TEAM_TOTAL_LABEL]].join(
        [old_sum, new_sum, actual_sum], how="inner"
    ).reset_index()
    out["old_ratio"] = out["old_receiving_sum"] / out["team_total_pred"]
    out["new_ratio"] = out["new_receiving_sum"] / out["team_total_pred"]
    out["actual_ratio"] = out["actual_receiving_sum"] / out[TEAM_TOTAL_LABEL]
    return out


def residual_quantiles(feat, quantiles=INTERVAL_QUANTILES):
    """Empirical (position, stat) -> (resid_low, resid_high, resid_std) from
    the SAME held-out 2025 backtest (train 2021-22/22-23/23-24, predict
    2025) used for the MAE table above - genuine out-of-sample errors, not
    train-fit residuals (which would be optimistically narrow). Used by
    predict.py to build pred_pg_low/pred_pg_high = pred_pg + resid_low/high
    for the veteran path. All 20 position/stat combos have n_test in
    61-170 (see PHASE4_REPORT.md's backtest table) - above INTERVAL_MIN_N,
    so no position/stat needs a parametric (normal-approximation) fallback;
    resid_std is still carried through in case a future season's smaller
    test set ever needs one."""
    rows = []
    for position, stats in TARGET_STATS.items():
        for stat in stats:
            y_col = f"{stat}_pg"
            if (position, stat) in REFRAMED_SHARE_STATS:
                # Reconstructed team_total x share prediction, same as
                # backtest_position_stat - residuals must be in RATE units
                # (predict.py adds resid_low/high directly onto pred_pg),
                # not share units.
                result = _predict_reframed_receiving(feat, position, stat, TRAIN_PAIRS, [TEST_PAIR])
                if result is None:
                    continue
                test, pred, _ = result
            else:
                train = build_transition_pairs(feat, position, stat, TRAIN_PAIRS)
                test = build_transition_pairs(feat, position, stat, [TEST_PAIR])
                if train.empty or test.empty:
                    continue
                model = LGBMRegressor(**LGBM_PARAMS)
                model.fit(train[ALL_FEATURES], train[y_col])
                pred = model.predict(test[ALL_FEATURES])
            resid = test[y_col].values - pred
            lo, hi = np.quantile(resid, quantiles)
            rows.append({
                "position": position, "stat": stat, "n_test": len(test),
                "resid_low": float(lo), "resid_high": float(hi), "resid_std": float(np.std(resid)),
                "low_n_flag": len(test) < INTERVAL_MIN_N,
            })
    return pd.DataFrame(rows)


def backtest_availability(feat):
    """Held-out games-played MAE per position vs carrying season-N games
    forward (Phase 11). Scored on ALL season-N players including those who
    never played again - the rows build_transition_pairs drops and which
    every other table here is therefore blind to.

    Fits on AVAILABILITY_FEATURES (Gate A), which includes the TEST year's
    preseason depth chart. That is not leakage: a week-1/early-August chart
    is public before the season it describes is played, so it is available
    to a genuine preseason projection - the same standing the live 2026 run
    gives src/depth_chart/starters_2026.csv. What the chart cannot see is
    the outcome being scored (games actually played), which is what would
    make it leakage."""
    rows = []
    for position in TARGET_STATS:
        train = build_availability_pairs(feat, position, TRAIN_PAIRS)
        test = build_availability_pairs(feat, position, [TEST_PAIR])
        if train.empty or test.empty:
            continue
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(train[AVAILABILITY_FEATURES], train[AVAILABILITY_LABEL])
        pred = np.clip(model.predict(test[AVAILABILITY_FEATURES]), 0, SEASON_GAMES)
        actual, naive = test[AVAILABILITY_LABEL], test["naive_pred"]
        rows.append({
            "position": position, "stat": "games", "n_test": len(test),
            "n_never_played_again": int((~test["played_again"]).sum()),
            "model_mae": mae(pred, actual), "naive_mae": mae(naive, actual),
            "model_wins": mae(pred, actual) < mae(naive, actual),
        })
    return pd.DataFrame(rows)


def backtest_season_totals(feat):
    """The question Phase 11 exists to answer: which framing best predicts
    SEASON value? Compares, on the held-out year and scored against actual
    season-N+1 totals for every season-N player (0 for those who never
    played again):
      rate x17          - what a per-game-only deliverable forces a reader
                          to do, and the pre-Phase-11 status quo
      rate x pred games - the shipped decomposition
      naive             - carry season-N's actual total forward
    """
    rows = []
    for position, stat in [("WR", "receiving_yards"), ("RB", "rushing_yards"),
                           ("TE", "receiving_yards"), ("QB", "passing_yards")]:
        av_train = build_availability_pairs(feat, position, TRAIN_PAIRS)
        av_test = build_availability_pairs(feat, position, [TEST_PAIR])
        rate_train = build_transition_pairs(feat, position, stat, TRAIN_PAIRS)
        if av_train.empty or av_test.empty or rate_train.empty:
            continue
        gm = LGBMRegressor(**LGBM_PARAMS).fit(av_train[AVAILABILITY_FEATURES], av_train[AVAILABILITY_LABEL])
        rm = LGBMRegressor(**LGBM_PARAMS).fit(rate_train[ALL_FEATURES], rate_train[f"{stat}_pg"])
        games_hat = np.clip(gm.predict(av_test[AVAILABILITY_FEATURES]), 0, SEASON_GAMES)
        rate_hat = np.clip(rm.predict(av_test[ALL_FEATURES]), 0, None)

        # Actual season-N+1 total, and season-N's own total as the naive
        # carry-forward. Both looked up off the feature frame directly:
        # av_test keeps players who vanished, whose total is a real 0.
        st = TEST_PAIR[1]
        actual_tot = feat[(feat.position == position) & (feat.season == st)].set_index("player_id")[stat]
        prior_tot = feat[(feat.position == position) & (feat.season == TEST_PAIR[0])].set_index("player_id")[stat]
        actual = av_test["player_id"].map(actual_tot).fillna(0.0).to_numpy()
        naive = av_test["player_id"].map(prior_tot).fillna(0.0).to_numpy()
        rows.append({
            "position": position, "stat": stat, "n_test": len(av_test),
            "rate_x17_mae": mae(rate_hat * SEASON_GAMES, actual),
            "rate_x_games_mae": mae(rate_hat * games_hat, actual),
            "naive_mae": mae(naive, actual),
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

    print("\n=== Empirical residual quantiles (for predict.py's veteran prediction intervals) ===")
    resid = residual_quantiles(feat)
    print(resid.to_string(index=False))
    os.makedirs(MODELS_DIR, exist_ok=True)
    resid_path = os.path.join(MODELS_DIR, "interval_residuals.csv")
    resid.to_csv(resid_path, index=False)
    print(f"Saved -> {resid_path}")

    print("\n=== Joint/multi-output Phase A go/no-go: coherence ratio on 2024-2025 holdout ===")
    print("(old = today's independent receiving_yards models, new = team_total x share reframing,")
    print(" actual = real 2025 outcomes - all three as sum(WR/TE/RB receiving_yards_pg) / team passing_yards_pg)")
    coh = coherence_ratio_backtest(feat)
    if coh.empty:
        print("(coherence backtest produced no rows - check TRAIN_PAIRS/TEST_PAIR data availability)")
    else:
        print(coh[["team", "old_ratio", "new_ratio", "actual_ratio"]].to_string(index=False))
        print(f"\nMean |ratio - actual_ratio| across teams: "
              f"old={coh['old_ratio'].sub(coh['actual_ratio']).abs().mean():.3f}, "
              f"new={coh['new_ratio'].sub(coh['actual_ratio']).abs().mean():.3f}")

    print("\n=== Availability backtest (Phase 11): games played, vs carrying season-N games forward ===")
    av = backtest_availability(feat)
    print(av.to_string(index=False) if not av.empty else "(no availability rows)")

    print("\n=== Conditional rate by preseason depth (Gate B calibration) ===")
    depth_cal = depth_rate_calibration(feat, conn)
    print(depth_cal.to_string(index=False) if not depth_cal.empty else "(no calibration rows)")
    if not depth_cal.empty:
        depth_path = os.path.join(MODELS_DIR, "depth_rate_calibration.csv")
        depth_cal.to_csv(depth_path, index=False)
        print(f"Saved -> {depth_path}")

    print("\n=== Season-TOTAL framing (Phase 11): which produces the best season value? ===")
    print("(scored on actual season totals incl. 0 for players who never played again)")
    st = backtest_season_totals(feat)
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
