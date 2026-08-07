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
    build_transition_pairs, build_team_transition_pairs, ALL_FEATURES, TEAM_FEATURES,
    REFRAMED_SHARE_STATS, RECEIVING_SHARE_LABEL, TEAM_TOTAL_LABEL, receiving_share_scale,
)
from src.projection.rookies import build_rookie_dataset, fit_rookie_baselines, predict_rookies
from src.projection.train import LGBM_PARAMS

TRAIN_PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024)]
TEST_PAIR = (2024, 2025)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
INTERVAL_QUANTILES = (0.10, 0.90)  # 80% empirical interval width - see PHASE5_REPORT.md for why
INTERVAL_MIN_N = 30  # veteran (position, stat) test-set n below this would need a parametric fallback (none do - min is 61)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def _fit_team_total_model(feat, train_pairs):
    team_train = build_team_transition_pairs(feat, train_pairs)
    if team_train.empty:
        return None
    model = RidgeCV(alphas=np.logspace(-2, 3, 20))
    model.fit(team_train[TEAM_FEATURES], team_train[TEAM_TOTAL_LABEL])
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
    than papered over: (1) shares here carry NO role/depth-chart discount
    (d_i=1) - no historical curated depth chart exists for held-out
    seasons; (2) no rookie-path implied shares enter the denominator - the
    veteran backtest frame has no rookie predictions for the held-out
    season. Achieved parity is "identical composition code path, empty
    discount/rookie inputs," not a byte-for-byte replica of the live run.

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
        f["team_total_pred"] = np.clip(team_model.predict(test[TEAM_FEATURES]), 0, None)
        f["position"], f["stat"], f["row"] = position, stat, test.index
        per_combo[(position, stat)] = test
        frames.append(f)

    if not frames:
        _REFRAMED_CACHE[key] = {}
        return {}

    allf = pd.concat(frames, ignore_index=True)
    scale, _ = receiving_share_scale(allf[["team", "share"]])
    allf["uncapped"] = allf["share"] * allf["team_total_pred"]
    allf["capped"] = allf["share"] * scale * allf["team_total_pred"]

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


def backtest_position_stat(feat, position, stat):
    y_col = f"{stat}_pg"

    if (position, stat) in REFRAMED_SHARE_STATS:
        result = _predict_reframed_receiving(feat, position, stat, TRAIN_PAIRS, [TEST_PAIR])
        if result is None:
            return None
        test, pred, pred_uncapped = result
    else:
        train = build_transition_pairs(feat, position, stat, TRAIN_PAIRS)
        test = build_transition_pairs(feat, position, stat, [TEST_PAIR])
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


def backtest_team_total(feat):
    """Team-level MAE-vs-naive for the new team_passing_yards model (joint/
    multi-output Phase A) - same held-out discipline as
    backtest_position_stat, at team-season grain instead of player-season."""
    train = build_team_transition_pairs(feat, TRAIN_PAIRS)
    test = build_team_transition_pairs(feat, [TEST_PAIR])
    if train.empty or test.empty:
        return None
    model = RidgeCV(alphas=np.logspace(-2, 3, 20))
    model.fit(train[TEAM_FEATURES], train[TEAM_TOTAL_LABEL])
    pred = model.predict(test[TEAM_FEATURES])
    naive = test["naive_pred"]
    actual = test[TEAM_TOTAL_LABEL]
    return {
        "position": "TEAM", "stat": "passing_yards", "n_test": len(test),
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
        old_pred.append(old_test[["team", "old_pred", "actual"]])

        # NEW: reframed share model x team-total model.
        result = _predict_reframed_receiving(feat, position, stat, TRAIN_PAIRS, [TEST_PAIR])
        if result is None:
            continue
        new_test, new_reconstructed, _ = result
        new_test = new_test.copy()
        new_test["new_pred"] = new_reconstructed
        new_pred.append(new_test[["team", "new_pred"]])

    if not old_pred or not new_pred:
        return pd.DataFrame()

    old_df = pd.concat(old_pred, ignore_index=True)
    new_df = pd.concat(new_pred, ignore_index=True)

    team_test = build_team_transition_pairs(feat, [TEST_PAIR])
    team_test = team_test.copy()
    team_test["team_total_pred"] = team_model.predict(team_test[TEAM_FEATURES])

    old_sum = old_df.groupby("team")["old_pred"].sum().rename("old_receiving_sum")
    new_sum = new_df.groupby("team")["new_pred"].sum().rename("new_receiving_sum")
    actual_sum = old_df.groupby("team")["actual"].sum().rename("actual_receiving_sum")

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

    conn.close()
    return vet, rook, resid, coh


if __name__ == "__main__":
    main()
