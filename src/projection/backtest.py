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
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.projection.data_prep import get_conn
from src.projection.features import build_player_season_features, TARGET_STATS
from src.projection.transitions import build_transition_pairs, ALL_FEATURES
from src.projection.rookies import build_rookie_dataset, fit_rookie_baselines, predict_rookies
from src.projection.train import LGBM_PARAMS

TRAIN_PAIRS = [(2021, 2022), (2022, 2023), (2023, 2024)]
TEST_PAIR = (2024, 2025)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def backtest_position_stat(feat, position, stat):
    y_col = f"{stat}_pg"
    train = build_transition_pairs(feat, position, stat, TRAIN_PAIRS)
    test = build_transition_pairs(feat, position, stat, [TEST_PAIR])
    if train.empty or test.empty:
        return None

    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[ALL_FEATURES], train[y_col])
    pred = model.predict(test[ALL_FEATURES])

    naive = test["naive_pred"]  # season_from's own pg rate, carried forward unchanged
    actual = test[y_col]

    return {
        "position": position, "stat": stat, "n_test": len(test),
        "model_mae": mae(pred, actual), "naive_mae": mae(naive, actual),
        "model_wins": mae(pred, actual) < mae(naive, actual),
    }


def run_veteran_backtest(feat):
    rows = []
    for position, stats in TARGET_STATS.items():
        for stat in stats:
            r = backtest_position_stat(feat, position, stat)
            if r:
                rows.append(r)
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

    conn.close()
    return vet, rook


if __name__ == "__main__":
    main()
