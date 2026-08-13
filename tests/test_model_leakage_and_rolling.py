import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd

from src.projection.ol_quality import team_season_ol_quality
from src.projection.corrections import fit_elite_shrinkage


class ModelLeakageAndRollingTests(unittest.TestCase):
    def test_empty_early_fold_disables_elite_correction(self):
        self.assertEqual(fit_elite_shrinkage(pd.DataFrame()), {})

    def test_ol_quality_uses_exact_season_coefficients(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        pd.DataFrame(
            [
                (2021, "A", 1, "REG", "p1", "G", 100),
                (2022, "A", 1, "REG", "p1", "G", 100),
            ],
            columns=["season", "team", "week", "game_type", "pfr_player_id", "position", "offense_snaps"],
        ).to_sql("snap_counts", conn, index=False)
        pd.DataFrame([("g1", "p1")], columns=["gsis_id", "pfr_id"]).to_sql("players", conn, index=False)
        pd.DataFrame(
            [
                (2021, "g1", 1.0, "pass_protection"),
                (2022, "g1", 9.0, "pass_protection"),
                (2021, "g1", 2.0, "run_blocking"),
                (2022, "g1", 8.0, "run_blocking"),
            ],
            columns=["season", "gsis_id", "coef", "submodel"],
        ).to_sql("ol_coefficients", conn, index=False)
        pd.DataFrame(
            [(2021, "A", "individual"), (2022, "A", "individual")],
            columns=["season", "team", "confidence_flag"],
        ).to_sql("ol_team_season_churn", conn, index=False)

        out = team_season_ol_quality(conn, [2021, 2022]).set_index("season")
        self.assertEqual(out.loc[2021, "ol_pass_protection_score"], 1.0)
        self.assertEqual(out.loc[2022, "ol_pass_protection_score"], 9.0)
        self.assertEqual(out.loc[2021, "ol_run_blocking_score"], 2.0)
        self.assertEqual(out.loc[2022, "ol_run_blocking_score"], 8.0)

    def test_rolling_fold_training_is_strictly_earlier(self):
        import src.projection.backtest as backtest

        calls = []

        def fake_position(_feat, position, stat, train_pairs, test_pair):
            calls.append((tuple(train_pairs), test_pair))
            return {"position": position, "stat": stat, "n_test": 1,
                    "model_mae": 1.0, "naive_mae": 2.0, "model_wins": True}

        with patch.object(backtest, "backtest_position_stat", fake_position), patch.object(
            backtest, "backtest_team_total", lambda *_a, **_k: None
        ):
            backtest.run_rolling_origin_backtest(
                pd.DataFrame(), test_pairs=[(2022, 2023), (2024, 2025)]
            )

        self.assertTrue(calls)
        for train_pairs, test_pair in calls:
            self.assertTrue(all(pair[1] <= test_pair[0] for pair in train_pairs))
            self.assertNotIn(test_pair, train_pairs)
