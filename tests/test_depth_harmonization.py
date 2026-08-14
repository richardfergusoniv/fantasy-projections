import sqlite3
import unittest

import pandas as pd

from src.projection.depth_history import _CHART_CACHE, load_preseason_depth_chart


class DepthHarmonizationTests(unittest.TestCase):
    def test_old_schema_ties_remain_one_tier(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        pd.DataFrame([
            dict(season=2024, week=1, game_type="REG", formation="Offense",
                 club_code="LA", position="WR", depth_team=1, gsis_id="z", full_name="Zulu"),
            dict(season=2024, week=1, game_type="REG", formation="Offense",
                 club_code="LA", position="WR", depth_team=1, gsis_id="a", full_name="Alpha"),
        ]).to_sql("depth_charts", conn, index=False)
        _CHART_CACHE.pop(2024, None)

        chart = load_preseason_depth_chart(2024, conn=conn).set_index("player_id")
        self.assertEqual(chart.loc["a", "depth_rank"], 1)
        self.assertEqual(chart.loc["z", "depth_rank"], 1)
        self.assertEqual(chart.loc["a", "availability_rank"], 1)
        self.assertEqual(chart.loc["z", "availability_rank"], 1)


if __name__ == "__main__":
    unittest.main()
