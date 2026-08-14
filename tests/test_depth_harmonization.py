import sqlite3
import unittest

import pandas as pd

from src.projection.depth_history import clear_depth_chart_cache, load_preseason_depth_chart


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
        clear_depth_chart_cache()

        chart = load_preseason_depth_chart(2024, conn=conn).set_index("player_id")
        self.assertEqual(chart.loc["a", "depth_rank"], 1)
        self.assertEqual(chart.loc["z", "depth_rank"], 1)
        self.assertEqual(chart.loc["a", "availability_rank"], 1)
        self.assertEqual(chart.loc["z", "availability_rank"], 1)

    def test_new_schema_as_of_uses_latest_snapshot(self):
        conn = sqlite3.connect(":memory:")
        self.addCleanup(conn.close)
        # Include old-schema columns so _load_old_schema can run (returns empty
        # for season=2026) before falling through to the daily-snapshot path.
        rows = []
        for dt, pid, rank in (
            ("2026-08-01", "early", 1),
            ("2026-08-15", "late", 1),
        ):
            rows.append(dict(
                season=None, week=None, game_type=None, formation=None,
                club_code=None, position=None, depth_team=None, full_name=None,
                team="SF", pos_abb="WR", pos_rank=rank, gsis_id=pid,
                player_name=pid, dt=dt,
            ))
        pd.DataFrame(rows).to_sql("depth_charts", conn, index=False)
        clear_depth_chart_cache()
        early = load_preseason_depth_chart(2026, conn=conn)
        self.assertTrue((early["player_id"] == "early").all())
        clear_depth_chart_cache()
        late = load_preseason_depth_chart(2026, conn=conn, as_of="2026-08-20")
        self.assertTrue((late["player_id"] == "late").all())
        self.assertIn("2026-08-15", late["source"].iloc[0])


if __name__ == "__main__":
    unittest.main()
